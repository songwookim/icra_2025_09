#!/usr/bin/env python3
"""Generate stiffness profiles using GLOBAL T_K with Eq.(16) Normalization.

This version:
1. Computes global T_K once from all demonstrations combined
2. Applies T_K to each demo to get raw k (small k)
3. Finds k_min, k_max across all demos
4. Normalizes each demo using Eq.(16): K = K_min + (K_max - K_min) / (k_max - k_min) * (k - k_min)
   where K (big K) is the final normalized stiffness for robot control
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, sosfiltfilt, savgol_filter

if os.environ.get("DISPLAY", "") == "":
    import matplotlib
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
PKG_DIR = SCRIPT_DIR.parent.parent  # .../src/hri_falcon_robot_bridge

EPS = 1e-8

# ---------------------------------------------------------------------------
# CSV utilities
# ---------------------------------------------------------------------------

def _load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("CSV contains no rows")
    return df

def _time_vector(df: pd.DataFrame) -> np.ndarray:
    if {"t_sec", "t_nanosec"}.issubset(df.columns):
        sec = df["t_sec"].to_numpy(dtype=float)
        nsec = df["t_nanosec"].to_numpy(dtype=float)
        t = sec + nsec * 1e-9
    elif "t" in df.columns:
        t = df["t"].to_numpy(dtype=float)
    elif "time" in df.columns:
        t = df["time"].to_numpy(dtype=float)
    else:
        t = np.arange(len(df), dtype=float)
    t0 = float(t[0])
    return t - t0

def _estimate_fs(time: np.ndarray) -> Optional[float]:
    if time.ndim != 1 or len(time) < 3:
        return None
    dt = np.diff(time)
    dt = dt[(dt > 0) & np.isfinite(dt)]
    if dt.size == 0:
        return None
    return float(1.0 / np.median(dt))

def _guess_emg_columns(df: pd.DataFrame, channels: Optional[List[int]] = None) -> List[str]:
    if not channels:
        channels = list(range(1, 9))
    cols: List[str] = []
    for ch in channels:
        for cand in (f"emg_ch{ch}", f"ch{ch}", f"emg{ch}", f"channel{ch}"):
            if cand in df.columns:
                cols.append(cand)
                break
    return cols

def _align_emg_to_time(emg: np.ndarray, time: np.ndarray) -> Optional[np.ndarray]:
    if emg is None or emg.size == 0:
        return None
    if emg.shape[0] == time.shape[0]:
        return emg
    if time.size < 2 or emg.shape[0] < 2:
        return None
    t_start = float(time[0])
    t_end = float(time[-1])
    if not np.isfinite(t_start) or not np.isfinite(t_end) or np.isclose(t_end, t_start):
        return None
    src_times = np.linspace(t_start, t_end, num=emg.shape[0], dtype=float)
    if not np.all(np.isfinite(src_times)):
        return None
    aligned = np.empty((time.shape[0], emg.shape[1]), dtype=float)
    for ch in range(emg.shape[1]):
        aligned[:, ch] = np.interp(time, src_times, emg[:, ch])
    return aligned


def _center_signal(signal: np.ndarray, baseline_fraction: float = 0.1) -> np.ndarray:
    samples = signal.shape[0]
    take = max(1, int(round(samples * baseline_fraction)))
    baseline = np.median(signal[:take, :], axis=0, keepdims=True)
    return signal - baseline


# ---------------------------------------------------------------------------
# Force processing
# ---------------------------------------------------------------------------

def _smooth_force(force: np.ndarray, fs: Optional[float], cutoff_hz: float = 3.0, order: int = 2) -> Optional[np.ndarray]:
    """Apply low-pass filter to force signals to reduce noise."""
    if force is None or force.size == 0:
        return None
    if fs is None or fs <= 0 or cutoff_hz <= 0 or fs <= 2 * cutoff_hz:
        return None
    try:
        sos = butter(order, cutoff_hz, btype="low", fs=fs, output="sos")
        return sosfiltfilt(sos, force, axis=0)
    except ValueError:
        return None


def _compute_normal_force(
    df: pd.DataFrame,
    prefix: Optional[str] = None,
    n: Tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> np.ndarray:
    """Compute baseline-removed normal force component Fn (signed) for given sensor prefix."""
    cand_prefixes = ("force_3", "s3", "force3") if prefix is None else (prefix,)
    picked_cols: Optional[List[str]] = None
    for p in cand_prefixes:
        cols = [f"{p}_fx", f"{p}_fy", f"{p}_fz"]
        if all(c in df.columns for c in cols):
            picked_cols = cols
            break
    if picked_cols is None:
        raise KeyError(f"No force columns found for prefix={prefix or cand_prefixes}")
    F = df[picked_cols].to_numpy(dtype=float)
    rest = max(1, int(len(F) * 0.1))
    Fc = F - F[:rest].mean(axis=0, keepdims=True)
    n_vec = np.asarray(n, dtype=float)
    n_norm = np.linalg.norm(n_vec)
    if not np.isfinite(n_norm) or n_norm < 1e-12:
        n_vec = np.array([0.0, 0.0, 1.0], dtype=float)
        n_norm = 1.0
    n_hat = n_vec / n_norm
    Fn = Fc @ n_hat
    return Fn


# ---------------------------------------------------------------------------
# EMG filtering
# ---------------------------------------------------------------------------

def _ultra_smooth_strong_emg(
    emg_mag: np.ndarray,
    fs: Optional[float],
    cutoff_hz: float = 1.0,
    window_sec: float = 0.8,
    sg_window_sec: float = 0.6,
    sg_poly: int = 3,
) -> Optional[np.ndarray]:
    """Strong smoothing: LPF -> moving average -> Savitzky–Golay polish -> ReLU."""
    if emg_mag is None or emg_mag.size == 0:
        return None
    sm = np.asarray(emg_mag, dtype=float)
    # LPF
    if fs is not None and fs > 0 and cutoff_hz > 0 and fs > 2 * cutoff_hz:
        try:
            sos = butter(2, cutoff_hz, btype="low", fs=fs, output="sos")
            sm = sosfiltfilt(sos, sm, axis=0)
        except ValueError:
            pass
    # Moving average
    eff_fs = fs if fs is not None and fs > 0 else 200.0
    win = max(1, int(round(window_sec * eff_fs)))
    if win > 1:
        sm = uniform_filter1d(sm, size=win, axis=0, mode="nearest")
    # Savitzky–Golay polish
    sgw = max(5, int(round(sg_window_sec * eff_fs)))
    if sgw % 2 == 0:
        sgw += 1
    if sgw > sg_poly + 2:
        try:
            sm = savgol_filter(sm, window_length=sgw, polyorder=sg_poly, axis=0, mode="interp")
        except Exception:
            pass
    return np.maximum(sm, 0.0)


# ---------------------------------------------------------------------------
# Linear-algebra helpers for T_F / T_K
# ---------------------------------------------------------------------------

def compute_tf(P: np.ndarray, F: np.ndarray, reg: float = 1e-6) -> np.ndarray:
    """Solve T_F from F = T_F · P using ridge regularization."""
    PPt = P @ P.T
    reg_eye = reg * np.eye(PPt.shape[0])
    return F @ P.T @ np.linalg.inv(PPt + reg_eye)


def compute_projection_from_tf(T_F: np.ndarray, reg: float = 1e-6) -> np.ndarray:
    """Return force subspace projector H_F from T_F with numerical symmetrization."""
    TF_TF_T = T_F @ T_F.T
    reg_eye = reg * np.eye(TF_TF_T.shape[0])
    inv_term = np.linalg.inv(TF_TF_T + reg_eye)
    H_F = T_F.T @ inv_term @ T_F
    H_F = 0.5 * (H_F + H_F.T)
    return H_F


def compute_k_basis_from_force_projector(
    H_F: np.ndarray,
    target_rank: int = 3,
    eig_threshold: float = 0.5,
    near_one_tol: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute orthogonal complement basis ("stiffness subspace") of force projector."""
    dim = H_F.shape[0]
    H_F = 0.5 * (H_F + H_F.T)
    H_K = np.eye(dim) - H_F
    H_K = 0.5 * (H_K + H_K.T)
    
    eigvals, eigvecs = np.linalg.eigh(H_K)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    
    keep_mask = (eigvals > eig_threshold) | (np.isclose(eigvals, 1.0, atol=near_one_tol))
    kept = np.count_nonzero(keep_mask)
    if kept < target_rank:
        raise ValueError(f"Complementary subspace insufficient rank: have {kept}, need {target_rank}.")
    
    basis_full = eigvecs[:, keep_mask]
    basis = basis_full[:, :target_rank].T
    
    return basis, H_K


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_variant(
    time: np.ndarray,
    forces_signed: np.ndarray,
    forces_smoothed: np.ndarray,
    force_normal_abs: Optional[np.ndarray],
    stiffness_raw: np.ndarray,
    stiffness_filtered: np.ndarray,
    emg_raw: np.ndarray,
    emg_filtered: np.ndarray,
    out_path: Path,
    title: str,
    filter_label: str,
) -> None:
    if time.ndim != 1 or time.size < 2:
        return
    fig, axes = plt.subplots(3, 1, figsize=(12, 8.0), sharex=True)
    ax_force, ax_stiff, ax_emg = axes
    
    colors = ["red", "green", "blue"]
    axis_labels = ["x", "y", "z"]
    
    for axis in range(min(3, forces_signed.shape[1])):
        color = colors[axis % len(colors)]
        label = axis_labels[axis % len(axis_labels)]
        ax_force.plot(time, forces_signed[:, axis], color=color, linewidth=0.8, linestyle="--", alpha=0.3, label=f"F_{label} raw" if axis == 0 else None)
        ax_force.plot(time, forces_smoothed[:, axis], color=color, linewidth=1.2, label=f"F_{label} smoothed")
    
    if force_normal_abs is not None and force_normal_abs.size == time.size:
        ax_force.plot(time, force_normal_abs, color="black", linewidth=1.0, linestyle=":", alpha=0.9, label="F_n")
    
    ax_force.set_ylabel("Force [N]")
    ax_force.grid(alpha=0.3)
    ax_force.legend(loc="upper right", fontsize=8)
    
    for axis in range(min(3, stiffness_raw.shape[1])):
        color = colors[axis % len(colors)]
        label = axis_labels[axis % len(axis_labels)]
        ax_stiff.plot(time, stiffness_raw[:, axis], color=color, linewidth=1.0, linestyle="--", alpha=0.1, label=f"K_{label} raw")
        ax_stiff.plot(time, stiffness_filtered[:, axis], color=color, linewidth=1.3, linestyle="-", label=f"K_{label} {filter_label}")
    
    ax_stiff.set_ylabel("Stiffness [N/m]")
    ax_stiff.grid(alpha=0.3)
    ax_stiff.legend(loc="upper right", fontsize=8)
    
    emg_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
    for ch in range(min(emg_raw.shape[1], 8)):
        color = emg_colors[ch % len(emg_colors)]
        ax_emg.plot(time, emg_raw[:, ch], color=color, linewidth=0.8, linestyle="--", alpha=0.1, label="EMG raw" if ch == 0 else None)
        ax_emg.plot(time, emg_filtered[:, ch], color=color, linewidth=1.2, linestyle="-", label=f"EMG {filter_label}" if ch == 0 else None)
    
    ax_emg.set_ylabel("EMG")
    ax_emg.set_xlabel("Time [s]")
    ax_emg.grid(alpha=0.1)
    ax_emg.legend(loc="upper right", fontsize=8)
    
    span = float(time[-1] - time[0])
    pad = 0.03 * span if span > 0 else 0.1
    ax_force.set_xlim(float(time[0]) - pad, float(time[-1]) + pad)
    
    fig.suptitle(f"{title} [{filter_label}]")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
WORKSPACE_ROOT = PKG_DIR.parent.parent
DEFAULT_INPUT = PKG_DIR / "outputs" / "logs" / "success"
K_INIT = 200.0
# (NEW) Normalization range for Eq.(16)
K_MIN = 50.0   # Robot minimum stiffness [N/m]
K_MAX = 800.0  # Robot maximum stiffness [N/m]

# ---------------------------------------------------------------------------
# EMG Normalization helpers
# ---------------------------------------------------------------------------

def _compute_emg_scaler(emg_list: List[np.ndarray], mode: str) -> Dict[str, np.ndarray]:
    """Compute per-channel statistics needed for normalization.

    Args:
        emg_list: list of (N_i, C) arrays AFTER smoothing & baseline centering & rectification.
        mode: one of 'none','zscore','unitvar','minmax','robust'.
    Returns:
        stats dict containing needed arrays; always includes 'mode'.
    """
    stats: Dict[str, np.ndarray] = {"mode": np.array([mode])}
    if mode == "none" or not emg_list:
        return stats
    concat = np.vstack(emg_list)  # (sum N, C)
    # Always store mean/std/min/max/median/mad for flexibility
    mean = concat.mean(axis=0)
    std = concat.std(axis=0) + EPS
    minv = concat.min(axis=0)
    maxv = concat.max(axis=0)
    median = np.median(concat, axis=0)
    mad = np.median(np.abs(concat - median), axis=0) + EPS
    stats.update({"mean": mean, "std": std, "min": minv, "max": maxv, "median": median, "mad": mad})
    return stats

def _apply_emg_normalization(X: np.ndarray, stats: Dict[str, np.ndarray]) -> np.ndarray:
    """Apply normalization to (N,C) EMG feature matrix using previously computed stats."""
    mode = stats.get("mode", np.array(["none"]))[0]
    if mode == "none":
        return X
    if mode == "zscore":
        return (X - stats["mean"]) / stats["std"]
    if mode == "unitvar":
        return X / stats["std"]
    if mode == "minmax":
        denom = stats["max"] - stats["min"] + EPS
        return (X - stats["min"]) / denom
    if mode == "robust":
        return (X - stats["median"]) / stats["mad"]
    return X


def compute_global_tk(input_dir: Path, emg_norm_mode: str = "none") -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]], Dict[str, np.ndarray]]:
    """Compute global T_K for each finger from all demonstrations combined.
    
    Returns:
        Dict with keys 'th', 'if', 'mf', each containing (T_K, H_K, T_F, H_F) tuple
    """
    print("\n[Phase 1] Computing global T_K per finger from all demonstrations...")
    
    sensor_map = {"th": "s1", "if": "s2", "mf": "s3"}
    finger_data = {finger: {"emg": [], "force": []} for finger in ["th", "if", "mf"]}
    all_emg_for_scaler: List[np.ndarray] = []  # collect for normalization stats
    
    csv_files = sorted(input_dir.glob("*.csv"))
    processed = 0
    
    for csv_path in csv_files:
        if csv_path.name.endswith("_paper_profile.csv"):
            continue
            
        try:
            df = _load_csv(csv_path)
            time = _time_vector(df)
            fs = _estimate_fs(time) or 200.0
            
            # Extract EMG
            emg_cols = _guess_emg_columns(df)
            if not emg_cols:
                continue
                
            emg_raw = df[emg_cols].to_numpy(dtype=float)
            aligned_emg = _align_emg_to_time(emg_raw, time)
            if aligned_emg is None:
                continue
                
            emg_centered = _center_signal(aligned_emg)
            emg_magnitude = np.abs(emg_centered)
            emg_smoothed = _ultra_smooth_strong_emg(emg_magnitude, fs)
            if emg_smoothed is None:
                emg_smoothed = emg_magnitude
            all_emg_for_scaler.append(emg_smoothed)
            
            # Extract full 3D force for each finger
            for finger, sensor in sensor_map.items():
                try:
                    force_cols = [f"{sensor}_fx", f"{sensor}_fy", f"{sensor}_fz"]
                    if all(c in df.columns for c in force_cols):
                        forces = df[force_cols].to_numpy(dtype=float)
                        rest = max(1, int(len(forces) * 0.1))
                        baseline = np.mean(forces[:rest, :], axis=0, keepdims=True)
                        forces_centered = forces - baseline
                        forces_smoothed = _smooth_force(forces_centered, fs)
                        if forces_smoothed is None:
                            forces_smoothed = forces_centered
                        finger_data[finger]["emg"].append(emg_smoothed)
                        finger_data[finger]["force"].append(forces_smoothed)
                except Exception:
                    continue
            
            processed += 1
            
        except Exception as exc:
            print(f"[skip] {csv_path.name}: {exc}")
            continue
    
    if processed == 0:
        raise ValueError("No valid demonstrations found for global T_K computation")
    
    print(f"[info] Loaded {processed} demonstrations for global T_K")

    # -------------------------------------------------------------------
    # EMG normalization stats (global across all demos) then apply
    # -------------------------------------------------------------------
    emg_stats = _compute_emg_scaler(all_emg_for_scaler, emg_norm_mode)
    if emg_norm_mode != "none":
        print(f"[emg-norm] mode={emg_norm_mode}; per-channel mean/std range: "
              f"mean∈[{emg_stats['mean'].min():.3f},{emg_stats['mean'].max():.3f}] std∈[{emg_stats['std'].min():.3f},{emg_stats['std'].max():.3f}]")
        # Apply normalization in-place for each stored EMG array
        for finger in ["th","if","mf"]:
            finger_data[finger]["emg"] = [_apply_emg_normalization(arr, emg_stats) for arr in finger_data[finger]["emg"]]
    
    # Compute T_K for each finger
    global_tk = {}
    for finger in ["th", "if", "mf"]:
        if not finger_data[finger]["emg"]:
            raise ValueError(f"No data collected for finger {finger}")
        
        P_global = np.vstack(finger_data[finger]["emg"]).T  # (d, N_total)
        F_list = finger_data[finger]["force"]
        F_global = np.vstack(F_list).T  # (3, N_total)
        
        print(f"[{finger}] Global data: EMG {P_global.shape}, Force {F_global.shape}")
        
        T_F = compute_tf(P_global, F_global)
        H_F = compute_projection_from_tf(T_F)
        T_K, H_K = compute_k_basis_from_force_projector(H_F, target_rank=3)
        
        global_tk[finger] = (T_K, H_K, T_F, H_F)
        print(f"[{finger}] T_K shape: {T_K.shape}, norm: {np.linalg.norm(T_K):.4f}")
    
    return global_tk, emg_stats


def compute_global_k_range(
    input_dir: Path,
    global_tk: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    emg_stats: Dict[str, np.ndarray],
    finger_norm_source: str = "components",
) -> Dict[str, Dict[str, float]]:
    """Compute global stiffness ranges per finger for normalization.

    finger_norm_source:
        'components' -> min/max over all component entries of k
        'norm'       -> min/max over Euclidean norm ||k|| values

    Returns per finger dict with keys:
        k_min_component, k_max_component, k_min_norm, k_max_norm,
        k_min_used, k_max_used
    """
    print("\n[Phase 2] Computing global per-finger stiffness ranges (source=%s)..." % finger_norm_source)

    finger_k_values_components = {finger: [] for finger in ["th", "if", "mf"]}
    finger_k_values_norm = {finger: [] for finger in ["th", "if", "mf"]}

    csv_files = sorted(input_dir.glob("*.csv"))
    for csv_path in csv_files:
        if csv_path.name.endswith("_paper_profile.csv"):
            continue
        try:
            df = _load_csv(csv_path)
            time = _time_vector(df)
            fs = _estimate_fs(time) or 200.0
            emg_cols = _guess_emg_columns(df)
            if not emg_cols:
                continue
            emg_raw = df[emg_cols].to_numpy(dtype=float)
            aligned_emg = _align_emg_to_time(emg_raw, time)
            if aligned_emg is None:
                continue
            emg_centered = _center_signal(aligned_emg)
            emg_magnitude = np.abs(emg_centered)
            emg_smoothed = _ultra_smooth_strong_emg(emg_magnitude, fs)
            if emg_smoothed is None:
                emg_smoothed = emg_magnitude
            emg_smoothed = _apply_emg_normalization(emg_smoothed, emg_stats)
            P_variant = emg_smoothed.T
            for finger in ["th", "if", "mf"]:
                T_K_finger = global_tk[finger][0]
                k_raw = (T_K_finger @ P_variant).T + K_INIT  # (N,3)
                finger_k_values_components[finger].append(k_raw)
                finger_k_values_norm[finger].append(np.linalg.norm(k_raw, axis=1, keepdims=True))
        except Exception:
            continue

    ranges: Dict[str, Dict[str, float]] = {}
    for finger in ["th", "if", "mf"]:
        if not finger_k_values_components[finger]:
            raise ValueError(f"No k values computed for finger {finger}")
        comp_all = np.vstack(finger_k_values_components[finger])
        norm_all = np.vstack(finger_k_values_norm[finger])
        k_min_component = float(np.min(comp_all))
        k_max_component = float(np.max(comp_all))
        k_min_norm = float(np.min(norm_all))
        k_max_norm = float(np.max(norm_all))
        if finger_norm_source == "norm":
            k_min_used, k_max_used = k_min_norm, k_max_norm
        else:
            k_min_used, k_max_used = k_min_component, k_max_component
        ranges[finger] = {
            "k_min_component": k_min_component,
            "k_max_component": k_max_component,
            "k_min_norm": k_min_norm,
            "k_max_norm": k_max_norm,
            "k_min_used": k_min_used,
            "k_max_used": k_max_used,
        }
        print(f"[{finger}] used_min={k_min_used:.2f} used_max={k_max_used:.2f} (comp_min={k_min_component:.2f}, comp_max={k_max_component:.2f}, norm_min={k_min_norm:.2f}, norm_max={k_max_norm:.2f})")
    return ranges


def normalize_stiffness_eq16(
    k: np.ndarray,
    k_min: float,
    k_max: float,
    K_min: float = K_MIN,
    K_max: float = K_MAX,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply Equation (16) normalization; also return 0-1 scaled value.

    Returns:
        K: normalized stiffness in robot range
        k_norm01: per-sample 0..1 scaled stiffness (unitless intention level)
    """
    denom = k_max - k_min
    if abs(denom) < EPS:
        K_center = (K_min + K_max) / 2.0
        return np.full_like(k, K_center), np.full((k.shape[0], 1), 0.5)
    k_norm01 = (k - k_min) / denom
    K = K_min + (K_max - K_min) * k_norm01
    return K, k_norm01


def process_file_with_global_tk_normalized(
    csv_path: Path,
    output_dir: Path,
    global_tk: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    k_ranges: Dict[str, Dict[str, float]],
    emg_stats: Dict[str, np.ndarray],
    finger_norm_source: str = "components",
) -> Optional[List[Path]]:
    """Process single file using global T_K and Eq.(16) normalization."""
    try:
        df = _load_csv(csv_path)
    except Exception as exc:
        print(f"[skip] {csv_path.name}: load failed ({exc})")
        return None

    time = _time_vector(df)
    fs = _estimate_fs(time) or 200.0

    # Extract deformation and end-effector data
    deform_ecc = df["deform_ecc"].to_numpy(dtype=float) if "deform_ecc" in df else np.zeros(len(df), dtype=float)
    
    ee_if_px = df["ee_if_px"].to_numpy(dtype=float) if "ee_if_px" in df else (df["ee_px"].to_numpy(dtype=float) if "ee_px" in df else np.zeros(len(df), dtype=float))
    ee_if_py = df["ee_if_py"].to_numpy(dtype=float) if "ee_if_py" in df else (df["ee_py"].to_numpy(dtype=float) if "ee_py" in df else np.zeros(len(df), dtype=float))
    ee_if_pz = df["ee_if_pz"].to_numpy(dtype=float) if "ee_if_pz" in df else (df["ee_pz"].to_numpy(dtype=float) if "ee_pz" in df else np.zeros(len(df), dtype=float))
    
    ee_mf_px = df["ee_mf_px"].to_numpy(dtype=float) if "ee_mf_px" in df else np.zeros(len(df), dtype=float)
    ee_mf_py = df["ee_mf_py"].to_numpy(dtype=float) if "ee_mf_py" in df else np.zeros(len(df), dtype=float)
    ee_mf_pz = df["ee_mf_pz"].to_numpy(dtype=float) if "ee_mf_pz" in df else np.zeros(len(df), dtype=float)
    
    ee_th_px = df["ee_th_px"].to_numpy(dtype=float) if "ee_th_px" in df else np.zeros(len(df), dtype=float)
    ee_th_py = df["ee_th_py"].to_numpy(dtype=float) if "ee_th_py" in df else np.zeros(len(df), dtype=float)
    ee_th_pz = df["ee_th_pz"].to_numpy(dtype=float) if "ee_th_pz" in df else np.zeros(len(df), dtype=float)

    # Extract EMG
    emg_cols = _guess_emg_columns(df)
    if not emg_cols:
        print(f"[skip] {csv_path.name}: no EMG columns")
        return None

    emg_raw = df[emg_cols].to_numpy(dtype=float)
    aligned_emg = _align_emg_to_time(emg_raw, time)
    if aligned_emg is None:
        print(f"[skip] {csv_path.name}: EMG alignment failed")
        return None

    emg_centered = _center_signal(aligned_emg)
    emg_magnitude = np.abs(emg_centered)
    emg_smoothed = _ultra_smooth_strong_emg(emg_magnitude, fs)
    if emg_smoothed is None:
        emg_smoothed = emg_magnitude
    # Apply SAME normalization used in global T_K computation
    emg_smoothed = _apply_emg_normalization(emg_smoothed, emg_stats)
    emg_magnitude_norm = _apply_emg_normalization(emg_magnitude, emg_stats)

    # Prepare EMG features
    P_variant = emg_smoothed.T
    P_raw = emg_magnitude_norm.T

    out_paths: List[Path] = []

    # Process each finger
    finger_names = ["th", "if", "mf"]
    sensor_names = ["s1", "s2", "s3"]
    all_forces = []
    all_forces_csv = []
    all_stiffness_normalized = []
    all_stiffness_norm01 = []
    all_stiffness_raw = []
    all_fn_abs = []
    
    for finger, sensor in zip(finger_names, sensor_names):
        # Get this finger's global T_K and k_range
        T_K_finger = global_tk[finger][0]
        k_min = k_ranges[finger]["k_min_used"]
        k_max = k_ranges[finger]["k_max_used"]
        
        # Compute raw k (small k)
        k_raw = (T_K_finger @ P_variant).T + K_INIT  # (N, 3)
        k_raw_unsmoothed = (T_K_finger @ P_raw).T + K_INIT
        
        # Apply Eq.(16) normalization to get K (big K)
        K_normalized, k_norm01 = normalize_stiffness_eq16(k_raw, k_min, k_max, K_MIN, K_MAX)
        
        # Store for plotting and CSV
        all_stiffness_raw.append(k_raw_unsmoothed)  # for comparison in plot
        all_stiffness_normalized.append(K_normalized)
        all_stiffness_norm01.append(k_norm01)  # (N,1)
        
        # Extract forces
        force_cols = [f"{sensor}_fx", f"{sensor}_fy", f"{sensor}_fz"]
        if all(col in df.columns for col in force_cols):
            forces = df[force_cols].to_numpy(dtype=float)
            rest = max(1, int(len(forces) * 0.1))
            baseline = np.mean(forces[:rest, :], axis=0, keepdims=True)
            forces_centered = forces - baseline
            
            forces_smoothed = _smooth_force(forces_centered, fs)
            if forces_smoothed is None:
                forces_smoothed = forces_centered
            all_forces_csv.append(forces_smoothed)
            all_forces.append(forces_smoothed)

            Fn = _compute_normal_force(df, prefix=sensor, n=(0.0, 0.0, 1.0))
            Fn_smoothed = _smooth_force(Fn, fs)
            if Fn_smoothed is None:
                Fn_smoothed = Fn
            all_fn_abs.append(Fn_smoothed)
        else:
            all_forces.append(np.zeros((len(time), 3)))
            all_forces_csv.append(np.zeros((len(time), 3)))
            all_fn_abs.append(np.zeros(len(time)))
    
    # Generate plots
    for idx, finger_name in enumerate(finger_names):
        out_png = output_dir / f"{csv_path.stem}_{finger_name}_normalized.png"
        try:
            _plot_variant(
                time,
                all_forces[idx],
                all_forces[idx],
                all_fn_abs[idx],
                all_stiffness_raw[idx],  # raw k (after normalization) for comparison
                all_stiffness_normalized[idx],  # normalized K
                emg_magnitude,
                emg_smoothed,
                out_png,
                csv_path.stem,
                f"{finger_name} Eq.(16) Normalized",
            )
            out_paths.append(out_png)
        except Exception as exc:
            print(f"[warn] {csv_path.name}: PNG for {finger_name} failed ({exc})")

    # Save CSV with normalized K values
    out_csv = output_dir / f"{csv_path.stem}.csv"
    try:
        all_forces_combined = np.hstack(all_forces_csv)
        data = {"time_s": time.astype(float)}
        
        for i, (finger_name, sensor) in enumerate(zip(finger_names, sensor_names)):
            data[f"{sensor}_fx"] = all_forces_combined[:, i*3+0].astype(float)
            data[f"{sensor}_fy"] = all_forces_combined[:, i*3+1].astype(float)
            data[f"{sensor}_fz"] = all_forces_combined[:, i*3+2].astype(float)
            # Save normalized K (big K)
            data[f"{finger_name}_k1"] = all_stiffness_normalized[i][:, 0].astype(float)
            data[f"{finger_name}_k2"] = all_stiffness_normalized[i][:, 1].astype(float)
            data[f"{finger_name}_k3"] = all_stiffness_normalized[i][:, 2].astype(float)
            # 0-1 intention level (if norm source='norm' it's based on vector norm min/max)
            data[f"{finger_name}_k_intent01"] = all_stiffness_norm01[i][:, 0].astype(float)

        # (NEW) Export EMG magnitude & smoothed (both already normalized if emg_norm_mode applied)
        try:
            if emg_magnitude_norm is not None and emg_smoothed is not None:
                for ch_idx, col_name in enumerate(emg_cols):
                    # Raw rectified magnitude after normalization
                    data[f"{col_name}"] = emg_magnitude_norm[:, ch_idx].astype(float)
                    # Strongly smoothed envelope
                    data[f"{col_name}_smooth"] = emg_smoothed[:, ch_idx].astype(float)
        except Exception as exc:
            print(f"[warn] {csv_path.name}: failed adding EMG columns ({exc})")
        
        data["deform_ecc"] = deform_ecc.astype(float)
        data["ee_if_px"] = ee_if_px.astype(float)
        data["ee_if_py"] = ee_if_py.astype(float)
        data["ee_if_pz"] = ee_if_pz.astype(float)
        data["ee_mf_px"] = ee_mf_px.astype(float)
        data["ee_mf_py"] = ee_mf_py.astype(float)
        data["ee_mf_pz"] = ee_mf_pz.astype(float)
        data["ee_th_px"] = ee_th_px.astype(float)
        data["ee_th_py"] = ee_th_py.astype(float)
        data["ee_th_pz"] = ee_th_pz.astype(float)
        
        out_df = pd.DataFrame(data)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(out_csv, index=False)
        out_paths.append(out_csv)
    except Exception as exc:
        print(f"[warn] {csv_path.name}: CSV save failed ({exc})")

    print(f"[ok] {csv_path.name}: {len(out_paths)} files (Normalized)")
    return out_paths


def main() -> None:
    global K_MIN, K_MAX
    
    parser = argparse.ArgumentParser(description="Generate stiffness with global T_K + Eq.(16) normalization")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PKG_DIR / "outputs" / "stiffness_profiles_normalized",
    )
    parser.add_argument("--k-min", type=float, default=K_MIN, help="Target minimum stiffness K_min [N/m]")
    parser.add_argument("--k-max", type=float, default=K_MAX, help="Target maximum stiffness K_max [N/m]")
    parser.add_argument("--emg-norm-mode", type=str, default="none", choices=["none","zscore","unitvar","minmax","robust"], help="Per-channel EMG normalization before T_K computation")
    parser.add_argument("--finger-norm-source", type=str, default="components", choices=["components","norm"], help="Stiffness range source: components (min/max over all axes) or norm (vector L2 norm)")
    args = parser.parse_args()
    
    # Update global normalization range if specified
    K_MIN = args.k_min
    K_MAX = args.k_max
    
    # Resolve input directory
    candidates: List[Path] = []
    if isinstance(args.input, Path):
        candidates.append(args.input)
        if not args.input.is_absolute():
            candidates.append(Path.cwd() / args.input)
            candidates.append(PKG_DIR / args.input)
            candidates.append(SCRIPT_DIR / args.input)
            candidates.append(WORKSPACE_ROOT / args.input)
    for c in candidates:
        if c.is_dir():
            input_dir = c
            break
    else:
        raise ValueError(f"Input must be directory: {args.input}")
    
    print(f"[info] Normalization range: K_min={K_MIN:.1f}, K_max={K_MAX:.1f} N/m")
    
    # Phase 1: Compute global T_K per finger
    global_tk, emg_stats = compute_global_tk(input_dir, emg_norm_mode=args.emg_norm_mode)
    
    # Phase 2: Compute global k_min, k_max per finger
    k_ranges = compute_global_k_range(input_dir, global_tk, emg_stats, finger_norm_source=args.finger_norm_source)

    # Save EMG normalization stats if used
    if args.emg_norm_mode != "none":
        stats_out = args.output_dir / "emg_normalization_stats.json"
        args.output_dir.mkdir(parents=True, exist_ok=True)
        try:
            import json
            serializable = {k: (v.tolist() if isinstance(v, np.ndarray) else str(v)) for k, v in emg_stats.items()}
            with open(stats_out, "w") as f:
                json.dump(serializable, f, indent=2)
            print(f"[emg-norm] Saved stats to {stats_out}")
        except Exception as exc:
            print(f"[warn] Could not save EMG normalization stats ({exc})")
    
    # Phase 3: Process all files with normalization
    print(f"\n[Phase 3] Processing files with Eq.(16) normalization...")
    csv_files = [p for p in sorted(input_dir.glob("*.csv")) if not p.name.endswith("_paper_profile.csv")]
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    for csv_path in csv_files:
        process_file_with_global_tk_normalized(csv_path, args.output_dir, global_tk, k_ranges, emg_stats, finger_norm_source=args.finger_norm_source)

    # Save stiffness scaler info
    scaler_out = args.output_dir / "per_finger_stiffness_scalers.json"
    try:
        import json
        serializable_ranges = {f: {k: float(v) for k, v in k_ranges[f].items()} for f in k_ranges}
        with open(scaler_out, "w") as f:
            json.dump({"finger_norm_source": args.finger_norm_source, "ranges": serializable_ranges}, f, indent=2)
        print(f"[scaler] Saved per-finger stiffness ranges to {scaler_out}")
    except Exception as exc:
        print(f"[warn] Could not save stiffness scaler info ({exc})")
    
    print(f"\n[Done] All files processed with GLOBAL T_K + Eq.(16) Normalization")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
