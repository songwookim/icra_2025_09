#!/usr/bin/env python3
"""Generate stiffness profiles using GLOBAL T_K (same transformation for all demos).

This version computes T_K once from all demonstrations combined, then applies
the same T_K to all demos for consistent EMG → Stiffness mapping.
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
VALIDATE_RESULTS = True


def compute_global_tk(input_dir: Path) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Compute global T_K for each finger from all demonstrations combined.
    
    Returns:
        Dict with keys 'th', 'if', 'mf', each containing (T_K, H_K, T_F, H_F) tuple
        T_K: (3, d) transformation matrix from EMG to stiffness
        H_K: (d, d) stiffness subspace projector
        T_F: (3, d) transformation matrix from EMG to force
        H_F: (d, d) force subspace projector
    """
    print("\n[Phase 1] Computing global T_K per finger from all demonstrations...")
    
    sensor_map = {"th": "s1", "if": "s2", "mf": "s3"}
    finger_data = {finger: {"emg": [], "force": []} for finger in ["th", "if", "mf"]}
    
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
    
    # Compute T_K for each finger
    global_tk = {}
    for finger in ["th", "if", "mf"]:
        if not finger_data[finger]["emg"]:
            raise ValueError(f"No data collected for finger {finger}")
        
        P_global = np.vstack(finger_data[finger]["emg"]).T  # (d, N_total)
        F_list = finger_data[finger]["force"]  # each (N_i, 3)
        F_global = np.vstack(F_list).T  # (3, N_total)
        
        print(f"[{finger}] Global data: EMG {P_global.shape}, Force {F_global.shape}")
        
        T_F = compute_tf(P_global, F_global)
        H_F = compute_projection_from_tf(T_F)
        T_K, H_K = compute_k_basis_from_force_projector(H_F, target_rank=3)
        
        global_tk[finger] = (T_K, H_K, T_F, H_F)
        print(f"[{finger}] T_K shape: {T_K.shape}, norm: {np.linalg.norm(T_K):.4f}")
    
    return global_tk


def process_file_with_global_tk(
    csv_path: Path,
    output_dir: Path,
    global_tk: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
) -> Optional[List[Path]]:
    """Process single file using global T_K per finger.
    
    Args:
        global_tk: Dict with keys 'th', 'if', 'mf', each containing (T_K, H_K, T_F, H_F)
    """
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

    # Prepare EMG features
    P_variant = emg_smoothed.T
    P_raw = emg_magnitude.T

    out_paths: List[Path] = []

    # Process each finger with its own global T_K
    finger_names = ["th", "if", "mf"]
    sensor_names = ["s1", "s2", "s3"]
    all_forces = []
    all_forces_csv = []
    all_stiffness = []
    all_stiffness_raw = []
    all_fn_abs = []
    
    for finger, sensor in zip(finger_names, sensor_names):
        # Get this finger's global T_K
        T_K_finger = global_tk[finger][0]
        
        # Compute stiffness for this finger using its T_K (allow negative values)
        stiffness_finger = (T_K_finger @ P_variant).T + K_INIT
        stiffness_finger_raw = (T_K_finger @ P_raw).T + K_INIT
        
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
            
            # For plotting: use raw forces (not absolute value)
            all_forces.append(forces_smoothed)

            Fn = _compute_normal_force(df, prefix=sensor, n=(0.0, 0.0, 1.0))
            Fn_smoothed = _smooth_force(Fn, fs)
            if Fn_smoothed is None:
                Fn_smoothed = Fn
            all_fn_abs.append(Fn_smoothed)  # Use raw Fn (not absolute value)
            
            # Use finger-specific stiffness (both smoothed and raw)
            all_stiffness.append(stiffness_finger)
            all_stiffness_raw.append(stiffness_finger_raw)
        else:
            all_forces.append(np.zeros((len(time), 3)))
            all_forces_csv.append(np.zeros((len(time), 3)))
            all_stiffness.append(stiffness_finger)
            all_stiffness_raw.append(stiffness_finger_raw)
            all_fn_abs.append(np.zeros(len(time)))
    
    # Generate plots (use each finger's own raw stiffness for comparison)
    for idx, finger_name in enumerate(finger_names):
        out_png = output_dir / f"{csv_path.stem}_{finger_name}_global_tk.png"
        try:
            _plot_variant(
                time,
                all_forces[idx],
                all_forces[idx],
                all_fn_abs[idx],
                all_stiffness_raw[idx],  # raw (finger-specific, for comparison)
                all_stiffness[idx],  # smoothed (finger-specific)
                emg_magnitude,
                emg_smoothed,
                out_png,
                csv_path.stem,
                f"{finger_name} GLOBAL-TK",
            )
            out_paths.append(out_png)
        except Exception as exc:
            print(f"[warn] {csv_path.name}: PNG for {finger_name} failed ({exc})")

    # Save CSV (use _paper_profile suffix for compatibility with benchmark script)
    out_csv = output_dir / f"{csv_path.stem}.csv"
    try:
        all_forces_combined = np.hstack(all_forces_csv)
        data = {"time_s": time.astype(float)}
        
        for i, (finger_name, sensor) in enumerate(zip(finger_names, sensor_names)):
            data[f"{sensor}_fx"] = all_forces_combined[:, i*3+0].astype(float)
            data[f"{sensor}_fy"] = all_forces_combined[:, i*3+1].astype(float)
            data[f"{sensor}_fz"] = all_forces_combined[:, i*3+2].astype(float)
            data[f"{finger_name}_k1"] = all_stiffness[i][:, 0].astype(float)
            data[f"{finger_name}_k2"] = all_stiffness[i][:, 1].astype(float)
            data[f"{finger_name}_k3"] = all_stiffness[i][:, 2].astype(float)
        
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

    print(f"[ok] {csv_path.name}: {len(out_paths)} files (GLOBAL T_K)")
    return out_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate stiffness with global T_K")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PKG_DIR / "outputs" / "stiffness_profiles",
    )
    parser.add_argument("--validate", action="store_true", help="Run geometric/projector validation and save metrics JSON")
    args = parser.parse_args()
    
    # Resolve input directory with fallbacks
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
    
    # Phase 1: Compute global T_K per finger
    global_tk = compute_global_tk(input_dir)
    
    # Optional validation of subspace geometry
    if args.validate:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        report = {}
        for finger, (T_K, H_K, T_F, H_F) in global_tk.items():
            # Orthogonality between force map and stiffness basis
            ortho = np.linalg.norm(T_F @ T_K.T)
            # Projector properties
            HF_idem = np.linalg.norm(H_F @ H_F - H_F) / (np.linalg.norm(H_F) + EPS)
            HK_idem = np.linalg.norm(H_K @ H_K - H_K) / (np.linalg.norm(H_K) + EPS)
            HF_HK_orth = np.linalg.norm(H_F @ H_K) / ((np.linalg.norm(H_F) * np.linalg.norm(H_K)) + EPS)
            # Complementarity
            d = T_K.shape[1]
            I = np.eye(d)
            comp = np.linalg.norm((H_F + H_K) - I) / (np.linalg.norm(I) + EPS)
            report[finger] = {
                "norm_TF_TK_T": float(ortho),
                "HF_idempotency": float(HF_idem),
                "HK_idempotency": float(HK_idem),
                "HF_HK_orthogonality": float(HF_HK_orth),
                "complementarity_HF_plus_HK_approx_I": float(comp),
                "shapes": {
                    "T_F": list(T_F.shape),
                    "T_K": list(T_K.shape),
                    "H_F": list(H_F.shape),
                    "H_K": list(H_K.shape),
                },
            }
        import json
        with open(args.output_dir / "global_tk_validation.json", "w") as f:
            json.dump(report, f, indent=2)
        print("[validate] Saved validation metrics to", args.output_dir / "global_tk_validation.json")
    
    # Phase 2: Process all files with global T_K
    print(f"\n[Phase 2] Processing files with global T_K...")
    csv_files = [p for p in sorted(input_dir.glob("*.csv")) if not p.name.endswith("_paper_profile.csv")]
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    for csv_path in csv_files:
        process_file_with_global_tk(csv_path, args.output_dir, global_tk)
    
    print(f"\n[Done] All files processed with GLOBAL T_K")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
