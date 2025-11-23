#!/usr/bin/env python3
"""Generate stiffness profiles using GLOBAL T_K with FORCE COMPONENTS MAPPED TO ALWAYS NEGATIVE.

요청 변경: "양수면 음수로, 음수는 음수로 유지" → 모든 축에 대해 F_processed = -abs(F_signed).
즉:
    원래 F > 0  →  -F (음수로 반전)
    원래 F < 0  →  F  (그대로 유지)  (수식상 동일하게 -abs(F) 결과가 원래 음수와 동일)

Differences vs previous negative-only version:
1. 이전: positive 성분을 0으로 클램프(min(F,0)).
2. 현재: positive 성분을 음수로 반전하여 정보(크기)를 유지 (모든 값이 음수).
3. 옵션 --neg-mode magnitude: 압축 크기를 양의 크기(abs(-abs(F)))로 사용 → 결과는 abs(F_processed) = abs(F_signed).
4. Per-finger global T_F, T_K는 F_processed (또는 magnitude 모드 시 |F_processed|)에 기반.
5. Stiffness: K_finger = T_K_finger · EMG_variant + K_INIT.

Outputs:
    - PNG: raw signed force vs transformed all-negative (또는 magnitude) force, stiffness, EMG
    - CSV: per-finger transformed force & stiffness
    - Validation JSON (옵션)
    - Stiffness 통계 JSON (per-file + aggregate)

Usage examples:
    python3 generate_stiffness_profiles_global_tk_force_negative.py
    python3 generate_stiffness_profiles_global_tk_force_negative.py --neg-mode magnitude --validate
"""

from __future__ import annotations

import argparse
import json
import os
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
WORKSPACE_ROOT = PKG_DIR.parent.parent

EPS = 1e-8
K_INIT = 200.0
DEFAULT_INPUT = PKG_DIR / "outputs" / "logs" / "success"


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
    t_start = float(time[0]); t_end = float(time[-1])
    if not np.isfinite(t_start) or not np.isfinite(t_end) or np.isclose(t_end, t_start):
        return None
    src_times = np.linspace(t_start, t_end, num=emg.shape[0], dtype=float)
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
# Force & EMG processing
# ---------------------------------------------------------------------------

def _smooth_force(force: np.ndarray, fs: Optional[float], cutoff_hz: float = 3.0, order: int = 2) -> Optional[np.ndarray]:
    if force is None or force.size == 0:
        return None
    if fs is None or fs <= 0 or cutoff_hz <= 0 or fs <= 2 * cutoff_hz:
        return None
    try:
        sos = butter(order, cutoff_hz, btype="low", fs=fs, output="sos")
        return sosfiltfilt(sos, force, axis=0)
    except ValueError:
        return None


def _ultra_smooth_strong_emg(
    emg_mag: np.ndarray,
    fs: Optional[float],
    cutoff_hz: float = 1.0,
    window_sec: float = 0.8,
    sg_window_sec: float = 0.6,
    sg_poly: int = 3,
) -> Optional[np.ndarray]:
    if emg_mag is None or emg_mag.size == 0:
        return None
    sm = np.asarray(emg_mag, dtype=float)
    if fs is not None and fs > 0 and cutoff_hz > 0 and fs > 2 * cutoff_hz:
        try:
            sos = butter(2, cutoff_hz, btype="low", fs=fs, output="sos")
            sm = sosfiltfilt(sos, sm, axis=0)
        except ValueError:
            pass
    eff_fs = fs if fs is not None and fs > 0 else 200.0
    win = max(1, int(round(window_sec * eff_fs)))
    if win > 1:
        sm = uniform_filter1d(sm, size=win, axis=0, mode="nearest")
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
# Linear algebra helpers
# ---------------------------------------------------------------------------

def compute_tf(P: np.ndarray, F: np.ndarray, reg: float = 1e-6) -> np.ndarray:
    PPt = P @ P.T
    reg_eye = reg * np.eye(PPt.shape[0])
    return F @ P.T @ np.linalg.inv(PPt + reg_eye)


def compute_projection_from_tf(T_F: np.ndarray, reg: float = 1e-6) -> np.ndarray:
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
    dim = H_F.shape[0]
    H_F = 0.5 * (H_F + H_F.T)
    H_K = np.eye(dim) - H_F
    H_K = 0.5 * (H_K + H_K.T)
    eigvals, eigvecs = np.linalg.eigh(H_K)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]; eigvecs = eigvecs[:, idx]
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
    forces_negative_used: np.ndarray,
    stiffness_raw: np.ndarray,
    stiffness_filtered: np.ndarray,
    emg_raw: np.ndarray,
    emg_filtered: np.ndarray,
    out_path: Path,
    title: str,
    label_suffix: str,
) -> None:
    if time.ndim != 1 or time.size < 2:
        return
    fig, axes = plt.subplots(3, 1, figsize=(12, 8.0), sharex=True)
    ax_force, ax_stiff, ax_emg = axes
    colors = ["red", "green", "blue"]
    axis_labels = ["x", "y", "z"]
    # Raw signed forces
    for axis in range(min(3, forces_signed.shape[1])):
        color = colors[axis % len(colors)]; label = axis_labels[axis % len(axis_labels)]
        ax_force.plot(time, forces_signed[:, axis], color=color, linewidth=0.7, linestyle="--", alpha=0.25,
                      label=f"F_{label} raw" if axis == 0 else None)
        ax_force.plot(time, forces_negative_used[:, axis], color=color, linewidth=1.3, linestyle="-",
                      label=f"F_{label} neg" if axis == 0 else None)
    ax_force.set_ylabel("Force [N]"); ax_force.grid(alpha=0.3); ax_force.legend(loc="upper right", fontsize=8)
    # Stiffness
    for axis in range(min(3, stiffness_raw.shape[1])):
        color = colors[axis % len(colors)]; lbl = axis_labels[axis % len(axis_labels)]
        ax_stiff.plot(time, stiffness_raw[:, axis], color=color, linewidth=0.8, linestyle="--", alpha=0.2,
                      label=f"K_{lbl} raw" if axis == 0 else None)
        ax_stiff.plot(time, stiffness_filtered[:, axis], color=color, linewidth=1.4, linestyle="-",
                      label=f"K_{lbl} filt" if axis == 0 else None)
    ax_stiff.set_ylabel("Stiffness [N/m]"); ax_stiff.grid(alpha=0.3); ax_stiff.legend(loc="upper right", fontsize=8)
    # EMG
    emg_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
    for ch in range(min(emg_raw.shape[1], 8)):
        color = emg_colors[ch % len(emg_colors)]
        ax_emg.plot(time, emg_raw[:, ch], color=color, linewidth=0.7, linestyle="--", alpha=0.15,
                    label="EMG raw" if ch == 0 else None)
        ax_emg.plot(time, emg_filtered[:, ch], color=color, linewidth=1.1, linestyle="-",
                    label="EMG filt" if ch == 0 else None)
    ax_emg.set_ylabel("EMG"); ax_emg.set_xlabel("Time [s]"); ax_emg.grid(alpha=0.15); ax_emg.legend(loc="upper right", fontsize=8)
    fig.suptitle(f"{title} [NEG {label_suffix}]")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=160); plt.close(fig)


# ---------------------------------------------------------------------------
# Global T_K computation (negative-only forces)
# ---------------------------------------------------------------------------

def compute_global_tk_negative(input_dir: Path, neg_mode: str, verbose: bool = False) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    print(f"\n[Phase 1] Computing global T_K per finger (NEG mode={neg_mode})...")
    sensor_map = {"th": "s1", "if": "s2", "mf": "s3"}
    finger_data = {finger: {"emg": [], "force": []} for finger in sensor_map.keys()}
    csv_files = sorted(input_dir.glob("*.csv"))
    processed = 0
    for csv_path in csv_files:
        if csv_path.name.endswith("_paper_profile.csv"):
            continue
        try:
            if verbose: print(f"[dbg] file={csv_path.name} step=load")
            df = _load_csv(csv_path)
            if verbose: print(f"[dbg] file={csv_path.name} step=time cols={len(df.columns)}")
            time = _time_vector(df); fs = _estimate_fs(time) or 200.0
            if verbose: print(f"[dbg] file={csv_path.name} step=emg fs={fs:.2f}")
            emg_cols = _guess_emg_columns(df)
            if verbose: print(f"[dbg] file={csv_path.name} emg_cols={emg_cols}")
            if len(emg_cols) == 0:
                if verbose: print(f"[dbg] file={csv_path.name} no_emg_cols")
                continue
            emg_raw = df[emg_cols].to_numpy(dtype=float)
            if verbose: print(f"[dbg] file={csv_path.name} emg_raw_shape={emg_raw.shape}")
            aligned_emg = _align_emg_to_time(emg_raw, time)
            if aligned_emg is None:
                if verbose: print(f"[dbg] file={csv_path.name} aligned_emg=None")
                continue
            if verbose: print(f"[dbg] file={csv_path.name} aligned_shape={aligned_emg.shape}")
            if verbose: print(f"[dbg] file={csv_path.name} step=center_signal")
            emg_centered = _center_signal(aligned_emg)
            if verbose: print(f"[dbg] file={csv_path.name} emg_centered_shape={emg_centered.shape}")
            emg_mag = np.abs(emg_centered)
            if verbose: print(f"[dbg] file={csv_path.name} emg_mag_shape={emg_mag.shape}")
            # Apply smoothing safely without boolean "or" on arrays (avoid ambiguous truth-value)
            _tmp_emg = _ultra_smooth_strong_emg(emg_mag, fs)
            emg_smoothed = _tmp_emg if _tmp_emg is not None else emg_mag
            if verbose: print(f"[dbg] file={csv_path.name} emg_smoothed_shape={emg_smoothed.shape}")
            for finger, sensor in sensor_map.items():
                if verbose: print(f"[dbg] file={csv_path.name} finger_loop_start finger={finger}")
                force_cols = [f"{sensor}_fx", f"{sensor}_fy", f"{sensor}_fz"]
                memberships = [c in df.columns for c in force_cols]
                if verbose: print(f"[dbg] file={csv_path.name} finger={finger} force_cols={force_cols} memberships={memberships}")
                if not all(memberships):
                    if verbose: print(f"[dbg] file={csv_path.name} finger={finger} missing_force_cols")
                    continue
                if verbose: print(f"[dbg] file={csv_path.name} finger={finger} retrieving_forces")
                forces = df[force_cols].to_numpy(dtype=float)
                if verbose: print(f"[dbg] file={csv_path.name} finger={finger} forces_shape={forces.shape}")
                rest = max(1, int(len(forces) * 0.1))
                baseline = np.mean(forces[:rest, :], axis=0, keepdims=True)
                forces_centered = forces - baseline
                if verbose: print(f"[dbg] file={csv_path.name} finger={finger} centered_shape={forces_centered.shape}")
                _tmp_force = _smooth_force(forces_centered, fs)
                forces_smoothed = _tmp_force if _tmp_force is not None else forces_centered
                if verbose: print(f"[dbg] file={csv_path.name} finger={finger} smoothed_shape={forces_smoothed.shape}")
                all_neg = -np.abs(forces_smoothed)
                if neg_mode == "magnitude":
                    neg_used = np.abs(all_neg)
                else:
                    neg_used = all_neg
                if verbose: print(f"[dbg] file={csv_path.name} finger={finger} neg_used_shape={neg_used.shape}")
                finger_data[finger]["emg"].append(emg_smoothed)
                finger_data[finger]["force"].append(neg_used)
                if verbose: print(f"[dbg] file={csv_path.name} finger={finger} appended_force_count={len(finger_data[finger]['force'])}")
            processed += 1
            if verbose: print(f"[dbg] file={csv_path.name} processed_count={processed}")
        except Exception as exc:
            print(f"[skip] {csv_path.name}: {exc}")
            continue
    if processed == 0:
        raise ValueError("No valid demonstrations for negative-only T_K computation")
    print(f"[info] Loaded {processed} demonstrations (NEG mode={neg_mode})")
    global_tk = {}
    for finger in sensor_map.keys():
        if not finger_data[finger]["emg"]:
            raise ValueError(f"No data collected for finger {finger}")
        P_global = np.vstack(finger_data[finger]["emg"]).T  # (d, N_total)
        F_global = np.vstack(finger_data[finger]["force"]).T  # (3, N_total) negative-only
        print(f"[{finger}] Global data: EMG {P_global.shape}, F_neg {F_global.shape}")
        T_F = compute_tf(P_global, F_global)
        H_F = compute_projection_from_tf(T_F)
        T_K, H_K = compute_k_basis_from_force_projector(H_F, target_rank=3)
        global_tk[finger] = (T_K, H_K, T_F, H_F)
        print(f"[{finger}] T_K shape: {T_K.shape}, norm {np.linalg.norm(T_K):.4f}")
    return global_tk


# ---------------------------------------------------------------------------
# Process single file
# ---------------------------------------------------------------------------

def process_file_with_global_tk_negative(
    csv_path: Path,
    output_dir: Path,
    global_tk: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    neg_mode: str,
) -> Optional[List[Path]]:
    try:
        df = _load_csv(csv_path)
    except Exception as exc:
        print(f"[skip] {csv_path.name}: load failed ({exc})")
        return None
    time = _time_vector(df); fs = _estimate_fs(time) or 200.0
    deform_ecc = df["deform_ecc"].to_numpy(dtype=float) if "deform_ecc" in df else np.zeros(len(df))
    ee_if_px = df.get("ee_if_px", df.get("ee_px", pd.Series(np.zeros(len(df)))))
    ee_if_py = df.get("ee_if_py", df.get("ee_py", pd.Series(np.zeros(len(df)))))
    ee_if_pz = df.get("ee_if_pz", df.get("ee_pz", pd.Series(np.zeros(len(df)))))
    ee_mf_px = df.get("ee_mf_px", pd.Series(np.zeros(len(df))))
    ee_mf_py = df.get("ee_mf_py", pd.Series(np.zeros(len(df))))
    ee_mf_pz = df.get("ee_mf_pz", pd.Series(np.zeros(len(df))))
    ee_th_px = df.get("ee_th_px", pd.Series(np.zeros(len(df))))
    ee_th_py = df.get("ee_th_py", pd.Series(np.zeros(len(df))))
    ee_th_pz = df.get("ee_th_pz", pd.Series(np.zeros(len(df))))
    emg_cols = _guess_emg_columns(df)
    if len(emg_cols) == 0:
        print(f"[skip] {csv_path.name}: no EMG columns")
        return None
    emg_raw = df[emg_cols].to_numpy(dtype=float)
    aligned_emg = _align_emg_to_time(emg_raw, time)
    if aligned_emg is None:
        print(f"[skip] {csv_path.name}: EMG alignment failed")
        return None
    emg_centered = _center_signal(aligned_emg)
    emg_mag = np.abs(emg_centered)
    _tmp_emg = _ultra_smooth_strong_emg(emg_mag, fs)
    emg_smoothed = _tmp_emg if _tmp_emg is not None else emg_mag
    P_variant = emg_smoothed.T
    P_raw = emg_mag.T
    sensor_map = {"th": "s1", "if": "s2", "mf": "s3"}
    finger_names = list(sensor_map.keys())
    all_forces_signed = []
    all_forces_neg_used = []
    all_stiffness = []
    all_stiffness_raw = []
    stiffness_stats = {}
    for finger in finger_names:
        sensor = sensor_map[finger]
        T_K_finger = global_tk[finger][0]
        stiffness_finger = (T_K_finger @ P_variant).T + K_INIT
        stiffness_finger_raw = (T_K_finger @ P_raw).T + K_INIT
        force_cols = [f"{sensor}_fx", f"{sensor}_fy", f"{sensor}_fz"]
        if all(col in df.columns for col in force_cols):
            forces = df[force_cols].to_numpy(dtype=float)
            rest = max(1, int(len(forces) * 0.1))
            baseline = np.mean(forces[:rest, :], axis=0, keepdims=True)
            forces_centered = forces - baseline
            _tmp_force = _smooth_force(forces_centered, fs)
            forces_smoothed = _tmp_force if _tmp_force is not None else forces_centered
            # All-negative mapping per file processing (consistent with global phase)
            all_neg = -np.abs(forces_smoothed)
            neg_used = np.abs(all_neg) if neg_mode == "magnitude" else all_neg
        else:
            forces_smoothed = np.zeros((len(time), 3))
            neg_used = np.zeros((len(time), 3))
        all_forces_signed.append(forces_smoothed)
        all_forces_neg_used.append(neg_used)
        all_stiffness.append(stiffness_finger)
        all_stiffness_raw.append(stiffness_finger_raw)
        stats = {
            "k1_min": float(stiffness_finger[:, 0].min()),
            "k1_max": float(stiffness_finger[:, 0].max()),
            "k1_mean": float(stiffness_finger[:, 0].mean()),
            "k2_min": float(stiffness_finger[:, 1].min()),
            "k2_max": float(stiffness_finger[:, 1].max()),
            "k2_mean": float(stiffness_finger[:, 1].mean()),
            "k3_min": float(stiffness_finger[:, 2].min()),
            "k3_max": float(stiffness_finger[:, 2].max()),
            "k3_mean": float(stiffness_finger[:, 2].mean()),
        }
        stiffness_stats[finger] = stats
    # Plots
    out_paths: List[Path] = []
    for idx, finger_name in enumerate(finger_names):
        out_png = output_dir / f"{csv_path.stem}_{finger_name}_force_neg.png"
        try:
            _plot_variant(
                time,
                all_forces_signed[idx],
                all_forces_neg_used[idx],
                all_stiffness_raw[idx],
                all_stiffness[idx],
                emg_mag,
                emg_smoothed,
                out_png,
                csv_path.stem,
                finger_name,
            )
            out_paths.append(out_png)
        except Exception as exc:
            print(f"[warn] {csv_path.name}: PNG for {finger_name} failed ({exc})")
    # CSV
    out_csv = output_dir / f"{csv_path.stem}.csv"
    try:
        data = {"time_s": time.astype(float)}
        for i, finger_name in enumerate(finger_names):
            sensor = sensor_map[finger_name]
            data[f"{sensor}_fx_neg"] = all_forces_neg_used[i][:, 0].astype(float)
            data[f"{sensor}_fy_neg"] = all_forces_neg_used[i][:, 1].astype(float)
            data[f"{sensor}_fz_neg"] = all_forces_neg_used[i][:, 2].astype(float)
            data[f"{finger_name}_k1"] = all_stiffness[i][:, 0].astype(float)
            data[f"{finger_name}_k2"] = all_stiffness[i][:, 1].astype(float)
            data[f"{finger_name}_k3"] = all_stiffness[i][:, 2].astype(float)
        data["deform_ecc"] = deform_ecc.astype(float)
        data["ee_if_px"] = ee_if_px.to_numpy(dtype=float)
        data["ee_if_py"] = ee_if_py.to_numpy(dtype=float)
        data["ee_if_pz"] = ee_if_pz.to_numpy(dtype=float)
        data["ee_mf_px"] = ee_mf_px.to_numpy(dtype=float)
        data["ee_mf_py"] = ee_mf_py.to_numpy(dtype=float)
        data["ee_mf_pz"] = ee_mf_pz.to_numpy(dtype=float)
        data["ee_th_px"] = ee_th_px.to_numpy(dtype=float)
        data["ee_th_py"] = ee_th_py.to_numpy(dtype=float)
        data["ee_th_pz"] = ee_th_pz.to_numpy(dtype=float)
        out_df = pd.DataFrame(data)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(out_csv, index=False)
        out_paths.append(out_csv)
    except Exception as exc:
        print(f"[warn] {csv_path.name}: CSV save failed ({exc})")
    # Metrics JSON per file
    try:
        metrics_path = output_dir / f"{csv_path.stem}_stiffness_stats_neg.json"
        with open(metrics_path, "w") as f:
            json.dump(stiffness_stats, f, indent=2)
        out_paths.append(metrics_path)
    except Exception as exc:
        print(f"[warn] {csv_path.name}: metrics save failed ({exc})")
    print(f"[ok] {csv_path.name}: {len(out_paths)} files (NEG mode={neg_mode})")
    return out_paths


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate stiffness with global T_K using negative-only force components")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=PKG_DIR / "outputs" / "stiffness_profiles_force_negative")
    parser.add_argument("--neg-mode", choices=["signed", "magnitude"], default="signed",
                        help="signed: keep negative values (positives -> 0). magnitude: convert negative values to positive magnitudes")
    parser.add_argument("--validate", action="store_true", help="Run projector validation and save metrics JSON")
    parser.add_argument("--verbose", action="store_true", help="Enable detailed debug logging")
    args = parser.parse_args()
    # Resolve input directory
    candidates: List[Path] = []
    if isinstance(args.input, Path):
        candidates.append(args.input)
        if not args.input.is_absolute():
            candidates.extend([
                Path.cwd() / args.input,
                PKG_DIR / args.input,
                SCRIPT_DIR / args.input,
                WORKSPACE_ROOT / args.input,
            ])
    for c in candidates:
        if c.is_dir():
            input_dir = c; break
    else:
        raise ValueError(f"Input must be directory: {args.input}")
    # Phase 1
    global_tk = compute_global_tk_negative(input_dir, args.neg_mode, verbose=args.verbose)
    # Optional validation
    if args.validate:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        report = {}
        for finger, (T_K, H_K, T_F, H_F) in global_tk.items():
            ortho = np.linalg.norm(T_F @ T_K.T)
            HF_idem = np.linalg.norm(H_F @ H_F - H_F) / (np.linalg.norm(H_F) + EPS)
            HK_idem = np.linalg.norm(H_K @ H_K - H_K) / (np.linalg.norm(H_K) + EPS)
            HF_HK_orth = np.linalg.norm(H_F @ H_K) / ((np.linalg.norm(H_F) * np.linalg.norm(H_K)) + EPS)
            d = T_K.shape[1]; I = np.eye(d)
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
        with open(args.output_dir / "global_tk_force_neg_validation.json", "w") as f:
            json.dump(report, f, indent=2)
        print("[validate] Saved validation metrics to", args.output_dir / "global_tk_force_neg_validation.json")
    # Phase 2
    print(f"\n[Phase 2] Processing files with global T_K (NEG mode={args.neg_mode})...")
    csv_files = [p for p in sorted(input_dir.glob("*.csv")) if not p.name.endswith("_paper_profile.csv")]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    aggregate_stats = {}
    for csv_path in csv_files:
        process_file_with_global_tk_negative(csv_path, args.output_dir, global_tk, args.neg_mode)
    # Summarize stiffness ranges across all per-file metrics
    per_file_metrics = list(args.output_dir.glob("*_stiffness_stats_neg.json"))
    if per_file_metrics:
        import math
        merged = {"th": [], "if": [], "mf": []}
        for mp in per_file_metrics:
            try:
                with open(mp, "r") as f:
                    data = json.load(f)
                for finger in merged.keys():
                    if finger in data:
                        merged[finger].append(data[finger])
            except Exception:
                continue
        summary = {}
        for finger, entries in merged.items():
            if not entries:
                continue
            for axis in ["k1", "k2", "k3"]:
                mins = [e[f"{axis}_min"] for e in entries if f"{axis}_min" in e]
                maxs = [e[f"{axis}_max"] for e in entries if f"{axis}_max" in e]
                means = [e[f"{axis}_mean"] for e in entries if f"{axis}_mean" in e]
                if mins and maxs and means:
                    summary[f"{finger}_{axis}"] = {
                        "global_min": float(np.min(mins)),
                        "global_max": float(np.max(maxs)),
                        "mean_of_means": float(np.mean(means)),
                        "count": len(entries),
                    }
        with open(args.output_dir / "aggregate_stiffness_summary_neg.json", "w") as f:
            json.dump(summary, f, indent=2)
        print("[summary] Aggregate stiffness summary saved to", args.output_dir / "aggregate_stiffness_summary_neg.json")
    print("\n[Done] All files processed with GLOBAL T_K using negative-only forces.")
    print("Output directory:", args.output_dir)


if __name__ == "__main__":
    main()
