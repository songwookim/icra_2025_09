#!/usr/bin/env python3
"""Generate stiffness profiles using EMG projection method (paper version).

For each demonstration CSV this script:
- extracts force magnitudes (baseline-removed, no low-pass filtering),
- aligns sEMG signals and builds transformation matrices T_F and T_K,
- projects multiple filtered EMG variants (raw magnitude, low-pass, moving-average,
    band-pass, and an ultra-smooth envelope) into stiffness trajectories, and
- saves comparison plots plus a CSV with the force traces and low-pass stiffness.

Four PNGs are emitted per input file, one per filtered EMG variant. Each PNG has
subplots for force magnitude, stiffness (raw-EMG dashed vs filtered solid), and EMG
(raw dashed vs filtered solid) with a small horizontal margin.
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
from scipy.signal import butter, sosfiltfilt

if os.environ.get("DISPLAY", "") == "":
    import matplotlib
    # Headless 환경에서는 Agg 백엔드 사용 (TKAgg는 시스템 의존성이 필요함)
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
PKG_DIR = SCRIPT_DIR.parent  # .../src/hri_falcon_robot_bridge
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

EPS = 1e-8
# 패키지 내부 outputs/logs/success 만 대상으로 사용
DEFAULT_INPUT = (PKG_DIR / "outputs" / "logs" / "success")
K_INIT = 200.0

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

def _compute_signed_force(df: pd.DataFrame) -> np.ndarray:
    force_prefixes = ("force_3", "s3", "force3")
    picked: Optional[List[str]] = None
    for prefix in force_prefixes:
        cand = [f"{prefix}_fx", f"{prefix}_fy", f"{prefix}_fz"]
        if all(col in df.columns for col in cand):
            picked = cand
            break
    if picked is None:
        raise KeyError("No force_3 columns found (expected fx, fy, fz)")
    force_raw = df[picked].to_numpy(dtype=float)
    rest_samples = max(1, int(len(force_raw) * 0.1))
    baseline = np.mean(force_raw[:rest_samples, :], axis=0, keepdims=True)
    forces_signed = np.abs(force_raw - baseline)
    return forces_signed


# ---------------------------------------------------------------------------
# EMG filtering helpers
# ---------------------------------------------------------------------------

def _lowpass_emg(emg: np.ndarray, fs: Optional[float], cutoff_hz: float = 5.0, order: int = 4) -> Optional[np.ndarray]:
    if emg is None or emg.size == 0:
        return None
    if fs is None or fs <= 0 or cutoff_hz <= 0 or fs <= 2 * cutoff_hz:
        return None
    sos = butter(order, cutoff_hz, btype="low", fs=fs, output="sos")
    return sosfiltfilt(sos, emg, axis=0)


def _moving_average_emg(emg: np.ndarray, fs: Optional[float], window_sec: float = 0.1) -> Optional[np.ndarray]:
    if emg is None or emg.size == 0:
        return None
    if fs is None or fs <= 0:
        window = max(1, int(round(window_sec * 200.0)))
    else:
        window = max(1, int(round(window_sec * fs)))
    if window <= 1:
        return emg.copy()
    return uniform_filter1d(emg, size=window, axis=0, mode="nearest")


def _bandpass_emg(emg: np.ndarray, fs: Optional[float], low_hz: float = 20.0, high_hz: float = 450.0, order: int = 4) -> Optional[np.ndarray]:
    if emg is None or emg.size == 0:
        return None
    if fs is None or fs <= 0:
        return None
    nyquist = 0.5 * fs
    high = min(high_hz, nyquist - 1e-3)
    low = max(low_hz, 1e-3)
    if high <= low or high <= 0:
        return None
    sos = butter(order, [low, high], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, emg, axis=0)


def _ultra_smooth_emg(emg_mag: np.ndarray, fs: Optional[float], cutoff_hz: float = 2.0, window_sec: float = 0.35) -> Optional[np.ndarray]:
    if emg_mag is None or emg_mag.size == 0:
        return None
    smoothed = np.asarray(emg_mag, dtype=float)
    if fs is not None and fs > 0 and cutoff_hz > 0 and fs > 2 * cutoff_hz:
        try:
            sos = butter(2, cutoff_hz, btype="low", fs=fs, output="sos")
            smoothed = sosfiltfilt(sos, smoothed, axis=0)
        except ValueError:
            pass
    effective_fs = fs if fs is not None and fs > 0 else 200.0
    window = max(1, int(round(window_sec * effective_fs)))
    if window > 1:
        smoothed = uniform_filter1d(smoothed, size=window, axis=0, mode="nearest")
    smoothed = np.maximum(smoothed, 0.0)
    return smoothed


# ---------------------------------------------------------------------------
# Linear-algebra helpers for T_F / T_K
# ---------------------------------------------------------------------------

def compute_tf(P: np.ndarray, F: np.ndarray, reg: float = 1e-6) -> np.ndarray:
    """Solve T_F from F = T_F · P using ridge regularization."""
    PPt = P @ P.T
    reg_eye = reg * np.eye(PPt.shape[0])
    return F @ P.T @ np.linalg.inv(PPt + reg_eye)


def compute_projection_from_tf(T_F: np.ndarray, reg: float = 1e-6) -> np.ndarray:
    TF_TF_T = T_F @ T_F.T
    reg_eye = reg * np.eye(TF_TF_T.shape[0])
    inv_term = np.linalg.inv(TF_TF_T + reg_eye)
    return T_F.T @ inv_term @ T_F


def compute_tk_from_projection(H_F: np.ndarray, target_rank: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    dim = H_F.shape[0]
    H_K = np.eye(dim) - H_F
    eigvals, eigvecs = np.linalg.eigh(H_K)
    idx = np.argsort(eigvals)[::-1]
    significant = [i for i in idx if eigvals[i] > 1e-6]
    if len(significant) < target_rank:
        raise ValueError("Complementary subspace has insufficient rank for T_K")
    basis = eigvecs[:, significant[:target_rank]].T
    return basis, H_K


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_variant(
    time: np.ndarray,
    forces_signed: np.ndarray,
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

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for axis in range(min(3, forces_signed.shape[1])):
        ax_force.plot(
            time,
            forces_signed[:, axis],
            color=colors[axis % len(colors)],
            linewidth=1.2,
            label=f"|F{axis + 1}|",
        )
    ax_force.set_ylabel("Force magnitude [N]")
    ax_force.grid(alpha=0.3)
    ax_force.legend(loc="upper right", fontsize=8)

    for axis in range(min(3, stiffness_raw.shape[1])):
        color = colors[axis % len(colors)]
        ax_stiff.plot(time, stiffness_raw[:, axis], color=color, linewidth=1.0, linestyle="--", alpha=0.1, label=f"K{axis + 1} raw")
        ax_stiff.plot(time, stiffness_filtered[:, axis], color=color, linewidth=1.3, linestyle="-", label=f"K{axis + 1} {filter_label}")
    ax_stiff.set_ylabel("Stiffness [N/m]")
    ax_stiff.grid(alpha=0.3)
    ax_stiff.legend(loc="upper right", fontsize=8)

    palette = plt.rcParams.get("axes.prop_cycle", None)
    emg_colors = palette.by_key().get("color", []) if palette else []
    if not emg_colors:
        emg_colors = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
        ]
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
    xmin = float(time[0]) - pad
    xmax = float(time[-1]) + pad
    ax_force.set_xlim(xmin, xmax)

    fig.suptitle(f"{title} [{filter_label}]")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Output writer
# ---------------------------------------------------------------------------

def _save_profile_csv(
    out_path: Path,
    time: np.ndarray,
    forces_signed: np.ndarray,
    stiffness_lp: np.ndarray,
) -> None:
    data = {
        "time_s": np.asarray(time, dtype=float),
        "Fx": forces_signed[:, 0].astype(float),
        "Fy": forces_signed[:, 1].astype(float),
        "Fz": forces_signed[:, 2].astype(float),
        "Kx_lp": stiffness_lp[:, 0].astype(float),
        "Ky_lp": stiffness_lp[:, 1].astype(float),
        "Kz_lp": stiffness_lp[:, 2].astype(float),
    }
    out_df = pd.DataFrame(data)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)


# ---------------------------------------------------------------------------
# Main processing per file
# ---------------------------------------------------------------------------

def process_file(csv_path: Path, output_dir: Path) -> Optional[List[Path]]:
    try:
        df = _load_csv(csv_path)
    except Exception as exc:
        print(f"[skip] {csv_path.name}: load failed ({exc})")
        return None

    time = _time_vector(df)
    fs = _estimate_fs(time)
    if fs is None:
        fs = 200.0
        print(f"[warn] {csv_path.name}: sampling rate unknown, assuming {fs:.1f} Hz")

    try:
        forces_signed = _compute_signed_force(df)
    except Exception as exc:
        print(f"[skip] {csv_path.name}: force extraction failed ({exc})")
        return None

    emg_cols = _guess_emg_columns(df)
    if not emg_cols:
        print(f"[skip] {csv_path.name}: no EMG columns detected")
        return None

    try:
        emg_raw_input = df[emg_cols].to_numpy(dtype=float)
    except Exception as exc:
        print(f"[skip] {csv_path.name}: EMG extraction failed ({exc})")
        return None

    aligned_emg = _align_emg_to_time(emg_raw_input, time)
    if aligned_emg is None:
        print(f"[skip] {csv_path.name}: unable to align EMG to force timeline")
        return None

    emg_centered = _center_signal(aligned_emg)
    emg_magnitude = np.abs(emg_centered)

    lp_emg = _lowpass_emg(emg_centered, fs)
    if lp_emg is not None:
        lp_emg = np.abs(lp_emg)

    ma_emg = _moving_average_emg(emg_centered, fs)
    if ma_emg is not None:
        ma_emg = np.abs(ma_emg)

    bp_emg = _bandpass_emg(emg_centered, fs)
    if bp_emg is not None:
        bp_emg = np.abs(bp_emg)

    ultra_emg = _ultra_smooth_emg(emg_magnitude, fs)

    variants: Dict[str, np.ndarray] = {}
    variants["lowpass"] = lp_emg if lp_emg is not None else emg_magnitude
    variants["moving_avg"] = ma_emg if ma_emg is not None else emg_magnitude
    if bp_emg is not None:
        variants["bandpass"] = bp_emg
    else:
        print(f"[warn] {csv_path.name}: band-pass filter unavailable (fs={fs:.2f} Hz)")
    variants["ultra_smooth"] = ultra_emg if ultra_emg is not None else emg_magnitude

    P_raw = emg_magnitude.T
    F_mat = forces_signed.T

    try:
        T_F = compute_tf(P_raw, F_mat)
        H_F = compute_projection_from_tf(T_F)
        T_K, _ = compute_tk_from_projection(H_F)
    except Exception as exc:
        print(f"[skip] {csv_path.name}: projection computation failed ({exc})")
        return None

    stiffness_raw = (T_K @ P_raw).T + K_INIT

    out_paths: List[Path] = []

    for variant_name, emg_variant in variants.items():
        P_variant = emg_variant.T
        stiffness_variant = (T_K @ P_variant).T + K_INIT
        out_png = output_dir / f"{csv_path.stem}_{variant_name}_paper.png"
        label = {
            "lowpass": "low-pass",
            "moving_avg": "moving-avg",
            "bandpass": "band-pass",
            "ultra_smooth": "ultra-smooth",
        }.get(variant_name, variant_name)
        try:
            _plot_variant(
                time,
                forces_signed,
                stiffness_raw,
                stiffness_variant,
                emg_magnitude,
                emg_variant,
                out_png,
                csv_path.stem,
                label,
            )
            out_paths.append(out_png)
        except Exception as exc:
            print(f"[warn] {csv_path.name}: plot for {variant_name} failed ({exc})")

        if variant_name == "lowpass":
            out_csv = output_dir / f"{csv_path.stem}_paper_profile.csv"
            try:
                _save_profile_csv(out_csv, time, forces_signed, stiffness_variant)
                out_paths.append(out_csv)
            except Exception as exc:
                print(f"[warn] {csv_path.name}: CSV save failed ({exc})")

    print(f"[ok] {csv_path.name}: generated {len(out_paths)} artifact(s)")
    for path in out_paths:
        print(f"       -> {path}")
    return out_paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def gather_csvs(input_path: Path) -> List[Path]:
    """Gather CSV files.
    - If input_path is a file: return it.
    - If input_path is a directory: gather recursively (support date subfolders).
    - If path doesn't exist: error out (we only process explicit/success dir).
    """
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        # Recursively collect CSVs (including dated subfolders)
        csvs = sorted([p for p in input_path.rglob("*.csv") if p.is_file()])
        return csvs
    raise FileNotFoundError(f"Input path not found: {input_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate stiffness profiles (paper ver.)")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
    help="CSV file or directory of logs (defaults to <pkg>/outputs/logs/success). Supports recursion.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(PKG_DIR / "outputs" / "analysis" / "stiffness_profiles"),
        help="Directory for output artifacts (defaults to <pkg>/outputs/analysis/stiffness_profiles)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_files = gather_csvs(args.input)
    if not csv_files:
        raise SystemExit(f"No CSV files found under {args.input}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Processing {len(csv_files)} file(s) from {args.input} ...")
    for csv_path in csv_files:
        process_file(csv_path, args.output_dir)


if __name__ == "__main__":
    main()
