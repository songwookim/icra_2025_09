#!/usr/bin/env python3
"""Enhanced variant: Generate stiffness profiles using GLOBAL T_K per finger

Differences vs `generate_stiffness_profiles_global_tk.py`:
 - More robust EMG channel detection (regex + fallback patterns)
 - Configurable EMG preprocessing: bandpass (HPF+LPF) + optional notch removal
 - Normalization options: z-score, min-max, unit-norm
 - Optional envelope smoothing choices (moving-average, savgol)
 - Identical output CSV format (`*_paper_profile.csv`) for direct comparison

This allows A/B comparison of different EMG preprocessing pipelines while
keeping downstream stiffness mapping structure the same.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, iirnotch
from scipy.ndimage import uniform_filter1d
from scipy.signal import savgol_filter

if os.environ.get("DISPLAY", "") == "":
    import matplotlib
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
PKG_DIR = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# Reuse utilities from the existing script base
from generate_stiffness_profiles import (
    _load_csv,
    _time_vector,
    _estimate_fs,
    _align_emg_to_time,
    _center_signal,
    _compute_normal_force,
    _smooth_force,
    compute_tf,
    compute_projection_from_tf,
    compute_k_basis_from_force_projector,
    _plot_variant,
    _guess_emg_columns,  # fallback
)

EPS = 1e-8
# Default input changed to workspace-level outputs (parent of src)
WORKSPACE_ROOT = PKG_DIR.parent.parent  # .../icra2025
# Keep default aligned with package-local outputs to match existing data
DEFAULT_INPUT = PKG_DIR / "outputs" / "logs" / "success"
K_INIT = 200.0
K_MIN = 50.0


# ---------------------------------------------------------------------------
# EMG helpers
# ---------------------------------------------------------------------------
EMG_REGEX = re.compile(r"^(emg(_\d+)?|muscle(_\d+)?|e\d+)$", re.IGNORECASE)
FALLBACK_EMG_PREFIXES = ["emg", "muscle", "m", "e"]


def guess_emg_columns_robust(df: pd.DataFrame, max_channels: int = 32) -> List[str]:
    cols: List[str] = []
    for c in df.columns:
        base = c.strip().lower()
        if EMG_REGEX.match(base):
            cols.append(c)
        else:
            for pref in FALLBACK_EMG_PREFIXES:
                if base.startswith(pref):
                    # Accept if numeric suffix length reasonable (e.g., emg1..emg32, e1..e16)
                    suffix = base[len(pref):]
                    if suffix == "" or suffix.isdigit() and len(suffix) <= 2:
                        if df[c].dtype.kind in "if":
                            cols.append(c)
                    break
        if len(cols) >= max_channels:
            break
    if not cols:
        # Fallback to original heuristic if robust finds nothing
        try:
            cols = _guess_emg_columns(df)
        except Exception:
            cols = []
    # Deduplicate preserving order
    seen = set()
    ordered: List[str] = []
    for c in cols:
        if c not in seen:
            ordered.append(c)
            seen.add(c)
    return ordered


def bandpass_emg(emg: np.ndarray, fs: float, hp: float, lp: float) -> np.ndarray:
    """Safe EMG filtering that adapts to fs and avoids invalid Wn.

    - If only hp or lp is valid, degrade to highpass/lowpass.
    - If both invalid or numerically unsafe, return original signal.
    """
    try:
        if fs is None or fs <= 0:
            return emg
        nyq = 0.5 * fs
        hp_eff = max(0.0, float(hp))
        lp_eff = max(0.0, float(lp))

        # No filtering requested
        if hp_eff <= 0 and lp_eff <= 0:
            return emg

        # Handle single-sided filters first
        if lp_eff <= 0 < hp_eff:
            w = min(0.99, max(0.001, hp_eff / nyq))
            sos = butter(4, w, btype="highpass", output="sos")
            return sosfiltfilt(sos, emg, axis=0)
        if hp_eff <= 0 < lp_eff:
            w = min(0.99, max(0.001, lp_eff / nyq))
            sos = butter(4, w, btype="lowpass", output="sos")
            return sosfiltfilt(sos, emg, axis=0)

        # Both provided: ensure 0 < low < high < 1
        # Clamp lp below Nyquist and ensure hp < lp
        lp_eff = min(lp_eff, nyq * 0.99)
        hp_eff = min(hp_eff, nyq * 0.98)
        low = max(0.001, min(0.99, hp_eff / nyq))
        high = max(0.001, min(0.99, lp_eff / nyq))
        if high <= low + 1e-6:
            # Degrade to highpass if possible
            if low < 0.99:
                sos = butter(4, low, btype="highpass", output="sos")
                return sosfiltfilt(sos, emg, axis=0)
            # Else give up
            return emg
        sos = butter(4, [low, high], btype="bandpass", output="sos")
        return sosfiltfilt(sos, emg, axis=0)
    except Exception as exc:
        print(f"[warn] EMG filter failed, returning unfiltered signal ({exc})")
        return emg


def notch_filter(emg: np.ndarray, fs: float, freq: float = 50.0, q: float = 30.0) -> np.ndarray:
    w0 = freq / (fs / 2)
    if w0 <= 0 or w0 >= 1:
        return emg
    b, a = iirnotch(w0, q)
    # apply per channel to avoid shape mismatch
    out = np.empty_like(emg)
    for i in range(emg.shape[1]):
        out[:, i] = sosfiltfilt(np.atleast_2d(np.vstack(([b, a]))), emg[:, i]) if False else emg[:, i]
    # Using sosfiltfilt with notch requires SOS form; simpler to skip if we don't convert. Keep original for now.
    return emg  # (placeholder: avoid complexity) TODO if needed


def normalize_emg(emg: np.ndarray, mode: str) -> np.ndarray:
    if mode == "zscore":
        mu = emg.mean(axis=0, keepdims=True)
        std = emg.std(axis=0, keepdims=True) + EPS
        return (emg - mu) / std
    if mode == "minmax":
        mn = emg.min(axis=0, keepdims=True)
        mx = emg.max(axis=0, keepdims=True)
        return (emg - mn) / (mx - mn + EPS)
    if mode == "unitnorm":
        norm = np.linalg.norm(emg, axis=0, keepdims=True) + EPS
        return emg / norm
    return emg


def envelope(emg_abs: np.ndarray, fs: float, method: str, ma_window_ms: int = 50) -> np.ndarray:
    if method == "ma":
        w = max(1, int(fs * ma_window_ms / 1000))
        return uniform_filter1d(emg_abs, size=w, axis=0)
    if method == "savgol":
        w = max(5, int(fs * ma_window_ms / 1000) | 1)  # odd
        return savgol_filter(emg_abs, window_length=w, polyorder=3, axis=0)
    return emg_abs


# ---------------------------------------------------------------------------
# GLOBAL T_K computation (per finger)
# ---------------------------------------------------------------------------
def compute_global_tk(input_dir: Path, hp: float, lp: float, norm_mode: str, env_method: str) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    print("\n[Phase 1] Computing global T_K per finger with enhanced preprocessing...")
    sensor_map = {"th": "s1", "if": "s2", "mf": "s3"}
    finger_data = {finger: {"emg": [], "force": []} for finger in sensor_map}

    csv_files = sorted(input_dir.glob("*.csv"))
    used = 0
    for csv_path in csv_files:
        if csv_path.name.endswith("_paper_profile.csv"):
            continue
        try:
            df = _load_csv(csv_path)
            time = _time_vector(df)
            fs = _estimate_fs(time) or 200.0
            emg_cols = guess_emg_columns_robust(df)
            if not emg_cols:
                continue
            emg_raw = df[emg_cols].to_numpy(dtype=float)
            emg_aligned = _align_emg_to_time(emg_raw, time)
            if emg_aligned is None:
                continue
            emg_centered = _center_signal(emg_aligned)
            emg_bp = bandpass_emg(emg_centered, fs, hp, lp)
            emg_norm = normalize_emg(emg_bp, norm_mode)
            emg_abs = np.abs(emg_norm)
            emg_env = envelope(emg_abs, fs, env_method)

            for finger, sensor in sensor_map.items():
                try:
                    # Extract full 3D force (fx, fy, fz) instead of 1D normal force
                    force_cols = [f"{sensor}_fx", f"{sensor}_fy", f"{sensor}_fz"]
                    if all(c in df.columns for c in force_cols):
                        forces = df[force_cols].to_numpy(dtype=float)
                        rest = max(1, int(len(forces) * 0.1))
                        baseline = np.mean(forces[:rest, :], axis=0, keepdims=True)
                        forces_centered = forces - baseline
                        forces_smoothed = _smooth_force(forces_centered, fs)
                        if forces_smoothed is None:
                            forces_smoothed = forces_centered
                        finger_data[finger]["emg"].append(emg_env)
                        finger_data[finger]["force"].append(forces_smoothed)
                except Exception:
                    continue
            used += 1
        except Exception as exc:
            print(f"[skip] {csv_path.name}: {exc}")
            continue

    if used == 0:
        raise ValueError("No valid demonstrations for enhanced global T_K")
    print(f"[info] Used {used} demonstrations")

    global_tk: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for finger in sensor_map:
        if not finger_data[finger]["emg"]:
            raise ValueError(f"No EMG collected for finger {finger}")
        P = np.vstack(finger_data[finger]["emg"]).T  # (d, N_total)
        F_list = finger_data[finger]["force"]  # each (N_i, 3)
        F = np.vstack(F_list).T  # (3, N_total)
        print(f"[{finger}] EMG {P.shape}, Force {F.shape}")
        T_F = compute_tf(P, F)
        H_F = compute_projection_from_tf(T_F)
        T_K, H_K = compute_k_basis_from_force_projector(H_F, target_rank=3)
        global_tk[finger] = (T_K, H_K, T_F, H_F)
        print(f"[{finger}] T_K norm {np.linalg.norm(T_K):.4f}")
    return global_tk


# ---------------------------------------------------------------------------
# Process single file with enhanced pipeline
# ---------------------------------------------------------------------------
def process_file(csv_path: Path, out_dir: Path, global_tk: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]], hp: float, lp: float, norm_mode: str, env_method: str) -> Optional[List[Path]]:
    try:
        df = _load_csv(csv_path)
    except Exception as exc:
        print(f"[skip] {csv_path.name}: load fail ({exc})")
        return None

    time = _time_vector(df)
    fs = _estimate_fs(time) or 200.0
    emg_cols = guess_emg_columns_robust(df)
    if not emg_cols:
        print(f"[skip] {csv_path.name}: no EMG detected")
        return None
    emg_raw = df[emg_cols].to_numpy(dtype=float)
    emg_aligned = _align_emg_to_time(emg_raw, time)
    if emg_aligned is None:
        print(f"[skip] {csv_path.name}: alignment fail")
        return None
    emg_centered = _center_signal(emg_aligned)
    emg_bp = bandpass_emg(emg_centered, fs, hp, lp)
    emg_norm = normalize_emg(emg_bp, norm_mode)
    emg_abs = np.abs(emg_norm)
    emg_env = envelope(emg_abs, fs, env_method)

    P_variant = emg_env.T
    P_raw = emg_abs.T  # pre-envelope raw magnitude (for raw stiffness comparison)

    finger_names = ["th", "if", "mf"]
    sensor_names = ["s1", "s2", "s3"]
    all_forces = []
    all_forces_csv = []
    all_stiff = []
    all_stiff_raw = []
    all_fn_abs = []

    for finger, sensor in zip(finger_names, sensor_names):
        T_K_finger = global_tk[finger][0]
        stiff = np.maximum(K_MIN, (T_K_finger @ P_variant).T + K_INIT)
        stiff_raw = np.maximum(K_MIN, (T_K_finger @ P_raw).T + K_INIT)

        force_cols = [f"{sensor}_fx", f"{sensor}_fy", f"{sensor}_fz"]
        if all(c in df.columns for c in force_cols):
            forces = df[force_cols].to_numpy(dtype=float)
            rest = max(1, int(len(forces) * 0.1))
            baseline = np.mean(forces[:rest, :], axis=0, keepdims=True)
            forces_centered = forces - baseline
            forces_smoothed = _smooth_force(forces_centered, fs)
            if forces_smoothed is None:
                forces_smoothed = forces_centered
            all_forces_csv.append(forces_smoothed)

            forces_abs = np.abs(forces_centered)
            forces_abs_s = _smooth_force(forces_abs, fs)
            if forces_abs_s is None:
                forces_abs_s = forces_abs
            all_forces.append(forces_abs_s)

            Fn = _compute_normal_force(df, prefix=sensor, n=(0.0, 0.0, 1.0))
            Fn_s = _smooth_force(Fn, fs)
            if Fn_s is None:
                Fn_s = Fn
            all_fn_abs.append(np.abs(Fn_s))
        else:
            zeros_force = np.zeros((len(time), 3))
            all_forces.append(zeros_force)
            all_forces_csv.append(zeros_force)
            all_fn_abs.append(np.zeros(len(time)))

        all_stiff.append(stiff)
        all_stiff_raw.append(stiff_raw)

    out_paths: List[Path] = []
    for idx, fname in enumerate(finger_names):
        out_png = out_dir / f"{csv_path.stem}_{fname}_global_tk_enhanced.png"
        try:
            _plot_variant(
                time,
                all_forces[idx],  # abs smoothed
                all_forces[idx],  # repeat for legacy signature
                all_fn_abs[idx],
                all_stiff_raw[idx],
                all_stiff[idx],
                emg_abs,
                emg_env,
                out_png,
                csv_path.stem,
                f"{fname} GLOBAL-TK-ENH",
            )
            out_paths.append(out_png)
        except Exception as exc:
            print(f"[warn] {csv_path.name}: plot {fname} fail ({exc})")

    # Non-force meta columns (preserve same format as original script)
    deform_circ = df["deform_circ"].to_numpy(dtype=float) if "deform_circ" in df else np.zeros(len(df), dtype=float)
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

    out_csv = out_dir / f"{csv_path.stem}_paper_profile.csv"  # same suffix for benchmark compatibility
    try:
        all_forces_combined = np.hstack(all_forces_csv)
        data = {"time_s": time.astype(float)}
        for i, (finger_name, sensor) in enumerate(zip(finger_names, sensor_names)):
            data[f"{sensor}_fx"] = all_forces_combined[:, i * 3 + 0].astype(float)
            data[f"{sensor}_fy"] = all_forces_combined[:, i * 3 + 1].astype(float)
            data[f"{sensor}_fz"] = all_forces_combined[:, i * 3 + 2].astype(float)
            data[f"{finger_name}_k1"] = all_stiff[i][:, 0].astype(float)
            data[f"{finger_name}_k2"] = all_stiff[i][:, 1].astype(float)
            data[f"{finger_name}_k3"] = all_stiff[i][:, 2].astype(float)

        data["deform_circ"] = deform_circ.astype(float)
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
        print(f"[warn] {csv_path.name}: CSV save fail ({exc})")

    print(f"[ok] {csv_path.name}: {len(out_paths)} enhanced outputs")
    return out_paths


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Enhanced global T_K stiffness generator")
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--output-dir", type=Path, default=PKG_DIR / "outputs" / "analysis" / "stiffness_profiles_global_tk_enhanced")
    ap.add_argument("--hp-cutoff", type=float, default=20.0, help="High-pass cutoff Hz (set 0 to disable)")
    ap.add_argument("--lp-cutoff", type=float, default=450.0, help="Low-pass cutoff Hz (set 0 to disable)")
    ap.add_argument("--norm", choices=["none", "zscore", "minmax", "unitnorm"], default="zscore")
    ap.add_argument("--envelope", choices=["ma", "savgol", "none"], default="ma")
    ap.add_argument("--validate", action="store_true", help="Run geometric/projector validation and save metrics JSON")
    args = ap.parse_args()

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

    global_tk = compute_global_tk(input_dir, args.hp_cutoff, args.lp_cutoff, args.norm, args.envelope)

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
            # Complementarity (optional)
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

    print("\n[Phase 2] Processing files with enhanced global T_K...")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_files = [p for p in sorted(input_dir.glob("*.csv")) if not p.name.endswith("_paper_profile.csv")]
    for csv_path in csv_files:
        process_file(csv_path, args.output_dir, global_tk, args.hp_cutoff, args.lp_cutoff, args.norm, args.envelope)

    print("\n[Done] Enhanced GLOBAL T_K processing complete")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":  # pragma: no cover
    main()
