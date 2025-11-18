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
PKG_DIR = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# Import utilities from original script
from generate_stiffness_profiles import (
    _load_csv,
    _time_vector,
    _estimate_fs,
    _guess_emg_columns,
    _align_emg_to_time,
    _center_signal,
    _compute_normal_force,
    _smooth_force,
    _ultra_smooth_strong_emg,
    compute_tf,
    compute_projection_from_tf,
    compute_k_basis_from_force_projector,
    _plot_variant,
)

EPS = 1e-8
DEFAULT_INPUT = PKG_DIR / "outputs" / "logs" / "success"
K_INIT = 200.0
K_MIN = 50.0
VALIDATE_RESULTS = True


def compute_global_tk(input_dir: Path) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Compute global T_K for each finger from all demonstrations combined.
    
    Returns:
        Dict with keys 'th', 'if', 'mf', each containing (T_K, H_K) tuple
        T_K: (3, d) transformation matrix from EMG to stiffness
        H_K: (d, d) stiffness subspace projector
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
            
            # Extract force for each finger
            for finger, sensor in sensor_map.items():
                try:
                    Fn = _compute_normal_force(df, prefix=sensor, n=(0.0, 0.0, 1.0))
                    Fn_smoothed = _smooth_force(Fn, fs)
                    if Fn_smoothed is None:
                        Fn_smoothed = Fn
                    
                    finger_data[finger]["emg"].append(emg_smoothed)
                    finger_data[finger]["force"].append(Fn_smoothed)
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
        F_global = np.concatenate(finger_data[finger]["force"])[None, :]  # (1, N_total)
        
        print(f"[{finger}] Global data: EMG {P_global.shape}, Force {F_global.shape}")
        
        T_F = compute_tf(P_global, F_global)
        H_F = compute_projection_from_tf(T_F)
        T_K, H_K = compute_k_basis_from_force_projector(H_F, target_rank=3)
        
        global_tk[finger] = (T_K, H_K)
        print(f"[{finger}] T_K shape: {T_K.shape}, norm: {np.linalg.norm(T_K):.4f}")
    
    return global_tk


def process_file_with_global_tk(
    csv_path: Path,
    output_dir: Path,
    global_tk: Dict[str, Tuple[np.ndarray, np.ndarray]],
) -> Optional[List[Path]]:
    """Process single file using global T_K per finger.
    
    Args:
        global_tk: Dict with keys 'th', 'if', 'mf', each containing (T_K, H_K)
    """
    try:
        df = _load_csv(csv_path)
    except Exception as exc:
        print(f"[skip] {csv_path.name}: load failed ({exc})")
        return None

    time = _time_vector(df)
    fs = _estimate_fs(time) or 200.0

    # Extract deformation and end-effector data
    deform_circ = df.get("deform_circ", np.zeros(len(df))).values
    deform_ecc = df.get("deform_ecc", np.zeros(len(df))).values
    
    ee_if_px = df.get("ee_if_px", df.get("ee_px", np.zeros(len(df)))).values
    ee_if_py = df.get("ee_if_py", df.get("ee_py", np.zeros(len(df)))).values
    ee_if_pz = df.get("ee_if_pz", df.get("ee_pz", np.zeros(len(df)))).values
    
    ee_mf_px = df.get("ee_mf_px", np.zeros(len(df))).values
    ee_mf_py = df.get("ee_mf_py", np.zeros(len(df))).values
    ee_mf_pz = df.get("ee_mf_pz", np.zeros(len(df))).values
    
    ee_th_px = df.get("ee_th_px", np.zeros(len(df))).values
    ee_th_py = df.get("ee_th_py", np.zeros(len(df))).values
    ee_th_pz = df.get("ee_th_pz", np.zeros(len(df))).values

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
    all_fn_abs = []
    
    for finger, sensor in zip(finger_names, sensor_names):
        # Get this finger's global T_K
        T_K_finger, _ = global_tk[finger]
        
        # Compute stiffness for this finger using its T_K
        stiffness_finger = np.maximum(K_MIN, (T_K_finger @ P_variant).T + K_INIT)
        stiffness_finger_raw = np.maximum(K_MIN, (T_K_finger @ P_raw).T + K_INIT)
        
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
            
            forces_abs = np.abs(forces_centered)
            forces_abs_smoothed = _smooth_force(forces_abs, fs)
            if forces_abs_smoothed is None:
                forces_abs_smoothed = forces_abs
            all_forces.append(forces_abs_smoothed)

            Fn = _compute_normal_force(df, prefix=sensor, n=(0.0, 0.0, 1.0))
            Fn_smoothed = _smooth_force(Fn, fs)
            if Fn_smoothed is None:
                Fn_smoothed = Fn
            all_fn_abs.append(np.abs(Fn_smoothed))
            
            # Use finger-specific stiffness
            all_stiffness.append(stiffness_finger)
        else:
            all_forces.append(np.zeros((len(time), 3)))
            all_forces_csv.append(np.zeros((len(time), 3)))
            all_stiffness.append(stiffness_finger)
            all_fn_abs.append(np.zeros(len(time)))
    
    # Generate plots (use first finger's raw stiffness for comparison)
    stiffness_raw_ref = np.maximum(K_MIN, (global_tk["th"][0] @ P_raw).T + K_INIT)
    
    for idx, finger_name in enumerate(finger_names):
        out_png = output_dir / f"{csv_path.stem}_{finger_name}_global_tk.png"
        try:
            _plot_variant(
                time,
                all_forces[idx],
                all_forces[idx],
                all_fn_abs[idx],
                stiffness_raw_ref,  # raw (for comparison)
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
    out_csv = output_dir / f"{csv_path.stem}_paper_profile.csv"
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
        print(f"[warn] {csv_path.name}: CSV save failed ({exc})")

    print(f"[ok] {csv_path.name}: {len(out_paths)} files (GLOBAL T_K)")
    return out_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate stiffness with global T_K")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PKG_DIR / "outputs" / "analysis" / "stiffness_profiles_global_tk",
    )
    args = parser.parse_args()
    
    if not args.input.is_dir():
        raise ValueError(f"Input must be directory: {args.input}")
    
    # Phase 1: Compute global T_K per finger
    global_tk = compute_global_tk(args.input)
    
    # Phase 2: Process all files with global T_K
    print(f"\n[Phase 2] Processing files with global T_K...")
    csv_files = [p for p in sorted(args.input.glob("*.csv")) if not p.name.endswith("_paper_profile.csv")]
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    for csv_path in csv_files:
        process_file_with_global_tk(csv_path, args.output_dir, global_tk)
    
    print(f"\n[Done] All files processed with GLOBAL T_K")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
