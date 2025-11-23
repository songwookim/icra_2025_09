#!/usr/bin/env python3
"""Analyze correlation between observations and stiffness targets."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple
import matplotlib.pyplot as plt

# Same columns as run_stiffness_policy_benchmarks.py
OBS_COLUMNS = [
    "s1_fx", "s1_fy", "s1_fz",
    "s2_fx", "s2_fy", "s2_fz",
    "s3_fx", "s3_fy", "s3_fz",
    "deform_circ", "deform_ecc",
    "ee_if_px", "ee_if_py", "ee_if_pz",
    "ee_mf_px", "ee_mf_py", "ee_mf_pz",
    "ee_th_px", "ee_th_py", "ee_th_pz",
]

ACTION_COLUMNS = [
    "th_k1", "th_k2", "th_k3",
    "if_k1", "if_k2", "if_k3",
    "mf_k1", "mf_k2", "mf_k3",
]


def load_all_data(log_dir: Path, stiffness_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load all demos and concatenate into single arrays."""
    all_obs = []
    all_act = []
    
    for csv_path in sorted(log_dir.glob("*.csv")):
        if csv_path.name.endswith("_paper_profile.csv"):
            continue
        
        try:
            raw = pd.read_csv(csv_path)
            stiff_path = stiffness_dir / f"{csv_path.stem}_paper_profile.csv"
            stiff = pd.read_csv(stiff_path)
            
            rows = min(len(raw), len(stiff))
            raw = raw.iloc[:rows].reset_index(drop=True)
            stiff = stiff.iloc[:rows].reset_index(drop=True)
            
            # Backward compatibility
            for finger in ["if", "mf", "th"]:
                for axis in ["px", "py", "pz"]:
                    new_col = f"ee_{finger}_{axis}"
                    old_col = f"ee_{axis}"
                    if new_col not in raw.columns and old_col in raw.columns:
                        raw[new_col] = raw[old_col]
            
            # Check columns
            missing_obs = [col for col in OBS_COLUMNS if col not in raw.columns and col not in stiff.columns]
            missing_act = [col for col in ACTION_COLUMNS if col not in stiff.columns]
            
            if missing_obs or missing_act:
                print(f"[skip] {csv_path.name}: missing columns")
                continue
            
            # Build obs/act arrays
            obs_parts = []
            for col in OBS_COLUMNS:
                if col in stiff.columns:
                    obs_parts.append(stiff[col].to_numpy(dtype=float).reshape(-1, 1))
                else:
                    obs_parts.append(raw[col].to_numpy(dtype=float).reshape(-1, 1))
            obs = np.hstack(obs_parts)
            act = stiff[ACTION_COLUMNS].to_numpy(dtype=float)
            
            # Filter invalid
            mask = np.isfinite(obs).all(axis=1) & np.isfinite(act).all(axis=1)
            obs = obs[mask]
            act = act[mask]
            
            if obs.shape[0] > 0:
                all_obs.append(obs)
                all_act.append(act)
                print(f"[load] {csv_path.stem}: {obs.shape[0]} samples")
        
        except Exception as exc:
            print(f"[skip] {csv_path.name}: {exc}")
            continue
    
    if not all_obs:
        raise RuntimeError("No valid demos loaded")
    
    obs_all = np.vstack(all_obs)
    act_all = np.vstack(all_act)
    
    print(f"\n[total] {obs_all.shape[0]} samples from {len(all_obs)} demos")
    print(f"  obs shape: {obs_all.shape}")
    print(f"  act shape: {act_all.shape}")
    
    return obs_all, act_all


def analyze_correlation(obs: np.ndarray, act: np.ndarray) -> None:
    """Compute and visualize correlation matrix."""
    # Combine obs and act into single dataframe
    df = pd.DataFrame(
        np.hstack([obs, act]),
        columns=OBS_COLUMNS + ACTION_COLUMNS
    )
    
    # Compute correlation matrix
    corr = df.corr()
    
    # Extract obs-act cross-correlations
    obs_act_corr = corr.loc[OBS_COLUMNS, ACTION_COLUMNS]
    
    print("\n" + "="*80)
    print("OBSERVATION-STIFFNESS CORRELATION ANALYSIS")
    print("="*80)
    
    # Find strongest correlations
    print("\nStrongest correlations (top 20):")
    abs_corr = obs_act_corr.abs().stack().sort_values(ascending=False)
    for i, ((obs_name, act_name), corr_val) in enumerate(abs_corr.head(20).items()):
        actual_val = obs_act_corr.loc[obs_name, act_name]
        print(f"{i+1:2d}. {obs_name:15s} <-> {act_name:8s}: {actual_val:+.4f} (|r|={corr_val:.4f})")
    
    # Summary statistics
    print(f"\nCorrelation summary:")
    print(f"  Mean |correlation|: {abs_corr.mean():.4f}")
    print(f"  Max  |correlation|: {abs_corr.max():.4f}")
    print(f"  Median |correlation|: {abs_corr.median():.4f}")
    
    # Check per-action statistics
    print(f"\nPer-action max |correlation|:")
    for act_col in ACTION_COLUMNS:
        max_corr = obs_act_corr[act_col].abs().max()
        max_obs = obs_act_corr[act_col].abs().idxmax()
        actual_val = obs_act_corr.loc[max_obs, act_col]
        print(f"  {act_col:8s}: {actual_val:+.4f} (with {max_obs})")
    
    # Plot heatmap
    plt.figure(figsize=(14, 8))
    im = plt.imshow(
        obs_act_corr.values,
        aspect='auto',
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
    )
    plt.colorbar(im, label="Pearson correlation")
    plt.xticks(range(len(ACTION_COLUMNS)), ACTION_COLUMNS, rotation=45, ha='right')
    plt.yticks(range(len(OBS_COLUMNS)), OBS_COLUMNS)
    plt.title("Observation-Stiffness Correlation Matrix", fontsize=14, fontweight="bold")
    plt.xlabel("Stiffness Actions", fontsize=12)
    plt.ylabel("Observations", fontsize=12)
    plt.tight_layout()
    
    output_dir = Path("/home/songwoo/ros2_ws/icra2025/src/hri_falcon_robot_bridge/outputs/models/stiffness_policies")
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "correlation_matrix.png"
    plt.savefig(plot_path, dpi=150)
    print(f"\n[saved] correlation heatmap to {plot_path}")
    
    # Check variance
    print(f"\n" + "="*80)
    print("DATA VARIANCE ANALYSIS")
    print("="*80)
    
    print(f"\nObservation variance:")
    for i, col in enumerate(OBS_COLUMNS):
        print(f"  {col:15s}: mean={obs[:, i].mean():8.4f}, std={obs[:, i].std():8.4f}, range=[{obs[:, i].min():8.4f}, {obs[:, i].max():8.4f}]")
    
    print(f"\nAction variance:")
    for i, col in enumerate(ACTION_COLUMNS):
        print(f"  {col:8s}: mean={act[:, i].mean():8.2f}, std={act[:, i].std():8.2f}, range=[{act[:, i].min():8.2f}, {act[:, i].max():8.2f}]")


def main():
    log_dir = Path("/home/songwoo/ros2_ws/icra2025/src/hri_falcon_robot_bridge/outputs/logs/success")
    stiffness_dir = Path("/home/songwoo/ros2_ws/icra2025/src/hri_falcon_robot_bridge/outputs/analysis/stiffness_profiles")
    
    obs, act = load_all_data(log_dir, stiffness_dir)
    analyze_correlation(obs, act)


if __name__ == "__main__":
    main()
