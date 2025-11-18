#!/usr/bin/env python3
"""Augment demonstration data (raw logs + stiffness profiles) for training.

This script applies physics-aware data augmentation to both:
- Raw demonstration logs (outputs/logs/success/*.csv)
- Stiffness profiles (outputs/analysis/stiffness_profiles_global_tk/*_paper_profile.csv)

Augmentation includes:
- Gaussian noise on force, stiffness, deformation, end-effector positions
- Magnitude scaling for force and stiffness
- Temporal variations (shifting, jittering)
- Physical constraints (stiffness >= 1.0 N/m)

Usage:
    python3 scripts/augment_demonstration_data.py --num-augment 5
    
Output:
    Augmented files saved with suffix _aug{i} in the same directories.
"""

import argparse
import sys
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd

# Add script directory to path for imports
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_LEARNING_DIR = SCRIPT_DIR / "3_model_learning"
if str(MODEL_LEARNING_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_LEARNING_DIR))

from augmentation_utils import DataAugmentor, Trajectory

PKG_DIR = SCRIPT_DIR.parent  # .../src/hri_falcon_robot_bridge
DEFAULT_LOG_DIR = PKG_DIR / "outputs" / "logs" / "success"
DEFAULT_STIFF_DIR = PKG_DIR / "outputs" / "analysis" / "stiffness_profiles_global_tk"


def augment_raw_log(
    log_path: Path,
    output_dir: Path,
    augmentor: DataAugmentor,
    num_augment: int,
    noise_force: float,
    noise_deform: float,
    noise_ee: float,
) -> List[Path]:
    """Augment a single raw demonstration log."""
    df = pd.read_csv(log_path)
    
    # Extract relevant columns for augmentation
    force_cols = [f"s{i}_{axis}" for i in [1, 2, 3] for axis in ["fx", "fy", "fz"]]
    deform_cols = ["deform_circ", "deform_ecc"]
    ee_cols = [f"ee_{finger}_{axis}" for finger in ["th", "if", "mf"] for axis in ["px", "py", "pz"]]
    
    # Build observation array (only columns that exist)
    obs_parts = []
    obs_col_names = []
    for col in force_cols + deform_cols + ee_cols:
        if col in df.columns:
            obs_parts.append(df[col].values.reshape(-1, 1))
            obs_col_names.append(col)
    
    if not obs_parts:
        print(f"[SKIP] {log_path.name}: No augmentable columns found")
        return []
    
    observations = np.hstack(obs_parts)
    
    # Dummy actions (not used, but required by augmentor)
    actions = np.zeros((len(df), 9))
    
    # Create trajectory
    traj = Trajectory(
        name=log_path.stem,
        observations=observations,
        actions=actions
    )
    
    # Augment (disable temporal operations to preserve exact length)
    aug_trajs = augmentor.augment_trajectory(
        traj,
        obs_columns=obs_col_names,
        num_augmentations=num_augment,
        noise_std_force=noise_force,
        noise_std_deform=noise_deform,
        noise_std_ee=noise_ee,
        noise_std_stiffness=0.0,  # No stiffness in raw logs
        enable_temporal_shift=False,  # CRITICAL: preserve length
        enable_temporal_jitter=False,  # CRITICAL: preserve length
    )
    
    augmented_files = []
    # Skip first (original)
    for i, aug_traj in enumerate(aug_trajs[1:], start=1):
        # Ensure lengths match
        if aug_traj.observations.shape[0] != len(df):
            print(f"  ⚠️  Length mismatch for aug {i}, skipping")
            continue
            
        # Create augmented dataframe
        aug_df = df.copy()
        for j, col in enumerate(obs_col_names):
            aug_df[col] = aug_traj.observations[:, j]
        
        # Save augmented file
        stem = log_path.stem
        aug_path = output_dir / f"{stem}_aug{i}.csv"
        aug_df.to_csv(aug_path, index=False)
        augmented_files.append(aug_path)
    
    return augmented_files


def augment_stiffness_profile(
    stiff_path: Path,
    output_dir: Path,
    augmentor: DataAugmentor,
    num_augment: int,
    noise_force: float,
    noise_deform: float,
    noise_ee: float,
    noise_stiffness: float,
) -> List[Path]:
    """Augment a single stiffness profile."""
    df = pd.read_csv(stiff_path)
    
    # Extract columns
    force_cols = [f"s{i}_{axis}" for i in [1, 2, 3] for axis in ["fx", "fy", "fz"]]
    stiff_cols = [f"{finger}_k{dof}" for finger in ["th", "if", "mf"] for dof in [1, 2, 3]]
    deform_cols = ["deform_circ", "deform_ecc"]
    ee_cols = [f"ee_{finger}_{axis}" for finger in ["th", "if", "mf"] for axis in ["px", "py", "pz"]]
    
    # Build observation array
    obs_parts = []
    obs_col_names = []
    for col in force_cols + deform_cols + ee_cols:
        if col in df.columns:
            obs_parts.append(df[col].values.reshape(-1, 1))
            obs_col_names.append(col)
    
    # Build action array (stiffness)
    act_parts = []
    act_col_names = []
    for col in stiff_cols:
        if col in df.columns:
            act_parts.append(df[col].values.reshape(-1, 1))
            act_col_names.append(col)
    
    if not obs_parts or not act_parts:
        print(f"[SKIP] {stiff_path.name}: Missing columns")
        return []
    
    observations = np.hstack(obs_parts)
    actions = np.hstack(act_parts)
    
    # Create trajectory
    traj = Trajectory(
        name=stiff_path.stem,
        observations=observations,
        actions=actions
    )
    
    # Augment (disable temporal operations to preserve exact length)
    aug_trajs = augmentor.augment_trajectory(
        traj,
        obs_columns=obs_col_names,
        num_augmentations=num_augment,
        noise_std_force=noise_force,
        noise_std_deform=noise_deform,
        noise_std_ee=noise_ee,
        noise_std_stiffness=noise_stiffness,
        enable_temporal_shift=False,  # CRITICAL: preserve length
        enable_temporal_jitter=False,  # CRITICAL: preserve length
    )
    
    augmented_files = []
    # Skip first (original)
    for i, aug_traj in enumerate(aug_trajs[1:], start=1):
        # Ensure lengths match
        if aug_traj.observations.shape[0] != len(df) or aug_traj.actions.shape[0] != len(df):
            print(f"  ⚠️  Length mismatch for aug {i}, skipping")
            continue
            
        # Create augmented dataframe
        aug_df = df.copy()
        for j, col in enumerate(obs_col_names):
            aug_df[col] = aug_traj.observations[:, j]
        for j, col in enumerate(act_col_names):
            aug_df[col] = aug_traj.actions[:, j]
        
        # Save augmented file
        stem = stiff_path.stem.replace("_paper_profile", "")
        aug_path = output_dir / f"{stem}_aug{i}_paper_profile.csv"
        aug_df.to_csv(aug_path, index=False)
        augmented_files.append(aug_path)
    
    return augmented_files


def main():
    parser = argparse.ArgumentParser(
        description="Augment demonstration data (raw logs + stiffness profiles)"
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=DEFAULT_LOG_DIR,
        help="Directory containing raw demonstration logs"
    )
    parser.add_argument(
        "--stiff-dir",
        type=Path,
        default=DEFAULT_STIFF_DIR,
        help="Directory containing stiffness profiles"
    )
    parser.add_argument(
        "--num-augment",
        type=int,
        default=5,
        help="Number of augmented copies per demonstration"
    )
    parser.add_argument(
        "--noise-force",
        type=float,
        default=0.02,
        help="Force noise level (relative to signal std, default: 0.02 = 2%%)"
    )
    parser.add_argument(
        "--noise-stiffness",
        type=float,
        default=0.05,
        help="Stiffness noise level (relative to signal std, default: 0.05 = 5%%)"
    )
    parser.add_argument(
        "--noise-deform",
        type=float,
        default=0.01,
        help="Deformation noise level (relative to signal std, default: 0.01 = 1%%)"
    )
    parser.add_argument(
        "--noise-ee",
        type=float,
        default=0.001,
        help="End-effector position noise level (absolute, meters, default: 0.001 = 1mm)"
    )
    parser.add_argument(
        "--skip-logs",
        action="store_true",
        help="Skip augmenting raw logs"
    )
    parser.add_argument(
        "--skip-stiffness",
        action="store_true",
        help="Skip augmenting stiffness profiles"
    )
    
    args = parser.parse_args()
    
    # Create augmentor
    augmentor = DataAugmentor(seed=42)
    
    print("="*80)
    print("DATA AUGMENTATION")
    print("="*80)
    print(f"Log directory: {args.log_dir}")
    print(f"Stiffness directory: {args.stiff_dir}")
    print(f"Number of augmentations per demo: {args.num_augment}")
    print(f"Noise levels - Force: {args.noise_force}, Stiffness: {args.noise_stiffness}")
    print(f"             - Deform: {args.noise_deform}, EE: {args.noise_ee} m")
    print("="*80)
    
    # Augment raw logs
    if not args.skip_logs:
        print("\n[1/2] Augmenting raw demonstration logs...")
        log_files = sorted(args.log_dir.glob("*.csv"))
        # Filter out already augmented files
        log_files = [f for f in log_files if "_aug" not in f.stem]
        
        total_created = 0
        for log_path in log_files:
            augmented = augment_raw_log(
                log_path,
                args.log_dir,
                augmentor,
                args.num_augment,
                args.noise_force,
                args.noise_deform,
                args.noise_ee,
            )
            total_created += len(augmented)
            if augmented:
                print(f"  ✅ {log_path.name} → {len(augmented)} augmented files")
        
        print(f"\n✅ Created {total_created} augmented log files")
    else:
        print("\n[1/2] Skipping raw logs (--skip-logs)")
    
    # Augment stiffness profiles
    if not args.skip_stiffness:
        print("\n[2/2] Augmenting stiffness profiles...")
        stiff_files = sorted(args.stiff_dir.glob("*_paper_profile.csv"))
        # Filter out already augmented files
        stiff_files = [f for f in stiff_files if "_aug" not in f.stem]
        
        total_created = 0
        for stiff_path in stiff_files:
            augmented = augment_stiffness_profile(
                stiff_path,
                args.stiff_dir,
                augmentor,
                args.num_augment,
                args.noise_force,
                args.noise_deform,
                args.noise_ee,
                args.noise_stiffness,
            )
            total_created += len(augmented)
            if augmented:
                print(f"  ✅ {stiff_path.name} → {len(augmented)} augmented files")
        
        print(f"\n✅ Created {total_created} augmented stiffness files")
    else:
        print("\n[2/2] Skipping stiffness profiles (--skip-stiffness)")
    
    print("\n" + "="*80)
    print("AUGMENTATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
