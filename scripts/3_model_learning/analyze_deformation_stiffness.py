#!/usr/bin/env python3
"""Analyze if deformation features can better predict stiffness.

Current features: circularity, eccentricity (shape descriptors)
Proposed features: Force/Deformation ratio, deformation magnitude

Theory: k = F / Δx (Hooke's Law)
If we can measure Δx from vision/force sensors, we can estimate k directly.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

LOG_DIR = Path("/home/songwoo/ros2_ws/icra2025/src/hri_falcon_robot_bridge/outputs/logs/success")
STIFF_DIR = Path("/home/songwoo/ros2_ws/icra2025/src/hri_falcon_robot_bridge/outputs/analysis/stiffness_profiles")
OUTPUT_DIR = Path("/home/songwoo/ros2_ws/icra2025/src/hri_falcon_robot_bridge/outputs/models/stiffness_policies")


def load_demo(csv_path: Path) -> pd.DataFrame:
    """Load raw CSV and stiffness profile, return merged dataframe."""
    raw = pd.read_csv(csv_path)
    stiff_path = STIFF_DIR / f"{csv_path.stem}_paper_profile.csv"
    
    if not stiff_path.exists():
        return None
    
    stiff = pd.read_csv(stiff_path)
    
    rows = min(len(raw), len(stiff))
    raw = raw.iloc[:rows].reset_index(drop=True)
    stiff = stiff.iloc[:rows].reset_index(drop=True)
    
    # Merge
    merged = pd.concat([raw, stiff], axis=1)
    
    # Remove duplicate columns (keep first)
    merged = merged.loc[:, ~merged.columns.duplicated()]
    
    return merged


def compute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute new deformation-based features."""
    
    # 1. Total force magnitude per finger
    df['th_force_mag'] = np.sqrt(df['s1_fx']**2 + df['s1_fy']**2 + df['s1_fz']**2)
    df['if_force_mag'] = np.sqrt(df['s2_fx']**2 + df['s2_fy']**2 + df['s2_fz']**2)
    df['mf_force_mag'] = np.sqrt(df['s3_fx']**2 + df['s3_fy']**2 + df['s3_fz']**2)
    
    # 2. Total stiffness per finger (average of k1, k2, k3)
    df['th_stiff_avg'] = (df['th_k1'] + df['th_k2'] + df['th_k3']) / 3
    df['if_stiff_avg'] = (df['if_k1'] + df['if_k2'] + df['if_k3']) / 3
    df['mf_stiff_avg'] = (df['mf_k1'] + df['mf_k2'] + df['mf_k3']) / 3
    
    # 3. Implied deformation from Hooke's law: Δx = F / k
    # Only compute where stiffness > threshold to avoid division by zero
    k_threshold = 60.0  # Higher than K_MIN=50 to avoid edge cases
    
    df['th_implied_deform'] = np.where(
        df['th_stiff_avg'] > k_threshold,
        df['th_force_mag'] / df['th_stiff_avg'],
        np.nan
    )
    df['if_implied_deform'] = np.where(
        df['if_stiff_avg'] > k_threshold,
        df['if_force_mag'] / df['if_stiff_avg'],
        np.nan
    )
    df['mf_implied_deform'] = np.where(
        df['mf_stiff_avg'] > k_threshold,
        df['mf_force_mag'] / df['mf_stiff_avg'],
        np.nan
    )
    
    # 4. Shape change rate (Δcircularity, Δeccentricity over time)
    if 'deform_circ' in df.columns and 'deform_ecc' in df.columns:
        df['deform_circ_vel'] = df['deform_circ'].diff().fillna(0)
        df['deform_ecc_vel'] = df['deform_ecc'].diff().fillna(0)
    else:
        df['deform_circ_vel'] = 0
        df['deform_ecc_vel'] = 0
    
    # 5. Force/stiffness ratio (proxy for deformation)
    k_threshold = 60.0
    df['th_force_stiff_ratio'] = np.where(
        df['th_stiff_avg'] > k_threshold,
        df['th_force_mag'] / df['th_stiff_avg'],
        np.nan
    )
    df['if_force_stiff_ratio'] = np.where(
        df['if_stiff_avg'] > k_threshold,
        df['if_force_mag'] / df['if_stiff_avg'],
        np.nan
    )
    df['mf_force_stiff_ratio'] = np.where(
        df['mf_stiff_avg'] > k_threshold,
        df['mf_force_mag'] / df['mf_stiff_avg'],
        np.nan
    )
    
    return df


def analyze_correlations(all_data: pd.DataFrame) -> None:
    """Compute correlations between deformation features and stiffness."""
    
    # Current features
    current_features = ['deform_circ', 'deform_ecc']
    
    # Proposed features
    new_features = [
        'th_force_mag', 'if_force_mag', 'mf_force_mag',
        'th_implied_deform', 'if_implied_deform', 'mf_implied_deform',
        'deform_circ_vel', 'deform_ecc_vel',
        'th_force_stiff_ratio', 'if_force_stiff_ratio', 'mf_force_stiff_ratio',
    ]
    
    # Target stiffness
    stiffness_cols = [
        'th_k1', 'th_k2', 'th_k3',
        'if_k1', 'if_k2', 'if_k3',
        'mf_k1', 'mf_k2', 'mf_k3',
    ]
    
    # Also add averaged stiffness
    avg_stiff_cols = ['th_stiff_avg', 'if_stiff_avg', 'mf_stiff_avg']
    
    print("="*80)
    print("DEFORMATION-STIFFNESS CORRELATION ANALYSIS")
    print("="*80)
    
    # Compute correlations
    feature_cols = [f for f in current_features + new_features if f in all_data.columns]
    target_cols = [c for c in stiffness_cols + avg_stiff_cols if c in all_data.columns]
    
    corr_matrix = all_data[feature_cols + target_cols].corr()
    feature_stiff_corr = corr_matrix.loc[feature_cols, target_cols]
    
    print("\n1. CURRENT FEATURES (circularity, eccentricity)")
    print("-" * 80)
    for feat in current_features:
        if feat not in all_data.columns:
            continue
        print(f"\n{feat}:")
        abs_corr = feature_stiff_corr.loc[feat].abs().sort_values(ascending=False)
        for i, (col, val) in enumerate(abs_corr.head(5).items()):
            actual = feature_stiff_corr.loc[feat, col]
            print(f"  {i+1}. {col:15s}: {actual:+.4f} (|r|={val:.4f})")
    
    print("\n\n2. PROPOSED FEATURES (Force magnitude, Deformation proxy)")
    print("-" * 80)
    
    # Group by finger
    for finger, prefix in [('Thumb', 'th'), ('Index', 'if'), ('Middle', 'mf')]:
        print(f"\n{finger} Finger:")
        finger_features = [f for f in new_features if f.startswith(prefix)]
        finger_stiff = [f"{prefix}_k1", f"{prefix}_k2", f"{prefix}_k3", f"{prefix}_stiff_avg"]
        
        for feat in finger_features:
            if feat not in all_data.columns:
                continue
            print(f"\n  {feat}:")
            for stiff_col in finger_stiff:
                if stiff_col not in all_data.columns:
                    continue
                corr_val = corr_matrix.loc[feat, stiff_col]
                print(f"    → {stiff_col:15s}: {corr_val:+.4f}")
    
    print("\n\n3. SUMMARY: Best Correlations")
    print("-" * 80)
    
    # Find top correlations across all features
    abs_corr_all = feature_stiff_corr.abs().stack().sort_values(ascending=False)
    
    print("\nTop 20 Feature-Stiffness Correlations:")
    for i, ((feat, stiff), abs_val) in enumerate(abs_corr_all.head(20).items()):
        actual = feature_stiff_corr.loc[feat, stiff]
        marker = "✓ NEW" if feat in new_features else "  CUR"
        print(f"{i+1:2d}. [{marker}] {feat:25s} <-> {stiff:15s}: {actual:+.4f} (|r|={abs_val:.4f})")
    
    # Statistics
    current_corrs = []
    new_corrs = []
    
    for feat in feature_cols:
        for stiff in target_cols:
            val = abs(feature_stiff_corr.loc[feat, stiff])
            if feat in current_features:
                current_corrs.append(val)
            else:
                new_corrs.append(val)
    
    print(f"\n\nCorrelation Statistics:")
    print(f"  Current features (circ, ecc):")
    print(f"    Mean |r|: {np.mean(current_corrs):.4f}")
    print(f"    Max  |r|: {np.max(current_corrs):.4f}")
    
    print(f"\n  Proposed features (force, deform proxy):")
    print(f"    Mean |r|: {np.mean(new_corrs):.4f}")
    print(f"    Max  |r|: {np.max(new_corrs):.4f}")
    
    improvement = (np.mean(new_corrs) - np.mean(current_corrs)) / np.mean(current_corrs) * 100
    print(f"\n  → Improvement: {improvement:+.1f}%")
    
    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Current features heatmap
    current_corr = feature_stiff_corr.loc[[f for f in current_features if f in all_data.columns]]
    im1 = axes[0].imshow(current_corr.values, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
    axes[0].set_xticks(range(len(current_corr.columns)))
    axes[0].set_xticklabels(current_corr.columns, rotation=45, ha='right', fontsize=8)
    axes[0].set_yticks(range(len(current_corr.index)))
    axes[0].set_yticklabels(current_corr.index, fontsize=10)
    axes[0].set_title("Current Features\n(circularity, eccentricity)", fontweight='bold')
    plt.colorbar(im1, ax=axes[0])
    
    # New features heatmap (force magnitude only for clarity)
    force_features = ['th_force_mag', 'if_force_mag', 'mf_force_mag']
    force_features = [f for f in force_features if f in all_data.columns]
    if force_features:
        force_corr = feature_stiff_corr.loc[force_features]
        im2 = axes[1].imshow(force_corr.values, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
        axes[1].set_xticks(range(len(force_corr.columns)))
        axes[1].set_xticklabels(force_corr.columns, rotation=45, ha='right', fontsize=8)
        axes[1].set_yticks(range(len(force_corr.index)))
        axes[1].set_yticklabels(force_corr.index, fontsize=10)
        axes[1].set_title("Proposed Features\n(force magnitude)", fontweight='bold')
        plt.colorbar(im2, ax=axes[1])
    
    plt.tight_layout()
    plot_path = OUTPUT_DIR / "deformation_correlation_comparison.png"
    plt.savefig(plot_path, dpi=150)
    print(f"\n[saved] comparison plot to {plot_path}")


def main():
    all_dfs = []
    
    print("[info] Loading demos and computing derived features...")
    for csv_path in sorted(LOG_DIR.glob("*.csv")):
        if csv_path.name.endswith("_paper_profile.csv"):
            continue
        
        df = load_demo(csv_path)
        if df is None:
            continue
        
        df = compute_derived_features(df)
        all_dfs.append(df)
        print(f"  [load] {csv_path.stem}: {len(df)} samples")
    
    if not all_dfs:
        print("[error] No valid demos found")
        return
    
    all_data = pd.concat(all_dfs, ignore_index=True)
    print(f"\n[total] {len(all_data)} samples from {len(all_dfs)} demos")
    
    # Filter out NaN in derived features only (keep original data)
    derived_cols = [
        'th_force_mag', 'if_force_mag', 'mf_force_mag',
        'th_implied_deform', 'if_implied_deform', 'mf_implied_deform',
        'deform_circ_vel', 'deform_ecc_vel',
        'th_force_stiff_ratio', 'if_force_stiff_ratio', 'mf_force_stiff_ratio',
    ]
    
    # Replace inf with nan
    all_data = all_data.replace([np.inf, -np.inf], np.nan)
    
    # Keep rows where at least some derived features are valid
    print(f"[info] Data contains NaN in derived features (expected due to k threshold)")
    print(f"[keep] All {len(all_data)} samples (will handle NaN in correlation)")
    
    analyze_correlations(all_data)


if __name__ == "__main__":
    main()
