#!/usr/bin/env python3
"""Verify if EMG-derived stiffness actually corresponds to demonstrated behavior.

The key question: Does k_emg explain the observed Force-Deformation relationship?

According to Hooke's Law: F = k × Δx
If k_emg is correct, then: F / Δx should ≈ k_emg

But we have a problem:
1. EMG → k_emg (via T_K projection matrix)
2. k_emg should control finger impedance
3. BUT: We don't have direct measurement of Δx (finger deformation)

We only have:
- deform_circ, deform_ecc (object shape, not finger deformation)
- EE position (doesn't show compliance)

This script checks:
1. Is EMG-derived stiffness consistent across demonstrations?
2. Does force magnitude correlate with EMG activity?
3. Is there a physical consistency check we can do?
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

LOG_DIR = Path("/home/songwoo/ros2_ws/icra2025/src/hri_falcon_robot_bridge/outputs/logs/success")
STIFF_DIR = Path("/home/songwoo/ros2_ws/icra2025/src/hri_falcon_robot_bridge/outputs/analysis/stiffness_profiles")
OUTPUT_DIR = Path("/home/songwoo/ros2_ws/icra2025/src/hri_falcon_robot_bridge/outputs/models/stiffness_policies")


def load_all_demos():
    """Load all demonstrations with force and stiffness data."""
    all_data = []
    
    for csv_path in sorted(LOG_DIR.glob("*.csv")):
        if csv_path.name.endswith("_paper_profile.csv"):
            continue
        
        try:
            raw = pd.read_csv(csv_path)
            stiff_path = STIFF_DIR / f"{csv_path.stem}_paper_profile.csv"
            
            if not stiff_path.exists():
                continue
            
            stiff = pd.read_csv(stiff_path)
            
            rows = min(len(raw), len(stiff))
            raw = raw.iloc[:rows].reset_index(drop=True)
            stiff = stiff.iloc[:rows].reset_index(drop=True)
            
            # Compute force magnitudes
            df = pd.DataFrame({
                'demo': csv_path.stem,
                'th_fx': raw['s1_fx'],
                'th_fy': raw['s1_fy'],
                'th_fz': raw['s1_fz'],
                'if_fx': raw['s2_fx'],
                'if_fy': raw['s2_fy'],
                'if_fz': raw['s2_fz'],
                'mf_fx': raw['s3_fx'],
                'mf_fy': raw['s3_fy'],
                'mf_fz': raw['s3_fz'],
                'th_k1': stiff['th_k1'],
                'th_k2': stiff['th_k2'],
                'th_k3': stiff['th_k3'],
                'if_k1': stiff['if_k1'],
                'if_k2': stiff['if_k2'],
                'if_k3': stiff['if_k3'],
                'mf_k1': stiff['mf_k1'],
                'mf_k2': stiff['mf_k2'],
                'mf_k3': stiff['mf_k3'],
            })
            
            df['th_force'] = np.sqrt(df['th_fx']**2 + df['th_fy']**2 + df['th_fz']**2)
            df['if_force'] = np.sqrt(df['if_fx']**2 + df['if_fy']**2 + df['if_fz']**2)
            df['mf_force'] = np.sqrt(df['mf_fx']**2 + df['mf_fy']**2 + df['mf_fz']**2)
            
            df['th_k_avg'] = (df['th_k1'] + df['th_k2'] + df['th_k3']) / 3
            df['if_k_avg'] = (df['if_k1'] + df['if_k2'] + df['if_k3']) / 3
            df['mf_k_avg'] = (df['mf_k1'] + df['mf_k2'] + df['mf_k3']) / 3
            
            all_data.append(df)
            
        except Exception as e:
            print(f"[skip] {csv_path.stem}: {e}")
            continue
    
    return pd.concat(all_data, ignore_index=True)


def analyze_stiffness_consistency(df):
    """Check if stiffness values are consistent with task requirements."""
    
    print("="*80)
    print("STIFFNESS GROUND TRUTH VERIFICATION")
    print("="*80)
    
    print("\n1. STIFFNESS DISTRIBUTION")
    print("-"*80)
    
    for finger, prefix in [('Thumb', 'th'), ('Index', 'if'), ('Middle', 'mf')]:
        k_avg = df[f'{prefix}_k_avg']
        print(f"\n{finger}:")
        print(f"  Mean: {k_avg.mean():.2f}")
        print(f"  Std:  {k_avg.std():.2f}")
        print(f"  Min:  {k_avg.min():.2f}")
        print(f"  Max:  {k_avg.max():.2f}")
        print(f"  Range: {k_avg.max() - k_avg.min():.2f}")
        
        # Check if stuck at K_MIN
        k_min_count = (k_avg <= 55).sum()
        k_min_pct = k_min_count / len(k_avg) * 100
        print(f"  At K_MIN (≤55): {k_min_count} samples ({k_min_pct:.1f}%)")
    
    print("\n\n2. FORCE-STIFFNESS RELATIONSHIP")
    print("-"*80)
    print("Question: Does higher stiffness → higher force? (as expected in impedance control)")
    
    for finger, prefix in [('Thumb', 'th'), ('Index', 'if'), ('Middle', 'mf')]:
        force = df[f'{prefix}_force']
        k_avg = df[f'{prefix}_k_avg']
        
        # Filter valid data
        mask = (np.isfinite(force)) & (np.isfinite(k_avg)) & (k_avg > 55)
        force_valid = force[mask]
        k_valid = k_avg[mask]
        
        if len(force_valid) > 0:
            corr = np.corrcoef(force_valid, k_valid)[0, 1]
            print(f"\n{finger}:")
            print(f"  Correlation (Force ↔ Stiffness): {corr:+.4f}")
            
            # Bin analysis: divide stiffness into quartiles
            k_q1, k_q2, k_q3 = np.percentile(k_valid, [25, 50, 75])
            
            f_low = force_valid[k_valid < k_q1].mean()
            f_mid = force_valid[(k_valid >= k_q1) & (k_valid < k_q3)].mean()
            f_high = force_valid[k_valid >= k_q3].mean()
            
            print(f"  Force when k<{k_q1:.0f}:  {f_low:.3f} N")
            print(f"  Force when k∈[{k_q1:.0f},{k_q3:.0f}]: {f_mid:.3f} N")
            print(f"  Force when k>{k_q3:.0f}: {f_high:.3f} N")
            
            if f_high > f_low:
                trend = "✓ Correct (higher k → higher F)"
            else:
                trend = "✗ WRONG (higher k → LOWER F)"
            print(f"  → {trend}")
    
    print("\n\n3. CROSS-DEMO VARIABILITY")
    print("-"*80)
    print("Question: Do different demos have different stiffness strategies?")
    
    demo_stats = df.groupby('demo').agg({
        'th_k_avg': ['mean', 'std'],
        'if_k_avg': ['mean', 'std'],
        'mf_k_avg': ['mean', 'std'],
        'th_force': 'mean',
        'if_force': 'mean',
        'mf_force': 'mean',
    }).round(2)
    
    print("\nPer-demo averages (first 10):")
    print(demo_stats.head(10))
    
    # Variance decomposition
    total_var_th = df['th_k_avg'].var()
    within_var_th = df.groupby('demo')['th_k_avg'].var().mean()
    between_var_th = total_var_th - within_var_th
    
    print(f"\nVariance decomposition (Thumb stiffness):")
    print(f"  Total variance: {total_var_th:.2f}")
    print(f"  Within-demo:    {within_var_th:.2f} ({within_var_th/total_var_th*100:.1f}%)")
    print(f"  Between-demo:   {between_var_th:.2f} ({between_var_th/total_var_th*100:.1f}%)")
    
    if between_var_th > within_var_th:
        print("  → HIGH inter-demo variance! Each demo has different stiffness strategy.")
    else:
        print("  → Low inter-demo variance. Stiffness is consistent across demos.")
    
    print("\n\n4. THE FUNDAMENTAL PROBLEM")
    print("-"*80)
    print("EMG-derived stiffness (k_emg) represents INTENT (muscle activation).")
    print("But observations (force, deform) represent OUTCOME (task execution).")
    print()
    print("The mapping is:")
    print("  EMG → k_emg (muscle stiffness)")
    print("  k_emg + Environment → Force (contact dynamics)")
    print("  Force + Object → Deformation (object compliance)")
    print()
    print("What we're trying to learn:")
    print("  (Force, Deform) → k_emg  [INVERSE problem, ill-posed]")
    print()
    print("Why it's hard:")
    print("  1. Same force can result from different (k_emg, deformation) pairs")
    print("  2. Deformation depends on object properties (unknown)")
    print("  3. k_emg is an INTERNAL state, not directly observable")
    print("  4. Each demo may use different stiffness strategy for same task")
    
    # Visualize
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    for i, (finger, prefix) in enumerate([('Thumb', 'th'), ('Index', 'if'), ('Middle', 'mf')]):
        force = df[f'{prefix}_force']
        k_avg = df[f'{prefix}_k_avg']
        
        # Filter valid
        mask = (np.isfinite(force)) & (np.isfinite(k_avg)) & (k_avg > 55)
        
        # Scatter plot
        axes[0, i].scatter(k_avg[mask], force[mask], alpha=0.1, s=1)
        axes[0, i].set_xlabel('Stiffness (k_avg)')
        axes[0, i].set_ylabel('Force magnitude (N)')
        axes[0, i].set_title(f'{finger} Finger')
        axes[0, i].grid(True, alpha=0.3)
        
        # Histogram
        axes[1, i].hist(k_avg[mask], bins=50, alpha=0.7, edgecolor='black')
        axes[1, i].axvline(k_avg[mask].mean(), color='red', linestyle='--', label=f'Mean={k_avg[mask].mean():.1f}')
        axes[1, i].axvline(55, color='orange', linestyle='--', label='K_MIN=55')
        axes[1, i].set_xlabel('Stiffness (k_avg)')
        axes[1, i].set_ylabel('Count')
        axes[1, i].legend()
        axes[1, i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = OUTPUT_DIR / "stiffness_ground_truth_verification.png"
    plt.savefig(plot_path, dpi=150)
    print(f"\n[saved] verification plot to {plot_path}")


def main():
    print("[info] Loading all demonstrations...")
    df = load_all_demos()
    print(f"[total] {len(df)} samples from {df['demo'].nunique()} demos")
    
    analyze_stiffness_consistency(df)


if __name__ == "__main__":
    main()
