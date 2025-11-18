#!/usr/bin/env python3
"""Check if EMG actually correlates with stiffness."""

import numpy as np
import pandas as pd
from pathlib import Path

LOG_DIR = Path("/home/songwoo/ros2_ws/icra2025/src/hri_falcon_robot_bridge/outputs/logs/success")
STIFF_DIR = Path("/home/songwoo/ros2_ws/icra2025/src/hri_falcon_robot_bridge/outputs/analysis/stiffness_profiles")

EMG_COLS = [f"emg_ch{i}" for i in range(1, 9)]
STIFF_COLS = [f"{finger}_k{i}" for finger in ['th', 'if', 'mf'] for i in [1, 2, 3]]

all_data = []

for csv_path in sorted(LOG_DIR.glob("*.csv")):
    if csv_path.name.endswith("_paper_profile.csv"):
        continue
    
    try:
        raw = pd.read_csv(csv_path)
        stiff_path = STIFF_DIR / f"{csv_path.stem}_paper_profile.csv"
        stiff = pd.read_csv(stiff_path)
        
        rows = min(len(raw), len(stiff))
        
        df = pd.DataFrame()
        for col in EMG_COLS:
            if col in raw.columns:
                df[col] = raw[col].iloc[:rows].values
        
        for col in STIFF_COLS:
            if col in stiff.columns:
                df[col] = stiff[col].iloc[:rows].values
        
        all_data.append(df)
    except Exception as e:
        continue

combined = pd.concat(all_data, ignore_index=True)
combined = combined.replace([np.inf, -np.inf], np.nan).dropna()

print(f"Total samples: {len(combined)}")
print(f"\nEMG-Stiffness Correlations:")
print("="*80)

corr_matrix = combined.corr()
emg_stiff_corr = corr_matrix.loc[EMG_COLS, STIFF_COLS]

# Find top correlations
abs_corr = emg_stiff_corr.abs().stack().sort_values(ascending=False)

print("\nTop 20 EMG-Stiffness Correlations:")
for i, ((emg, stiff), abs_val) in enumerate(abs_corr.head(20).items()):
    actual = emg_stiff_corr.loc[emg, stiff]
    print(f"{i+1:2d}. {emg:8s} <-> {stiff:8s}: {actual:+.4f} (|r|={abs_val:.4f})")

print(f"\nSummary:")
print(f"  Mean |correlation|: {abs_corr.mean():.4f}")
print(f"  Max  |correlation|: {abs_corr.max():.4f}")
print(f"  Median |correlation|: {abs_corr.median():.4f}")

print(f"\nPer-stiffness max correlation:")
for stiff_col in STIFF_COLS:
    max_corr = emg_stiff_corr[stiff_col].abs().max()
    max_emg = emg_stiff_corr[stiff_col].abs().idxmax()
    actual_val = emg_stiff_corr.loc[max_emg, stiff_col]
    print(f"  {stiff_col:8s}: {actual_val:+.4f} (with {max_emg})")

# Check EMG statistics
print(f"\n\nEMG Signal Statistics:")
print("="*80)
for emg in EMG_COLS:
    vals = combined[emg]
    print(f"{emg}: mean={vals.mean():.4f}, std={vals.std():.4f}, min={vals.min():.4f}, max={vals.max():.4f}")
