#!/usr/bin/env python3
"""Aggregate statistics over global T_K stiffness profile CSVs.

Scans an output directory produced by generate_stiffness_profiles_global_tk.py and
computes per-finger per-axis stats plus k_norm stats.

Output JSON keys:
  files_processed, rows_total
  per_finger: {finger: {axis: {min,max,mean,std}, k_norm: {...}}}
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

def k_norm(k1,k2,k3):
    return np.sqrt(k1**2 + k2**2 + k3**2)

def stats(arr):
    return {
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr)),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True, help='Directory with stiffness profile CSVs')
    ap.add_argument('--out', required=True, help='Output JSON path')
    args = ap.parse_args()

    d = Path(args.dir)
    csvs = sorted([p for p in d.glob('*.csv') if not p.name.endswith('_validation.json')])
    if not csvs:
        raise SystemExit('No CSV files found for summary')

    finger_map = {'th':['th_k1','th_k2','th_k3'], 'if':['if_k1','if_k2','if_k3'], 'mf':['mf_k1','mf_k2','mf_k3']}
    per_finger = {f:{} for f in finger_map.keys()}
    rows_total = 0

    for fp in csvs:
        try:
            df = pd.read_csv(fp)
        except Exception:
            continue
        rows_total += len(df)
        for f, cols in finger_map.items():
            if not all(c in df.columns for c in cols):
                continue
            k1 = df[cols[0]].to_numpy(float)
            k2 = df[cols[1]].to_numpy(float)
            k3 = df[cols[2]].to_numpy(float)
            if f not in per_finger or not per_finger[f]:
                per_finger[f] = {'k1':[], 'k2':[], 'k3':[], 'k_norm':[]}
            per_finger[f]['k1'].append(k1)
            per_finger[f]['k2'].append(k2)
            per_finger[f]['k3'].append(k3)
            per_finger[f]['k_norm'].append(k_norm(k1,k2,k3))

    out = {'files_processed': len(csvs), 'rows_total': rows_total, 'per_finger': {}}
    for f, data in per_finger.items():
        if not data:
            continue
        agg = {axis: np.concatenate(series) for axis, series in data.items()}
        out['per_finger'][f] = {axis: stats(vals) for axis, vals in agg.items()}

    with open(args.out,'w') as f:
        json.dump(out, f, indent=2)
    print('[ok] summary written:', args.out)

if __name__ == '__main__':
    main()
