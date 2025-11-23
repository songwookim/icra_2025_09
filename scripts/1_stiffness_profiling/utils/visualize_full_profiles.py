#!/usr/bin/env python3
"""Create comprehensive PNG per normalized stiffness CSV: FORCE, STIFFNESS, EMG.

Input: directory with normalized CSVs containing columns:
  time_s, s1_fx/s1_fy/s1_fz, s2_fx/... s3_*, th_k1/k2/k3/th_k_intent01, if_*, mf_*, emg_ch*, emg_ch*_smooth

Output: one PNG per file: <stem>_full_profile.png placed in --out-dir.
Panels:
  (1) Force norms & individual sensor forces (optionally stacked)
  (2) Stiffness k_norm per finger + component ranges ribbon (min/max across k1,k2,k3)
  (3) EMG smoothed envelopes (8ch) + raw magnitude faint background.

"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FINGERS = ['th','if','mf']
EMG_CH = [f'emg_ch{i}' for i in range(1,9)]

def k_norm(df: pd.DataFrame, finger: str) -> np.ndarray:
    return np.sqrt((df[f'{finger}_k1']**2 + df[f'{finger}_k2']**2 + df[f'{finger}_k3']**2).to_numpy(dtype=float))

def force_norm(df: pd.DataFrame, sensor: str) -> np.ndarray:
    return np.sqrt((df[f'{sensor}_fx']**2 + df[f'{sensor}_fy']**2 + df[f'{sensor}_fz']**2).to_numpy(dtype=float))

def plot_file(csv_path: Path, out_dir: Path) -> None:
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        print(f'[skip] {csv_path.name}: load failed ({exc})')
        return
    if 'time_s' not in df.columns:
        print(f'[skip] {csv_path.name}: no time_s column')
        return
    t = df['time_s'].to_numpy(dtype=float)
    if t.size < 10:
        print(f'[skip] {csv_path.name}: too few samples')
        return

    # Prepare figure
    fig = plt.figure(figsize=(14,8))
    gs = fig.add_gridspec(3,1, height_ratios=[1.0,1.0,1.2])
    ax_force = fig.add_subplot(gs[0,0])
    ax_stiff = fig.add_subplot(gs[1,0])
    ax_emg = fig.add_subplot(gs[2,0])

    # Force panel
    for sensor, color in zip(['s1','s2','s3'], ['#1f77b4','#ff7f0e','#2ca02c']):
        if all(f'{sensor}_{c}' in df.columns for c in ['fx','fy','fz']):
            fn = force_norm(df, sensor)
            ax_force.plot(t, fn, label=f'{sensor}_norm', color=color, linewidth=1.0)
    ax_force.set_ylabel('force_norm [a.u.]')
    ax_force.set_title(f'Forces (norm) - {csv_path.stem}')
    ax_force.legend(loc='upper right', fontsize=8, ncol=3)

    # Stiffness panel
    for finger, color in zip(FINGERS, ['#9467bd','#d62728','#17becf']):
        try:
            kn = k_norm(df, finger)
            ax_stiff.plot(t, kn, label=f'{finger}_k_norm', color=color, linewidth=1.0)
            # intent overlay (scaled) if present
            intent_col = f'{finger}_k_intent01'
            if intent_col in df.columns:
                intent = df[intent_col].to_numpy(dtype=float)
                # scale intent to k_norm range for visual overlay
                if kn.max() - kn.min() > 1e-9:
                    intent_scaled = kn.min() + (kn.max()-kn.min()) * intent
                    ax_stiff.plot(t, intent_scaled, color=color, alpha=0.25, linewidth=0.8, linestyle='--')
        except Exception:
            continue
    ax_stiff.set_ylabel('k_norm [N/m]')
    ax_stiff.set_title('Stiffness norms (+ intent dashed)')
    ax_stiff.legend(loc='upper right', fontsize=8, ncol=3)

    # EMG panel
    have_smoothed = any(f'{ch}_smooth' in df.columns for ch in EMG_CH)
    for ch in EMG_CH:
        if ch in df.columns:
            raw = df[ch].to_numpy(dtype=float)
            if have_smoothed and f'{ch}_smooth' in df.columns:
                sm = df[f'{ch}_smooth'].to_numpy(dtype=float)
                ax_emg.plot(t, sm, linewidth=0.9, label=f'{ch}_smooth')
                ax_emg.plot(t, raw, linewidth=0.4, alpha=0.25, color=ax_emg.lines[-1].get_color())
            else:
                ax_emg.plot(t, raw, linewidth=0.8, label=ch)
    ax_emg.set_ylabel('EMG (norm units)')
    ax_emg.set_xlabel('time [s]')
    ax_emg.set_title('EMG envelopes (solid) & raw magnitude (faint)')
    ax_emg.legend(fontsize=7, ncol=4, loc='upper right')

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / f'{csv_path.stem}_full_profile.png'
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f'[ok] {csv_path.name} -> {out_png.name}')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-dir', type=Path, required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    args = ap.parse_args()
    files = sorted([p for p in Path(args.input_dir).glob('*.csv')])
    if not files:
        raise SystemExit('No CSV files found.')
    for fp in files:
        plot_file(fp, Path(args.out_dir))
    print('[done] Full profile PNG generation complete')

if __name__ == '__main__':
    main()
