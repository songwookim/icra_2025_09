#!/usr/bin/env python3
"""Visualize parametric intent→stiffness mappings vs actual proxy stiffness for one demo.

Loads:
  - parametric_remap_summary.json (weights, per-finger parametric fits)
  - One success log CSV containing EMG + force (e.g. s1_fx, s1_fy, s1_fz ... emg_ch1..8)

Figure layout (saved as PNG):
  Row 1 (3 subplots): per-finger time series of proxy stiffness (force magnitude) vs parametric best curve evaluated on intent.
  Row 2 (3 subplots): mapping domain scatter (intent01 vs normalized stiffness) + isotonic baseline + parametric candidates (piecewise/logistic/exp/power/linear).

Proxy stiffness:
  For each finger (th/if/mf) use baseline-removed |F| magnitude of its force triple (s1,s2,s3) then scale to [0,1] by provided k_min/k_max in summary.

Usage:
  python3 visualize_parametric_mapping_example.py \
      --summary outputs/parametric_remap_success/parametric_remap_summary.json \
      --csv src/hri_falcon_robot_bridge/outputs/logs/success/20251122_022712_synced.csv \
      --out outputs/parametric_remap_success/parametric_mapping_example.png

"""
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt

EPS = 1e-9

def _load_json(path: Path) -> Dict:
    with open(path, 'r') as f:
        return json.load(f)

def _load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError('CSV empty')
    return df

def _time(df: pd.DataFrame) -> np.ndarray:
    if {'t_sec','t_nanosec'}.issubset(df.columns):
        t = df['t_sec'].to_numpy(float) + df['t_nanosec'].to_numpy(float) * 1e-9
    elif 'time_s' in df.columns:
        t = df['time_s'].to_numpy(float)
    else:
        t = np.arange(len(df), dtype=float)
    return t - float(t[0])

def _estimate_fs(t: np.ndarray) -> float:
    if len(t) < 3: return 200.0
    dt = np.diff(t)
    dt = dt[(dt>0)&np.isfinite(dt)]
    if dt.size==0: return 200.0
    return 1.0/np.median(dt)

def _baseline_remove(arr: np.ndarray, frac: float = 0.1) -> np.ndarray:
    n = arr.shape[0]
    take = max(1,int(round(n*frac)))
    base = arr[:take].mean(axis=0, keepdims=True)
    return arr - base

def _lpf(arr: np.ndarray, fs: float, cut: float=3.0) -> np.ndarray:
    if fs <= 2*cut: return arr
    try:
        sos = butter(2, cut, btype='low', fs=fs, output='sos')
        return sosfiltfilt(sos, arr, axis=0)
    except Exception:
        return arr

def _intent(emg: np.ndarray, weights: List[float]) -> np.ndarray:
    w = np.asarray(weights, dtype=float)
    return emg @ w

def _norm01(x: np.ndarray) -> np.ndarray:
    xmin, xmax = float(x.min()), float(x.max())
    if xmax - xmin < EPS: return np.zeros_like(x)
    return (x - xmin) / (xmax - xmin)

def _piecewise_eval(x: np.ndarray, xp: List[float], yp: List[float]) -> np.ndarray:
    return np.interp(x, xp, yp)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--summary', required=True)
    ap.add_argument('--csv', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--show', action='store_true', help='Display instead of Agg save')
    args = ap.parse_args()

    summary = _load_json(Path(args.summary))
    df = _load_csv(Path(args.csv))
    t = _time(df)
    fs = _estimate_fs(t)

    # EMG
    emg_cols = [c for c in df.columns if c.startswith('emg_ch')]
    if not emg_cols:
        raise ValueError('No EMG channels in CSV')
    emg_raw = df[emg_cols].to_numpy(float)
    emg_center = _baseline_remove(emg_raw)
    emg_mag = np.abs(emg_center)
    # Minimal smoothing (LPF each channel)
    emg_smooth = _lpf(emg_mag, fs, cut=2.0)

    intent = _intent(emg_smooth, summary['weights'])
    intent01 = _norm01(intent)

    fingers = ['th','if','mf']
    sensor_map = {'th':'s1','if':'s2','mf':'s3'}

    # Collect per-finger proxy stiffness (force magnitude baseline removed)
    finger_proxy = {}
    for f in fingers:
        s = sensor_map[f]
        cols = [f'{s}_fx', f'{s}_fy', f'{s}_fz']
        if not all(c in df.columns for c in cols):
            finger_proxy[f] = np.zeros(len(df))
            continue
        F = df[cols].to_numpy(float)
        Fc = _baseline_remove(F)
        Fm = np.sqrt(np.sum(Fc**2, axis=1))
        Fm_lpf = _lpf(Fm[:,None], fs, cut=3.0)[:,0]
        finger_proxy[f] = Fm_lpf

    # Prepare figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5))
    top_axes = axes[0]
    map_axes = axes[1]

    for idx, f in enumerate(fingers):
        ax_t = top_axes[idx]
        ax_m = map_axes[idx]
        pdata = summary['fingers'].get(f)
        if pdata is None:
            ax_t.text(0.5,0.5,'No data',ha='center',va='center')
            continue
        k_min = pdata['k_min']
        k_max = pdata['k_max']
        # Normalize proxy to [0,1] using k_min/k_max (clip)
        proxy = finger_proxy[f]
        proxy_scaled = np.clip((proxy - k_min)/(k_max - k_min + EPS), 0, 1)

        # Best function
        best_name = pdata['best_function']
        param_set = pdata['parametric']
        # Evaluate candidates on intent01
        curves = {}
        for name, info in param_set.items():
            if name == 'linear':
                curves[name] = intent01
            elif name == 'power':
                a = info['alpha']
                curves[name] = intent01 ** a
            elif name == 'exp':
                b = info['beta']
                curves[name] = 1.0 - np.exp(-b*intent01)
            elif name == 'logistic':
                a = info['a']; b = info['b']
                raw = 1/(1+np.exp(-a*(intent01 - b)))
                curves[name] = _norm01(raw)
            elif name == 'piecewise':
                xp = info['xp']; yp = info['yp']
                curves[name] = _piecewise_eval(intent01, xp, yp)
        # Time series: proxy vs best parametric scaled
        best_curve = curves.get(best_name, intent01)
        best_scaled = k_min + (k_max - k_min) * best_curve
        proxy_abs = k_min + (k_max - k_min) * proxy_scaled
        ax_t.plot(t, proxy_abs, color='#1f77b4', linewidth=1.2, label='proxy stiffness')
        ax_t.plot(t, best_scaled, color='#d62728', linewidth=1.2, alpha=0.8, label=f'{best_name} mapped')
        ax_t.set_title(f'{f} time series')
        ax_t.set_xlabel('t [s]')
        ax_t.set_ylabel('stiffness (scaled)')
        ax_t.grid(alpha=0.2)
        ax_t.legend(fontsize=8)

        # Mapping scatter
        ax_m.scatter(intent01, proxy_scaled, s=6, alpha=0.3, color='#1f77b4', label='proxy points')
        # Plot curves
        color_map = {'piecewise':'#ff7f0e','logistic':'#d62728','exp':'#2ca02c','power':'#9467bd','linear':'#7f7f7f'}
        x_line = np.linspace(0,1,300)
        for name, info in param_set.items():
            if name == 'linear':
                y_line = x_line
            elif name == 'power':
                y_line = x_line ** info['alpha']
            elif name == 'exp':
                y_line = 1.0 - np.exp(-info['beta']*x_line)
            elif name == 'logistic':
                a = info['a']; b = info['b']
                raw = 1/(1+np.exp(-a*(x_line - b)))
                y_line = _norm01(raw)
            elif name == 'piecewise':
                y_line = _piecewise_eval(x_line, info['xp'], info['yp'])
            else:
                continue
            ax_m.plot(x_line, y_line, label=name, color=color_map.get(name,'black'), linewidth=1.1, alpha=0.9 if name==best_name else 0.5)
        ax_m.set_title(f'{f} mapping domain')
        ax_m.set_xlabel('intent01')
        ax_m.set_ylabel('norm stiffness (0-1)')
        ax_m.grid(alpha=0.2)
        ax_m.legend(fontsize=7, ncol=2)

    fig.suptitle('Parametric Mapping Example (proxy force→stiffness)', fontsize=14)
    fig.tight_layout(rect=(0,0,1,0.96))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    if args.show:
        plt.show()
    plt.close(fig)
    print(f'[ok] Saved figure to {out_path}')

if __name__ == '__main__':
    main()
