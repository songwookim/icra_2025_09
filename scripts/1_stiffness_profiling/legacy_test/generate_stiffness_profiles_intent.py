#!/usr/bin/env python3
"""Generate integrated intent–stiffness profile dataset.

Reads:
  - parametric_remap_summary.json (piecewise params, weights, intent_min/max)
  - success logs (CSV with emg_ch*, s1_fx,s1_fy,s1_fz, s2_*, s3_*)

Outputs (out-dir):
  - intent_stiffness_aggregated.csv : concatenated rows from all logs with columns:
        file_id, time_s, intent_raw, intent01,
        th_k_ratio, th_k_target,
        if_k_ratio, if_k_target,
        mf_k_ratio, mf_k_target
        (optional) th_proxy, if_proxy, mf_proxy (force magnitude baselines)
  - intent_stiffness_summary.json : global stats (min/max/mean per finger & intent distribution)

Piecewise mapping uses per-finger xp, yp from summary[parametric][piecewise].
If piecewise missing, falls back to best_function curve; if none, linear.

Usage:
  python3 generate_stiffness_profiles_intent.py \
      --summary src/hri_falcon_robot_bridge/outputs/parametric_remap_success/parametric_remap_summary.json \
      --logs-dir src/hri_falcon_robot_bridge/outputs/logs/success \
      --out-dir src/hri_falcon_robot_bridge/outputs/stiffness_profiles_intent
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd

EPS = 1e-9

def _load_json(p: Path) -> Dict:
    with open(p,'r') as f: return json.load(f)

def _load_csv(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p)
    if df.empty:
        raise ValueError(f"CSV empty: {p.name}")
    return df

def _time(df: pd.DataFrame) -> np.ndarray:
    if {'t_sec','t_nanosec'}.issubset(df.columns):
        t = df['t_sec'].to_numpy(float) + df['t_nanosec'].to_numpy(float)*1e-9
    elif 'time_s' in df.columns:
        t = df['time_s'].to_numpy(float)
    else:
        t = np.arange(len(df), dtype=float)
    return t - float(t[0])

def _baseline(arr: np.ndarray, frac=0.1) -> np.ndarray:
    n = arr.shape[0]
    take = max(1,int(round(n*frac)))
    base = arr[:take].mean(axis=0, keepdims=True)
    return arr - base

def _intent(emg: np.ndarray, weights: np.ndarray, imin: float, imax: float) -> (np.ndarray, np.ndarray):
    raw = emg @ weights
    norm = (raw - imin)/(imax - imin + EPS)
    norm = np.clip(norm,0.0,1.0)
    return raw, norm

def _piecewise(x: np.ndarray, xp: List[float], yp: List[float]) -> np.ndarray:
    return np.interp(x, xp, yp)

def _recover_curve(name: str, info: Dict, x: np.ndarray) -> np.ndarray:
    if name == 'linear':
        return x
    if name == 'power':
        return x ** info['alpha']
    if name == 'exp':
        return 1.0 - np.exp(-info['beta']*x)
    if name == 'logistic':
        a = info['a']; b = info['b']
        raw = 1/(1+np.exp(-a*(x-b)))
        mn,mx = raw.min(), raw.max()
        return (raw - mn)/(mx - mn + EPS)
    if name == 'piecewise':
        return _piecewise(x, info['xp'], info['yp'])
    return x

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--summary', required=True)
    ap.add_argument('--logs-dir', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--include-proxy', action='store_true', help='Include force magnitude proxies')
    args = ap.parse_args()

    summary = _load_json(Path(args.summary))
    weights = np.asarray(summary['weights'], dtype=float)
    intent_min = summary['intent_min']
    intent_max = summary['intent_max']
    fingers_meta = summary.get('fingers', {})

    logs_dir = Path(args.logs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_files = sorted([p for p in logs_dir.glob('*.csv')])
    if not csv_files:
        raise ValueError('No log CSVs found')

    sensor_map = {'th':'s1','if':'s2','mf':'s3'}
    rows = []

    for csv_path in csv_files:
        try:
            df = _load_csv(csv_path)
        except Exception as e:
            print(f'[skip] {csv_path.name}: {e}')
            continue
        # EMG columns
        emg_cols = [c for c in df.columns if c.startswith('emg_ch')]
        if not emg_cols:
            print(f'[skip] {csv_path.name}: no emg')
            continue
        emg = df[emg_cols].to_numpy(float)
        emg_center = _baseline(emg)
        emg_mag = np.abs(emg_center)
        # Simple smoothing (optional): skip heavy smoothing to preserve tempo
        time = _time(df)
        intent_raw, intent01 = _intent(emg_mag, weights, intent_min, intent_max)

        # Per-finger mapping
        per_finger = {}
        per_finger_ratio = {}
        per_finger_proxy = {}
        for f in ['th','if','mf']:
            meta = fingers_meta.get(f)
            if meta is None:
                # default linear if missing
                per_finger_ratio[f] = intent01
                per_finger[f] = intent01
                continue
            k_min = meta['k_min']; k_max = meta['k_max']
            param = meta['parametric']
            # Prefer piecewise; else fallback to best_function; else linear
            if 'piecewise' in param:
                ratio = _recover_curve('piecewise', param['piecewise'], intent01)
            else:
                best = meta.get('best_function','linear')
                if best in param:
                    ratio = _recover_curve(best, param[best], intent01)
                else:
                    ratio = intent01
            per_finger_ratio[f] = np.clip(ratio,0.0,1.0)
            per_finger[f] = k_min + per_finger_ratio[f]*(k_max - k_min)
            # Optional proxy from force magnitude
            if args.include_proxy:
                s = sensor_map[f]
                colsF = [f'{s}_fx', f'{s}_fy', f'{s}_fz']
                if all(c in df.columns for c in colsF):
                    F = df[colsF].to_numpy(float)
                    Fc = _baseline(F)
                    mag = np.sqrt(np.sum(Fc**2, axis=1))
                    per_finger_proxy[f] = mag
                else:
                    per_finger_proxy[f] = np.zeros(len(df))

        for i in range(len(df)):
            row = {
                'file_id': csv_path.stem,
                'time_s': float(time[i]),
                'intent_raw': float(intent_raw[i]),
                'intent01': float(intent01[i]),
                'th_k_ratio': float(per_finger_ratio['th'][i]),
                'th_k_target': float(per_finger['th'][i]),
                'if_k_ratio': float(per_finger_ratio['if'][i]),
                'if_k_target': float(per_finger['if'][i]),
                'mf_k_ratio': float(per_finger_ratio['mf'][i]),
                'mf_k_target': float(per_finger['mf'][i]),
            }
            if args.include_proxy:
                row['th_proxy'] = float(per_finger_proxy['th'][i])
                row['if_proxy'] = float(per_finger_proxy['if'][i])
                row['mf_proxy'] = float(per_finger_proxy['mf'][i])
            rows.append(row)
        print(f'[ok] {csv_path.name}: {len(df)} rows integrated')

    agg_df = pd.DataFrame(rows)
    agg_csv = out_dir / 'intent_stiffness_aggregated.csv'
    agg_df.to_csv(agg_csv, index=False)

    # Summary stats
    stats = {
        'count_rows': int(len(agg_df)),
        'intent_raw_min': float(agg_df['intent_raw'].min()),
        'intent_raw_max': float(agg_df['intent_raw'].max()),
        'intent01_mean': float(agg_df['intent01'].mean()),
    }
    for f in ['th','if','mf']:
        stats[f] = {
            'k_target_min': float(agg_df[f'{f}_k_target'].min()),
            'k_target_max': float(agg_df[f'{f}_k_target'].max()),
            'k_target_mean': float(agg_df[f'{f}_k_target'].mean()),
        }
    with open(out_dir / 'intent_stiffness_summary.json','w') as f:
        json.dump(stats, f, indent=2)
    print(f'[done] Aggregated CSV: {agg_csv}')
    print(f'[done] Summary JSON: {out_dir / "intent_stiffness_summary.json"}')

if __name__ == '__main__':
    main()
