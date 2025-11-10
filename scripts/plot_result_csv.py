#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Matplotlib fallback for headless if needed
if os.environ.get('DISPLAY','') == '':
    import matplotlib
    matplotlib.use('Agg')


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Basic normalization: add t (sec + nsec*1e-9)
    if 't_sec' in df.columns and 't_nanosec' in df.columns:
        try:
            df['t'] = df['t_sec'].astype(float) + df['t_nanosec'].astype(float) * 1e-9
            t0 = float(df['t'].iloc[0])
            df['t_rel'] = df['t'] - t0
        except Exception:
            pass
    return df


def _series_or_zeros(df: pd.DataFrame, col: str) -> pd.Series:
    """Return numeric series for a column; keep NaNs for missing/invalid values.
    Keeping NaNs avoids plotting long flat zeros before data arrives."""
    if col in df.columns:
        try:
            return pd.to_numeric(df[col], errors='coerce')
        except Exception:
            return pd.to_numeric(pd.Series(df[col]), errors='coerce')
    else:
        return pd.Series(np.nan, index=df.index)


def _mag3(df: pd.DataFrame, cols, smooth: int = 0):
    try:
        x = _series_or_zeros(df, cols[0])
        y = _series_or_zeros(df, cols[1])
        z = _series_or_zeros(df, cols[2])
        m = (x.pow(2) + y.pow(2) + z.pow(2)).pow(0.5)
        if isinstance(smooth, int) and smooth and smooth > 1:
            m = m.rolling(window=smooth, min_periods=1, center=False).mean()
        return m
    except Exception:
        return pd.Series(np.nan, index=df.index)


def _mag3_with_fallback(df: pd.DataFrame, std_cols, fb_cols, smooth: int = 0):
    """Compute magnitude using standard columns first; if missing, try fallback columns.
    Returns None if neither set is fully available."""
    cols = None
    if all(c in df.columns for c in std_cols):
        cols = std_cols
    elif fb_cols and all(c in df.columns for c in fb_cols):
        cols = fb_cols
    if cols is None:
        return None
    return _mag3(df, cols, smooth)


def _components3_with_fallback(df: pd.DataFrame, std_cols, fb_cols, smooth: int = 0):
    """Return per-axis series (x, y, z) using primary or fallback column names."""
    cols = None
    if all(c in df.columns for c in std_cols):
        cols = std_cols
    elif fb_cols and all(c in df.columns for c in fb_cols):
        cols = fb_cols
    if cols is None:
        return None
    comps = []
    for col in cols:
        series = _series_or_zeros(df, col)
        if isinstance(smooth, int) and smooth and smooth > 1:
            series = series.rolling(window=smooth, min_periods=1, center=False).mean()
        comps.append(series)
    return comps


def plot_force(df: pd.DataFrame, axarr, sensors=(1,2,3), smooth: int = 0, torque_scale: float = 1.0):
    """센서별로 |F|, |T|, |F|+|T| (또는 고정 스케일 s 적용) 스칼라를 그립니다. 축은 [s1, s2, s3] 세 줄을 가정합니다.
    torque_scale는 |T|에 곱하는 고정 배율로, 기본 1.0 (단순 합)."""
    t = df.get('t_rel', df.index)
    component_labels = [('Fx', '#1f77b4'), ('Fy', '#ff9896'), ('Fz', '#9467bd')]
    
    sensor_configs = [
        (1, 0, ['s1_fx','s1_fy','s1_fz'], ['s1_tx','s1_ty','s1_tz'], 's1 magnitudes'),
        (2, 1, ['s2_fx','s2_fy','s2_fz'], ['s2_tx','s2_ty','s2_tz'], 's2 magnitudes'),
        (3, 2, ['s3_fx','s3_fy','s3_fz'], ['s3_tx','s3_ty','s3_tz'], 's3 magnitudes'),
    ]
    
    for sensor_id, ax_idx, f_cols, t_cols, title in sensor_configs:
        if sensor_id not in sensors or ax_idx >= len(axarr):
            continue
            
        f_mag = _mag3_with_fallback(df, f_cols, None, smooth)
        f_components = _components3_with_fallback(df, f_cols, None, smooth)
        t_mag = _mag3_with_fallback(df, t_cols, None, smooth)
        
        ax = axarr[ax_idx]
        lines = 0
        scale = float(torque_scale) if torque_scale is not None else 1.0
        
        if f_mag is not None:
            ax.plot(t, f_mag, label=f'|F| (s{sensor_id})', color='#1f77b4', linewidth=1.5)
            lines += 1
        if f_components is not None:
            for (label, color), series in zip(component_labels, f_components):
                ax.plot(t, series, label=f'{label} (s{sensor_id})', color=color, linestyle='--', linewidth=1.0, alpha=0.85)
                lines += 1
        if t_mag is not None:
            ax.plot(t, t_mag, label=f'|T| (s{sensor_id})', color='#ff7f0e', linewidth=1.5)
            lines += 1
        # Combined
        if f_mag is not None and t_mag is not None:
            comb = f_mag + (scale * t_mag)
            comb_label = f'|F| + |T| (s{sensor_id})' if abs(scale - 1.0) < 1e-9 else f'|F| + {scale:.3g}·|T| (s{sensor_id})'
            ax.plot(t, comb, label=comb_label, color='#2ca02c', linewidth=2.2, alpha=0.9)
            lines += 1
            
        ax.set_title(title)
        ax.set_xlabel('t [s]')
        ax.set_ylabel('force / torque')
        ax.grid(True, alpha=0.3)
        if lines:
            ax.legend(loc='best', fontsize=8)


def plot_deform(df: pd.DataFrame, ax):
    t = df.get('t_rel', df.index)
    lines = 0
    if 'deform_circ' in df.columns:
        ax.plot(t, df['deform_circ'], label='circularity')
        lines += 1
    if 'deform_ecc' in df.columns:
        ax.plot(t, df['deform_ecc'], label='eccentricity')
        lines += 1
    ax.set_title('Deformity metrics')
    ax.set_xlabel('t [s]')
    if lines:
        ax.legend(loc='best', fontsize=8)
    # 요청에 따라 deform은 0~1 범위로 고정
    ax.set_ylim(0.0, 1.0)


def plot_emg(df: pd.DataFrame, ax):
    t = df.get('t_rel', df.index)
    chs = [c for c in df.columns if c.startswith('emg_ch')]
    if not chs:
        ax.set_title('EMG (no data)')
        return
    for c in chs:
        ax.plot(t, df[c], label=c, alpha=0.8)
    ax.set_title('EMG channels')
    ax.set_xlabel('t [s]')
    ax.legend(ncol=4, fontsize=7, loc='upper right')


def plot_ee(df: pd.DataFrame, ax):
    """Plot EE position components for legacy and new dual topics (MF/TH)."""
    t = df.get('t_rel', df.index)
    groups = [
        ('ee',   ['ee_px','ee_py','ee_pz'],   'EE '),
        ('ee_mf',['ee_mf_px','ee_mf_py','ee_mf_pz'], 'EE MF '),
        ('ee_th',['ee_th_px','ee_th_py','ee_th_pz'], 'EE TH '),
    ]
    colors = {
        'ee_px': '#1f77b4', 'ee_py': '#2ca02c', 'ee_pz': '#ff7f0e',
        'ee_mf_px': '#17becf', 'ee_mf_py': '#98df8a', 'ee_mf_pz': '#ffbb78',
        'ee_th_px': '#9467bd', 'ee_th_py': '#8c564b', 'ee_th_pz': '#c49c94',
    }
    plotted = 0
    for prefix, cols, label_prefix in groups:
        if all(c in df.columns for c in cols):
            for c in cols:
                ax.plot(t, df[c], label=label_prefix + c.split('_')[-1], color=colors.get(c, None), linewidth=1.5)
            plotted += 1
    if plotted == 0:
        ax.set_title('EE position (no data)')
        return
    ax.set_title('EE position (px, py, pz)')
    ax.set_xlabel('t [s]')
    ax.legend(loc='best', fontsize=8, ncol=3)


def _find_latest_csv(search_roots):
    candidates = []
    for root in search_roots:
        root = Path(root)
        if not root.exists():
            continue
        # Prefer logger-named files first
        candidates.extend(root.rglob('*_synced.csv'))
        if not candidates:
            candidates.extend(root.rglob('*.csv'))
    if not candidates:
        return None
    candidates = [p for p in candidates if p.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def main():
    p = argparse.ArgumentParser(description='Plot synced CSV outputs from data_logger_node')
    p.add_argument('csv', nargs='?', default='', type=str,
                   help='Path to CSV (omit to auto-pick latest from outputs/logs)')
    p.add_argument('--save', type=str, default='', help='If set, save figure to this file instead of showing')
    p.add_argument('--force-smooth', type=int, default=0, help='Rolling mean window for |F|,|T| (0=off)')
    p.add_argument('--torque-scale', type=float, default=1.0, help='|T| 배율 (결합=|F| + s·|T|, 기본 1.0)')
    args = p.parse_args()

    if args.csv:
        csv_path = Path(args.csv)
    else:
        # Try to autodetect the latest CSV from multiple likely locations
        cwd = Path.cwd()
        script_dir = Path(__file__).resolve().parent
        pkg_dir = script_dir.parent
        search_roots = [
            cwd / 'outputs' / 'logs',
            cwd,
            pkg_dir / 'outputs' / 'logs',
            pkg_dir,
        ]
        latest = _find_latest_csv(search_roots)
        if latest is None:
            print('No CSV path provided and no CSV found under outputs/logs. Provide a CSV path.', file=sys.stderr)
            sys.exit(2)
        print(f'Auto-selected latest CSV: {latest}')
        csv_path = latest

    if not csv_path.exists():
        print(f'CSV not found: {csv_path}', file=sys.stderr)
        sys.exit(2)

    df = load_csv(csv_path)

    # Figure 1: Force (s1, s2, s3) + EMG
    fig1, axes1 = plt.subplots(4, 1, figsize=(12, 12), constrained_layout=True)
    plot_force(df, axes1[0:3], sensors=(1,2,3), smooth=args.force_smooth, torque_scale=args.torque_scale)
    plot_emg(df, axes1[3])
    fig1.suptitle('Force Sensors & EMG', fontsize=14, fontweight='bold')

    # Figure 2: Deformity + EE positions (if, mf, th)
    fig2, axes2 = plt.subplots(2, 1, figsize=(12, 6), constrained_layout=True)
    plot_deform(df, axes2[0])
    plot_ee(df, axes2[1])
    fig2.suptitle('Deformity Metrics & EE Positions', fontsize=14, fontweight='bold')

    if args.save:
        # Save both figures with suffix
        out = Path(args.save)
        stem = out.stem
        ext = out.suffix or '.png'
        out1 = out.parent / f'{stem}_force_emg{ext}'
        out2 = out.parent / f'{stem}_deform_ee{ext}'
        out1.parent.mkdir(parents=True, exist_ok=True)
        out2.parent.mkdir(parents=True, exist_ok=True)
        fig1.savefig(out1, dpi=150)
        fig2.savefig(out2, dpi=150)
        print(f'Saved figures -> {out1}, {out2}')
    else:
        plt.show()


if __name__ == '__main__':
    main()
