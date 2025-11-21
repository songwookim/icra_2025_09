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
    """센서별 Fx/Fy/Fz 성분만 그립니다 (|F|, 토크, 합산 라인 모두 제거)."""
    # Ensure numpy array for matplotlib (avoid pandas Series multi-dim indexing issue)
    t = df.get('t_rel', df.index)
    t_np = np.asarray(t)
    component_labels = [('Fx', '#1f77b4'), ('Fy', '#ff9896'), ('Fz', '#9467bd')]
    
    sensor_configs = [
        (1, 0, ['s1_fx','s1_fy','s1_fz'], ['s1_tx','s1_ty','s1_tz'], 's1 magnitudes'),
        (2, 1, ['s2_fx','s2_fy','s2_fz'], ['s2_tx','s2_ty','s2_tz'], 's2 magnitudes'),
        (3, 2, ['s3_fx','s3_fy','s3_fz'], ['s3_tx','s3_ty','s3_tz'], 's3 magnitudes'),
    ]
    
    for sensor_id, ax_idx, f_cols, t_cols, title in sensor_configs:
        if sensor_id not in sensors or ax_idx >= len(axarr):
            continue
            
        # |F| 제거 -> 성분만 사용
        f_components = _components3_with_fallback(df, f_cols, None, smooth)
        # 토크 관련 플롯은 사용자 요청으로 비활성화
        
        ax = axarr[ax_idx]
        lines = 0
        scale = float(torque_scale) if torque_scale is not None else 1.0
        
        if f_components is not None:
            for (label, color), series in zip(component_labels, f_components):
                ax.plot(t_np, np.asarray(series), label=f'{label} (s{sensor_id})', color=color, linestyle='--', linewidth=1.0, alpha=0.85)
                lines += 1
        # 토크 합산(|Tx|+|Ty|+|Tz|) 라인 제거됨
            
        ax.set_title(title)
        ax.set_xlabel('t [s]')
        ax.set_ylabel('force / torque')
        ax.grid(True, alpha=0.3)
        if lines:
            ax.legend(loc='best', fontsize=8)


def plot_deform(df: pd.DataFrame, ax):
    """Plot only eccentricity (circularity removed)."""
    t_np = np.asarray(df.get('t_rel', df.index))
    if 'deform_ecc' in df.columns:
        ax.plot(t_np, np.asarray(df['deform_ecc']), label='eccentricity', color='#1f77b4')
        ax.legend(loc='best', fontsize=8)
    ax.set_title('Deformity (eccentricity)')
    ax.set_xlabel('t [s]')
    ax.set_ylim(0.0, 1.0)


def plot_emg(df: pd.DataFrame, ax):
    t = df.get('t_rel', df.index)
    t_np = np.asarray(t)
    chs = [c for c in df.columns if c.startswith('emg_ch')]
    if not chs:
        ax.set_title('EMG (no data)')
        return
    for c in chs:
        ax.plot(t_np, np.asarray(df[c]), label=c, alpha=0.8)
    ax.set_title('EMG channels')
    ax.set_xlabel('t [s]')
    ax.legend(ncol=4, fontsize=7, loc='upper right')


def plot_ee_3d(df: pd.DataFrame, ax3d):
    """Plot EE trajectories (IF legacy ee_* or ee_if_*, MF ee_mf_*, TH ee_th_*) in 3D."""
    groups = [
        ('IF', ['ee_if_px','ee_if_py','ee_if_pz'], ['ee_px','ee_py','ee_pz']),  # new format, fallback to legacy
        ('MF', ['ee_mf_px','ee_mf_py','ee_mf_pz'], None),
        ('TH', ['ee_th_px','ee_th_py','ee_th_pz'], None),
    ]
    color_map = {
        'IF': '#1f77b4',
        'MF': '#2ca02c',
        'TH': '#ff7f0e',
    }
    # We now vary alpha over time instead of hue. Hue remains constant per finger.
    any_plotted = False
    # Optional time array for gradient (normalize 0~1)
    t_rel = df.get('t_rel', None)
    for label, cols, fallback_cols in groups:
        # Try primary columns first, then fallback (for IF backward compatibility)
        active_cols = None
        if all(c in df.columns for c in cols):
            active_cols = cols
        elif fallback_cols and all(c in df.columns for c in fallback_cols):
            active_cols = fallback_cols
        
        if active_cols is None:
            continue
            
        arr = df[active_cols].to_numpy(dtype=float)
        # Build segment list for gradient coloring
        if arr.shape[0] < 2:
            # Fallback: just a point
            ax3d.scatter(arr[:,0], arr[:,1], arr[:,2], label=label, color=color_map.get(label,'#555555'), s=12)
            any_plotted = True
            continue
        from mpl_toolkits.mplot3d.art3d import Line3DCollection
        import matplotlib.colors as mcolors
        import matplotlib.cm as cm
        segments = np.stack([arr[:-1, :], arr[1:, :]], axis=1)  # (N-1, 2, 3)
        nseg = segments.shape[0]
        if t_rel is not None and len(t_rel) == arr.shape[0]:
            # Normalize time to 0~1 and map midpoints of segments
            t_norm = (t_rel - t_rel.min()) / (max(1e-12, (t_rel.max() - t_rel.min())))
            t_mid = (t_norm[:-1] + t_norm[1:]) * 0.5
        else:
            t_mid = np.linspace(0.0, 1.0, nseg)
        base_rgb = np.array(mcolors.to_rgb(color_map.get(label,'#555555')))
        # Alpha ramp (earlier segments more 투명, 후반 진하게)
        alphas = 0.25 + 0.75 * t_mid  # range ~0.25 -> 1.0
        colors = [(*base_rgb, a) for a in alphas]
        # Line3DCollection expects a sequence of (2,3) arrays, convert ndarray to list
        segments_list = [segments[i] for i in range(nseg)]
        lc = Line3DCollection(segments_list, colors=colors, linewidth=2.0)
        ax3d.add_collection3d(lc)
        # Dummy handle for legend (single solid color)
        ax3d.plot([], [], [], color=color_map.get(label,'#555555'), label=label, linewidth=2.0)
        any_plotted = True
    if not any_plotted:
        ax3d.set_title('EE 3D (no data)')
        return
    ax3d.set_title('EE Trajectories (3D)')
    ax3d.set_xlabel('X [m]')
    ax3d.set_ylabel('Y [m]')
    ax3d.set_zlabel('Z [m]')
    ax3d.legend(loc='upper right', fontsize=8)
    try:
        # Equal aspect for better spatial perception
        xs, ys, zs = [], [], []
        for label, cols, fallback_cols in groups:
            active_cols = None
            if all(c in df.columns for c in cols):
                active_cols = cols
            elif fallback_cols and all(c in df.columns for c in fallback_cols):
                active_cols = fallback_cols
            if active_cols is None:
                continue
            arr = df[active_cols].to_numpy(dtype=float)
            xs.append(arr[:,0]); ys.append(arr[:,1]); zs.append(arr[:,2])
        if xs:
            xmin, xmax = min(a.min() for a in xs), max(a.max() for a in xs)
            ymin, ymax = min(a.min() for a in ys), max(a.max() for a in ys)
            zmin, zmax = min(a.min() for a in zs), max(a.max() for a in zs)
            max_range = max(xmax - xmin, ymax - ymin, zmax - zmin)
            cx = (xmin + xmax)/2.0; cy = (ymin + ymax)/2.0; cz = (zmin + zmax)/2.0
            ax3d.set_xlim(cx - max_range/2.0, cx + max_range/2.0)
            ax3d.set_ylim(cy - max_range/2.0, cy + max_range/2.0)
            ax3d.set_zlim(cz - max_range/2.0, cz + max_range/2.0)
    except Exception:
        pass


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
    p.add_argument('csv', nargs='?', default='/home/songwoo/ros2_ws/icra2025/src/hri_falcon_robot_bridge/outputs/analysis/stiffness_profiles_global_tk/20251122_023936_synced_aug4_paper_profile.csv', type=str,
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

    # Figure 2: Deformity (eccentricity only) + EE 3D trajectories
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig2 = plt.figure(figsize=(12, 6))
    ax_deform = fig2.add_subplot(1, 2, 1)
    ax_ee3d = fig2.add_subplot(1, 2, 2, projection='3d')
    plot_deform(df, ax_deform)
    plot_ee_3d(df, ax_ee3d)
    fig2.suptitle('Deformity (eccentricity) & EE 3D', fontsize=14, fontweight='bold')

    if args.save:
        # Save both figures with suffix
        out = Path(args.save)
        stem = out.stem
        ext = out.suffix or '.png'
        out1 = out.parent / f'{stem}_force_emg{ext}'
        out2 = out.parent / f'{stem}_deform_ee{ext}'
        out1.parent.mkdir(parents=True, exist_ok=True)
        out2.parent.mkdir(parents=True, exist_ok=True)
        fig1.savefig(str(out1), dpi=150)
        fig2.savefig(str(out2), dpi=150)
        print(f'Saved figures -> {out1}, {out2}')
    else:
        plt.show()


if __name__ == '__main__':
    main()
