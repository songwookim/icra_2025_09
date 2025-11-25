#!/usr/bin/env python3
"""Plot aggregated stiffness (k) profiles across all signaligned CSV demos.

Scans a directory (default: outputs/stiffness_profiles_signaligned) for files:
  *_signaligned.csv and *_signaligned_aug*.csv

Extracts columns:
  th_k1..k3, if_k1..k3, mf_k1..k3 and builds per-finger/time aligned arrays.

Outputs:
  1) Line plot: all demos (light), mean (bold), median (dashed), IQR band, min-max band.
  2) Distribution (multi-modal) snapshot: violin per stiffness dimension at user-specified time slices.

Usage:
    # 기본: 진행률(0~1) 정규화 후 보간
    python plot_stiffness_profiles.py \
        --dir /home/songwoo/ros2_ws/icra2025/src/hri_falcon_robot_bridge/outputs/stiffness_profiles_signaligned \
        --target_len 200 \
        --time_slices 0.0 0.25 0.5 0.75 1.0

    # 절대 시간 스케일 (정규화 없이) 사용:
    python plot_stiffness_profiles.py \
        --dir ... \
        --no_time_norm \
        --target_len 300 \
        --time_slices 0 30 60 90 120  # 단위: 초

Optional:
    --exclude_aug    원본(_signaligned.csv)만 사용
    --save_prefix    출력 파일명 prefix
    --no_time_norm   time_s 그대로(공통 겹치는 최소 duration 기준) 절대 시간 보간
    --raw_time_only  보간/정규화 없이 각 데모 원본 시간축 그대로 라인 플롯 (분포/violin 생략)
    --raw_mean_len   raw_time_only 모드에서 mean 계산용 보간 그리드 길이 (기본 500)

Generates PNGs in the same directory.
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Dict

K_COLS = [
    'th_k1','th_k2','th_k3',
    'if_k1','if_k2','if_k3',
    'mf_k1','mf_k2','mf_k3'
]

def load_csvs(root: Path, exclude_aug: bool):
    files = []
    for f in sorted(root.glob('*_signaligned*.csv')):
        name = f.name
        if exclude_aug and 'aug' in name:
            continue
        if 'signaligned.csv' in name or 'signaligned_aug' in name:
            files.append(f)
    return files

def resample_time_normalized(df: pd.DataFrame, target_len: int) -> Optional[pd.DataFrame]:
    """진행률(0~1) 정규화 후 target_len에 맞춰 선형 보간"""
    t = df['time_s'].values
    if len(t) < 2:
        return None
    t_norm = ((t - t[0]) / (t[-1] - t[0] + 1e-12)).astype(float)
    new_t = np.linspace(0, 1, target_len).astype(float)
    out = {'time_axis': new_t}
    for c in K_COLS:
        v = np.asarray(df[c].values, dtype=float)
        out[c] = np.interp(new_t, t_norm, v)
    return pd.DataFrame(out)

def resample_time_absolute(df: pd.DataFrame, target_len: int, common_end: float) -> Optional[pd.DataFrame]:
    """절대 시간(초) 기반 보간: 0~common_end 범위 공통 시간축으로 맞춤.
    모든 데모의 최소 duration(common_end)까지만 사용하여 겹치는 구간 통계 왜곡 방지."""
    t = df['time_s'].values.astype(float)
    if len(t) < 2:
        return None
    t_rel = (t - t[0]).astype(float)
    # 데모 자체 길이
    dur = t_rel[-1]
    # 공통 구간 초과분 제거 위해 마스크
    mask = t_rel <= common_end + 1e-12
    t_rel = t_rel[mask]
    out_time = np.linspace(0, common_end, target_len).astype(float)
    out = {'time_axis': out_time}
    for c in K_COLS:
        v = np.asarray(df[c].values[mask], dtype=float)
        out[c] = np.interp(out_time, t_rel, v)
    return pd.DataFrame(out)

def aggregate(dfs: List[pd.DataFrame]) -> Dict[str, np.ndarray]:
    # Stack: N x T x D
    arr = np.stack([d[K_COLS].values for d in dfs], axis=0)  # (N, T, 9)
    stats = {
        'all': arr,
        'mean': np.mean(arr, axis=0),
        'median': np.median(arr, axis=0),
        'q25': np.quantile(arr, 0.25, axis=0),
        'q75': np.quantile(arr, 0.75, axis=0),
        'min': np.min(arr, axis=0),
        'max': np.max(arr, axis=0)
    }
    return stats

def plot_time_series(stats: Dict[str, np.ndarray], time_axis: np.ndarray, save_path: Path, absolute: bool):
    all_arr = stats['all']  # (N, T, 9)
    mean = stats['mean']
    median = stats['median']
    q25 = stats['q25']
    q75 = stats['q75']
    minv = stats['min']
    maxv = stats['max']

    fig, axes = plt.subplots(3, 3, figsize=(15, 10), sharex=True)
    axes = axes.flatten()
    for i, col in enumerate(K_COLS):
        ax = axes[i]
        # Individual demos
        for demo in all_arr:
            ax.plot(time_axis, demo[:, i], color='tab:blue', alpha=0.12, linewidth=0.8)
        # IQR band
        ax.fill_between(time_axis, q25[:, i], q75[:, i], color='orange', alpha=0.25, label='IQR' if i==0 else None)
        # Min-max band (thin)
        ax.fill_between(time_axis, minv[:, i], maxv[:, i], color='gray', alpha=0.1, label='Min-Max' if i==0 else None)
        # Mean & median
        ax.plot(time_axis, mean[:, i], color='red', linewidth=2.0, label='Mean' if i==0 else None)
        ax.plot(time_axis, median[:, i], color='black', linestyle='--', linewidth=1.5, label='Median' if i==0 else None)
        ax.set_title(col)
        ax.grid(alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4)
    fig.suptitle('Stiffness Profiles (Absolute Time)' if absolute else 'Stiffness Profiles (Normalized Progress)')
    fig.text(0.5, 0.02, 'Time (s)' if absolute else 'Normalized Progress (0→1)', ha='center')
    fig.tight_layout(rect=(0,0.04,1,0.95))
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)

def plot_violin_slices(stats: Dict[str, np.ndarray], time_axis: np.ndarray, slices: List[float], save_path: Path, absolute: bool):
    all_arr = stats['all']  # (N, T, 9)
    T = all_arr.shape[1]
    fig, axes_obj = plt.subplots(1, len(slices), figsize=(4*len(slices)+2, 6), sharey=True)
    if isinstance(axes_obj, np.ndarray):
        axes_list = list(axes_obj.flat)
    else:
        axes_list = [axes_obj]
    for ax, s in zip(axes_list, slices):
        # s is in [0,1]; map to index
        if absolute:
            # s는 초 단위이므로 비율로 변환
            total = time_axis[-1]
            ratio = s / (total + 1e-12)
            idx = int(np.clip(ratio, 0, 1) * (T - 1))
        else:
            # s는 0~1 progress
            idx = int(np.clip(s, 0, 1) * (T - 1))
        data_at_t = all_arr[:, idx, :]  # (N, 9)
        ax.violinplot([data_at_t[:, i] for i in range(9)], showmeans=True, showextrema=False)
        ax.set_title(f'Time={s:.2f}s' if absolute else f'Progress={s:.2f}')
        ax.set_xticks(range(1, 10))
        ax.set_xticklabels([c for c in K_COLS], rotation=45, ha='right')
        ax.grid(alpha=0.3)
    fig.suptitle('Multi-Modal Distribution (Violin) at Time Slices (Absolute Time)' if absolute else 'Multi-Modal Distribution (Violin) at Progress Slices')
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', type=str, default='outputs/stiffness_profiles_signaligned')
    ap.add_argument('--target_len', type=int, default=200)
    ap.add_argument('--exclude_aug', action='store_true')
    ap.add_argument('--save_prefix', type=str, default='aggregate_k_profiles')
    ap.add_argument('--time_slices', type=float, nargs='*', default=[0.0, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument('--no_time_norm', action='store_true', help='절대 시간(초) 스케일 사용 (진행률 정규화 비활성)')
    ap.add_argument('--raw_time_only', action='store_true', help='원본 데모 시간축 그대로 플롯 (스케일/violin 미사용)')
    ap.add_argument('--raw_mean_len', type=int, default=500)
    args = ap.parse_args()

    root = Path(args.dir).resolve()
    files = load_csvs(root, args.exclude_aug)
    if len(files) == 0:
        print(f'[WARN] No CSV files found in {root}')
        return
    print(f'[INFO] Using {len(files)} files')

    raw_dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            for c in K_COLS:
                if c not in df.columns:
                    raise ValueError(f'Missing column {c} in {f.name}')
            raw_dfs.append(df)
        except Exception as e:
            print(f'[ERROR] {f.name}: {e}')
    if len(raw_dfs) == 0:
        print('[ERROR] No valid trajectories')
        return

    # RAW TIME ONLY 모드: 원본 시간축 그대로 플롯 후 종료
    if args.raw_time_only:
        rel_time_dfs = []
        durations = []
        for df in raw_dfs:
            t = df['time_s'].values.astype(float)
            t_rel = t - t[0]
            out = {'time_rel': t_rel}
            for c in K_COLS:
                out[c] = df[c].values.astype(float)
            rel_time_dfs.append(pd.DataFrame(out))
            durations.append(t_rel[-1])
        common_end = min(durations)
        # Mean 계산: 공통 최소 duration 구간을 기준으로 균일 그리드로 각 데모 보간 후 평균
        grid = np.linspace(0, common_end, args.raw_mean_len).astype(float)
        stacked = []
        for rdf in rel_time_dfs:
            t_rel = rdf['time_rel'].values
            mask = t_rel <= common_end + 1e-12
            t_rel = t_rel[mask]
            arr = []
            for c in K_COLS:
                v = rdf[c].values[mask]
                arr.append(np.interp(grid, t_rel, v))
            stacked.append(np.stack(arr, axis=1))  # (len(grid), 9)
        stacked_arr = np.stack(stacked, axis=0)  # (N, G, 9)
        mean_arr = stacked_arr.mean(axis=0)      # (G, 9)
        median_arr = np.median(stacked_arr, axis=0)
        q25_arr = np.quantile(stacked_arr, 0.25, axis=0)
        q75_arr = np.quantile(stacked_arr, 0.75, axis=0)

        fig, axes = plt.subplots(3, 3, figsize=(15,10), sharex=False)
        axes = axes.flatten()
        for i, col in enumerate(K_COLS):
            ax = axes[i]
            for rdf in rel_time_dfs:
                ax.plot(rdf['time_rel'].values, rdf[col].values, color='tab:blue', alpha=0.18, linewidth=0.8)
            ax.fill_between(grid, q25_arr[:, i], q75_arr[:, i], color='orange', alpha=0.25, label='IQR' if i==0 else None)
            ax.plot(grid, mean_arr[:, i], color='red', linewidth=2.0, label='Mean' if i==0 else None)
            ax.plot(grid, median_arr[:, i], color='black', linestyle='--', linewidth=1.3, label='Median' if i==0 else None)
            ax.set_title(col)
            ax.grid(alpha=0.3)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=4)
        fig.suptitle('Stiffness Profiles (Raw Demo Time)')
        fig.text(0.5, 0.02, f'Time (s) | common_end={common_end:.2f}s', ha='center')
        fig.tight_layout(rect=(0,0.04,1,0.95))
        raw_path = root / f'{args.save_prefix}_raw_timeseries.png'
        fig.savefig(str(raw_path), dpi=150)
        plt.close(fig)
        print(f'[OK] Saved {raw_path}')
        print('[INFO] raw_time_only 모드: violin/normalized/absolute 플롯 생략')
        return

    if args.no_time_norm:
        # 공통 겹치는 최소 duration
        durations = [df['time_s'].iloc[-1] - df['time_s'].iloc[0] for df in raw_dfs]
        common_end = min(durations)
        print(f'[INFO] Absolute time mode: common_end={common_end:.3f}s (min of {len(durations)} demos)')
        resampled = []
        for df in raw_dfs:
            rd = resample_time_absolute(df, args.target_len, common_end)
            if rd is not None:
                resampled.append(rd)
    else:
        resampled = []
        for df in raw_dfs:
            rd = resample_time_normalized(df, args.target_len)
            if rd is not None:
                resampled.append(rd)
    if len(resampled) == 0:
        print('[ERROR] No valid trajectories')
        return

    stats = aggregate(resampled)
    time_axis = resampled[0]['time_axis'].values
    # Time series plot
    ts_path = root / f'{args.save_prefix}_timeseries.png'
    plot_time_series(stats, time_axis, ts_path, absolute=args.no_time_norm)
    print(f'[OK] Saved {ts_path}')
    # Violin slices
    vs_path = root / f'{args.save_prefix}_violin.png'
    if len(args.time_slices) == 0:
        print('[INFO] No time_slices provided -> skip violin plot')
    else:
        slices = args.time_slices if args.no_time_norm else [min(1.0, max(0.0, s)) for s in args.time_slices]
        plot_violin_slices(stats, time_axis, slices, vs_path, absolute=args.no_time_norm)
        print(f'[OK] Saved {vs_path}')

if __name__ == '__main__':
    main()
