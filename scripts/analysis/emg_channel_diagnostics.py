#!/usr/bin/env python3
"""EMG 채널 진단: 저상관 원인(노이즈/공선성/분포왜곡) 파악용 종합 리포트.

집계:
 1) 채널별 기본 통계: mean, std, coeff_var, min, max, median, q05, q95, skew, kurtosis
 2) 채널간 Pearson 상관행렬 + 고유치/설명분산(EMG 차원 유효랭크 추정)
 3) 채널별 stiffness(k_norm per finger) Spearman 상관 비교 테이블
 4) 상위 공선성 채널쌍 (|corr| >= threshold)
 5) 히트맵 PNG (corr matrix)

Usage:
  python3 emg_channel_diagnostics.py --limit 5 --out-dir outputs/analysis --pattern *_synced.csv
"""
import argparse
import json
import os
from pathlib import Path
from typing import List, Optional, Dict
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
import matplotlib
if os.environ.get("DISPLAY", "") == "":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

STIFFNESS_DIR = Path("/home/songwoo/ros2_ws/icra2025/src/hri_falcon_robot_bridge/outputs/stiffness_profiles")
LOGS_DIR = Path("/home/songwoo/ros2_ws/icra2025/src/hri_falcon_robot_bridge/outputs/logs")

# ------------------------- Data Utilities -------------------------

def find_matching_log(stiffness_file: Path) -> Optional[Path]:
    basename = stiffness_file.name
    for root, _dirs, files in os.walk(LOGS_DIR):
        if basename in files:
            return Path(root) / basename
    return None

def load_stiffness(stiffness_file: Path) -> pd.DataFrame:
    df = pd.read_csv(stiffness_file)
    time_col = None
    for cand in ["time_s", "t_sec"]:
        if cand in df.columns:
            time_col = cand; break
    if time_col is None:
        raise ValueError(f"No time column in {stiffness_file}")
    df = df.copy(); df.rename(columns={time_col: "time_s"}, inplace=True)
    for prefix in ["th", "if", "mf"]:
        k1, k2, k3 = df[f"{prefix}_k1"], df[f"{prefix}_k2"], df[f"{prefix}_k3"]
        df[f"{prefix}_k_norm"] = np.sqrt(k1*k1 + k2*k2 + k3*k3)
    return df

def load_emg_log(log_file: Path) -> pd.DataFrame:
    df = pd.read_csv(log_file)
    if "t_sec" not in df.columns:
        raise ValueError(f"Missing t_sec in {log_file}")
    t = df["t_sec"].astype(float)
    if "t_nanosec" in df.columns:
        t = t + df["t_nanosec"].astype(float)*1e-9
    df["time_s"] = t
    emg_cols = [c for c in df.columns if c.startswith("emg_ch")]
    if not emg_cols:
        raise ValueError(f"No EMG columns in {log_file}")
    return df[["time_s"] + emg_cols]

def interpolate_emg_to_stiffness(stiff_df: pd.DataFrame, emg_df: pd.DataFrame) -> pd.DataFrame:
    stiff_df = stiff_df.sort_values("time_s"); emg_df = emg_df.sort_values("time_s")
    t_target = stiff_df["time_s"].to_numpy(); t_src = emg_df["time_s"].to_numpy()
    out = stiff_df.copy()
    for col in [c for c in emg_df.columns if c.startswith("emg_ch")]:
        src = emg_df[col].to_numpy()
        out[col] = np.interp(t_target, t_src, src, left=src[0], right=src[-1])
    return out

def aggregate(files: List[Path], limit: Optional[int]) -> pd.DataFrame:
    frames = []; used = 0
    for f in files:
        if limit is not None and used >= limit: break
        log = find_matching_log(f)
        if not log: continue
        try:
            stiff = load_stiffness(f); emg = load_emg_log(log)
            merged = interpolate_emg_to_stiffness(stiff, emg)
            frames.append(merged); used += 1
        except Exception:
            continue
    if not frames:
        raise RuntimeError("No valid stiffness+EMG pairs aggregated")
    return pd.concat(frames, ignore_index=True)

# ------------------------- Diagnostics -------------------------

def channel_basic_stats(df: pd.DataFrame, emg_cols: List[str]) -> List[Dict[str, float]]:
    rows = []
    for col in emg_cols:
        x = df[col].to_numpy(dtype=float)
        mean = float(np.mean(x))
        std = float(np.std(x))
        coeff_var = float(std / (abs(mean) + 1e-9))
        med = float(np.median(x))
        mn = float(np.min(x)); mx = float(np.max(x))
        q05 = float(np.quantile(x, 0.05)); q95 = float(np.quantile(x, 0.95))
        sk = float(skew(x)) if x.size > 10 else 0.0
        ku = float(kurtosis(x)) if x.size > 10 else 0.0
        rows.append({"channel": col, "mean": mean, "std": std, "coeff_var": coeff_var,
                     "median": med, "min": mn, "max": mx, "q05": q05, "q95": q95, "skew": sk, "kurtosis": ku})
    return rows

def stiffness_correlations(df: pd.DataFrame, emg_cols: List[str]) -> Dict[str, Dict[str, float]]:
    out = {}
    for finger in ["th", "if", "mf"]:
        target = df[f"{finger}_k_norm"].to_numpy(dtype=float)
        finger_map = {}
        for col in emg_cols:
            finger_map[col] = float(pd.Series(df[col]).corr(pd.Series(target), method="spearman"))
        out[finger] = finger_map
    return out

def collinearity_pairs(corr_df: pd.DataFrame, threshold: float = 0.9) -> List[Dict[str, object]]:
    rows = []
    for i, c1 in enumerate(corr_df.columns):
        for j, c2 in enumerate(corr_df.columns):
            if j <= i: continue
            val = corr_df.loc[c1, c2]
            if abs(val) >= threshold:
                rows.append({"ch1": c1, "ch2": c2, "pearson": float(val)})
    rows.sort(key=lambda r: abs(r["pearson"]), reverse=True)
    return rows

def corr_eigendecomp(corr_df: pd.DataFrame) -> Dict[str, object]:
    vals, vecs = np.linalg.eigh(corr_df.to_numpy())
    vals = np.sort(vals)[::-1]
    total = np.sum(vals)
    explained = (vals / total).tolist()
    eff_rank = int(np.sum(vals > 1e-2))
    return {"eigenvalues": vals.tolist(), "explained": explained, "effective_rank_lambda>1e-2": eff_rank}

# ------------------------- Plotting -------------------------

def plot_corr_heatmap(corr_df: pd.DataFrame, out_path: Path):
    fig, ax = plt.subplots(figsize=(7,6))
    im = ax.imshow(corr_df.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr_df.columns)))
    ax.set_yticks(range(len(corr_df.columns)))
    ax.set_xticklabels(corr_df.columns, rotation=90, fontsize=8)
    ax.set_yticklabels(corr_df.columns, fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson")
    ax.set_title("EMG Channel Pearson Correlation")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

# ------------------------- Main -------------------------

def parse_args():
    p = argparse.ArgumentParser(description="EMG channel diagnostics")
    p.add_argument("--limit", type=int, default=5)
    p.add_argument("--pattern", type=str, default="*_synced.csv")
    p.add_argument("--out-dir", type=Path, default=STIFFNESS_DIR.parent / "analysis")
    p.add_argument("--collinearity-th", type=float, default=0.9)
    return p.parse_args()

def main():
    args = parse_args()
    files = sorted(STIFFNESS_DIR.glob(args.pattern))
    if not files: raise SystemExit("No stiffness files found")
    df = aggregate(files, args.limit)
    emg_cols = [c for c in df.columns if c.startswith("emg_ch")]
    basic_rows = channel_basic_stats(df, emg_cols)
    corr_df = df[emg_cols].corr(method="pearson")
    stiff_corrs = stiffness_correlations(df, emg_cols)
    col_pairs = collinearity_pairs(corr_df, threshold=args.collinearity_th)
    eig_info = corr_eigendecomp(corr_df)
    # Spearman 평균 ranking
    avg_spearman = {ch: float(np.mean([stiff_corrs[f][ch] for f in stiff_corrs.keys()])) for ch in emg_cols}
    ranking = sorted(avg_spearman.items(), key=lambda kv: abs(kv[1]), reverse=True)
    out_dir = args.out_dir; out_dir.mkdir(parents=True, exist_ok=True)
    plot_path = out_dir / "emg_channel_corr_heatmap.png"
    plot_corr_heatmap(corr_df, plot_path)
    ts = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"emg_channel_diagnostics_{ts}.json"
    csv_basic = out_dir / f"emg_channel_basic_{ts}.csv"
    csv_stiff = out_dir / f"emg_channel_stiff_corr_{ts}.csv"
    csv_pairs = out_dir / f"emg_channel_collinearity_pairs_{ts}.csv"
    pd.DataFrame(basic_rows).to_csv(csv_basic, index=False)
    stiff_rows = []
    for finger, mp in stiff_corrs.items():
        for ch, val in mp.items():
            stiff_rows.append({"finger": finger, "channel": ch, "spearman_k_norm": val})
    pd.DataFrame(stiff_rows).to_csv(csv_stiff, index=False)
    pd.DataFrame(col_pairs).to_csv(csv_pairs, index=False)
    report = {
        "basic_stats": basic_rows,
        "stiffness_correlations": stiff_corrs,
        "avg_spearman_ranking": ranking,
        "collinearity_pairs_threshold": args.collinearity_th,
        "collinearity_pairs": col_pairs,
        "corr_eigendecomp": eig_info,
        "heatmap_png": str(plot_path),
    }
    with json_path.open("w") as f:
        json.dump(report, f, indent=2)
    print("Saved EMG diagnostics:")
    print("  JSON:", json_path)
    print("  Basic CSV:", csv_basic)
    print("  Stiffness Corr CSV:", csv_stiff)
    print("  Collinearity CSV:", csv_pairs)
    print("  Heatmap:", plot_path)

if __name__ == "__main__":
    main()
