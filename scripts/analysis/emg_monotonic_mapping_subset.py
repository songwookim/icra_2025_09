#!/usr/bin/env python3
"""Channel subset NNLS + isotonic regression mapping.

손가락별로 EMG 채널과 stiffness k_norm 간 Spearman 상관을 계산하여 상위 top-k 채널만 선택.
선택 채널로 NNLS 가중치 재학습 후 PAV(Isotonic) 적용하여 기존(전체 채널) 대비 성능 비교.

출력:
  - JSON: per_finger (full vs subset) metrics + 채널 랭킹
  - CSV: subset 가중치 / metrics

Usage:
  python3 emg_monotonic_mapping_subset.py --limit 5 --top-k 4
"""
import argparse
import json
from pathlib import Path
import os
from datetime import datetime
from typing import List, Optional, Dict
import numpy as np
import pandas as pd
from scipy.optimize import nnls

STIFFNESS_DIR = Path("/home/songwoo/ros2_ws/icra2025/src/hri_falcon_robot_bridge/outputs/stiffness_profiles")
LOGS_DIR = Path("/home/songwoo/ros2_ws/icra2025/src/hri_falcon_robot_bridge/outputs/logs")
OUT_DIR = STIFFNESS_DIR.parent / "analysis"

# --------------------------- Data Utilities ---------------------------

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
        if limit is not None and used >= limit:
            break
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

# --------------------------- Modeling ---------------------------

def fit_nnls(df: pd.DataFrame, channels: List[str], target_col: str) -> Dict[str, float]:
    X = df[channels].to_numpy(); y = df[target_col].to_numpy()
    w, _ = nnls(X, y)
    return {ch: float(val) for ch, val in zip(channels, w)}


def compute_intent(df: pd.DataFrame, weights: Dict[str, float]) -> np.ndarray:
    intent = np.zeros(len(df))
    for ch, w in weights.items():
        if ch in df.columns:
            intent += df[ch].to_numpy(dtype=float) * w
    mn, mx = intent.min(), intent.max()
    if mx > mn: intent = (intent - mn)/(mx - mn)
    else: intent[:] = 0.0
    return intent


def pav_isotonic(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    order = np.argsort(x); xs = x[order]; ys = y[order].astype(float).copy()
    starts = list(range(len(ys))); ends = list(range(len(ys)))
    i = 0
    while i < len(starts) - 1:
        if ys[ends[i]] > ys[starts[i+1]]:
            new_start = starts[i]; new_end = ends[i+1]
            block = range(new_start, new_end+1)
            avg = float(np.mean(ys[block]))
            for idx in block: ys[idx] = avg
            starts[i] = new_start; ends[i] = new_end
            del starts[i+1]; del ends[i+1]
            if i > 0: i -= 1
        else:
            i += 1
    # interpolate back to original order
    ys_mono = np.interp(x, xs, ys)
    return ys_mono


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3: return 0.0
    return float(pd.Series(a).corr(pd.Series(b), method="spearman"))

# --------------------------- Channel Ranking ---------------------------

def rank_channels(df: pd.DataFrame, target_col: str, channels: List[str]) -> List[Dict[str, float]]:
    rows = []
    for ch in channels:
        corr = spearman(df[ch].to_numpy(), df[target_col].to_numpy())
        rows.append({"channel": ch, "spearman": corr})
    rows.sort(key=lambda r: abs(r["spearman"]), reverse=True)
    return rows

# --------------------------- Main Flow ---------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Subset NNLS + isotonic mapping")
    p.add_argument("--limit", type=int, default=5)
    p.add_argument("--pattern", type=str, default="*_synced.csv")
    p.add_argument("--top-k", type=int, default=4, help="Per-finger 상위 채널 수")
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    return p.parse_args()


def main():
    args = parse_args()
    files = sorted(STIFFNESS_DIR.glob(args.pattern))
    if not files: raise SystemExit("No stiffness files found")
    df = aggregate(files, args.limit)
    channels = [c for c in df.columns if c.startswith("emg_ch")]
    result = {"fingers": {}, "meta": {"limit": args.limit, "top_k": args.top_k}}
    metrics_rows = []
    weight_rows = []
    for finger in ["th", "if", "mf"]:
        target = f"{finger}_k_norm"
        if target not in df.columns:
            continue
        ranking = rank_channels(df, target, channels)
        top_subset = [r["channel"] for r in ranking[:args.top_k]]
        # full model
        w_full = fit_nnls(df, channels, target)
        intent_full = compute_intent(df, w_full)
        mono_full = pav_isotonic(intent_full, df[target].to_numpy())
        m_full_raw = spearman(intent_full, df[target].to_numpy())
        m_full_mono = spearman(intent_full, mono_full)
        # subset model
        w_sub = fit_nnls(df, top_subset, target)
        intent_sub = compute_intent(df, w_sub)
        mono_sub = pav_isotonic(intent_sub, df[target].to_numpy())
        m_sub_raw = spearman(intent_sub, df[target].to_numpy())
        m_sub_mono = spearman(intent_sub, mono_sub)
        result["fingers"][finger] = {
            "ranking": ranking,
            "subset_channels": top_subset,
            "weights_full": w_full,
            "weights_subset": w_sub,
            "metrics": {
                "full_raw": m_full_raw,
                "full_mono": m_full_mono,
                "subset_raw": m_sub_raw,
                "subset_mono": m_sub_mono,
                "raw_delta": m_sub_raw - m_full_raw,
                "mono_delta": m_sub_mono - m_full_mono,
            },
        }
        metrics_rows.append({
            "finger": finger,
            "full_raw": m_full_raw,
            "full_mono": m_full_mono,
            "subset_raw": m_sub_raw,
            "subset_mono": m_sub_mono,
            "raw_delta": m_sub_raw - m_full_raw,
            "mono_delta": m_sub_mono - m_full_mono,
        })
        for ch, w in w_sub.items():
            weight_rows.append({"finger": finger, "channel": ch, "weight_subset": w})
    out_dir = args.out_dir; out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    with (out_dir / f"emg_monotonic_subset_{ts}.json").open("w") as f:
        json.dump(result, f, indent=2)
    pd.DataFrame(metrics_rows).to_csv(out_dir / f"emg_monotonic_subset_metrics_{ts}.csv", index=False)
    pd.DataFrame(weight_rows).to_csv(out_dir / f"emg_monotonic_subset_weights_{ts}.csv", index=False)
    print("Subset mapping metrics:")
    for r in metrics_rows:
        print(f"{r['finger']}: raw {r['full_raw']:.3f}->{r['subset_raw']:.3f} (Δ={r['raw_delta']:.3f}), mono {r['full_mono']:.3f}->{r['subset_mono']:.3f} (Δ={r['mono_delta']:.3f})")

if __name__ == "__main__":
    main()
