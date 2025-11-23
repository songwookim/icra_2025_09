#!/usr/bin/env python3
"""Derive monotonic EMG->stiffness intent mapping using NNLS weights + PAV isotonic regression.

Generates per-finger weight vectors (non-negative) and monotonic calibrated curves.
Outputs JSON + CSV summaries for diagnostics:
 - nnls weights
 - raw vs monotonic Spearman correlations (intent vs k_norm)
 - EMG channel collinearity matrix (Pearson among EMG channels)
 - Piecewise mapping breakpoints (intent value vs monotonic stiffness)

Usage:
  python3 emg_monotonic_mapping.py --limit 5 --out-dir /desired/output/dir

"""
import argparse
import json
import os
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import nnls


STIFFNESS_DIR = Path("/home/songwoo/ros2_ws/icra2025/src/hri_falcon_robot_bridge/outputs/stiffness_profiles")
LOGS_DIR = Path("/home/songwoo/ros2_ws/icra2025/src/hri_falcon_robot_bridge/outputs/logs")


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
            time_col = cand
            break
    if time_col is None:
        raise ValueError(f"No time column in {stiffness_file}")
    df = df.copy()
    df.rename(columns={time_col: "time_s"}, inplace=True)
    for prefix in ["th", "if", "mf"]:
        for comp in [1, 2, 3]:
            col = f"{prefix}_k{comp}"
            if col not in df.columns:
                raise ValueError(f"Missing {col} in {stiffness_file}")
        k1, k2, k3 = (df[f"{prefix}_k1"], df[f"{prefix}_k2"], df[f"{prefix}_k3"])
        df[f"{prefix}_k_norm"] = np.sqrt(k1 * k1 + k2 * k2 + k3 * k3)
    return df


def load_emg_log(log_file: Path) -> pd.DataFrame:
    df = pd.read_csv(log_file)
    if "t_sec" not in df.columns:
        raise ValueError(f"Missing t_sec in {log_file}")
    base_t = df["t_sec"].astype(float)
    if "t_nanosec" in df.columns:
        base_t = base_t + df["t_nanosec"].astype(float) * 1e-9
    df["time_s"] = base_t
    emg_cols = [c for c in df.columns if c.startswith("emg_ch")]
    if not emg_cols:
        raise ValueError(f"No EMG columns in {log_file}")
    return df[["time_s"] + emg_cols]


def interpolate_emg_to_stiffness(stiff_df: pd.DataFrame, emg_df: pd.DataFrame) -> pd.DataFrame:
    stiff_df = stiff_df.sort_values("time_s")
    emg_df = emg_df.sort_values("time_s")
    t_target = stiff_df["time_s"].to_numpy()
    t_src = emg_df["time_s"].to_numpy()
    out = stiff_df.copy()
    for col in [c for c in emg_df.columns if c.startswith("emg_ch")]:
        src = emg_df[col].to_numpy()
        out[col] = np.interp(t_target, t_src, src, left=src[0], right=src[-1])
    return out


def aggregate(files: List[Path], limit: Optional[int]) -> pd.DataFrame:
    frames = []
    count = 0
    for f in files:
        if limit is not None and count >= limit:
            break
        log = find_matching_log(f)
        if not log:
            continue
        try:
            stiff = load_stiffness(f)
            emg = load_emg_log(log)
            merged = interpolate_emg_to_stiffness(stiff, emg)
            frames.append(merged)
            count += 1
        except Exception:
            continue
    if not frames:
        raise RuntimeError("No valid stiffness+EMG pairs aggregated")
    return pd.concat(frames, ignore_index=True)


def compute_emg_collinearity(df: pd.DataFrame) -> pd.DataFrame:
    emg_cols = [c for c in df.columns if c.startswith("emg_ch")]
    return df[emg_cols].corr(method="pearson")


def fit_nnls(df: pd.DataFrame, channels: List[str], target_col: str) -> Dict[str, float]:
    X = df[channels].to_numpy()
    y = df[target_col].to_numpy()
    w, _ = nnls(X, y)  # non-negative weights
    return {ch: float(val) for ch, val in zip(channels, w)}


def compute_intent(df: pd.DataFrame, weights: Dict[str, float]) -> np.ndarray:
    intent = np.zeros(len(df))
    for ch, w in weights.items():
        intent += df[ch].to_numpy() * w
    # Normalize intent to [0,1] for stability
    min_v, max_v = intent.min(), intent.max()
    if max_v > min_v:
        intent = (intent - min_v) / (max_v - min_v)
    else:
        intent[:] = 0.0
    return intent


def pav_isotonic(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Pool Adjacent Violators for increasing monotonic relation.
    Returns piecewise (x_sorted, y_mono)."""
    order = np.argsort(x)
    xs = x[order]
    ys = y[order].astype(float).copy()
    # Blocks represented by start/end indices
    starts = list(range(len(ys)))
    ends = list(range(len(ys)))
    i = 0
    while i < len(starts) - 1:
        if ys[ends[i]] > ys[starts[i + 1]]:  # violation
            # merge blocks i and i+1
            new_start = starts[i]
            new_end = ends[i + 1]
            block_indices = range(new_start, new_end + 1)
            avg = float(np.mean(ys[block_indices]))
            for idx in block_indices:
                ys[idx] = avg
            # replace blocks
            starts[i] = new_start
            ends[i] = new_end
            del starts[i + 1]
            del ends[i + 1]
            if i > 0:
                i -= 1
        else:
            i += 1
    return xs, ys


def evaluate_monotonic(intent: np.ndarray, target: np.ndarray, mono_target: np.ndarray) -> Dict[str, float]:
    def safe_spearman(a, b):
        if len(a) < 3:
            return 0.0
        return float(pd.Series(a).corr(pd.Series(b), method="spearman"))
    return {
        "spearman_raw": safe_spearman(intent, target),
        "spearman_mono": safe_spearman(intent, mono_target),
    }


def build_mapping(xs: np.ndarray, ys: np.ndarray, n_points: int = 50) -> Dict[str, List[float]]:
    # Unique breakpoints
    uniq_x, indices = np.unique(xs, return_index=True)
    uniq_y = ys[indices]
    return {
        "breakpoints_x": uniq_x.tolist(),
        "breakpoints_y": uniq_y.tolist(),
    }


def parse_args():
    p = argparse.ArgumentParser(description="NNLS + isotonic regression for EMG->stiffness intent mapping.")
    p.add_argument("--limit", type=int, default=5, help="Max number of stiffness profile base files")
    p.add_argument("--channels", type=str, default="emg_ch1,emg_ch2,emg_ch3,emg_ch4,emg_ch5,emg_ch6,emg_ch7,emg_ch8", help="Comma-separated EMG channel list")
    p.add_argument("--out-dir", type=str, default=str(STIFFNESS_DIR.parent / "analysis"), help="Output directory")
    p.add_argument("--pattern", type=str, default="*_synced.csv", help="File glob pattern in stiffness dir")
    return p.parse_args()


def main():
    args = parse_args()
    channels = [c.strip() for c in args.channels.split(',') if c.strip()]
    files = sorted(STIFFNESS_DIR.glob(args.pattern))
    if not files:
        raise SystemExit("No stiffness files found")
    df = aggregate(files, args.limit)
    # Diagnostics: EMG collinearity
    emg_corr = compute_emg_collinearity(df)

    results = {"finger_models": {}, "emg_collinearity": emg_corr.to_dict()}
    intent_matrix = {}
    mono_mappings = {}
    metrics = {}
    for finger in ["th", "if", "mf"]:
        target_col = f"{finger}_k_norm"
        weights = fit_nnls(df, channels, target_col)
        intent = compute_intent(df, weights)
        xs, ys_mono = pav_isotonic(intent, df[target_col].to_numpy())
        # For evaluation produce monotonic target aligned to original order (interpolate)
        # Build interpolation using piecewise constant segments
        mono_interp = np.interp(intent, xs, ys_mono)
        eval_metrics = evaluate_monotonic(intent, df[target_col].to_numpy(), mono_interp)
        mapping = build_mapping(xs, ys_mono)
        results["finger_models"][finger] = {
            "weights": weights,
            "metrics": eval_metrics,
            "mapping": mapping,
        }
        intent_matrix[finger] = intent.tolist()
        mono_mappings[finger] = {"intent": xs.tolist(), "stiffness_mono": ys_mono.tolist()}
        metrics[finger] = eval_metrics

    # Global summary
    summary = {
        "per_finger": results["finger_models"],
        "emg_collinearity_pearson": results["emg_collinearity"],
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    with (out_dir / f"emg_monotonic_mapping_{ts}.json").open("w") as f:
        json.dump(summary, f, indent=2)
    # Export weights table
    weights_rows = []
    for finger, data in results["finger_models"].items():
        for ch, w in data["weights"].items():
            d = {"finger": finger, "channel": ch, "weight": w}
            weights_rows.append(d)
    pd.DataFrame(weights_rows).to_csv(out_dir / f"emg_monotonic_weights_{ts}.csv", index=False)
    # Metrics CSV
    m_rows = []
    for finger, m in metrics.items():
        m_rows.append({"finger": finger, **m})
    pd.DataFrame(m_rows).to_csv(out_dir / f"emg_monotonic_metrics_{ts}.csv", index=False)
    print("Finger metrics (Spearman raw -> mono):")
    for finger, m in metrics.items():
        print(f"{finger}: {m['spearman_raw']:.4f} -> {m['spearman_mono']:.4f}")


if __name__ == "__main__":
    main()
