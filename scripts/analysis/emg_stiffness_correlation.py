#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


STIFFNESS_DIR = Path("/home/songwoo/ros2_ws/icra2025/src/hri_falcon_robot_bridge/outputs/stiffness_profiles")
LOGS_DIR = Path("/home/songwoo/ros2_ws/icra2025/src/hri_falcon_robot_bridge/outputs/logs")


def find_matching_log(stiffness_file: Path) -> Optional[Path]:
    """Find a log CSV (with EMG) sharing same basename anywhere under LOGS_DIR.
    Returns first match or None.
    """
    basename = stiffness_file.name
    # Walk through logs directory once building a map (cache could be added later)
    for root, _dirs, files in os.walk(LOGS_DIR):
        if basename in files:
            return Path(root) / basename
    return None


def load_stiffness(stiffness_file: Path) -> pd.DataFrame:
    df = pd.read_csv(stiffness_file)
    # Expect columns like th_k1,th_k2,th_k3, if_k1.., mf_k1.. existing.
    required_sets = [
        ["th_k1", "th_k2", "th_k3"],
        ["if_k1", "if_k2", "if_k3"],
        ["mf_k1", "mf_k2", "mf_k3"],
    ]
    for cols in required_sets:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing stiffness columns {missing} in {stiffness_file}")
    # time column
    time_col = None
    for cand in ["time_s", "t_sec"]:  # some variants might reuse t_sec
        if cand in df.columns:
            time_col = cand
            break
    if time_col is None:
        raise ValueError(f"No time column found in {stiffness_file}")
    df = df.copy()
    df.rename(columns={time_col: "time_s"}, inplace=True)
    # k-norm computation
    for prefix in ["th", "if", "mf"]:
        k1, k2, k3 = df[f"{prefix}_k1"], df[f"{prefix}_k2"], df[f"{prefix}_k3"]
        df[f"{prefix}_k_norm"] = np.sqrt(k1 * k1 + k2 * k2 + k3 * k3)
    return df


def load_emg_log(log_file: Path) -> pd.DataFrame:
    df = pd.read_csv(log_file)
    # EMG time: combine seconds + nanoseconds where available
    if "t_sec" in df.columns:
        base_t = df["t_sec"].astype(float)
        if "t_nanosec" in df.columns:
            base_t = base_t + df["t_nanosec"].astype(float) * 1e-9
        df["time_s"] = base_t
    else:
        raise ValueError(f"Log file {log_file} missing t_sec for time base")
    emg_cols = [c for c in df.columns if c.startswith("emg_ch")]
    if not emg_cols:
        raise ValueError(f"No EMG channels in {log_file}")
    return df[["time_s"] + emg_cols]


def interpolate_emg_to_stiffness(stiff_df: pd.DataFrame, emg_df: pd.DataFrame) -> pd.DataFrame:
    # Ensure sorted
    stiff_df = stiff_df.sort_values("time_s")
    emg_df = emg_df.sort_values("time_s")
    t_target = stiff_df["time_s"].to_numpy()
    t_src = emg_df["time_s"].to_numpy()
    result = stiff_df.copy()
    for col in [c for c in emg_df.columns if c.startswith("emg_ch")]:
        src = emg_df[col].to_numpy()
        # Handle potential edge extrapolation: clamp
        interp = np.interp(t_target, t_src, src, left=src[0], right=src[-1])
        result[col] = interp
    return result


def aggregate_files(files: List[Path], limit: Optional[int] = None) -> Tuple[pd.DataFrame, List[dict]]:
    rows = []
    metadata = []
    count = 0
    for f in files:
        if limit is not None and count >= limit:
            break
        try:
            stiff = load_stiffness(f)
        except Exception as e:
            metadata.append({"file": str(f), "status": "skip_stiffness", "error": str(e)})
            continue
        log_path = find_matching_log(f)
        if log_path is None:
            metadata.append({"file": str(f), "status": "skip_no_log"})
            continue
        try:
            emg = load_emg_log(log_path)
        except Exception as e:
            metadata.append({"file": str(f), "log": str(log_path), "status": "skip_emg", "error": str(e)})
            continue
        merged = interpolate_emg_to_stiffness(stiff, emg)
        merged["source_file"] = f.name
        rows.append(merged)
        metadata.append({"file": str(f), "log": str(log_path), "status": "ok", "n_rows": len(merged)})
        count += 1
    if not rows:
        raise RuntimeError("No valid stiffness + EMG pairs aggregated")
    big = pd.concat(rows, ignore_index=True)
    return big, metadata


def compute_correlations(df: pd.DataFrame) -> dict:
    emg_cols = [c for c in df.columns if c.startswith("emg_ch")]
    stiffness_norm_cols = ["th_k_norm", "if_k_norm", "mf_k_norm"]
    # Pearson
    pearson = df[emg_cols + stiffness_norm_cols].corr(method="pearson")
    spearman = df[emg_cols + stiffness_norm_cols].corr(method="spearman")
    result = {
        "pearson": pearson.to_dict(),
        "spearman": spearman.to_dict(),
    }
    # Channel ranking by mean absolute pearson against k_norms
    rankings = []
    for ch in emg_cols:
        vals = [pearson.at[ch, k] for k in stiffness_norm_cols]
        rankings.append({"channel": ch, "mean_abs_corr": float(np.mean(np.abs(vals)))})
    rankings.sort(key=lambda r: r["mean_abs_corr"], reverse=True)
    result["ranking_mean_abs_corr"] = rankings
    return result


def write_outputs(out_dir: Path, corr: dict, metadata: List[dict]):
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    # JSON summary
    with (out_dir / f"emg_stiffness_correlation_{ts}.json").open("w") as f:
        json.dump({"correlations": corr, "metadata": metadata}, f, indent=2)
    # Plain CSV for easier viewing (pearson only)
    pearson_df = pd.DataFrame(corr["pearson"])
    pearson_df.to_csv(out_dir / f"emg_stiffness_pearson_{ts}.csv", index=True)
    spearman_df = pd.DataFrame(corr["spearman"])
    spearman_df.to_csv(out_dir / f"emg_stiffness_spearman_{ts}.csv", index=True)
    ranking_df = pd.DataFrame(corr["ranking_mean_abs_corr"])
    ranking_df.to_csv(out_dir / f"emg_stiffness_channel_ranking_{ts}.csv", index=False)


def parse_args():
    p = argparse.ArgumentParser(description="Compute EMG vs stiffness correlation across synced sessions.")
    p.add_argument("--limit", type=int, default=5, help="Max number of stiffness profile files to process")
    p.add_argument("--out-dir", type=str, default=str(STIFFNESS_DIR.parent / "analysis"), help="Output directory")
    p.add_argument("--pattern", type=str, default="*_synced.csv", help="Filename glob under stiffness_profiles")
    return p.parse_args()


def main():
    args = parse_args()
    pattern = args.pattern
    files = sorted(STIFFNESS_DIR.glob(pattern))
    if not files:
        raise SystemExit(f"No stiffness profile files found for pattern {pattern}")
    aggregated, metadata = aggregate_files(files, limit=args.limit)
    corr = compute_correlations(aggregated)
    write_outputs(Path(args.out_dir), corr, metadata)
    # Minimal stdout summary
    print("Processed files:")
    for m in metadata:
        print(m)
    print("Ranking (mean_abs_corr):")
    for r in corr["ranking_mean_abs_corr"]:
        print(r)


if __name__ == "__main__":
    main()
