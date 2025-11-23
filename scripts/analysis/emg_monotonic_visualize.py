#!/usr/bin/env python3
"""Visualize EMG intent vs stiffness (raw + monotonic calibrated).

Loads latest (or user-specified) JSON produced by `emg_monotonic_mapping.py` and re-aggregates
stiffness+EMG pairs to recreate the raw intent signal using saved NNLS weights. Then applies the
piecewise monotonic mapping and plots:
  - Scatter: raw intent vs k_norm
  - Monotonic curve (piecewise constant or stepped)
  - Overlay monotonic calibrated stiffness vs raw intent order
  - Text box with Spearman raw/mono metrics

Usage examples:
  python3 emg_monotonic_visualize.py --limit 3
  python3 emg_monotonic_visualize.py --mapping-json /path/to/emg_monotonic_mapping_YYYYMMDD_HHMMSS.json
"""
import argparse
import json
import os
from pathlib import Path
from typing import Optional, List, Dict
import numpy as np
import pandas as pd
import matplotlib
if os.environ.get("DISPLAY", "") == "":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

STIFFNESS_DIR = Path("/home/songwoo/ros2_ws/icra2025/src/hri_falcon_robot_bridge/outputs/stiffness_profiles")
LOGS_DIR = Path("/home/songwoo/ros2_ws/icra2025/src/hri_falcon_robot_bridge/outputs/logs")
ANALYSIS_DIR = STIFFNESS_DIR.parent / "analysis"


def find_latest_mapping_json(explicit: Optional[Path]) -> Path:
    if explicit:
        return explicit
    if not ANALYSIS_DIR.is_dir():
        raise SystemExit(f"Analysis directory not found: {ANALYSIS_DIR}")
    files = sorted(ANALYSIS_DIR.glob("emg_monotonic_mapping_*.json"))
    if not files:
        raise SystemExit("No mapping JSON files found. Run emg_monotonic_mapping.py first.")
    return files[-1]


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
    emg_cols = [c for c in df.columns if c.startswith("emg_ch")]  # channels inferred from mapping weights
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
        raise RuntimeError("No valid stiffness+EMG pairs aggregated for visualization")
    return pd.concat(frames, ignore_index=True)


def compute_intent(df: pd.DataFrame, weights: Dict[str, float]) -> np.ndarray:
    intent = np.zeros(len(df))
    for ch, w in weights.items():
        if ch not in df.columns:
            continue
        intent += df[ch].to_numpy(dtype=float) * w
    # normalize to [0,1]
    mn, mx = intent.min(), intent.max()
    if mx > mn:
        intent = (intent - mn) / (mx - mn)
    else:
        intent[:] = 0.0
    return intent


def apply_piecewise_mapping(intent: np.ndarray, mapping: Dict[str, List[float]]) -> np.ndarray:
    bx = np.array(mapping.get("breakpoints_x", []), dtype=float)
    by = np.array(mapping.get("breakpoints_y", []), dtype=float)
    if bx.size == 0 or by.size == 0:
        return np.zeros_like(intent)
    # piecewise constant interpolation
    return np.interp(intent, bx, by)


def plot_finger(intent: np.ndarray, k_norm: np.ndarray, k_mono: np.ndarray, mapping: Dict[str, List[float]], finger: str, out_dir: Path, metrics: Dict[str, float]):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7,5))
    ax.scatter(intent, k_norm, s=6, alpha=0.35, color="#1f77b4", label="raw k_norm")
    # mapping curve
    bx = np.array(mapping.get("breakpoints_x", []), dtype=float)
    by = np.array(mapping.get("breakpoints_y", []), dtype=float)
    if bx.size and by.size:
        ax.plot(bx, by, color="#d62728", linewidth=2.0, label="monotonic mapping")
    ax.scatter(intent, k_mono, s=5, alpha=0.25, color="#ff7f0e", label="mono interp")
    ax.set_title(f"Finger {finger.upper()} Intent Mapping")
    ax.set_xlabel("Intent (normalized)")
    ax.set_ylabel("Stiffness norm (N/m)")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=8)
    txt = f"Spearman raw={metrics.get('spearman_raw',0):.3f}\nSpearman mono={metrics.get('spearman_mono',0):.3f}"
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top", ha="left", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", alpha=0.7, ec="#999"))
    out_path = out_dir / f"intent_mapping_{finger}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def parse_args():
    p = argparse.ArgumentParser(description="Visualize EMG intent monotonic mapping")
    p.add_argument("--limit", type=int, default=5, help="Max number of stiffness files to aggregate")
    p.add_argument("--pattern", type=str, default="*_synced.csv", help="Glob pattern for stiffness profiles")
    p.add_argument("--mapping-json", type=Path, default=None, help="Explicit mapping JSON path (auto-latest if omitted)")
    p.add_argument("--out-dir", type=Path, default=ANALYSIS_DIR / "plots", help="Directory to write plots")
    return p.parse_args()


def main():
    args = parse_args()
    mapping_json = find_latest_mapping_json(args.mapping_json)
    with mapping_json.open("r") as f:
        mapping_data = json.load(f)
    files = sorted(STIFFNESS_DIR.glob(args.pattern))
    if not files:
        raise SystemExit("No stiffness files found for visualization")
    df = aggregate(files, args.limit)
    per_finger = mapping_data.get("per_finger", {})
    generated = []
    for finger, info in per_finger.items():
        weights = info.get("weights", {})
        mapping = info.get("mapping", {})
        metrics = info.get("metrics", {})
        target_col = f"{finger}_k_norm"
        if target_col not in df.columns:
            print(f"[warn] Missing target column for finger {finger}; skipping")
            continue
        intent = compute_intent(df, weights)
        k_norm = df[target_col].to_numpy(dtype=float)
        k_mono = apply_piecewise_mapping(intent, mapping)
        out_path = plot_finger(intent, k_norm, k_mono, mapping, finger, args.out_dir, metrics)
        generated.append(str(out_path))
    print("Generated plots:")
    for p in generated:
        print("  -", p)


if __name__ == "__main__":
    main()
