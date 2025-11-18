#!/usr/bin/env python3
"""Automate Leave-One-Trial-Out cross-validation for stiffness policy benchmarks.

This script runs the benchmark pipeline 15 times (once per demo), holding out
a different trajectory each time. Results are saved to timestamped subdirectories
and aggregated into a summary CSV with mean±std metrics.

Usage:
    python3 run_loto_cv.py --models gmr --gmm-components 3 --gmm-covariance diag
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_SCRIPT = SCRIPT_DIR / "run_stiffness_policy_benchmarks.py"
DEFAULT_LOG_DIR = Path("outputs") / "logs" / "success"
DEFAULT_OUTPUT_DIR = Path("outputs") / "models" / "stiffness_policies" / "loto_cv"


def gather_demo_names(log_dir: Path) -> List[str]:
    """Return sorted list of demo stems from log_dir."""
    if not log_dir.exists():
        raise RuntimeError(f"Log directory '{log_dir}' does not exist.")
    csvs = sorted(log_dir.glob("*.csv"))
    if not csvs:
        raise RuntimeError(f"No CSV files found in '{log_dir}'.")
    stems = [p.stem for p in csvs if not p.name.endswith("_paper_profile.csv")]
    if not stems:
        raise RuntimeError(f"No demo CSVs (non-profile) found in '{log_dir}'.")
    return stems


def run_single_fold(
    demo: str,
    args: argparse.Namespace,
    fold_idx: int,
    total_folds: int,
) -> Dict[str, Any]:
    """Run benchmark with --eval-demo=<demo>, return parsed metrics."""
    output_subdir = args.output_dir / f"fold_{fold_idx:02d}_{demo}"
    output_subdir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(BENCHMARK_SCRIPT),
        "--eval-demo",
        demo,
        "--mode",
        args.mode,
        "--models",
        args.models,
        "--log-dir",
        str(args.log_dir),
        "--stiffness-dir",
        str(args.stiffness_dir),
        "--output-dir",
        str(output_subdir),
        "--stride",
        str(args.stride),
        "--sequence-window",
        str(args.sequence_window),
        "--seed",
        str(args.seed),
        "--gmm-components",
        str(args.gmm_components),
        "--gmm-covariance",
        args.gmm_covariance,
        "--gmm-samples",
        str(args.gmm_samples),
    ]
    # Add BC-specific arguments if BC is in the models list
    if "bc" in args.models:
        cmd.extend(["--bc-epochs", str(args.bc_epochs)])
    if args.save_predictions:
        cmd.append("--save-predictions")
    # Always disable tensorboard for robustness when torch may be absent
    cmd.append("--no-tensorboard")

    print(f"\n[LOTO-CV {fold_idx+1}/{total_folds}] Evaluating {demo}")
    print(f"  command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"[warn] fold {fold_idx} (demo={demo}) failed with exit code {result.returncode}")
        return {}

    # Parse the latest summary JSON
    summaries = sorted(output_subdir.glob("benchmark_summary_*.json"))
    if not summaries:
        print(f"[warn] no summary JSON found for fold {fold_idx}")
        return {}
    latest = summaries[-1]
    with latest.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    models = data.get("models", {})
    # Flatten metrics per model
    flat: Dict[str, Any] = {}
    for model_name, metrics in models.items():
        for metric_name, value in metrics.items():
            flat[f"{model_name}_{metric_name}"] = float(value)
    flat["demo"] = demo
    flat["fold"] = fold_idx
    return flat


def aggregate_results(records: List[Dict[str, Any]], out_csv: Path) -> None:
    """Compute mean±std across folds and save to CSV."""
    if not records:
        print("[warn] no valid folds to aggregate; creating empty summary CSV.")
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=["metric","mean","std","min","max","count"]).to_csv(out_csv, index=False)
        return
    df = pd.DataFrame(records)
    demo_col = df.pop("demo") if "demo" in df.columns else None
    fold_col = df.pop("fold") if "fold" in df.columns else None
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    summary_rows: List[Dict[str, Any]] = []
    for col in numeric_cols:
        vals = df[col].dropna()
        if vals.size == 0:
            continue
        summary_rows.append(
            {
                "metric": col,
                "mean": float(vals.mean()),
                "std": float(vals.std(ddof=1) if vals.size > 1 else 0.0),
                "min": float(vals.min()),
                "max": float(vals.max()),
                "count": int(vals.size),
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_csv, index=False, float_format="%.4f")
    print(f"\n[done] aggregated LOTO-CV summary saved to {out_csv}")
    print("\nKey metrics (mean ± std):")
    for _, row in summary_df.iterrows():
        if any(
            key in row["metric"].lower() for key in ["rmse", "mae", "r2", "nll"]
        ):
            print(f"  {row['metric']:30s}: {row['mean']:.4f} ± {row['std']:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Leave-One-Trial-Out CV for stiffness policy benchmarks."
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=DEFAULT_LOG_DIR,
        help="Directory with raw demonstration CSVs",
    )
    parser.add_argument(
        "--stiffness-dir",
        type=Path,
        default=Path("outputs") / "analysis" / "stiffness_profiles",
        help="Directory with *_paper_profile.csv outputs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Root directory for LOTO-CV results",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="gmr",
        help="Comma-separated models to evaluate (e.g., gmr,bc,lstm_gmm)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="unified",
        choices=["unified", "per-finger"],
        help="Benchmark mode: unified (20D->9D) or per-finger (3×8D->3D)",
    )
    parser.add_argument(
        "--bc-epochs",
        type=int,
        default=100,
        help="Training epochs for BC model",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Subsample stride for demonstrations",
    )
    parser.add_argument(
        "--sequence-window",
        type=int,
        default=1,
        help="Temporal window length for sequence models",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (used for any internal splits)",
    )
    parser.add_argument(
        "--gmm-components",
        type=int,
        default=3,
        help="Number of GMM mixture components",
    )
    parser.add_argument(
        "--gmm-covariance",
        type=str,
        default="diag",
        choices=["full", "diag", "spherical", "tied"],
        help="GMM covariance type",
    )
    parser.add_argument(
        "--gmm-samples",
        type=int,
        default=16,
        help="Samples per query for stochastic GMM",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save per-sample predictions to CSV (passed to benchmark script)",
    )
    parser.add_argument(
        "--no-tensorboard",
        action="store_true",
        help="Disable TensorBoard logging (passed to benchmark script)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    demos = gather_demo_names(args.log_dir)
    print(f"[info] running LOTO-CV with {len(demos)} folds")
    print(f"  demos: {', '.join(demos)}")
    print(f"  models: {args.models}")
    print(f"  output: {args.output_dir}")

    records: List[Dict[str, Any]] = []
    for idx, demo in enumerate(demos):
        fold_result = run_single_fold(demo, args, idx, len(demos))
        if fold_result:
            records.append(fold_result)

    aggregate_csv = args.output_dir / "loto_cv_summary.csv"
    aggregate_results(records, aggregate_csv)
    print(f"\n[done] LOTO-CV complete. Results in {args.output_dir}")


if __name__ == "__main__":
    main()
