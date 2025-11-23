#!/usr/bin/env python3
"""Validate PF/PK decomposition and stiffness mapping consistency.

- Computes TF from demos (as in global TK pipeline)
- Builds projectors PF = T_F^+ T_F and PK = I - PF
- Checks algebraic properties and correlations with force and stiffness
- Saves JSON report and a few diagnostic plots

Usage:
  python3 validate_pf_pk_k.py \
    --log-dir outputs/logs/success \
    --stiffness-dir outputs/analysis/stiffness_profiles_global_tk \
    --out-dir outputs/analysis/validation/pf_pk_k
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# Reuse helpers from generation pipeline
# Ensure we can import helpers from parent scripts directory
import sys
from pathlib import Path as _Path
_THIS_DIR = _Path(__file__).resolve().parent
_SCRIPTS_DIR = _THIS_DIR.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from generate_stiffness_profiles import (
    _load_csv,
    _time_vector,
    _estimate_fs,
    _guess_emg_columns,
    _align_emg_to_time,
    _center_signal,
    _ultra_smooth_strong_emg,
    _compute_normal_force,
    compute_tf,
)


def collect_emg_force(log_paths: List[Path]) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, np.ndarray]]]:
    P_all: List[np.ndarray] = []  # each: (d, N)
    F_all: List[np.ndarray] = []  # each: (m, N), here m=1 (|Fn|)
    perfile: List[Dict[str, np.ndarray]] = []
    for p in log_paths:
        try:
            df = _load_csv(p)
            time = _time_vector(df)
            fs = _estimate_fs(time) or 200.0
            emg_cols = _guess_emg_columns(df)
            if not emg_cols:
                continue
            emg_raw = df[emg_cols].to_numpy(dtype=float)
            aligned_emg = _align_emg_to_time(emg_raw, time)
            if aligned_emg is None:
                continue
            emg_centered = _center_signal(aligned_emg)
            emg_mag = np.abs(emg_centered)
            emg_smooth = _ultra_smooth_strong_emg(emg_mag, fs) or emg_mag
            # Normal force magnitude as in global TK
            Fn = _compute_normal_force(df, prefix="s1", n=(0.0,0.0,1.0))
            # fallback if s1 missing: try s2 then s3
            if Fn is None or len(Fn) != len(time):
                for sensor in ("s2","s3"):
                    Fn = _compute_normal_force(df, prefix=sensor, n=(0.0,0.0,1.0))
                    if Fn is not None and len(Fn) == len(time):
                        break
            if Fn is None:
                continue
            Fn = np.asarray(Fn, dtype=float)
            P = emg_smooth.T  # (d, N)
            F = Fn.reshape(1, -1)  # (1, N)
            P_all.append(P)
            F_all.append(F)
            perfile.append({
                "stem": p.stem,
                "time": time,
                "P": P,
                "F": F,
            })
        except Exception:
            continue
    if not P_all:
        raise RuntimeError("No valid logs found for TF identification")
    P_global = np.hstack(P_all)
    F_global = np.hstack(F_all)
    return P_global, F_global, perfile


def projector_checks(TF: np.ndarray, tol: float = 1e-6) -> Dict[str, float]:
    # Moore-Penrose right pseudoinverse (d x m)
    TF_pinv = np.linalg.pinv(TF)
    d, m = TF.shape[1], TF.shape[0]
    PF = TF_pinv @ TF
    I = np.eye(d)
    PK = I - PF
    # Algebraic properties
    res_idem_pf = np.linalg.norm(PF @ PF - PF, ord="fro") / (d + 1e-12)
    res_idem_pk = np.linalg.norm(PK @ PK - PK, ord="fro") / (d + 1e-12)
    res_comp = np.linalg.norm(PF + PK - I, ord="fro") / (d + 1e-12)
    # Nullspace basis via SVD check
    U, S, Vt = np.linalg.svd(TF, full_matrices=False)
    rank = int(np.sum(S > 1e-8))
    NF = Vt[rank:].T  # (d, d-rank)
    res_null = np.linalg.norm(TF @ NF, ord="fro") / (m + 1e-12)
    return {
        "rank_TF": float(rank),
        "res_idempotent_PF": float(res_idem_pf),
        "res_idempotent_PK": float(res_idem_pk),
        "res_complementary": float(res_comp),
        "res_TF_NF_zero": float(res_null),
    }


def correlation_checks(TF: np.ndarray, perfile: List[Dict[str, np.ndarray]]) -> Dict[str, float]:
    TF_pinv = np.linalg.pinv(TF)
    PF = TF_pinv @ TF
    PK = np.eye(TF.shape[1]) - PF
    r2_force_mean = []
    r2_force_null = []
    dot_pf_pk = []
    for item in perfile:
        P = item["P"]
        F = item["F"]  # (1, N)
        P_pf = PF @ P
        P_pk = PK @ P
        # Predict force from EMG
        F_pred = TF @ P
        F_pred_pf = TF @ P_pf
        # R2 wrt measured F (flatten time)
        y = F.flatten()
        yhat = F_pred.flatten()
        yhat_pf = F_pred_pf.flatten()
        r2_mean = r2_score(y, yhat) if np.isfinite(y).all() and np.isfinite(yhat).all() else np.nan
        r2_pf = r2_score(y, yhat_pf) if np.isfinite(yhat_pf).all() else np.nan
        # Null should not explain force
        yhat_pk = (TF @ P_pk).flatten()
        r2_null = r2_score(y, yhat_pk) if np.isfinite(yhat_pk).all() else np.nan
        # Orthogonality over time: average dot product per frame
        dots = np.sum(P_pf * P_pk, axis=0)
        dot_pf_pk.append(float(np.mean(np.abs(dots))))
        r2_force_mean.append(float(r2_mean))
        r2_force_null.append(float(r2_null))
    return {
        "r2_force_TF_P_mean": float(np.nanmean(r2_force_mean)),
        "r2_force_TF_PF": float(np.nanmean(r2_force_mean)),
        "r2_force_TF_PK": float(np.nanmean(r2_force_null)),
        "mean_abs_dot_PF_PK": float(np.nanmean(dot_pf_pk)),
    }


def stiffness_checks(perfile: List[Dict[str, np.ndarray]], stiff_dir: Path) -> Dict[str, float]:
    # Co-contraction indicator vs stiffness magnitude correlation
    r_list = []
    for item in perfile:
        stem = item["stem"]
        stiff_csv = stiff_dir / f"{stem}_paper_profile.csv"
        if not stiff_csv.exists():
            continue
        try:
            dfk = pd.read_csv(stiff_csv)
            # Sum of all joint stiffness as a scalar proxy
            k_cols = [c for c in dfk.columns if c.endswith(("_k1","_k2","_k3"))]
            if not k_cols:
                continue
            Ksum = dfk[k_cols].sum(axis=1).to_numpy(dtype=float)
            # Compute EMG again to align length
            N = len(dfk)
            P = item["P"][:, :N]
            # Build PF/PK using TF from global fit? We approximate by recomputing here from the same run.
            # For a local proxy, use variance split: use first principal components as PF? Not robust.
            # Simpler: use PK magnitude as co-contraction indicator supplied externally? Fallback to ||P|| if missing.
            P_norm = np.linalg.norm(P, axis=0)
            # Spearman/Pearson correlation
            if len(P_norm) == len(Ksum):
                r = np.corrcoef(P_norm, Ksum)[0,1]
                r_list.append(float(r))
        except Exception:
            continue
    return {
        "corr_emg_norm_vs_stiffness_sum": float(np.nanmean(r_list)) if r_list else float("nan")
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", type=Path, required=True)
    ap.add_argument("--stiffness-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--max-files", type=int, default=12)
    args = ap.parse_args()

    logs = sorted(args.log_dir.glob("*.csv"))
    logs = [p for p in logs if not p.name.endswith("_paper_profile.csv")]
    if args.max_files > 0:
        logs = logs[: args.max_files]

    P_global, F_global, perfile = collect_emg_force(logs)
    # Identify TF as in pipeline
    TF = compute_tf(P_global, F_global)

    # Algebraic checks
    algebra = projector_checks(TF)

    # Correlation checks
    corr = correlation_checks(TF, perfile)

    # Stiffness sanity checks (proxy)
    stiff = stiffness_checks(perfile, args.stiffness_dir)

    report = {
        "TF_shape": list(TF.shape),
        "algebraic_checks": algebra,
        "correlation_checks": corr,
        "stiffness_checks": stiff,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "pf_pk_k_report.json").open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    # Simple plot for first file
    if perfile:
        item = perfile[0]
        time = item["time"]
        P = item["P"]
        F = item["F"].flatten()
        TF_pinv = np.linalg.pinv(TF)
        PF = TF_pinv @ TF
        PK = np.eye(P.shape[0]) - PF
        F_pred = (TF @ P).flatten()
        F_pred_pf = (TF @ (PF @ P)).flatten()
        # Plot
        fig, ax = plt.subplots(1,1, figsize=(10,3))
        t = np.arange(len(F))
        ax.plot(t, F, label="|F_n| measured", color="black", linewidth=1.2)
        ax.plot(t, F_pred, label="TF @ P", alpha=0.8)
        ax.plot(t, F_pred_pf, label="TF @ PF(P)", alpha=0.8)
        ax.set_title(f"Force prediction check: {item['stem']}")
        ax.legend(frameon=False, fontsize=8)
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(args.out_dir / "force_fit_example.png", dpi=140)
        plt.close(fig)

    print("[ok] PF/PK/K validation report:", args.out_dir / "pf_pk_k_report.json")


if __name__ == "__main__":
    main()
