#!/usr/bin/env python3
"""
Parametric intent→stiffness remapping.

Pipeline:
1. Load previously normalized stiffness profiles + EMG (same aggregation logic as monotonic mapping).
2. Reconstruct intent signal using existing NNLS weights (from monotonic mapping JSON) or refit if not provided.
3. Derive per-finger baseline target curve via isotonic regression on (intent, k_norm_scaled).
4. Estimate K_min/K_max automatically using robust percentiles (p5, p95) unless user overrides.
5. Fit candidate parametric functions f(x;θ) in [0,1] domain minimizing MAE to isotonic target.
   Functions (each naturally monotonic increasing):
     - linear: f(x)=x
     - power: f(x)=x**alpha, alpha>0
     - exp: f(x)=1 - exp(-beta*x), beta>0
     - logistic: f(x)= (1/(1+exp(-a*(x-b))) - offset) / scale  (normalized to [0,1])
     - piecewise: simplified piecewise linear through reduced breakpoints (uniform quantiles of isotonic curve)
6. Rescale function output to physical stiffness range [K_min, K_max] per finger.
7. Export JSON summary (parameters, metrics) and CSV sampled curves.

Usage examples:
  python3 parametric_intent_remap.py \
      --input-dir outputs/stiffness_profiles_global_tk_normalized \
      --mapping-json outputs/emg_monotonic_mapping_full.json \
      --functions linear,power,exp,logistic,piecewise \
      --out-dir outputs/parametric_remap

Optional overrides:
  --kmin-thumb 0.8 --kmax-thumb 5.2  (and similarly for index,middle)

Outputs:
  JSON: parametric_remap_summary.json
  CSV:  parametric_remap_samples.csv  (intent,x,finger,function,value_norm,value_scaled)
  PNG:  parametric_fit_[finger].png   (isotonic reference vs best functions)
"""

import argparse
import json
import math
import os
import glob
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _percentile_range(values: np.ndarray, p_low=5.0, p_high=95.0) -> Tuple[float, float]:
    return float(np.percentile(values, p_low)), float(np.percentile(values, p_high))


def _isotonic(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Pool Adjacent Violators (PAV) isotonic regression (increasing)."""
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]
    # Blocks initialization
    y_out = y_sorted.copy().astype(float)
    n = len(y_out)
    block_start = list(range(n))
    block_end = list(range(n))
    # Use list for block means so we can delete/merge easily
    block_mean = list(y_out.copy())
    m = n
    i = 0
    while i < m - 1:
        if block_mean[i] > block_mean[i + 1]:
            # Merge blocks
            total_len = block_end[i + 1] - block_start[i] + 1
            merged_mean = (
                np.sum(y_out[block_start[i] : block_end[i + 1] + 1]) / total_len
            )
            block_mean[i] = merged_mean
            block_end[i] = block_end[i + 1]
            # Remove block i+1
            del block_mean[i + 1]
            del block_start[i + 1]
            del block_end[i + 1]
            m -= 1
            # Move backward if needed
            if i > 0:
                i -= 1
        else:
            i += 1
    # Expand means back to points
    y_iso = np.zeros_like(y_out)
    for bs, be, bm in zip(block_start, block_end, block_mean):
        y_iso[bs : be + 1] = bm
    # Reorder to original x ordering
    y_final = np.zeros_like(y_iso)
    y_final[order] = y_iso
    return y_final


def _grid_search_power(x, y_ref):
    best = None
    for alpha in np.linspace(0.5, 3.0, 51):
        y_pred = x ** alpha
        mae = np.mean(np.abs(y_pred - y_ref))
        if best is None or mae < best[0]:
            best = (mae, float(alpha), y_pred)
    return {"mae": best[0], "alpha": best[1], "y_pred": best[2].tolist()}


def _grid_search_exp(x, y_ref):
    best = None
    for beta in np.linspace(0.5, 5.0, 91):
        y_pred = 1.0 - np.exp(-beta * x)
        mae = np.mean(np.abs(y_pred - y_ref))
        if best is None or mae < best[0]:
            best = (mae, float(beta), y_pred)
    return {"mae": best[0], "beta": best[1], "y_pred": best[2].tolist()}


def _grid_search_logistic(x, y_ref):
    best = None
    for a in np.linspace(2.0, 12.0, 51):
        for b in np.linspace(0.3, 0.7, 41):
            raw = 1.0 / (1.0 + np.exp(-a * (x - b)))
            # Normalize to [0,1]
            raw_min = raw.min()
            raw_max = raw.max()
            if raw_max - raw_min < 1e-6:
                continue
            y_pred = (raw - raw_min) / (raw_max - raw_min)
            mae = np.mean(np.abs(y_pred - y_ref))
            if best is None or mae < best[0]:
                best = (mae, float(a), float(b), y_pred)
    if best is None:
        return {"mae": None}
    return {"mae": best[0], "a": best[1], "b": best[2], "y_pred": best[3].tolist()}


def _piecewise_from_isotonic(x, y_iso, n_points=6):
    """Reduce isotonic curve to piecewise-linear with fixed quantile breakpoints."""
    if n_points < 2:
        raise ValueError("n_points must be >=2")
    qs = np.linspace(0, 1, n_points)
    xp = np.quantile(x, qs)
    yp = []
    for xv in xp:
        # local average around each xv
        mask = (x >= xv - 1e-6) & (x <= xv + 1e-6)
        if mask.any():
            yp.append(float(np.mean(y_iso[mask])))
        else:
            # interpolate
            yp.append(float(np.interp(xv, x, y_iso)))
    xp = xp.tolist()
    # Evaluate piecewise on original x
    y_pred = np.interp(x, xp, yp)
    mae = np.mean(np.abs(y_pred - y_iso))
    return {"mae": mae, "xp": xp, "yp": yp, "y_pred": y_pred.tolist()}


def _compute_metrics(y_ref, y_pred):
    y_ref_np = np.asarray(y_ref)
    y_pred_np = np.asarray(y_pred)
    mae = np.mean(np.abs(y_pred_np - y_ref_np))
    rmse = math.sqrt(np.mean((y_pred_np - y_ref_np) ** 2))
    # Spearman
    rank_ref = np.argsort(np.argsort(y_ref_np))
    rank_pred = np.argsort(np.argsort(y_pred_np))
    spearman = np.corrcoef(rank_ref, rank_pred)[0, 1]
    return {"mae": float(mae), "rmse": float(rmse), "spearman": float(spearman)}


def _load_nnls_weights(mapping_json_path: str) -> np.ndarray:
    if not os.path.exists(mapping_json_path):
        return None
    with open(mapping_json_path, "r") as f:
        js = json.load(f)
    if "nnls_weights" in js:
        return np.array(js["nnls_weights"], dtype=float)
    return None


def _refit_nnls(emg_mat: np.ndarray, target: np.ndarray) -> np.ndarray:
    # Simple NNLS via scipy if available, else custom coordinate descent.
    try:
        from scipy.optimize import nnls
        w, _ = nnls(emg_mat, target)
        return w
    except Exception:
        # Fallback: projected gradient descent
        w = np.zeros(emg_mat.shape[1])
        lr = 1e-2
        for _ in range(2000):
            grad = -2 * emg_mat.T @ (target - emg_mat @ w)
            w -= lr * grad
            w[w < 0] = 0
        return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True, help="Directory with normalized stiffness & EMG CSV logs")
    ap.add_argument("--mapping-json", help="Existing monotonic mapping JSON with nnls_weights")
    ap.add_argument("--functions", default="linear,power,exp,logistic,piecewise", help="Comma list of functions to fit")
    ap.add_argument("--out-dir", required=True, help="Output directory")
    ap.add_argument("--kmin-thumb", type=float)
    ap.add_argument("--kmax-thumb", type=float)
    ap.add_argument("--kmin-index", type=float)
    ap.add_argument("--kmax-index", type=float)
    ap.add_argument("--kmin-middle", type=float)
    ap.add_argument("--kmax-middle", type=float)
    ap.add_argument("--sample-n", type=int, default=200, help="Samples for curve export")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Aggregate CSVs (expect naming pattern) from input-dir
    csv_paths = sorted(glob.glob(os.path.join(args.input_dir, "*.csv")))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in {args.input_dir}")
    dfs = []
    for p in csv_paths:
        try:
            df = pd.read_csv(p)
            # Basic column checks (flexible: expect emg channels + k_norm columns)
            if not any(c.startswith("emg_ch") for c in df.columns):
                continue
            dfs.append(df)
        except Exception:
            continue
    if not dfs:
        raise RuntimeError("No valid EMG/stiffness CSVs aggregated")
    data = pd.concat(dfs, ignore_index=True)

    # Identify EMG channels
    emg_cols = [c for c in data.columns if c.startswith("emg_ch")]
    if not emg_cols:
        raise RuntimeError("No EMG channels detected")

    # Finger stiffness norm columns (assumed naming)
    finger_cols = [c for c in data.columns if c.endswith("_k_norm01")]
    if not finger_cols:
        # Fallback: attempt raw k_norm columns
        finger_cols = [c for c in data.columns if c.endswith("_k_norm")]
    # If still empty, construct k_norm on-the-fly from k1,k2,k3 components (e.g., th_k1, th_k2, th_k3)
    if not finger_cols:
        # Detect finger prefixes by pattern *_k1,*_k2,*_k3 triples
        prefixes = []
        for col in data.columns:
            if col.endswith('_k1'):
                pref = col[:-3]
                if all((pref + suf) in data.columns for suf in ['_k1','_k2','_k3']):
                    prefixes.append(pref)
        prefixes = sorted(set(prefixes))
        if not prefixes:
            # Attempt fallback: force sensor triples s1_fx,s1_fy,s1_fz etc -> pseudo stiffness norms
            sensor_prefixes = []
            for col in data.columns:
                if col.endswith('_fx'):
                    base = col[:-3]  # remove _fx
                    if all(f"{base}{axis}" in data.columns for axis in ['_fx','_fy','_fz']):
                        sensor_prefixes.append(base)
            sensor_prefixes = sorted(set(sensor_prefixes))
            # Expect at least 1 triple
            if not sensor_prefixes:
                raise RuntimeError("No stiffness component triples or force sensor triples found; cannot build norms")
            # Map first three sensor triples to fingers (assumed order: s1->th, s2->if, s3->mf). If fewer than 3, map what exists.
            mapping_order = ['th','if','mf']
            constructed_cols = []
            for idx, sp in enumerate(sensor_prefixes[:3]):
                fx = data[f"{sp}_fx"].to_numpy(dtype=float)
                fy = data[f"{sp}_fy"].to_numpy(dtype=float)
                fz = data[f"{sp}_fz"].to_numpy(dtype=float)
                kn = np.sqrt(fx**2 + fy**2 + fz**2)
                finger_tag = mapping_order[idx]
                col_name = f"{finger_tag}_k_norm"
                data[col_name] = kn
                constructed_cols.append(col_name)
            finger_cols = constructed_cols
            prefixes = []  # skip further processing
        if not prefixes and not finger_cols:
            raise RuntimeError("Unable to derive finger norm columns")
        # Compute k_norm columns from component triples only if prefixes were found
        if prefixes:
            constructed_cols = []
            for pref in prefixes:
                k1 = data[f"{pref}_k1"].to_numpy(dtype=float)
                k2 = data[f"{pref}_k2"].to_numpy(dtype=float)
                k3 = data[f"{pref}_k3"].to_numpy(dtype=float)
                kn = np.sqrt(k1**2 + k2**2 + k3**2)
                col_name = f"{pref}_k_norm"
                data[col_name] = kn
                constructed_cols.append(col_name)
            # If sensor-based norms existed, append; else replace
            if finger_cols:
                finger_cols.extend(constructed_cols)
            else:
                finger_cols = constructed_cols

    # Build intent via NNLS weights (global across channels)
    emg_mat = data[emg_cols].to_numpy(dtype=float)
    # Coarse target for NNLS weight fitting:
    # Prefer mean of k_norm; if not available (should be now), fallback to force magnitude sensors s*_f* pattern
    if finger_cols:
        coarse_target = data[finger_cols].mean(axis=1).to_numpy(dtype=float)
    else:
        force_sets = []
        # Identify sN_fx, sN_fy, sN_fz groups
        for col in data.columns:
            if col.endswith('_fx'):
                base = col[:-3]
                if all(f"{base}{axis}" in data.columns for axis in ['_fx','_fy','_fz']):
                    fx = data[f"{base}_fx"].to_numpy(float)
                    fy = data[f"{base}_fy"].to_numpy(float)
                    fz = data[f"{base}_fz"].to_numpy(float)
                    force_sets.append(np.sqrt(fx**2 + fy**2 + fz**2))
        if not force_sets:
            raise RuntimeError("No finger norms or force triples found for coarse target")
        coarse_target = np.mean(force_sets, axis=0)

    weights = None
    if args.mapping_json:
        weights = _load_nnls_weights(args.mapping_json)
    if weights is None:
        weights = _refit_nnls(emg_mat, coarse_target)

    intent_raw = emg_mat @ weights
    intent_min, intent_max = np.min(intent_raw), np.max(intent_raw)
    if intent_max - intent_min < 1e-9:
        raise RuntimeError("Intent signal has zero variance")
    intent01 = (intent_raw - intent_min) / (intent_max - intent_min)

    # Prepare output containers
    functions_requested = [f.strip() for f in args.functions.split(",") if f.strip()]
    summary = {
        "weights": weights.tolist(),
        "intent_min": float(intent_min),
        "intent_max": float(intent_max),
        "functions": functions_requested,
        "fingers": {},
    }

    sample_x = np.linspace(0, 1, args.sample_n)
    samples_rows = []

    # Debug: ensure finger_cols populated
    if not finger_cols:
        print("[parametric_intent_remap][warn] finger_cols empty after all fallbacks")
    else:
        print(f"[parametric_intent_remap] finger_cols detected: {finger_cols}")

    for fcol in finger_cols:
        finger_name = fcol.replace("_k_norm01", "").replace("_k_norm", "")
        y_raw = data[fcol].to_numpy(dtype=float)
        # If column not already scaled 0-1, rescale
        if y_raw.min() < -1e-6 or y_raw.max() > 1 + 1e-6:
            yr_min, yr_max = y_raw.min(), y_raw.max()
            y_norm = (y_raw - yr_min) / (yr_max - yr_min + 1e-12)
        else:
            y_norm = y_raw
        # Isotonic baseline
        y_iso = _isotonic(intent01, y_norm)
        # Auto K range if not provided
        alias_map = {"th": "thumb", "if": "index", "mf": "middle"}
        kmin_override = getattr(args, f"kmin_{finger_name}", None)
        kmax_override = getattr(args, f"kmax_{finger_name}", None)
        if kmin_override is None and finger_name in alias_map:
            kmin_override = getattr(args, f"kmin_{alias_map[finger_name]}", None)
        if kmax_override is None and finger_name in alias_map:
            kmax_override = getattr(args, f"kmax_{alias_map[finger_name]}", None)
        if kmin_override is not None and kmax_override is not None:
            k_min = float(kmin_override)
            k_max = float(kmax_override)
        else:
            k_min, k_max = _percentile_range(y_raw, 5.0, 95.0)
        if k_max - k_min < 1e-9:
            k_min, k_max = float(np.min(y_raw)), float(np.max(y_raw))

        finger_entry = {
            "k_min": k_min,
            "k_max": k_max,
            "isotonic_mae": float(np.mean(np.abs(y_iso - y_norm))),
            "parametric": {},
        }

        # Fit requested functions
        if "linear" in functions_requested:
            y_pred_lin = intent01.copy()  # already 0-1
            metrics_lin = _compute_metrics(y_iso, y_pred_lin)
            finger_entry["parametric"]["linear"] = {"metrics": metrics_lin}

        if "power" in functions_requested:
            res = _grid_search_power(intent01, y_iso)
            metrics_pow = _compute_metrics(y_iso, res["y_pred"])
            finger_entry["parametric"]["power"] = {"alpha": res["alpha"], "metrics": metrics_pow}

        if "exp" in functions_requested:
            res = _grid_search_exp(intent01, y_iso)
            metrics_exp = _compute_metrics(y_iso, res["y_pred"])
            finger_entry["parametric"]["exp"] = {"beta": res["beta"], "metrics": metrics_exp}

        if "logistic" in functions_requested:
            res = _grid_search_logistic(intent01, y_iso)
            if res.get("mae") is not None:
                metrics_log = _compute_metrics(y_iso, res["y_pred"])
                finger_entry["parametric"]["logistic"] = {"a": res["a"], "b": res["b"], "metrics": metrics_log}

        if "piecewise" in functions_requested:
            res = _piecewise_from_isotonic(intent01, y_iso, n_points=6)
            metrics_pw = _compute_metrics(y_iso, res["y_pred"])
            finger_entry["parametric"]["piecewise"] = {"xp": res["xp"], "yp": res["yp"], "metrics": metrics_pw}

        # Pick best by MAE
        best_name = None
        best_mae = None
        for name, entry in finger_entry["parametric"].items():
            mae = entry["metrics"]["mae"]
            if best_name is None or mae < best_mae:
                best_name = name
                best_mae = mae
        finger_entry["best_function"] = best_name
        finger_entry["best_mae"] = best_mae
        summary["fingers"][finger_name] = finger_entry

        # Generate sampled curves for each function, scaled to [k_min,k_max]
        for name, entry in finger_entry["parametric"].items():
            # Reconstruct predicted curve over sample_x
            if name == "linear":
                y_curve = sample_x
            elif name == "power":
                y_curve = sample_x ** entry["alpha"]
            elif name == "exp":
                y_curve = 1.0 - np.exp(-entry["beta"] * sample_x)
            elif name == "logistic":
                a = entry["a"]
                b = entry["b"]
                raw = 1.0 / (1.0 + np.exp(-a * (sample_x - b)))
                raw_min = raw.min()
                raw_max = raw.max()
                y_curve = (raw - raw_min) / (raw_max - raw_min + 1e-12)
            elif name == "piecewise":
                y_curve = np.interp(sample_x, entry["xp"], entry["yp"])  # already 0-1-ish
                # Ensure bounds
                y_curve = np.clip(y_curve, 0.0, 1.0)
            else:
                continue
            y_scaled = k_min + (k_max - k_min) * y_curve
            for xv, yn, ys in zip(sample_x, y_curve, y_scaled):
                samples_rows.append(
                    {
                        "finger": finger_name,
                        "function": name,
                        "intent": float(xv),
                        "value_norm": float(yn),
                        "value_scaled": float(ys),
                    }
                )

        # Plot isotonic vs best few functions
        plt.figure(figsize=(6, 4))
        plt.plot(intent01, y_iso, label="isotonic", linewidth=2)
        colors = {
            "linear": "#1f77b4",
            "power": "#ff7f0e",
            "exp": "#2ca02c",
            "logistic": "#d62728",
            "piecewise": "#9467bd",
        }
        for name in finger_entry["parametric"].keys():
            # Re-evaluate predictions on intent01 domain
            if name == "linear":
                yp = intent01
            elif name == "power":
                yp = intent01 ** finger_entry["parametric"][name]["alpha"]
            elif name == "exp":
                yp = 1.0 - np.exp(-finger_entry["parametric"][name]["beta"] * intent01)
            elif name == "logistic":
                a = finger_entry["parametric"][name]["a"]
                b = finger_entry["parametric"][name]["b"]
                raw = 1.0 / (1.0 + np.exp(-a * (intent01 - b)))
                raw = (raw - raw.min()) / (raw.max() - raw.min() + 1e-12)
                yp = raw
            elif name == "piecewise":
                xp = finger_entry["parametric"][name]["xp"]
                yp_list = finger_entry["parametric"][name]["yp"]
                yp = np.interp(intent01, xp, yp_list)
            else:
                continue
            plt.plot(intent01, yp, label=name, alpha=0.8, color=colors.get(name))
        plt.title(f"Parametric Fits - {finger_name}")
        plt.xlabel("intent01")
        plt.ylabel("stiffness_norm (0-1)")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, f"parametric_fit_{finger_name}.png"))
        plt.close()

    # Write summary + samples
    with open(os.path.join(args.out_dir, "parametric_remap_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    samples_df = pd.DataFrame(samples_rows)
    samples_df.to_csv(os.path.join(args.out_dir, "parametric_remap_samples.csv"), index=False)

    print("[parametric_intent_remap] Completed. Summary & samples written.")


if __name__ == "__main__":
    main()
