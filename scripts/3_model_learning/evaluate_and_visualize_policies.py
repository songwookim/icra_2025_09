#!/usr/bin/env python3
"""Evaluate trained stiffness policies (unified + per-finger) and visualize predictions.

Usage example:
  python3 evaluate_and_visualize_policies.py \
      --unified-artifact-dir outputs/models/policy_learning_global_tk_unified/artifacts/20251119_163418 \
      --per-finger-artifact-dir outputs/models/policy_learning_global_tk_per_finger/artifacts/20251119_205940 \
      --log-dir outputs/logs/success \
      --stiffness-dir outputs/analysis/stiffness_profiles_global_tk \
      --out-dir outputs/analysis/policy_eval/global_tk

The script relies on manifest.json + scalers.pkl inside artifact directories produced by
run_stiffness_policy_benchmarks.py. It reconstructs the test trajectories listed in the manifest
for fair comparison (no need to re-split). For per-finger mode it loads each finger sub-manifest.

Metrics: RMSE, MAE, R2, NLL (where available: GMM/GMR/MDN/GP). Sequence models (diffusion_t, lstm_gmm)
are evaluated only at their valid indices (final step per window) consistent with training script logic.

Visualization: For each model a 9D stiffness plot (3 fingers × 3 joints) overlaying target vs prediction.
Per-finger models get separate plots per finger. Plots saved under out-dir.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # type: ignore

# Import model classes from training script (they reside in same directory)
from run_stiffness_policy_benchmarks import (
    GMMConditional,
    BehaviorCloningBaseline,
    IBCBaseline,
    DiffusionPolicyBaseline,
    LSTMGMMBaseline,
    MDNBaseline,
    GPBaseline,
)


OBS_COLUMNS_DEFAULT = [
    "s1_fx","s1_fy","s1_fz","s2_fx","s2_fy","s2_fz","s3_fx","s3_fy","s3_fz",
    "deform_ecc",
    "ee_if_px","ee_if_py","ee_if_pz","ee_mf_px","ee_mf_py","ee_mf_pz","ee_th_px","ee_th_py","ee_th_pz",
]
ACTION_COLUMNS_DEFAULT = [
    "th_k1","th_k2","th_k3","if_k1","if_k2","if_k3","mf_k1","mf_k2","mf_k3",
]

FINGER_ACTION_MAP = {
    "th": ["th_k1","th_k2","th_k3"],
    "if": ["if_k1","if_k2","if_k3"],
    "mf": ["mf_k1","mf_k2","mf_k3"],
}


@dataclass
class Trajectory:
    name: str
    observations: np.ndarray
    actions: np.ndarray


def load_pair(log_csv: Path, stiff_csv: Path, obs_cols: Sequence[str], act_cols: Sequence[str]) -> Optional[Trajectory]:
    try:
        raw = pd.read_csv(log_csv)
        stiff = pd.read_csv(stiff_csv)
    except Exception:
        return None
    rows = min(len(raw), len(stiff))
    if rows < 5:
        return None
    raw = raw.iloc[:rows].reset_index(drop=True)
    stiff = stiff.iloc[:rows].reset_index(drop=True)
    obs_parts = []
    for col in obs_cols:
        if col in stiff.columns:
            obs_parts.append(stiff[col].to_numpy(dtype=float).reshape(-1,1))
        elif col in raw.columns:
            obs_parts.append(raw[col].to_numpy(dtype=float).reshape(-1,1))
        else:
            return None
    obs = np.hstack(obs_parts)
    if not all(c in stiff.columns for c in act_cols):
        return None
    act = stiff[act_cols].to_numpy(dtype=float)
    mask = np.isfinite(obs).all(axis=1) & np.isfinite(act).all(axis=1)
    obs = obs[mask]; act = act[mask]
    if obs.shape[0] < 5:
        return None
    return Trajectory(name=log_csv.stem, observations=obs, actions=act)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def build_sequence(trajs: Sequence[Trajectory], window: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    obs_seq: List[np.ndarray] = []
    act_seq: List[np.ndarray] = []
    act_raw: List[np.ndarray] = []
    indices: List[int] = []
    offset = 0
    for t in trajs:
        length = t.observations.shape[0]
        for idx in range(window-1, length):
            obs_seq.append(t.observations[idx-window+1:idx+1])
            act_seq.append(t.actions[idx])
            act_raw.append(t.actions[idx])
            indices.append(offset+idx)
        offset += length
    if not obs_seq:
        return np.zeros((0,window,trajs[0].observations.shape[1])), np.zeros((0,trajs[0].actions.shape[1])), np.zeros((0,trajs[0].actions.shape[1])), np.zeros((0,),dtype=int)
    return np.stack(obs_seq), np.stack(act_seq), np.stack(act_raw), np.asarray(indices)


def load_unified_artifacts(path: Path) -> Dict[str, Any]:
    manifest_path = path / "manifest.json"
    scalers_path = path / "scalers.pkl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    if not scalers_path.exists():
        raise FileNotFoundError(f"Missing scalers: {scalers_path}")
    with manifest_path.open("r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    with scalers_path.open("rb") as fh:
        scalers = pickle.load(fh)
    return {"manifest": manifest, "scalers": scalers}


def evaluate_unified(artifact_dir: Path, log_dir: Path, stiff_dir: Path, out_dir: Path) -> Dict[str, Dict[str, float]]:
    data = load_unified_artifacts(artifact_dir)
    manifest = data["manifest"]
    scalers = data["scalers"]
    obs_scaler = scalers["obs_scaler"]
    act_scaler = scalers["act_scaler"]
    obs_cols = manifest.get("obs_columns", OBS_COLUMNS_DEFAULT)
    act_cols = manifest.get("action_columns", ACTION_COLUMNS_DEFAULT)
    test_names = manifest.get("test_trajectories", [])
    trajectories: List[Trajectory] = []
    for name in test_names:
        log_csv = log_dir / f"{name}.csv"
        stiff_csv = stiff_dir / f"{name}_paper_profile.csv"
        traj = load_pair(log_csv, stiff_csv, obs_cols, act_cols)
        if traj:
            trajectories.append(traj)
    if not trajectories:
        raise RuntimeError("No test trajectories reconstructed for unified evaluation.")
    obs = np.concatenate([t.observations for t in trajectories], axis=0)
    act = np.concatenate([t.actions for t in trajectories], axis=0)
    obs_s = obs_scaler.transform(obs)
    act_s = act_scaler.transform(act)

    results: Dict[str, Dict[str, float]] = {}
    # Store predictions for combined comparison plot
    unified_predictions: Dict[str, np.ndarray] = {}

    # Helper to save plot
    def plot_actions(pred: np.ndarray, model_name: str):
        fig, axes = plt.subplots(3,3, figsize=(10,8), sharex=True)
        for i, joint in enumerate(act_cols):
            r = i//3; c=i%3
            axes[r][c].plot(act[:,i], label="GT (from demo)", linewidth=1.2, color="black")
            axes[r][c].plot(pred[:,i], label="prediction", linewidth=0.9, alpha=0.8)
            axes[r][c].set_title(joint, fontsize=9)
            axes[r][c].set_ylabel("K [N/m]", fontsize=8)
        axes[0][0].legend(frameon=False, fontsize=8)
        axes[2][1].set_xlabel("Time step", fontsize=8)
        fig.suptitle(f"Unified {model_name}: Predicted vs Ground-Truth Stiffness", fontsize=11)
        fig.tight_layout()
        fig.savefig(out_dir / f"unified_{model_name}_comparison.png", dpi=130)
        plt.close(fig)

    # Load models
    models_info: Dict[str, Any] = manifest.get("models", {})
    for key, meta in models_info.items():
        kind = meta.get("kind", key)
        path_rel = meta.get("path")
        if not path_rel:
            continue
        model_path = artifact_dir / path_rel
        if not model_path.exists():
            continue
        try:
            if kind in {"gmr", "gmm"} and model_path.suffix == ".pkl":
                with model_path.open("rb") as fh:
                    gmm_model = pickle.load(fh)
                mode = "mean" if kind == "gmr" else "sample"
                pred_scaled = gmm_model.predict(obs_s, mode=mode, n_samples=meta.get("n_samples", 8))
                pred = act_scaler.inverse_transform(pred_scaled)
                m = compute_metrics(act, pred)
                # NLL if available
                try:
                    m["nll"] = gmm_model.nll(obs_s, act_s)
                except Exception:
                    m["nll"] = float("nan")
                results[key] = m
                unified_predictions[key] = pred
                plot_actions(pred, key)
            elif kind == "bc":
                import torch
                checkpoint = torch.load(model_path, map_location="cpu")
                cfg = checkpoint.get("config", {})
                bc = BehaviorCloningBaseline(obs_dim=obs_s.shape[1], act_dim=act_s.shape[1], hidden_dim=cfg.get("hidden_dim",256), depth=cfg.get("depth",3), lr=1e-3, batch_size=256, epochs=0, weight_decay=1e-4, seed=0, log_name="bc")
                bc.model.load_state_dict(checkpoint["state_dict"])  # type: ignore
                pred_scaled = bc.predict(obs_s)
                pred = act_scaler.inverse_transform(pred_scaled)
                m = compute_metrics(act, pred); m["nll"] = float("nan")
                results[key] = m
                unified_predictions[key] = pred
                plot_actions(pred, key)
            elif kind == "ibc":
                import torch
                checkpoint = torch.load(model_path, map_location="cpu")
                cfg = checkpoint.get("config", {})
                ibc = IBCBaseline(obs_dim=obs_s.shape[1], act_dim=act_s.shape[1], hidden_dim=cfg.get("hidden_dim",256), depth=cfg.get("depth",3), lr=1e-3, batch_size=cfg.get("batch_size",256), epochs=0, noise_std=cfg.get("noise_std",0.5), langevin_steps=cfg.get("langevin_steps",30), step_size=cfg.get("step_size",1e-2), seed=0, log_name="ibc")
                ibc.model.load_state_dict(checkpoint["state_dict"])  # type: ignore
                pred_scaled = ibc.predict(obs_s)
                pred = act_scaler.inverse_transform(pred_scaled)
                m = compute_metrics(act, pred); m["nll"] = float("nan")
                results[key] = m
                unified_predictions[key] = pred
                plot_actions(pred, key)
            elif kind == "mdn":
                import torch
                checkpoint = torch.load(model_path, map_location="cpu")
                cfg = checkpoint.get("config", {})
                mdn = MDNBaseline(obs_dim=obs_s.shape[1], act_dim=act_s.shape[1], hidden_units=cfg.get("hidden_units",[128,128,64]), n_components=cfg.get("n_components",5), covariance_type=cfg.get("covariance_type","diag"), activation=cfg.get("activation","relu"), dropout=cfg.get("dropout",0.1), learning_rate=1e-3, weight_decay=1e-4, mixture_reg=0.01, covariance_floor=1e-4)
                mdn.model.load_state_dict(checkpoint["model_state"])  # type: ignore
                pred_scaled = mdn.predict(obs_s, mode="mean")
                pred = act_scaler.inverse_transform(pred_scaled)
                m = compute_metrics(act, pred)
                try:
                    m["nll"] = mdn.nll(obs_s, act_s)
                except Exception:
                    m["nll"] = float("nan")
                results[key] = m
                unified_predictions[key] = pred
                plot_actions(pred, key)
            elif kind == "diffusion":
                import torch
                checkpoint = torch.load(model_path, map_location="cpu")
                cfg = checkpoint.get("config", {})
                temporal = bool(cfg.get("temporal", False))
                model = DiffusionPolicyBaseline(obs_dim=cfg.get("obs_dim", obs_s.shape[1]), act_dim=cfg.get("act_dim", act_s.shape[1]), timesteps=cfg.get("timesteps",50), hidden_dim=cfg.get("hidden_dim",256), lr=1e-3, batch_size=256, epochs=0, seed=0, log_name="diffusion", temporal=temporal)
                model.model.load_state_dict(checkpoint["state_dict"])  # type: ignore
                if temporal:
                    window = int(cfg.get("seq_len", 1))
                    # Build sequence dataset
                    # Re-split trajectories flatten again for seq building
                    seq_obs, _, act_seq_raw, indices = build_sequence(trajectories, window)
                    if seq_obs.shape[0] == 0:
                        continue
                    seq_obs_s = obs_scaler.transform(seq_obs.reshape(-1, seq_obs.shape[-1])).reshape(seq_obs.shape)
                    pred_scaled = model.predict(seq_obs_s, n_samples=4, sampler=meta.get("sampler","ddpm"), eta=float(meta.get("eta",0.0)))
                    pred = act_scaler.inverse_transform(pred_scaled)
                    # Expand to full length alignment
                    full_pred = np.full_like(act, np.nan)
                    full_pred[indices] = pred
                    m = compute_metrics(act[indices], pred)
                    m["nll"] = float("nan")
                    results[key] = m
                    unified_predictions[key] = full_pred
                    plot_actions(full_pred, key)
                else:
                    pred_scaled = model.predict(obs_s, n_samples=4, sampler=meta.get("sampler","ddpm"), eta=float(meta.get("eta",0.0)))
                    pred = act_scaler.inverse_transform(pred_scaled)
                    m = compute_metrics(act, pred); m["nll"] = float("nan")
                    results[key] = m
                    unified_predictions[key] = pred
                    plot_actions(pred, key)
            elif kind == "lstm_gmm":
                import torch
                checkpoint = pickle.load(open(model_path, "rb")) if model_path.suffix == ".pkl" else torch.load(model_path, map_location="cpu")
                cfg = checkpoint.get("config", {})
                window = int(cfg.get("seq_len", 1))
                seq_obs, _, act_seq_raw, indices = build_sequence(trajectories, window)
                if seq_obs.shape[0] == 0:
                    continue
                seq_obs_s = obs_scaler.transform(seq_obs.reshape(-1, seq_obs.shape[-1])).reshape(seq_obs.shape)
                lstm = LSTMGMMBaseline(obs_dim=seq_obs.shape[-1], act_dim=act_s.shape[1], seq_len=window, n_components=cfg.get("n_components",5), hidden_dim=cfg.get("hidden_dim",256), n_layers=cfg.get("n_layers",1), lr=1e-3, batch_size=256, epochs=0, seed=0, log_name="lstm_gmm")
                lstm.model.load_state_dict(checkpoint["state_dict"])  # type: ignore
                pred_seq_scaled = lstm.predict(seq_obs_s, mode="mean", n_samples=8)
                pred_seq = act_scaler.inverse_transform(pred_seq_scaled)
                full_pred = np.full_like(act, np.nan); full_pred[indices] = pred_seq
                m = compute_metrics(act[indices], pred_seq); m["nll"] = float("nan")
                results[key] = m
                unified_predictions[key] = full_pred
                plot_actions(full_pred, key)
            elif kind == "gp":
                with model_path.open("rb") as fh:
                    gp_obj = pickle.load(fh)
                gp = gp_obj["model"]
                pred_gp, pred_std = gp.predict(obs, return_std=True)
                m = compute_metrics(act, pred_gp)
                try:
                    m["nll"] = gp.nll(obs, act)
                except Exception:
                    m["nll"] = float("nan")
                m["mean_uncertainty"] = float(np.mean(pred_std))
                results[key] = m
                unified_predictions[key] = pred_gp
                plot_actions(pred_gp, key)
        except Exception as exc:  # robust skip on individual model failure
            results[key] = {"rmse": float("nan"), "mae": float("nan"), "r2": float("nan"), "nll": float("nan"), "error": str(exc)}

    # Save summary
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "unified_metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    # Combined multi-model comparison plot (overlay all predictions)
    if unified_predictions:
        fig, axes = plt.subplots(3,3, figsize=(11,9), sharex=True)
        model_keys_sorted = sorted(unified_predictions.keys())
        for i, joint in enumerate(act_cols):
            r = i//3; c = i%3
            axes[r][c].plot(act[:, i], label="GT (demo)", linewidth=1.4, color="black", alpha=0.9)
            for mk in model_keys_sorted:
                pred_arr = unified_predictions[mk][:, i]
                # Mask NaNs for sequence/temporal expansions
                if np.isnan(pred_arr).any():
                    valid_mask = ~np.isnan(pred_arr)
                    axes[r][c].plot(np.where(valid_mask)[0], pred_arr[valid_mask], label=mk, linewidth=0.8, alpha=0.7)
                else:
                    axes[r][c].plot(pred_arr, label=mk, linewidth=0.8, alpha=0.7)
            axes[r][c].set_title(joint, fontsize=9)
            axes[r][c].set_ylabel("K [N/m]", fontsize=7)
        axes[2][1].set_xlabel("Time step", fontsize=8)
        # Single legend outside
        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=min(5, len(labels)), frameon=False, fontsize=8)
        fig.suptitle("Unified: All Models vs Ground-Truth Stiffness (from Demonstration)", fontsize=11)
        fig.tight_layout(rect=(0,0,1,0.94))
        fig.savefig(out_dir / "unified_all_models_comparison.png", dpi=140)
        plt.close(fig)
    return results


def evaluate_per_finger(artifact_dir: Path, log_dir: Path, stiff_dir: Path, out_dir: Path) -> Dict[str, Dict[str, Dict[str, float]]]:
    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    for finger in ["th","if","mf"]:
        finger_dir = artifact_dir / finger
        manifest_path = finger_dir / "manifest.json"
        scalers_path = finger_dir / "scalers.pkl"
        if not manifest_path.exists() or not scalers_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        scalers = pickle.load(open(scalers_path, "rb"))
        obs_scaler = scalers["obs_scaler"]; act_scaler = scalers["act_scaler"]
        test_names = manifest.get("test_trajectories", [])
        finger_act_cols = FINGER_ACTION_MAP[finger]
        # Derive per-finger observation columns strictly from scaler feature count
        n_features = int(getattr(obs_scaler, 'n_features_in_', 0))
        # Default expected pattern (7 features) per finger from training script FINGER_CONFIG
        if n_features == 7:
            if finger == 'th':
                finger_obs_cols = ["s1_fx","s1_fy","s1_fz","ee_th_px","ee_th_py","ee_th_pz","deform_ecc"]
            elif finger == 'if':
                finger_obs_cols = ["s2_fx","s2_fy","s2_fz","ee_if_px","ee_if_py","ee_if_pz","deform_ecc"]
            elif finger == 'mf':
                finger_obs_cols = ["s3_fx","s3_fy","s3_fz","ee_mf_px","ee_mf_py","ee_mf_pz","deform_ecc"]
        else:
            # Fallback: attempt to read from manifest if present
            finger_obs_cols = manifest.get("obs_columns", []) or manifest.get("finger_obs_columns", []) or []
            if not finger_obs_cols:
                finger_obs_cols = OBS_COLUMNS_DEFAULT[:n_features]
        trajs: List[Trajectory] = []
        for name in test_names:
            log_csv = log_dir / (name.replace(f"_{finger}", "") + ".csv") if not (log_dir / f"{name}.csv").exists() else log_dir / f"{name}.csv"
            stiff_csv = stiff_dir / (name.replace(f"_{finger}", "") + "_paper_profile.csv") if not (stiff_dir / f"{name}_paper_profile.csv").exists() else stiff_dir / f"{name}_paper_profile.csv"
            traj = load_pair(log_csv, stiff_csv, finger_obs_cols, finger_act_cols)
            if traj:
                trajs.append(traj)
        if not trajs:
            continue
        obs = np.concatenate([t.observations for t in trajs], axis=0)
        act = np.concatenate([t.actions for t in trajs], axis=0)
        obs_s = obs_scaler.transform(obs)
        act_s = act_scaler.transform(act)
        finger_results: Dict[str, Dict[str, float]] = {}
        finger_predictions: Dict[str, np.ndarray] = {}
        for key, meta in manifest.get("models", {}).items():
            kind = meta.get("kind", key)
            model_path = finger_dir / meta.get("path", "")
            if not model_path.exists():
                continue
            try:
                if kind == "bc":
                    import torch
                    ck = torch.load(model_path, map_location="cpu")
                    cfg = ck.get("config", {})
                    bc = BehaviorCloningBaseline(obs_dim=obs_s.shape[1], act_dim=act_s.shape[1], hidden_dim=cfg.get("hidden_dim",256), depth=cfg.get("depth",3), lr=1e-3, batch_size=256, epochs=0, weight_decay=1e-4, seed=0, log_name="bc")
                    bc.model.load_state_dict(ck["state_dict"])  # type: ignore
                    pred_scaled = bc.predict(obs_s); pred = act_scaler.inverse_transform(pred_scaled)
                    m = compute_metrics(act, pred); m["nll"] = float("nan")
                    finger_results[key] = m
                    finger_predictions[key] = pred
                elif kind == "diffusion":
                    import torch
                    ck = torch.load(model_path, map_location="cpu")
                    cfg = ck.get("config", {})
                    model = DiffusionPolicyBaseline(obs_dim=obs_s.shape[1], act_dim=act_s.shape[1], timesteps=cfg.get("timesteps",50), hidden_dim=cfg.get("hidden_dim",256), lr=1e-3, batch_size=256, epochs=0, seed=0, log_name="diffusion", temporal=False)
                    model.model.load_state_dict(ck["state_dict"])  # type: ignore
                    pred_scaled = model.predict(obs_s, n_samples=4, sampler="ddpm", eta=0.0)
                    pred = act_scaler.inverse_transform(pred_scaled)
                    m = compute_metrics(act, pred); m["nll"] = float("nan")
                    finger_results[key] = m
                    finger_predictions[key] = pred
                elif kind == "lstm_gmm":
                    import torch
                    ck = torch.load(model_path, map_location="cpu")
                    cfg = ck.get("config", {})
                    window = int(cfg.get("seq_len", 10))
                    seq_obs, _, act_seq_raw, indices = build_sequence(trajs, window)
                    if seq_obs.shape[0] == 0:
                        continue
                    seq_obs_s = obs_scaler.transform(seq_obs.reshape(-1, seq_obs.shape[-1])).reshape(seq_obs.shape)
                    lstm = LSTMGMMBaseline(obs_dim=seq_obs.shape[-1], act_dim=act_s.shape[1], seq_len=window, n_components=cfg.get("n_components",5), hidden_dim=cfg.get("hidden_dim",256), n_layers=cfg.get("n_layers",1), lr=1e-3, batch_size=256, epochs=0, seed=0, log_name="lstm_gmm")
                    lstm.model.load_state_dict(ck["state_dict"])  # type: ignore
                    pred_seq_scaled = lstm.predict(seq_obs_s, mode="mean", n_samples=8)
                    pred_seq = act_scaler.inverse_transform(pred_seq_scaled)
                    m = compute_metrics(act[indices], pred_seq); m["nll"] = float("nan")
                    finger_results[key] = m
                    # For sequence models we don't expand yet; store raw sequential predictions aligned to indices later for combined plot
                    full_pred_seq = np.full_like(act, np.nan); full_pred_seq[indices] = pred_seq
                    finger_predictions[key] = full_pred_seq
            except Exception as exc:
                finger_results[key] = {"rmse": float("nan"), "mae": float("nan"), "r2": float("nan"), "nll": float("nan"), "error": str(exc)}
        results[finger] = finger_results
        # Plot all models for this finger (individual files)
        plot_dir = out_dir / "per_finger_plots"; plot_dir.mkdir(parents=True, exist_ok=True)
        for model_key, model_meta in manifest.get("models", {}).items():
            if model_key not in finger_results:
                continue
            kind = model_meta.get("kind", model_key)
            model_path = finger_dir / model_meta.get("path", "")
            if not model_path.exists():
                continue
            try:
                import torch
                ck = torch.load(model_path, map_location="cpu")
                cfg = ck.get("config", {})
                if kind == "bc":
                    bc = BehaviorCloningBaseline(obs_dim=obs_s.shape[1], act_dim=act_s.shape[1], hidden_dim=cfg.get("hidden_dim",256), depth=cfg.get("depth",3), lr=1e-3, batch_size=256, epochs=0, weight_decay=1e-4, seed=0, log_name="bc")
                    bc.model.load_state_dict(ck["state_dict"])  # type: ignore
                    pred_scaled = bc.predict(obs_s); pred = act_scaler.inverse_transform(pred_scaled)
                    fig, axes = plt.subplots(1,3, figsize=(9,3))
                    for i, joint in enumerate(finger_act_cols):
                        axes[i].plot(act[:,i], label="GT (demo)", linewidth=1.2, color="black")
                        axes[i].plot(pred[:,i], label="prediction", linewidth=0.9, alpha=0.8)
                        axes[i].set_title(f"{finger}-{joint}", fontsize=10)
                        axes[i].set_ylabel("K [N/m]", fontsize=8)
                    axes[0].legend(frameon=False, fontsize=8)
                    axes[1].set_xlabel("Time step", fontsize=8)
                    fig.suptitle(f"Finger {finger.upper()} {model_key}: Predicted vs GT Stiffness", fontsize=11)
                    fig.tight_layout(); fig.savefig(plot_dir / f"{finger}_{model_key}_comparison.png", dpi=130); plt.close(fig)
                elif kind == "diffusion":
                    model = DiffusionPolicyBaseline(obs_dim=obs_s.shape[1], act_dim=act_s.shape[1], timesteps=cfg.get("timesteps",50), hidden_dim=cfg.get("hidden_dim",256), lr=1e-3, batch_size=256, epochs=0, seed=0, log_name="diffusion", temporal=False)
                    model.model.load_state_dict(ck["state_dict"])  # type: ignore
                    pred_scaled = model.predict(obs_s, n_samples=4, sampler=model_meta.get("sampler","ddpm"), eta=float(model_meta.get("eta",0.0)))
                    pred = act_scaler.inverse_transform(pred_scaled)
                    fig, axes = plt.subplots(1,3, figsize=(9,3))
                    for i, joint in enumerate(finger_act_cols):
                        axes[i].plot(act[:,i], label="GT (demo)", linewidth=1.2, color="black")
                        axes[i].plot(pred[:,i], label="prediction", linewidth=0.9, alpha=0.8)
                        axes[i].set_title(f"{finger}-{joint}", fontsize=10)
                        axes[i].set_ylabel("K [N/m]", fontsize=8)
                    axes[0].legend(frameon=False, fontsize=8)
                    axes[1].set_xlabel("Time step", fontsize=8)
                    fig.suptitle(f"Finger {finger.upper()} {model_key}: Predicted vs GT Stiffness", fontsize=11)
                    fig.tight_layout(); fig.savefig(plot_dir / f"{finger}_{model_key}_comparison.png", dpi=130); plt.close(fig)
                elif kind == "lstm_gmm":
                    window = int(cfg.get("seq_len",10))
                    seq_obs, _, act_seq_raw, indices = build_sequence(trajs, window)
                    if seq_obs.shape[0] > 0:
                        seq_obs_s = obs_scaler.transform(seq_obs.reshape(-1, seq_obs.shape[-1])).reshape(seq_obs.shape)
                        lstm = LSTMGMMBaseline(obs_dim=seq_obs.shape[-1], act_dim=act_s.shape[1], seq_len=window, n_components=cfg.get("n_components",5), hidden_dim=cfg.get("hidden_dim",256), n_layers=cfg.get("n_layers",1), lr=1e-3, batch_size=256, epochs=0, seed=0, log_name="lstm_gmm")
                        lstm.model.load_state_dict(ck["state_dict"])  # type: ignore
                        pred_seq_scaled = lstm.predict(seq_obs_s, mode="mean", n_samples=8)
                        pred_seq = act_scaler.inverse_transform(pred_seq_scaled)
                        full_pred = np.full_like(act, np.nan); full_pred[indices] = pred_seq
                        fig, axes = plt.subplots(1,3, figsize=(9,3))
                        for i, joint in enumerate(finger_act_cols):
                            axes[i].plot(act[:,i], label="GT (demo)", linewidth=1.2, color="black")
                            axes[i].plot(full_pred[:,i], label="prediction", linewidth=0.9, alpha=0.8)
                            axes[i].set_title(f"{finger}-{joint}", fontsize=10)
                            axes[i].set_ylabel("K [N/m]", fontsize=8)
                        axes[0].legend(frameon=False, fontsize=8)
                        axes[1].set_xlabel("Time step", fontsize=8)
                        fig.suptitle(f"Finger {finger.upper()} {model_key}: Predicted vs GT Stiffness", fontsize=11)
                        fig.tight_layout(); fig.savefig(plot_dir / f"{finger}_{model_key}_comparison.png", dpi=130); plt.close(fig)
            except Exception:
                pass  # skip failed visualization for this model
        # Combined multi-model overlay plot for this finger
        if finger_predictions:
            fig, axes = plt.subplots(1,3, figsize=(11,3.2), sharex=True)
            model_keys_sorted = sorted(finger_predictions.keys())
            for i, joint in enumerate(finger_act_cols):
                axes[i].plot(act[:, i], label="GT (demo)", linewidth=1.4, color="black", alpha=0.9)
                for mk in model_keys_sorted:
                    pred_arr = finger_predictions[mk][:, i]
                    if np.isnan(pred_arr).any():
                        valid_mask = ~np.isnan(pred_arr)
                        axes[i].plot(np.where(valid_mask)[0], pred_arr[valid_mask], label=mk, linewidth=0.9, alpha=0.7)
                    else:
                        axes[i].plot(pred_arr, label=mk, linewidth=0.9, alpha=0.7)
                axes[i].set_title(f"{finger}-{joint}", fontsize=10)
                axes[i].set_ylabel("K [N/m]", fontsize=8)
            axes[1].set_xlabel("Time step", fontsize=8)
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="upper center", ncol=min(5, len(labels)), frameon=False, fontsize=8)
            fig.suptitle(f"Finger {finger.upper()}: All Models vs Ground-Truth Stiffness", fontsize=11)
            fig.tight_layout(rect=(0,0,1,0.86))
            fig.savefig(plot_dir / f"{finger}_all_models_comparison.png", dpi=140)
            plt.close(fig)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "per_finger_metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    return results


def main():
    ap = argparse.ArgumentParser(description="Evaluate and visualize stiffness policies")
    ap.add_argument("--unified-artifact-dir", type=Path, required=True)
    ap.add_argument("--per-finger-artifact-dir", type=Path, default=None)
    ap.add_argument("--log-dir", type=Path, required=True)
    ap.add_argument("--stiffness-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[eval] unified artifacts: {args.unified_artifact_dir}")
    unified_results = evaluate_unified(args.unified_artifact_dir, args.log_dir, args.stiffness_dir, args.out_dir)
    print("[eval] unified metrics:")
    for k,v in unified_results.items():
        print(f"  {k}: rmse={v.get('rmse'):.4f} mae={v.get('mae'):.4f} r2={v.get('r2'):.4f}")
    if args.per_finger_artifact_dir:
        print(f"[eval] per-finger artifacts: {args.per_finger_artifact_dir}")
        pf_results = evaluate_per_finger(args.per_finger_artifact_dir, args.log_dir, args.stiffness_dir, args.out_dir)
        print("[eval] per-finger metrics:")
        for finger, models in pf_results.items():
            for k,v in models.items():
                print(f"  {finger}-{k}: rmse={v.get('rmse'):.4f} mae={v.get('mae'):.4f} r2={v.get('r2'):.4f}")
    print(f"[done] metrics & plots saved to {args.out_dir}")


if __name__ == "__main__":
    main()
