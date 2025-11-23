#!/usr/bin/env python3
"""Visualise saved stiffness policy predictions on a held-out demonstration.

Outputs
- Console: per-model RMSE/MAE/R² and per-finger RMSE/MAE
- Figure: grid plot of GT vs predictions for selected models and action dims
- Artifact discovery: supports both 'artifacts' and 'artifacts_global' roots
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import os
if os.environ.get("DISPLAY", "") == "":  # headless-safe backend
    import matplotlib
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:  # pragma: no cover - optional dependency
    import torch  # type: ignore[import]
except ImportError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]

import run_stiffness_policy_benchmarks as benchmarks

ACTION_COLUMNS = benchmarks.ACTION_COLUMNS
OBS_COLUMNS = benchmarks.OBS_COLUMNS
DEFAULT_STIFFNESS_DIR = benchmarks.DEFAULT_STIFFNESS_DIR
DEFAULT_OUTPUT_DIR = benchmarks.DEFAULT_OUTPUT_DIR

GMMConditional = benchmarks.GMMConditional
Trajectory = benchmarks.Trajectory
build_sequence_dataset = benchmarks.build_sequence_dataset
compute_metrics = benchmarks.compute_metrics
compute_offsets = benchmarks.compute_offsets
load_dataset = benchmarks.load_dataset
load_dataset_per_finger = benchmarks.load_dataset_per_finger
scale_trajectories = benchmarks.scale_trajectories

BehaviorCloningBaseline = getattr(benchmarks, "BehaviorCloningBaseline", None)
IBCBaseline = getattr(benchmarks, "IBCBaseline", None)
DiffusionPolicyBaseline = getattr(benchmarks, "DiffusionPolicyBaseline", None)
LSTMGMMBaseline = getattr(benchmarks, "LSTMGMMBaseline", None)
GPBaseline = getattr(benchmarks, "GPBaseline", None)
MDNBaseline = getattr(benchmarks, "MDNBaseline", None)

# Override DEFAULT_ARTIFACT_ROOT to point to actual artifact location
_PKG_ROOT = Path(__file__).resolve().parents[2]  # .../src/hri_falcon_robot_bridge
DEFAULT_ARTIFACT_ROOT_UNIFIED = _PKG_ROOT / "outputs" / "models" / "policy_learning_unified" / "artifacts"
DEFAULT_ARTIFACT_ROOT_PER_FINGER = _PKG_ROOT / "outputs" / "models" / "policy_learning_per_finger" / "artifacts"
DEFAULT_PLOT_DIR = DEFAULT_OUTPUT_DIR / "plots"

# Finger configuration for per-finger mode
FINGER_LIST = ["th", "if", "mf"]


def _normalise_finger_token(token: str) -> str:
    return token.strip().lower()


def _group_action_columns(action_columns: Sequence[str]) -> Dict[str, List[int]]:
    groups: Dict[str, List[int]] = {}
    for idx, col in enumerate(action_columns):
        prefix = _normalise_finger_token(col.split("_", 1)[0]) if "_" in col else _normalise_finger_token(col)
        groups.setdefault(prefix, []).append(idx)
    return groups


def _resolve_finger_selection(spec: Optional[str], available: Sequence[str]) -> List[str]:
    if not available:
        raise RuntimeError("No finger groups available for selection.")
    tokens = [] if spec is None else [t for t in spec.split(",") if t.strip()]
    if not tokens:
        return list(available)
    normalized = [_normalise_finger_token(tok) for tok in tokens]
    if "all" in normalized:
        return list(available)
    selection: List[str] = []
    missing: List[str] = []
    for token in normalized:
        if token in available and token not in selection:
            selection.append(token)
        elif token not in available:
            missing.append(token)
    if missing:
        print(
            "[warn] skipping unavailable fingers: {} (available: {})".format(
                ", ".join(sorted(set(missing))), ", ".join(available)
            )
        )
    if not selection:
        raise RuntimeError("Finger selection matched no available groups.")
    return selection


def _fingerwise_metrics(target: np.ndarray, pred: np.ndarray, finger_slices: Dict[str, List[int]]) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    for finger, rel_indices in finger_slices.items():
        if not rel_indices:
            continue
        finger_target = target[:, rel_indices]
        finger_pred = pred[:, rel_indices]
        mask = np.isfinite(finger_pred).all(axis=1)
        valid = int(np.count_nonzero(mask))
        if valid == 0:
            summary[finger] = {"rmse": float("nan"), "mae": float("nan"), "count": 0.0}
            continue
        diff = finger_target[mask] - finger_pred[mask]
        summary[finger] = {
            "rmse": float(np.sqrt(np.mean(diff ** 2))),
            "mae": float(np.mean(np.abs(diff))),
            "count": float(valid),
        }
    return summary


def _ensure_baseline_available(name: str, cls: Any) -> Any:
    if cls is None or torch is None:
        raise RuntimeError(f"{name} baseline is unavailable. Install PyTorch to evaluate this model.")
    return cls


def _latest_artifact_dir(base_dir: Path) -> Path:
    """Return latest run directory under 'artifacts' or fallback to 'artifacts_global'."""
    candidates: list[Path] = []
    if base_dir.exists():
        candidates.extend([p for p in base_dir.iterdir() if p.is_dir()])
    alt = base_dir.with_name(base_dir.name + "_global")
    if alt.exists():
        candidates.extend([p for p in alt.iterdir() if p.is_dir()])
    if not candidates:
        raise RuntimeError(
            f"No saved model runs found under '{base_dir}' or '{alt}'. Run benchmarks first or pass --artifact-dir."
        )
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _load_manifest(artifact_dir: Path) -> Dict[str, Any]:
    manifest_path = artifact_dir / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(f"Manifest file not found in '{artifact_dir}'.")
    with manifest_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)
def _load_scalers(artifact_dir: Path, manifest: Dict[str, Any]) -> Tuple[Any, Any]:
    scalers_path = artifact_dir / manifest["scalers"]
    if not scalers_path.exists():
        raise RuntimeError(f"Scaler file '{scalers_path}' referenced in manifest is missing.")
    with scalers_path.open("rb") as fh:
        scalers = pickle.load(fh)
    return scalers["obs_scaler"], scalers["act_scaler"]


def _select_eval_demo(
    trajectories: Sequence[Trajectory],
    desired: Optional[str],
    finger_suffix: Optional[str] = None,
) -> str:
    """Select evaluation demo, optionally matching finger suffix for per-finger mode."""
    if desired:
        eval_name = Path(desired).stem
        # If requesting an augmented variant that is not present, fall back to base stem
        if eval_name.endswith("_synced") is False and "_aug" in eval_name:
            base_candidate = eval_name.split("_aug")[0]
            if any(t.name == base_candidate for t in trajectories):
                eval_name = base_candidate
        if finger_suffix:
            eval_name_with_finger = f"{eval_name}_{finger_suffix}"
            if any(t.name == eval_name_with_finger for t in trajectories):
                return eval_name_with_finger
        if any(t.name == eval_name for t in trajectories):
            return eval_name
        available = ", ".join(sorted(t.name for t in trajectories))
        raise RuntimeError(f"Requested evaluation demo '{desired}' not available. Options: {available}")

    candidates: List[Tuple[float, str]] = []
    for traj in trajectories:
        traj_name = traj.name
        if finger_suffix and traj_name.endswith(f"_{finger_suffix}"):
            base_name = traj_name[:-len(f"_{finger_suffix}")]
        else:
            base_name = traj_name
        if not base_name.endswith("_synced"):
            continue
        candidates.append((0.0, traj_name))  # no timestamp, simple ordering
    if not candidates:
        raise RuntimeError("Could not infer a demonstration to visualise. No _synced demos found.")
    candidates.sort(reverse=True)
    return candidates[0][1]


def _metrics_with_mask(target: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    mask = np.isfinite(pred).all(axis=1)
    if np.count_nonzero(mask) == 0:
        return {"rmse": float("nan"), "mae": float("nan"), "r2": float("nan")}
    return compute_metrics(target[mask], pred[mask])


def _plot_model_grid(
    time_idx: np.ndarray,
    target: np.ndarray,
    predictions: Dict[str, np.ndarray],
    model_order: Sequence[str],
    axes_labels: Sequence[str],
    out_path: Path,
    observations: Optional[np.ndarray] = None,
    obs_labels: Optional[Sequence[str]] = None,
) -> None:
    """Plot grid of model predictions with optional observation subplot.
    
    Args:
        observations: (T, obs_dim) observation array to display
        obs_labels: Labels for observation dimensions
    """
    rows = len(model_order)
    act_dim = target.shape[1]
    
    # Add extra row for observations if provided
    total_rows = rows + 1 if observations is not None else rows
    
    fig, axes = plt.subplots(total_rows, act_dim, figsize=(4.2 * act_dim, 2.4 * total_rows), sharex=True)
    axes_array = np.atleast_2d(axes)
    
    # Plot observations in first row if provided
    row_offset = 0
    if observations is not None and obs_labels is not None:
        obs_to_plot = min(act_dim, observations.shape[1])  # Match action dimension count
        for col in range(act_dim):
            ax = axes_array[0, col]
            if col < obs_to_plot:
                ax.plot(time_idx, observations[:, col], linewidth=1.0, color='gray', alpha=0.7)
                ax.set_title(f"Obs: {obs_labels[col]}", fontsize=9)
            else:
                ax.axis('off')
            ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.5)
            if col == 0:
                ax.set_ylabel("Observations", fontsize=9)
        row_offset = 1
    
    # Plot model predictions
    for row, model_name in enumerate(model_order):
        plot_row = row + row_offset
        for col in range(act_dim):
            ax = axes_array[plot_row, col]
            ax.plot(time_idx, target[:, col], label="GT (demo)", linewidth=1.2)
            ax.plot(
                time_idx,
                predictions[model_name][:, col],
                label=model_name,
                linewidth=1.2,
                linestyle="--",
                alpha=0.9,
            )
            if row_offset == 0 and row == 0:
                ax.set_title(axes_labels[col])
            elif row_offset > 0 and row == 0:
                # Observations row already has title, so add action label
                current_title = ax.get_title()
                if current_title:
                    ax.set_title(f"{current_title}\nAction: {axes_labels[col]}", fontsize=9)
                else:
                    ax.set_title(f"Action: {axes_labels[col]}", fontsize=9)
            if plot_row == total_rows - 1:
                ax.set_xlabel("sample index")
            if col == 0:
                ax.set_ylabel(model_name)
            ax.grid(True, linestyle=":", linewidth=0.6)
            if row == 0 and col == act_dim - 1:
                ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def _torch_load(path: Path) -> Dict[str, Any]:
    if torch is None:
        raise RuntimeError("PyTorch is required to load neural model checkpoints.")
    return torch.load(path, map_location="cpu")  # type: ignore[call-arg]


def evaluate_models(args: argparse.Namespace) -> None:
    mode = args.mode.lower()
    
    if mode == "per-finger":
        # Per-finger mode: evaluate each finger separately
        evaluate_per_finger_models(args)
        return
    
    # Unified mode (default)
    default_root = DEFAULT_ARTIFACT_ROOT_UNIFIED
    artifact_dir = args.artifact_dir if args.artifact_dir else _latest_artifact_dir(default_root)
    print(f"[info] Mode: unified")
    print(f"[info] Using artifact directory: {artifact_dir}")
    manifest = _load_manifest(artifact_dir)
    action_columns = manifest.get("action_columns", ACTION_COLUMNS)
    if not action_columns:
        raise RuntimeError("Manifest does not describe any action columns to evaluate.")
    finger_groups = _group_action_columns(action_columns)
    if not finger_groups:
        raise RuntimeError("Could not determine finger groups from provided action columns.")
    available_fingers = list(finger_groups.keys())
    selected_fingers = _resolve_finger_selection(args.fingers, available_fingers)
    selected_indices_abs: List[int] = []
    selected_labels: List[str] = []
    for finger in selected_fingers:
        idxs = finger_groups.get(finger)
        if not idxs:
            continue
        selected_indices_abs.extend(idxs)
        selected_labels.extend(action_columns[i] for i in idxs)
    if not selected_indices_abs:
        raise RuntimeError("Finger selection resulted in zero action dimensions.")
    finger_slices_rel: Dict[str, List[int]] = {}
    for rel_idx, label in enumerate(selected_labels):
        prefix = _normalise_finger_token(label.split("_", 1)[0]) if "_" in label else _normalise_finger_token(label)
        finger_slices_rel.setdefault(prefix, []).append(rel_idx)
    # Preserve requested order
    finger_slices_rel = {finger: finger_slices_rel.get(finger, []) for finger in selected_fingers}

    obs_scaler, act_scaler = _load_scalers(artifact_dir, manifest)
    window = int(manifest.get("sequence_window", 1))
    diffusion_sampler = args.diffusion_sampler.lower()
    diffusion_eta = max(0.0, float(args.diffusion_eta))

    trajectories = load_dataset(args.stiffness_dir, args.stride, include_aug=args.augment)
    manifest_tests = manifest.get("test_trajectories", [])
    desired_demo = args.eval_demo or (manifest_tests[0] if manifest_tests else None)
    eval_name = _select_eval_demo(trajectories, desired_demo)
    print(f"[info] evaluating demonstration '{eval_name}'")

    try:
        test_traj = next(t for t in trajectories if t.name == eval_name)
    except StopIteration as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"Evaluation trajectory '{eval_name}' could not be loaded.") from exc

    test_obs = test_traj.observations
    test_act_full = test_traj.actions
    test_obs_s = obs_scaler.transform(test_obs)

    test_scaled = scale_trajectories([test_traj], obs_scaler, act_scaler)
    test_offsets = compute_offsets([test_traj])
    seq_obs, _, _, seq_indices = build_sequence_dataset(
        test_scaled,
        [test_traj],
        max(1, window),
        test_offsets,
    )

    available_models = list(manifest.get("models", {}).keys())
    if not available_models:
        raise RuntimeError(f"Manifest at '{artifact_dir}' does not list any models to evaluate.")

    requested = {token.strip().lower() for token in args.models.split(",") if token.strip()}
    if not requested or "all" in requested:
        model_order = available_models
    else:
        model_order = [name for name in available_models if name.lower() in requested]
        available_map = {name.lower(): name for name in available_models}
        missing = sorted(req for req in requested if req not in available_map)
        if missing:
            print(f"[warn] skipping unavailable models: {', '.join(missing)}")
    if not model_order:
        raise RuntimeError("No valid models selected for visualisation.")

    predictions: Dict[str, np.ndarray] = {}
    metrics_summary: Dict[str, Dict[str, float]] = {}
    finger_metrics_summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    gmm_cache: Dict[str, GMMConditional] = {}
    time_idx = np.arange(test_act_full.shape[0])
    target_subset = test_act_full[:, selected_indices_abs]

    for model_name in model_order:
        entry = manifest["models"][model_name]
        kind = entry.get("kind", model_name).lower()
        artifact_path = artifact_dir / entry["path"]
        if not artifact_path.exists():
            raise RuntimeError(f"Artifact file for model '{model_name}' not found: {artifact_path}")

        if kind in {"gmm", "gmr"}:
            cache_key = entry["path"]
            gmm_model = gmm_cache.get(cache_key)
            if gmm_model is None:
                with artifact_path.open("rb") as fh:
                    gmm_model = pickle.load(fh)
                if not isinstance(gmm_model, GMMConditional):
                    raise RuntimeError(f"Artifact '{artifact_path}' does not contain a GMMConditional model.")
                gmm_cache[cache_key] = gmm_model
            if kind == "gmr":
                mode = "mean"
                n_samples = 1
            else:
                mode = entry.get("mode", "sample")
                n_samples = int(entry.get("n_samples", 16))
            pred_scaled = gmm_model.predict(test_obs_s, mode=mode, n_samples=n_samples)
            pred = act_scaler.inverse_transform(pred_scaled)
            pred_subset = pred[:, selected_indices_abs]
            predictions[model_name] = pred_subset
            metrics_summary[model_name] = compute_metrics(target_subset, pred_subset)
            finger_metrics_summary[model_name] = _fingerwise_metrics(target_subset, pred_subset, finger_slices_rel)
            continue

        if kind == "bc":
            bc_cls = _ensure_baseline_available("Behavior cloning", BehaviorCloningBaseline)
            state = _torch_load(artifact_path)
            config = state.get("config", {})
            bc = bc_cls(
                obs_dim=int(config.get("obs_dim", test_obs_s.shape[1])),
                act_dim=int(config.get("act_dim", test_act_full.shape[1])),
                hidden_dim=int(config.get("hidden_dim", 256)),
                depth=int(config.get("depth", 3)),
                lr=float(config.get("lr", 1e-3)),
                batch_size=int(config.get("batch_size", 256)),
                epochs=int(config.get("epochs", 0)),
                seed=int(config.get("seed", 0)),
                device="cpu",
                log_name="bc_eval",
            )
            bc.model.load_state_dict(state["state_dict"])
            pred_scaled = bc.predict(test_obs_s)
            pred = act_scaler.inverse_transform(pred_scaled)
            pred_subset = pred[:, selected_indices_abs]
            predictions[model_name] = pred_subset
            metrics_summary[model_name] = compute_metrics(target_subset, pred_subset)
            finger_metrics_summary[model_name] = _fingerwise_metrics(target_subset, pred_subset, finger_slices_rel)
            continue

        if kind == "ibc":
            ibc_cls = _ensure_baseline_available("IBC", IBCBaseline)
            state = _torch_load(artifact_path)
            config = state.get("config", {})
            ibc = ibc_cls(
                obs_dim=int(config.get("obs_dim", test_obs_s.shape[1])),
                act_dim=int(config.get("act_dim", test_act_full.shape[1])),
                hidden_dim=int(config.get("hidden_dim", 256)),
                depth=int(config.get("depth", 3)),
                lr=float(config.get("lr", 1e-3)),
                batch_size=int(config.get("batch_size", 256)),
                epochs=int(config.get("epochs", 0)),
                noise_std=float(config.get("noise_std", 0.5)),
                langevin_steps=int(config.get("langevin_steps", 30)),
                step_size=float(config.get("step_size", 1e-2)),
                seed=int(config.get("seed", 0)),
                device="cpu",
                log_name="ibc_eval",
            )
            ibc.model.load_state_dict(state["state_dict"])
            pred_scaled = ibc.predict(test_obs_s, n_samples=int(entry.get("n_samples", 1)))
            pred = act_scaler.inverse_transform(pred_scaled)
            pred_subset = pred[:, selected_indices_abs]
            predictions[model_name] = pred_subset
            metrics_summary[model_name] = compute_metrics(target_subset, pred_subset)
            finger_metrics_summary[model_name] = _fingerwise_metrics(target_subset, pred_subset, finger_slices_rel)
            continue

        if kind == "diffusion":
            diff_cls = _ensure_baseline_available("Diffusion policy", DiffusionPolicyBaseline)
            state = _torch_load(artifact_path)
            config = state.get("config", {})
            temporal = bool(config.get("temporal", entry.get("temporal", False)))
            diff = diff_cls(
                obs_dim=int(
                    config.get(
                        "obs_dim",
                        seq_obs.shape[-1] if temporal and seq_obs.size else test_obs_s.shape[1],
                    )
                ),
                act_dim=int(config.get("act_dim", test_act_full.shape[1])),
                timesteps=int(config.get("timesteps", 50)),
                hidden_dim=int(config.get("hidden_dim", 256)),
                time_dim=int(config.get("time_dim", 64)),
                lr=float(config.get("lr", 1e-3)),
                batch_size=int(config.get("batch_size", 256)),
                epochs=int(config.get("epochs", 0)),
                seed=int(config.get("seed", 0)),
                temporal=temporal,
                device="cpu",
                log_name=f"{model_name}_eval",
            )
            diff.model.load_state_dict(state["state_dict"])
            sample_count = int(entry.get("n_samples", 4))
            if temporal:
                if seq_obs.shape[0] == 0:
                    full_pred = np.full_like(test_act_full, np.nan)
                else:
                    pred_seq_scaled = diff.predict(
                        seq_obs,
                        n_samples=sample_count,
                        sampler=diffusion_sampler,
                        eta=diffusion_eta,
                    )
                    pred_seq = act_scaler.inverse_transform(pred_seq_scaled)
                    full_pred = np.full_like(test_act_full, np.nan)
                    full_pred[seq_indices] = pred_seq
                pred_subset = full_pred[:, selected_indices_abs]
                predictions[model_name] = pred_subset
                metrics_summary[model_name] = _metrics_with_mask(target_subset, pred_subset)
                finger_metrics_summary[model_name] = _fingerwise_metrics(target_subset, pred_subset, finger_slices_rel)
            else:
                pred_scaled = diff.predict(test_obs_s, n_samples=sample_count)
                pred = act_scaler.inverse_transform(pred_scaled)
                pred_subset = pred[:, selected_indices_abs]
                predictions[model_name] = pred_subset
                metrics_summary[model_name] = compute_metrics(target_subset, pred_subset)
                finger_metrics_summary[model_name] = _fingerwise_metrics(target_subset, pred_subset, finger_slices_rel)
            continue

        if kind == "lstm_gmm":
            lstm_cls = _ensure_baseline_available("LSTM-GMM", LSTMGMMBaseline)
            state = _torch_load(artifact_path)
            config = state.get("config", {})
            seq_len = int(config.get("seq_len", entry.get("seq_len", window)))
            lstm = lstm_cls(
                obs_dim=int(
                    config.get(
                        "obs_dim",
                        seq_obs.shape[-1] if seq_obs.size else test_obs_s.shape[1],
                    )
                ),
                act_dim=int(config.get("act_dim", test_act_full.shape[1])),
                seq_len=seq_len,
                n_components=int(config.get("n_components", 5)),
                hidden_dim=int(config.get("hidden_dim", 256)),
                n_layers=int(config.get("n_layers", 1)),
                lr=float(config.get("lr", 1e-3)),
                batch_size=int(config.get("batch_size", 256)),
                epochs=int(config.get("epochs", 0)),
                seed=int(config.get("seed", 0)),
                device="cpu",
                log_name="lstm_gmm_eval",
            )
            lstm.model.load_state_dict(state["state_dict"])
            sample_count = int(entry.get("n_samples", config.get("n_components", 5)))
            if seq_obs.shape[0] == 0:
                full_pred = np.full_like(test_act_full, np.nan)
            else:
                pred_seq_scaled = lstm.predict(seq_obs, mode="mean", n_samples=sample_count)
                pred_seq = act_scaler.inverse_transform(pred_seq_scaled)
                full_pred = np.full_like(test_act_full, np.nan)
                full_pred[seq_indices] = pred_seq
            pred_subset = full_pred[:, selected_indices_abs]
            predictions[model_name] = pred_subset
            metrics_summary[model_name] = _metrics_with_mask(target_subset, pred_subset)
            finger_metrics_summary[model_name] = _fingerwise_metrics(target_subset, pred_subset, finger_slices_rel)
            continue

        if kind == "gp":
            gp_cls = _ensure_baseline_available("Gaussian Process", GPBaseline)
            with artifact_path.open("rb") as fh:
                gp_data = pickle.load(fh)
            gp_model = gp_data.get("model")
            if gp_model is None:
                raise RuntimeError(f"GP artifact '{artifact_path}' does not contain a 'model' key.")
            # GP operates on raw (unscaled) observations
            pred, pred_std = gp_model.predict(test_obs, return_std=True)
            pred_subset = pred[:, selected_indices_abs]
            predictions[model_name] = pred_subset
            metrics_summary[model_name] = compute_metrics(target_subset, pred_subset)
            finger_metrics_summary[model_name] = _fingerwise_metrics(target_subset, pred_subset, finger_slices_rel)
            continue

        if kind == "mdn":
            mdn_cls = _ensure_baseline_available("Mixture Density Network", MDNBaseline)
            state = _torch_load(artifact_path)
            config = state.get("config", {})
            mdn = mdn_cls(
                obs_dim=int(config.get("obs_dim", test_obs_s.shape[1])),
                act_dim=int(config.get("act_dim", test_act_full.shape[1])),
                hidden_units=config.get("hidden_units", [128, 128, 64]),
                n_components=int(config.get("n_components", 5)),
                covariance_type=str(config.get("covariance_type", "diag")),
                activation=str(config.get("activation", "relu")),
                dropout=float(config.get("dropout", 0.1)),
                lr=float(config.get("lr", 1e-3)),
                batch_size=int(config.get("batch_size", 256)),
                epochs=int(config.get("epochs", 0)),
                seed=int(config.get("seed", 0)),
                device="cpu",
                log_name="mdn_eval",
            )
            mdn.model.load_state_dict(state["model_state"])
            pred_scaled = mdn.predict(test_obs_s, mode="mean", n_samples=int(entry.get("n_samples", 1)))
            pred = act_scaler.inverse_transform(pred_scaled)
            pred_subset = pred[:, selected_indices_abs]
            predictions[model_name] = pred_subset
            metrics_summary[model_name] = compute_metrics(target_subset, pred_subset)
            finger_metrics_summary[model_name] = _fingerwise_metrics(target_subset, pred_subset, finger_slices_rel)
            continue

        raise RuntimeError(f"Unsupported model entry '{model_name}' (kind='{kind}') in manifest.")

    # Prepare observation data for visualization
    obs_columns = manifest.get("obs_columns", OBS_COLUMNS)
    obs_subset_indices = list(range(min(len(selected_indices_abs), test_obs.shape[1])))
    obs_subset_labels = [obs_columns[i] if i < len(obs_columns) else f"obs_{i}" for i in obs_subset_indices]
    
    plot_dir = args.output_dir if args.output_dir else (artifact_dir / "plots")
    plot_path = plot_dir / f"{eval_name}_comparison.png"
    _plot_model_grid(
        time_idx, 
        target_subset, 
        predictions, 
        model_order, 
        selected_labels, 
        plot_path,
        observations=test_obs[:, obs_subset_indices] if args.show_obs else None,
        obs_labels=obs_subset_labels if args.show_obs else None,
    )

    for model_name in model_order:
        metrics = metrics_summary.get(
            model_name,
            {"rmse": float("nan"), "mae": float("nan"), "r2": float("nan")},
        )
        print(
            f"[{model_name}] rmse={metrics['rmse']:.4f} "
            f"mae={metrics['mae']:.4f} r2={metrics['r2']:.4f}"
        )
        finger_stats = finger_metrics_summary.get(model_name, {})
        for finger in selected_fingers:
            stats = finger_stats.get(finger)
            if not stats:
                continue
            print(
                f"    ({finger.upper()}) rmse={stats['rmse']:.4f} "
                f"mae={stats['mae']:.4f} samples={int(stats['count'])}"
            )
    print(f"[done] plot saved to {plot_path}")


def evaluate_per_finger_models(args: argparse.Namespace) -> None:
    """Evaluate per-finger trained models with finger-specific artifacts."""
    # Get base artifact directory
    default_root = DEFAULT_ARTIFACT_ROOT_PER_FINGER
    base_artifact_dir = args.artifact_dir if args.artifact_dir else _latest_artifact_dir(default_root)
    print(f"[info] Mode: per-finger")
    print(f"[info] Base artifact directory: {base_artifact_dir}")
    
    # Determine which fingers to evaluate
    requested_fingers = args.fingers.lower().split(",") if args.fingers and args.fingers.lower() != "all" else FINGER_LIST
    requested_fingers = [f.strip() for f in requested_fingers if f.strip() in FINGER_LIST]
    
    if not requested_fingers:
        raise RuntimeError(f"No valid fingers selected. Available: {', '.join(FINGER_LIST)}")
    
    print(f"[info] Evaluating fingers: {', '.join(requested_fingers)}")
    
    # Collect all finger results for unified visualization
    all_finger_data: Dict[str, Dict[str, Any]] = {}
    
    # Process each finger independently
    for finger in requested_fingers:
        finger_artifact_dir = base_artifact_dir / finger
        if not finger_artifact_dir.exists():
            print(f"[warn] Skipping {finger.upper()}: artifact directory not found at {finger_artifact_dir}")
            continue
        
        print(f"\n{'='*80}")
        print(f"[info] Evaluating finger: {finger.upper()}")
        print(f"[info] Artifact directory: {finger_artifact_dir}")
        print(f"{'='*80}")
        
        # Load finger-specific manifest
        manifest = _load_manifest(finger_artifact_dir)
        action_columns = manifest.get("action_columns", ACTION_COLUMNS)
        if not action_columns:
            print(f"[warn] Skipping {finger.upper()}: no action columns in manifest")
            continue
        
        obs_scaler, act_scaler = _load_scalers(finger_artifact_dir, manifest)
        window = int(manifest.get("sequence_window", 1))
        diffusion_sampler = args.diffusion_sampler.lower()
        diffusion_eta = max(0.0, float(args.diffusion_eta))
        
        # Load dataset using per-finger loader to get correct observation dimensions
        trajectories = load_dataset_per_finger(args.stiffness_dir, args.stride, finger, include_aug=args.augment)
        manifest_tests = manifest.get("test_trajectories", [])
        desired_demo = args.eval_demo or (manifest_tests[0] if manifest_tests else None)
        eval_name = _select_eval_demo(trajectories, desired_demo, finger_suffix=finger)
        # Avoid double suffixing (function may already return suffixed name)
        eval_name_finger = eval_name if eval_name.endswith(f"_{finger}") else f"{eval_name}_{finger}"
        print(f"[info] Using demonstration: '{eval_name_finger}'")
        
        try:
            test_traj = next(t for t in trajectories if t.name == eval_name_finger)
        except StopIteration:
            print(f"[warn] Skipping {finger.upper()}: trajectory '{eval_name_finger}' not found")
            continue
        
        test_obs = test_traj.observations
        test_act_full = test_traj.actions
        test_obs_s = obs_scaler.transform(test_obs)
        
        test_scaled = scale_trajectories([test_traj], obs_scaler, act_scaler)
        test_offsets = compute_offsets([test_traj])
        seq_obs, _, _, seq_indices = build_sequence_dataset(
            test_scaled,
            [test_traj],
            max(1, window),
            test_offsets,
        )
        
        available_models = list(manifest.get("models", {}).keys())
        if not available_models:
            print(f"[warn] Skipping {finger.upper()}: no models in manifest")
            continue
        
        requested = {token.strip().lower() for token in args.models.split(",") if token.strip()}
        if not requested or "all" in requested:
            model_order = available_models
        else:
            model_order = [name for name in available_models if name.lower() in requested]
        
        if not model_order:
            print(f"[warn] Skipping {finger.upper()}: no valid models selected")
            continue
        
        predictions: Dict[str, np.ndarray] = {}
        metrics_summary: Dict[str, Dict[str, float]] = {}
        gmm_cache: Dict[str, GMMConditional] = {}
        time_idx = np.arange(test_act_full.shape[0])
        
        # Evaluate each model for this finger
        for model_name in model_order:
            entry = manifest["models"][model_name]
            kind = entry.get("kind", model_name).lower()
            artifact_path = finger_artifact_dir / entry["path"]
            
            if not artifact_path.exists():
                print(f"[warn] Skipping {model_name}: artifact not found at {artifact_path}")
                continue
            
            try:
                # Use same evaluation logic as unified mode
                if kind in {"gmm", "gmr"}:
                    cache_key = entry["path"]
                    gmm_model = gmm_cache.get(cache_key)
                    if gmm_model is None:
                        with artifact_path.open("rb") as fh:
                            gmm_model = pickle.load(fh)
                        if not isinstance(gmm_model, GMMConditional):
                            gmm_model = gmm_model.get("model")  # Try extracting from dict
                        gmm_cache[cache_key] = gmm_model
                    mode_str = "mean" if kind == "gmr" else entry.get("mode", "sample")
                    n_samples = 1 if kind == "gmr" else int(entry.get("n_samples", 16))
                    pred_scaled = gmm_model.predict(test_obs_s, mode=mode_str, n_samples=n_samples)
                    pred = act_scaler.inverse_transform(pred_scaled)
                    predictions[model_name] = pred
                    metrics_summary[model_name] = compute_metrics(test_act_full, pred)
                
                elif kind == "bc":
                    bc_cls = _ensure_baseline_available("Behavior cloning", BehaviorCloningBaseline)
                    state = _torch_load(artifact_path)
                    config = state.get("config", {})
                    bc = bc_cls(
                        obs_dim=int(config.get("obs_dim", test_obs_s.shape[1])),
                        act_dim=int(config.get("act_dim", test_act_full.shape[1])),
                        hidden_dim=int(config.get("hidden_dim", 256)),
                        depth=int(config.get("depth", 3)),
                        lr=float(config.get("lr", 1e-3)),
                        batch_size=int(config.get("batch_size", 256)),
                        epochs=int(config.get("epochs", 0)),
                        seed=int(config.get("seed", 0)),
                        device="cpu",
                        log_name="bc_eval",
                    )
                    bc.model.load_state_dict(state["state_dict"])
                    pred_scaled = bc.predict(test_obs_s)
                    pred = act_scaler.inverse_transform(pred_scaled)
                    predictions[model_name] = pred
                    metrics_summary[model_name] = compute_metrics(test_act_full, pred)
                
                elif kind == "ibc":
                    ibc_cls = _ensure_baseline_available("IBC", IBCBaseline)
                    state = _torch_load(artifact_path)
                    config = state.get("config", {})
                    ibc = ibc_cls(
                        obs_dim=int(config.get("obs_dim", test_obs_s.shape[1])),
                        act_dim=int(config.get("act_dim", test_act_full.shape[1])),
                        hidden_dim=int(config.get("hidden_dim", 256)),
                        depth=int(config.get("depth", 3)),
                        lr=float(config.get("lr", 1e-3)),
                        batch_size=int(config.get("batch_size", 256)),
                        epochs=int(config.get("epochs", 0)),
                        noise_std=float(config.get("noise_std", 0.5)),
                        langevin_steps=int(config.get("langevin_steps", 30)),
                        step_size=float(config.get("step_size", 1e-2)),
                        seed=int(config.get("seed", 0)),
                        device="cpu",
                        log_name="ibc_eval",
                    )
                    ibc.model.load_state_dict(state["state_dict"])
                    pred_scaled = ibc.predict(test_obs_s, n_samples=int(entry.get("n_samples", 1)))
                    pred = act_scaler.inverse_transform(pred_scaled)
                    predictions[model_name] = pred
                    metrics_summary[model_name] = compute_metrics(test_act_full, pred)
                
                elif kind == "diffusion":
                    diff_cls = _ensure_baseline_available("Diffusion policy", DiffusionPolicyBaseline)
                    state = _torch_load(artifact_path)
                    config = state.get("config", {})
                    temporal = bool(config.get("temporal", entry.get("temporal", False)))
                    diff = diff_cls(
                        obs_dim=int(config.get("obs_dim", seq_obs.shape[-1] if temporal and seq_obs.size else test_obs_s.shape[1])),
                        act_dim=int(config.get("act_dim", test_act_full.shape[1])),
                        timesteps=int(config.get("timesteps", 50)),
                        hidden_dim=int(config.get("hidden_dim", 256)),
                        time_dim=int(config.get("time_dim", 64)),
                        lr=float(config.get("lr", 1e-3)),
                        batch_size=int(config.get("batch_size", 256)),
                        epochs=int(config.get("epochs", 0)),
                        seed=int(config.get("seed", 0)),
                        temporal=temporal,
                        device="cpu",
                        log_name=f"{model_name}_eval",
                    )
                    diff.model.load_state_dict(state["state_dict"])
                    sample_count = int(entry.get("n_samples", 4))
                    if temporal:
                        if seq_obs.shape[0] == 0:
                            pred = np.full_like(test_act_full, np.nan)
                        else:
                            pred_seq_scaled = diff.predict(seq_obs, n_samples=sample_count, sampler=diffusion_sampler, eta=diffusion_eta)
                            pred_seq = act_scaler.inverse_transform(pred_seq_scaled)
                            pred = np.full_like(test_act_full, np.nan)
                            pred[seq_indices] = pred_seq
                        predictions[model_name] = pred
                        metrics_summary[model_name] = _metrics_with_mask(test_act_full, pred)
                    else:
                        pred_scaled = diff.predict(test_obs_s, n_samples=sample_count)
                        pred = act_scaler.inverse_transform(pred_scaled)
                        predictions[model_name] = pred
                        metrics_summary[model_name] = compute_metrics(test_act_full, pred)
                
                elif kind == "lstm_gmm":
                    lstm_cls = _ensure_baseline_available("LSTM-GMM", LSTMGMMBaseline)
                    state = _torch_load(artifact_path)
                    config = state.get("config", {})
                    seq_len = int(config.get("seq_len", entry.get("seq_len", window)))
                    lstm = lstm_cls(
                        obs_dim=int(config.get("obs_dim", seq_obs.shape[-1] if seq_obs.size else test_obs_s.shape[1])),
                        act_dim=int(config.get("act_dim", test_act_full.shape[1])),
                        seq_len=seq_len,
                        n_components=int(config.get("n_components", 5)),
                        hidden_dim=int(config.get("hidden_dim", 256)),
                        n_layers=int(config.get("n_layers", 1)),
                        lr=float(config.get("lr", 1e-3)),
                        batch_size=int(config.get("batch_size", 256)),
                        epochs=int(config.get("epochs", 0)),
                        seed=int(config.get("seed", 0)),
                        device="cpu",
                        log_name="lstm_gmm_eval",
                    )
                    lstm.model.load_state_dict(state["state_dict"])
                    sample_count = int(entry.get("n_samples", config.get("n_components", 5)))
                    if seq_obs.shape[0] == 0:
                        pred = np.full_like(test_act_full, np.nan)
                    else:
                        pred_seq_scaled = lstm.predict(seq_obs, mode="mean", n_samples=sample_count)
                        pred_seq = act_scaler.inverse_transform(pred_seq_scaled)
                        pred = np.full_like(test_act_full, np.nan)
                        pred[seq_indices] = pred_seq
                    predictions[model_name] = pred
                    metrics_summary[model_name] = _metrics_with_mask(test_act_full, pred)
                
            except Exception as exc:
                print(f"[warn] Failed to evaluate {model_name} for {finger.upper()}: {exc}")
                continue
        
        # Store finger results for unified plot
        if predictions:
            all_finger_data[finger] = {
                "predictions": predictions,
                "metrics": metrics_summary,
                "target": test_act_full,
                "time_idx": time_idx,
                "action_columns": action_columns,
                "model_order": model_order,
                "eval_name": eval_name,
            }
            
            # Print metrics
            for model_name in model_order:
                metrics = metrics_summary.get(model_name, {"rmse": float("nan"), "mae": float("nan"), "r2": float("nan")})
                print(f"[{model_name}] rmse={metrics['rmse']:.4f} mae={metrics['mae']:.4f} r2={metrics['r2']:.4f}")
    
    # Create unified plot showing all fingers together
    if all_finger_data:
        print(f"\n{'='*80}")
        print("[info] Creating unified plot for all fingers")
        print(f"{'='*80}")
        
        # Use first finger's model order (should be same for all)
        first_finger = list(all_finger_data.keys())[0]
        model_order = all_finger_data[first_finger]["model_order"]
        eval_name = all_finger_data[first_finger]["eval_name"]
        
        # Build combined predictions and targets
        combined_predictions: Dict[str, List[np.ndarray]] = {}
        combined_targets: List[np.ndarray] = []
        combined_labels: List[str] = []
        
        for finger in requested_fingers:
            if finger not in all_finger_data:
                continue
            data = all_finger_data[finger]
            combined_targets.append(data["target"])
            combined_labels.extend([f"{finger.upper()}_{col}" for col in data["action_columns"]])
            
            for model_name in model_order:
                if model_name not in combined_predictions:
                    combined_predictions[model_name] = []
                combined_predictions[model_name].append(data["predictions"][model_name])
        
        # Concatenate along action dimension (axis=1)
        combined_target = np.hstack(combined_targets)
        combined_preds_final: Dict[str, np.ndarray] = {}
        for model_name in model_order:
            combined_preds_final[model_name] = np.hstack(combined_predictions[model_name])
        
        # Use first finger's time index (should be same for all if same demo)
        time_idx = all_finger_data[first_finger]["time_idx"]
        
        # Create unified plot
        plot_dir = args.output_dir if args.output_dir else (base_artifact_dir / "plots")
        plot_path = plot_dir / f"{eval_name}_all_fingers_comparison.png"
        _plot_model_grid(
            time_idx,
            combined_target,
            combined_preds_final,
            model_order,
            combined_labels,
            plot_path,
        )
        
        print(f"[done] Unified plot saved to {plot_path}")
        
        # Print overall metrics summary
        print(f"\n{'='*80}")
        print("[info] Overall metrics summary")
        print(f"{'='*80}")
        for model_name in model_order:
            model_metrics = []
            for finger in requested_fingers:
                if finger in all_finger_data:
                    metrics = all_finger_data[finger]["metrics"].get(model_name)
                    if metrics:
                        model_metrics.append((finger, metrics))
            
            if model_metrics:
                avg_rmse = np.mean([m[1]["rmse"] for m in model_metrics if not np.isnan(m[1]["rmse"])])
                avg_mae = np.mean([m[1]["mae"] for m in model_metrics if not np.isnan(m[1]["mae"])])
                avg_r2 = np.mean([m[1]["r2"] for m in model_metrics if not np.isnan(m[1]["r2"])])
                print(f"[{model_name}] avg_rmse={avg_rmse:.4f} avg_mae={avg_mae:.4f} avg_r2={avg_r2:.4f}")
                for finger, metrics in model_metrics:
                    print(f"    ({finger.upper()}) rmse={metrics['rmse']:.4f} mae={metrics['mae']:.4f} r2={metrics['r2']:.4f}")



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualise saved stiffness policy predictions for a single demonstration."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="unified",
        choices=["unified", "per-finger"],
        help="Evaluation mode: 'unified' (single model) or 'per-finger' (separate models per finger).",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default="/home/songwoo/ros2_ws/icra2025/src/hri_falcon_robot_bridge/outputs/models/policy_learning_unified/artifacts/20251122_181241",
        help="Optional explicit artifact directory (timestamped run). If omitted, latest run is auto-selected per mode.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="all",
        help="Comma separated list of model names to plot (or 'all').",
    )
    parser.add_argument(
        "--fingers",
        type=str,
        default="all",
        help="Comma separated finger codes to visualise (e.g., 'th,if,mf'). Defaults to all available.",
    )
    parser.add_argument(
        "--show-obs",
        action="store_true",
        default=True,
        help="Display observations in the first row of the comparison plot.",
    )
    parser.add_argument(
        "--diffusion-sampler",
        type=str,
        default="ddpm",
        choices=["ddpm", "ddim"],
        help="Sampler to use when generating diffusion policy rollouts.",
    )
    parser.add_argument(
        "--diffusion-eta",
        type=float,
        default=0.0,
        help="Eta parameter for DDIM sampling (0 for deterministic). Ignored for DDPM.",
    )
    parser.add_argument(
        "--eval-demo",
        type=str,
        default=None,
        help="Optional demonstration stem (without .csv). Defaults to manifest entry or latest _synced.",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Include on-disk augmented stiffness profiles (*_augN.csv) when loading dataset",
    )
    parser.add_argument(
        "--stiffness-dir",
        type=Path,
        default=DEFAULT_STIFFNESS_DIR,
        help="Directory containing stiffness reconstruction CSVs.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Stride for subsampling demonstrations before evaluation.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to store the generated comparison plot (defaults to artifacts/<run>/plots).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    evaluate_models(parse_args())
