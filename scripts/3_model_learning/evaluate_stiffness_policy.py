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
DEFAULT_LOG_DIR = benchmarks.DEFAULT_LOG_DIR
DEFAULT_STIFFNESS_DIR = benchmarks.DEFAULT_STIFFNESS_DIR
DEFAULT_OUTPUT_DIR = benchmarks.DEFAULT_OUTPUT_DIR

GMMConditional = benchmarks.GMMConditional
Trajectory = benchmarks.Trajectory
build_sequence_dataset = benchmarks.build_sequence_dataset
compute_metrics = benchmarks.compute_metrics
compute_offsets = benchmarks.compute_offsets
load_dataset = benchmarks.load_dataset
scale_trajectories = benchmarks.scale_trajectories

BehaviorCloningBaseline = getattr(benchmarks, "BehaviorCloningBaseline", None)
IBCBaseline = getattr(benchmarks, "IBCBaseline", None)
DiffusionPolicyBaseline = getattr(benchmarks, "DiffusionPolicyBaseline", None)
LSTMGMMBaseline = getattr(benchmarks, "LSTMGMMBaseline", None)

DEFAULT_ARTIFACT_ROOT = DEFAULT_OUTPUT_DIR / "artifacts"
DEFAULT_PLOT_DIR = DEFAULT_OUTPUT_DIR / "plots"


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
    log_dir: Path,
    desired: Optional[str],
) -> str:
    if desired:
        eval_name = Path(desired).stem
        
        # Direct match
        if any(t.name == eval_name for t in trajectories):
            return eval_name
        
        # Try to match base name (remove augmentation suffix like _aug1, _aug2, etc.)
        # This allows using augmented demos for evaluation
        if '_aug' in eval_name:
            # Extract base name (e.g., "demo_aug2" -> "demo")
            base_name = eval_name.rsplit('_aug', 1)[0]
            matches = [t.name for t in trajectories if t.name == base_name or t.name.startswith(base_name + '_aug')]
            if matches:
                print(f"[info] Augmented demo '{eval_name}' requested but using closest match: '{matches[0]}'")
                return matches[0]
        
        # No match found
        available = ", ".join(sorted(t.name for t in trajectories))
        raise RuntimeError(f"Requested evaluation demo '{desired}' not available. Options: {available}")

    candidates: List[Tuple[float, str]] = []
    for traj in trajectories:
        log_path = log_dir / f"{traj.name}.csv"
        if log_path.exists():
            try:
                candidates.append((log_path.stat().st_mtime, traj.name))
            except FileNotFoundError:  # pragma: no cover - race guard
                continue
    if not candidates:
        raise RuntimeError("Could not infer a demonstration to visualise.")
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
) -> None:
    rows = len(model_order)
    act_dim = target.shape[1]
    fig, axes = plt.subplots(rows, act_dim, figsize=(4.2 * act_dim, 2.4 * rows), sharex=True)
    axes_array = np.atleast_2d(axes)
    for row, model_name in enumerate(model_order):
        for col in range(act_dim):
            ax = axes_array[row, col]
            ax.plot(time_idx, target[:, col], label="target", linewidth=1.2)
            ax.plot(
                time_idx,
                predictions[model_name][:, col],
                label=model_name,
                linewidth=1.2,
                linestyle="--",
                alpha=0.9,
            )
            if row == 0:
                ax.set_title(axes_labels[col])
            if row == rows - 1:
                ax.set_xlabel("sample index")
            if col == 0:
                ax.set_ylabel(model_name)
            ax.grid(True, linestyle=":", linewidth=0.6)
            if row == 0 and col == act_dim - 1:
                ax.legend(loc="upper right")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def _torch_load(path: Path) -> Dict[str, Any]:
    if torch is None:
        raise RuntimeError("PyTorch is required to load neural model checkpoints.")
    return torch.load(path, map_location="cpu")  # type: ignore[call-arg]


def evaluate_models(args: argparse.Namespace) -> None:
    artifact_dir = args.artifact_dir if args.artifact_dir else _latest_artifact_dir(DEFAULT_ARTIFACT_ROOT)
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

    trajectories = load_dataset(args.log_dir, args.stiffness_dir, args.stride)
    manifest_tests = manifest.get("test_trajectories", [])
    desired_demo = args.eval_demo or (manifest_tests[0] if manifest_tests else None)
    eval_name = _select_eval_demo(trajectories, args.log_dir, desired_demo)
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

        raise RuntimeError(f"Unsupported model entry '{model_name}' (kind='{kind}') in manifest.")

    plot_dir = args.output_dir if args.output_dir else (artifact_dir / "plots")
    plot_path = plot_dir / f"{eval_name}_comparison.png"
    _plot_model_grid(time_idx, target_subset, predictions, model_order, selected_labels, plot_path)

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualise saved stiffness policy predictions for a single demonstration."
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=None,
        help="Directory containing saved model artifacts (defaults to latest timestamped run).",
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
        help="Explicit demonstration stem to visualise (defaults to latest or manifest entry).",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=DEFAULT_LOG_DIR,
        help="Directory containing raw demonstration CSV logs.",
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
