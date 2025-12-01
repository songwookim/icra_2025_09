#!/usr/bin/env python3
"""Evaluate Diffusion Policy variants: Single Action vs Action Chunking comparison.

This script compares:
- diffusion (non-temporal, single action)
- diffusion_t (temporal/GRU encoder, single action)
- diffusion with action_horizon > 1 (action chunking)
- diffusion_t with action_horizon > 1 (temporal + action chunking)
- Low-pass filter effect on predictions

Outputs:
- Console: per-model RMSE/MAE/R², inference time statistics
- Figure: comparison plot of GT vs predictions
- Summary table of Single Action vs Action Chunking performance
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import os
if os.environ.get("DISPLAY", "") == "":
    import matplotlib
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt  # For low-pass filter

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None
    F = None

import run_stiffness_policy_benchmarks as benchmarks

# Import shared utilities
ACTION_COLUMNS = benchmarks.ACTION_COLUMNS
OBS_COLUMNS = benchmarks.OBS_COLUMNS
DEFAULT_STIFFNESS_DIR = benchmarks.DEFAULT_STIFFNESS_DIR
DEFAULT_OUTPUT_DIR = benchmarks.DEFAULT_OUTPUT_DIR

Trajectory = benchmarks.Trajectory
build_sequence_dataset = benchmarks.build_sequence_dataset
compute_metrics = benchmarks.compute_metrics
compute_offsets = benchmarks.compute_offsets
load_dataset = benchmarks.load_dataset
scale_trajectories = benchmarks.scale_trajectories

DiffusionPolicyBaseline = getattr(benchmarks, "DiffusionPolicyBaseline", None)

_PKG_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_ROOT = _PKG_ROOT / "outputs" / "models" / "policy_learning_unified" / "artifacts"


def apply_lowpass_filter(data: np.ndarray, cutoff_hz: float = 2.0, fs: float = 100.0, order: int = 2) -> np.ndarray:
    """Apply Butterworth low-pass filter to smooth predictions.
    
    Args:
        data: (T, D) array of predictions
        cutoff_hz: Cutoff frequency in Hz
        fs: Sampling frequency in Hz (default 100Hz for robot control)
        order: Filter order
    
    Returns:
        Filtered data with same shape
    """
    if data.shape[0] < 15:  # Need enough samples for filtering
        return data
    
    nyquist = fs / 2.0
    normalized_cutoff = cutoff_hz / nyquist
    normalized_cutoff = min(normalized_cutoff, 0.99)  # Ensure valid range
    
    b, a = butter(order, normalized_cutoff, btype='low')
    
    # Apply filter to each dimension
    filtered = np.zeros_like(data)
    for dim in range(data.shape[1]):
        # Handle NaN values
        valid_mask = ~np.isnan(data[:, dim])
        if np.sum(valid_mask) < 15:
            filtered[:, dim] = data[:, dim]
            continue
        
        # Use filtfilt for zero-phase filtering
        try:
            filtered[valid_mask, dim] = filtfilt(b, a, data[valid_mask, dim])
        except Exception:
            filtered[:, dim] = data[:, dim]
    
    return filtered


def _latest_artifact_dir(base_dir: Path) -> Path:
    """Return latest run directory."""
    candidates = [p for p in base_dir.iterdir() if p.is_dir()] if base_dir.exists() else []
    if not candidates:
        raise RuntimeError(f"No saved model runs found under '{base_dir}'.")
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
    with scalers_path.open("rb") as fh:
        scalers = pickle.load(fh)
    return scalers["obs_scaler"], scalers["act_scaler"]


def _torch_load(path: Path) -> Dict[str, Any]:
    if torch is None:
        raise RuntimeError("PyTorch is required.")
    return torch.load(path, map_location="cpu")


class DiffusionEvaluator:
    """Evaluator for Diffusion Policy with various configurations."""
    
    def __init__(
        self,
        artifact_dir: Path,
        device: str = "cpu",
    ):
        self.artifact_dir = artifact_dir
        self.device = device
        self.manifest = _load_manifest(artifact_dir)
        self.obs_scaler, self.act_scaler = _load_scalers(artifact_dir, self.manifest)
        self.window = int(self.manifest.get("sequence_window", 16))
        
    def _load_diffusion_model(
        self,
        model_name: str,
        override_horizon: Optional[int] = None,
    ) -> Tuple[Any, Dict[str, Any], bool, int]:
        """Load a diffusion model from artifacts."""
        entry = self.manifest["models"].get(model_name)
        if entry is None:
            raise RuntimeError(f"Model '{model_name}' not found in manifest.")
        
        artifact_path = self.artifact_dir / entry["path"]
        if not artifact_path.exists():
            raise RuntimeError(f"Artifact not found: {artifact_path}")
        
        state = _torch_load(artifact_path)
        config = state.get("config", {})
        
        temporal = bool(config.get("temporal", entry.get("temporal", False)))
        original_horizon = int(config.get("action_horizon", entry.get("action_horizon", 1)))
        action_horizon = override_horizon if override_horizon is not None else original_horizon
        
        # Re-create model with potentially different horizon
        diff = DiffusionPolicyBaseline(
            obs_dim=int(config.get("obs_dim", 19)),
            act_dim=int(config.get("act_dim", 9)),
            timesteps=int(config.get("timesteps", 50)),
            hidden_dim=int(config.get("hidden_dim", 256)),
            time_dim=int(config.get("time_dim", 64)),
            lr=float(config.get("lr", 1e-3)),
            batch_size=int(config.get("batch_size", 256)),
            epochs=0,
            seed=int(config.get("seed", 0)),
            temporal=temporal,
            action_horizon=action_horizon,
            device=self.device,
            log_name=f"{model_name}_eval",
        )
        
        # Load weights (only if horizon matches, otherwise we can't load directly)
        if action_horizon == original_horizon:
            diff.model.load_state_dict(state["state_dict"])
        else:
            print(f"[warn] Cannot load weights for horizon={action_horizon} (trained with {original_horizon})")
            # For fair comparison, we'll need models trained with different horizons
            
        return diff, config, temporal, action_horizon
    
    def predict_single_action(
        self,
        model: Any,
        obs: np.ndarray,
        sampler: str = "ddim",
        eta: float = 0.0,
        n_samples: int = 4,
    ) -> Tuple[np.ndarray, float]:
        """Predict using single action mode (horizon=1)."""
        model.model.eval()
        
        start_time = time.perf_counter()
        pred_scaled = model.predict(obs, n_samples=n_samples, sampler=sampler, eta=eta)
        elapsed = time.perf_counter() - start_time
        
        pred = self.act_scaler.inverse_transform(pred_scaled)
        return pred, elapsed
    
    def predict_simulated_chunking(
        self,
        model: Any,
        obs: np.ndarray,
        horizon: int,
        sampler: str = "ddim",
        eta: float = 0.0,
        n_samples: int = 4,
        use_temporal_ensembling: bool = False,
    ) -> Tuple[np.ndarray, float]:
        """Simulate action chunking with a single-action model.
        
        This simulates what would happen if we ran the model less frequently
        (every `horizon` steps) and held the action constant between runs.
        With temporal ensembling, we average overlapping predictions from recent steps.
        """
        model.model.eval()
        
        n_steps = obs.shape[0]
        predictions = []
        action_buffer: List[Tuple[int, np.ndarray]] = []  # For temporal ensembling
        last_action = None
        
        start_time = time.perf_counter()
        
        for t in range(n_steps):
            # In simulated chunking, we only run inference every `horizon` steps
            should_run_inference = (t % horizon == 0) or use_temporal_ensembling
            
            if should_run_inference:
                # Run single-action prediction
                obs_t = obs[t:t+1]
                pred_scaled = model.predict(obs_t, n_samples=n_samples, sampler=sampler, eta=eta)
                current_action = self.act_scaler.inverse_transform(pred_scaled)[0]
                
                if use_temporal_ensembling:
                    # Store this prediction for future averaging
                    action_buffer.append((t, current_action))
                    
                    # Keep only recent predictions within horizon
                    action_buffer = [(ts, act) for ts, act in action_buffer if t - ts < horizon]
                    
                    # Average all valid predictions
                    if action_buffer:
                        ensembled_action = np.mean([act for _, act in action_buffer], axis=0)
                        predictions.append(ensembled_action)
                    else:
                        predictions.append(current_action)
                else:
                    # Without TE: just store for reuse
                    last_action = current_action
                    predictions.append(last_action)
            else:
                # Reuse last action (simulating open-loop execution within chunk)
                predictions.append(last_action)
        
        elapsed = time.perf_counter() - start_time
        
        pred = np.array(predictions)
        return pred, elapsed
    
    def predict_with_chunking(
        self,
        model: Any,
        obs: np.ndarray,
        action_horizon: int,
        sampler: str = "ddim",
        eta: float = 0.0,
        use_temporal_ensembling: bool = True,
    ) -> Tuple[np.ndarray, float]:
        """Predict using action chunking with optional temporal ensembling."""
        model.model.eval()
        
        n_steps = obs.shape[0]
        act_dim = model.act_dim
        predictions = []
        action_buffer: List[np.ndarray] = []
        
        start_time = time.perf_counter()
        
        with torch.no_grad():
            for t in range(n_steps):
                obs_t = obs[t:t+1]
                obs_tensor = torch.from_numpy(obs_t.astype(np.float32)).to(model.device)
                
                # Generate full action sequence
                model_act_dim = act_dim * action_horizon
                x = torch.randn(1, model_act_dim, device=model.device)
                
                for t_inv in reversed(range(model.timesteps)):
                    t_tensor = torch.full((1,), t_inv, device=model.device, dtype=torch.long)
                    pred_noise = model.model(obs_tensor, x, t_tensor)
                    
                    alpha_hat = model.alpha_cumprod[t_inv]
                    sqrt_alpha_hat = model.sqrt_alpha_cumprod[t_inv]
                    sqrt_one_minus = model.sqrt_one_minus_alpha_cumprod[t_inv]
                    pred_x0 = (x - pred_noise * sqrt_one_minus) / sqrt_alpha_hat
                    
                    if sampler == "ddim":
                        if t_inv > 0:
                            alpha_prev = model.alpha_cumprod_prev[t_inv]
                            base = (1.0 - alpha_prev) / (1.0 - alpha_hat) * (1.0 - alpha_hat / alpha_prev)
                            base = torch.clamp(base, min=0.0)
                            sigma = eta * torch.sqrt(base)
                            noise = torch.randn_like(x) if eta > 0.0 else torch.zeros_like(x)
                            dir_coeff = torch.sqrt(torch.clamp(1.0 - alpha_prev - sigma**2, min=1e-6))
                            x = torch.sqrt(alpha_prev) * pred_x0 + dir_coeff * pred_noise + sigma * noise
                        else:
                            x = pred_x0
                    else:  # DDPM
                        coef1 = model.posterior_mean_coef1[t_inv]
                        coef2 = model.posterior_mean_coef2[t_inv]
                        mean = coef1 * pred_x0 + coef2 * x
                        if t_inv > 0:
                            noise = torch.randn_like(x)
                            var = model.posterior_variance[t_inv]
                            x = mean + torch.sqrt(torch.clamp(var, min=1e-6)) * noise
                        else:
                            x = mean
                
                # Reshape to (horizon, act_dim)
                action_seq = x.cpu().numpy().reshape(action_horizon, act_dim)
                action_buffer.append(action_seq)
                
                if use_temporal_ensembling:
                    # Average overlapping predictions
                    valid_actions = []
                    for i, buffered_seq in enumerate(action_buffer):
                        offset = t - (len(action_buffer) - 1 - i)
                        if 0 <= offset < action_horizon:
                            valid_actions.append(buffered_seq[offset])
                    
                    if valid_actions:
                        ensembled_action = np.mean(valid_actions, axis=0)
                    else:
                        ensembled_action = action_seq[0]
                    predictions.append(ensembled_action)
                else:
                    # Just use first action of chunk
                    predictions.append(action_seq[0])
                
                # Keep buffer size limited
                if len(action_buffer) > action_horizon:
                    action_buffer.pop(0)
        
        elapsed = time.perf_counter() - start_time
        
        pred_scaled = np.array(predictions)
        pred = self.act_scaler.inverse_transform(pred_scaled)
        return pred, elapsed


def evaluate_diffusion_policies(args: argparse.Namespace) -> None:
    """Main evaluation function."""
    if torch is None:
        raise RuntimeError("PyTorch is required for diffusion policy evaluation.")
    
    # Setup
    artifact_dir = Path(args.artifact_dir) if args.artifact_dir else _latest_artifact_dir(DEFAULT_ARTIFACT_ROOT)
    print(f"[info] Using artifact directory: {artifact_dir}")
    
    evaluator = DiffusionEvaluator(artifact_dir, device=args.device)
    manifest = evaluator.manifest
    
    # Load test data
    trajectories = load_dataset(args.stiffness_dir, args.stride, include_aug=False)
    manifest_tests = manifest.get("test_trajectories", [])
    desired_demo = args.eval_demo or (manifest_tests[0] if manifest_tests else None)
    
    # Find eval demo
    if desired_demo:
        eval_name = Path(desired_demo).stem
    else:
        eval_name = next((t.name for t in trajectories if t.name.endswith("_synced")), None)
    
    if eval_name is None:
        raise RuntimeError("No suitable evaluation demo found.")
    
    test_traj = next((t for t in trajectories if t.name == eval_name), None)
    if test_traj is None:
        # Try with _signaligned suffix
        eval_name_alt = eval_name.replace("_synced", "_synced_signaligned")
        test_traj = next((t for t in trajectories if t.name == eval_name_alt), None)
        if test_traj:
            eval_name = eval_name_alt
    
    if test_traj is None:
        available = [t.name for t in trajectories]
        raise RuntimeError(f"Demo '{eval_name}' not found. Available: {available[:5]}...")
    
    print(f"[info] Evaluating on demo: {eval_name}")
    print(f"[info] Demo length: {test_traj.observations.shape[0]} timesteps")
    
    test_obs = test_traj.observations
    test_act = test_traj.actions
    test_obs_s = evaluator.obs_scaler.transform(test_obs)
    
    # Build sequence dataset for temporal models
    test_scaled = scale_trajectories([test_traj], evaluator.obs_scaler, evaluator.act_scaler)
    test_offsets = compute_offsets([test_traj])
    seq_obs, _, _, seq_indices = build_sequence_dataset(
        test_scaled, [test_traj], evaluator.window, test_offsets
    )
    
    # Find diffusion models in manifest
    diffusion_models = []
    for name, entry in manifest.get("models", {}).items():
        kind = entry.get("kind", "").lower()
        if kind == "diffusion":
            diffusion_models.append(name)
    
    if not diffusion_models:
        raise RuntimeError("No diffusion models found in manifest.")
    
    print(f"\n[info] Found diffusion models: {diffusion_models}")
    
    # Evaluation configurations
    # Note: For proper chunking comparison, we simulate chunking behavior even with horizon=1 models
    # by running the model multiple times and applying receding horizon logic
    configs = [
        {"name": "Single Action", "horizon": 1, "temporal_ensemble": False, "simulate_chunking": False},
    ]
    
    # Only add simulated chunking if requested (slower)
    if args.include_chunking:
        configs.extend([
            {"name": "Sim Chunk H=4", "horizon": 4, "temporal_ensemble": False, "simulate_chunking": True},
            {"name": "Sim Chunk H=8", "horizon": 8, "temporal_ensemble": False, "simulate_chunking": True},
        ])
    
    # Results storage
    results: Dict[str, Dict[str, Dict[str, Any]]] = {}
    predictions: Dict[str, Dict[str, np.ndarray]] = {}
    predictions_filtered: Dict[str, Dict[str, np.ndarray]] = {}  # Low-pass filtered
    
    sampler = args.sampler
    eta = args.eta
    n_samples = args.n_samples
    lowpass_cutoff = args.lowpass_cutoff
    
    print(f"\n[info] Sampler: {sampler}, eta: {eta}, n_samples: {n_samples}")
    print(f"[info] Low-pass filter cutoff: {lowpass_cutoff} Hz")
    print("=" * 100)
    
    # Filter to only evaluate specific models if requested
    models_to_eval = diffusion_models
    if args.model_filter:
        filter_list = [m.strip().lower() for m in args.model_filter.split(",")]
        models_to_eval = [m for m in diffusion_models if m.lower() in filter_list]
        print(f"[info] Filtering to models: {models_to_eval}")
    
    # Evaluate each diffusion model
    for model_name in models_to_eval:
        print(f"\n{'='*80}")
        print(f"[{model_name}] Loading model...")
        
        try:
            model, config, temporal, trained_horizon = evaluator._load_diffusion_model(model_name)
        except Exception as e:
            print(f"[warn] Failed to load {model_name}: {e}")
            continue
        
        print(f"[{model_name}] Temporal: {temporal}, Trained horizon: {trained_horizon}")
        
        results[model_name] = {}
        predictions[model_name] = {}
        
        # Choose input based on temporal mode
        if temporal:
            if seq_obs.shape[0] == 0:
                print(f"[warn] No sequence data for temporal model {model_name}")
                continue
            eval_obs = seq_obs
            eval_indices = seq_indices
            target = test_act[seq_indices]
        else:
            eval_obs = test_obs_s
            eval_indices = np.arange(len(test_obs_s))
            target = test_act
        
        print(f"[{model_name}] Eval samples: {eval_obs.shape[0]}")
        
        # Test each configuration
        for cfg in configs:
            cfg_name = cfg["name"]
            horizon = cfg["horizon"]
            use_te = cfg["temporal_ensemble"]
            simulate = cfg.get("simulate_chunking", False)
            
            print(f"  [{cfg_name}] Running inference...")
            
            try:
                if horizon == 1 and not simulate:
                    # Standard single action prediction
                    pred, elapsed = evaluator.predict_single_action(
                        model, eval_obs, sampler=sampler, eta=eta, n_samples=n_samples
                    )
                elif simulate:
                    # Simulated chunking: use single-action model but apply chunking logic
                    pred, elapsed = evaluator.predict_simulated_chunking(
                        model, eval_obs, horizon, sampler=sampler, eta=eta, 
                        n_samples=n_samples, use_temporal_ensembling=use_te
                    )
                else:
                    # True action chunking (requires model trained with horizon > 1)
                    if trained_horizon != horizon:
                        print(f"    [skip] Model trained with horizon={trained_horizon}, not {horizon}")
                        continue
                    pred, elapsed = evaluator.predict_with_chunking(
                        model, eval_obs, horizon, sampler=sampler, eta=eta, use_temporal_ensembling=use_te
                    )
                
                # Compute metrics
                metrics = compute_metrics(target, pred)
                avg_time_per_step = elapsed / eval_obs.shape[0] * 1000  # ms
                
                results[model_name][cfg_name] = {
                    "rmse": metrics["rmse"],
                    "mae": metrics["mae"],
                    "r2": metrics["r2"],
                    "total_time_s": elapsed,
                    "avg_time_ms": avg_time_per_step,
                    "horizon": horizon,
                    "temporal_ensemble": use_te,
                }
                predictions[model_name][cfg_name] = pred
                
                print(f"    RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}, "
                      f"R²: {metrics['r2']:.4f}, Time: {elapsed:.2f}s ({avg_time_per_step:.2f}ms/step)")
                
                # Apply low-pass filter and compute metrics
                pred_filtered = apply_lowpass_filter(pred, cutoff_hz=lowpass_cutoff)
                metrics_filtered = compute_metrics(target, pred_filtered)
                
                cfg_name_lp = f"{cfg_name} + LP"
                results[model_name][cfg_name_lp] = {
                    "rmse": metrics_filtered["rmse"],
                    "mae": metrics_filtered["mae"],
                    "r2": metrics_filtered["r2"],
                    "total_time_s": elapsed,  # Same inference time
                    "avg_time_ms": avg_time_per_step,
                    "horizon": horizon,
                    "temporal_ensemble": use_te,
                    "lowpass": True,
                }
                predictions[model_name][cfg_name_lp] = pred_filtered
                
                print(f"    [+LP] RMSE: {metrics_filtered['rmse']:.4f}, MAE: {metrics_filtered['mae']:.4f}, "
                      f"R²: {metrics_filtered['r2']:.4f} (cutoff={lowpass_cutoff}Hz)")
                
            except Exception as e:
                print(f"    [error] {e}")
                continue
    
    # Print summary table
    print("\n" + "=" * 100)
    print("SUMMARY: Single Action vs Action Chunking Comparison")
    print("=" * 100)
    
    header = f"{'Model':<20} {'Config':<25} {'RMSE':>10} {'MAE':>10} {'R²':>10} {'Time(ms/step)':>15}"
    print(header)
    print("-" * 100)
    
    for model_name, model_results in results.items():
        for cfg_name, metrics in model_results.items():
            print(f"{model_name:<20} {cfg_name:<25} {metrics['rmse']:>10.4f} {metrics['mae']:>10.4f} "
                  f"{metrics['r2']:>10.4f} {metrics['avg_time_ms']:>15.2f}")
        print("-" * 100)
    
    # Create comparison plots
    if args.plot:
        _create_comparison_plots(
            test_act, predictions, results, 
            artifact_dir / "plots", eval_name,
            seq_indices if diffusion_models and "diffusion_t" in diffusion_models else None
        )
        
        # Create finger-wise comparison for specific models
        _create_finger_comparison_plots(
            test_act, predictions, results,
            artifact_dir / "plots", eval_name,
            seq_indices if diffusion_models and "diffusion_t" in diffusion_models else None
        )


def _create_finger_comparison_plots(
    target: np.ndarray,
    predictions: Dict[str, Dict[str, np.ndarray]],
    results: Dict[str, Dict[str, Dict[str, Any]]],
    plot_dir: Path,
    eval_name: str,
    temporal_indices: Optional[np.ndarray] = None,
) -> None:
    """Create finger-wise comparison plots for specific models (diffusion_t + LP vs diffusion_c_ddim + LP).
    
    Shows Thumb (k_th), Index (k_if), Middle (k_mf) stiffness separately for each model.
    """
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    # Action labels based on ACTION_COLUMNS
    action_labels = [
        "k_th_x", "k_th_y", "k_th_z",  # Thumb
        "k_if_x", "k_if_y", "k_if_z",  # Index
        "k_mf_x", "k_mf_y", "k_mf_z",  # Middle
    ]
    
    # Finger groupings
    fingers = {
        "Thumb (TH)": [0, 1, 2],      # k_th_x, k_th_y, k_th_z
        "Index (IF)": [3, 4, 5],      # k_if_x, k_if_y, k_if_z
        "Middle (MF)": [6, 7, 8],     # k_mf_x, k_mf_y, k_mf_z
    }
    
    # Models to compare: prefer +LP versions
    models_to_compare = []
    for model_name, model_preds in predictions.items():
        for cfg_name in model_preds.keys():
            if "LP" in cfg_name or "+ LP" in cfg_name:
                models_to_compare.append((model_name, cfg_name))
    
    # If no LP models, use Single Action
    if not models_to_compare:
        for model_name, model_preds in predictions.items():
            for cfg_name in model_preds.keys():
                if "Single Action" in cfg_name and "LP" not in cfg_name:
                    models_to_compare.append((model_name, cfg_name))
    
    if not models_to_compare:
        print("[warn] No models found for finger comparison")
        return
    
    # Color scheme for models
    model_colors = {
        "diffusion_t": "#1f77b4",      # Blue
        "diffusion_t_ddim": "#1f77b4", # Blue
        "diffusion_c": "#ff7f0e",      # Orange
        "diffusion_c_ddim": "#ff7f0e", # Orange
        "diffusion_t_h4": "#2ca02c",   # Green
        "diffusion_t_h8": "#d62728",   # Red
    }
    
    time_idx = np.arange(target.shape[0])
    
    # Create a plot for each finger comparing all models
    for finger_name, dims in fingers.items():
        n_dims = len(dims)
        fig, axes = plt.subplots(n_dims, 1, figsize=(16, 4 * n_dims), sharex=True)
        if n_dims == 1:
            axes = [axes]
        
        for i, dim in enumerate(dims):
            ax = axes[i]
            
            # Ground truth
            ax.plot(time_idx, target[:, dim], 'k-', linewidth=2.5, label='Ground Truth', alpha=0.9, zorder=10)
            
            # Plot each model's prediction
            for model_name, cfg_name in models_to_compare:
                if model_name not in predictions or cfg_name not in predictions[model_name]:
                    continue
                    
                pred = predictions[model_name][cfg_name]
                color = model_colors.get(model_name, "#7f7f7f")
                label = f"{model_name}"
                if "LP" in cfg_name:
                    label += " +LP"
                
                # Get metrics for this model
                metrics = results.get(model_name, {}).get(cfg_name, {})
                r2 = metrics.get("r2", 0)
                rmse = metrics.get("rmse", 0)
                
                # Handle temporal models with different lengths
                if pred.shape[0] < target.shape[0] and temporal_indices is not None:
                    full_pred = np.full(target.shape[0], np.nan)
                    full_pred[temporal_indices[:pred.shape[0]]] = pred[:, dim]
                    ax.plot(time_idx, full_pred, '--', linewidth=1.8, color=color,
                           label=f'{label} (R²={r2:.3f})', alpha=0.85)
                else:
                    ax.plot(time_idx[:pred.shape[0]], pred[:, dim], '--', linewidth=1.8, color=color,
                           label=f'{label} (R²={r2:.3f})', alpha=0.85)
            
            ax.set_ylabel(action_labels[dim], fontsize=11)
            ax.grid(True, linestyle=':', alpha=0.5)
            ax.legend(loc='upper right', fontsize=9)
            ax.set_title(f"{action_labels[dim]} - Stiffness Prediction", fontsize=10)
        
        axes[-1].set_xlabel("Time step", fontsize=11)
        plt.suptitle(f"{finger_name} Stiffness: Model Comparison", fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        finger_safe = finger_name.replace(" ", "_").replace("(", "").replace(")", "")
        plot_path = plot_dir / f"{eval_name}_{finger_safe}_comparison.png"
        plt.savefig(plot_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"[info] Finger comparison plot saved: {plot_path}")
    
    # Create a combined 3x3 grid plot (all fingers, all dimensions)
    fig, axes = plt.subplots(3, 3, figsize=(20, 12), sharex=True)
    
    finger_names = ["Thumb (TH)", "Index (IF)", "Middle (MF)"]
    xyz_labels = ["X", "Y", "Z"]
    
    for finger_idx, (finger_name, dims) in enumerate(fingers.items()):
        for xyz_idx, dim in enumerate(dims):
            ax = axes[finger_idx, xyz_idx]
            
            # Ground truth
            ax.plot(time_idx, target[:, dim], 'k-', linewidth=2, label='GT', alpha=0.9)
            
            # Plot each model
            for model_name, cfg_name in models_to_compare:
                if model_name not in predictions or cfg_name not in predictions[model_name]:
                    continue
                
                pred = predictions[model_name][cfg_name]
                color = model_colors.get(model_name, "#7f7f7f")
                short_name = model_name.replace("diffusion_", "diff_")
                if "LP" in cfg_name:
                    short_name += "+LP"
                
                if pred.shape[0] < target.shape[0] and temporal_indices is not None:
                    full_pred = np.full(target.shape[0], np.nan)
                    full_pred[temporal_indices[:pred.shape[0]]] = pred[:, dim]
                    ax.plot(time_idx, full_pred, '--', linewidth=1.5, color=color, label=short_name, alpha=0.8)
                else:
                    ax.plot(time_idx[:pred.shape[0]], pred[:, dim], '--', linewidth=1.5, color=color, 
                           label=short_name, alpha=0.8)
            
            ax.set_title(f"{finger_name.split()[0]} - {xyz_labels[xyz_idx]}", fontsize=10)
            ax.grid(True, linestyle=':', alpha=0.4)
            if finger_idx == 0 and xyz_idx == 2:
                ax.legend(loc='upper right', fontsize=8)
            if finger_idx == 2:
                ax.set_xlabel("Time step")
            if xyz_idx == 0:
                ax.set_ylabel("Stiffness")
    
    plt.suptitle("All Fingers Stiffness Comparison: Ground Truth vs Models (+LP)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    combined_path = plot_dir / f"{eval_name}_all_fingers_grid.png"
    plt.savefig(combined_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[info] Combined finger grid plot saved: {combined_path}")


def _create_comparison_plots(
    target: np.ndarray,
    predictions: Dict[str, Dict[str, np.ndarray]],
    results: Dict[str, Dict[str, Dict[str, Any]]],
    plot_dir: Path,
    eval_name: str,
    temporal_indices: Optional[np.ndarray] = None,
) -> None:
    """Create comparison plots."""
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Metrics comparison bar chart
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    all_labels = []
    rmse_values = []
    mae_values = []
    r2_values = []
    
    for model_name, model_results in results.items():
        for cfg_name, metrics in model_results.items():
            label = f"{model_name}\n{cfg_name}"
            all_labels.append(label)
            rmse_values.append(metrics["rmse"])
            mae_values.append(metrics["mae"])
            r2_values.append(metrics["r2"])
    
    x = np.arange(len(all_labels))
    
    axes[0].bar(x, rmse_values, color='steelblue')
    axes[0].set_ylabel("RMSE")
    axes[0].set_title("RMSE Comparison")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(all_labels, rotation=45, ha='right', fontsize=8)
    
    axes[1].bar(x, mae_values, color='darkorange')
    axes[1].set_ylabel("MAE")
    axes[1].set_title("MAE Comparison")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(all_labels, rotation=45, ha='right', fontsize=8)
    
    axes[2].bar(x, r2_values, color='forestgreen')
    axes[2].set_ylabel("R²")
    axes[2].set_title("R² Comparison")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(all_labels, rotation=45, ha='right', fontsize=8)
    
    plt.tight_layout()
    metrics_path = plot_dir / f"{eval_name}_diffusion_metrics_comparison.png"
    plt.savefig(metrics_path, dpi=200)
    plt.close()
    print(f"[info] Metrics plot saved: {metrics_path}")
    
    # 2. Inference time comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    
    time_values = []
    for model_name, model_results in results.items():
        for cfg_name, metrics in model_results.items():
            time_values.append(metrics["avg_time_ms"])
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(time_values)))
    bars = ax.bar(x, time_values, color=colors)
    ax.set_ylabel("Time (ms/step)")
    ax.set_title("Inference Time Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(all_labels, rotation=45, ha='right', fontsize=8)
    
    # Add value labels on bars
    for bar, val in zip(bars, time_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{val:.1f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    time_path = plot_dir / f"{eval_name}_diffusion_time_comparison.png"
    plt.savefig(time_path, dpi=200)
    plt.close()
    print(f"[info] Time plot saved: {time_path}")
    
    # 3. Trajectory comparison (first 3 action dimensions - one finger)
    n_dims = min(3, target.shape[1])
    fig, axes = plt.subplots(n_dims, 1, figsize=(14, 3 * n_dims), sharex=True)
    if n_dims == 1:
        axes = [axes]
    
    time_idx = np.arange(target.shape[0])
    action_labels = ["th_k1", "th_k2", "th_k3", "if_k1", "if_k2", "if_k3", "mf_k1", "mf_k2", "mf_k3"]
    
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for dim in range(n_dims):
        ax = axes[dim]
        ax.plot(time_idx, target[:, dim], 'k-', linewidth=2, label='Ground Truth', alpha=0.8)
        
        color_idx = 0
        for model_name, model_preds in predictions.items():
            for cfg_name, pred in model_preds.items():
                # Handle different lengths (temporal models have fewer predictions)
                if pred.shape[0] < target.shape[0] and temporal_indices is not None:
                    full_pred = np.full(target.shape[0], np.nan)
                    full_pred[temporal_indices[:pred.shape[0]]] = pred[:, dim]
                    ax.plot(time_idx, full_pred, linestyle='--', linewidth=1.2, 
                           color=colors[color_idx % 10], label=f'{model_name} - {cfg_name}', alpha=0.7)
                else:
                    ax.plot(time_idx[:pred.shape[0]], pred[:, dim], linestyle='--', linewidth=1.2,
                           color=colors[color_idx % 10], label=f'{model_name} - {cfg_name}', alpha=0.7)
                color_idx += 1
        
        ax.set_ylabel(action_labels[dim] if dim < len(action_labels) else f"Action {dim}")
        ax.grid(True, linestyle=':', alpha=0.5)
        ax.legend(loc='upper right', fontsize=7, ncol=2)
    
    axes[-1].set_xlabel("Time step")
    plt.suptitle("Diffusion Policy Predictions: Single Action vs Action Chunking", fontsize=12)
    plt.tight_layout()
    
    traj_path = plot_dir / f"{eval_name}_diffusion_trajectory_comparison.png"
    plt.savefig(traj_path, dpi=200)
    plt.close()
    print(f"[info] Trajectory plot saved: {traj_path}")
    
    # 4. Individual model trajectory plots (separate plot per model)
    for model_name, model_preds in predictions.items():
        for cfg_name, pred in model_preds.items():
            fig, axes = plt.subplots(n_dims, 1, figsize=(14, 3 * n_dims), sharex=True)
            if n_dims == 1:
                axes = [axes]
            
            for dim in range(n_dims):
                ax = axes[dim]
                ax.plot(time_idx, target[:, dim], 'k-', linewidth=2, label='Ground Truth', alpha=0.8)
                
                # Handle different lengths (temporal models have fewer predictions)
                if pred.shape[0] < target.shape[0] and temporal_indices is not None:
                    full_pred = np.full(target.shape[0], np.nan)
                    full_pred[temporal_indices[:pred.shape[0]]] = pred[:, dim]
                    ax.plot(time_idx, full_pred, 'r--', linewidth=1.5, label=f'{model_name}', alpha=0.9)
                else:
                    ax.plot(time_idx[:pred.shape[0]], pred[:, dim], 'r--', linewidth=1.5,
                           label=f'{model_name}', alpha=0.9)
                
                ax.set_ylabel(action_labels[dim] if dim < len(action_labels) else f"Action {dim}")
                ax.grid(True, linestyle=':', alpha=0.5)
                ax.legend(loc='upper right', fontsize=9)
            
            axes[-1].set_xlabel("Time step")
            safe_cfg = cfg_name.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "")
            plt.suptitle(f"{model_name} - {cfg_name}", fontsize=12)
            plt.tight_layout()
            
            individual_path = plot_dir / f"{eval_name}_{model_name}_{safe_cfg}_trajectory.png"
            plt.savefig(individual_path, dpi=200)
            plt.close()
            print(f"[info] Individual plot saved: {individual_path}")


def evaluate_realtime_simulation(args: argparse.Namespace) -> None:
    """Evaluate diffusion policy as if running in real-time (one observation at a time).
    
    This simulates what happens in run_policy_node.py:
    - Observations come one at a time
    - History buffer accumulates observations
    - Initial padding when history < sequence_window
    - Low-pass filter applied in real-time (IIR)
    """
    if torch is None:
        raise RuntimeError("PyTorch is required for diffusion policy evaluation.")
    
    from collections import deque
    from scipy.signal import butter, lfilter, lfilter_zi
    
    # Setup
    artifact_dir = Path(args.artifact_dir) if args.artifact_dir else _latest_artifact_dir(DEFAULT_ARTIFACT_ROOT)
    print(f"[REALTIME SIM] Using artifact directory: {artifact_dir}")
    
    evaluator = DiffusionEvaluator(artifact_dir, device=args.device)
    manifest = evaluator.manifest
    sequence_window = int(manifest.get("sequence_window", 16))
    
    # Load test data
    trajectories = load_dataset(args.stiffness_dir, args.stride, include_aug=False)
    manifest_tests = manifest.get("test_trajectories", [])
    desired_demo = args.eval_demo or (manifest_tests[0] if manifest_tests else None)
    
    if desired_demo:
        eval_name = Path(desired_demo).stem
    else:
        eval_name = next((t.name for t in trajectories if t.name.endswith("_synced")), None)
    
    if eval_name is None:
        raise RuntimeError("No suitable evaluation demo found.")
    
    test_traj = next((t for t in trajectories if t.name == eval_name), None)
    if test_traj is None:
        eval_name_alt = eval_name.replace("_synced", "_synced_signaligned")
        test_traj = next((t for t in trajectories if t.name == eval_name_alt), None)
        if test_traj:
            eval_name = eval_name_alt
    
    if test_traj is None:
        raise RuntimeError(f"Demo '{eval_name}' not found.")
    
    print(f"[REALTIME SIM] Demo: {eval_name}, length: {test_traj.observations.shape[0]} timesteps")
    print(f"[REALTIME SIM] Sequence window: {sequence_window}")
    
    test_obs = test_traj.observations
    test_act = test_traj.actions
    test_obs_s = evaluator.obs_scaler.transform(test_obs)
    
    # Find diffusion_t model
    model_name = "diffusion_t"
    if model_name not in manifest.get("models", {}):
        model_name = next((n for n in manifest["models"] if "diffusion" in n.lower()), None)
    
    if model_name is None:
        raise RuntimeError("No diffusion model found.")
    
    print(f"[REALTIME SIM] Loading model: {model_name}")
    model, config, temporal, action_horizon = evaluator._load_diffusion_model(model_name)
    model.model.eval()
    
    # Initialize real-time simulation state (mimics run_policy_node.py)
    obs_history: deque = deque(maxlen=sequence_window)
    
    # Low-pass filter setup (IIR for real-time)
    rate_hz = 50.0  # Same as launch file
    nyquist = rate_hz / 2.0
    lowpass_cutoff = args.lowpass_cutoff
    normalized_cutoff = min(lowpass_cutoff / nyquist, 0.99)
    lp_b, lp_a = butter(2, normalized_cutoff, btype='low')
    lp_zi = lfilter_zi(lp_b, lp_a)
    lp_state = np.tile(lp_zi[:, np.newaxis], (1, 9))  # (order, 9)
    
    # Simulation parameters
    n_inference_steps = args.n_inference_steps  # DDIM steps
    sampler = args.sampler
    
    print(f"[REALTIME SIM] Sampler: {sampler}, n_inference_steps: {n_inference_steps}")
    print(f"[REALTIME SIM] Low-pass filter: {lowpass_cutoff}Hz cutoff")
    print("=" * 80)
    
    # Run real-time simulation
    predictions_raw = []
    predictions_filtered = []
    inference_times = []
    
    n_steps = test_obs_s.shape[0]
    
    for t in range(n_steps):
        # 1. Add new observation to history (like run_policy_node)
        obs_history.append(test_obs_s[t].copy())
        
        # 2. Build sequence input with padding if needed
        if len(obs_history) < sequence_window:
            # Pad with first observation
            pad_count = sequence_window - len(obs_history)
            padded_hist = [obs_history[0]] * pad_count + list(obs_history)
            obs_seq = np.array(padded_hist)[np.newaxis, :, :]  # (1, seq_len, obs_dim)
        else:
            obs_seq = np.array(list(obs_history))[np.newaxis, :, :]
        
        # 3. Run inference
        start_time = time.perf_counter()
        with torch.no_grad():
            pred_scaled = model.predict(
                obs_seq, 
                n_samples=1, 
                sampler=sampler, 
                eta=0.0,
                n_inference_steps=n_inference_steps
            )
        elapsed = time.perf_counter() - start_time
        inference_times.append(elapsed * 1000)  # ms
        
        # Handle temporal model output
        if len(pred_scaled.shape) == 3:
            act_scaled = pred_scaled[0, 0, :]  # Take first action of sequence
        else:
            act_scaled = pred_scaled.reshape(-1)
        
        # 4. Inverse scale
        act_raw = evaluator.act_scaler.inverse_transform(act_scaled.reshape(1, -1))[0]
        predictions_raw.append(act_raw)
        
        # 5. Apply IIR low-pass filter (real-time, sample by sample)
        act_filtered = np.zeros(9)
        for dim in range(9):
            filtered_val, lp_state[:, dim] = lfilter(
                lp_b, lp_a, [act_raw[dim]], zi=lp_state[:, dim]
            )
            act_filtered[dim] = filtered_val[0]
        predictions_filtered.append(act_filtered)
        
        # Progress
        if (t + 1) % 100 == 0:
            avg_time = np.mean(inference_times[-100:])
            print(f"  Step {t+1}/{n_steps}, avg inference: {avg_time:.1f}ms")
    
    predictions_raw = np.array(predictions_raw)
    predictions_filtered = np.array(predictions_filtered)
    
    # Compute metrics
    metrics_raw = compute_metrics(test_act, predictions_raw)
    metrics_filtered = compute_metrics(test_act, predictions_filtered)
    
    print("\n" + "=" * 80)
    print("REALTIME SIMULATION RESULTS")
    print("=" * 80)
    print(f"{'Config':<30} {'RMSE':>10} {'MAE':>10} {'R²':>10}")
    print("-" * 80)
    print(f"{'Raw (no filter)':<30} {metrics_raw['rmse']:>10.4f} {metrics_raw['mae']:>10.4f} {metrics_raw['r2']:>10.4f}")
    print(f"{'IIR Low-pass ({:.1f}Hz)'.format(lowpass_cutoff):<30} {metrics_filtered['rmse']:>10.4f} {metrics_filtered['mae']:>10.4f} {metrics_filtered['r2']:>10.4f}")
    print("-" * 80)
    print(f"Avg inference time: {np.mean(inference_times):.1f}ms (min: {np.min(inference_times):.1f}, max: {np.max(inference_times):.1f})")
    
    # Create comparison plot
    plot_dir = artifact_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 12), sharex=True)
    
    action_labels = ["k_th_x", "k_th_y", "k_th_z", "k_if_x", "k_if_y", "k_if_z", "k_mf_x", "k_mf_y", "k_mf_z"]
    time_idx = np.arange(n_steps)
    
    for idx in range(9):
        row, col = idx // 3, idx % 3
        ax = axes[row, col]
        
        # Ground truth
        ax.plot(time_idx, test_act[:, idx], 'k-', linewidth=2, label='Ground Truth', alpha=0.9)
        
        # Raw prediction (real-time sim)
        ax.plot(time_idx, predictions_raw[:, idx], 'r--', linewidth=1.2, 
                label=f'Raw (R²={metrics_raw["r2"]:.3f})', alpha=0.7)
        
        # Filtered prediction (real-time sim)
        ax.plot(time_idx, predictions_filtered[:, idx], 'b-', linewidth=1.5, 
                label=f'LP {lowpass_cutoff}Hz (R²={metrics_filtered["r2"]:.3f})', alpha=0.8)
        
        ax.set_title(action_labels[idx], fontsize=10)
        ax.grid(True, linestyle=':', alpha=0.4)
        if idx == 2:
            ax.legend(loc='upper right', fontsize=8)
        if row == 2:
            ax.set_xlabel("Time step")
        if col == 0:
            ax.set_ylabel("Stiffness")
    
    plt.suptitle(f"Real-time Simulation: {model_name} (DDIM {n_inference_steps} steps)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    plot_path = plot_dir / f"{eval_name}_realtime_simulation.png"
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\n[info] Real-time simulation plot saved: {plot_path}")
    
    # Also create comparison: batch vs realtime
    # Load batch prediction for comparison
    print("\n[COMPARISON] Running batch prediction for comparison...")
    
    # Build proper sequence dataset for batch
    test_scaled = scale_trajectories([test_traj], evaluator.obs_scaler, evaluator.act_scaler)
    test_offsets = compute_offsets([test_traj])
    seq_obs_batch, _, _, seq_indices = build_sequence_dataset(
        test_scaled, [test_traj], sequence_window, test_offsets
    )
    
    # Batch prediction
    start_time = time.perf_counter()
    with torch.no_grad():
        batch_pred_scaled = model.predict(seq_obs_batch, n_samples=1, sampler=sampler, eta=0.0)
    batch_elapsed = time.perf_counter() - start_time
    
    if len(batch_pred_scaled.shape) == 3:
        batch_pred_scaled = batch_pred_scaled[:, 0, :]
    batch_pred = evaluator.act_scaler.inverse_transform(batch_pred_scaled)
    
    # Apply batch low-pass (offline, zero-phase)
    batch_pred_lp = apply_lowpass_filter(batch_pred, cutoff_hz=lowpass_cutoff)
    
    # Align lengths for comparison
    target_batch = test_act[seq_indices]
    metrics_batch_raw = compute_metrics(target_batch, batch_pred)
    metrics_batch_lp = compute_metrics(target_batch, batch_pred_lp)
    
    print("\n" + "=" * 80)
    print("BATCH vs REALTIME COMPARISON")
    print("=" * 80)
    print(f"{'Config':<40} {'RMSE':>10} {'MAE':>10} {'R²':>10}")
    print("-" * 80)
    print(f"{'Batch (offline, no filter)':<40} {metrics_batch_raw['rmse']:>10.4f} {metrics_batch_raw['mae']:>10.4f} {metrics_batch_raw['r2']:>10.4f}")
    print(f"{'Batch (offline, LP {:.1f}Hz zero-phase)'.format(lowpass_cutoff):<40} {metrics_batch_lp['rmse']:>10.4f} {metrics_batch_lp['mae']:>10.4f} {metrics_batch_lp['r2']:>10.4f}")
    print("-" * 40)
    print(f"{'Realtime (padded history, no filter)':<40} {metrics_raw['rmse']:>10.4f} {metrics_raw['mae']:>10.4f} {metrics_raw['r2']:>10.4f}")
    print(f"{'Realtime (padded history, IIR LP)':<40} {metrics_filtered['rmse']:>10.4f} {metrics_filtered['mae']:>10.4f} {metrics_filtered['r2']:>10.4f}")
    print("=" * 80)
    print(f"Batch total time: {batch_elapsed*1000:.1f}ms for {len(seq_obs_batch)} samples ({batch_elapsed*1000/len(seq_obs_batch):.2f}ms/sample)")
    
    # Create batch vs realtime comparison plot
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
    
    # Plot first 3 dimensions (thumb)
    for dim in range(3):
        ax = axes[dim]
        
        # Ground truth (full length)
        ax.plot(time_idx, test_act[:, dim], 'k-', linewidth=2.5, label='Ground Truth', alpha=0.9)
        
        # Batch prediction (aligned to seq_indices)
        batch_time = seq_indices
        ax.plot(batch_time, batch_pred_lp[:, dim], 'g--', linewidth=1.5, 
                label=f'Batch LP (R²={metrics_batch_lp["r2"]:.3f})', alpha=0.8)
        
        # Realtime prediction (full length)
        ax.plot(time_idx, predictions_filtered[:, dim], 'b-', linewidth=1.5, 
                label=f'Realtime LP (R²={metrics_filtered["r2"]:.3f})', alpha=0.8)
        
        ax.set_ylabel(action_labels[dim], fontsize=11)
        ax.grid(True, linestyle=':', alpha=0.5)
        ax.legend(loc='upper right', fontsize=9)
    
    axes[-1].set_xlabel("Time step", fontsize=11)
    plt.suptitle(f"Batch vs Real-time Simulation: {model_name}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    compare_path = plot_dir / f"{eval_name}_batch_vs_realtime.png"
    plt.savefig(compare_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[info] Batch vs Realtime plot saved: {compare_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Diffusion Policy: Single Action vs Action Chunking comparison"
    )
    parser.add_argument(
        "--artifact-dir",
        type=str,
        # default=None, # Select latest by default
        default="/home/songwoo/ros2_ws/icra2025/src/hri_falcon_robot_bridge/outputs/models/policy_learning_unified/artifacts/20251130_063538",
        help="Artifact directory containing trained models. Defaults to latest.",
    )
    parser.add_argument(
        "--stiffness-dir",
        type=Path,
        default=DEFAULT_STIFFNESS_DIR,
        # default="/home/songwoo/ros2_ws/icra2025/src/hri_falcon_robot_bridge/outputs/stiffness_profiles_signaligned/20251122_023936_synced_signaligned.csv",
        help="Directory containing stiffness profile CSVs.",
    )
    parser.add_argument(
        "--eval-demo",
        type=str,
        # default=None,
        default="20251122_023936_synced_signaligned",
        help="Specific demonstration to evaluate on.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=4,
        help="Stride for subsampling.",
    )
    parser.add_argument(
        "--model-filter",
        type=str,
        default=None,
        help="Comma-separated list of model names to evaluate (e.g., 'diffusion_t,diffusion_t_ddim').",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch is not None and torch.cuda.is_available() else "cpu",
        help="Device to use for inference.",
    )
    parser.add_argument(
        "--sampler",
        type=str,
        default="ddim",
        choices=["ddpm", "ddim"],
        help="Diffusion sampler to use.",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.0,
        help="Eta parameter for DDIM (0 = deterministic).",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=4,
        help="Number of samples to average for single action prediction.",
    )
    parser.add_argument(
        "--include-chunking",
        action="store_true",
        default=False,
        help="Include simulated chunking configurations in evaluation.",
    )
    parser.add_argument(
        "--lowpass-cutoff",
        type=float,
        default=1.0,
        help="Cutoff frequency for low-pass filter (Hz). Default: 1.0Hz (very smooth)",
    )
    parser.add_argument(
        "--n-inference-steps",
        type=int,
        default=10,
        help="Number of DDIM inference steps (default: 10 for fast inference).",
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        default=False,
        help="Run real-time simulation mode (one obs at a time with padding).",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        default=True,
        help="Generate comparison plots.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_false",
        dest="plot",
        help="Skip plot generation.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.realtime:
        evaluate_realtime_simulation(args)
    else:
        evaluate_diffusion_policies(args)
