#!/usr/bin/env python3
"""Benchmark conditional stiffness policies on demonstration data.
-----------------------------
#!/bin/bash
set -euo pipefail

cd /home/songwoo/ros2_ws/icra2025

echo "[1/2] Unified 시작"
python3 src/hri_falcon_robot_bridge/scripts/3_model_learning/run_stiffness_policy_benchmarks.py \
  --models all --mode unified

echo "[2/2] Per-finger 시작"
python3 src/hri_falcon_robot_bridge/scripts/3_model_learning/run_stiffness_policy_benchmarks.py \
  --models all --mode per-finger

echo "✅ 완료"
--------------------------
This script pairs the raw demonstrations under ``outputs/logs/success`` with the
low-pass stiffness reconstructions stored in ``outputs/stiffness_profiles``.
Observations ``O`` are built from force magnitudes, deformity descriptors, and end
-effector positions, while actions ``a`` are the reconstructed stiffness profiles.

Implemented policies:
- ``gmm``: samples from a Gaussian mixture model of ``p(a|o)``.
- ``gmr``: Gaussian mixture regression using the conditional expectation of ``a``.
- ``diffusion``: lightweight diffusion policy (conditional denoising).

All models operate in a standardised feature space and report RMSE/MAE/R2/NLL. Use
``--save-predictions`` to export per-sample predictions for later plotting.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, TYPE_CHECKING, Literal, cast
import random

import numpy as np  # type: ignore[import]
import pandas as pd  # type: ignore[import]
from scipy.special import logsumexp  # type: ignore[import]
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # type: ignore[import]
from sklearn.mixture import GaussianMixture  # type: ignore[import]
from sklearn.model_selection import train_test_split  # type: ignore[import]
from sklearn.preprocessing import StandardScaler  # type: ignore[import]
try:
    import yaml  # type: ignore[import]
except ImportError:  # pragma: no cover - PyYAML optional but encouraged
    yaml = None

# Robust torch / tensorboard import separation: failure of tensorboard won't nullify torch.
try:  # torch core
    import torch  # type: ignore[import]
    import torch.nn as nn  # type: ignore[import]
    import torch.nn.functional as F  # type: ignore[import]
    from torch.utils.data import DataLoader, TensorDataset  # type: ignore[import]
    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    torch = cast(Any, None)
    nn = cast(Any, None)
    F = cast(Any, None)
    DataLoader = cast(Any, None)
    TensorDataset = cast(Any, None)
    TORCH_AVAILABLE = False

try:  # tensorboard (optional)
    from torch.utils.tensorboard import SummaryWriter  # type: ignore[import]
except ImportError:  # pragma: no cover
    SummaryWriter = cast(Any, None)

if TORCH_AVAILABLE:
    class BehaviorCloningModel(nn.Module):
        def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int, depth: int):
            super().__init__()
            layers = []
            in_dim = obs_dim
            for _ in range(max(1, depth)):
                layers.append(nn.Linear(in_dim, hidden_dim))
                layers.append(nn.ReLU())
                in_dim = hidden_dim
            layers.append(nn.Linear(in_dim, act_dim))
            self.net = nn.Sequential(*layers)

        def forward(self, obs):
            return self.net(obs)

    class BehaviorCloningBaseline:
        def __init__(
            self,
            obs_dim: int,
            act_dim: int,
            hidden_dim: int = 256,
            depth: int = 3,
            lr: float = 1e-3,
            batch_size: int = 256,
            epochs: int = 200,
            weight_decay: float = 1e-4,
            device: Optional[str] = None,
            seed: int = 0,
            log_name: str = "bc",
        ):
            torch.manual_seed(seed + 1)
            self.obs_dim = obs_dim
            self.act_dim = act_dim
            self.batch_size = batch_size
            self.epochs = epochs
            self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
            self.model = BehaviorCloningModel(obs_dim, act_dim, hidden_dim, depth).to(self.device)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
            self.criterion = nn.MSELoss()
            self.log_name = log_name

        def fit(self, obs: np.ndarray, act: np.ndarray, writer: Any = None, verbose: bool = True) -> None:
            dataset = TensorDataset(
                torch.from_numpy(obs.astype(np.float32)),
                torch.from_numpy(act.astype(np.float32)),
            )
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)
            for epoch in range(1, self.epochs + 1):
                loss_accum = 0.0
                batches = 0
                self.model.train()
                for batch_obs, batch_act in loader:
                    batch_obs = batch_obs.to(self.device)
                    batch_act = batch_act.to(self.device)
                    preds = self.model(batch_obs)
                    loss = self.criterion(preds, batch_act)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    loss_accum += float(loss.detach().cpu())
                    batches += 1
                avg_loss = loss_accum / max(1, batches)
                if verbose and (epoch == 1 or epoch % 25 == 0 or epoch == self.epochs):
                    print(f"[bc] epoch {epoch:04d}/{self.epochs} - loss {avg_loss:.6f}")
                if writer is not None:
                    writer.add_scalar(f"{self.log_name}/train_loss", avg_loss, epoch)

        @torch.no_grad()
        def predict(self, obs: np.ndarray) -> np.ndarray:
            self.model.eval()
            obs_tensor = torch.from_numpy(obs.astype(np.float32)).to(self.device)
            preds = self.model(obs_tensor)
            return preds.detach().cpu().numpy()


# Resolve project root robustly, independent of current working directory.
# Script path: <repo>/src/hri_falcon_robot_bridge/scripts/3_model_learning/run_stiffness_policy_benchmarks.py
# parents[4] points to the repository root (…/icra2025)
_THIS_FILE = Path(__file__).resolve()
try:
    _PROJECT_ROOT = _THIS_FILE.parents[4]  # …/icra2025
except Exception:
    _PROJECT_ROOT = Path.cwd()
_PKG_ROOT = _THIS_FILE.parents[2]  # …/src/hri_falcon_robot_bridge
CONFIG_DIR = _PKG_ROOT / "scripts" / "3_model_learning" / "configs" / "stiffness_policy"

# Prefer outputs under the package (…/src/hri_falcon_robot_bridge/outputs) if present,
# otherwise fall back to project root (…/icra2025/outputs).
_STIFF_CANDIDATES = [
    _PKG_ROOT / "outputs" / "stiffness_profiles_signaligned",
]
_OUT_CANDIDATES = [
    _PKG_ROOT / "outputs" / "models",
    _PROJECT_ROOT / "outputs" / "models",
]


def _parse_simple_yaml(path: Path) -> Dict[str, Any]:
    """Minimal parser for simple key: value configs when PyYAML is unavailable."""
    result: Dict[str, Any] = {}
    with path.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.split("#", 1)[0].strip()
            if not key or not value:
                continue
            lower = value.lower()
            if lower in {"true", "false"}:
                parsed: Any = lower == "true"
            else:
                try:
                    parsed = int(value, 0)
                except ValueError:
                    try:
                        parsed = float(value)
                    except ValueError:
                        parsed = value
            result[key] = parsed
    return result


def _load_yaml_config(name: str) -> Dict[str, Any]:
    path = CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        return {}
    if yaml is None:
        data = _parse_simple_yaml(path)
    else:
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config {path} must contain a mapping, got {type(data).__name__}")
    return data


TRAINING_DEFAULTS = _load_yaml_config("training_defaults")
BC_CONFIG = _load_yaml_config("behavior_cloning")
DIFF_COND_CONFIG = _load_yaml_config("diffusion_conditional")
DIFF_TEMP_CONFIG = _load_yaml_config("diffusion_temporal")
GMM_CONFIG = _load_yaml_config("gmm")
IBC_CONFIG = _load_yaml_config("ibc")
LSTM_GMM_CONFIG = _load_yaml_config("lstm_gmm")
GP_CONFIG = _load_yaml_config("gp")
MDN_CONFIG = _load_yaml_config("mdn")

def _pick_first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return paths[0]

DEFAULT_STIFFNESS_DIR = _pick_first_existing(_STIFF_CANDIDATES)
DEFAULT_OUTPUT_DIR = _pick_first_existing(_OUT_CANDIDATES)
DEFAULT_TENSORBOARD_DIR = DEFAULT_OUTPUT_DIR / "tensorboard"
# Observation columns: force (s1/s2/s3) from stiffness profile + deform/ee from raw demo
OBS_COLUMNS = [
    "s1_fx",
    "s1_fy",
    "s1_fz",
    "s2_fx",
    "s2_fy",
    "s2_fz",
    "s3_fx",
    "s3_fy",
    "s3_fz",
    "deform_ecc",
    "ee_if_px",
    "ee_if_py",
    "ee_if_pz",
    "ee_mf_px",
    "ee_mf_py",
    "ee_mf_pz",
    "ee_th_px",
    "ee_th_py",
    "ee_th_pz",
]
# Action columns: thumb (th), index (if), middle (mf) finger stiffness from stiffness profile
ACTION_COLUMNS = [
    "th_k1",
    "th_k2",
    "th_k3",
    "if_k1",
    "if_k2",
    "if_k3",
    "mf_k1",
    "mf_k2",
    "mf_k3",
]

# Per-finger observation and action columns (for independent models)
FINGER_CONFIG = {
    "th": {
        "obs": ["s1_fx", "s1_fy", "s1_fz", "ee_th_px", "ee_th_py", "ee_th_pz", "deform_ecc"],
        "act": ["th_k1", "th_k2", "th_k3"],
    },
    "if": {
        "obs": ["s2_fx", "s2_fy", "s2_fz", "ee_if_px", "ee_if_py", "ee_if_pz", "deform_ecc"],
        "act": ["if_k1", "if_k2", "if_k3"],
    },
    "mf": {
        "obs": ["s3_fx", "s3_fy", "s3_fz", "ee_mf_px", "ee_mf_py", "ee_mf_pz", "deform_ecc"],
        "act": ["mf_k1", "mf_k2", "mf_k3"],
    },
}
EPS = 1e-8


@dataclass
class Trajectory:
    name: str
    observations: np.ndarray
    actions: np.ndarray


def compute_offsets(trajs: Sequence[Trajectory]) -> List[int]:
    offsets: List[int] = []
    total = 0
    for traj in trajs:
        offsets.append(total)
        total += traj.actions.shape[0]
    return offsets


def build_sequence_dataset(
    trajs_scaled: Sequence[Trajectory],
    trajs_raw: Sequence[Trajectory],
    window: int,
    offsets: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if window < 1:
        raise ValueError("window must be >= 1")
    seq_obs: List[np.ndarray] = []
    seq_act_scaled: List[np.ndarray] = []
    seq_act_raw: List[np.ndarray] = []
    indices: List[int] = []
    for idx, (traj_s, traj_r, offset) in enumerate(zip(trajs_scaled, trajs_raw, offsets)):
        obs_s = traj_s.observations
        act_s = traj_s.actions
        act_r = traj_r.actions
        length = obs_s.shape[0]
        if length < window:
            continue
        for t in range(window - 1, length):
            seq_obs.append(obs_s[t - window + 1 : t + 1])
            seq_act_scaled.append(act_s[t])
            seq_act_raw.append(act_r[t])
            indices.append(offset + t)
    if not seq_obs:
        return (
            np.zeros((0, window, trajs_scaled[0].observations.shape[1] if trajs_scaled else 0), dtype=float),
            np.zeros((0, trajs_scaled[0].actions.shape[1] if trajs_scaled else 0), dtype=float),
            np.zeros((0, trajs_raw[0].actions.shape[1] if trajs_raw else 0), dtype=float),
            np.zeros((0,), dtype=int),
        )
    return (
        np.stack(seq_obs, axis=0),
        np.stack(seq_act_scaled, axis=0),
        np.stack(seq_act_raw, axis=0),
        np.asarray(indices, dtype=int),
    )


def _resolve_stiffness_csv(stiffness_dir: Path, demo_stem: str) -> Path:
    """Resolve stiffness profile path for a given demonstration stem.

    Supports current naming ("<stem>.csv"), legacy paper suffix
    ("<stem>_paper_profile.csv"), and augmented variants
    ("<stem>_augN.csv").

    Resolution priority:
    1. Base file "<stem>.csv"
    2. Legacy "<stem>_paper_profile.csv" (backward compatibility)
    3. First augmented variant "<stem>_aug*.csv" (lowest priority fallback)

    Note: Returning a single augmented file keeps existing loader
    interface unchanged. If multiple augmented variants should be
    treated as separate trajectories, loader logic must be extended.
    """
    base = stiffness_dir / f"{demo_stem}.csv"
    if base.exists():
        return base

    legacy = stiffness_dir / f"{demo_stem}_paper_profile.csv"
    if legacy.exists():
        return legacy

    aug_matches = sorted(stiffness_dir.glob(f"{demo_stem}_aug*.csv"))
    if aug_matches:
        # Fallback to the first augmented variant if no base/legacy present
        return aug_matches[0]

    raise FileNotFoundError(
        f"Missing stiffness profile for {demo_stem}: tried base, legacy, and aug variants in {stiffness_dir}"
    )


def _load_single_demo(log_path: Path, stiffness_dir: Path, stride: int) -> Optional[Trajectory]:
    try:
        raw = pd.read_csv(log_path)
    except Exception as exc:  # pragma: no cover - IO guard
        print(f"[skip] {log_path.name}: load failed ({exc})")
        return None

    try:
        stiff = pd.read_csv(_resolve_stiffness_csv(stiffness_dir, log_path.stem))
    except Exception as exc:  # pragma: no cover - IO guard
        print(f"[skip] {log_path.name}: stiffness load failed ({exc})")
        return None

    rows = min(len(raw), len(stiff))
    if rows < 5:
        print(f"[skip] {log_path.name}: insufficient paired samples ({rows})")
        return None

    raw = raw.iloc[:rows].reset_index(drop=True)
    stiff = stiff.iloc[:rows].reset_index(drop=True)

    # Backward compatibility: if ee_if_px/py/pz is missing, use ee_px/py/pz
    for finger in ["if", "mf", "th"]:
        ee_finger_prefix = f"ee_{finger}_"
        legacy_prefix = "ee_"
        for axis in ["px", "py", "pz"]:
            new_col = f"{ee_finger_prefix}{axis}"
            old_col = f"{legacy_prefix}{axis}"
            if new_col not in raw.columns and old_col in raw.columns:
                raw[new_col] = raw[old_col]

    missing_obs = [col for col in OBS_COLUMNS if col not in raw.columns and col not in stiff.columns]
    if missing_obs:
        print(f"[skip] {log_path.name}: missing observation columns {missing_obs}")
        return None

    missing_act = [col for col in ACTION_COLUMNS if col not in stiff.columns]
    if missing_act:
        print(f"[skip] {log_path.name}: missing action columns {missing_act}")
        return None

    obs_parts: List[np.ndarray] = []
    for col in OBS_COLUMNS:
        if col in stiff.columns:
            obs_parts.append(stiff[col].to_numpy(dtype=float).reshape(-1, 1))
        else:
            obs_parts.append(raw[col].to_numpy(dtype=float).reshape(-1, 1))
    obs = np.hstack(obs_parts)

    act = stiff[ACTION_COLUMNS].to_numpy(dtype=float)

    mask = np.isfinite(obs).all(axis=1) & np.isfinite(act).all(axis=1)
    obs = obs[mask]
    act = act[mask]
    if stride > 1:
        obs = obs[::stride]
        act = act[::stride]
    if obs.shape[0] < 5:
        print(f"[skip] {log_path.name}: too few samples after filtering ({obs.shape[0]})")
        return None

    return Trajectory(name=log_path.stem, observations=obs, actions=act)


def load_dataset(stiffness_dir: Path, stride: int, include_aug: bool = False) -> List[Trajectory]:
    """Load demonstrations from stiffness_profiles directory.
    
    Each CSV contains both observations and actions. Optionally include *_augN.csv variants.
    Base files: <stem>_synced.csv or <stem>.csv
    Augmented: <stem>_synced_augN.csv
    """
    trajectories: List[Trajectory] = []
    # Collect all CSV files
    all_csvs = sorted(stiffness_dir.glob("*.csv"))
    
    # Group by base stem (remove _augN suffix)
    from collections import defaultdict
    stem_groups = defaultdict(list)
    for csv_path in all_csvs:
        stem = csv_path.stem
        # Extract base stem (remove _augN if present)
        if "_aug" in stem:
            base_stem = stem.split("_aug")[0]
        else:
            base_stem = stem
        stem_groups[base_stem].append(csv_path)
    
    for base_stem, csv_list in stem_groups.items():
        # Filter: base only or base + augmented variants
        if include_aug:
            paths_to_load = csv_list
        else:
            # Only load base (non-aug) file
            paths_to_load = [p for p in csv_list if "_aug" not in p.stem]
        
        for csv_path in paths_to_load:
            try:
                df = pd.read_csv(csv_path)
            except Exception:
                continue
            
            if len(df) < 5:
                continue
            
            # Backward compatibility for ee finger pose columns
            for finger in ["if", "mf", "th"]:
                prefix = f"ee_{finger}_"
                for axis in ["px", "py", "pz"]:
                    col = prefix + axis
                    legacy_col = "ee_" + axis
                    if col not in df.columns and legacy_col in df.columns:
                        df[col] = df[legacy_col]
            
            # Check required columns
            missing_obs = [c for c in OBS_COLUMNS if c not in df.columns]
            missing_act = [c for c in ACTION_COLUMNS if c not in df.columns]
            if missing_obs or missing_act:
                continue
            
            # Extract observations and actions
            obs = df[OBS_COLUMNS].to_numpy(dtype=float)
            act = df[ACTION_COLUMNS].to_numpy(dtype=float)
            
            # Filter NaN/Inf
            mask = np.isfinite(obs).all(axis=1) & np.isfinite(act).all(axis=1)
            obs = obs[mask]
            act = act[mask]
            
            if stride > 1:
                obs = obs[::stride]
                act = act[::stride]
            
            if obs.shape[0] < 5:
                continue
            
            trajectories.append(Trajectory(name=csv_path.stem, observations=obs, actions=act))
    
    if not trajectories:
        raise RuntimeError(f"No valid demonstrations found in {stiffness_dir}")
    return trajectories


def _load_single_demo_per_finger(
    log_path: Path, stiffness_dir: Path, stride: int, finger: str
) -> Optional[Trajectory]:
    """Load a single demo for one finger (th/if/mf) with finger-specific obs/act columns."""
    if finger not in FINGER_CONFIG:
        raise ValueError(f"Invalid finger: {finger}. Must be one of {list(FINGER_CONFIG.keys())}")
    
    obs_cols = FINGER_CONFIG[finger]["obs"]
    act_cols = FINGER_CONFIG[finger]["act"]
    
    try:
        raw = pd.read_csv(log_path)
    except Exception as exc:
        print(f"[skip] {log_path.name} ({finger}): load failed ({exc})")
        return None

    try:
        stiff = pd.read_csv(_resolve_stiffness_csv(stiffness_dir, log_path.stem))
    except Exception as exc:
        print(f"[skip] {log_path.name} ({finger}): stiffness load failed ({exc})")
        return None

    rows = min(len(raw), len(stiff))
    if rows < 5:
        print(f"[skip] {log_path.name} ({finger}): insufficient paired samples ({rows})")
        return None

    raw = raw.iloc[:rows].reset_index(drop=True)
    stiff = stiff.iloc[:rows].reset_index(drop=True)

    # Backward compatibility: if ee_if_px/py/pz is missing, use ee_px/py/pz
    ee_finger_prefix = f"ee_{finger}_"
    legacy_prefix = "ee_"
    for axis in ["px", "py", "pz"]:
        new_col = f"{ee_finger_prefix}{axis}"
        old_col = f"{legacy_prefix}{axis}"
        if new_col not in raw.columns and old_col in raw.columns:
            raw[new_col] = raw[old_col]

    # Check if all required columns exist
    missing_obs = [col for col in obs_cols if col not in raw.columns and col not in stiff.columns]
    if missing_obs:
        print(f"[skip] {log_path.name} ({finger}): missing observation columns {missing_obs}")
        return None

    missing_act = [col for col in act_cols if col not in stiff.columns]
    if missing_act:
        print(f"[skip] {log_path.name} ({finger}): missing action columns {missing_act}")
        return None

    # Build observation array
    obs_parts: List[np.ndarray] = []
    for col in obs_cols:
        if col in stiff.columns:
            obs_parts.append(stiff[col].to_numpy(dtype=float).reshape(-1, 1))
        else:
            obs_parts.append(raw[col].to_numpy(dtype=float).reshape(-1, 1))
    obs = np.hstack(obs_parts)

    # Build action array
    act = stiff[act_cols].to_numpy(dtype=float)

    # Filter invalid values
    mask = np.isfinite(obs).all(axis=1) & np.isfinite(act).all(axis=1)
    obs = obs[mask]
    act = act[mask]
    
    if stride > 1:
        obs = obs[::stride]
        act = act[::stride]
        
    if obs.shape[0] < 5:
        print(f"[skip] {log_path.name} ({finger}): too few samples after filtering ({obs.shape[0]})")
        return None

    return Trajectory(name=f"{log_path.stem}_{finger}", observations=obs, actions=act)


def load_dataset_per_finger(stiffness_dir: Path, stride: int, finger: str, include_aug: bool = False) -> List[Trajectory]:
    """Load finger-specific demonstrations from stiffness_profiles directory."""
    if finger not in FINGER_CONFIG:
        raise ValueError(f"Invalid finger: {finger}. Must be one of {list(FINGER_CONFIG.keys())}")
    
    obs_cols = FINGER_CONFIG[finger]["obs"]
    act_cols = FINGER_CONFIG[finger]["act"]
    trajectories: List[Trajectory] = []
    
    all_csvs = sorted(stiffness_dir.glob("*.csv"))
    from collections import defaultdict
    stem_groups = defaultdict(list)
    for csv_path in all_csvs:
        stem = csv_path.stem
        if "_aug" in stem:
            base_stem = stem.split("_aug")[0]
        else:
            base_stem = stem
        stem_groups[base_stem].append(csv_path)
    
    for base_stem, csv_list in stem_groups.items():
        if include_aug:
            paths_to_load = csv_list
        else:
            paths_to_load = [p for p in csv_list if "_aug" not in p.stem]
        
        for csv_path in paths_to_load:
            try:
                df = pd.read_csv(csv_path)
            except Exception:
                continue
            
            if len(df) < 5:
                continue
            
            # Backward compatibility for ee finger pose columns
            ee_prefix = f"ee_{finger}_"
            for axis in ["px", "py", "pz"]:
                new_col = f"{ee_prefix}{axis}"
                legacy_col = "ee_" + axis
                if new_col not in df.columns and legacy_col in df.columns:
                    df[new_col] = df[legacy_col]
            
            missing_obs = [c for c in obs_cols if c not in df.columns]
            missing_act = [c for c in act_cols if c not in df.columns]
            if missing_obs or missing_act:
                continue
            
            obs = df[obs_cols].to_numpy(dtype=float)
            act = df[act_cols].to_numpy(dtype=float)
            
            mask = np.isfinite(obs).all(axis=1) & np.isfinite(act).all(axis=1)
            obs = obs[mask]
            act = act[mask]
            
            if stride > 1:
                obs = obs[::stride]
                act = act[::stride]
            
            if obs.shape[0] < 5:
                continue
            
            traj_name = f"{csv_path.stem}_{finger}"
            trajectories.append(Trajectory(name=traj_name, observations=obs, actions=act))
    
    if not trajectories:
        raise RuntimeError(f"No valid demonstrations found for finger '{finger}' in {stiffness_dir}")
    return trajectories


def flatten_trajectories(trajs: Sequence[Trajectory]) -> Tuple[np.ndarray, np.ndarray]:
    obs = np.concatenate([t.observations for t in trajs], axis=0)
    act = np.concatenate([t.actions for t in trajs], axis=0)
    return obs, act


def scale_trajectories(
    trajs: Sequence[Trajectory],
    obs_scaler: StandardScaler,
    act_scaler: StandardScaler,
) -> List[Trajectory]:
    scaled: List[Trajectory] = []
    for traj in trajs:
        scaled.append(
            Trajectory(
                name=traj.name,
                observations=obs_scaler.transform(traj.observations),
                actions=act_scaler.transform(traj.actions),
            )
        )
    return scaled


def split_train_test(trajs: Sequence[Trajectory], test_size: float, seed: int) -> Tuple[List[Trajectory], List[Trajectory]]:
    if not (0.0 < test_size < 1.0):
        raise ValueError("test_size must lie in (0,1)")
    names = [t.name for t in trajs]
    train_names, test_names = train_test_split(names, test_size=test_size, random_state=seed)
    name_to_traj = {t.name: t for t in trajs}
    train = [name_to_traj[n] for n in train_names]
    test = [name_to_traj[n] for n in test_names]
    if not train or not test:
        raise RuntimeError("Train/test split produced empty partition. Adjust --test-size.")
    return train, test


class GMMConditional:
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        n_components: int,
        covariance_type: str,
        reg_covar: float,
        random_state: Optional[int] = None,
    ):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        cov_type_literal = cast(Literal["full", "tied", "diag", "spherical"], covariance_type)
        self.model = GaussianMixture(
            n_components=n_components,
            covariance_type=cov_type_literal,
            reg_covar=reg_covar,
            max_iter=512,
            n_init=4,
            init_params="kmeans",
            random_state=random_state,
        )
        self._components: List[Dict[str, np.ndarray]] = []

    def fit(self, obs: np.ndarray, act: np.ndarray) -> None:
        joint = np.hstack([obs, act])
        self.model.fit(joint)
        self._prepare_components()

    def _prepare_components(self) -> None:
        self._components.clear()
        covs = self.model.covariances_
        means = self.model.means_
        weights = self.model.weights_
        n_components = weights.shape[0]
        D = self.obs_dim + self.act_dim
        for k in range(n_components):
            mu = means[k]
            # robustly handle covariance array shapes from sklearn
            if covs.ndim == 3:
                cov = covs[k]
            elif covs.ndim == 2:
                if covs.shape[0] == n_components and covs.shape[1] == D:
                    cov = np.diag(covs[k])
                elif covs.shape == (D, D):
                    cov = covs
                else:
                    cov = np.diag(covs[k % covs.shape[0]])
            elif covs.ndim == 1:
                cov = np.eye(D) * covs[k]
            else:
                cov = np.eye(D) * 1e-3

            mu_obs = mu[: self.obs_dim]
            mu_act = mu[self.obs_dim :]
            cov_oo = cov[: self.obs_dim, : self.obs_dim] + np.eye(self.obs_dim) * 1e-6
            cov_ao = cov[self.obs_dim :, : self.obs_dim]
            cov_aa = cov[self.obs_dim :, self.obs_dim :] + np.eye(self.act_dim) * 1e-6
            try:
                chol = np.linalg.cholesky(cov_oo)
            except np.linalg.LinAlgError:
                cov_oo += np.eye(self.obs_dim) * 1e-5
                chol = np.linalg.cholesky(cov_oo)
            cov_oo_inv = np.linalg.inv(cov_oo)
            log_det = 2.0 * np.sum(np.log(np.diag(chol)))
            gain = cov_ao @ cov_oo_inv
            cond_cov = cov_aa - gain @ cov_ao.T
            cond_cov = (cond_cov + cond_cov.T) * 0.5
            self._components.append(
                {
                    "mu_obs": mu_obs,
                    "mu_act": mu_act,
                    "cov_oo": cov_oo,
                    "cov_oo_inv": cov_oo_inv,
                    "gain": gain,
                    "cond_cov": cond_cov,
                    "log_det_cov_oo": log_det,
                    "weight": weights[k],
                }
            )

    def _log_gaussian(self, obs: np.ndarray, comp: Dict[str, np.ndarray]) -> float:
        diff = obs - comp["mu_obs"]
        mahal = float(diff.T @ comp["cov_oo_inv"] @ diff)
        return -0.5 * (self.obs_dim * math.log(2.0 * math.pi) + comp["log_det_cov_oo"] + mahal)

    def _condition(self, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        log_weights: List[float] = []
        means: List[np.ndarray] = []
        covs: List[np.ndarray] = []
        for comp in self._components:
            log_prob = math.log(comp["weight"] + EPS) + self._log_gaussian(obs, comp)
            log_weights.append(log_prob)
            mean = comp["mu_act"] + comp["gain"] @ (obs - comp["mu_obs"])
            cov = comp["cond_cov"]
            cov = cov + np.eye(self.act_dim) * 1e-6
            covs.append(cov)
            means.append(mean)
        log_weights_arr = np.array(log_weights)
        log_norm = logsumexp(log_weights_arr)
        weights = np.exp(log_weights_arr - log_norm)
        means_arr = np.stack(means, axis=0)
        covs_arr = np.stack(covs, axis=0)
        return weights, means_arr, covs_arr

    def predict(self, obs: np.ndarray, mode: str = "mean", n_samples: int = 1) -> np.ndarray:
        preds: List[np.ndarray] = []
        for row in obs:
            weights, means, covs = self._condition(row)
            if mode == "mean":
                preds.append(weights @ means)
                continue
            draws = []
            for _ in range(max(1, n_samples)):
                comp_idx = np.random.choice(len(weights), p=weights)
                sample = np.random.multivariate_normal(means[comp_idx], covs[comp_idx])
                draws.append(sample)
            preds.append(np.mean(draws, axis=0))
        return np.vstack(preds)

    def nll(self, obs: np.ndarray, act: np.ndarray) -> float:
        log_probs: List[float] = []
        for row_o, row_a in zip(obs, act):
            weights, means, covs = self._condition(row_o)
            component_logs = []
            for w, mean, cov in zip(weights, means, covs):
                diff = row_a - mean
                try:
                    chol = np.linalg.cholesky(cov)
                except np.linalg.LinAlgError:
                    cov = cov + np.eye(self.act_dim) * 1e-5
                    chol = np.linalg.cholesky(cov)
                cov_inv = np.linalg.inv(cov)
                mahal = float(diff.T @ cov_inv @ diff)
                log_det = 2.0 * np.sum(np.log(np.diag(chol)))
                log_pdf = -0.5 * (self.act_dim * math.log(2.0 * math.pi) + log_det + mahal)
                component_logs.append(math.log(w + EPS) + log_pdf)
            log_probs.append(logsumexp(component_logs))
        return -float(np.mean(log_probs))

if torch is not None:
    class SinusoidalTimeEmbedding(nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            self.dim = dim

        def forward(self, timesteps):
            half = self.dim // 2
            freqs = torch.exp(
                torch.arange(half, dtype=torch.float32, device=timesteps.device)
                * -(math.log(10000.0) / max(1, half - 1))
            )
            args = timesteps.float().unsqueeze(-1) * freqs.unsqueeze(0)
            emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
            if self.dim % 2 == 1:
                emb = torch.cat([emb, torch.zeros_like(emb[..., :1])], dim=-1)
            return emb


    class ConditionalDiffusionModel(nn.Module):
        def __init__(
            self,
            obs_dim: int,
            act_dim: int,
            hidden_dim: int,
            time_dim: int,
            temporal: bool = False,
        ):
            super().__init__()
            self.temporal = temporal
            if temporal:
                self.obs_encoder = nn.GRU(obs_dim, hidden_dim, batch_first=True)
                self.obs_proj = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.SiLU(),
                )
            else:
                self.obs_encoder = nn.Sequential(
                    nn.Linear(obs_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.SiLU(),
                )
                self.obs_proj = nn.Identity()
            self.time_embed = SinusoidalTimeEmbedding(time_dim)
            self.net = nn.Sequential(
                nn.Linear(hidden_dim + act_dim + time_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, act_dim),
            )

        def forward(self, obs, noisy_action, timesteps):
            if self.temporal:
                if obs.dim() == 2:
                    obs = obs.unsqueeze(1)
                _, h = self.obs_encoder(obs)
                obs_feat = self.obs_proj(h[-1])
            else:
                obs_feat = self.obs_proj(self.obs_encoder(obs))
            time_feat = self.time_embed(timesteps)
            x = torch.cat([obs_feat, noisy_action, time_feat], dim=-1)
            return self.net(x)


    class DiffusionPolicyBaseline:
        def __init__(
            self,
            obs_dim: int,
            act_dim: int,
            timesteps: int = 50,
            hidden_dim: int = 256,
            time_dim: int = 64,
            lr: float = 1e-3,
            batch_size: int = 256,
            epochs: int = 300,
            device: Optional[str] = None,
            seed: int = 0,
            log_name: str = "diffusion",
            temporal: bool = False,
            action_horizon: int = 1,
        ):
            torch.manual_seed(seed)
            self.obs_dim = obs_dim
            self.act_dim = act_dim
            self.timesteps = timesteps
            self.batch_size = batch_size
            self.epochs = epochs
            self.action_horizon = action_horizon
            self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
            # Model output dimension: act_dim * action_horizon for chunking
            model_act_dim = act_dim * action_horizon
            self.model = ConditionalDiffusionModel(obs_dim, model_act_dim, hidden_dim, time_dim, temporal=temporal).to(self.device)
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
            self.log_name = log_name
            self.betas: Any = None
            self.alphas: Any = None
            self.alpha_cumprod: Any = None
            self.alpha_cumprod_prev: Any = None
            self.sqrt_alpha_cumprod: Any = None
            self.sqrt_one_minus_alpha_cumprod: Any = None
            self.posterior_variance: Any = None
            self.posterior_log_variance_clipped: Any = None
            self.posterior_mean_coef1: Any = None
            self.posterior_mean_coef2: Any = None
            self._build_schedule()

        def _build_schedule(self) -> None:
            betas = torch.linspace(1e-4, 0.02, self.timesteps, dtype=torch.float32)
            alphas = 1.0 - betas
            alpha_cumprod = torch.cumprod(alphas, dim=0)
            alpha_cumprod_prev = torch.cat([torch.ones(1, dtype=torch.float32), alpha_cumprod[:-1]], dim=0)
            self.register_buffer("betas", betas)
            self.register_buffer("alphas", alphas)
            self.register_buffer("alpha_cumprod", alpha_cumprod)
            self.register_buffer("alpha_cumprod_prev", alpha_cumprod_prev)
            self.register_buffer("sqrt_alpha_cumprod", torch.sqrt(alpha_cumprod))
            self.register_buffer("sqrt_one_minus_alpha_cumprod", torch.sqrt(1.0 - alpha_cumprod))
            self.register_buffer("posterior_variance", betas * (1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod))
            self.register_buffer(
                "posterior_log_variance_clipped",
                torch.log(torch.clamp(self.posterior_variance, min=1e-6)),
            )
            coef1 = betas * torch.sqrt(alpha_cumprod_prev) / (1.0 - alpha_cumprod)
            coef2 = (1.0 - alpha_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alpha_cumprod)
            self.register_buffer("posterior_mean_coef1", coef1)
            self.register_buffer("posterior_mean_coef2", coef2)

        def register_buffer(self, name: str, tensor) -> None:
            setattr(self, name, tensor.to(self.device))

        def fit(self, obs: np.ndarray, act: np.ndarray, writer: Any = None, verbose: bool = True) -> None:
            """Train diffusion policy.
            
            For action_horizon > 1 (action chunking):
            - obs: (B, obs_dim) or (B, seq_len, obs_dim) for temporal
            - act: (B, act_dim) - we create consecutive action sequences
            
            When action_horizon > 1, we slide window through actions to create
            target action chunks: [a_t, a_{t+1}, ..., a_{t+horizon-1}]
            """
            if self.action_horizon > 1:
                # Create consecutive action sequences for action chunking
                # obs[i] -> predict [act[i], act[i+1], ..., act[i+horizon-1]]
                n_samples = obs.shape[0] - self.action_horizon + 1
                if n_samples <= 0:
                    raise ValueError(f"Not enough samples for action_horizon={self.action_horizon}. "
                                     f"Need at least {self.action_horizon} samples, got {obs.shape[0]}")
                
                # Create chunked actions: (n_samples, horizon * act_dim)
                act_chunks = np.zeros((n_samples, self.action_horizon * self.act_dim), dtype=np.float32)
                for i in range(n_samples):
                    # Stack horizon consecutive actions and flatten
                    act_chunks[i] = act[i:i+self.action_horizon].flatten()
                
                # Truncate observations to match
                obs_chunked = obs[:n_samples]
                
                if verbose:
                    print(f"[info] Action chunking: {obs.shape[0]} samples -> {n_samples} chunks (horizon={self.action_horizon})")
                
                obs_tensor = torch.from_numpy(obs_chunked.astype(np.float32))
                act_tensor = torch.from_numpy(act_chunks)
            else:
                obs_tensor = torch.from_numpy(obs.astype(np.float32))
                act_tensor = torch.from_numpy(act.astype(np.float32))
            
            dataset = TensorDataset(obs_tensor, act_tensor)
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
            for epoch in range(1, self.epochs + 1):
                loss_accum = 0.0
                batches = 0
                for batch_obs, batch_act in loader:
                    batch_obs = batch_obs.to(self.device)
                    batch_act = batch_act.to(self.device)
                    t = torch.randint(0, self.timesteps, (batch_obs.size(0),), device=self.device)
                    noise = torch.randn_like(batch_act)
                    alpha_hat = self.sqrt_alpha_cumprod[t].unsqueeze(-1)
                    sigma_hat = self.sqrt_one_minus_alpha_cumprod[t].unsqueeze(-1)
                    noisy = alpha_hat * batch_act + sigma_hat * noise
                    pred_noise = self.model(batch_obs, noisy, t)
                    loss = F.mse_loss(pred_noise, noise)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    loss_accum += float(loss.detach().cpu())
                    batches += 1
                if verbose and (epoch == 1 or epoch % 25 == 0 or epoch == self.epochs):
                    avg_loss = loss_accum / max(1, batches)
                    print(f"[diffusion] epoch {epoch:04d}/{self.epochs} - loss {avg_loss:.6f}")
                if writer is not None:
                    avg_loss = loss_accum / max(1, batches)
                    writer.add_scalar(f"{self.log_name}/train_loss", avg_loss, epoch)

        @torch.no_grad()
        def predict(
            self,
            obs: np.ndarray,
            n_samples: int = 1,
            sampler: str = "ddpm",
            eta: float = 0.0,
        ) -> np.ndarray:
            self.model.eval()
            obs_tensor = torch.from_numpy(obs.astype(np.float32)).to(self.device)
            batch = obs_tensor.size(0)
            preds: List[Any] = []
            mode = sampler.lower()
            if mode not in {"ddpm", "ddim"}:
                raise ValueError(f"Unsupported sampler '{sampler}'. Choose 'ddpm' or 'ddim'.")
            eta = max(0.0, float(eta))
            model_act_dim = self.act_dim * self.action_horizon
            for _ in range(max(1, n_samples)):
                x = torch.randn(batch, model_act_dim, device=self.device)
                for t_inv in reversed(range(self.timesteps)):
                    t = torch.full((batch,), t_inv, device=self.device, dtype=torch.long)
                    pred_noise = self.model(obs_tensor, x, t)
                    alpha_hat = self.alpha_cumprod[t_inv]
                    sqrt_alpha_hat = self.sqrt_alpha_cumprod[t_inv]
                    sqrt_one_minus = self.sqrt_one_minus_alpha_cumprod[t_inv]
                    pred_x0 = (x - pred_noise * sqrt_one_minus) / sqrt_alpha_hat
                    if mode == "ddpm":
                        coef1 = self.posterior_mean_coef1[t_inv]
                        coef2 = self.posterior_mean_coef2[t_inv]
                        mean = coef1 * pred_x0 + coef2 * x
                        if t_inv > 0:
                            noise = torch.randn_like(x)
                            var = self.posterior_variance[t_inv]
                            x = mean + torch.sqrt(torch.clamp(var, min=1e-6)) * noise
                        else:
                            x = mean
                    else:  # DDIM
                        if t_inv > 0:
                            alpha_prev = self.alpha_cumprod_prev[t_inv]
                            base = (1.0 - alpha_prev) / (1.0 - alpha_hat) * (1.0 - alpha_hat / alpha_prev)
                            base = torch.clamp(base, min=0.0)
                            sigma = eta * torch.sqrt(base)
                            noise = torch.randn_like(x) if eta > 0.0 else torch.zeros_like(x)
                            dir_coeff = torch.sqrt(torch.clamp(1.0 - alpha_prev - sigma**2, min=1e-6))
                            x = torch.sqrt(alpha_prev) * pred_x0 + dir_coeff * pred_noise + sigma * noise
                        else:
                            x = pred_x0
                preds.append(x.cpu())
            stacked = torch.stack(preds, dim=0).mean(dim=0)  # (B, act_dim * horizon)
            # Reshape to (B, horizon, act_dim) and return only first action
            if self.action_horizon > 1:
                stacked_reshaped = stacked.reshape(batch, self.action_horizon, self.act_dim)
                return stacked_reshaped[:, 0, :].numpy()  # (B, act_dim)
            else:
                return stacked.numpy()


    class LSTMGMMHead(nn.Module):
        def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int, n_layers: int, n_components: int):
            super().__init__()
            self.n_components = n_components
            self.act_dim = act_dim
            self.encoder = nn.LSTM(obs_dim, hidden_dim, num_layers=max(1, n_layers), batch_first=True)
            self.hidden_to_mean = nn.Linear(hidden_dim, n_components * act_dim)
            self.hidden_to_logvar = nn.Linear(hidden_dim, n_components * act_dim)
            self.hidden_to_logits = nn.Linear(hidden_dim, n_components)

        def forward(self, seq_obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            _, (h_n, _) = self.encoder(seq_obs)
            h = h_n[-1]
            mean = self.hidden_to_mean(h).view(-1, self.n_components, self.act_dim)
            logvar = self.hidden_to_logvar(h).view(-1, self.n_components, self.act_dim)
            logits = self.hidden_to_logits(h)
            return mean, logvar, logits


    class LSTMGMMBaseline:
        def __init__(
            self,
            obs_dim: int,
            act_dim: int,
            seq_len: int,
            n_components: int = 5,
            hidden_dim: int = 256,
            n_layers: int = 1,
            lr: float = 1e-3,
            batch_size: int = 256,
            epochs: int = 200,
            device: Optional[str] = None,
            seed: int = 0,
            log_name: str = "lstm_gmm",
        ):
            torch.manual_seed(seed + 123)
            self.obs_dim = obs_dim
            self.act_dim = act_dim
            self.seq_len = seq_len
            self.n_components = n_components
            self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
            self.model = LSTMGMMHead(obs_dim, act_dim, hidden_dim, n_layers, n_components).to(self.device)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
            self.batch_size = batch_size
            self.epochs = epochs
            self.log_name = log_name

        def _negative_log_likelihood(self, mean, logvar, logits, target):
            var = logvar.exp().clamp(min=1e-6)
            diff = target.unsqueeze(1) - mean
            log_component = -0.5 * ((diff ** 2) / var + logvar + math.log(2.0 * math.pi))
            log_component = log_component.sum(dim=-1)
            log_weights = torch.log_softmax(logits, dim=-1)
            log_probs = torch.logsumexp(log_weights + log_component, dim=-1)
            return -log_probs.mean()

        def fit(self, seq_obs: np.ndarray, act: np.ndarray, writer: Any = None, verbose: bool = True) -> None:
            if seq_obs.shape[0] == 0:
                raise RuntimeError("LSTM-GMM requires at least one sequence. Increase data or reduce window.")
            dataset = TensorDataset(
                torch.from_numpy(seq_obs.astype(np.float32)),
                torch.from_numpy(act.astype(np.float32)),
            )
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
            for epoch in range(1, self.epochs + 1):
                total_loss = 0.0
                batches = 0
                for seq_batch, act_batch in loader:
                    seq_batch = seq_batch.to(self.device)
                    act_batch = act_batch.to(self.device)
                    mean, logvar, logits = self.model(seq_batch)
                    loss = self._negative_log_likelihood(mean, logvar, logits, act_batch)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    total_loss += float(loss.detach().cpu())
                    batches += 1
                avg_loss = total_loss / max(1, batches)
                if verbose and (epoch == 1 or epoch % 25 == 0 or epoch == self.epochs):
                    print(f"[lstm_gmm] epoch {epoch:04d}/{self.epochs} - loss {avg_loss:.6f}")
                if writer is not None:
                    writer.add_scalar(f"{self.log_name}/train_loss", avg_loss, epoch)

        @torch.no_grad()
        def predict(self, seq_obs: np.ndarray, mode: str = "mean", n_samples: int = 8) -> np.ndarray:
            if seq_obs.shape[0] == 0:
                return np.zeros((0, self.act_dim), dtype=float)
            self.model.eval()
            seq_tensor = torch.from_numpy(seq_obs.astype(np.float32)).to(self.device)
            mean, logvar, logits = self.model(seq_tensor)
            weights = torch.softmax(logits, dim=-1)
            if mode == "sample":
                draws: List[torch.Tensor] = []
                var = logvar.exp().clamp(min=1e-6)
                for _ in range(max(1, n_samples)):
                    comp = torch.distributions.Categorical(weights).sample()
                    comp_mean = mean[torch.arange(mean.size(0)), comp]
                    comp_std = var[torch.arange(var.size(0)), comp].sqrt()
                    sample = torch.randn_like(comp_mean) * comp_std + comp_mean
                    draws.append(sample)
                pred = torch.stack(draws, dim=0).mean(dim=0)
            else:
                pred = torch.sum(weights.unsqueeze(-1) * mean, dim=1)
            return pred.detach().cpu().numpy()


    class IBCScoreNet(nn.Module):
        def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256, depth: int = 3):
            super().__init__()
            layers: List[nn.Module] = []
            in_dim = obs_dim + act_dim
            for _ in range(max(1, depth)):
                layers.append(nn.Linear(in_dim, hidden_dim))
                layers.append(nn.SiLU())
                in_dim = hidden_dim
            layers.append(nn.Linear(in_dim, act_dim))
            self.net = nn.Sequential(*layers)

        def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
            return self.net(torch.cat([obs, act], dim=-1))


    class IBCBaseline:
        def __init__(
            self,
            obs_dim: int,
            act_dim: int,
            hidden_dim: int = 256,
            depth: int = 3,
            lr: float = 1e-3,
            batch_size: int = 256,
            epochs: int = 300,
            noise_std: float = 0.5,
            langevin_steps: int = 30,
            step_size: float = 1e-2,
            device: Optional[str] = None,
            seed: int = 0,
            log_name: str = "ibc",
        ):
            torch.manual_seed(seed + 777)
            self.obs_dim = obs_dim
            self.act_dim = act_dim
            self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
            self.model = IBCScoreNet(obs_dim, act_dim, hidden_dim, depth).to(self.device)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
            self.batch_size = batch_size
            self.epochs = epochs
            self.noise_std = noise_std
            self.langevin_steps = langevin_steps
            self.step_size = step_size
            self.log_name = log_name

        def fit(self, obs: np.ndarray, act: np.ndarray, writer: Any = None, verbose: bool = True) -> None:
            dataset = TensorDataset(
                torch.from_numpy(obs.astype(np.float32)),
                torch.from_numpy(act.astype(np.float32)),
            )
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
            noise_std = torch.tensor(self.noise_std, dtype=torch.float32, device=self.device)
            for epoch in range(1, self.epochs + 1):
                total_loss = 0.0
                batches = 0
                for obs_batch, act_batch in loader:
                    obs_batch = obs_batch.to(self.device)
                    act_batch = act_batch.to(self.device)
                    noise = torch.randn_like(act_batch) * noise_std
                    noisy_act = act_batch + noise
                    pred_noise = self.model(obs_batch, noisy_act)
                    loss = F.mse_loss(pred_noise, noise)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    total_loss += float(loss.detach().cpu())
                    batches += 1
                avg_loss = total_loss / max(1, batches)
                if verbose and (epoch == 1 or epoch % 25 == 0 or epoch == self.epochs):
                    print(f"[ibc] epoch {epoch:04d}/{self.epochs} - loss {avg_loss:.6f}")
                if writer is not None:
                    writer.add_scalar(f"{self.log_name}/train_loss", avg_loss, epoch)

        @torch.no_grad()
        def predict(self, obs: np.ndarray, n_samples: int = 1) -> np.ndarray:
            obs_tensor = torch.from_numpy(obs.astype(np.float32)).to(self.device)
            batch = obs_tensor.size(0)
            # Initialize from Gaussian prior; could be replaced by a deterministic regressor (e.g., BC) for faster convergence.
            act = torch.randn(batch, self.act_dim, device=self.device) * self.noise_std
            sigma_sq = self.noise_std ** 2
            for _ in range(max(1, self.langevin_steps)):
                # Model trained to predict added noise ε ~ N(0, σ^2); score ≈ ∇_a log p(a|obs) = -ε / σ^2
                pred_noise = self.model(obs_tensor, act)
                score = -pred_noise / sigma_sq
                act = act + self.step_size * score + math.sqrt(2 * self.step_size) * self.noise_std * torch.randn_like(act)
            return act.cpu().numpy()


class GPBaseline:
    """Gaussian Process Regression for stiffness prediction with uncertainty quantification.

    Notes on scalability:
    - Exact GPR has O(N^2) memory and O(N^3) time. We therefore cap the number of
      training points by random subsampling (subset-of-data) and perform batched prediction
      to avoid building an enormous K(X*, X_train) at once.
    """
    
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        kernel_type: str = "rbf_ard",
        length_scale_bounds: Tuple[float, float] = (0.01, 100.0),
        nu: float = 1.5,
        alpha: float = 1e-4,
        normalize_y: bool = True,
        n_restarts_optimizer: int = 5,
        # GP scalability guards
        max_train_points: int = 4000,
        subsample_strategy: str = "random",
        batch_predict_size: int = 2048,
        random_state: int = 42,
    ):
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C
        
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.max_train_points = int(max(1, max_train_points))
        self.subsample_strategy = str(subsample_strategy)
        self.batch_predict_size = int(max(1, batch_predict_size))
        self.random_state = int(random_state)
        # Coerce numeric parameters (YAML may parse scientific notation as string)
        try:
            alpha = float(alpha)
        except Exception:
            alpha = 1e-4
        if isinstance(length_scale_bounds, (list, tuple)):
            length_scale_bounds = tuple(float(x) for x in length_scale_bounds)
        try:
            nu = float(nu)
        except Exception:
            nu = 1.5
        try:
            n_restarts_optimizer = int(n_restarts_optimizer)
        except Exception:
            n_restarts_optimizer = 5

        self.normalize_y = bool(normalize_y)
        self.n_restarts = n_restarts_optimizer
        self.alpha = alpha
        
        # Build kernel
        if kernel_type == "rbf_ard":
            # Automatic Relevance Determination: separate length scale per input dimension
            kernel = C(1.0, (1e-3, 1e3)) * RBF(
                length_scale=np.ones(obs_dim),
                length_scale_bounds=length_scale_bounds,
            )
        elif kernel_type == "rbf":
            kernel = C(1.0, (1e-3, 1e3)) * RBF(
                length_scale=1.0,
                length_scale_bounds=length_scale_bounds,
            )
        elif kernel_type == "matern":
            kernel = C(1.0, (1e-3, 1e3)) * Matern(
                length_scale=np.ones(obs_dim),
                length_scale_bounds=length_scale_bounds,
                nu=nu,
            )
        else:
            raise ValueError(f"Unknown kernel_type: {kernel_type}")
        
        # One GP per action dimension
        self.gps = []
        for _ in range(act_dim):
            gp = GaussianProcessRegressor(
                kernel=kernel,
                alpha=alpha,
                normalize_y=normalize_y,
                n_restarts_optimizer=n_restarts_optimizer,
                random_state=42,
            )
            self.gps.append(gp)
    
    def fit(self, obs: np.ndarray, act: np.ndarray) -> None:
        """Fit one GP per action dimension with optional subsampling to cap memory/time."""
        N = obs.shape[0]
        if N > self.max_train_points:
            # Subsample indices once and reuse for all output dimensions
            if self.subsample_strategy == "random":
                rng = np.random.RandomState(self.random_state)
                idx = rng.choice(N, size=self.max_train_points, replace=False)
            else:
                # Fallback to random for now; can add KMeans-based coreset later
                rng = np.random.RandomState(self.random_state)
                idx = rng.choice(N, size=self.max_train_points, replace=False)
            obs_fit = obs[idx]
            act_fit = act[idx]
            print(f"[gp] subsampling {N} -> {self.max_train_points} points to bound O(N^2) memory")
        else:
            obs_fit = obs
            act_fit = act

        for i, gp in enumerate(self.gps):
            gp.fit(obs_fit, act_fit[:, i])
    
    def predict(self, obs: np.ndarray, return_std: bool = False) -> np.ndarray:
        """Predict with optional uncertainty using batching to control memory usage."""
        n = obs.shape[0]
        BS = self.batch_predict_size
        if return_std:
            means_per_dim = [np.empty((n,), dtype=np.float64) for _ in self.gps]
            stds_per_dim = [np.empty((n,), dtype=np.float64) for _ in self.gps]
            for start in range(0, n, BS):
                end = min(n, start + BS)
                Xb = obs[start:end]
                for d, gp in enumerate(self.gps):
                    mean_b, std_b = gp.predict(Xb, return_std=True)
                    means_per_dim[d][start:end] = mean_b
                    stds_per_dim[d][start:end] = std_b
            means = np.column_stack(means_per_dim)
            stds = np.column_stack(stds_per_dim)
            return means, stds
        else:
            preds_per_dim = [np.empty((n,), dtype=np.float64) for _ in self.gps]
            for start in range(0, n, BS):
                end = min(n, start + BS)
                Xb = obs[start:end]
                for d, gp in enumerate(self.gps):
                    preds_per_dim[d][start:end] = gp.predict(Xb)
            return np.column_stack(preds_per_dim)
    
    def nll(self, obs: np.ndarray, act: np.ndarray) -> float:
        """Compute test-set negative log likelihood using predictive distribution.
        
        For each test point, computes -log p(y* | x*, D) where D is the training data.
        This is the proper test NLL for comparison with other models.
        
        Note: This is computationally more expensive than the previous version which
        only returned the training log marginal likelihood -log p(y | X, theta).
        """
        total_nll = 0.0
        n_test = obs.shape[0]
        
        # Batch predictions to avoid memory issues
        BS = self.batch_predict_size
        for start in range(0, n_test, BS):
            end = min(n_test, start + BS)
            obs_batch = obs[start:end]
            act_batch = act[start:end]
            
            for i, gp in enumerate(self.gps):
                # Get predictive mean and std for this dimension
                mean_pred, std_pred = gp.predict(obs_batch, return_std=True)
                
                # Compute Gaussian log likelihood: -log N(y | mu, sigma^2)
                # = -0.5 * [(y - mu)^2 / sigma^2 + log(sigma^2) + log(2*pi)]
                variance = std_pred ** 2 + 1e-8  # Add small noise for numerical stability
                log_prob = -0.5 * (
                    ((act_batch[:, i] - mean_pred) ** 2) / variance
                    + np.log(variance)
                    + np.log(2 * np.pi)
                )
                total_nll += -np.sum(log_prob)
        
        return float(total_nll / n_test)  # Average NLL per test point


if torch is not None:
    
    class MDNHead(nn.Module):
        """Mixture Density Network head for multimodal predictions."""
        
        def __init__(
            self,
            input_dim: int,
            output_dim: int,
            n_components: int,
            covariance_type: str = "diag",
        ):
            super().__init__()
            self.output_dim = output_dim
            self.n_components = n_components
            self.covariance_type = covariance_type
            
            # Mixture weights (logits)
            self.pi_head = nn.Linear(input_dim, n_components)
            
            # Means (n_components × output_dim)
            self.mu_head = nn.Linear(input_dim, n_components * output_dim)
            
            # Covariances
            if covariance_type == "diag":
                # Log variance (n_components × output_dim)
                self.sigma_head = nn.Linear(input_dim, n_components * output_dim)
            elif covariance_type == "full":
                # Lower triangular Cholesky factor (n_components × output_dim × output_dim)
                n_tril = output_dim * (output_dim + 1) // 2
                self.sigma_head = nn.Linear(input_dim, n_components * n_tril)
            else:
                raise ValueError(f"Unknown covariance_type: {covariance_type}")
        
        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """
            Returns:
                pi: (batch, n_components) mixture weights (softmax)
                mu: (batch, n_components, output_dim) means
                sigma: (batch, n_components, output_dim) or (batch, n_components, output_dim, output_dim)
            """
            batch_size = x.shape[0]
            
            # Mixture weights
            pi = F.softmax(self.pi_head(x), dim=-1)  # (batch, K)
            
            # Means
            mu = self.mu_head(x).view(batch_size, self.n_components, self.output_dim)
            
            # Covariances
            if self.covariance_type == "diag":
                # Softplus ensures positivity
                log_sigma = self.sigma_head(x).view(batch_size, self.n_components, self.output_dim)
                sigma = F.softplus(log_sigma) + 1e-4  # (batch, K, D)
            else:  # full
                raise NotImplementedError("Full covariance not yet implemented")
            
            return pi, mu, sigma
    
    
    class MDNBaseline:
        """Mixture Density Network baseline for multimodal stiffness prediction."""
        
        def __init__(
            self,
            obs_dim: int,
            act_dim: int,
            hidden_units: List[int],
            n_components: int,
            covariance_type: str = "diag",
            activation: str = "relu",
            dropout: float = 0.1,
            learning_rate: float = 0.001,
            weight_decay: float = 0.0001,
            mixture_reg: float = 0.01,
            covariance_floor: float = 1e-4,
        ):
            self.obs_dim = obs_dim
            self.act_dim = act_dim
            self.n_components = n_components
            self.covariance_type = covariance_type
            self.lr = learning_rate
            self.weight_decay = weight_decay
            # Robust numeric coercion (config 값이 문자열로 들어오는 경우 대비)
            try:
                self.mixture_reg = float(mixture_reg)
            except Exception:
                self.mixture_reg = 0.01
            try:
                self.covariance_floor = float(covariance_floor)
            except Exception:
                self.covariance_floor = 1e-4
            
            # Build encoder
            layers: List[nn.Module] = []
            in_dim = obs_dim
            for h_dim in hidden_units:
                layers.append(nn.Linear(in_dim, h_dim))
                if activation == "relu":
                    layers.append(nn.ReLU())
                elif activation == "tanh":
                    layers.append(nn.Tanh())
                elif activation == "elu":
                    layers.append(nn.ELU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                in_dim = h_dim
            
            self.encoder = nn.Sequential(*layers)
            self.mdn_head = MDNHead(in_dim, act_dim, n_components, covariance_type)
            self.model = nn.Sequential(self.encoder, self.mdn_head)
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
            )
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
        
        def _mdn_loss(
            self,
            pi: torch.Tensor,
            mu: torch.Tensor,
            sigma: torch.Tensor,
            target: torch.Tensor,
        ) -> torch.Tensor:
            """Negative log likelihood for MDN (diagonal covariance).
            
            Treats sigma as standard deviation (std) for consistency with sampling.
            """
            # pi: (batch, K)
            # mu: (batch, K, D)
            # sigma: (batch, K, D) - interpreted as STANDARD DEVIATION
            # target: (batch, D)
            
            target = target.unsqueeze(1)  # (batch, 1, D)
            
            # Gaussian log prob per component
            diff = target - mu  # (batch, K, D)
            # Clamp sigma (std) to avoid numerical issues
            sigma_safe = torch.clamp(sigma, min=self.covariance_floor)
            # Convert std to variance for log probability calculation
            variance = sigma_safe ** 2
            log_prob = -0.5 * (
                diff**2 / variance
                + torch.log(variance)
                + math.log(2 * math.pi)
            ).sum(dim=-1)  # (batch, K)
            
            # Mixture log prob
            log_pi = torch.log(pi + 1e-8)  # (batch, K)
            log_mixture = torch.logsumexp(log_pi + log_prob, dim=-1)  # (batch,)
            
            # Entropy regularization (encourage diversity)
            entropy = -(pi * log_pi).sum(dim=-1).mean()
            
            return -log_mixture.mean() - self.mixture_reg * entropy
        
        def fit(
            self,
            obs_train: np.ndarray,
            act_train: np.ndarray,
            obs_val: np.ndarray,
            act_val: np.ndarray,
            epochs: int,
            batch_size: int,
            patience: int = 30,
            min_delta: float = 1e-4,
            writer: Optional[Any] = None,
        ) -> None:
            """Train MDN with early stopping."""
            train_dataset = TensorDataset(
                torch.FloatTensor(obs_train),
                torch.FloatTensor(act_train),
            )
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            
            obs_val_t = torch.FloatTensor(obs_val).to(self.device)
            act_val_t = torch.FloatTensor(act_val).to(self.device)
            
            best_val_loss = float("inf")
            patience_counter = 0
            
            for epoch in range(epochs):
                self.model.train()
                train_loss = 0.0
                for obs_b, act_b in train_loader:
                    obs_b = obs_b.to(self.device)
                    act_b = act_b.to(self.device)
                    
                    self.optimizer.zero_grad()
                    features = self.encoder(obs_b)
                    pi, mu, sigma = self.mdn_head(features)
                    loss = self._mdn_loss(pi, mu, sigma, act_b)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    
                    train_loss += loss.item() * obs_b.shape[0]
                
                train_loss /= len(train_dataset)
                
                # Validation
                self.model.eval()
                with torch.no_grad():
                    features_val = self.encoder(obs_val_t)
                    pi_val, mu_val, sigma_val = self.mdn_head(features_val)
                    val_loss = self._mdn_loss(pi_val, mu_val, sigma_val, act_val_t).item()
                
                if writer:
                    writer.add_scalar("MDN/train_loss", train_loss, epoch)
                    writer.add_scalar("MDN/val_loss", val_loss, epoch)
                
                # Early stopping
                if val_loss < best_val_loss - min_delta:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break
                
                if (epoch + 1) % 20 == 0:
                    print(f"  Epoch {epoch+1}/{epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
        
        def predict(self, obs: np.ndarray, mode: str = "mean") -> np.ndarray:
            """Predict with mode='mean' (mixture mean) or 'sample' (sample from mixture)."""
            self.model.eval()
            with torch.no_grad():
                obs_t = torch.FloatTensor(obs).to(self.device)
                features = self.encoder(obs_t)
                pi, mu, sigma = self.mdn_head(features)
                
                if mode == "mean":
                    # Weighted mean of components
                    pred = (pi.unsqueeze(-1) * mu).sum(dim=1)  # (batch, D)
                elif mode == "sample":
                    # Sample from mixture
                    batch_size = obs_t.shape[0]
                    # Choose component per sample
                    component_idx = torch.multinomial(pi, 1).squeeze(-1)  # (batch,)
                    # Select corresponding mu and sigma
                    mu_selected = mu[torch.arange(batch_size), component_idx]  # (batch, D)
                    sigma_selected = sigma[torch.arange(batch_size), component_idx]  # (batch, D)
                    # Sample from Gaussian
                    pred = mu_selected + sigma_selected * torch.randn_like(mu_selected)
                else:
                    raise ValueError(f"Unknown mode: {mode}")
                
                return pred.cpu().numpy()
        
        def nll(self, obs: np.ndarray, act: np.ndarray) -> float:
            """Negative log likelihood."""
            self.model.eval()
            with torch.no_grad():
                obs_t = torch.FloatTensor(obs).to(self.device)
                act_t = torch.FloatTensor(act).to(self.device)
                features = self.encoder(obs_t)
                pi, mu, sigma = self.mdn_head(features)
                loss = self._mdn_loss(pi, mu, sigma, act_t)
                return loss.item()


def compute_metrics(target: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    mse = mean_squared_error(target, pred)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(target, pred)
    r2 = r2_score(target, pred)
    return {"rmse": rmse, "mae": mae, "r2": r2}


def evaluate_gmm(
    model: GMMConditional,
    obs_test_scaled: np.ndarray,
    act_test_scaled: np.ndarray,
    mode: str,
    n_samples: int,
    act_scaler: Optional[StandardScaler] = None,
    act_test_raw: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Evaluate GMM/GMR predictions.

    Metrics are reported in the ORIGINAL (unscaled) action space when both
    ``act_scaler`` and ``act_test_raw`` are provided. Otherwise they fall back
    to the scaled space (legacy behaviour).
    Negative log-likelihood (nll) remains computed in the scaled space (model space).
    """
    pred_scaled = model.predict(obs_test_scaled, mode=mode, n_samples=n_samples)
    if act_scaler is not None and act_test_raw is not None:
        pred_raw = act_scaler.inverse_transform(pred_scaled)
        metrics = compute_metrics(act_test_raw, pred_raw)
    else:
        metrics = compute_metrics(act_test_scaled, pred_scaled)
    metrics["nll"] = model.nll(obs_test_scaled, act_test_scaled)
    return metrics


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_writer(base_dir: Optional[Path], run_name: str):
    if base_dir is None or SummaryWriter is None:
        return None
    subdir = ensure_dir(base_dir / run_name)
    return SummaryWriter(log_dir=str(subdir))


def save_predictions(
    out_path: Path,
    obs: np.ndarray,
    act_true: np.ndarray,
    act_pred: np.ndarray,
    obs_columns: Sequence[str],
    act_columns: Sequence[str],
) -> None:
    df_obs = pd.DataFrame(obs, columns=[f"obs_{c}" for c in obs_columns])
    df_true = pd.DataFrame(act_true, columns=[f"target_{c}" for c in act_columns])
    df_pred = pd.DataFrame(act_pred, columns=[f"pred_{c}" for c in act_columns])
    df = pd.concat([df_obs, df_true, df_pred], axis=1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run stiffness policy benchmarks.")
    training_defaults = TRAINING_DEFAULTS or {}

    def _first_available_int(candidates: Sequence[Any], fallback: int) -> int:
        for value in candidates:
            if value is None:
                continue
            return int(value)
        return fallback

    seed_default = int(training_defaults.get("seed", 0))
    stride_default = int(training_defaults.get("stride", 1))
    test_size_default = float(training_defaults.get("test_size", 0.25))
    sequence_window_default = _first_available_int(
        (
            training_defaults.get("sequence_window"),
            DIFF_TEMP_CONFIG.get("sequence_window"),
            LSTM_GMM_CONFIG.get("sequence_window"),
        ),
        1,
    )
    save_predictions_default = bool(training_defaults.get("save_predictions", False))
    tensorboard_default = bool(training_defaults.get("use_tensorboard", True))

    gmm_components_default = int(GMM_CONFIG.get("components", 8))
    gmm_covariance_default = GMM_CONFIG.get("covariance_type", "full")
    gmm_samples_default = int(GMM_CONFIG.get("samples", 16))
    gmm_reg_covar_default = float(GMM_CONFIG.get("reg_covar", 1e-5))

    bc_hidden_default = int(BC_CONFIG.get("hidden_dim", 256))
    bc_depth_default = int(BC_CONFIG.get("hidden_layers", 3))
    bc_batch_default = int(BC_CONFIG.get("batch_size", 256))
    bc_lr_default = float(BC_CONFIG.get("learning_rate", 1e-3))
    bc_epochs_default = int(BC_CONFIG.get("epochs", 200))
    bc_weight_decay_default = float(BC_CONFIG.get("weight_decay", 1e-4))

    diff_steps_default = int(DIFF_COND_CONFIG.get("steps", 75))
    diff_hidden_default = int(DIFF_COND_CONFIG.get("hidden_dim", 256))
    diff_batch_default = int(DIFF_COND_CONFIG.get("batch_size", 256))
    diff_lr_default = float(DIFF_COND_CONFIG.get("learning_rate", 1e-3))
    diff_epochs_default = int(DIFF_COND_CONFIG.get("epochs", 200))
    diff_eta_default = float(DIFF_COND_CONFIG.get("eta", 0.0))

    ibc_hidden_default = int(IBC_CONFIG.get("hidden_dim", 256))
    ibc_depth_default = int(IBC_CONFIG.get("hidden_layers", 3))
    ibc_batch_default = IBC_CONFIG.get("batch_size")
    ibc_lr_default = float(IBC_CONFIG.get("learning_rate", 1e-3))
    ibc_epochs_default = int(IBC_CONFIG.get("epochs", 300))
    ibc_noise_default = float(IBC_CONFIG.get("noise_std", 0.5))
    ibc_langevin_steps_default = int(IBC_CONFIG.get("langevin_steps", 30))
    ibc_step_size_default = float(IBC_CONFIG.get("langevin_step_size", 1e-2))

    lstm_components_default = int(LSTM_GMM_CONFIG.get("components", 5))
    lstm_hidden_default = int(LSTM_GMM_CONFIG.get("hidden_dim", 256))
    lstm_layers_default = int(LSTM_GMM_CONFIG.get("lstm_layers", 1))
    lstm_epochs_default = int(LSTM_GMM_CONFIG.get("epochs", 200))
    lstm_lr_default = float(LSTM_GMM_CONFIG.get("learning_rate", 1e-3))

    parser.add_argument(
        "--stiffness-dir",
        type=Path,
        default=DEFAULT_STIFFNESS_DIR,
        help="Directory with stiffness profile CSVs (contains both obs and actions)",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for benchmark artifacts")
    parser.add_argument(
        "--mode",
        type=str,
        default="unified",
        choices=["unified", "per-finger"],
        help="Training mode: 'unified' (all obs->all act, 20D->9D) or 'per-finger' (3 independent models, 8D->3D each)",
    )
    parser.add_argument(
        "--use-emg",
        action="store_true",
        help="Add EMG signals to observations (oracle experiment: 20D→28D or 8D→16D). Should achieve high R² if model architecture is correct.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="diffusion_c,diffusion_t",
        help="Comma-separated models: gmm,gmr,bc,diffusion_c,diffusion_t,diffusion_t_h4,diffusion_t_h8,lstm_gmm,ibc,gp,mdn,all (default: all)",
    )
    parser.add_argument("--test-size", type=float, default=test_size_default, help="Fraction of trajectories reserved for testing")
    parser.add_argument(
        "--eval-demo",
        type=str,
        default=None,
        help="Optional CSV stem to reserve as the only evaluation trajectory (overrides --test-size)",
    )
    parser.add_argument("--stride", type=int, default=stride_default, help="Subsample demonstrations by stride")
    parser.add_argument("--sequence-window", type=int, default=sequence_window_default, help="Temporal window length for sequence models")
    parser.add_argument("--seed", type=int, default=seed_default, help="Random seed")
    parser.add_argument("--gmm-components", type=int, default=gmm_components_default, help="Number of mixture components")
    parser.add_argument(
        "--gmm-covariance",
        type=str,
        default=gmm_covariance_default,
        choices=["full", "diag", "spherical", "tied"],
        help="Covariance type",
    )
    parser.add_argument("--gmm-samples", type=int, default=gmm_samples_default, help="Samples per query for stochastic GMM benchmark")
    parser.add_argument("--gmm-reg-covar", type=float, default=gmm_reg_covar_default, help="Regularisation term for GMM covariances")
    parser.add_argument("--diffusion-epochs", type=int, default=diff_epochs_default, help="Training epochs for the diffusion policy")
    parser.add_argument("--diffusion-steps", type=int, default=diff_steps_default, help="Diffusion timetable length")
    parser.add_argument("--diffusion-hidden", type=int, default=diff_hidden_default, help="Hidden width for the diffusion policy network")
    parser.add_argument("--diffusion-batch", type=int, default=diff_batch_default, help="Batch size for diffusion policy training")
    parser.add_argument("--diffusion-lr", type=float, default=diff_lr_default, help="Learning rate for the diffusion policy")
    parser.add_argument("--diffusion-eta", type=float, default=diff_eta_default, help="DDIM eta used for deterministic/stochastic trade-off")
    parser.add_argument("--bc-epochs", type=int, default=bc_epochs_default, help="Training epochs for behavior cloning baseline")
    parser.add_argument("--bc-hidden", type=int, default=bc_hidden_default, help="Hidden width for behavior cloning MLP")
    parser.add_argument("--bc-depth", type=int, default=bc_depth_default, help="Number of hidden layers for behavior cloning MLP")
    parser.add_argument("--bc-batch", type=int, default=bc_batch_default, help="Batch size for behavior cloning training")
    parser.add_argument("--bc-lr", type=float, default=bc_lr_default, help="Learning rate for behavior cloning baseline")
    parser.add_argument("--bc-weight-decay", type=float, default=bc_weight_decay_default, help="Weight decay for behavior cloning baseline")
    parser.add_argument("--lstm-gmm-components", type=int, default=lstm_components_default, help="Mixture components for LSTM-GMM baseline")
    parser.add_argument("--lstm-gmm-hidden", type=int, default=lstm_hidden_default, help="Hidden width for LSTM encoder")
    parser.add_argument("--lstm-gmm-layers", type=int, default=lstm_layers_default, help="Number of LSTM layers")
    parser.add_argument("--lstm-gmm-epochs", type=int, default=lstm_epochs_default, help="Training epochs for LSTM-GMM baseline")
    parser.add_argument("--lstm-gmm-lr", type=float, default=lstm_lr_default, help="Learning rate for LSTM-GMM baseline")
    parser.add_argument("--ibc-epochs", type=int, default=ibc_epochs_default, help="Training epochs for IBC baseline")
    parser.add_argument("--ibc-hidden", type=int, default=ibc_hidden_default, help="Hidden width for IBC score network")
    parser.add_argument("--ibc-depth", type=int, default=ibc_depth_default, help="Hidden layers for IBC score network")
    parser.add_argument("--ibc-lr", type=float, default=ibc_lr_default, help="Learning rate for IBC baseline")
    parser.add_argument("--ibc-noise-std", type=float, default=ibc_noise_default, help="Noise std used in IBC training and sampling")
    parser.add_argument("--ibc-langevin-steps", type=int, default=ibc_langevin_steps_default, help="Langevin iterations for IBC sampling")
    parser.add_argument("--ibc-step-size", type=float, default=ibc_step_size_default, help="Step size for IBC Langevin updates")
    parser.add_argument(
        "--ibc-batch",
        type=int,
        default=int(ibc_batch_default) if ibc_batch_default is not None else None,
        help="Batch size override for IBC baseline (defaults to its config value or --bc-batch)",
    )
    parser.set_defaults(tensorboard=tensorboard_default)
    parser.add_argument(
        "--tensorboard",
        dest="tensorboard",
        action="store_true",
        help="Enable TensorBoard logging for neural models",
    )
    parser.add_argument(
        "--no-tensorboard",
        dest="tensorboard",
        action="store_false",
        help="Disable TensorBoard logging",
    )
    parser.add_argument(
        "--tensorboard-dir",
        type=Path,
        default=DEFAULT_TENSORBOARD_DIR,
        help="Base directory for TensorBoard logs (used when --tensorboard)",
    )
    parser.set_defaults(save_predictions=save_predictions_default)
    parser.add_argument(
        "--save-predictions",
        dest="save_predictions",
        action="store_true",
        help="Persist per-sample predictions to CSV",
    )
    parser.add_argument(
        "--no-save-predictions",
        dest="save_predictions",
        action="store_false",
        help="Disable CSV export of predictions",
    )
    
    # Data augmentation flag (disk-based only)
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Include on-disk augmented stiffness profiles (*_augN.csv) as separate trajectories",
        default=True,
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="이미 존재하는 모델 아티팩트가 있으면 학습을 건너뛰고 로드하여 평가만 수행합니다.",
    )
    parser.add_argument(
        "--resume-artifact-dir",
        type=Path,
        default=None,
        help="중단된 실행을 재개할 기존 artifact 디렉터리 경로 (예: artifacts/20251119_XXXXXX).",
    )
    parser.add_argument(
        "--exclude-models",
        type=str,
        default="",
        help="실행에서 제외할 모델들 (콤마 구분). 예: gp,mdn",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    # Configure observation columns based on --use-emg flag
    global OBS_COLUMNS
    if args.use_emg:
        try:
            OBS_COLUMNS = OBS_COLUMNS_WITH_EMG  # type: ignore[name-defined]
            print(f"[info] ⚡ ORACLE MODE: Using EMG observations ({len(OBS_COLUMNS)}D)")
            print(f"[info] Expected: High R² if model architecture is correct")
        except NameError:
            print("[warn] EMG column list OBS_COLUMNS_WITH_EMG 미정의 → EMG 비활성화 후 계속 진행")
            args.use_emg = False
    else:
        print(f"[info] Standard mode: Using sensor observations ({len(OBS_COLUMNS)}D)")
    
    def set_global_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        if torch is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

    set_global_seed(int(args.seed))
    models_requested = {m.strip().lower() for m in args.models.split(",") if m.strip()}
    exclude_requested = {m.strip().lower() for m in str(getattr(args, "exclude_models", "")).split(",") if m.strip()}
    if "diffusion" in models_requested:
        models_requested.remove("diffusion")
        models_requested.add("diffusion_c")
    if "all" in models_requested or not models_requested:
        # 'all' 기본 집합에서 gp, mdn 제외 (요청사항)
        models_requested = {"gmm", "gmr", "bc", "ibc", "diffusion_c", "diffusion_t", "lstm_gmm"}
    # Add chunking variants if explicitly requested
    if "diffusion_chunking" in models_requested or "chunking" in models_requested:
        models_requested.discard("diffusion_chunking")
        models_requested.discard("chunking")
        models_requested.add("diffusion_t_h4")
        models_requested.add("diffusion_t_h8")
    # Exclude any requested heavy models
    if exclude_requested:
        before = sorted(models_requested)
        models_requested -= exclude_requested
        after = sorted(models_requested)
        removed = sorted(exclude_requested)
        print(f"[info] Excluding models: {', '.join(removed)} (remaining: {', '.join(after)})")

    # Branch based on mode
    obs_dim = len(OBS_COLUMNS)
    if args.mode == "per-finger":
        print(f"[info] Running in per-finger mode (3 independent models: th, if, mf)")
        if args.use_emg:
            print(f"[warn] Per-finger mode with EMG not implemented yet. Using unified mode.")
            args.mode = "unified"
        else:
            run_per_finger_benchmarks(args, models_requested)
            return
    
    if args.mode == "unified":
        print(f"[info] Running in unified mode (single model: {obs_dim}D obs -> 9D act)")
        run_unified_benchmarks(args, models_requested)


def _is_global_stiffness_dir(stiffness_dir: Path) -> bool:
    name = stiffness_dir.name.lower()
    full = str(stiffness_dir).lower()
    return ("global_tk" in name) or ("global_tk" in full) or (name.endswith("_global") or "_global" in name)


def _resolve_artifact_and_tb_dirs(
    base_output: Path,
    base_tb: Path,
    stiffness_dir: Path,
    timestamp: str,
    mode: str,
) -> Tuple[Path, Optional[Path]]:
    """Return artifact root and tensorboard run dir.
    
    Simple 2-way directory separation (unified vs per-finger):
        policy_learning_unified/
        policy_learning_per_finger/
    """
    mode_suffix = "per_finger" if mode == "per-finger" else "unified"
    full_dir_name = f"policy_learning_{mode_suffix}"

    # Store mode-specific results directly under `outputs/models/<policy_learning_mode>`
    policy_learning_dir = base_output / full_dir_name
    ensure_dir(policy_learning_dir)
    artifacts_root = ensure_dir(policy_learning_dir / "artifacts" / timestamp)
    tb_parent = policy_learning_dir / "tensorboard"

    tb_run = ensure_dir(tb_parent / timestamp) if SummaryWriter is not None else None
    return artifacts_root, tb_run


def run_unified_benchmarks(args, models_requested: set) -> None:
    """Original unified training: all obs -> all act (20D -> 9D)."""
    def set_global_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        if torch is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

    trajectories = load_dataset(args.stiffness_dir, stride=max(1, args.stride), include_aug=args.augment)

    evaluation_meta: Dict[str, Any]
    if args.eval_demo:
        eval_name = args.eval_demo
        if eval_name.endswith(".csv"):
            eval_name = Path(eval_name).stem
        candidate = next((t for t in trajectories if t.name == eval_name), None)
        if candidate is None:
            available = ", ".join(sorted(t.name for t in trajectories))
            raise RuntimeError(
                f"Requested eval demo '{args.eval_demo}' not found. Available trajectories: {available}"
            )
        train_traj = [t for t in trajectories if t.name != candidate.name]
        if not train_traj:
            raise RuntimeError("Need at least one trajectory for training when using --eval-demo.")
        test_traj = [candidate]
        evaluation_meta = {"mode": "single_demo", "demo": candidate.name}
        print(f"[info] reserving '{candidate.name}' as evaluation trajectory")
    else:
        train_traj, test_traj = split_train_test(trajectories, args.test_size, args.seed)
        evaluation_meta = {"mode": "random_split", "test_size": args.test_size, "seed": args.seed}

    train_obs, train_act = flatten_trajectories(train_traj)
    test_obs, test_act = flatten_trajectories(test_traj)

    obs_scaler = StandardScaler()
    act_scaler = StandardScaler()
    train_obs_s = obs_scaler.fit_transform(train_obs)
    test_obs_s = obs_scaler.transform(test_obs)
    train_act_s = act_scaler.fit_transform(train_act)
    test_act_s = act_scaler.transform(test_act)

    train_traj_scaled = scale_trajectories(train_traj, obs_scaler, act_scaler)
    test_traj_scaled = scale_trajectories(test_traj, obs_scaler, act_scaler)

    window = max(1, args.sequence_window)
    train_offsets = compute_offsets(train_traj)
    test_offsets = compute_offsets(test_traj)
    train_seq_obs, train_seq_act_s, train_seq_act_raw, _ = build_sequence_dataset(
        train_traj_scaled,
        train_traj,
        window,
        train_offsets,
    )
    test_seq_obs, test_seq_act_s, test_seq_act_raw, test_seq_indices = build_sequence_dataset(
        test_traj_scaled,
        test_traj,
        window,
        test_offsets,
    )

    results: Dict[str, Dict[str, float]] = {}
    diffusion_eta = max(0.0, float(args.diffusion_eta))
    run_tensorboard_dir: Optional[Path] = None
    ensure_dir(args.output_dir)
    resume_dir: Optional[Path] = args.resume_artifact_dir
    if resume_dir is not None:
        if not resume_dir.exists():
            raise RuntimeError(f"--resume-artifact-dir not found: {resume_dir}")
        artifacts_root = resume_dir
        timestamp = artifacts_root.name
        print(f"[resume] Using existing artifact directory: {artifacts_root}")
        scalers_path = artifacts_root / "scalers.pkl"
        if not scalers_path.exists():
            raise RuntimeError(f"Missing scalers.pkl in resume dir: {scalers_path}")
        with scalers_path.open("rb") as fh:
            scalers_loaded = pickle.load(fh)
        obs_scaler = scalers_loaded["obs_scaler"]
        act_scaler = scalers_loaded["act_scaler"]
        # 재계산(안전): 이미 있는 스케일러로 테스트셋 transform 보장
        test_obs_s = obs_scaler.transform(test_obs)
        test_act_s = act_scaler.transform(test_act)
        train_traj_scaled = scale_trajectories(train_traj, obs_scaler, act_scaler)
        test_traj_scaled = scale_trajectories(test_traj, obs_scaler, act_scaler)
        # TensorBoard 재시작은 생략 (기존 로그 유지)
        run_tensorboard_dir = None
        manifest_path_existing = artifacts_root / "manifest.json"
        if manifest_path_existing.exists():
            with manifest_path_existing.open("r", encoding="utf-8") as fh:
                manifest = json.load(fh)
            # timestamp 동기화
            manifest["timestamp"] = timestamp
            print(f"[resume] Loaded manifest models: {', '.join(manifest.get('models', {}).keys())}")
        else:
            manifest = {
                "timestamp": timestamp,
                "scalers": "scalers.pkl",
                "sequence_window": window,
                "models": {},
                "train_trajectories": [t.name for t in train_traj],
                "test_trajectories": [t.name for t in test_traj],
                "obs_columns": OBS_COLUMNS,
                "action_columns": ACTION_COLUMNS,
            }
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        if args.tensorboard:
            if torch is None or SummaryWriter is None:
                print("[warn] tensorboard requested but unavailable; proceeding without logs.")
                args.tensorboard = False
        artifacts_root, run_tensorboard_dir = _resolve_artifact_and_tb_dirs(
            base_output=args.output_dir,
            base_tb=args.tensorboard_dir,
            stiffness_dir=args.stiffness_dir,
            timestamp=timestamp,
            mode="unified",
        )
        scalers_path = artifacts_root / "scalers.pkl"
        with scalers_path.open("wb") as fh:
            pickle.dump({"obs_scaler": obs_scaler, "act_scaler": act_scaler}, fh)
        manifest: Dict[str, Any] = {
            "timestamp": timestamp,
            "scalers": scalers_path.name,
            "sequence_window": window,
            "models": {},
            "train_trajectories": [t.name for t in train_traj],
            "test_trajectories": [t.name for t in test_traj],
            "obs_columns": OBS_COLUMNS,
            "action_columns": ACTION_COLUMNS,
        }

    if {"gmm", "gmr"} & models_requested:
        gmm_artifact = artifacts_root / "gmm.pkl"
        if args.skip_existing and gmm_artifact.exists():
            print(f"[skip] GMM artifact exists → load {gmm_artifact.name}")
            with gmm_artifact.open("rb") as fh:
                gmm_model = pickle.load(fh)
        else:
            print("[info] training Gaussian mixture on joint space ...")
            gmm_model = GMMConditional(
                obs_dim=train_obs_s.shape[1],
                act_dim=train_act_s.shape[1],
                n_components=args.gmm_components,
                covariance_type=args.gmm_covariance,
                reg_covar=args.gmm_reg_covar,
            )
            gmm_model.fit(train_obs_s, train_act_s)
            with gmm_artifact.open("wb") as fh:
                pickle.dump(gmm_model, fh)

        if "gmr" in models_requested:
            metrics = evaluate_gmm(
                gmm_model,
                test_obs_s,
                test_act_s,
                mode="mean",
                n_samples=1,
                act_scaler=act_scaler,
                act_test_raw=test_act,
            )
            results["gmr"] = metrics
            print(
                f"[gmr] rmse={metrics['rmse']:.4f} mae={metrics['mae']:.4f} "
                f"r2={metrics['r2']:.4f} nll={metrics['nll']:.4f}"
            )
            preds = gmm_model.predict(test_obs_s, mode="mean")
            preds_raw = act_scaler.inverse_transform(preds)
            if args.save_predictions:
                save_predictions(
                    args.output_dir / f"gmr_predictions_{timestamp}.csv",
                    test_obs,
                    test_act,
                    preds_raw,
                    OBS_COLUMNS,
                    ACTION_COLUMNS,
                )
            manifest["models"]["gmr"] = {
                "kind": "gmr",
                "path": gmm_artifact.name,
            }

        if "gmm" in models_requested:
            metrics = evaluate_gmm(
                gmm_model,
                test_obs_s,
                test_act_s,
                mode="sample",
                n_samples=args.gmm_samples,
                act_scaler=act_scaler,
                act_test_raw=test_act,
            )
            results["gmm"] = metrics
            print(
                f"[gmm] rmse={metrics['rmse']:.4f} mae={metrics['mae']:.4f} "
                f"r2={metrics['r2']:.4f} nll={metrics['nll']:.4f}"
            )
            preds = gmm_model.predict(test_obs_s, mode="sample", n_samples=args.gmm_samples)
            preds_raw = act_scaler.inverse_transform(preds)
            if args.save_predictions:
                save_predictions(
                    args.output_dir / f"gmm_predictions_{timestamp}.csv",
                    test_obs,
                    test_act,
                    preds_raw,
                    OBS_COLUMNS,
                    ACTION_COLUMNS,
                )
            manifest["models"]["gmm"] = {
                "kind": "gmm",
                "path": gmm_artifact.name,
                "mode": "sample",
                "n_samples": args.gmm_samples,
                "n_components": args.gmm_components,
                "covariance_type": args.gmm_covariance,
            }

    if "bc" in models_requested:
        if torch is None:
            raise RuntimeError("Behavior cloning baseline requires PyTorch.")
        bc_artifact = artifacts_root / "bc.pt"
        bc_writer = None
        if args.skip_existing and bc_artifact.exists():
            print(f"[skip] BC artifact exists → load {bc_artifact.name}")
            checkpoint = torch.load(bc_artifact, map_location="cpu")
            bc_config = checkpoint.get("config", {})
            bc = BehaviorCloningBaseline(
                obs_dim=bc_config.get("obs_dim", train_obs_s.shape[1]),
                act_dim=bc_config.get("act_dim", train_act_s.shape[1]),
                hidden_dim=bc_config.get("hidden_dim", args.bc_hidden),
                depth=bc_config.get("depth", args.bc_depth),
                lr=bc_config.get("lr", args.bc_lr),
                batch_size=bc_config.get("batch_size", args.bc_batch),
                epochs=0,
                weight_decay=bc_config.get("weight_decay", args.bc_weight_decay),
                seed=bc_config.get("seed", args.seed),
                log_name="bc",
            )
            bc.model.load_state_dict(checkpoint["state_dict"])  # type: ignore
        else:
            print("[info] training behavior cloning baseline ...")
            bc_config = {
                "obs_dim": train_obs_s.shape[1],
                "act_dim": train_act_s.shape[1],
                "hidden_dim": args.bc_hidden,
                "depth": args.bc_depth,
                "lr": args.bc_lr,
                "batch_size": args.bc_batch,
                "epochs": args.bc_epochs,
                "weight_decay": args.bc_weight_decay,
                "seed": args.seed,
            }
            bc = BehaviorCloningBaseline(
                obs_dim=bc_config["obs_dim"],
                act_dim=bc_config["act_dim"],
                hidden_dim=bc_config["hidden_dim"],
                depth=bc_config["depth"],
                lr=bc_config["lr"],
                batch_size=bc_config["batch_size"],
                epochs=bc_config["epochs"],
                weight_decay=bc_config["weight_decay"],
                seed=bc_config["seed"],
                log_name="bc",
            )
            bc_writer = make_writer(run_tensorboard_dir, "bc")
            bc.fit(train_obs_s, train_act_s, writer=bc_writer)
            torch.save({"config": bc_config, "state_dict": {k: v.cpu() for k, v in bc.model.state_dict().items()}}, bc_artifact)
            if bc_writer is not None:
                bc_writer.flush(); bc_writer.close()
        manifest["models"]["bc"] = {
            "kind": "bc",
            "path": bc_artifact.name,
        }
        pred_scaled_bc = bc.predict(test_obs_s)
        pred_bc = act_scaler.inverse_transform(pred_scaled_bc)
        metrics_bc = compute_metrics(test_act, pred_bc)
        metrics_bc["nll"] = float("nan")
        results["bc"] = metrics_bc
        print(
            f"[bc] rmse={metrics_bc['rmse']:.4f} "
            f"mae={metrics_bc['mae']:.4f} r2={metrics_bc['r2']:.4f}"
        )
        if args.save_predictions:
            save_predictions(
                args.output_dir / f"bc_predictions_{timestamp}.csv",
                test_obs,
                test_act,
                pred_bc,
                OBS_COLUMNS,
                ACTION_COLUMNS,
            )
        if bc_writer is not None:
            bc_writer.flush()
            bc_writer.close()

    if "ibc" in models_requested:
        if torch is None:
            raise RuntimeError("IBC baseline requires PyTorch.")
        ibc_artifact = artifacts_root / "ibc.pt"
        ibc_writer = None
        if args.skip_existing and ibc_artifact.exists():
            print(f"[skip] IBC artifact exists → load {ibc_artifact.name}")
            checkpoint = torch.load(ibc_artifact, map_location="cpu")
            ibc_config = checkpoint.get("config", {})
            ibc_batch = ibc_config.get("batch_size", args.ibc_batch if args.ibc_batch is not None else args.bc_batch)
            ibc = IBCBaseline(
                obs_dim=ibc_config.get("obs_dim", train_obs_s.shape[1]),
                act_dim=ibc_config.get("act_dim", train_act_s.shape[1]),
                hidden_dim=ibc_config.get("hidden_dim", args.ibc_hidden),
                depth=ibc_config.get("depth", args.ibc_depth),
                lr=ibc_config.get("lr", args.ibc_lr),
                batch_size=ibc_batch,
                epochs=0,
                noise_std=ibc_config.get("noise_std", args.ibc_noise_std),
                langevin_steps=ibc_config.get("langevin_steps", args.ibc_langevin_steps),
                step_size=ibc_config.get("step_size", args.ibc_step_size),
                seed=ibc_config.get("seed", args.seed),
                log_name="ibc",
            )
            ibc.model.load_state_dict(checkpoint["state_dict"])  # type: ignore
        else:
            print("[info] training IBC baseline ...")
            ibc_batch = args.ibc_batch if args.ibc_batch is not None else args.bc_batch
            ibc_config = {
                "obs_dim": train_obs_s.shape[1],
                "act_dim": train_act_s.shape[1],
                "hidden_dim": args.ibc_hidden,
                "depth": args.ibc_depth,
                "lr": args.ibc_lr,
                "batch_size": ibc_batch,
                "epochs": args.ibc_epochs,
                "noise_std": args.ibc_noise_std,
                "langevin_steps": args.ibc_langevin_steps,
                "step_size": args.ibc_step_size,
                "seed": args.seed,
            }
            ibc = IBCBaseline(
                obs_dim=ibc_config["obs_dim"],
                act_dim=ibc_config["act_dim"],
                hidden_dim=ibc_config["hidden_dim"],
                depth=ibc_config["depth"],
                lr=ibc_config["lr"],
                batch_size=ibc_config["batch_size"],
                epochs=ibc_config["epochs"],
                noise_std=ibc_config["noise_std"],
                langevin_steps=ibc_config["langevin_steps"],
                step_size=ibc_config["step_size"],
                seed=ibc_config["seed"],
                log_name="ibc",
            )
            ibc_writer = make_writer(run_tensorboard_dir, "ibc")
            ibc.fit(train_obs_s, train_act_s, writer=ibc_writer)
            torch.save({"config": ibc_config, "state_dict": {k: v.cpu() for k, v in ibc.model.state_dict().items()}}, ibc_artifact)
            if ibc_writer is not None:
                ibc_writer.flush(); ibc_writer.close()
        manifest["models"]["ibc"] = {
            "kind": "ibc",
            "path": ibc_artifact.name,
        }
        pred_scaled_ibc = ibc.predict(test_obs_s)
        pred_ibc = act_scaler.inverse_transform(pred_scaled_ibc)
        metrics_ibc = compute_metrics(test_act, pred_ibc)
        metrics_ibc["nll"] = float("nan")
        results["ibc"] = metrics_ibc
        print(
            f"[ibc] rmse={metrics_ibc['rmse']:.4f} "
            f"mae={metrics_ibc['mae']:.4f} r2={metrics_ibc['r2']:.4f}"
        )
        if args.save_predictions:
            save_predictions(
                args.output_dir / f"ibc_predictions_{timestamp}.csv",
                test_obs,
                test_act,
                pred_ibc,
                OBS_COLUMNS,
                ACTION_COLUMNS,
            )
        if ibc_writer is not None:
            ibc_writer.flush()
            ibc_writer.close()

    # GP Baseline (Gaussian Process Regression)
    if "gp" in models_requested:
        print("[info] training GP baseline ...")
        gp_config = GP_CONFIG or {}
        # Reduce GP training time: fewer restarts and smaller subsample for large datasets
        n_samples = train_obs.shape[0]
        max_gp_points = min(2000, int(gp_config.get("max_train_points", 4000)))
        n_restarts = 0 if n_samples > 100000 else gp_config.get("n_restarts_optimizer", 2)
        
        print(f"[gp] using max_train_points={max_gp_points}, n_restarts={n_restarts} (dataset size: {n_samples})")
        
        gp_artifact = artifacts_root / "gp.pkl"
        if args.skip_existing and gp_artifact.exists():
            print(f"[skip] GP artifact exists → load {gp_artifact.name}")
            with gp_artifact.open("rb") as fh:
                gp_loaded = pickle.load(fh)
            gp = gp_loaded["model"]
        else:
            gp = GPBaseline(
                obs_dim=train_obs.shape[1],
                act_dim=train_act.shape[1],
                kernel_type=gp_config.get("kernel_type", "rbf_ard"),
                length_scale_bounds=tuple(gp_config.get("length_scale_bounds", [0.01, 100.0])),
                nu=gp_config.get("nu", 1.5),
                alpha=gp_config.get("alpha", 1e-4),
                normalize_y=gp_config.get("normalize_y", True),
                n_restarts_optimizer=n_restarts,
                max_train_points=max_gp_points,
                subsample_strategy=str(gp_config.get("subsample_strategy", "random")),
                batch_predict_size=int(gp_config.get("batch_predict_size", 2048)),
                random_state=int(getattr(args, "seed", 42)),
            )
            gp.fit(train_obs, train_act)
        
        # Predict
        pred_gp, pred_gp_std = gp.predict(test_obs, return_std=True)
        metrics_gp = compute_metrics(test_act, pred_gp)
        metrics_gp["nll"] = gp.nll(test_obs, test_act)
        metrics_gp["mean_uncertainty"] = float(pred_gp_std.mean())
        results["gp"] = metrics_gp
        print(
            f"[gp] rmse={metrics_gp['rmse']:.4f} "
            f"mae={metrics_gp['mae']:.4f} r2={metrics_gp['r2']:.4f} "
            f"uncertainty={metrics_gp['mean_uncertainty']:.4f}"
        )
        
        # Save model
        if not (args.skip_existing and gp_artifact.exists()):
            with gp_artifact.open("wb") as fh:
                pickle.dump({"model": gp, "config": gp_config}, fh)
        
        if args.save_predictions:
            save_predictions(
                artifacts_root / f"gp_predictions_{timestamp}.csv",
                test_obs,
                test_act,
                pred_gp,
                OBS_COLUMNS,
                ACTION_COLUMNS,
            )

    # MDN Baseline (Mixture Density Network)
    if "mdn" in models_requested:
        if torch is None:
            raise RuntimeError("MDN baseline requires PyTorch.")
        mdn_artifact = artifacts_root / "mdn.pt"
        mdn_config = MDN_CONFIG or {}
        mdn_writer = None
        if args.skip_existing and mdn_artifact.exists():
            print(f"[skip] MDN artifact exists → load {mdn_artifact.name}")
            checkpoint = torch.load(mdn_artifact, map_location="cpu")
            cfg = checkpoint.get("config", {})
            mdn = MDNBaseline(
                obs_dim=cfg.get("obs_dim", train_obs_s.shape[1]),
                act_dim=cfg.get("act_dim", train_act_s.shape[1]),
                hidden_units=cfg.get("hidden_units", [128, 128, 64]),
                n_components=cfg.get("n_components", 5),
                covariance_type=cfg.get("covariance_type", "diag"),
                activation=cfg.get("activation", "relu"),
                dropout=cfg.get("dropout", 0.1),
                learning_rate=mdn_config.get("learning_rate", 0.001),
                weight_decay=mdn_config.get("weight_decay", 0.0001),
                mixture_reg=mdn_config.get("mixture_reg", 0.01),
                covariance_floor=mdn_config.get("covariance_floor", 1e-4),
            )
            mdn.model.load_state_dict(checkpoint["model_state"])  # type: ignore
        else:
            print("[info] training MDN baseline ...")
            mdn = MDNBaseline(
                obs_dim=train_obs_s.shape[1],
                act_dim=train_act_s.shape[1],
                hidden_units=mdn_config.get("hidden_units", [128, 128, 64]),
                n_components=mdn_config.get("n_components", 5),
                covariance_type=mdn_config.get("covariance_type", "diag"),
                activation=mdn_config.get("activation", "relu"),
                dropout=mdn_config.get("dropout", 0.1),
                learning_rate=mdn_config.get("learning_rate", 0.001),
                weight_decay=mdn_config.get("weight_decay", 0.0001),
                mixture_reg=mdn_config.get("mixture_reg", 0.01),
                covariance_floor=mdn_config.get("covariance_floor", 1e-4),
            )
            mdn_writer = make_writer(run_tensorboard_dir, "mdn")
            mdn.fit(
                train_obs_s,
                train_act_s,
                test_obs_s,
                test_act_s,
                epochs=int(mdn_config.get("epochs", 200)),
                batch_size=int(mdn_config.get("batch_size", 32)),
                patience=int(mdn_config.get("patience", 30)),
                min_delta=float(mdn_config.get("min_delta", 1e-4)),
                writer=mdn_writer,
            )
            torch.save({"model_state": mdn.model.state_dict(), "config": {
                "obs_dim": train_obs_s.shape[1], "act_dim": train_act_s.shape[1],
                "hidden_units": mdn_config.get("hidden_units", [128,128,64]),
                "n_components": mdn_config.get("n_components", 5),
                "covariance_type": mdn_config.get("covariance_type", "diag"),
                "activation": mdn_config.get("activation", "relu"),
                "dropout": mdn_config.get("dropout", 0.1)}}, mdn_artifact)
            if mdn_writer is not None:
                mdn_writer.flush(); mdn_writer.close()
        
        # Predict (mean mode)
        pred_mdn_scaled = mdn.predict(test_obs_s, mode="mean")
        pred_mdn = act_scaler.inverse_transform(pred_mdn_scaled)
        metrics_mdn = compute_metrics(test_act, pred_mdn)
        metrics_mdn["nll"] = mdn.nll(test_obs_s, test_act_s)
        results["mdn"] = metrics_mdn
        print(
            f"[mdn] rmse={metrics_mdn['rmse']:.4f} "
            f"mae={metrics_mdn['mae']:.4f} r2={metrics_mdn['r2']:.4f} "
            f"nll={metrics_mdn['nll']:.4f}"
        )
        
        # Save model
        manifest["models"]["mdn"] = {"kind": "mdn", "path": mdn_artifact.name}
        
        if args.save_predictions:
            save_predictions(
                artifacts_root / f"mdn_predictions_{timestamp}.csv",
                test_obs,
                test_act,
                pred_mdn,
                OBS_COLUMNS,
                ACTION_COLUMNS,
            )
        
        if mdn_writer is not None:
            mdn_writer.flush()
            mdn_writer.close()

    if "diffusion_c" in models_requested:
        if torch is None:
            raise RuntimeError("Requested diffusion policy but PyTorch is not installed.")
        diffusion_c_artifact = artifacts_root / "diffusion_c.pt"
        diff_c_writer = None
        if args.skip_existing and diffusion_c_artifact.exists():
            print(f"[skip] diffusion_c artifact exists → load {diffusion_c_artifact.name}")
            checkpoint = torch.load(diffusion_c_artifact, map_location="cpu")
            cfg = checkpoint.get("config", {})
            diffusion_c = DiffusionPolicyBaseline(
                obs_dim=cfg.get("obs_dim", train_obs_s.shape[1]),
                act_dim=cfg.get("act_dim", train_act_s.shape[1]),
                timesteps=cfg.get("timesteps", args.diffusion_steps),
                hidden_dim=cfg.get("hidden_dim", args.diffusion_hidden),
                lr=args.diffusion_lr,
                batch_size=args.diffusion_batch,
                epochs=0,
                seed=args.seed,
                log_name="diffusion_c",
                temporal=False,
            )
            diffusion_c.model.load_state_dict(checkpoint["state_dict"])  # type: ignore
        else:
            print("[info] training diffusion policy (conditional) ...")
            diffusion_c = DiffusionPolicyBaseline(
                obs_dim=train_obs_s.shape[1],
                act_dim=train_act_s.shape[1],
                timesteps=args.diffusion_steps,
                hidden_dim=args.diffusion_hidden,
                lr=args.diffusion_lr,
                batch_size=args.diffusion_batch,
                epochs=args.diffusion_epochs,
                seed=args.seed,
                log_name="diffusion_c",
                temporal=False,
            )
            diff_c_writer = make_writer(run_tensorboard_dir, "diffusion_c")
            diffusion_c.fit(train_obs_s, train_act_s, writer=diff_c_writer)
            if diff_c_writer is not None:
                diff_c_writer.flush(); diff_c_writer.close()
        diffusion_c_config = {
            "obs_dim": train_obs_s.shape[1],
            "act_dim": train_act_s.shape[1],
            "timesteps": args.diffusion_steps,
            "hidden_dim": args.diffusion_hidden,
            "time_dim": diffusion_c.model.time_embed.dim if hasattr(diffusion_c.model.time_embed, "dim") else 64,
            "lr": args.diffusion_lr,
            "batch_size": args.diffusion_batch,
            "epochs": args.diffusion_epochs,
            "seed": args.seed,
            "temporal": False,
        }
        torch.save(
            {
                "config": diffusion_c_config,
                "state_dict": {k: v.cpu() for k, v in diffusion_c.model.state_dict().items()},
            },
            diffusion_c_artifact,
        )
        manifest["models"]["diffusion_c"] = {
            "kind": "diffusion",
            "path": diffusion_c_artifact.name,
            "temporal": False,
            "sampler": "ddpm",
            "eta": 0.0,
        }
        pred_scaled_c_ddpm = diffusion_c.predict(test_obs_s, n_samples=4, sampler="ddpm", eta=0.0)
        pred_c_ddpm = act_scaler.inverse_transform(pred_scaled_c_ddpm)
        metrics_c_ddpm = compute_metrics(test_act, pred_c_ddpm)
        metrics_c_ddpm["nll"] = float("nan")
        results["diffusion_c"] = metrics_c_ddpm
        results["diffusion_c_ddpm"] = metrics_c_ddpm
        print(
            f"[diffusion_c|ddpm] rmse={metrics_c_ddpm['rmse']:.4f} "
            f"mae={metrics_c_ddpm['mae']:.4f} r2={metrics_c_ddpm['r2']:.4f}"
        )
        if args.save_predictions:
            save_predictions(
                args.output_dir / f"diffusion_c_predictions_{timestamp}.csv",
                test_obs,
                test_act,
                pred_c_ddpm,
                OBS_COLUMNS,
                ACTION_COLUMNS,
            )
        pred_scaled_c_ddim = diffusion_c.predict(test_obs_s, n_samples=4, sampler="ddim", eta=diffusion_eta)
        pred_c_ddim = act_scaler.inverse_transform(pred_scaled_c_ddim)
        metrics_c_ddim = compute_metrics(test_act, pred_c_ddim)
        metrics_c_ddim["nll"] = float("nan")
        results["diffusion_c_ddim"] = metrics_c_ddim
        manifest["models"]["diffusion_c_ddim"] = {
            "kind": "diffusion",
            "path": diffusion_c_artifact.name,
            "temporal": False,
            "sampler": "ddim",
            "eta": diffusion_eta,
        }
        print(
            f"[diffusion_c|ddim] rmse={metrics_c_ddim['rmse']:.4f} "
            f"mae={metrics_c_ddim['mae']:.4f} r2={metrics_c_ddim['r2']:.4f}"
        )
        if args.save_predictions:
            save_predictions(
                args.output_dir / f"diffusion_c_ddim_predictions_{timestamp}.csv",
                test_obs,
                test_act,
                pred_c_ddim,
                OBS_COLUMNS,
                ACTION_COLUMNS,
            )
        if diff_c_writer is not None:
            diff_c_writer.flush()
            diff_c_writer.close()

    if "diffusion_t" in models_requested:
        if torch is None:
            raise RuntimeError("Requested temporal diffusion policy but PyTorch is not installed.")
        if train_seq_obs.shape[0] == 0:
            raise RuntimeError(
                "Temporal diffusion policy requires sequence data in the training split. "
                "Reduce --sequence-window or provide longer demonstrations."
            )
        if test_seq_obs.shape[0] == 0:
            raise RuntimeError("Temporal diffusion policy requires sequence data. Increase --sequence-window or trajectory length.")
        diffusion_t_artifact = artifacts_root / "diffusion_t.pt"
        diff_t_writer = None
        if args.skip_existing and diffusion_t_artifact.exists():
            print(f"[skip] diffusion_t artifact exists → load {diffusion_t_artifact.name}")
            checkpoint = torch.load(diffusion_t_artifact, map_location="cpu")
            cfg = checkpoint.get("config", {})
            diffusion_t = DiffusionPolicyBaseline(
                obs_dim=cfg.get("obs_dim", train_seq_obs.shape[-1]),
                act_dim=cfg.get("act_dim", train_seq_act_s.shape[1] if train_seq_act_s.size else train_act_s.shape[1]),
                timesteps=cfg.get("timesteps", args.diffusion_steps),
                hidden_dim=cfg.get("hidden_dim", args.diffusion_hidden),
                lr=args.diffusion_lr,
                batch_size=args.diffusion_batch,
                epochs=0,
                seed=args.seed,
                log_name="diffusion_t",
                temporal=True,
            )
            diffusion_t.model.load_state_dict(checkpoint["state_dict"])  # type: ignore
        else:
            print("[info] training diffusion policy (temporal) ...")
            diffusion_t = DiffusionPolicyBaseline(
                obs_dim=train_seq_obs.shape[-1],
                act_dim=train_seq_act_s.shape[1] if train_seq_act_s.size else train_act_s.shape[1],
                timesteps=args.diffusion_steps,
                hidden_dim=args.diffusion_hidden,
                lr=args.diffusion_lr,
                batch_size=args.diffusion_batch,
                epochs=args.diffusion_epochs,
                seed=args.seed,
                log_name="diffusion_t",
                temporal=True,
            )
            diff_t_writer = make_writer(run_tensorboard_dir, "diffusion_t")
            diffusion_t.fit(train_seq_obs, train_seq_act_s, writer=diff_t_writer)
            if diff_t_writer is not None:
                diff_t_writer.flush(); diff_t_writer.close()
        diffusion_t_config = {
            "obs_dim": train_seq_obs.shape[-1],
            "act_dim": train_seq_act_s.shape[1] if train_seq_act_s.size else train_act_s.shape[1],
            "timesteps": args.diffusion_steps,
            "hidden_dim": args.diffusion_hidden,
            "time_dim": diffusion_t.model.time_embed.dim if hasattr(diffusion_t.model.time_embed, "dim") else 64,
            "lr": args.diffusion_lr,
            "batch_size": args.diffusion_batch,
            "epochs": args.diffusion_epochs,
            "seed": args.seed,
            "temporal": True,
            "seq_len": window,
        }
        torch.save(
            {
                "config": diffusion_t_config,
                "state_dict": {k: v.cpu() for k, v in diffusion_t.model.state_dict().items()},
            },
            diffusion_t_artifact,
        )
        manifest["models"]["diffusion_t"] = {
            "kind": "diffusion",
            "path": diffusion_t_artifact.name,
            "temporal": True,
            "seq_len": window,
            "sampler": "ddpm",
            "eta": 0.0,
        }
        pred_seq_scaled_ddpm = diffusion_t.predict(test_seq_obs, n_samples=4, sampler="ddpm", eta=0.0)
        pred_seq_ddpm = act_scaler.inverse_transform(pred_seq_scaled_ddpm)
        metrics_t_ddpm = compute_metrics(test_seq_act_raw, pred_seq_ddpm)
        metrics_t_ddpm["nll"] = float("nan")
        results["diffusion_t"] = metrics_t_ddpm
        results["diffusion_t_ddpm"] = metrics_t_ddpm
        full_pred = np.full_like(test_act, np.nan)
        full_pred[test_seq_indices] = pred_seq_ddpm
        print(
            f"[diffusion_t|ddpm] rmse={metrics_t_ddpm['rmse']:.4f} "
            f"mae={metrics_t_ddpm['mae']:.4f} r2={metrics_t_ddpm['r2']:.4f}"
        )
        if args.save_predictions:
            save_predictions(
                args.output_dir / f"diffusion_t_predictions_{timestamp}.csv",
                test_obs,
                test_act,
                full_pred,
                OBS_COLUMNS,
                ACTION_COLUMNS,
            )
        pred_seq_scaled_ddim = diffusion_t.predict(test_seq_obs, n_samples=4, sampler="ddim", eta=diffusion_eta)
        pred_seq_ddim = act_scaler.inverse_transform(pred_seq_scaled_ddim)
        metrics_t_ddim = compute_metrics(test_seq_act_raw, pred_seq_ddim)
        metrics_t_ddim["nll"] = float("nan")
        results["diffusion_t_ddim"] = metrics_t_ddim 
        manifest["models"]["diffusion_t_ddim"] = {
            "kind": "diffusion",
            "path": diffusion_t_artifact.name,
            "temporal": True,
            "seq_len": window,
            "sampler": "ddim",
            "eta": diffusion_eta,
        }
        full_pred_ddim = np.full_like(test_act, np.nan)
        full_pred_ddim[test_seq_indices] = pred_seq_ddim
        print(
            f"[diffusion_t|ddim] rmse={metrics_t_ddim['rmse']:.4f} "
            f"mae={metrics_t_ddim['mae']:.4f} r2={metrics_t_ddim['r2']:.4f}"
        )
        if args.save_predictions:
            save_predictions(
                args.output_dir / f"diffusion_t_ddim_predictions_{timestamp}.csv",
                test_obs,
                test_act,
                full_pred_ddim,
                OBS_COLUMNS,
                ACTION_COLUMNS,
            )
        if diff_t_writer is not None:
            diff_t_writer.flush()
            diff_t_writer.close()

    # ============================================================================
    # Action Chunking Diffusion Policies (horizon > 1)
    # ============================================================================
    for horizon_val in [4, 8]:
        model_key = f"diffusion_t_h{horizon_val}"
        if model_key not in models_requested:
            continue
        if torch is None:
            raise RuntimeError(f"Requested {model_key} but PyTorch is not installed.")
        if train_seq_obs.shape[0] == 0:
            raise RuntimeError(
                f"{model_key} requires sequence data in the training split. "
                "Reduce --sequence-window or provide longer demonstrations."
            )
        if test_seq_obs.shape[0] == 0:
            raise RuntimeError(f"{model_key} requires sequence data. Increase --sequence-window or trajectory length.")
        
        artifact_path = artifacts_root / f"{model_key}.pt"
        writer = None
        
        if args.skip_existing and artifact_path.exists():
            print(f"[skip] {model_key} artifact exists → load {artifact_path.name}")
            checkpoint = torch.load(artifact_path, map_location="cpu")
            cfg = checkpoint.get("config", {})
            trainer = DiffusionPolicyBaseline(
                obs_dim=cfg.get("obs_dim", train_seq_obs.shape[-1]),
                act_dim=cfg.get("act_dim", train_seq_act_s.shape[1] if train_seq_act_s.size else train_act_s.shape[1]),
                timesteps=cfg.get("timesteps", args.diffusion_steps),
                hidden_dim=cfg.get("hidden_dim", args.diffusion_hidden),
                lr=args.diffusion_lr,
                batch_size=args.diffusion_batch,
                epochs=0,
                seed=args.seed,
                log_name=model_key,
                temporal=True,
                action_horizon=horizon_val,
            )
            trainer.model.load_state_dict(checkpoint["state_dict"])
        else:
            print(f"[info] training diffusion policy (temporal, horizon={horizon_val}) ...")
            trainer = DiffusionPolicyBaseline(
                obs_dim=train_seq_obs.shape[-1],
                act_dim=train_seq_act_s.shape[1] if train_seq_act_s.size else train_act_s.shape[1],
                timesteps=args.diffusion_steps,
                hidden_dim=args.diffusion_hidden,
                lr=args.diffusion_lr,
                batch_size=args.diffusion_batch,
                epochs=args.diffusion_epochs,
                seed=args.seed,
                log_name=model_key,
                temporal=True,
                action_horizon=horizon_val,
            )
            writer = make_writer(run_tensorboard_dir, model_key)
            trainer.fit(train_seq_obs, train_seq_act_s, writer=writer)
            if writer is not None:
                writer.flush()
                writer.close()
        
        # Save config and model
        config = {
            "obs_dim": train_seq_obs.shape[-1],
            "act_dim": train_seq_act_s.shape[1] if train_seq_act_s.size else train_act_s.shape[1],
            "timesteps": args.diffusion_steps,
            "hidden_dim": args.diffusion_hidden,
            "time_dim": trainer.model.time_embed.dim if hasattr(trainer.model.time_embed, "dim") else 64,
            "lr": args.diffusion_lr,
            "batch_size": args.diffusion_batch,
            "epochs": args.diffusion_epochs,
            "seed": args.seed,
            "temporal": True,
            "seq_len": window,
            "action_horizon": horizon_val,
        }
        torch.save(
            {
                "config": config,
                "state_dict": {k: v.cpu() for k, v in trainer.model.state_dict().items()},
            },
            artifact_path,
        )
        
        # Register in manifest
        manifest["models"][model_key] = {
            "kind": "diffusion",
            "path": artifact_path.name,
            "temporal": True,
            "seq_len": window,
            "action_horizon": horizon_val,
            "sampler": "ddim",
            "eta": diffusion_eta,
        }
        
        # Evaluate with DDIM sampler
        pred_seq_scaled = trainer.predict(test_seq_obs, n_samples=4, sampler="ddim", eta=diffusion_eta)
        pred_seq = act_scaler.inverse_transform(pred_seq_scaled)
        metrics_chunk = compute_metrics(test_seq_act_raw, pred_seq)
        metrics_chunk["nll"] = float("nan")
        results[model_key] = metrics_chunk
        
        full_pred_chunk = np.full_like(test_act, np.nan)
        full_pred_chunk[test_seq_indices] = pred_seq
        
        print(
            f"[{model_key}|ddim] rmse={metrics_chunk['rmse']:.4f} "
            f"mae={metrics_chunk['mae']:.4f} r2={metrics_chunk['r2']:.4f}"
        )
        
        if args.save_predictions:
            save_predictions(
                args.output_dir / f"{model_key}_predictions_{timestamp}.csv",
                test_obs,
                test_act,
                full_pred_chunk,
                OBS_COLUMNS,
                ACTION_COLUMNS,
            )

    if "lstm_gmm" in models_requested:
        if torch is None:
            raise RuntimeError("LSTM-GMM baseline requires PyTorch.")
        if train_seq_obs.shape[0] == 0:
            raise RuntimeError(
                "LSTM-GMM baseline requires sequence data in the training split. "
                "Reduce --sequence-window or provide longer demonstrations."
            )
        if test_seq_obs.shape[0] == 0:
            raise RuntimeError("LSTM-GMM baseline requires sequence data. Increase --sequence-window or trajectory length.")
        lstm_artifact = artifacts_root / "lstm_gmm.pt"
        lstm_writer = None
        if args.skip_existing and lstm_artifact.exists():
            print(f"[skip] LSTM-GMM artifact exists → load {lstm_artifact.name}")
            checkpoint = torch.load(lstm_artifact, map_location="cpu")
            cfg = checkpoint.get("config", {})
            lstm_gmm = LSTMGMMBaseline(
                obs_dim=cfg.get("obs_dim", train_seq_obs.shape[-1]),
                act_dim=cfg.get("act_dim", train_seq_act_s.shape[1] if train_seq_act_s.size else train_act_s.shape[1]),
                seq_len=cfg.get("seq_len", window),
                n_components=cfg.get("n_components", args.lstm_gmm_components),
                hidden_dim=cfg.get("hidden_dim", args.lstm_gmm_hidden),
                n_layers=cfg.get("n_layers", args.lstm_gmm_layers),
                lr=args.lstm_gmm_lr,
                batch_size=args.bc_batch,
                epochs=0,
                seed=args.seed,
                log_name="lstm_gmm",
            )
            lstm_gmm.model.load_state_dict(checkpoint["state_dict"])  # type: ignore
        else:
            print("[info] training LSTM-GMM baseline ...")
            lstm_gmm = LSTMGMMBaseline(
                obs_dim=train_seq_obs.shape[-1],
                act_dim=train_seq_act_s.shape[1] if train_seq_act_s.size else train_act_s.shape[1],
                seq_len=window,
                n_components=args.lstm_gmm_components,
                hidden_dim=args.lstm_gmm_hidden,
                n_layers=args.lstm_gmm_layers,
                lr=args.lstm_gmm_lr,
                batch_size=args.bc_batch,
                epochs=args.lstm_gmm_epochs,
                seed=args.seed,
                log_name="lstm_gmm",
            )
            lstm_writer = make_writer(run_tensorboard_dir, "lstm_gmm")
            lstm_gmm.fit(train_seq_obs, train_seq_act_s, writer=lstm_writer)
            if lstm_writer is not None:
                lstm_writer.flush(); lstm_writer.close()
        lstm_config = {
            "obs_dim": train_seq_obs.shape[-1],
            "act_dim": train_seq_act_s.shape[1] if train_seq_act_s.size else train_act_s.shape[1],
            "seq_len": window,
            "n_components": args.lstm_gmm_components,
            "hidden_dim": args.lstm_gmm_hidden,
            "n_layers": args.lstm_gmm_layers,
            "lr": args.lstm_gmm_lr,
            "batch_size": args.bc_batch,
            "epochs": args.lstm_gmm_epochs,
            "seed": args.seed,
        }
        torch.save(
            {
                "config": lstm_config,
                "state_dict": {k: v.cpu() for k, v in lstm_gmm.model.state_dict().items()},
            },
            lstm_artifact,
        )
        manifest["models"]["lstm_gmm"] = {
            "kind": "lstm_gmm",
            "path": lstm_artifact.name,
            "seq_len": window,
        }
        pred_seq_scaled = lstm_gmm.predict(test_seq_obs, mode="mean", n_samples=args.gmm_samples)
        pred_seq = act_scaler.inverse_transform(pred_seq_scaled)
        metrics_lstm = compute_metrics(test_seq_act_raw, pred_seq)
        metrics_lstm["nll"] = float("nan")
        results["lstm_gmm"] = metrics_lstm
        full_pred = np.full_like(test_act, np.nan)
        full_pred[test_seq_indices] = pred_seq
        print(
            f"[lstm_gmm] rmse={metrics_lstm['rmse']:.4f} "
            f"mae={metrics_lstm['mae']:.4f} r2={metrics_lstm['r2']:.4f}"
        )
        if args.save_predictions:
            save_predictions(
                args.output_dir / f"lstm_gmm_predictions_{timestamp}.csv",
                test_obs,
                test_act,
                full_pred,
                OBS_COLUMNS,
                ACTION_COLUMNS,
            )
        if lstm_writer is not None:
            lstm_writer.flush()
            lstm_writer.close()

    manifest_path = artifacts_root / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    summary = {
        "timestamp": timestamp,
        "stiffness_dir": str(args.stiffness_dir),
        "train_samples": int(train_obs.shape[0]),
        "test_samples": int(test_obs.shape[0]),
        "train_trajectories": [t.name for t in train_traj],
        "test_trajectories": [t.name for t in test_traj],
        "evaluation": evaluation_meta,
        "artifact_dir": str(artifacts_root),
        "manifest": manifest_path.name,
        "models": results,
        "hyperparameters": {
            "gmm_components": args.gmm_components,
            "gmm_covariance": args.gmm_covariance,
            "gmm_samples": args.gmm_samples,
            "diffusion_epochs": args.diffusion_epochs,
            "diffusion_steps": args.diffusion_steps,
            "diffusion_hidden": args.diffusion_hidden,
            "diffusion_batch": args.diffusion_batch,
            "diffusion_lr": args.diffusion_lr,
            "bc_hidden": args.bc_hidden,
            "bc_depth": args.bc_depth,
            "lstm_gmm_components": args.lstm_gmm_components,
            "lstm_gmm_hidden": args.lstm_gmm_hidden,
            "lstm_gmm_layers": args.lstm_gmm_layers,
            "sequence_window": window,
            "ibc_noise_std": args.ibc_noise_std,
            "ibc_langevin_steps": args.ibc_langevin_steps,
            "ibc_step_size": args.ibc_step_size,
        },
    }
    out_json = args.output_dir / f"benchmark_summary_{timestamp}.json"
    with out_json.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[done] summary saved to {out_json}")
    print(f"[done] artifacts stored in {artifacts_root}")


def run_per_finger_benchmarks(args, models_requested: set) -> None:
    """Per-finger training: 3 independent models (th: 8D->3D, if: 8D->3D, mf: 8D->3D)."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Resolve artifact/tensorboard directories
    artifacts_root, run_tensorboard_dir = _resolve_artifact_and_tb_dirs(
        base_output=args.output_dir,
        base_tb=args.tensorboard_dir,
        stiffness_dir=args.stiffness_dir,
        timestamp=timestamp,
        mode="per-finger",
    )
    ensure_dir(artifacts_root)
    
    all_results = {}
    all_manifests = {}
    
    for finger in ["th", "if", "mf"]:
        print(f"\n{'='*80}")
        print(f"[info] Training models for finger: {finger.upper()}")
        print(f"{'='*80}")
        
        # Load finger-specific data
        trajectories = load_dataset_per_finger(
            args.stiffness_dir,
            stride=max(1, args.stride),
            finger=finger,
            include_aug=args.augment,
        )
        
        # Train/test split
        if args.eval_demo:
            eval_name = args.eval_demo
            if eval_name.endswith(".csv"):
                eval_name = Path(eval_name).stem
            eval_name_finger = f"{eval_name}_{finger}"
            candidate = next((t for t in trajectories if t.name == eval_name_finger), None)
            if candidate is None:
                print(f"[skip] {finger}: eval demo '{args.eval_demo}' not found")
                continue
            train_traj = [t for t in trajectories if t.name != candidate.name]
            if not train_traj:
                print(f"[skip] {finger}: need at least one trajectory for training")
                continue
            test_traj = [candidate]
        else:
            train_traj, test_traj = split_train_test(trajectories, args.test_size, args.seed)
        
        train_obs, train_act = flatten_trajectories(train_traj)
        test_obs, test_act = flatten_trajectories(test_traj)
        
        print(f"[{finger}] Train: {train_obs.shape[0]} samples ({len(train_traj)} demos), Test: {test_obs.shape[0]} samples ({len(test_traj)} demos)")
        print(f"[{finger}] Obs dim: {train_obs.shape[1]}D, Act dim: {train_act.shape[1]}D")
        
        # Scaling
        obs_scaler = StandardScaler()
        act_scaler = StandardScaler()
        train_obs_s = obs_scaler.fit_transform(train_obs)
        test_obs_s = obs_scaler.transform(test_obs)
        train_act_s = act_scaler.fit_transform(train_act)
        test_act_s = act_scaler.transform(test_act)
        
        # Save scalers for this finger
        finger_artifact_dir = ensure_dir(artifacts_root / finger)
        scalers_path = finger_artifact_dir / "scalers.pkl"
        with scalers_path.open("wb") as fh:
            pickle.dump({"obs_scaler": obs_scaler, "act_scaler": act_scaler}, fh)
        
        finger_results = {}
        finger_manifest = {
            "timestamp": timestamp,
            "finger": finger,
            "scalers": scalers_path.name,
            "models": {},
            "train_trajectories": [t.name for t in train_traj],
            "test_trajectories": [t.name for t in test_traj],
        }
        
        # Prepare sequence data for LSTM-GMM and temporal models
        window = int(LSTM_GMM_CONFIG.get("sequence_window", 10))
        
        # Build scaled trajectories for sequence dataset
        train_traj_scaled = [
            Trajectory(
                name=t.name,
                observations=obs_scaler.transform(t.observations),
                actions=act_scaler.transform(t.actions),
            )
            for t in train_traj
        ]
        test_traj_scaled = [
            Trajectory(
                name=t.name,
                observations=obs_scaler.transform(t.observations),
                actions=act_scaler.transform(t.actions),
            )
            for t in test_traj
        ]
        
        train_offsets = compute_offsets(train_traj)
        test_offsets = compute_offsets(test_traj)
        train_seq_obs, train_seq_act_s, train_seq_act_raw, _ = build_sequence_dataset(
            train_traj_scaled,
            train_traj,
            window,
            train_offsets,
        )
        test_seq_obs, test_seq_act_s, test_seq_act_raw, test_seq_indices = build_sequence_dataset(
            test_traj_scaled,
            test_traj,
            window,
            test_offsets,
        )
        
        # Train GMM/GMR models
        if {"gmm", "gmr"} & models_requested:
            print(f"[{finger}] training Gaussian Mixture Models ...")
            try:
                gmm_policy = GMMConditional(
                    obs_dim=train_obs_s.shape[1],
                    act_dim=train_act_s.shape[1],
                    n_components=args.gmm_components,
                    covariance_type=args.gmm_covariance,
                    reg_covar=args.gmm_reg_covar,
                )
                gmm_policy.fit(train_obs_s, train_act_s)
                
                # Save GMM artifact
                gmm_artifact = finger_artifact_dir / "gmm.pkl"
                gmm_config = {
                    "obs_dim": train_obs_s.shape[1],
                    "act_dim": train_act_s.shape[1],
                    "n_components": args.gmm_components,
                    "covariance_type": args.gmm_covariance,
                    "reg_covar": args.gmm_reg_covar,
                }
                with gmm_artifact.open("wb") as fh:
                    pickle.dump({"config": gmm_config, "model": gmm_policy}, fh)
                finger_manifest["models"]["gmm"] = {
                    "kind": "gmm",
                    "path": gmm_artifact.name,
                }
                
                # Evaluate GMM (stochastic sampling)
                if "gmm" in models_requested:
                    gmm_metrics = evaluate_gmm(
                        gmm_policy,
                        test_obs_s,
                        test_act_s,
                        mode="sample",
                        n_samples=args.gmm_samples,
                        act_scaler=act_scaler,
                        act_test_raw=test_act,
                    )
                    print(f"[{finger}] gmm rmse={gmm_metrics['rmse']:.4f} mae={gmm_metrics['mae']:.4f} r2={gmm_metrics['r2']:.4f}")
                    finger_results["gmm"] = gmm_metrics
                
                # Evaluate GMR (deterministic regression)
                if "gmr" in models_requested:
                    gmr_metrics = evaluate_gmm(
                        gmm_policy,
                        test_obs_s,
                        test_act_s,
                        mode="regression",
                        n_samples=1,
                        act_scaler=act_scaler,
                        act_test_raw=test_act,
                    )
                    print(f"[{finger}] gmr rmse={gmr_metrics['rmse']:.4f} mae={gmr_metrics['mae']:.4f} r2={gmr_metrics['r2']:.4f}")
                    finger_results["gmr"] = gmr_metrics
            except Exception as exc:
                print(f"[{finger}] gmm/gmr training failed: {exc}")
        
        # Train BC model
        if "bc" in models_requested:
            print(f"[{finger}] training behavior cloning baseline ...")
            try:
                bc_policy = BehaviorCloningBaseline(
                    obs_dim=train_obs_s.shape[1],
                    act_dim=train_act_s.shape[1],
                    hidden_dim=args.bc_hidden,
                    depth=args.bc_depth,
                    lr=args.bc_lr,
                    batch_size=args.bc_batch,
                    epochs=args.bc_epochs,
                    weight_decay=args.bc_weight_decay,
                    log_name=f"bc_{finger}",
                )
                bc_writer = make_writer(run_tensorboard_dir, f"bc_{finger}")
                bc_policy.fit(train_obs_s, train_act_s, writer=bc_writer, verbose=False)
                
                # Save BC model
                bc_artifact = finger_artifact_dir / "bc.pt"
                bc_config = {
                    "obs_dim": train_obs_s.shape[1],
                    "act_dim": train_act_s.shape[1],
                    "hidden_dim": args.bc_hidden,
                    "depth": args.bc_depth,
                    "lr": args.bc_lr,
                    "batch_size": args.bc_batch,
                    "epochs": args.bc_epochs,
                    "weight_decay": args.bc_weight_decay,
                    "seed": args.seed,
                }
                torch.save(
                    {
                        "config": bc_config,
                        "state_dict": {k: v.cpu() for k, v in bc_policy.model.state_dict().items()},
                    },
                    bc_artifact,
                )
                finger_manifest["models"]["bc"] = {
                    "kind": "bc",
                    "path": bc_artifact.name,
                }
                
                # Evaluate
                pred_act_s = bc_policy.predict(test_obs_s)
                pred_act = act_scaler.inverse_transform(pred_act_s)
                
                metrics = compute_metrics(test_act, pred_act)
                print(f"[{finger}] bc rmse={metrics['rmse']:.4f} mae={metrics['mae']:.4f} r2={metrics['r2']:.4f}")
                finger_results["bc"] = metrics
                
                if bc_writer is not None:
                    bc_writer.flush()
                    bc_writer.close()
            except Exception as exc:
                print(f"[{finger}] bc training failed: {exc}")
        
        # Train LSTM-GMM model
        if "lstm_gmm" in models_requested:
            if train_seq_obs.shape[0] == 0 or test_seq_obs.shape[0] == 0:
                print(f"[{finger}] lstm_gmm skipped (insufficient sequence data)")
            else:
                print(f"[{finger}] training LSTM-GMM baseline ...")
                try:
                    lstm_gmm = LSTMGMMBaseline(
                        obs_dim=train_seq_obs.shape[-1],
                        act_dim=train_seq_act_s.shape[1],
                        seq_len=window,
                        n_components=args.lstm_gmm_components,
                        hidden_dim=args.lstm_gmm_hidden,
                        n_layers=args.lstm_gmm_layers,
                        lr=args.lstm_gmm_lr,
                        batch_size=args.bc_batch,
                        epochs=args.lstm_gmm_epochs,
                        seed=args.seed,
                        log_name=f"lstm_gmm_{finger}",
                    )
                    lstm_writer = make_writer(run_tensorboard_dir, f"lstm_gmm_{finger}")
                    lstm_gmm.fit(train_seq_obs, train_seq_act_s, writer=lstm_writer, verbose=False)
                    
                    # Save model
                    lstm_artifact = finger_artifact_dir / "lstm_gmm.pt"
                    lstm_config = {
                        "obs_dim": train_seq_obs.shape[-1],
                        "act_dim": train_seq_act_s.shape[1],
                        "seq_len": window,
                        "n_components": args.lstm_gmm_components,
                        "hidden_dim": args.lstm_gmm_hidden,
                        "n_layers": args.lstm_gmm_layers,
                        "lr": args.lstm_gmm_lr,
                        "batch_size": args.bc_batch,
                        "epochs": args.lstm_gmm_epochs,
                        "seed": args.seed,
                    }
                    torch.save(
                        {
                            "config": lstm_config,
                            "state_dict": {k: v.cpu() for k, v in lstm_gmm.model.state_dict().items()},
                        },
                        lstm_artifact,
                    )
                    finger_manifest["models"]["lstm_gmm"] = {
                        "kind": "lstm_gmm",
                        "path": lstm_artifact.name,
                        "seq_len": window,
                    }
                    
                    # Evaluate
                    pred_seq_scaled = lstm_gmm.predict(test_seq_obs, mode="mean", n_samples=args.gmm_samples)
                    pred_seq = act_scaler.inverse_transform(pred_seq_scaled)
                    metrics = compute_metrics(test_seq_act_raw, pred_seq)
                    metrics["nll"] = float("nan")
                    print(f"[{finger}] lstm_gmm rmse={metrics['rmse']:.4f} mae={metrics['mae']:.4f} r2={metrics['r2']:.4f}")
                    finger_results["lstm_gmm"] = metrics
                    
                    if lstm_writer is not None:
                        lstm_writer.flush()
                        lstm_writer.close()
                except Exception as exc:
                    print(f"[{finger}] lstm_gmm training failed: {exc}")
        
        # Train Diffusion model
        if "diffusion_c" in models_requested:
            print(f"[{finger}] training diffusion policy (conditional) ...")
            try:
                diffusion_c = DiffusionPolicyBaseline(
                    obs_dim=train_obs_s.shape[1],
                    act_dim=train_act_s.shape[1],
                    timesteps=args.diffusion_steps,
                    hidden_dim=args.diffusion_hidden,
                    lr=args.diffusion_lr,
                    batch_size=args.diffusion_batch,
                    epochs=args.diffusion_epochs,
                    seed=args.seed,
                    log_name=f"diffusion_c_{finger}",
                    temporal=False,
                )
                diff_c_writer = make_writer(run_tensorboard_dir, f"diffusion_c_{finger}")
                diffusion_c.fit(train_obs_s, train_act_s, writer=diff_c_writer, verbose=False)
                
                # Save model
                diffusion_c_artifact = finger_artifact_dir / "diffusion_c.pt"
                diffusion_c_config = {
                    "obs_dim": train_obs_s.shape[1],
                    "act_dim": train_act_s.shape[1],
                    "timesteps": args.diffusion_steps,
                    "hidden_dim": args.diffusion_hidden,
                    "time_dim": diffusion_c.model.time_embed.dim if hasattr(diffusion_c.model.time_embed, "dim") else 64,
                    "lr": args.diffusion_lr,
                    "batch_size": args.diffusion_batch,
                    "epochs": args.diffusion_epochs,
                    "seed": args.seed,
                    "temporal": False,
                }
                torch.save(
                    {
                        "config": diffusion_c_config,
                        "state_dict": {k: v.cpu() for k, v in diffusion_c.model.state_dict().items()},
                    },
                    diffusion_c_artifact,
                )
                finger_manifest["models"]["diffusion_c"] = {
                    "kind": "diffusion",
                    "path": diffusion_c_artifact.name,
                    "temporal": False,
                    "sampler": "ddpm",
                    "eta": 0.0,
                }
                
                # Evaluate with DDPM sampler
                pred_scaled_c_ddpm = diffusion_c.predict(test_obs_s, n_samples=4, sampler="ddpm", eta=0.0)
                pred_c_ddpm = act_scaler.inverse_transform(pred_scaled_c_ddpm)
                metrics_c_ddpm = compute_metrics(test_act, pred_c_ddpm)
                metrics_c_ddpm["nll"] = float("nan")
                finger_results["diffusion_c_ddpm"] = metrics_c_ddpm
                print(f"[{finger}] diffusion_c|ddpm rmse={metrics_c_ddpm['rmse']:.4f} mae={metrics_c_ddpm['mae']:.4f} r2={metrics_c_ddpm['r2']:.4f}")
                
                # Evaluate with DDIM sampler
                diffusion_eta = 1.0
                pred_scaled_c_ddim = diffusion_c.predict(test_obs_s, n_samples=4, sampler="ddim", eta=diffusion_eta)
                pred_c_ddim = act_scaler.inverse_transform(pred_scaled_c_ddim)
                metrics_c_ddim = compute_metrics(test_act, pred_c_ddim)
                metrics_c_ddim["nll"] = float("nan")
                finger_results["diffusion_c_ddim"] = metrics_c_ddim
                finger_manifest["models"]["diffusion_c_ddim"] = {
                    "kind": "diffusion",
                    "path": diffusion_c_artifact.name,
                    "temporal": False,
                    "sampler": "ddim",
                    "eta": diffusion_eta,
                }
                print(f"[{finger}] diffusion_c|ddim rmse={metrics_c_ddim['rmse']:.4f} mae={metrics_c_ddim['mae']:.4f} r2={metrics_c_ddim['r2']:.4f}")
                
                if diff_c_writer is not None:
                    diff_c_writer.flush()
                    diff_c_writer.close()
            except Exception as exc:
                print(f"[{finger}] diffusion_c training failed: {exc}")
        
        # Train Diffusion Temporal model
        if "diffusion_t" in models_requested:
            if train_seq_obs.shape[0] == 0 or test_seq_obs.shape[0] == 0:
                print(f"[{finger}] diffusion_t skipped (insufficient sequence data)")
            else:
                print(f"[{finger}] training diffusion policy (temporal) ...")
                try:
                    diffusion_t = DiffusionPolicyBaseline(
                        obs_dim=train_seq_obs.shape[-1],
                        act_dim=train_seq_act_s.shape[1],
                        timesteps=args.diffusion_steps,
                        hidden_dim=args.diffusion_hidden,
                        lr=args.diffusion_lr,
                        batch_size=args.diffusion_batch,
                        epochs=args.diffusion_epochs,
                        seed=args.seed,
                        log_name=f"diffusion_t_{finger}",
                        temporal=True,
                    )
                    diff_t_writer = make_writer(run_tensorboard_dir, f"diffusion_t_{finger}")
                    diffusion_t.fit(train_seq_obs, train_seq_act_s, writer=diff_t_writer, verbose=False)
                    
                    # Save model
                    diffusion_t_artifact = finger_artifact_dir / "diffusion_t.pt"
                    diffusion_t_config = {
                        "obs_dim": train_seq_obs.shape[-1],
                        "act_dim": train_seq_act_s.shape[1],
                        "timesteps": args.diffusion_steps,
                        "hidden_dim": args.diffusion_hidden,
                        "time_dim": diffusion_t.model.time_embed.dim if hasattr(diffusion_t.model.time_embed, "dim") else 64,
                        "lr": args.diffusion_lr,
                        "batch_size": args.diffusion_batch,
                        "epochs": args.diffusion_epochs,
                        "seed": args.seed,
                        "temporal": True,
                        "seq_len": window,
                    }
                    torch.save(
                        {
                            "config": diffusion_t_config,
                            "state_dict": {k: v.cpu() for k, v in diffusion_t.model.state_dict().items()},
                        },
                        diffusion_t_artifact,
                    )
                    finger_manifest["models"]["diffusion_t"] = {
                        "kind": "diffusion",
                        "path": diffusion_t_artifact.name,
                        "temporal": True,
                        "sampler": "ddpm",
                        "eta": 0.0,
                        "seq_len": window,
                    }
                    
                    # Evaluate with DDPM sampler
                    pred_scaled_t_ddpm = diffusion_t.predict(test_seq_obs, n_samples=4, sampler="ddpm", eta=0.0)
                    pred_t_ddpm = act_scaler.inverse_transform(pred_scaled_t_ddpm)
                    metrics_t_ddpm = compute_metrics(test_seq_act_raw, pred_t_ddpm)
                    metrics_t_ddpm["nll"] = float("nan")
                    finger_results["diffusion_t_ddpm"] = metrics_t_ddpm
                    print(f"[{finger}] diffusion_t|ddpm rmse={metrics_t_ddpm['rmse']:.4f} mae={metrics_t_ddpm['mae']:.4f} r2={metrics_t_ddpm['r2']:.4f}")
                    
                    # Evaluate with DDIM sampler
                    diffusion_eta = 1.0
                    pred_scaled_t_ddim = diffusion_t.predict(test_seq_obs, n_samples=4, sampler="ddim", eta=diffusion_eta)
                    pred_t_ddim = act_scaler.inverse_transform(pred_scaled_t_ddim)
                    metrics_t_ddim = compute_metrics(test_seq_act_raw, pred_t_ddim)
                    metrics_t_ddim["nll"] = float("nan")
                    finger_results["diffusion_t_ddim"] = metrics_t_ddim
                    finger_manifest["models"]["diffusion_t_ddim"] = {
                        "kind": "diffusion",
                        "path": diffusion_t_artifact.name,
                        "temporal": True,
                        "sampler": "ddim",
                        "eta": diffusion_eta,
                        "seq_len": window,
                    }
                    print(f"[{finger}] diffusion_t|ddim rmse={metrics_t_ddim['rmse']:.4f} mae={metrics_t_ddim['mae']:.4f} r2={metrics_t_ddim['r2']:.4f}")
                    
                    if diff_t_writer is not None:
                        diff_t_writer.flush()
                        diff_t_writer.close()
                except Exception as exc:
                    print(f"[{finger}] diffusion_t training failed: {exc}")
        
        # Train IBC model
        if "ibc" in models_requested:
            print(f"[{finger}] training IBC baseline ...")
            try:
                ibc_batch = args.ibc_batch if args.ibc_batch is not None else args.bc_batch
                ibc_policy = IBCBaseline(
                    obs_dim=train_obs_s.shape[1],
                    act_dim=train_act_s.shape[1],
                    hidden_dim=args.ibc_hidden,
                    depth=args.ibc_depth,
                    lr=args.ibc_lr,
                    batch_size=ibc_batch,
                    epochs=args.ibc_epochs,
                    noise_std=args.ibc_noise_std,
                    langevin_steps=args.ibc_langevin_steps,
                    step_size=args.ibc_step_size,
                    seed=args.seed,
                    log_name=f"ibc_{finger}",
                )
                ibc_writer = make_writer(run_tensorboard_dir, f"ibc_{finger}")
                ibc_policy.fit(train_obs_s, train_act_s, writer=ibc_writer, verbose=False)
                
                # Save IBC model
                ibc_artifact = finger_artifact_dir / "ibc.pt"
                ibc_config = {
                    "obs_dim": train_obs_s.shape[1],
                    "act_dim": train_act_s.shape[1],
                    "hidden_dim": args.ibc_hidden,
                    "depth": args.ibc_depth,
                    "lr": args.ibc_lr,
                    "batch_size": ibc_batch,
                    "epochs": args.ibc_epochs,
                    "noise_std": args.ibc_noise_std,
                    "langevin_steps": args.ibc_langevin_steps,
                    "step_size": args.ibc_step_size,
                    "seed": args.seed,
                }
                torch.save(
                    {
                        "config": ibc_config,
                        "state_dict": {k: v.cpu() for k, v in ibc_policy.model.state_dict().items()},
                    },
                    ibc_artifact,
                )
                finger_manifest["models"]["ibc"] = {
                    "kind": "ibc",
                    "path": ibc_artifact.name,
                }
                
                # Evaluate
                pred_act_s = ibc_policy.predict(test_obs_s)
                pred_act = act_scaler.inverse_transform(pred_act_s)
                
                metrics = compute_metrics(test_act, pred_act)
                metrics["nll"] = float("nan")
                print(f"[{finger}] ibc rmse={metrics['rmse']:.4f} mae={metrics['mae']:.4f} r2={metrics['r2']:.4f}")
                finger_results["ibc"] = metrics
                
                if ibc_writer is not None:
                    ibc_writer.flush()
                    ibc_writer.close()
            except Exception as exc:
                print(f"[{finger}] ibc training failed: {exc}")
        
        # Store results
        all_results[finger] = finger_results
        all_manifests[finger] = finger_manifest
        
        # Save manifest for this finger
        manifest_path = finger_artifact_dir / "manifest.json"
        with manifest_path.open("w", encoding="utf-8") as fh:
            json.dump(finger_manifest, fh, indent=2)
    
    # Save combined summary
    summary = {
        "mode": "per-finger",
        "timestamp": timestamp,
        "artifacts_root": str(artifacts_root),
        "results_per_finger": all_results,
        "manifests_per_finger": all_manifests,
        "config": {
            "test_size": args.test_size,
            "seed": args.seed,
            "stride": args.stride,
            "bc_epochs": args.bc_epochs,
            "bc_hidden": args.bc_hidden,
            "bc_depth": args.bc_depth,
            "bc_batch": args.bc_batch,
            "bc_lr": args.bc_lr,
        },
    }
    
    out_json = args.output_dir / f"benchmark_summary_per_finger_{timestamp}.json"
    with out_json.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\n[done] per-finger summary saved to {out_json}")
    print(f"[done] artifacts stored in {artifacts_root}")
    
    # Print aggregate results
    print(f"\n{'='*80}")
    print("AGGREGATE RESULTS (per-finger mode)")
    print(f"{'='*80}")
    for finger in ["th", "if", "mf"]:
        if finger in all_results and "bc" in all_results[finger]:
            m = all_results[finger]["bc"]
            print(f"{finger.upper()}: RMSE={m['rmse']:.2f}, MAE={m['mae']:.2f}, R²={m['r2']:.4f}")


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()

