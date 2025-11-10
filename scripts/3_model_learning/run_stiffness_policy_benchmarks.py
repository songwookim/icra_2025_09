#!/usr/bin/env python3
"""Benchmark conditional stiffness policies on demonstration data.

This script pairs the raw demonstrations under ``outputs/logs/success`` with the
low-pass stiffness reconstructions stored in ``outputs/analysis/stiffness_profiles``.
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
import random

import numpy as np  # type: ignore[import]
import pandas as pd  # type: ignore[import]
from scipy.special import logsumexp  # type: ignore[import]
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # type: ignore[import]
from sklearn.mixture import GaussianMixture  # type: ignore[import]
from sklearn.model_selection import train_test_split  # type: ignore[import]
from sklearn.preprocessing import StandardScaler  # type: ignore[import]

try:
    import torch  # type: ignore[import]
    import torch.nn as nn  # type: ignore[import]
    import torch.nn.functional as F  # type: ignore[import]
    from torch.utils.data import DataLoader, TensorDataset  # type: ignore[import]
    
    # New Behavior Cloning Model
    
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
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
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
    from torch.utils.tensorboard import SummaryWriter  # type: ignore[import]
except ImportError:  # pragma: no cover - torch optional
    torch = cast(Any, None)
    nn = cast(Any, None)
    F = cast(Any, None)
    DataLoader = cast(Any, None)
    TensorDataset = cast(Any, None)
    SummaryWriter = cast(Any, None)


DEFAULT_LOG_DIR = Path("outputs") / "logs" / "success"
DEFAULT_STIFFNESS_DIR = Path("outputs") / "analysis" / "stiffness_profiles"
DEFAULT_OUTPUT_DIR = Path("outputs") / "models" / "stiffness_policies"
DEFAULT_TENSORBOARD_DIR = DEFAULT_OUTPUT_DIR / "tensorboard"
OBS_COLUMNS = ["Fx", "Fy", "Fz", "deform_circ", "deform_ecc", "ee_px", "ee_py", "ee_pz"]
ACTION_COLUMNS = ["Kx_lp", "Ky_lp", "Kz_lp"]
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
    cand = stiffness_dir / f"{demo_stem}_paper_profile.csv"
    if not cand.exists():
        raise FileNotFoundError(f"Missing stiffness profile for {demo_stem}: {cand}")
    return cand


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


def load_dataset(log_dir: Path, stiffness_dir: Path, stride: int) -> List[Trajectory]:
    trajectories: List[Trajectory] = []
    for csv_path in sorted(log_dir.glob("*.csv")):
        if csv_path.name.endswith("_paper_profile.csv"):
            continue
        traj = _load_single_demo(csv_path, stiffness_dir, stride)
        if traj is not None:
            trajectories.append(traj)
    if not trajectories:
        raise RuntimeError("No valid demonstrations found. Ensure stiffness profiles exist.")
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
        ):
            torch.manual_seed(seed)
            self.obs_dim = obs_dim
            self.act_dim = act_dim
            self.timesteps = timesteps
            self.batch_size = batch_size
            self.epochs = epochs
            self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
            self.model = ConditionalDiffusionModel(obs_dim, act_dim, hidden_dim, time_dim, temporal=temporal).to(self.device)
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
            dataset = TensorDataset(
                torch.from_numpy(obs.astype(np.float32)),
                torch.from_numpy(act.astype(np.float32)),
            )
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
            for _ in range(max(1, n_samples)):
                x = torch.randn(batch, self.act_dim, device=self.device)
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
            stacked = torch.stack(preds, dim=0).mean(dim=0)
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
            act = torch.randn(batch, self.act_dim, device=self.device) * self.noise_std
            for _ in range(max(1, self.langevin_steps)):
                act = act + self.step_size * self.model(obs_tensor, act)
                act = act + math.sqrt(2 * self.step_size) * self.noise_std * torch.randn_like(act)
            return act.cpu().numpy()


def compute_metrics(target: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    mse = mean_squared_error(target, pred)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(target, pred)
    r2 = r2_score(target, pred)
    return {"rmse": rmse, "mae": mae, "r2": r2}


def evaluate_gmm(
    model: GMMConditional,
    obs_test: np.ndarray,
    act_test: np.ndarray,
    mode: str,
    n_samples: int,
) -> Dict[str, float]:
    pred = model.predict(obs_test, mode=mode, n_samples=n_samples)
    metrics = compute_metrics(act_test, pred)
    metrics["nll"] = model.nll(obs_test, act_test)
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
    # 참고: configs/stiffness_policy/*.yaml 파일에 모델별 기본 하이퍼파라미터가 정리되어 있다.
    # 스크립트가 YAML을 자동으로 로드하지는 않으므로, 값을 바꾸고 싶으면 여기 대응되는
    # CLI 인자(예: --diffusion-epochs, --ibc-batch 등)를 직접 지정해서 덮어써야 한다.
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR, help="Directory with raw demonstrations")
    parser.add_argument(
        "--stiffness-dir",
        type=Path,
        default=DEFAULT_STIFFNESS_DIR,
        help="Directory with *_paper_profile.csv outputs",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for benchmark artifacts")
    parser.add_argument(
        "--models",
        type=str,
        default="all",
        help="Comma-separated models: gmm,gmr,bc,diffusion_c,diffusion_t,lstm_gmm,ibc,all (default: all)",
    )
    parser.add_argument("--test-size", type=float, default=0.25, help="Fraction of trajectories reserved for testing")
    parser.add_argument(
        "--eval-demo",
        type=str,
        default=None,
        help="Optional CSV stem to reserve as the only evaluation trajectory (overrides --test-size)",
    )
    parser.add_argument("--stride", type=int, default=1, help="Subsample demonstrations by stride")
    parser.add_argument("--sequence-window", type=int, default=1, help="Temporal window length for sequence models")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--gmm-components", type=int, default=8, help="Number of mixture components")
    parser.add_argument("--gmm-covariance", type=str, default="full", choices=["full", "diag", "spherical", "tied"], help="Covariance type")
    parser.add_argument("--gmm-samples", type=int, default=16, help="Samples per query for stochastic GMM benchmark")
    parser.add_argument("--diffusion-epochs", type=int, default=200, help="Training epochs for the diffusion policy")
    parser.add_argument("--diffusion-steps", type=int, default=75, help="Diffusion timetable length")
    parser.add_argument("--diffusion-hidden", type=int, default=256, help="Hidden width for the diffusion policy network")
    parser.add_argument("--diffusion-batch", type=int, default=256, help="Batch size for diffusion policy training")
    parser.add_argument("--diffusion-lr", type=float, default=1e-3, help="Learning rate for the diffusion policy")
    parser.add_argument("--diffusion-eta", type=float, default=0.0, help="DDIM eta used for deterministic/stochastic trade-off")
    parser.add_argument("--bc-epochs", type=int, default=200, help="Training epochs for behavior cloning baseline")
    parser.add_argument("--bc-hidden", type=int, default=256, help="Hidden width for behavior cloning MLP")
    parser.add_argument("--bc-depth", type=int, default=3, help="Number of hidden layers for behavior cloning MLP")
    parser.add_argument("--bc-batch", type=int, default=256, help="Batch size for behavior cloning training")
    parser.add_argument("--bc-lr", type=float, default=1e-3, help="Learning rate for behavior cloning baseline")
    parser.add_argument("--lstm-gmm-components", type=int, default=5, help="Mixture components for LSTM-GMM baseline")
    parser.add_argument("--lstm-gmm-hidden", type=int, default=256, help="Hidden width for LSTM encoder")
    parser.add_argument("--lstm-gmm-layers", type=int, default=1, help="Number of LSTM layers")
    parser.add_argument("--lstm-gmm-epochs", type=int, default=200, help="Training epochs for LSTM-GMM baseline")
    parser.add_argument("--ibc-epochs", type=int, default=300, help="Training epochs for IBC baseline")
    parser.add_argument("--ibc-hidden", type=int, default=256, help="Hidden width for IBC score network")
    parser.add_argument("--ibc-depth", type=int, default=3, help="Hidden layers for IBC score network")
    parser.add_argument("--ibc-lr", type=float, default=1e-3, help="Learning rate for IBC baseline")
    parser.add_argument("--ibc-noise-std", type=float, default=0.5, help="Noise std used in IBC training and sampling")
    parser.add_argument("--ibc-langevin-steps", type=int, default=30, help="Langevin iterations for IBC sampling")
    parser.add_argument("--ibc-step-size", type=float, default=1e-2, help="Step size for IBC Langevin updates")
    parser.add_argument("--ibc-batch", type=int, default=None, help="Batch size override for IBC baseline (defaults to --bc-batch)")
    parser.add_argument(
        "--tensorboard",
        dest="tensorboard",
        action="store_true",
        default=True,
        help="Enable TensorBoard logging for neural models (default: enabled)",
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
    parser.add_argument("--save-predictions", action="store_true", help="Persist per-sample predictions to CSV")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    def set_global_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        if torch is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

    set_global_seed(int(args.seed))
    models_requested = {m.strip().lower() for m in args.models.split(",") if m.strip()}
    if "diffusion" in models_requested:
        models_requested.remove("diffusion")
        models_requested.add("diffusion_c")
    if "all" in models_requested or not models_requested:
        models_requested = {"gmm", "gmr", "bc", "ibc", "diffusion_c", "diffusion_t", "lstm_gmm"}

    trajectories = load_dataset(args.log_dir, args.stiffness_dir, stride=max(1, args.stride))

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
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    diffusion_eta = max(0.0, float(args.diffusion_eta))
    run_tensorboard_dir: Optional[Path] = None
    if args.tensorboard:
        if torch is None or SummaryWriter is None:
            raise RuntimeError("TensorBoard requested but torch or tensorboard is unavailable.")
        ensure_dir(args.tensorboard_dir)
        run_tensorboard_dir = ensure_dir(args.tensorboard_dir / timestamp)
    ensure_dir(args.output_dir)
    artifacts_root = ensure_dir(args.output_dir / "artifacts" / timestamp)
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
        print("[info] training Gaussian mixture on joint space ...")
        gmm_model = GMMConditional(
            obs_dim=train_obs_s.shape[1],
            act_dim=train_act_s.shape[1],
            n_components=args.gmm_components,
            covariance_type=args.gmm_covariance,
            reg_covar=1e-5,
        )
        gmm_model.fit(train_obs_s, train_act_s)
        gmm_artifact = artifacts_root / "gmm.pkl"
        with gmm_artifact.open("wb") as fh:
            pickle.dump(gmm_model, fh)

        if "gmr" in models_requested:
            metrics = evaluate_gmm(gmm_model, test_obs_s, test_act_s, mode="mean", n_samples=1)
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
            metrics = evaluate_gmm(gmm_model, test_obs_s, test_act_s, mode="sample", n_samples=args.gmm_samples)
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
        print("[info] training behavior cloning baseline ...")
        bc_config = {
            "obs_dim": train_obs_s.shape[1],
            "act_dim": train_act_s.shape[1],
            "hidden_dim": args.bc_hidden,
            "depth": args.bc_depth,
            "lr": args.bc_lr,
            "batch_size": args.bc_batch,
            "epochs": args.bc_epochs,
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
            seed=bc_config["seed"],
            log_name="bc",
        )
        bc_writer = make_writer(run_tensorboard_dir, "bc")
        bc.fit(train_obs_s, train_act_s, writer=bc_writer)
        bc_artifact = artifacts_root / "bc.pt"
        torch.save(
            {
                "config": bc_config,
                "state_dict": {k: v.cpu() for k, v in bc.model.state_dict().items()},
            },
            bc_artifact,
        )
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
        ibc_artifact = artifacts_root / "ibc.pt"
        torch.save(
            {
                "config": ibc_config,
                "state_dict": {k: v.cpu() for k, v in ibc.model.state_dict().items()},
            },
            ibc_artifact,
        )
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

    if "diffusion_c" in models_requested:
        if torch is None:
            raise RuntimeError("Requested diffusion policy but PyTorch is not installed.")
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
        diffusion_c_artifact = artifacts_root / "diffusion_c.pt"
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
        diffusion_t_artifact = artifacts_root / "diffusion_t.pt"
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
        print("[info] training LSTM-GMM baseline ...")
        lstm_gmm = LSTMGMMBaseline(
            obs_dim=train_seq_obs.shape[-1],
            act_dim=train_seq_act_s.shape[1] if train_seq_act_s.size else train_act_s.shape[1],
            seq_len=window,
            n_components=args.lstm_gmm_components,
            hidden_dim=args.lstm_gmm_hidden,
            n_layers=args.lstm_gmm_layers,
            lr=args.bc_lr,
            batch_size=args.bc_batch,
            epochs=args.lstm_gmm_epochs,
            seed=args.seed,
            log_name="lstm_gmm",
        )
        lstm_writer = make_writer(run_tensorboard_dir, "lstm_gmm")
        lstm_gmm.fit(train_seq_obs, train_seq_act_s, writer=lstm_writer)
        lstm_artifact = artifacts_root / "lstm_gmm.pt"
        lstm_config = {
            "obs_dim": train_seq_obs.shape[-1],
            "act_dim": train_seq_act_s.shape[1] if train_seq_act_s.size else train_act_s.shape[1],
            "seq_len": window,
            "n_components": args.lstm_gmm_components,
            "hidden_dim": args.lstm_gmm_hidden,
            "n_layers": args.lstm_gmm_layers,
            "lr": args.bc_lr,
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
        "log_dir": str(args.log_dir),
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


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
