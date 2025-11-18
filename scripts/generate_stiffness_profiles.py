#!/usr/bin/env python3
"""Generate stiffness profiles using EMG projection method (paper version).

For each demonstration CSV this script:
- extracts force magnitudes (baseline-removed, no low-pass filtering),
- aligns sEMG signals and builds transformation matrices T_F and T_K,
- projects multiple filtered EMG variants (raw magnitude, low-pass, moving-average,
    band-pass, ultra-smooth envelope, and a stronger ultra-smooth envelope) into stiffness trajectories, and
- saves comparison plots plus a CSV with the force traces and low-pass stiffness.

One PNG is emitted per EMG variant (currently: low-pass, moving-avg, band-pass, ultra-smooth, ultra-smooth-strong).
Each PNG has subplots for force magnitude, stiffness (raw-EMG dashed vs filtered solid), and EMG
(raw dashed vs filtered solid) with a small horizontal margin.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, sosfiltfilt, savgol_filter

if os.environ.get("DISPLAY", "") == "":
    import matplotlib
    # Headless 환경에서는 Agg 백엔드 사용 (TKAgg는 시스템 의존성이 필요함)
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
PKG_DIR = SCRIPT_DIR.parent  # .../src/hri_falcon_robot_bridge
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

EPS = 1e-8
# 패키지 내부 outputs/logs/success 만 대상으로 사용
DEFAULT_INPUT = (PKG_DIR / "outputs" / "logs" / "success")
K_INIT = 200.0
K_MIN = 50.0  # 물리적으로 유효한 최소 강성 (N/m)
VALIDATE_RESULTS = True  # set True via --validate flag

# ---------------------------------------------------------------------------
# CSV utilities
# ---------------------------------------------------------------------------

def _load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("CSV contains no rows")
    return df


def _time_vector(df: pd.DataFrame) -> np.ndarray:
    if {"t_sec", "t_nanosec"}.issubset(df.columns):
        sec = df["t_sec"].to_numpy(dtype=float)
        nsec = df["t_nanosec"].to_numpy(dtype=float)
        t = sec + nsec * 1e-9
    elif "t" in df.columns:
        t = df["t"].to_numpy(dtype=float)
    elif "time" in df.columns:
        t = df["time"].to_numpy(dtype=float)
    else:
        t = np.arange(len(df), dtype=float)
    t0 = float(t[0])
    return t - t0


def _estimate_fs(time: np.ndarray) -> Optional[float]:
    if time.ndim != 1 or len(time) < 3:
        return None
    dt = np.diff(time)
    dt = dt[(dt > 0) & np.isfinite(dt)]
    if dt.size == 0:
        return None
    return float(1.0 / np.median(dt))


def _guess_emg_columns(df: pd.DataFrame, channels: Optional[List[int]] = None) -> List[str]:
    if not channels:
        channels = list(range(1, 9))
    cols: List[str] = []
    for ch in channels:
        for cand in (f"emg_ch{ch}", f"ch{ch}", f"emg{ch}", f"channel{ch}"):
            if cand in df.columns:
                cols.append(cand)
                break
    return cols


def _align_emg_to_time(emg: np.ndarray, time: np.ndarray) -> Optional[np.ndarray]:
    if emg is None or emg.size == 0:
        return None
    if emg.shape[0] == time.shape[0]:
        return emg
    if time.size < 2 or emg.shape[0] < 2:
        return None
    t_start = float(time[0])
    t_end = float(time[-1])
    if not np.isfinite(t_start) or not np.isfinite(t_end) or np.isclose(t_end, t_start):
        return None
    src_times = np.linspace(t_start, t_end, num=emg.shape[0], dtype=float)
    if not np.all(np.isfinite(src_times)):
        return None
    aligned = np.empty((time.shape[0], emg.shape[1]), dtype=float)
    for ch in range(emg.shape[1]):
        aligned[:, ch] = np.interp(time, src_times, emg[:, ch])
    return aligned


def _center_signal(signal: np.ndarray, baseline_fraction: float = 0.1) -> np.ndarray:
    samples = signal.shape[0]
    take = max(1, int(round(samples * baseline_fraction)))
    baseline = np.median(signal[:take, :], axis=0, keepdims=True)
    return signal - baseline


# ---------------------------------------------------------------------------
# Force processing
# ---------------------------------------------------------------------------

def _compute_signed_force(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Extract force and return absolute values for both calculation and plotting.
    
    Returns:
        Tuple of (forces_for_calc, forces_for_plot) - both are absolute values
    """
    force_prefixes = ("force_3", "s3", "force3")
    picked: Optional[List[str]] = None
    for prefix in force_prefixes:
        cand = [f"{prefix}_fx", f"{prefix}_fy", f"{prefix}_fz"]
        if all(col in df.columns for col in cand):
            picked = cand
            break
    if picked is None:
        raise KeyError("No force_3 columns found (expected fx, fy, fz)")
    force_raw = df[picked].to_numpy(dtype=float)
    rest_samples = max(1, int(len(force_raw) * 0.1))
    baseline = np.mean(force_raw[:rest_samples, :], axis=0, keepdims=True)
    forces_centered = force_raw - baseline
    
    # Use absolute value for both calculation and plotting
    forces_abs = np.abs(forces_centered)
    return forces_abs, forces_abs


def _smooth_force(force: np.ndarray, fs: Optional[float], cutoff_hz: float = 3.0, order: int = 2) -> Optional[np.ndarray]:
    """Apply low-pass filter to force signals to reduce noise."""
    if force is None or force.size == 0:
        return None
    if fs is None or fs <= 0 or cutoff_hz <= 0 or fs <= 2 * cutoff_hz:
        return None
    try:
        sos = butter(order, cutoff_hz, btype="low", fs=fs, output="sos")
        return sosfiltfilt(sos, force, axis=0)
    except ValueError:
        return None


def _compute_normal_force(
    df: pd.DataFrame,
    prefix: Optional[str] = None,
    n: Tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> np.ndarray:
    """Compute baseline-removed normal force component Fn (signed) for given sensor prefix.

    Args:
        df: input dataframe
        prefix: e.g., 's3', 's1', 'force_3'. If None, tries common prefixes.
        n: normal vector to project onto (will be normalized)

    Returns:
        Fn: (N,) signed normal force component after baseline removal.
    """
    cand_prefixes = ("force_3", "s3", "force3") if prefix is None else (prefix,)
    picked_cols: Optional[List[str]] = None
    for p in cand_prefixes:
        cols = [f"{p}_fx", f"{p}_fy", f"{p}_fz"]
        if all(c in df.columns for c in cols):
            picked_cols = cols
            break
    if picked_cols is None:
        raise KeyError(f"No force columns found for prefix={prefix or cand_prefixes}")
    F = df[picked_cols].to_numpy(dtype=float)
    rest = max(1, int(len(F) * 0.1))
    Fc = F - F[:rest].mean(axis=0, keepdims=True)
    n_vec = np.asarray(n, dtype=float)
    n_norm = np.linalg.norm(n_vec)
    if not np.isfinite(n_norm) or n_norm < 1e-12:
        n_vec = np.array([0.0, 0.0, 1.0], dtype=float)
        n_norm = 1.0
    n_hat = n_vec / n_norm
    Fn = Fc @ n_hat  # (N,)
    return Fn


# ---------------------------------------------------------------------------
# EMG filtering helpers
# ---------------------------------------------------------------------------

def _lowpass_emg(emg: np.ndarray, fs: Optional[float], cutoff_hz: float = 5.0, order: int = 4) -> Optional[np.ndarray]:
    if emg is None or emg.size == 0:
        return None
    if fs is None or fs <= 0 or cutoff_hz <= 0 or fs <= 2 * cutoff_hz:
        return None
    sos = butter(order, cutoff_hz, btype="low", fs=fs, output="sos")
    return sosfiltfilt(sos, emg, axis=0)


def _moving_average_emg(emg: np.ndarray, fs: Optional[float], window_sec: float = 0.1) -> Optional[np.ndarray]:
    if emg is None or emg.size == 0:
        return None
    if fs is None or fs <= 0:
        window = max(1, int(round(window_sec * 200.0)))
    else:
        window = max(1, int(round(window_sec * fs)))
    if window <= 1:
        return emg.copy()
    return uniform_filter1d(emg, size=window, axis=0, mode="nearest")


def _bandpass_emg(emg: np.ndarray, fs: Optional[float], low_hz: float = 20.0, high_hz: float = 450.0, order: int = 4) -> Optional[np.ndarray]:
    if emg is None or emg.size == 0:
        return None
    if fs is None or fs <= 0:
        return None
    nyquist = 0.5 * fs
    high = min(high_hz, nyquist - 1e-3)
    low = max(low_hz, 1e-3)
    if high <= low or high <= 0:
        return None
    sos = butter(order, [low, high], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, emg, axis=0)


def _ultra_smooth_emg(emg_mag: np.ndarray, fs: Optional[float], cutoff_hz: float = 2.0, window_sec: float = 0.35) -> Optional[np.ndarray]:
    if emg_mag is None or emg_mag.size == 0:
        return None
    smoothed = np.asarray(emg_mag, dtype=float)
    if fs is not None and fs > 0 and cutoff_hz > 0 and fs > 2 * cutoff_hz:
        try:
            sos = butter(2, cutoff_hz, btype="low", fs=fs, output="sos")
            smoothed = sosfiltfilt(sos, smoothed, axis=0)
        except ValueError:
            pass
    effective_fs = fs if fs is not None and fs > 0 else 200.0
    window = max(1, int(round(window_sec * effective_fs)))
    if window > 1:
        smoothed = uniform_filter1d(smoothed, size=window, axis=0, mode="nearest")
    smoothed = np.maximum(smoothed, 0.0)
    return smoothed


def _ultra_smooth_strong_emg(
    emg_mag: np.ndarray,
    fs: Optional[float],
    cutoff_hz: float = 1.0,
    window_sec: float = 0.8,
    sg_window_sec: float = 0.6,
    sg_poly: int = 3,
) -> Optional[np.ndarray]:
    """Strong smoothing: LPF -> moving average -> Savitzky–Golay polish -> ReLU."""
    if emg_mag is None or emg_mag.size == 0:
        return None
    sm = np.asarray(emg_mag, dtype=float)
    # LPF
    if fs is not None and fs > 0 and cutoff_hz > 0 and fs > 2 * cutoff_hz:
        try:
            sos = butter(2, cutoff_hz, btype="low", fs=fs, output="sos")
            sm = sosfiltfilt(sos, sm, axis=0)
        except ValueError:
            pass
    # Moving average
    eff_fs = fs if fs is not None and fs > 0 else 200.0
    win = max(1, int(round(window_sec * eff_fs)))
    if win > 1:
        sm = uniform_filter1d(sm, size=win, axis=0, mode="nearest")
    # Savitzky–Golay polish
    sgw = max(5, int(round(sg_window_sec * eff_fs)))
    if sgw % 2 == 0:
        sgw += 1
    if sgw > sg_poly + 2:
        try:
            sm = savgol_filter(sm, window_length=sgw, polyorder=sg_poly, axis=0, mode="interp")
        except Exception:
            pass
    return np.maximum(sm, 0.0)


# ---------------------------------------------------------------------------
# Linear-algebra helpers for T_F / T_K
# ---------------------------------------------------------------------------

def compute_tf(P: np.ndarray, F: np.ndarray, reg: float = 1e-6) -> np.ndarray:
    """Solve T_F from F = T_F · P using ridge regularization."""
    PPt = P @ P.T
    reg_eye = reg * np.eye(PPt.shape[0])
    return F @ P.T @ np.linalg.inv(PPt + reg_eye)


def compute_projection_from_tf(T_F: np.ndarray, reg: float = 1e-6) -> np.ndarray:
    """Return force subspace projector H_F from T_F with numerical symmetrization.

    We compute H_F = T_F^T (T_F T_F^T + λI)^{-1} T_F (ridge regularized) and then
    symmetrize to mitigate floating point drift from exact idempotence.
    """
    TF_TF_T = T_F @ T_F.T
    reg_eye = reg * np.eye(TF_TF_T.shape[0])
    inv_term = np.linalg.inv(TF_TF_T + reg_eye)
    H_F = T_F.T @ inv_term @ T_F
    # Symmetrize for numerical stability
    H_F = 0.5 * (H_F + H_F.T)
    return H_F


def compute_k_basis_from_force_projector(
    H_F: np.ndarray,
    target_rank: int = 3,
    eig_threshold: float = 0.5,
    near_one_tol: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute orthogonal complement basis ("stiffness subspace") of force projector.

    Args:
        H_F: (d,d) approximate projector onto force subspace (should be symmetric).
        target_rank: number of basis vectors to return for stiffness subspace.
        eig_threshold: lower bound separating 0 vs 1 eigenvalues (use >0.5 by default).
        near_one_tol: tolerance when checking eigenvalues close to 1.

    Returns:
        basis: (r,d) row-orthonormal basis vectors spanning complement subspace.
        H_K: (d,d) projector onto complement (I - H_F) symmetrized.
    """
    dim = H_F.shape[0]
    # 1) 대칭화 (수치 안정)
    H_F = 0.5 * (H_F + H_F.T)
    # 2) 보완 사영
    H_K = np.eye(dim) - H_F
    H_K = 0.5 * (H_K + H_K.T)

    # 3) 고유분해 (대칭 행렬)
    eigvals, eigvecs = np.linalg.eigh(H_K)  # 오름차순
    idx = np.argsort(eigvals)[::-1]         # 내림차순 정렬
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # 4) 고유값이 1에 가깝거나, 0.5 초과인 성분 선택 (둘 중 하나라도 True)
    keep_mask = (eigvals > eig_threshold) | (np.isclose(eigvals, 1.0, atol=near_one_tol))
    kept = np.count_nonzero(keep_mask)
    if kept < target_rank:
        raise ValueError(f"Complementary subspace insufficient rank: have {kept}, need {target_rank}.")

    basis_full = eigvecs[:, keep_mask]   # (d, r_full), 열 직교 정규
    basis = basis_full[:, :target_rank].T  # (r, d) 행 직교 정규 (행들이 기저 벡터)

    return basis, H_K


def compute_tk_from_projection(H_F: np.ndarray, target_rank: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """Backward-compatible wrapper calling improved basis computation.

    Existing code expects (T_K_basis, H_K). Here we delegate to
    compute_k_basis_from_force_projector with stable eigen threshold.
    """
    return compute_k_basis_from_force_projector(H_F, target_rank=target_rank)


def compute_scalar_stiffness_map(
    basis: np.ndarray,
    P: np.ndarray,
    kappa: np.ndarray,
) -> np.ndarray:
    """Estimate scalar stiffness linear map T_K (1,d) from subspace basis.

    Args:
        basis: (r,d) row-orthonormal basis vectors for stiffness subspace.
        P: (d,N) EMG feature matrix.
        kappa: (N,) target scalar stiffness samples (e.g., log(K) or raw K).

    Returns:
        T_K: (1,d) linear map so that T_K @ P ≈ kappa.

    Procedure:
        1) Project features: Z = basis @ P (r,N).
        2) Solve least squares Z.T w ≈ kappa for w (r,).
        3) Expand back to original space: T_K = w^T basis.
    """
    if basis.ndim != 2:
        raise ValueError("basis must be 2D (r,d)")
    if P.ndim != 2:
        raise ValueError("P must be 2D (d,N)")
    if kappa.ndim != 1 or kappa.shape[0] != P.shape[1]:
        raise ValueError("kappa shape mismatch with P's sample count")
    Z = basis @ P  # (r,N)
    # Least squares solve
    w, *_ = np.linalg.lstsq(Z.T, kappa, rcond=None)
    T_K_row = w @ basis  # (d,)
    return T_K_row[None, :]


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def validate_stiffness_physics(stiffness: np.ndarray, forces: np.ndarray) -> Dict[str, float]:
    """Basic physical sanity checks for stiffness time series.

    Returns keys:
        - frac_positive: fraction of entries >= 0
        - in_range_frac: fraction within [50, 2000] N/m (heuristic)
        - corr_force_stiff: Pearson correlation between scalarized F and K
          (use norms to avoid axis mismatch)
        - smooth_p95: 95th percentile of |ΔK| per-step (lower is smoother),
          computed after a light MAD-based outlier rejection
    """
    out: Dict[str, float] = {}
    if stiffness is None or forces is None:
        return {"frac_positive": 0.0, "in_range_frac": 0.0, "corr_force_stiff": 0.0, "smooth_p95": float("inf")}
    K = np.asarray(stiffness, dtype=float)
    F = np.asarray(forces, dtype=float)

    # Scalarize K and F for correlation and smoothness metrics
    K_scalar = np.linalg.norm(K, axis=1) if K.ndim == 2 else K.reshape(-1)
    F_scalar = np.linalg.norm(F, axis=1) if F.ndim == 2 else np.abs(F).reshape(-1)

    total = K_scalar.size if K_scalar.size > 0 else 1
    out["frac_positive"] = float(np.count_nonzero(K_scalar >= 0) / total)
    in_rng = (K_scalar >= 50.0) & (K_scalar <= 2000.0)
    out["in_range_frac"] = float(np.count_nonzero(in_rng) / total)

    # Scalar correlation (robust to axis mismatch)
    if K_scalar.size > 3 and np.std(F_scalar) > 1e-9 and np.std(K_scalar) > 1e-9:
        try:
            corr = float(np.corrcoef(F_scalar, K_scalar)[0, 1])
        except Exception:
            corr = 0.0
    else:
        corr = 0.0
    out["corr_force_stiff"] = corr if np.isfinite(corr) else 0.0

    # Robust smoothness: compute p95 of |ΔK| after MAD-based outlier rejection
    dK = np.diff(K_scalar, axis=0)
    if dK.size:
        med = float(np.median(dK))
        mad = float(np.median(np.abs(dK - med))) + 1e-9
        # Consistent MAD scale factor for normal distribution
        thresh = 3.0 * 1.4826 * mad
        keep = np.abs(dK - med) <= thresh
        vals = np.abs(dK[keep]) if np.any(keep) else np.abs(dK)
        out["smooth_p95"] = float(np.percentile(vals, 95)) if vals.size else 0.0
    else:
        out["smooth_p95"] = 0.0
    return out


def validate_emg_stiffness_mapping(emg: np.ndarray, stiffness: np.ndarray) -> float:
    """Return correlation between EMG magnitude and stiffness magnitude over time."""
    if emg is None or stiffness is None:
        return 0.0
    E = np.asarray(emg, dtype=float)
    K = np.asarray(stiffness, dtype=float)
    if E.ndim != 2 or K.ndim != 2 or E.shape[0] != K.shape[0]:
        n = min(E.shape[0] if E.ndim > 0 else 0, K.shape[0] if K.ndim > 0 else 0)
        E = E[:n]
        K = K[:n]
    Emag = np.linalg.norm(E, axis=1)
    Kmag = np.linalg.norm(K, axis=1)
    if len(Emag) < 3 or np.std(Emag) < 1e-9 or np.std(Kmag) < 1e-9:
        return 0.0
    corr = float(np.corrcoef(Emag, Kmag)[0, 1])
    return corr if np.isfinite(corr) else 0.0


def validate_projection_quality(H_F: np.ndarray, H_K: np.ndarray) -> Dict[str, float]:
    """Check projector properties for H_F and H_K (idempotence, orthogonality, completeness)."""
    out: Dict[str, float] = {}
    if H_F is None or H_K is None:
        return {"idemp_F": float("inf"), "idemp_K": float("inf"), "orth": float("inf"), "comp": float("inf")}
    idemp_F = np.linalg.norm(H_F @ H_F - H_F, ord="fro")
    idemp_K = np.linalg.norm(H_K @ H_K - H_K, ord="fro")
    orth = np.linalg.norm(H_F @ H_K, ord="fro")
    comp = np.linalg.norm(H_F + H_K - np.eye(H_F.shape[0]), ord="fro")
    out["idemp_F"] = float(idemp_F)
    out["idemp_K"] = float(idemp_K)
    out["orth"] = float(orth)
    out["comp"] = float(comp)
    return out


# ---------------------------------------------------------------------------
# Least-squares fit diagnostics (for T_F estimation)
# ---------------------------------------------------------------------------

def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute basic regression metrics for a 1D fit.

    Returns keys:
        - rmse: root mean squared error
        - mae: mean absolute error
        - r2: coefficient of determination
        - nrmse_std: RMSE normalized by std(y_true)
    """
    out: Dict[str, float] = {"rmse": float("nan"), "mae": float("nan"), "r2": float("nan"), "nrmse_std": float("nan")}
    if y_true is None or y_pred is None:
        return out
    yt = np.asarray(y_true, dtype=float).reshape(-1)
    yp = np.asarray(y_pred, dtype=float).reshape(-1)
    n = min(yt.size, yp.size)
    if n == 0:
        return out
    yt = yt[:n]
    yp = yp[:n]
    if not np.all(np.isfinite(yt)) or not np.all(np.isfinite(yp)):
        return out
    err = yp - yt
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    var = float(np.var(yt))
    r2 = float("nan")
    if var > 1e-12:
        sse = float(np.sum(err ** 2))
        sst = float(np.sum((yt - float(np.mean(yt))) ** 2)) + 1e-12
        r2 = float(1.0 - sse / sst)
    std = float(np.std(yt))
    nrmse_std = float(rmse / std) if std > 1e-12 else float("nan")
    out.update({"rmse": rmse, "mae": mae, "r2": r2, "nrmse_std": nrmse_std})
    return out
 

def _plot_variant(
    time: np.ndarray,
    forces_signed: np.ndarray,
    forces_smoothed: np.ndarray,
    force_normal_abs: Optional[np.ndarray],
    stiffness_raw: np.ndarray,
    stiffness_filtered: np.ndarray,
    emg_raw: np.ndarray,
    emg_filtered: np.ndarray,
    out_path: Path,
    title: str,
    filter_label: str,
) -> None:
    if time.ndim != 1 or time.size < 2:
        return
    fig, axes = plt.subplots(3, 1, figsize=(12, 8.0), sharex=True)
    ax_force, ax_stiff, ax_emg = axes

    colors = ["red", "green", "blue"]  # RGB
    axis_labels = ["x", "y", "z"]

    for axis in range(min(3, forces_signed.shape[1])):
        color = colors[axis % len(colors)]
        label = axis_labels[axis % len(axis_labels)]
        # Plot raw force as dashed line
        ax_force.plot(
            time,
            forces_signed[:, axis],
            color=color,
            linewidth=0.8,
            linestyle="--",
            alpha=0.3,
            label=f"|F_{label}| raw" if axis == 0 else None,
        )
        # Plot smoothed force as solid line
        ax_force.plot(
            time,
            forces_smoothed[:, axis],
            color=color,
            linewidth=1.2,
            label=f"|F_{label}| smoothed",
        )
    # Overlay |F_n| if provided
    if force_normal_abs is not None and force_normal_abs.size == time.size:
        ax_force.plot(
            time,
            force_normal_abs,
            color="black",
            linewidth=1.0,
            linestyle=":",
            alpha=0.9,
            label="|F_n|",
        )

    ax_force.set_ylabel("Force [N]")
    ax_force.grid(alpha=0.3)
    ax_force.legend(loc="upper right", fontsize=8)

    for axis in range(min(3, stiffness_raw.shape[1])):
        color = colors[axis % len(colors)]
        label = axis_labels[axis % len(axis_labels)]
        ax_stiff.plot(time, stiffness_raw[:, axis], color=color, linewidth=1.0, linestyle="--", alpha=0.1, label=f"K_{label} raw")
        ax_stiff.plot(time, stiffness_filtered[:, axis], color=color, linewidth=1.3, linestyle="-", label=f"K_{label} {filter_label}")
    ax_stiff.set_ylabel("Stiffness [N/m]")
    ax_stiff.grid(alpha=0.3)
    ax_stiff.legend(loc="upper right", fontsize=8)

    palette = plt.rcParams.get("axes.prop_cycle", None)
    emg_colors = palette.by_key().get("color", []) if palette else []
    if not emg_colors:
        emg_colors = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
        ]
    for ch in range(min(emg_raw.shape[1], 8)):
        color = emg_colors[ch % len(emg_colors)]
        ax_emg.plot(time, emg_raw[:, ch], color=color, linewidth=0.8, linestyle="--", alpha=0.1, label="EMG raw" if ch == 0 else None)
        ax_emg.plot(time, emg_filtered[:, ch], color=color, linewidth=1.2, linestyle="-", label=f"EMG {filter_label}" if ch == 0 else None)
    ax_emg.set_ylabel("EMG")
    ax_emg.set_xlabel("Time [s]")
    ax_emg.grid(alpha=0.1)
    ax_emg.legend(loc="upper right", fontsize=8)

    span = float(time[-1] - time[0])
    pad = 0.03 * span if span > 0 else 0.1
    xmin = float(time[0]) - pad
    xmax = float(time[-1]) + pad
    ax_force.set_xlim(xmin, xmax)

    fig.suptitle(f"{title} [{filter_label}]")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Output writer
# ---------------------------------------------------------------------------

def _save_profile_csv(
    out_path: Path,
    time: np.ndarray,
    forces_signed: np.ndarray,
    stiffness_lp: np.ndarray,
) -> None:
    data = {
        "time_s": np.asarray(time, dtype=float),
        "Fx": forces_signed[:, 0].astype(float),
        "Fy": forces_signed[:, 1].astype(float),
        "Fz": forces_signed[:, 2].astype(float),
        "Kx_lp": stiffness_lp[:, 0].astype(float),
        "Ky_lp": stiffness_lp[:, 1].astype(float),
        "Kz_lp": stiffness_lp[:, 2].astype(float),
    }
    out_df = pd.DataFrame(data)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)


# ---------------------------------------------------------------------------
# Main processing per file
# ---------------------------------------------------------------------------

def process_file(csv_path: Path, output_dir: Path) -> Optional[List[Path]]:
    try:
        df = _load_csv(csv_path)
    except Exception as exc:
        print(f"[skip] {csv_path.name}: load failed ({exc})")
        return None

    time = _time_vector(df)
    fs = _estimate_fs(time)
    if fs is None:
        fs = 200.0
        print(f"[warn] {csv_path.name}: sampling rate unknown, assuming {fs:.1f} Hz")

    # Extract deformation and end-effector position data if available
    deform_circ = df["deform_circ"].values if "deform_circ" in df.columns else np.zeros(len(df))
    deform_ecc = df["deform_ecc"].values if "deform_ecc" in df.columns else np.zeros(len(df))
    # IF finger (ee_if_px/py/pz, fallback to old ee_px/py/pz for backward compatibility)
    ee_if_px = df["ee_if_px"].values if "ee_if_px" in df.columns else (df["ee_px"].values if "ee_px" in df.columns else np.zeros(len(df)))
    ee_if_py = df["ee_if_py"].values if "ee_if_py" in df.columns else (df["ee_py"].values if "ee_py" in df.columns else np.zeros(len(df)))
    ee_if_pz = df["ee_if_pz"].values if "ee_if_pz" in df.columns else (df["ee_pz"].values if "ee_pz" in df.columns else np.zeros(len(df)))
    # MF finger (ee_mf_px/py/pz)
    ee_mf_px = df["ee_mf_px"].values if "ee_mf_px" in df.columns else np.zeros(len(df))
    ee_mf_py = df["ee_mf_py"].values if "ee_mf_py" in df.columns else np.zeros(len(df))
    ee_mf_pz = df["ee_mf_pz"].values if "ee_mf_pz" in df.columns else np.zeros(len(df))
    # TH finger (ee_th_px/py/pz)
    ee_th_px = df["ee_th_px"].values if "ee_th_px" in df.columns else np.zeros(len(df))
    ee_th_py = df["ee_th_py"].values if "ee_th_py" in df.columns else np.zeros(len(df))
    ee_th_pz = df["ee_th_pz"].values if "ee_th_pz" in df.columns else np.zeros(len(df))

    # Compute signed normal component Fn for main force (uses any available prefix)
    try:
        Fn_main = _compute_normal_force(df, prefix=None, n=(0.0, 0.0, 1.0))  # (N,)
    except Exception as exc:
        print(f"[skip] {csv_path.name}: normal force extraction failed ({exc})")
        return None
    forces_calc_smoothed = _smooth_force(Fn_main, fs)
    if forces_calc_smoothed is None:
        print(f"[warn] {csv_path.name}: normal force smoothing failed, using raw normal force")
        forces_calc_smoothed = Fn_main

    emg_cols = _guess_emg_columns(df)
    if not emg_cols:
        print(f"[skip] {csv_path.name}: no EMG columns detected")
        return None

    try:
        emg_raw_input = df[emg_cols].to_numpy(dtype=float)
    except Exception as exc:
        print(f"[skip] {csv_path.name}: EMG extraction failed ({exc})")
        return None

    aligned_emg = _align_emg_to_time(emg_raw_input, time)
    if aligned_emg is None:
        print(f"[skip] {csv_path.name}: unable to align EMG to force timeline")
        return None

    emg_centered = _center_signal(aligned_emg)
    emg_magnitude = np.abs(emg_centered)

    # ultra_smooth_strong variant만 처리
    ultra_strong_emg = _ultra_smooth_strong_emg(emg_magnitude, fs)
    emg_variant = ultra_strong_emg if ultra_strong_emg is not None else emg_magnitude

    P_raw = emg_magnitude.T
    # Use signed normal force (row vector) for T_F calculation
    F_mat = np.asarray(forces_calc_smoothed, dtype=float)[None, :]

    try:
        T_F = compute_tf(P_raw, F_mat)
        H_F = compute_projection_from_tf(T_F)
        T_K, H_K = compute_k_basis_from_force_projector(H_F, target_rank=3)
    except Exception as exc:
        print(f"[skip] {csv_path.name}: projection computation failed ({exc})")
        return None

    # ultra_smooth_strong variant로 stiffness 계산
    P_variant = emg_variant.T
    stiffness_variant = np.maximum(K_MIN, (T_K @ P_variant).T + K_INIT)
    stiffness_raw = np.maximum(K_MIN, (T_K @ P_raw).T + K_INIT)

    # Optional validation printouts
    if VALIDATE_RESULTS:
        print(f"\n[Validation] {csv_path.name}:")
        physics = validate_stiffness_physics(stiffness_variant, forces_calc_smoothed)
        print(
            f"  Physics: frac_positive={physics['frac_positive']:.2f}, in_range_frac={physics['in_range_frac']:.2f}, "
            f"corr_force_stiff={physics['corr_force_stiff']:.3f}, smooth_p95={physics['smooth_p95']:.2f}"
        )
        proj = validate_projection_quality(H_F, H_K)
        print(
            f"  Projection: idemp_F={proj['idemp_F']:.2e}, idemp_K={proj['idemp_K']:.2e}, "
            f"orth={proj['orth']:.2e}, comp={proj['comp']:.2e}"
        )
        # LS diagnostics for F ≈ T_F·P
        F_fit = (T_F @ P_raw).reshape(-1)
        tf_metrics = _regression_metrics(Fn_main, F_fit)
        print(
            "  T_F fit (main normal): "
            f"RMSE={tf_metrics['rmse']:.3f} NRMSE(std)={tf_metrics['nrmse_std']:.3f} R²={tf_metrics['r2']:.3f}"
        )
        emg_corr = validate_emg_stiffness_mapping(emg_variant, stiffness_variant)
        print(f"  EMG-Stiff correlation (mag): {emg_corr:.3f}")

    out_paths: List[Path] = []

    # 각 손가락별로 force와 stiffness 처리
    finger_names = ["th", "if", "mf"]  # thumb, index, middle
    sensor_names = ["s1", "s2", "s3"]
    all_forces = []
    all_forces_for_csv = []  # 원본 부호 유지 (CSV 저장용)
    all_stiffness = []
    all_fn_abs = []
    
    per_finger_metrics: List[Tuple[str, Dict[str, float]]] = []
    for sensor in sensor_names:
        force_cols = [f"{sensor}_fx", f"{sensor}_fy", f"{sensor}_fz"]
        if all(col in df.columns for col in force_cols):
            # Baseline removal (preserves sign)
            forces_finger = df[force_cols].to_numpy(dtype=float)
            rest_samples = max(1, int(len(forces_finger) * 0.1))
            baseline = np.mean(forces_finger[:rest_samples, :], axis=0, keepdims=True)
            forces_finger_centered = forces_finger - baseline
            
            # For CSV: smooth the signed centered force
            forces_finger_smoothed_signed = _smooth_force(forces_finger_centered, fs)
            if forces_finger_smoothed_signed is None:
                forces_finger_smoothed_signed = forces_finger_centered
            all_forces_for_csv.append(forces_finger_smoothed_signed)
            
            # For plotting: use absolute values
            forces_finger_abs = np.abs(forces_finger_centered)
            forces_finger_smoothed_abs = _smooth_force(forces_finger_abs, fs)
            if forces_finger_smoothed_abs is None:
                forces_finger_smoothed_abs = forces_finger_abs
            all_forces.append(forces_finger_smoothed_abs)

            # For calculation: signed normal force component per finger
            Fn_finger = _compute_normal_force(df, prefix=sensor, n=(0.0, 0.0, 1.0))
            Fn_finger_smoothed = _smooth_force(Fn_finger, fs)
            if Fn_finger_smoothed is None:
                Fn_finger_smoothed = Fn_finger
            all_fn_abs.append(np.abs(Fn_finger_smoothed))

            F_finger = np.asarray(Fn_finger_smoothed, dtype=float)[None, :]
            try:
                T_F_finger = compute_tf(P_raw, F_finger)
                H_F_finger = compute_projection_from_tf(T_F_finger)
                T_K_finger, _ = compute_k_basis_from_force_projector(H_F_finger, target_rank=3)
                stiffness_finger = np.maximum(K_MIN, (T_K_finger @ P_variant).T + K_INIT)
                all_stiffness.append(stiffness_finger)
                if VALIDATE_RESULTS:
                    F_fit_f = (T_F_finger @ P_raw).reshape(-1)
                    per_finger_metrics.append((sensor, _regression_metrics(Fn_finger_smoothed, F_fit_f)))
            except Exception:
                # 실패 시 기본 stiffness 사용
                all_stiffness.append(stiffness_variant)
        else:
            all_forces.append(np.zeros((len(time), 3)))
            all_forces_for_csv.append(np.zeros((len(time), 3)))
            all_stiffness.append(stiffness_variant)
            all_fn_abs.append(np.zeros(len(time)))

    # Print per-finger LS diagnostics together
    if VALIDATE_RESULTS and per_finger_metrics:
        msg = []
        for sensor, m in per_finger_metrics:
            try:
                msg.append(
                    f"{sensor}: RMSE={m['rmse']:.3f} NRMSE(std)={m['nrmse_std']:.3f} R²={m['r2']:.3f}"
                )
            except Exception:
                pass
        if msg:
            print("  T_F fit per finger: " + " | ".join(msg))
    
    # PNG 플롯 3개 생성 (각 손가락별) - 양수 force 사용
    for idx, (finger_name, sensor) in enumerate(zip(finger_names, sensor_names)):
        out_png = output_dir / f"{csv_path.stem}_{finger_name}_smooth.png"
        try:
            _plot_variant(
                time,
                all_forces[idx],  # plot (abs, per-axis)
                all_forces[idx],  # plot (abs, per-axis smoothed)
                all_fn_abs[idx],  # overlay |F_n|
                stiffness_raw,
                all_stiffness[idx],
                emg_magnitude,
                emg_variant,
                out_png,
                csv_path.stem,
                f"{finger_name} ultra-smooth-strong",
            )
            out_paths.append(out_png)
        except Exception as exc:
            print(f"[warn] {csv_path.name}: PNG plot for {finger_name} failed ({exc})")

    # 하나의 CSV 파일 생성 (모든 손가락 데이터 포함)
    # Use signed smoothed forces for CSV (consistent with raw demo coordinate system)
    all_forces_csv_combined = np.hstack(all_forces_for_csv)  # shape: (N, 9)
    
    # CSV 저장
    out_csv = output_dir / f"{csv_path.stem}_paper_profile.csv"
    try:
        data = {"time_s": np.asarray(time, dtype=float)}
        for i, (finger_name, sensor) in enumerate(zip(finger_names, sensor_names)):
            data[f"{sensor}_fx"] = all_forces_csv_combined[:, i*3+0].astype(float)
            data[f"{sensor}_fy"] = all_forces_csv_combined[:, i*3+1].astype(float)
            data[f"{sensor}_fz"] = all_forces_csv_combined[:, i*3+2].astype(float)
            data[f"{finger_name}_k1"] = all_stiffness[i][:, 0].astype(float)
            data[f"{finger_name}_k2"] = all_stiffness[i][:, 1].astype(float)
            data[f"{finger_name}_k3"] = all_stiffness[i][:, 2].astype(float)
        # Add deformation and end-effector position columns
        data["deform_circ"] = np.asarray(deform_circ, dtype=float)
        data["deform_ecc"] = np.asarray(deform_ecc, dtype=float)
        # IF finger EE position
        data["ee_if_px"] = np.asarray(ee_if_px, dtype=float)
        data["ee_if_py"] = np.asarray(ee_if_py, dtype=float)
        data["ee_if_pz"] = np.asarray(ee_if_pz, dtype=float)
        # MF finger EE position
        data["ee_mf_px"] = np.asarray(ee_mf_px, dtype=float)
        data["ee_mf_py"] = np.asarray(ee_mf_py, dtype=float)
        data["ee_mf_pz"] = np.asarray(ee_mf_pz, dtype=float)
        # TH finger EE position
        data["ee_th_px"] = np.asarray(ee_th_px, dtype=float)
        data["ee_th_py"] = np.asarray(ee_th_py, dtype=float)
        data["ee_th_pz"] = np.asarray(ee_th_pz, dtype=float)
        out_df = pd.DataFrame(data)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(out_csv, index=False)
        out_paths.append(out_csv)
    except Exception as exc:
        print(f"[warn] {csv_path.name}: CSV save failed ({exc})")

    print(f"[ok] {csv_path.name}: generated {len(out_paths)} artifact(s)")
    for path in out_paths:
        print(f"       -> {path}")
    return out_paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def gather_csvs(input_path: Path) -> List[Path]:
    """Gather CSV files.
    - If input_path is a file: return it.
    - If input_path is a directory: gather recursively (support date subfolders).
    - If path doesn't exist: error out (we only process explicit/success dir).
    """
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        # Recursively collect CSVs (including dated subfolders)
        csvs = sorted([p for p in input_path.rglob("*.csv") if p.is_file()])
        return csvs
    raise FileNotFoundError(f"Input path not found: {input_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate stiffness profiles (paper ver.)")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
    help="CSV file or directory of logs (defaults to <pkg>/outputs/logs/success). Supports recursion.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(PKG_DIR / "outputs" / "analysis" / "stiffness_profiles"),
        help="Directory for output artifacts (defaults to <pkg>/outputs/analysis/stiffness_profiles)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Print validation metrics (physics, projection, EMG-stiffness) per file.",
        default=True,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    global VALIDATE_RESULTS
    VALIDATE_RESULTS = bool(getattr(args, "validate", False))
    csv_files = gather_csvs(args.input)
    if not csv_files:
        raise SystemExit(f"No CSV files found under {args.input}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Processing {len(csv_files)} file(s) from {args.input} ...")
    for csv_path in csv_files:
        process_file(csv_path, args.output_dir)


if __name__ == "__main__":
    main()
