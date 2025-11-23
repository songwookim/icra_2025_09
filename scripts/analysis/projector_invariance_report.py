#!/usr/bin/env python3
"""Force 변환별 Force Subspace Projector (H_F) 불변성 리포트.

Raw / |F| / -|F| (negative-only) 세 가지 변환에 대해 동일 EMG 특징 행렬 P_global 사용 시
각 손가락(finger)에 대한 T_F, H_F를 계산하고 차이를 수치화한다.

지표 (finger 단위):
  - fro_norm: ||H_F_variant - H_F_raw||_F
  - fro_rel:  fro_norm / ||H_F_raw||_F
  - eig_L2:  sqrt(sum((eig_variant - eig_raw)^2)) (고유치 내림차순 정렬 후 비교)
  - trace_diff: |trace(H_F_variant) - trace(H_F_raw)|
  - principal_angle_max: T_F 열공간과 variant 열공간 사이 최대 주각 (deg)
  - canonical_corr_mean: T_F vs variant 열공간 정규화 후 상관 평균

최종 출력 JSON + CSV.

Usage:
  python3 projector_invariance_report.py --input outputs/logs/success --out-dir outputs/analysis
"""
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

SCRIPT_DIR = Path(__file__).resolve().parent
PKG_DIR = SCRIPT_DIR.parent.parent  # .../src/hri_falcon_robot_bridge
DEFAULT_INPUT = PKG_DIR / "outputs" / "logs" / "success"
EPS = 1e-8

# ------------------------------ Utils ------------------------------

def _load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("CSV empty")
    return df

def _time_vector(df: pd.DataFrame) -> np.ndarray:
    if {"t_sec", "t_nanosec"}.issubset(df.columns):
        t = df["t_sec"].to_numpy(dtype=float) + df["t_nanosec"].to_numpy(dtype=float)*1e-9
    elif "t_sec" in df.columns:
        t = df["t_sec"].to_numpy(dtype=float)
    elif "time_s" in df.columns:
        t = df["time_s"].to_numpy(dtype=float)
    else:
        t = np.arange(len(df), dtype=float)
    return t - t[0]

def _estimate_fs(t: np.ndarray) -> Optional[float]:
    if t.ndim != 1 or len(t) < 3: return None
    dt = np.diff(t)
    dt = dt[(dt > 0) & np.isfinite(dt)]
    if dt.size == 0: return None
    return float(1.0/np.median(dt))

def _guess_emg_columns(df: pd.DataFrame) -> List[str]:
    cols = []
    for ch in range(1, 9):
        nm = f"emg_ch{ch}"
        if nm in df.columns:
            cols.append(nm)
    return cols

def _align_emg_to_time(emg: np.ndarray, time: np.ndarray) -> Optional[np.ndarray]:
    if emg.shape[0] == time.shape[0]:
        return emg
    if time.size < 2: return None
    src_times = np.linspace(time[0], time[-1], num=emg.shape[0], dtype=float)
    out = np.empty((time.shape[0], emg.shape[1]), dtype=float)
    for i in range(emg.shape[1]):
        out[:, i] = np.interp(time, src_times, emg[:, i])
    return out

def _center(signal: np.ndarray, frac: float = 0.1) -> np.ndarray:
    n = signal.shape[0]
    take = max(1, int(round(n*frac)))
    base = np.median(signal[:take, :], axis=0, keepdims=True)
    return signal - base

def _smooth_force(force: np.ndarray, fs: Optional[float], cutoff: float = 3.0) -> np.ndarray:
    if fs is None or fs <= 0 or fs <= 2*cutoff:
        return force
    try:
        sos = butter(2, cutoff, btype="low", fs=fs, output="sos")
        return sosfiltfilt(sos, force, axis=0)
    except Exception:
        return force

# ------------------------------ Linear Algebra ------------------------------

def compute_tf(P: np.ndarray, F: np.ndarray, reg: float = 1e-6) -> np.ndarray:
    PPt = P @ P.T
    return F @ P.T @ np.linalg.inv(PPt + reg * np.eye(PPt.shape[0]))

def compute_projection_from_tf(T_F: np.ndarray, reg: float = 1e-6) -> np.ndarray:
    A = T_F @ T_F.T
    H_F = T_F.T @ np.linalg.inv(A + reg * np.eye(A.shape[0])) @ T_F
    H_F = 0.5*(H_F + H_F.T)
    return H_F

def principal_angles(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    # Orthonormal bases via SVD
    Ua, _, _ = np.linalg.svd(A, full_matrices=False)
    Ub, _, _ = np.linalg.svd(B, full_matrices=False)
    M = Ua.T @ Ub
    s = np.linalg.svd(M, compute_uv=False)
    s = np.clip(s, -1.0, 1.0)
    return np.degrees(np.arccos(s))  # angles in degrees

# ------------------------------ Core ------------------------------

def aggregate_global(input_dir: Path) -> Dict[str, Dict[str, List[np.ndarray]]]:
    sensor_map = {"th": "s1", "if": "s2", "mf": "s3"}
    data = {f: {"emg": [], "force_raw": []} for f in sensor_map.keys()}
    csv_files = sorted(input_dir.glob("*.csv"))
    for p in csv_files:
        if p.name.endswith("_paper_profile.csv"): continue
        try:
            df = _load_csv(p)
            t = _time_vector(df); fs = _estimate_fs(t) or 200.0
            emg_cols = _guess_emg_columns(df)
            if not emg_cols: continue
            emg_raw = df[emg_cols].to_numpy(dtype=float)
            emg_aligned = _align_emg_to_time(emg_raw, t)
            if emg_aligned is None: continue
            emg_centered = _center(emg_aligned)
            emg_mag = np.abs(emg_centered)
            # ultra-smooth (simplified: moving average with large window)
            win = max(1, int(round(0.6 * (fs or 200))))
            if win > 1:
                kernel = np.ones((win,))/win
                emg_smoothed = np.vstack([np.convolve(emg_mag[:, i], kernel, mode='same') for i in range(emg_mag.shape[1])]).T
            else:
                emg_smoothed = emg_mag
            for finger, sensor in sensor_map.items():
                cols = [f"{sensor}_fx", f"{sensor}_fy", f"{sensor}_fz"]
                if not all(c in df.columns for c in cols):
                    continue
                F = df[cols].to_numpy(dtype=float)
                rest = max(1, int(len(F)*0.1))
                base = np.mean(F[:rest, :], axis=0, keepdims=True)
                F_centered = F - base
                F_smoothed = _smooth_force(F_centered, fs)
                data[finger]["emg"].append(emg_smoothed)
                data[finger]["force_raw"].append(F_smoothed)
        except Exception:
            continue
    return data

def build_global_matrices(data: Dict[str, Dict[str, List[np.ndarray]]]) -> Dict[str, Dict[str, np.ndarray]]:
    out = {}
    for finger, comp in data.items():
        if not comp["emg"] or not comp["force_raw"]: continue
        P_global = np.vstack(comp["emg"]).T  # (d, N)
        F_raw = np.vstack(comp["force_raw"]).T  # (3, N)
        F_abs = np.abs(F_raw)
        F_neg = -np.abs(F_raw)  # negative-only
        out[finger] = {"P": P_global, "F_raw": F_raw, "F_abs": F_abs, "F_neg": F_neg}
    return out

def metrics_for_variants(P: np.ndarray, F_raw: np.ndarray, F_variant: np.ndarray) -> Dict[str, float]:
    T_raw = compute_tf(P, F_raw)
    H_raw = compute_projection_from_tf(T_raw)
    T_var = compute_tf(P, F_variant)
    H_var = compute_projection_from_tf(T_var)
    eig_raw, _ = np.linalg.eigh(H_raw)
    eig_var, _ = np.linalg.eigh(H_var)
    eig_raw = np.sort(eig_raw)[::-1]
    eig_var = np.sort(eig_var)[::-1]
    fro_norm = float(np.linalg.norm(H_var - H_raw))
    fro_rel = float(fro_norm / (np.linalg.norm(H_raw) + EPS))
    eig_L2 = float(np.sqrt(np.sum((eig_var - eig_raw)**2)))
    trace_diff = float(abs(np.trace(H_var) - np.trace(H_raw)))
    angles = principal_angles(T_raw, T_var)
    principal_angle_max = float(angles[0]) if angles.size else 0.0
    canonical_corr_mean = float(np.mean(np.cos(np.radians(angles)))) if angles.size else 1.0
    return {
        "fro_norm": fro_norm,
        "fro_rel": fro_rel,
        "eig_L2": eig_L2,
        "trace_diff": trace_diff,
        "principal_angle_max_deg": principal_angle_max,
        "canonical_corr_mean": canonical_corr_mean,
    }

def run(input_dir: Path, out_dir: Path) -> Tuple[Path, Path]:
    data = aggregate_global(input_dir)
    globals_ = build_global_matrices(data)
    report = {}
    rows = []
    for finger, mats in globals_.items():
        P = mats["P"]; F_raw = mats["F_raw"]; F_abs = mats["F_abs"]; F_neg = mats["F_neg"]
        m_abs = metrics_for_variants(P, F_raw, F_abs)
        m_neg = metrics_for_variants(P, F_raw, F_neg)
        report[finger] = {"abs_vs_raw": m_abs, "neg_vs_raw": m_neg, "dims": {"P": list(P.shape), "F_raw": list(F_raw.shape)}}
        rows.append({"finger": finger, "variant": "abs", **m_abs})
        rows.append({"finger": finger, "variant": "neg", **m_neg})
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "projector_invariance_report.json"
    csv_path = out_dir / "projector_invariance_report.csv"
    with json_path.open("w") as f: json.dump(report, f, indent=2)
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return json_path, csv_path

def parse_args():
    p = argparse.ArgumentParser(description="Force projector invariance report")
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Directory of demo CSVs")
    p.add_argument("--out-dir", type=Path, default=PKG_DIR / "outputs" / "analysis", help="Output directory")
    return p.parse_args()

def main():
    args = parse_args()
    json_path, csv_path = run(args.input, args.out_dir)
    print("Saved invariance report:")
    print("  JSON:", json_path)
    print("  CSV :", csv_path)

if __name__ == "__main__":
    main()
