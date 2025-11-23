#!/usr/bin/env python3
"""MuJoCo Diffusion Policy stiffness control demo.

Loads a trained diffusion policy artifact (diffusion_c.pt or diffusion_t.pt) plus scalers
and generates stiffness actions (9D: th/if/mf k1..k3) either:
  - replay: for a provided observation CSV (open-loop sampled once then replay)
  - online: per-step sampling conditioned on current fingertip positions (forces/deform placeholders zero)

Supports DDPM or DDIM sampler and multiple sample averaging.

Artifact autodetect: searches outputs/.../artifacts for latest diffusion_c.pt (or diffusion_t.pt if temporal flag requested).

Usage examples:
  python3 mujoco_diffusion_control_demo.py --mode online --sampler ddim --eta 0.2 --n-samples 4
  python3 mujoco_diffusion_control_demo.py --mode replay --obs-csv path/to/trajectory.csv --sampler ddpm

Notes:
  - Online mode constructs an observation vector with fingertip site positions + zeros for unavailable sensors.
  - Temporal diffusion artifacts (diffusion_t.pt) expect sequence input; we pass single-step which becomes length-1 sequence.
  - Stiffness post-processing: simple Kp/Kd derivation (same as BC demo). You may clamp via --min-k if needed.
"""
from __future__ import annotations
import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, List, Optional
import numpy as np
import pandas as pd  # type: ignore
import mujoco
import mujoco.viewer

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F  # noqa: F401
except ImportError:
    torch = None
    nn = None

# ------------------- Constants -------------------
_THIS_FILE = Path(__file__).resolve()
_PKG_ROOT = _THIS_FILE.parents[2]
_OUTPUTS_ROOT = _PKG_ROOT / 'outputs'
DEFAULT_MJCF_PATH = '/home/songwoo/git/ur_dclaw/dclaw_finger_description/urdf/dclaw_finger_mjcf_finall_inertia_edit.xml'
SITE_NAMES = ['FFtip','MFtip','THtip']
FINGER_PREFIXES = ['if','mf','th']
OBS_FALLBACK = [
    's1_fx','s1_fy','s1_fz',
    's2_fx','s2_fy','s2_fz',
    's3_fx','s3_fy','s3_fz',
    'deform_ecc',
    'ee_if_px','ee_if_py','ee_if_pz',
    'ee_mf_px','ee_mf_py','ee_mf_pz',
    'ee_th_px','ee_th_py','ee_th_pz'
]
ACT_COLUMNS = [
    'th_k1','th_k2','th_k3',
    'if_k1','if_k2','if_k3',
    'mf_k1','mf_k2','mf_k3'
]

# ------------------- Diffusion model (inference only) -------------------
if torch is not None:
    # Reuse definition from benchmark script via import for reliability.
    import sys
    sys.path.append(str(_THIS_FILE.parent))
    from run_stiffness_policy_benchmarks import DiffusionPolicyBaseline  # type: ignore
else:
    DiffusionPolicyBaseline = None  # type: ignore

# ------------------- Helpers -------------------

def autodetect_artifact(temporal: bool=False) -> Optional[Path]:
    names = ['diffusion_t.pt','diffusion_c.pt'] if temporal else ['diffusion_c.pt','diffusion_t.pt']
    roots = [
        _OUTPUTS_ROOT / 'artifacts',
        _OUTPUTS_ROOT / 'models' / 'policy_learning_unified' / 'artifacts',
        _OUTPUTS_ROOT / 'policy_learning_global_tk_unified' / 'artifacts',
    ]
    best = None
    for r in roots:
        if not r.exists(): continue
        for d in r.iterdir():
            if not d.is_dir(): continue
            for nm in names:
                p = d / nm
                if p.exists():
                    m = p.stat().st_mtime
                    if best is None or m > best[0]: best = (m, p)
    return best[1] if best else None

def load_diffusion(artifact_path: Path):
    if not artifact_path.exists():
        raise RuntimeError(f'Artifact not found: {artifact_path}')
    if DiffusionPolicyBaseline is None or torch is None:
        raise RuntimeError('PyTorch diffusion baseline unavailable.')
    checkpoint = torch.load(artifact_path, map_location='cpu')
    cfg = checkpoint.get('config', {})
    infer = DiffusionPolicyBaseline(
        obs_dim=cfg.get('obs_dim'),
        act_dim=cfg.get('act_dim'),
        timesteps=cfg.get('timesteps', 50),
        hidden_dim=cfg.get('hidden_dim', 256),
        time_dim=cfg.get('time_dim', 64),
        lr=1e-4,
        batch_size=1,
        epochs=0,
        seed=cfg.get('seed', 0),
        log_name='diffusion_infer',
        temporal=cfg.get('temporal', False),
    )
    state_dict = checkpoint.get('state_dict', checkpoint)
    infer.model.load_state_dict(state_dict)  # type: ignore
    return infer, cfg

def load_scalers(art_dir: Path):
    pk = art_dir / 'scalers.pkl'
    if not pk.exists(): raise RuntimeError(f'scalers.pkl missing in {art_dir}')
    with pk.open('rb') as fh:
        scalers = __import__('pickle').load(fh)
    return scalers['obs_scaler'], scalers['act_scaler']

def build_obs_from_sim(obs_columns: List[str], site_positions: np.ndarray) -> np.ndarray:
    mapping = {
        'ee_if_px': site_positions[0,0],'ee_if_py': site_positions[0,1],'ee_if_pz': site_positions[0,2],
        'ee_mf_px': site_positions[1,0],'ee_mf_py': site_positions[1,1],'ee_mf_pz': site_positions[1,2],
        'ee_th_px': site_positions[2,0],'ee_th_py': site_positions[2,1],'ee_th_pz': site_positions[2,2],
    }
    arr=[]
    for c in obs_columns:
        if c.startswith('ee_'): arr.append(mapping.get(c,0.0))
        elif c.startswith('s') and '_f' in c: arr.append(0.0)
        elif c=='deform_ecc': arr.append(0.0)
        else: arr.append(0.0)
    return np.asarray(arr, dtype=float)

def load_replay_obs(csv_path: Path, obs_columns: List[str]) -> np.ndarray:
    df = pd.read_csv(csv_path)
    for finger in ['if','mf','th']:
        for axis in ['px','py','pz']:
            new_col = f'ee_{finger}_{axis}'; legacy=f'ee_{axis}'
            if new_col not in df.columns and legacy in df.columns: df[new_col]=df[legacy]
    miss=[c for c in obs_columns if c not in df.columns]
    if miss: raise RuntimeError(f'Missing columns in replay CSV: {miss}')
    return df[obs_columns].to_numpy(dtype=float)

def autodetect_latest_position_csv() -> Optional[Path]:
    """Auto-detect the latest position CSV from outputs/logs/success."""
    success_dir = _OUTPUTS_ROOT / 'logs' / 'success'
    if not success_dir.exists():
        return None
    csv_files = sorted(success_dir.glob('*_synced.csv'), key=lambda p: p.stat().st_mtime, reverse=True)
    return csv_files[0] if csv_files else None

def load_desired_positions(csv_path: Path) -> dict:
    """Load desired EE positions for all fingers from CSV. Returns dict with keys 'if','mf','th'."""
    if not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path)
    positions = {}
    for finger in FINGER_PREFIXES:
        cols = [f'ee_{finger}_px', f'ee_{finger}_py', f'ee_{finger}_pz']
        if all(c in df.columns for c in cols):
            positions[finger] = df[cols].to_numpy(dtype=float)
    return positions

def add_trajectory_visualization(scene, desired_positions: dict, finger_colors: dict):
    """Add sphere markers for desired trajectory visualization."""
    sample_step = max(1, min([pos.shape[0] for pos in desired_positions.values()]) // 50)

    for finger, positions in desired_positions.items():
        color = finger_colors.get(finger, [0.6, 0.6, 0.6, 0.7])
        for i in range(0, positions.shape[0], sample_step):
            if scene.ngeom >= scene.maxgeom:
                break
            pos = positions[i]
            geom = scene.geoms[scene.ngeom]
            try:
                geom.type = mujoco.mjtGeom.mjGEOM_SPHERE  # type: ignore[attr-defined]
            except AttributeError:
                geom.type = 0
            geom.size[0] = 0.003
            geom.pos[:] = pos
            if hasattr(geom, 'mat'):
                try:
                    if geom.mat.shape == (3, 3):
                        geom.mat[:, :] = np.eye(3)
                    elif np.prod(geom.mat.shape) == 9:
                        geom.mat[:] = np.eye(3).reshape(-1)
                except Exception:
                    pass
            geom.rgba[:] = color
            scene.ngeom += 1

# ------------------- CLI -------------------

def parse_args():
    p=argparse.ArgumentParser(description='MuJoCo diffusion policy stiffness control demo')
    p.add_argument('--artifact', type=Path, default=None, help='Path to diffusion_c.pt or diffusion_t.pt (auto-detect if omitted)')
    p.add_argument('--mode', choices=['online','replay'], default='online')
    p.add_argument('--obs-csv', type=Path, default=None, help='Observation CSV for replay mode')
    p.add_argument('--position-csv', type=Path, default=None, help='Desired position CSV (with ee_if/mf/th_px/py/pz columns)')
    p.add_argument('--sampler', choices=['ddpm','ddim'], default='ddpm')
    p.add_argument('--eta', type=float, default=0.0, help='DDIM eta (noise strength)')
    p.add_argument('--n-samples', type=int, default=1, help='Number of diffusion samples to average')
    p.add_argument('--stiffness-scale', type=float, default=1.0)
    p.add_argument('--min-k', type=float, default=1.0, help='Clamp each stiffness component to at least this')
    p.add_argument('--finger-target-offset', type=float, default=0.0, help='Goal offset in +z')
    p.add_argument('--max-steps', type=int, default=5000)
    p.add_argument('--temporal', action='store_true', help='Prefer temporal artifact (diffusion_t.pt) if available')
    p.add_argument('--mjcf-path', type=Path, default=Path(DEFAULT_MJCF_PATH))
    return p.parse_args()

# ------------------- Main -------------------

def main():
    args=parse_args()
    art_path = args.artifact or autodetect_artifact(temporal=args.temporal)
    if art_path is None: raise RuntimeError('Could not auto-detect diffusion artifact.')
    art_dir = art_path.parent
    print(f'[info] Using diffusion artifact: {art_path}')
    infer, cfg = load_diffusion(art_path)
    obs_scaler, act_scaler = load_scalers(art_dir)
    manifest_path = art_dir / 'manifest.json'
    obs_columns = OBS_FALLBACK
    if manifest_path.exists():
        try:
            with manifest_path.open('r',encoding='utf-8') as fh:
                manifest=json.load(fh)
            if 'obs_columns' in manifest: obs_columns = manifest['obs_columns']
        except Exception: pass
    print(f'[info] Model cfg: temporal={cfg.get("temporal", False)} timesteps={cfg.get("timesteps")} sampler={args.sampler}')

    if args.mode=='replay' and args.obs_csv is None:
        raise RuntimeError('--obs-csv required for replay mode')

    # Load desired positions (auto-detect if not provided)
    position_csv = args.position_csv or autodetect_latest_position_csv()
    desired_positions = {}
    if position_csv:
        print(f'[info] Using position CSV: {position_csv}')
        desired_positions = load_desired_positions(position_csv)
        if desired_positions:
            print(f'[info] Loaded desired positions for fingers: {", ".join(desired_positions.keys())}')
    else:
        print('[warn] No position CSV found, using current position + offset')

    if args.mode=='replay':
        raw_obs = load_replay_obs(args.obs_csv, obs_columns)
        obs_scaled = obs_scaler.transform(raw_obs)
        pred_scaled = infer.predict(obs_scaled, n_samples=args.n_samples, sampler=args.sampler, eta=args.eta)
        pred = act_scaler.inverse_transform(pred_scaled)
        stiffness_seq = pred * args.stiffness_scale
        print(f'[info] Generated stiffness sequence shape={stiffness_seq.shape}')
    else:
        stiffness_seq = None

    if not args.mjcf_path.exists(): raise RuntimeError(f'MJCF not found: {args.mjcf_path}')
    mj_model = mujoco.MjModel.from_xml_path(str(args.mjcf_path))  # type: ignore
    mj_data = mujoco.MjData(mj_model)  # type: ignore
    mujoco.mj_forward(mj_model, mj_data)  # type: ignore

    site_ids=[]
    for nm in SITE_NAMES:
        try:
            sid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, nm)  # type: ignore
        except Exception:
            sid = -1; print(f'[warn] Missing site {nm}')
        site_ids.append(sid)

    # Color coding for trajectory visualization
    finger_colors = {
        'if': [1.0, 0.0, 0.0, 0.4],  # red
        'mf': [0.0, 1.0, 0.0, 0.4],  # green
        'th': [0.0, 0.0, 1.0, 0.4],  # blue
    }

    print('[info] Starting diffusion control simulation... (ESC to exit)')
    step=0
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:  # type: ignore
        while viewer.is_running() and step < args.max_steps:
            t0=time.time()
            
            # Add trajectory visualization on first frame
            if step == 0 and desired_positions:
                scene = viewer.user_scn
                add_trajectory_visualization(scene, desired_positions, finger_colors)
            
            if stiffness_seq is not None:
                if step >= stiffness_seq.shape[0]:
                    print('[info] Replay finished.')
                    break
                act9 = stiffness_seq[step]
            else:
                positions=[]
                for sid in site_ids:
                    positions.append(mj_data.site_xpos[sid].copy() if sid>=0 else np.zeros(3))
                site_pos = np.vstack(positions)
                obs_vec = build_obs_from_sim(obs_columns, site_pos)
                obs_scaled = obs_scaler.transform(obs_vec.reshape(1,-1))
                pred_scaled = infer.predict(obs_scaled, n_samples=args.n_samples, sampler=args.sampler, eta=args.eta)
                pred_raw = act_scaler.inverse_transform(pred_scaled)
                act9 = pred_raw[0] * args.stiffness_scale
            th_K, if_K, mf_K = act9[0:3], act9[3:6], act9[6:9]
            # Apply PD per finger
            total_tau = np.zeros(mj_model.nv)
            for prefix, K, sid in zip(FINGER_PREFIXES, [if_K, mf_K, th_K], site_ids):
                if sid < 0: continue
                current = mj_data.site_xpos[sid].copy()
                # Use desired position from CSV if available, else current + offset
                if prefix in desired_positions and step < desired_positions[prefix].shape[0]:
                    goal = desired_positions[prefix][step]
                else:
                    goal = current + np.array([0.0,0.0,args.finger_target_offset])
                jacp = np.zeros((3, mj_model.nv)); jacr = np.zeros((3, mj_model.nv))
                mujoco.mj_jacSite(mj_model, mj_data, jacp, jacr, sid)  # type: ignore
                vel = jacp @ mj_data.qvel
                pos_err = goal - current
                K_clamped = np.clip(K, args.min_k, None)
                Kp = np.diag(K_clamped)
                Kd = np.diag(2.0 * np.sqrt(K_clamped))
                F = Kp @ pos_err - Kd @ vel
                tau = jacp.T @ F
                total_tau[:tau.shape[0]] += tau
            mj_data.qfrc_applied[:] = total_tau
            mujoco.mj_step(mj_model, mj_data)  # type: ignore
            step += 1
            if step % 200 == 0: print(f'[step {step}] th_K={th_K} if_K={if_K} mf_K={mf_K}')
            viewer.sync()
            remain = mj_model.opt.timestep - (time.time() - t0)
            if remain>0: time.sleep(remain)
    print('[info] Simulation finished.')

if __name__ == '__main__':
    main()
