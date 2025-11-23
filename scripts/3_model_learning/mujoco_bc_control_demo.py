#!/usr/bin/env python3
"""MuJoCo BC control demo.

Loads a trained behavior cloning (BC) stiffness policy (unified 9D) and applies
predicted stiffness (th/if/mf 3×3) to drive Cartesian impedance targets of each finger.

Two modes:
  replay  : Use an observation CSV (with columns matching training OBS_COLUMNS) to
            generate a stiffness sequence once (open-loop) then replay in sim.
  online  : Build observation vector each step from current fingertip positions plus
            zeros for force/deformity (best-effort if real sensors unavailable) and
            query BC model every step (closed-loop surrogate).

Artifact autodetect:
  If --artifacts-dir omitted, searches candidate dirs under outputs/ for the latest
  directory containing bc.pt and scalers.pkl.

Usage examples:
  python3 mujoco_bc_control_demo.py --mode replay --obs-csv outputs/analysis/stiffness_profiles_global_tk/sample.csv
  python3 mujoco_bc_control_demo.py --mode online --finger-target-offset 0.05

Notes:
  - Real-time force/deform observations are replaced with zeros in online mode.
  - EE positions are taken from MuJoCo sites FFtip, MFtip, THtip.
  - If observation scaler expects force magnitudes, zeros may produce out-of-distribution predictions.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from typing import List, Optional, Tuple
import time
import numpy as np
import pandas as pd  # type: ignore
import mujoco
import mujoco.viewer

try:
    import torch
    import torch.nn as nn
except ImportError:  # Provide stubs so static analysis doesn't error; runtime will raise when used.
    class _NoGradCtx:
        def __enter__(self):
            raise RuntimeError('PyTorch not available for BC demo.')
        def __exit__(self, exc_type, exc, tb):
            return False
    class _TorchStub:
        def no_grad(self):
            return _NoGradCtx()
        def from_numpy(self, *_a, **_k):
            raise RuntimeError('PyTorch not available for BC demo.')
        def load(self, *_a, **_k):
            raise RuntimeError('PyTorch not available for BC demo.')
    class _LinearStub:
        def __init__(self, *_a, **_k):
            raise RuntimeError('PyTorch not available for BC demo.')
    class _ReLUStub:
        def __init__(self, *_a, **_k):
            raise RuntimeError('PyTorch not available for BC demo.')
    class _SequentialStub:
        def __init__(self, *_a, **_k):
            raise RuntimeError('PyTorch not available for BC demo.')
        def __call__(self, *_a, **_k):
            raise RuntimeError('PyTorch not available for BC demo.')
    class _NNStub:
        Linear = _LinearStub
        ReLU = _ReLUStub
        Sequential = _SequentialStub
    torch = _TorchStub()  # type: ignore
    nn = _NNStub()  # type: ignore

# Project root resolution (same heuristic as other scripts)
_THIS_FILE = Path(__file__).resolve()
_PKG_ROOT = _THIS_FILE.parents[2]
_OUTPUTS_ROOT = _PKG_ROOT / 'outputs'

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
SITE_NAMES = ['FFtip','MFtip','THtip']
FINGER_PREFIXES = ['if','mf','th']  # order aligned with ACT_COLUMNS groups

DEFAULT_MJCF_PATH = '/home/songwoo/git/ur_dclaw/dclaw_finger_description/urdf/dclaw_finger_mjcf_finall_inertia_edit.xml'

class BehaviorCloningModel(nn.Module):  # type: ignore[misc]
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

def autodetect_artifacts() -> Optional[Path]:
    candidates_roots = [
        _OUTPUTS_ROOT / 'artifacts',
        _OUTPUTS_ROOT / 'models' / 'policy_learning_unified' / 'artifacts',
        _OUTPUTS_ROOT / 'policy_learning_global_tk_unified' / 'artifacts',
    ]
    best: Optional[Tuple[float, Path]] = None
    for root in candidates_roots:
        if not root.exists():
            continue
        for d in root.iterdir():
            if not d.is_dir():
                continue
            bc = d / 'bc.pt'
            sc = d / 'scalers.pkl'
            if bc.exists() and sc.exists():
                mtime = bc.stat().st_mtime
                if best is None or mtime > best[0]:
                    best = (mtime, d)
    return best[1] if best else None

def load_bc_artifacts(art_dir: Path):
    if torch is None:
        raise RuntimeError('PyTorch not available for BC demo.')
    bc_path = art_dir / 'bc.pt'
    scalers_path = art_dir / 'scalers.pkl'
    manifest_path = art_dir / 'manifest.json'
    if not bc_path.exists() or not scalers_path.exists():
        raise RuntimeError(f'Missing bc.pt or scalers.pkl in {art_dir}')
    checkpoint = torch.load(bc_path, map_location='cpu')
    cfg = checkpoint.get('config', {})
    obs_dim = int(cfg.get('obs_dim', len(OBS_FALLBACK)))
    act_dim = int(cfg.get('act_dim', len(ACT_COLUMNS)))
    hidden_dim = int(cfg.get('hidden_dim', 256))
    depth = int(cfg.get('depth', 3))
    model = BehaviorCloningModel(obs_dim, act_dim, hidden_dim, depth)
    state_dict = checkpoint.get('state_dict', checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    with scalers_path.open('rb') as fh:
        scalers = json.load(fh) if scalers_path.suffix == '.json' else __import__('pickle').load(fh)
    obs_scaler = scalers['obs_scaler']
    act_scaler = scalers['act_scaler']
    obs_columns = OBS_FALLBACK
    if manifest_path.exists():
        try:
            with manifest_path.open('r', encoding='utf-8') as fh:
                manifest = json.load(fh)
            if 'obs_columns' in manifest:
                obs_columns = manifest['obs_columns']
        except Exception:
            pass
    return model, obs_scaler, act_scaler, obs_columns

def build_obs_from_sim(model_obs_cols: List[str], site_positions: np.ndarray) -> np.ndarray:
    # site_positions shape (3,3): IF, MF, TH order
    mapping = {
        'ee_if_px': site_positions[0,0], 'ee_if_py': site_positions[0,1], 'ee_if_pz': site_positions[0,2],
        'ee_mf_px': site_positions[1,0], 'ee_mf_py': site_positions[1,1], 'ee_mf_pz': site_positions[1,2],
        'ee_th_px': site_positions[2,0], 'ee_th_py': site_positions[2,1], 'ee_th_pz': site_positions[2,2],
    }
    arr = []
    for col in model_obs_cols:
        if col.startswith('ee_'):
            arr.append(mapping.get(col, 0.0))
        elif col.startswith('s') and ('_f' in col):
            arr.append(0.0)  # no force sensor online
        elif col == 'deform_ecc':
            arr.append(0.0)
        else:
            arr.append(0.0)
    return np.asarray(arr, dtype=float)

def load_replay_observations(csv_path: Path, obs_columns: List[str]) -> np.ndarray:
    df = pd.read_csv(csv_path)
    # Backward compatibility for legacy ee_px/py/pz
    for finger in ['if','mf','th']:
        for axis in ['px','py','pz']:
            new_col = f'ee_{finger}_{axis}'
            legacy = f'ee_{axis}'
            if new_col not in df.columns and legacy in df.columns:
                df[new_col] = df[legacy]
    missing = [c for c in obs_columns if c not in df.columns]
    if missing:
        raise RuntimeError(f'Missing observation columns in replay CSV: {missing}')
    obs = df[obs_columns].to_numpy(dtype=float)
    return obs

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
    # Sample every N steps to avoid too many markers (cap ~50 per finger)
    sample_step = max(1, min([pos.shape[0] for pos in desired_positions.values()]) // 50)

    for finger, positions in desired_positions.items():
        color = finger_colors.get(finger, [0.6, 0.6, 0.6, 0.7])
        for i in range(0, positions.shape[0], sample_step):
            if scene.ngeom >= scene.maxgeom:
                break
            pos = positions[i]
            geom = scene.geoms[scene.ngeom]
            # Set sphere type (newer mujoco python bindings expose enum differently)
            try:
                geom.type = mujoco.mjtGeom.mjGEOM_SPHERE  # type: ignore[attr-defined]
            except AttributeError:
                # Fallback: older binding may require int value 0 for sphere
                geom.type = 0
            # Radius for sphere (only first element used for mjGEOM_SPHERE)
            geom.size[0] = 0.003
            geom.pos[:] = pos
            # Robust identity orientation assignment (binding shape can be (3,3) or (9,))
            if hasattr(geom, 'mat'):
                try:
                    if geom.mat.shape == (3, 3):
                        geom.mat[:, :] = np.eye(3)
                    elif np.prod(geom.mat.shape) == 9:
                        geom.mat[:] = np.eye(3).reshape(-1)
                except Exception:
                    pass
            # RGBA
            geom.rgba[:] = color
            scene.ngeom += 1

def parse_args():
    p = argparse.ArgumentParser(description='MuJoCo Behavior Cloning stiffness control demo')
    p.add_argument('--artifacts-dir', type=Path, default="/home/songwoo/ros2_ws/icra2025/src/hri_falcon_robot_bridge/outputs/models/policy_learning_unified/artifacts/20251122_181241", help='Directory containing bc.pt, scalers.pkl, manifest.json (auto-detect if omitted)')
    p.add_argument('--mode', type=str, default='online', choices=['online','replay'], help='Control mode (online model queries vs replay predicted sequence)')
    p.add_argument('--obs-csv', type=Path, default=None, help='Observation CSV for replay mode')
    p.add_argument('--position-csv', type=Path, default=None, help='Desired position CSV (with ee_if/mf/th_px/py/pz columns)')
    p.add_argument('--mjcf-path', type=Path, default=Path(DEFAULT_MJCF_PATH), help='MJCF model path')
    p.add_argument('--repeat', action='store_true', help='Loop sequence (replay mode)')
    p.add_argument('--stiffness-scale', type=float, default=1.0, help='Scale factor applied to predicted stiffness')
    p.add_argument('--finger-target-offset', type=float, default=0.0, help='Small offset added to desired fingertip goal positions (online mode)')
    p.add_argument('--max-steps', type=int, default=5000, help='Safety cap on simulation steps')
    return p.parse_args()

def main():
    args = parse_args()
    art_dir = args.artifacts_dir or autodetect_artifacts()
    if art_dir is None:
        raise RuntimeError('Could not auto-detect artifacts directory with bc.pt')
    print(f'[info] Using artifact dir: {art_dir}')
    model, obs_scaler, act_scaler, obs_columns = load_bc_artifacts(art_dir)
    print(f'[info] Loaded BC model: obs_dim={len(obs_columns)} act_dim={len(ACT_COLUMNS)}')

    if args.mode == 'replay' and args.obs_csv is None:
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

    if args.mode == 'replay':
        raw_obs = load_replay_observations(args.obs_csv, obs_columns)
        obs_scaled = obs_scaler.transform(raw_obs)
        with torch.no_grad():
            pred_scaled = model(torch.from_numpy(obs_scaled.astype(np.float32)))
        pred = act_scaler.inverse_transform(pred_scaled.numpy())
        stiffness_seq = pred * args.stiffness_scale
        print(f'[info] Generated stiffness sequence from replay CSV: {stiffness_seq.shape}')
    else:
        stiffness_seq = None  # generated online

    if not args.mjcf_path.exists():
        raise RuntimeError(f'MJCF not found: {args.mjcf_path}')
    mj_model = mujoco.MjModel.from_xml_path(str(args.mjcf_path))  # type: ignore[attr-defined]
    mj_data = mujoco.MjData(mj_model)  # type: ignore[attr-defined]
    mujoco.mj_forward(mj_model, mj_data)  # type: ignore[attr-defined]

    # Resolve fingertip site ids
    site_ids = []
    for nm in SITE_NAMES:
        try:
            sid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, nm)  # type: ignore
        except Exception:
            sid = -1
            print(f'[warn] Missing site: {nm}')
        site_ids.append(sid)

    # Color coding for trajectory visualization
    finger_colors = {
        'if': [1.0, 0.0, 0.0, 0.4],  # red for index finger
        'mf': [0.0, 1.0, 0.0, 0.4],  # green for middle finger
        'th': [0.0, 0.0, 1.0, 0.4],  # blue for thumb
    }

    print('[info] Starting BC control simulation... (ESC to exit)')
    step = 0
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:  # type: ignore
        while viewer.is_running() and step < args.max_steps:
            start = time.time()
            
            # Add trajectory visualization on first frame
            if step == 0 and desired_positions:
                scene = viewer.user_scn
                add_trajectory_visualization(scene, desired_positions, finger_colors)
            
            # Build or fetch stiffness vector (9D)
            if stiffness_seq is not None:
                if step >= stiffness_seq.shape[0]:
                    if args.repeat:
                        step = 0
                    else:
                        print('[info] Replay sequence finished.')
                        break
                act9 = stiffness_seq[step]
            else:
                # Online mode: build observation from current positions
                positions = []
                for sid in site_ids:
                    if sid >= 0:
                        positions.append(mj_data.site_xpos[sid].copy())
                    else:
                        positions.append(np.zeros(3))
                site_pos = np.vstack(positions)  # (3,3)
                obs_vec = build_obs_from_sim(obs_columns, site_pos)
                obs_scaled = obs_scaler.transform(obs_vec.reshape(1, -1))
                with torch.no_grad():
                    pred_scaled = model(torch.from_numpy(obs_scaled.astype(np.float32)))
                pred_raw = act_scaler.inverse_transform(pred_scaled.numpy())
                act9 = pred_raw[0] * args.stiffness_scale
            # Split per finger
            th_K = act9[0:3]
            if_K = act9[3:6]
            mf_K = act9[6:9]
            # Apply simple Cartesian PD for each site separately
            total_tau = np.zeros(mj_model.nv)
            for idx, (prefix, K, sid) in enumerate(zip(FINGER_PREFIXES, [if_K, mf_K, th_K], site_ids)):
                if sid < 0:
                    continue
                current_pos = mj_data.site_xpos[sid].copy()
                # Use desired position from CSV if available, else current + offset
                if prefix in desired_positions and step < desired_positions[prefix].shape[0]:
                    goal = desired_positions[prefix][step]
                else:
                    goal = current_pos + np.array([0.0, 0.0, args.finger_target_offset])
                jacp = np.zeros((3, mj_model.nv))
                jacr = np.zeros((3, mj_model.nv))
                mujoco.mj_jacSite(mj_model, mj_data, jacp, jacr, sid)  # type: ignore
                vel = jacp @ mj_data.qvel
                pos_err = goal - current_pos
                Kp = np.diag(np.clip(K, 1.0, None))
                Kd = np.diag(2.0 * np.sqrt(np.clip(K, 1.0, None)))
                F = Kp @ pos_err - Kd @ vel
                tau = jacp.T @ F
                total_tau[:tau.shape[0]] += tau
            mj_data.qfrc_applied[:] = total_tau
            mujoco.mj_step(mj_model, mj_data)  # type: ignore[attr-defined]
            step += 1
            if step % 200 == 0:
                print(f'[step {step}] th_K={th_K} if_K={if_K} mf_K={mf_K}')
            viewer.sync()
            remain = mj_model.opt.timestep - (time.time() - start)
            if remain > 0:
                time.sleep(remain)
    print('[info] Simulation finished.')

if __name__ == '__main__':
    main()
