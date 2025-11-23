#!/usr/bin/env python3
"""MuJoCo impedance control demonstration replaying stiffness profiles.

Simplified per user request:
  - No observation/model inference; stiffness K(t) is taken directly from a CSV.
  - CSV is assumed to contain columns like th_k1, th_k2, th_k3 (or if_k*, mf_k*).
  - Selected finger prefix provides the 3 DOF stiffness values for the demo arm.

If no CSV provided, a default sinusoidal stiffness pattern is used.

Requirements:
    pip install mujoco mujoco-python-viewer pandas

Usage examples:
    python3 mujoco_impedance_demo.py --stiffness-csv outputs/analysis/stiffness_profiles_global_tk/paper_profile.csv \
        --finger-prefix th --repeat --stiffness-scale 1.0

    python mujoco_impedance_demo.py  # (uses synthetic stiffness)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional
import os

import numpy as np

import pandas as pd  # type: ignore
HAS_PANDAS = True

# (Plotting removed per user request)


import mujoco
import mujoco.viewer
HAS_MUJOCO = True



# Default external MJCF (user provided path)
DEFAULT_MUJOCO_MODEL_PATH = '/home/songwoo/git/ur_dclaw/dclaw_finger_description/urdf/dclaw_finger_mjcf_finall_inertia_edit.xml'

# Resolve package root for consistent output paths
_THIS_FILE = Path(__file__).resolve()
_PKG_ROOT = _THIS_FILE.parents[2]  # .../src/hri_falcon_robot_bridge
_OUTPUT_ROOT = _PKG_ROOT / 'outputs'
_PLOT_DIR = _OUTPUT_ROOT / 'plots'  # unused
DEFAULT_POSITION_LOG_DIR = _PKG_ROOT / 'outputs' / 'logs' / 'success'



class CartesianImpedanceController:
    """Cartesian space impedance controller with variable stiffness."""

    def __init__(self, mj_model, mj_data, finger_prefix: str = "th"):
        self.model = mj_model
        self.data = mj_data
        self.nq = mj_model.nq
        self.nv = mj_model.nv
        self.nu = mj_model.nu
        self.finger_prefix = finger_prefix
        self.K_cart = np.array([50.0, 50.0, 50.0], dtype=float)
        self.x_desired = np.zeros(3, dtype=float)
        self.xd_desired = np.zeros(3, dtype=float)
        # Find EE site based on finger prefix
        ee_site_name = f"ee_{finger_prefix}"
        try:
            self.ee_site_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, ee_site_name)  # type: ignore
            print(f"[info] Using EE site: {ee_site_name}")
        except:
            self.ee_site_id = -1
            print(f"[warn] Site '{ee_site_name}' not found; using body 0 position as fallback")
def load_stiffness_csv(csv_path: Path, finger_prefix: str) -> np.ndarray:
    """Load stiffness sequence (k1,k2,k3) for given finger prefix from CSV."""
    if not csv_path.exists():
        raise RuntimeError(f"Stiffness CSV not found: {csv_path}")
    print(f"[info] Loading stiffness from: {csv_path}")
    df = pd.read_csv(csv_path)  # type: ignore
    cols = [f"{finger_prefix}_k1", f"{finger_prefix}_k2", f"{finger_prefix}_k3"]
    if all(c in df.columns for c in cols):
        return df[cols].to_numpy(dtype=float)  # type: ignore
    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)  # type: ignore
    if len(numeric_cols) < 3:
        raise RuntimeError("Not enough numeric columns to derive stiffness")
    return df[numeric_cols[:3]].to_numpy(dtype=float)  # type: ignore


def load_demo_positions(csv_path: Path, finger_prefix: str) -> Optional[np.ndarray]:
    """Load end-effector Cartesian positions for a finger prefix (ee_<prefix>_px/py/pz)."""
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)  # type: ignore
    cols = [f"ee_{finger_prefix}_px", f"ee_{finger_prefix}_py", f"ee_{finger_prefix}_pz"]
    if not all(c in df.columns for c in cols):
        return None
    return df[cols].to_numpy(dtype=float)  # type: ignore


def generate_synthetic_stiffness(num_steps: int) -> np.ndarray:
    """Generate a simple synthetic stiffness pattern if no CSV provided."""
    t = np.linspace(0, 2 * np.pi, num_steps)
    k1 = 40 + 20 * (np.sin(t) * 0.5 + 0.5)
    k2 = 50 + 25 * (np.sin(t + np.pi/3) * 0.5 + 0.5)
    k3 = 60 + 30 * (np.sin(t + 2*np.pi/3) * 0.5 + 0.5)
    return np.stack([k1, k2, k3], axis=1)
def _load_mujoco_model(args) -> tuple:
    """Load external MJCF; raise if missing (fallback removed by user request)."""
    mjcf_path = getattr(args, 'mjcf_path', None) or DEFAULT_MUJOCO_MODEL_PATH
    if not (mjcf_path and os.path.isfile(mjcf_path)):
        raise FileNotFoundError(f"External MJCF not found: {mjcf_path}. Provide --mjcf-path pointing to a valid file.")
    print(f"[info] Loading external MJCF: {mjcf_path}")
    model = mujoco.MjModel.from_xml_path(mjcf_path)  # type: ignore[attr-defined]
    data = mujoco.MjData(model)  # type: ignore[attr-defined]
    return model, data

def run_simulation(args: argparse.Namespace) -> None:
    print("[info] Creating MuJoCo environment...")
    print(f"[info] Configuration:")
    print(f"  - Finger prefix: {args.finger_prefix}")
    print(f"  - Stiffness CSV: {args.stiffness_csv if args.stiffness_csv else 'None (using default K=50)'}")
    # Auto-detect latest position CSV if none provided
    position_csv = args.position_csv
    if position_csv is None:
        search_dir = DEFAULT_POSITION_LOG_DIR
        latest = None
        if search_dir.exists():
            candidates = [p for p in search_dir.rglob('*.csv') if p.is_file()]
            if candidates:
                latest = max(candidates, key=lambda p: p.stat().st_mtime)
        if latest is not None:
            position_csv = latest
            print(f"[info] Auto-selected latest position CSV: {position_csv}")
        else:
            print(f"[warn] No position CSV provided and none found under {search_dir}")
    else:
        print(f"[info] Using provided position CSV: {position_csv}")

    # (Analysis/plotting removed)
    
    mj_model, mj_data = _load_mujoco_model(args)
    controller = CartesianImpedanceController(mj_model, mj_data, finger_prefix=args.finger_prefix)

    # Load stiffness sequence
    if args.stiffness_csv:
        if not args.stiffness_csv.exists():
            raise RuntimeError(f"Stiffness CSV not found: {args.stiffness_csv}")
        stiffness_seq = load_stiffness_csv(args.stiffness_csv, args.finger_prefix)
        print(f"[info] Loaded stiffness sequence: {stiffness_seq.shape} (rows,3) from {args.stiffness_csv}")
    else:
        # Use constant default stiffness
        stiffness_seq = np.tile(np.array([50.0, 50.0, 50.0]), (args.synthetic_steps, 1))
        print(f"[info] Using default constant stiffness: K=[50, 50, 50] for {args.synthetic_steps} steps")

    # Load position sequences for all three fingers
    demo_pos_dict = {}  # Dict mapping prefix to position array
    if position_csv:
        if not position_csv.exists():
            raise RuntimeError(f"Position CSV not found: {position_csv}")
        # Load positions for each finger
        for prefix in ["if", "mf", "th"]:
            pos = load_demo_positions(position_csv, prefix)
            if pos is not None:
                demo_pos_dict[prefix] = pos
                print(f"[info] Loaded demo positions for ee_{prefix}: {pos.shape}")
            else:
                print(f"[warn] Could not extract position data for ee_{prefix} from {position_csv}")
    
    # Plot EE trajectories if position data loaded
    if demo_pos_dict:
        print("[info] Position data loaded:")
        for prefix, arr in demo_pos_dict.items():
            print(f"  ee_{prefix}: {arr.shape[0]} samples")

    # Apply scaling
    stiffness_seq = stiffness_seq * args.stiffness_scale

    # Initialize qpos (zeros)
    mj_data.qpos[:] = 0.0
    mujoco.mj_forward(mj_model, mj_data)  # type: ignore[attr-defined]

    # Compute actual demo duration and adjust timestep
    if demo_pos_dict:
        # Use the first available position array to determine num_steps
        first_pos = next(iter(demo_pos_dict.values()))
        num_steps = min(stiffness_seq.shape[0], first_pos.shape[0])
        # Read time_s column if available to get real duration
        demo_duration = 15.0  # default assumption
        if args.stiffness_csv and args.stiffness_csv.exists():
            try:
                df_time = pd.read_csv(args.stiffness_csv, usecols=['time_s'])  # type: ignore
                if len(df_time) > 0:
                    demo_duration = float(df_time['time_s'].iloc[-1])
                    print(f"[info] Demo duration from CSV: {demo_duration:.2f}s")
            except:
                pass
        # Override model timestep to match demo
        target_dt = demo_duration / max(num_steps, 1)
        mj_model.opt.timestep = target_dt
        print(f"[info] Adjusted timestep to {target_dt*1000:.2f}ms to match demo duration")
    else:
        num_steps = stiffness_seq.shape[0]

    print("[info] Starting visualization (ESC to exit)...")
    
    # Initialize tracking variables for three fingers
    # Mapping: FFtip -> ee_if, MFtip -> ee_mf, THtip -> ee_th
    tips = ["FFtip", "MFtip", "THtip"]
    tip_prefixes = ["if", "mf", "th"]  # CSV column prefixes for each tip
    end_effector_ids = []
    for tip_name in tips:
        try:
            site_id = mj_model.site(tip_name).id
            end_effector_ids.append(site_id)
        except:
            print(f"[warn] Site '{tip_name}' not found")
            end_effector_ids.append(-1)
    
    # Initialize Jacobian matrices
    jacp = np.zeros([3, 3, mj_model.nv])
    
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:  # type: ignore
        step = 0
        # Add target visualization sphere
        target_geom_id = -1
        try:
            target_geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "target")  # type: ignore
        except:
            pass

        while viewer.is_running():
            start = time.time()
            
            # Check if we've reached the end of the sequence
            if step >= num_steps:
                if args.repeat:
                    step = 0  # Reset to beginning
                else:
                    print(f"[info] Reached end of sequence ({num_steps} steps). Exiting...")
                    break
            
            idx = step
            
            # Get stiffness from sequence
            K = stiffness_seq[idx]

            # Impedance control loop for all fingers
            tau_imp = np.zeros(mj_model.nv)
            
            for finger_idx, tip in enumerate(tips):
                if end_effector_ids[finger_idx] < 0:
                    continue
                
                # Get goal position for this specific finger
                prefix = tip_prefixes[finger_idx]
                if prefix in demo_pos_dict:
                    goal = demo_pos_dict[prefix][idx]
                else:
                    # Fallback to circular trajectory if no CSV data
                    t = step * mj_model.opt.timestep
                    goal = np.array([
                        0.2 + 0.1 * np.sin(0.5 * t),
                        0.2 + 0.1 * np.cos(0.5 * t),
                        0.4 + 0.05 * np.sin(t)
                    ])
                
                # Position error (goal - current)
                current_pos = mj_data.site(tip).xpos.copy()
                x_error = goal - current_pos
                
                # Compute Jacobian
                jacp_temp = np.zeros((3, mj_model.nv))
                jacr_temp = np.zeros((3, mj_model.nv))
                mujoco.mj_jacSite(mj_model, mj_data, jacp_temp, jacr_temp, end_effector_ids[finger_idx])  # type: ignore
                jacp[finger_idx, :] = jacp_temp
                
                # Velocity
                xvel = jacp[finger_idx, :] @ mj_data.qvel
                
                # Simple Cartesian PD control with increased gains
                # F = Kp * position_error - Kd * velocity
                Kp = np.diag(K )  # Increase proportional gain significantly
                Kd = np.diag(2.0 * np.sqrt(K))  # Critical damping
                
                F_cartesian = Kp @ x_error - Kd @ xvel
                
                # Map to joint torques via Jacobian transpose
                tau_finger = jacp[finger_idx, :].T @ F_cartesian
                tau_imp += tau_finger
                
                # Debug: print torque magnitude for first finger on step 200
                if step == 200 and finger_idx == 0:
                    print(f"DEBUG [step {step}] {tip}: x_error={x_error}, ||F||={np.linalg.norm(F_cartesian):.4f}, ||tau||={np.linalg.norm(tau_finger):.4f}")
            
            # Apply combined torque
            mj_data.qfrc_applied[:] = tau_imp
            mujoco.mj_step(mj_model, mj_data)  # type: ignore[attr-defined]
            step += 1

            if step % 200 == 0:
                # Print actual fingertip positions
                tip_positions = []
                tip_goals = []
                for finger_idx, tip in enumerate(tips):
                    if end_effector_ids[finger_idx] >= 0:
                        tip_positions.append(mj_data.site(tip).xpos.copy())
                        prefix = tip_prefixes[finger_idx]
                        if prefix in demo_pos_dict:
                            tip_goals.append(demo_pos_dict[prefix][idx])
                        else:
                            tip_goals.append(None)
                
                if demo_pos_dict:
                    # Calculate errors for each finger to its own target
                    errors = []
                    for i, (pos, goal) in enumerate(zip(tip_positions, tip_goals)):
                        if goal is not None:
                            errors.append(np.linalg.norm(pos - goal))
                    avg_err = np.mean(errors) if errors else 0.0
                    print(f"[step {step}] K={K}, Fingertips={len(tip_positions)}, avg_err={avg_err:.4f}")
                    for i, (pos, goal) in enumerate(zip(tip_positions, tip_goals)):
                        if goal is not None:
                            err = np.linalg.norm(pos - goal)
                            print(f"  {tips[i]}: pos={pos}, goal={goal}, err={err:.4f}")
                else:
                    print(f"[step {step}] K={K}, Fingertips:")
                    for i, pos in enumerate(tip_positions):
                        print(f"  {tips[i]}: {pos}")

            viewer.sync()
            remain = mj_model.opt.timestep - (time.time() - start)
            if remain > 0:
                time.sleep(remain)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MuJoCo impedance control demo (stiffness replay)")
    p.add_argument("--stiffness-csv", type=Path, default=None, help="CSV with stiffness profiles (<prefix>_k1, <prefix>_k2, <prefix>_k3 columns).")
    p.add_argument("--position-csv", type=Path, default=None, help="CSV with EE position trajectory (ee_<prefix>_px/py/pz columns).")
    p.add_argument("--finger-prefix", type=str, default="th", help="Finger prefix to select (th, if, mf).")
    p.add_argument("--repeat", default=True, action="store_true", help="Loop stiffness sequence when finished.")
    p.add_argument("--synthetic-steps", type=int, default=2000, help="Length of synthetic stiffness if CSV not provided.")
    p.add_argument("--stiffness-scale", type=float, default=1.0, help="Scale factor applied to stiffness values.")
    p.add_argument("--mjcf-path", type=str, default=None, help="Override path for external MJCF model (defaults to user-provided constant).")
    # (All plotting-related CLI options removed)
    return p.parse_args()


if __name__ == "__main__":
    run_simulation(parse_args())
