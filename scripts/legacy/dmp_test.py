#!/usr/bin/env python3
"""
Track 1: DMP Motion Learning (Auto Goal Extension Version)
----------------------------
1. CSV 데이터 로드 (x_demo, F_demo, K_demo)
2. 가상 목표 궤적(x_attr) 역산: x_attr = x_demo + F / K
3. DMP 학습
4. [NEW] 목표점 자동 연장: 진행 방향 벡터를 계산하여 목표 지점을 더 깊게 설정
5. 결과 저장 및 시각화
"""

import argparse
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import interp1d
from glob import glob
import os

# 멀티 손가락 처리 대상 (엄지 th, 검지 if, 중지 mf)
FINGERS = ["th", "if", "mf"]

# ======================================================
# 0. Trajectory Alignment Functions
# ======================================================
def get_mean_trajectory_simple(demo_list, target_len=200):
    """모든 데모를 같은 길이로 리샘플링 후 평균 계산"""
    if len(demo_list) == 0:
        raise ValueError("demo_list is empty")
    
    interpolated_trajs = []
    
    for traj in demo_list:
        T = len(traj)
        if T < 2:
            continue
            
        x_old = np.linspace(0, 1, T)
        x_new = np.linspace(0, 1, target_len)
        f = interp1d(x_old, traj, axis=0, kind='linear')
        traj_new = f(x_new)
        interpolated_trajs.append(traj_new)
    
    if len(interpolated_trajs) == 0:
        raise ValueError("No valid trajectories after filtering")
        
    mean_traj = np.mean(np.stack(interpolated_trajs), axis=0)
    print(f"✅ Aligned {len(interpolated_trajs)} demos → target_len={target_len}")
    return mean_traj

def visualize_alignment_quality(demo_list, mean_traj=None, dmp_output=None):
    """정렬 상태 시각화"""
    plt.figure(figsize=(12, 4))
    dims = ['X', 'Y', 'Z']
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(f"Axis {dims[i]} (Normalized)")
        
        all_vals = np.concatenate([traj[:, i] for traj in demo_list])
        if mean_traj is not None: all_vals = np.concatenate([all_vals, mean_traj[:, i]])
        if dmp_output is not None: all_vals = np.concatenate([all_vals, dmp_output[:, i]])
        
        v_min, v_max = all_vals.min(), all_vals.max()
        v_range = v_max - v_min if v_max > v_min else 1.0
        
        for traj in demo_list:
            progress = np.linspace(0, 1, len(traj))
            normalized = (traj[:, i] - v_min) / v_range
            plt.plot(progress, normalized, 'b-', alpha=0.15, linewidth=1)
        
        if mean_traj is not None:
            progress = np.linspace(0, 1, len(mean_traj))
            norm_mean = (mean_traj[:, i] - v_min) / v_range
            plt.plot(progress, norm_mean, 'orange', linewidth=2, linestyle='--', label='Mean')
            
        if dmp_output is not None:
            progress = np.linspace(0, 1, len(dmp_output))
            norm_dmp = (dmp_output[:, i] - v_min) / v_range
            plt.plot(progress, norm_dmp, 'r-', linewidth=2, label='DMP')
            
        plt.ylim(-0.05, 1.05)
        plt.grid(True, alpha=0.3)
        if i == 0: plt.legend()
    plt.tight_layout()
    plt.show()

# ======================================================
# 1. DMP Class
# ======================================================
class DiscreteDMP:
    def __init__(self, n_bfs=50, alpha_y=25.0, beta_y=6.25):
        self.n_bfs = n_bfs
        self.alpha_y = alpha_y
        self.beta_y = beta_y
        self.w = None
        self.a_x = 1.0 

    def _gaussian_basis(self, x):
        centers = np.exp(-self.a_x * np.linspace(0, 1, self.n_bfs))
        widths = (np.diff(centers)[0] if self.n_bfs > 1 else 1.0) ** 2
        h = np.exp(-((x[:, None] - centers[None, :]) ** 2) / (2 * widths))
        return h

    def train(self, trajectory, dt=0.02):
        n_steps, n_dims = trajectory.shape
        self.y0 = trajectory[0]
        self.goal = trajectory[-1]
        self.dt = dt
        self.tau = n_steps * dt

        dy = np.gradient(trajectory, axis=0) / dt
        ddy = np.gradient(dy, axis=0) / dt
        x = np.exp(-self.a_x * np.linspace(0, 1, n_steps))
        
        f_target = ddy - self.alpha_y * (self.beta_y * (self.goal - trajectory) - dy)
        
        psi = self._gaussian_basis(x)
        self.w = np.zeros((self.n_bfs, n_dims))
        for d in range(n_dims):
            X = psi * x[:, None]
            Y = f_target[:, d]
            self.w[:, d] = np.linalg.inv(X.T @ X + 1e-5 * np.eye(self.n_bfs)) @ (X.T @ Y)
        print(f"✅ DMP Training Done. Weights shape: {self.w.shape}")

    def save(self, path):
        data = {
            "w": self.w, "y0": self.y0, "goal": self.goal,
            "dt": self.dt, "tau": self.tau, "n_bfs": self.n_bfs,
            "alpha_y": self.alpha_y, "beta_y": self.beta_y,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"💾 Model saved to: {path}")

    def rollout(self, dt=None, tau=None, goal_offset=np.array([0.0, 0.0, 0.0])):
        if dt is None: dt = self.dt
        if tau is None: tau = self.tau
        n_steps = int(tau / dt)
        y = self.y0.copy()
        dy = np.zeros_like(y)
        path = []
        x = 1.0
        
        # 목표 지점에 오프셋 적용 (여기서 연장된 목표가 적용됨)
        adjusted_goal = self.goal + goal_offset
        
        for _ in range(n_steps):
            path.append(y.copy())
            x_next = x - self.a_x * x * (dt / tau)
            psi = self._gaussian_basis(np.array([x]))[0]
            f = np.dot(psi * x, self.w) / (np.sum(psi) + 1e-10)
            ddy = self.alpha_y * (self.beta_y * (adjusted_goal - y) - dy) + f
            dy += ddy * (dt / tau)
            y += dy * (dt / tau)
            x = x_next
        return np.array(path)

# ======================================================
# 2. 데이터 처리
# ======================================================
def process_data(csv_path, finger, force_scale=1.0):
    df = pd.read_csv(csv_path)
    sensor_map = {'th': 's1', 'if': 's2', 'mf': 's3'}
    s_idx = sensor_map[finger]

    pos_cols = [f'ee_{finger}_px', f'ee_{finger}_py', f'ee_{finger}_pz']
    if not all(c in df.columns for c in pos_cols):
        pos_cols = ['ee_px', 'ee_py', 'ee_pz']
    x_demo = df[pos_cols].values

    force_cols = [f'{s_idx}_fx', f'{s_idx}_fy', f'{s_idx}_fz']
    F_demo = df[force_cols].values * force_scale

    stiff_cols = [f'{finger}_k1', f'{finger}_k2', f'{finger}_k3']
    K_demo = df[stiff_cols].values

    valid_mask = np.isfinite(x_demo).all(axis=1) & \
                 np.isfinite(F_demo).all(axis=1) & \
                 np.isfinite(K_demo).all(axis=1)
    
    x_demo = x_demo[valid_mask]
    F_demo = F_demo[valid_mask]
    K_demo = K_demo[valid_mask]

    K_safe = np.maximum(K_demo, 1.0)
    x_attr_raw = x_demo + (F_demo / K_safe)

    return x_demo, x_attr_raw

# ======================================================
# 3. 메인 함수 (수정됨: 자동 목표 연장)
# ======================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, help='Path to single demo CSV')
    parser.add_argument('--csv_pattern', type=str, default="/home/songwoo/ros2_ws/icra2025/src/hri_falcon_robot_bridge/outputs/stiffness_profiles_signaligned/*.csv")
    parser.add_argument('--n_bfs', type=int, default=50)
    parser.add_argument('--out_dir', type=str, default='/home/songwoo/ros2_ws/icra2025/src/hri_falcon_robot_bridge/scripts/legacy/dmp_models')
    parser.add_argument('--force_scale', type=float, default=1.0)
    parser.add_argument('--target_len', type=int, default=1500)
    parser.add_argument('--visualize_alignment', action='store_true', default=True)
    
    # [설정] 진행 방향 연장 거리 (기본값 0.03m = 3cm)
    parser.add_argument('--goal_extension', type=float, default=0.03, help='Extend goal along trajectory direction (m)')
    
    # 수동 오프셋 (필요시 사용, 기본 0)
    parser.add_argument('--goal_offset_x', type=float, default=0.0)
    parser.add_argument('--goal_offset_y', type=float, default=0.0)
    parser.add_argument('--goal_offset_z', type=float, default=0.0)
    
    parser.add_argument('--save_plots', action='store_true', default=True)
    parser.add_argument('--plot_dir', type=str, default='/home/songwoo/ros2_ws/icra2025/src/hri_falcon_robot_bridge/scripts/legacy/plots')
    args = parser.parse_args()

    manual_offset = np.array([args.goal_offset_x, args.goal_offset_y, args.goal_offset_z])
    
    print(f"🔧 Config: Goal Extension = {args.goal_extension*100:.1f} cm (along trajectory)")

    # -------------------------------------------------------------------
    # Multi-CSV Mode
    # -------------------------------------------------------------------
    if args.csv_pattern:
        print(f"[Multi-Demo Mode] Pattern: {args.csv_pattern}")
        csv_files = sorted([f for f in glob(args.csv_pattern) if 'aug' not in Path(f).name])
        if len(csv_files) == 0:
            print("❌ No CSV files found.")
            return

        out_path = Path(args.out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        finger_mean = {}
        finger_reproduced = {}
        finger_raw_lists = {}

        for finger in FINGERS:
            print(f"\n=== Finger {finger} ===")
            x_attr_list = []
            for csv_path in csv_files:
                try:
                    _, x_attr = process_data(csv_path, finger, args.force_scale)
                    x_attr_list.append(x_attr)
                except Exception:
                    continue
            
            if not x_attr_list:
                print(f"  Skipping {finger} (No data)")
                continue

            # 1. 평균 궤적 생성
            x_attr_mean = get_mean_trajectory_simple(x_attr_list, target_len=args.target_len)
            
            # 2. [핵심] 진행 방향 벡터 계산 및 목표 연장
            start_pt = x_attr_mean[0]
            end_pt = x_attr_mean[-1]
            direction_vec = end_pt - start_pt
            norm = np.linalg.norm(direction_vec)
            
            if norm > 1e-6:
                unit_vec = direction_vec / norm
            else:
                unit_vec = np.zeros(3)
                
            # 최종 오프셋 = (방향 벡터 * 연장 거리) + 수동 오프셋
            auto_offset = unit_vec * args.goal_extension
            final_offset = auto_offset + manual_offset
            
            print(f"  🚀 Directional Extension: {auto_offset * 100} cm")

            # 3. DMP 학습
            dmp = DiscreteDMP(n_bfs=args.n_bfs)
            dmp.train(x_attr_mean, dt=0.02)
            
            # 4. 연장된 목표로 궤적 생성
            x_reproduced = dmp.rollout(goal_offset=final_offset)

            # 저장
            save_name = out_path / f"dmp_{finger}_multi_{len(csv_files)}demos.pkl"
            dmp.save(save_name)
            
            finger_mean[finger] = x_attr_mean
            finger_reproduced[finger] = x_reproduced
            finger_raw_lists[finger] = x_attr_list

        # 시각화 (3D Plot)
        if len(finger_mean) > 0 and args.save_plots:
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure(figsize=(15, 6))
            
            for idx, finger in enumerate(finger_mean.keys()):
                mean_traj = finger_mean[finger]
                reproduced = finger_reproduced[finger]
                
                ax = fig.add_subplot(1, 3, idx+1, projection='3d')
                ax.set_title(f"{finger.upper()} (Extension {args.goal_extension}m)")
                
                # Raw Demos
                for raw in finger_raw_lists[finger]:
                    ax.plot(raw[:,0], raw[:,1], raw[:,2], color='gray', alpha=0.1)
                
                # Mean Traj (Original Goal)
                ax.plot(mean_traj[:,0], mean_traj[:,1], mean_traj[:,2], 'k--', label='Original Mean')
                
                # DMP Result (Extended Goal)
                ax.plot(reproduced[:,0], reproduced[:,1], reproduced[:,2], 'r-', linewidth=2, label='Extended DMP')
                
                # 마커
                ax.scatter(*reproduced[0], color='green', s=50, label='Start')
                ax.scatter(*mean_traj[-1], color='black', marker='x', s=50, label='Orig Goal')
                ax.scatter(*reproduced[-1], color='blue', marker='s', s=80, label='New Goal')
                
                if idx == 0: ax.legend()

            os.makedirs(args.plot_dir, exist_ok=True)
            plot_path = Path(args.plot_dir) / f"dmp_extended_{args.goal_extension}m.png"
            plt.savefig(plot_path)
            print(f"📊 Plot saved: {plot_path}")
            
        return

    # -------------------------------------------------------------------
    # Single CSV Mode
    # -------------------------------------------------------------------
    if args.csv:
        print(f"[Single Demo Mode] {args.csv}")
        out_path = Path(args.out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        
        for finger in FINGERS:
            try:
                _, x_attr_raw = process_data(args.csv, finger, args.force_scale)
            except Exception:
                continue
            
            if len(x_attr_raw) < 2: continue

            # 목표 연장 계산
            start_pt = x_attr_raw[0]
            end_pt = x_attr_raw[-1]
            direction = end_pt - start_pt
            norm = np.linalg.norm(direction)
            unit_vec = (direction / norm) if norm > 1e-6 else np.zeros(3)
            
            final_offset = (unit_vec * args.goal_extension) + manual_offset
            
            dmp = DiscreteDMP(n_bfs=args.n_bfs)
            dmp.train(x_attr_raw, dt=0.02)
            dmp.save(out_path / f"dmp_{finger}_{Path(args.csv).stem}.pkl")
            
            # 시각화 생략 (Multi모드와 로직 동일)
            print(f"  ✅ {finger} trained. Ext: {unit_vec * args.goal_extension}")

if __name__ == "__main__":
    main()