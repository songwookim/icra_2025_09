#!/usr/bin/env python3
"""
Track 1: DMP Motion Learning (Auto Goal Extension Version)
----------------------------
1. CSV 데이터 로드 (x_demo, F_demo, K_demo)
2. 가상 목표 궤적(x_attr) 역산: x_attr = x_demo + F / K
3. DMP 학습
4. [NEW] 목표점 자동 연장: 진행 방향 벡터를 계산하여 목표 지점을 더 깊게 설정
5. 개별 CSV별 플롯 생성 (compare_dmp_kf.py 스타일)
6. 결과 저장 및 시각화
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
import re

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
        if mean_traj is not None:
            all_vals = np.concatenate([all_vals, mean_traj[:, i]])
        if dmp_output is not None:
            all_vals = np.concatenate([all_vals, dmp_output[:, i]])
        
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
        if i == 0:
            plt.legend()
    
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

    def train(self, trajectory, dt=0.02, goal_offset=np.array([0.0, 0.0, 0.0])):
        """
        DMP 학습
        
        Args:
            trajectory: (N, 3) 학습 궤적
            dt: 시간 간격
            goal_offset: 목표점 연장 벡터 (학습 시 적용)
        """
        n_steps, n_dims = trajectory.shape
        self.y0 = trajectory[0]
        # 연장된 목표점을 DMP의 goal로 설정
        self.goal = trajectory[-1] + goal_offset
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
        
        print(f"✅ DMP Training Done. Weights shape: {self.w.shape}, Extended Goal: {self.goal}")

    def save(self, path):
        data = {
            "w": self.w,
            "y0": self.y0,
            "goal": self.goal,
            "dt": self.dt,
            "tau": self.tau,
            "n_bfs": self.n_bfs,
            "alpha_y": self.alpha_y,
            "beta_y": self.beta_y,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"💾 Model saved to: {path}")

    def rollout(self, dt=None, tau=None, tau_scale=1.0, hold_time=0.0, 
                speed_profile='constant', accel_factor=2.0):
        """
        DMP 궤적 생성 (학습 시 설정된 연장된 goal 사용)
        
        Args:
            dt: 시간 간격
            tau: 전체 시간
            tau_scale: 속도 조절 (0.5=2배 빠름, 2.0=2배 느림)
            hold_time: 끝 위치 유지 시간 (초)
            speed_profile: 속도 프로파일 ('constant', 'accelerating', 'decelerating', 'sigmoid')
                - 'constant': 일정 속도
                - 'accelerating': 초반 느림 → 후반 빠름
                - 'decelerating': 초반 빠름 → 후반 느림
                - 'sigmoid': S자 곡선 (smooth acceleration)
            accel_factor: 가속/감속 강도 (1.0=선형, 2.0=제곱, 3.0=세제곱)
        """
        if dt is None:
            dt = self.dt
        if tau is None:
            tau = self.tau
        
        tau = tau * tau_scale  # Apply speed scaling
        n_steps = int(tau / dt)
        y = self.y0.copy()
        dy = np.zeros_like(y)
        path = []
        x = 1.0
        
        # 학습 시 저장된 goal 사용 (이미 연장된 값)
        for step_idx in range(n_steps):
            path.append(y.copy())
            
            # 진행도 계산 (0.0 ~ 1.0)
            progress = step_idx / max(n_steps - 1, 1)
            
            # 속도 프로파일에 따른 동적 tau 조절
            if speed_profile == 'accelerating':
                # 초반 느림(tau 큼) → 후반 빠름(tau 작음)
                # progress^accel_factor: 0→0, 0.5→0.25(accel=2), 1→1
                tau_dynamic = tau * (1.0 + (1.0 - progress**accel_factor) * 2.0)
            elif speed_profile == 'decelerating':
                # 초반 빠름(tau 작음) → 후반 느림(tau 큼)
                tau_dynamic = tau * (1.0 + progress**accel_factor * 2.0)
            elif speed_profile == 'sigmoid':
                # S자 곡선: 중간에 가장 빠름
                sigmoid = 1.0 / (1.0 + np.exp(-10 * (progress - 0.5)))
                tau_dynamic = tau * (2.0 - sigmoid)
            else:  # 'constant'
                tau_dynamic = tau
            
            x_next = x - self.a_x * x * (dt / tau_dynamic)
            psi = self._gaussian_basis(np.array([x]))[0]
            f = np.dot(psi * x, self.w) / (np.sum(psi) + 1e-10)
            ddy = self.alpha_y * (self.beta_y * (self.goal - y) - dy) + f
            dy += ddy * (dt / tau_dynamic)
            y += dy * (dt / tau_dynamic)
            x = x_next
        
        traj = np.array(path)
        
        # Hold at final position if hold_time > 0
        if hold_time > 0.0:
            hold_steps = int(hold_time / dt)
            final_pos = traj[-1].copy()
            hold_traj = np.tile(final_pos, (hold_steps, 1))
            traj = np.vstack([traj, hold_traj])
        
        return traj


# ======================================================
# 2. 데이터 처리
# ======================================================
def process_data(csv_path, finger, force_scale=1.0):
    """CSV에서 x_demo와 x_attr 계산"""
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

    valid_mask = (np.isfinite(x_demo).all(axis=1) & 
                  np.isfinite(F_demo).all(axis=1) & 
                  np.isfinite(K_demo).all(axis=1))
    
    x_demo = x_demo[valid_mask]
    F_demo = F_demo[valid_mask]
    K_demo = K_demo[valid_mask]

    K_safe = np.maximum(K_demo, 1.0)
    x_attr_raw = x_demo + (F_demo / K_safe)

    return x_demo, x_attr_raw


def compute_auto_offset(trajectory, extension_distance):
    """
    궤적의 진행 방향을 계산하여 자동 목표 연장 벡터 생성
    
    Args:
        trajectory: (N, 3) 궤적
        extension_distance: 연장 거리 (m)
    
    Returns:
        auto_offset: (3,) 연장 벡터
    """
    start_pt = trajectory[0]
    end_pt = trajectory[-1]
    direction_vec = end_pt - start_pt
    norm = np.linalg.norm(direction_vec)
    
    if norm > 1e-6:
        unit_vec = direction_vec / norm
    else:
        unit_vec = np.zeros(3)
    
    auto_offset = unit_vec * extension_distance
    return auto_offset


# ======================================================
# 3. 메인 함수
# ======================================================
def main():
    parser = argparse.ArgumentParser(description="DMP Learning with Auto Goal Extension")
    parser.add_argument('--csv', type=str, help='Path to single demo CSV')
    parser.add_argument('--csv_pattern', type=str, 
                        default="/home/songwoo/ros2_ws/icra2025/src/hri_falcon_robot_bridge/outputs/stiffness_profiles_signaligned/*.csv")
    parser.add_argument('--n_bfs', type=int, default=50)
    parser.add_argument('--out_dir', type=str, 
                        default='/home/songwoo/ros2_ws/icra2025/src/hri_falcon_robot_bridge/scripts/legacy/dmp_models')
    parser.add_argument('--force_scale', type=float, default=1.0)
    parser.add_argument('--target_len', type=int, default=1500)
    parser.add_argument('--visualize_alignment', action='store_true', default=False)
    
    # [핵심] 진행 방향 연장 거리 (기본값 0.03m = 3cm)
    parser.add_argument('--goal_extension', type=float, default=0.2, 
                        help='Extend goal along trajectory direction (m) - default for all fingers')
    
    # 손가락별 개별 연장 거리 (옵션)
    parser.add_argument('--goal_extension_th', type=float, default=None,
                        help='Goal extension for thumb (th) - overrides --goal_extension')
    parser.add_argument('--goal_extension_if', type=float, default=None,
                        help='Goal extension for index finger (if) - overrides --goal_extension')
    parser.add_argument('--goal_extension_mf', type=float, default=None,
                        help='Goal extension for middle finger (mf) - overrides --goal_extension')
    
    # 수동 오프셋 (전체 손가락 공통, 기본 0)
    parser.add_argument('--goal_offset_x', type=float, default=0.0)
    parser.add_argument('--goal_offset_y', type=float, default=0.0)
    parser.add_argument('--goal_offset_z', type=float, default=0.0)
    
    # 손가락별 축별 오프셋 (특정 손가락의 특정 축만 조정, 예: mf의 z를 -0.1)
    parser.add_argument('--th_offset_x', type=float, default=0.0, help='Thumb X-axis offset (m)')
    parser.add_argument('--th_offset_y', type=float, default=0.0, help='Thumb Y-axis offset (m)')
    parser.add_argument('--th_offset_z', type=float, default=0.025, help='Thumb Z-axis offset (m)')
    
    parser.add_argument('--if_offset_x', type=float, default=0.05, help='Index finger X-axis offset (m)') # 
    parser.add_argument('--if_offset_y', type=float, default=0.0, help='Index finger Y-axis offset (m)')
    parser.add_argument('--if_offset_z', type=float, default=-0.025, help='Index finger Z-axis offset (m)')
    
    parser.add_argument('--mf_offset_x', type=float, default=0.0, help='Middle finger X-axis offset (m)')
    parser.add_argument('--mf_offset_y', type=float, default=0.0, help='Middle finger Y-axis offset (m)')
    parser.add_argument('--mf_offset_z', type=float, default=-0.025, help='Middle finger Z-axis offset (m, e.g., -0.1 to lower)') # -0.25
    
    # DMP 실행 설정
    parser.add_argument('--tau_scale', type=float, default=0.25,
                        help='DMP speed scaling: 0.5=2x faster, 2.0=2x slower')
    parser.add_argument('--hold_time', type=float, default=7.5,
                        help='Hold time at final position (seconds)')
    
    # [NEW] 속도 프로파일 설정
    parser.add_argument('--speed_profile', type=str, default='decelerating',
                        choices=['constant', 'accelerating', 'decelerating', 'sigmoid'],
                        help='Speed profile during trajectory execution:\n'
                             '  constant: uniform speed\n'
                             '  accelerating: slow start → fast end\n'
                             '  decelerating: fast start → slow end\n'
                             '  sigmoid: smooth S-curve acceleration')
    parser.add_argument('--accel_factor', type=float, default=2.0,
                        help='Acceleration/deceleration strength (1.0=linear, 2.0=quadratic, 3.0=cubic)')
    
    # Plot 설정
    parser.add_argument('--save_plots', action='store_true', default=True,
                        help='Save plots without showing')
    parser.add_argument('--plot_dir', type=str, 
                        default='/home/songwoo/ros2_ws/icra2025/src/hri_falcon_robot_bridge/scripts/legacy/plots')
    parser.add_argument('--per_demo_plots', action='store_true', default=False,
                        help='Generate individual plots per CSV (like compare_dmp_kf.py)')
    
    args = parser.parse_args()

    # 전체 공통 오프셋
    manual_offset = np.array([args.goal_offset_x, args.goal_offset_y, args.goal_offset_z])
    
    # 손가락별 연장 거리 설정
    finger_extensions = {
        'th': args.goal_extension_th if args.goal_extension_th is not None else args.goal_extension,
        'if': args.goal_extension_if if args.goal_extension_if is not None else args.goal_extension,
        'mf': args.goal_extension_mf if args.goal_extension_mf is not None else args.goal_extension,
    }
    
    # 손가락별 축별 오프셋 설정
    finger_offsets = {
        'th': np.array([args.th_offset_x, args.th_offset_y, args.th_offset_z]),
        'if': np.array([args.if_offset_x, args.if_offset_y, args.if_offset_z]),
        'mf': np.array([args.mf_offset_x, args.mf_offset_y, args.mf_offset_z]),
    }
    
    print(f"🔧 Config:")
    print(f"   Default Goal Extension: {args.goal_extension*100:.1f} cm")
    print(f"   Per-Finger Extensions:")
    print(f"      Thumb (th):  {finger_extensions['th']*100:.1f} cm")
    print(f"      Index (if):  {finger_extensions['if']*100:.1f} cm")
    print(f"      Middle (mf): {finger_extensions['mf']*100:.1f} cm")
    print(f"   Global Manual Offset: [{args.goal_offset_x:.3f}, {args.goal_offset_y:.3f}, {args.goal_offset_z:.3f}] m")
    print(f"   Per-Finger Axis Offsets:")
    print(f"      Thumb (th):  [{args.th_offset_x:.3f}, {args.th_offset_y:.3f}, {args.th_offset_z:.3f}] m")
    print(f"      Index (if):  [{args.if_offset_x:.3f}, {args.if_offset_y:.3f}, {args.if_offset_z:.3f}] m")
    print(f"      Middle (mf): [{args.mf_offset_x:.3f}, {args.mf_offset_y:.3f}, {args.mf_offset_z:.3f}] m")
    print(f"   Speed Profile: {args.speed_profile} (accel_factor={args.accel_factor})")
    print(f"   Tau Scale: {args.tau_scale}, Hold Time: {args.hold_time}s")

    # ========================================
    # Multi-CSV Mode
    # ========================================
    if args.csv_pattern:
        print(f"\n[Multi-Demo Mode] Pattern: {args.csv_pattern}")
        csv_files = sorted([f for f in glob(args.csv_pattern) if 'aug' not in Path(f).name])
        
        if len(csv_files) == 0:
            print("❌ No CSV files found.")
            return
        
        print(f"📂 Found {len(csv_files)} demo files (excluding 'aug')")

        out_path = Path(args.out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        os.makedirs(args.plot_dir, exist_ok=True)

        # ========================================
        # Per-demo plots (개별 CSV마다 플롯 생성)
        # ========================================
        if args.per_demo_plots:
            print("\n" + "="*60)
            print("📊 Generating per-demo plots (like compare_dmp_kf.py)...")
            print("="*60)
            
            for csv_idx, csv_path in enumerate(csv_files, 1):
                csv_basename = Path(csv_path).stem
                print(f"\n[{csv_idx}/{len(csv_files)}] Processing {csv_basename}...")
                
                # Process all fingers for this CSV
                demo_data = {}
                for finger in FINGERS:
                    try:
                        x_demo, x_attr_raw = process_data(csv_path, finger, args.force_scale)
                        
                        # Compute auto offset for this demo (손가락별 연장 거리)
                        auto_offset = compute_auto_offset(x_attr_raw, finger_extensions[finger])
                        # 최종 오프셋 = 자동 연장 + 전체 공통 오프셋 + 손가락별 축 오프셋
                        final_offset = auto_offset + manual_offset + finger_offsets[finger]
                        
                        # Train DMP on this single demo (with extended goal)
                        dmp = DiscreteDMP(n_bfs=args.n_bfs)
                        dmp.train(x_attr_raw, dt=0.02, goal_offset=final_offset)
                        x_attr_dmp = dmp.rollout(tau_scale=args.tau_scale, hold_time=args.hold_time,
                                                speed_profile=args.speed_profile, accel_factor=args.accel_factor)
                        
                        demo_data[finger] = {
                            'pos_demo': x_demo,
                            'x_attr_raw': x_attr_raw,
                            'x_attr_dmp': x_attr_dmp,
                            'auto_offset': auto_offset
                        }
                    except Exception as e:
                        print(f"  ✗ {finger}: failed ({e})")
                        continue
                
                if not demo_data:
                    print(f"  ⚠️  No valid finger data, skipping plot")
                    continue
                
                # Generate plot (similar to compare_dmp_kf.py style)
                from mpl_toolkits.mplot3d import Axes3D
                fig = plt.figure(figsize=(18, 18))
                finger_colors = {'th': 'blue', 'if': 'red', 'mf': 'green'}
                
                # 3D Plot - All three fingers (top-left)
                ax3d = fig.add_subplot(3, 3, 1, projection='3d')
                for finger in demo_data.keys():
                    d = demo_data[finger]
                    color = finger_colors[finger]
                    ax3d.plot(d['pos_demo'][:, 0], d['pos_demo'][:, 1], d['pos_demo'][:, 2], 
                              '--', color=color, label=f'{finger.upper()} Demo', alpha=0.4, linewidth=1.5)
                    ax3d.plot(d['x_attr_dmp'][:, 0], d['x_attr_dmp'][:, 1], d['x_attr_dmp'][:, 2], 
                              '-', color=color, label=f'{finger.upper()} DMP', linewidth=2)
                
                ax3d.set_title(f"{csv_basename} - All Fingers (Ext: th={finger_extensions['th']*100:.0f}cm, if={finger_extensions['if']*100:.0f}cm, mf={finger_extensions['mf']*100:.0f}cm)", fontweight='bold')
                ax3d.set_xlabel('X [m]')
                ax3d.set_ylabel('Y [m]')
                ax3d.set_zlabel('Z [m]')
                ax3d.legend(fontsize=8)
                ax3d.grid(True, alpha=0.3)
                
                # Individual finger plots (3x3 grid)
                for idx, finger in enumerate(['th', 'if', 'mf']):
                    if finger not in demo_data:
                        continue
                    
                    d = demo_data[finger]
                    time = np.arange(len(d['pos_demo']))
                    color = finger_colors[finger]
                    
                    # 3D plot for individual finger (top row: positions 2, 3, 4)
                    ax = fig.add_subplot(3, 3, idx + 2, projection='3d')
                    ax.plot(d['pos_demo'][:, 0], d['pos_demo'][:, 1], d['pos_demo'][:, 2], 
                            'k--', label='Demo', alpha=0.5, linewidth=1.5)
                    ax.plot(d['x_attr_raw'][:, 0], d['x_attr_raw'][:, 1], d['x_attr_raw'][:, 2], 
                            '-', color=color, alpha=0.3, linewidth=1, label='K/F Raw')
                    ax.plot(d['x_attr_dmp'][:, 0], d['x_attr_dmp'][:, 1], d['x_attr_dmp'][:, 2], 
                            '-', color=color, linewidth=2, label='DMP Extended')
                    
                    # Mark original and extended goal
                    ax.scatter(*d['x_attr_raw'][-1], color='black', marker='x', s=50, label='Orig Goal')
                    ax.scatter(*d['x_attr_dmp'][-1], color='blue', marker='s', s=80, label='Extended Goal')
                    
                    ax.set_title(f"{finger.upper()} - 3D", fontweight='bold')
                    ax.set_xlabel('X')
                    ax.set_ylabel('Y')
                    ax.set_zlabel('Z')
                    ax.legend(fontsize=7)
                    ax.grid(True, alpha=0.3)
                    
                    # 1D per-axis plot for individual finger (middle row: positions 5, 6, 7)
                    ax1d = fig.add_subplot(3, 3, idx + 5)
                    labels = ['X', 'Y', 'Z']
                    axis_colors = ['r', 'g', 'b']
                    
                    for i in range(3):
                        ax1d.plot(time, d['pos_demo'][:, i], color=axis_colors[i], linestyle='--', 
                                  alpha=0.4, linewidth=1, label=f'Demo {labels[i]}')
                        ax1d.plot(time, d['x_attr_raw'][:, i], color=axis_colors[i], alpha=0.2, 
                                  linewidth=0.8, label=f'K/F {labels[i]}')
                        ax1d.plot(time, d['x_attr_dmp'][:, i], color=axis_colors[i], linewidth=1.5, 
                                  label=f'DMP {labels[i]}')
                    
                    ax1d.set_title(f"{finger.upper()} - Per Axis", fontweight='bold')
                    ax1d.set_xlabel("Time steps")
                    ax1d.set_ylabel("Position [m]")
                    ax1d.legend(loc='best', fontsize=6, ncol=3)
                    ax1d.grid(True, alpha=0.3)
                
                plt.tight_layout()
                
                # Save or show
                match = re.search(r'(\d{8}_\d{6})', csv_basename)
                timestamp = match.group(1) if match else csv_basename
                plot_path = Path(args.plot_dir) / f"dmp_kf_comparison_all_fingers_{timestamp}.png"
                
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                print(f"  💾 Saved: {plot_path}")
                
                if not args.save_plots:
                    plt.show()  # Show interactively
                
                plt.close(fig)
            
            print(f"\n✅ Generated {len(csv_files)} individual plots!")

        # ========================================
        # Global averaged plot (기존 로직)
        # ========================================
        print("\n" + "="*60)
        print("📊 Generating global averaged plot...")
        print("="*60)

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
                    print(f"  ✓ {finger}: {Path(csv_path).name} (len={len(x_attr)})")
                except Exception as e:
                    print(f"  ✗ {finger}: {Path(csv_path).name} skipped ({e})")
                    continue
            
            if len(x_attr_list) == 0:
                print(f"  ❌ No valid trajectories for finger {finger}, skipping")
                continue

            # 1. 평균 궤적 생성
            print(f"  Computing mean trajectory (target_len={args.target_len}) ...")
            x_attr_mean = get_mean_trajectory_simple(x_attr_list, target_len=args.target_len)
            
            # 2. [핵심] 진행 방향 벡터 계산 및 목표 연장 (손가락별)
            auto_offset = compute_auto_offset(x_attr_mean, finger_extensions[finger])
            # 최종 오프셋 = 자동 연장 + 전체 공통 오프셋 + 손가락별 축 오프셋
            final_offset = auto_offset + manual_offset + finger_offsets[finger]
            
            print(f"  🚀 Auto Extension: [{auto_offset[0]*100:.2f}, {auto_offset[1]*100:.2f}, {auto_offset[2]*100:.2f}] cm")
            print(f"  🎯 Finger-specific Offset: [{finger_offsets[finger][0]*100:.2f}, {finger_offsets[finger][1]*100:.2f}, {finger_offsets[finger][2]*100:.2f}] cm")
            print(f"  📍 Final Total Offset: [{final_offset[0]*100:.2f}, {final_offset[1]*100:.2f}, {final_offset[2]*100:.2f}] cm")

            # 3. DMP 학습 (연장된 goal로 학습)
            dmp = DiscreteDMP(n_bfs=args.n_bfs)
            dmp.train(x_attr_mean, dt=0.02, goal_offset=final_offset)
            
            # 4. 궤적 생성 (이미 연장된 goal 사용, 속도 조절 및 끝 유지)
            x_reproduced = dmp.rollout(tau_scale=args.tau_scale, hold_time=args.hold_time,
                                      speed_profile=args.speed_profile, accel_factor=args.accel_factor)

            if args.visualize_alignment:
                print("  [Alignment Check] plotting demos + mean + DMP output...")
                visualize_alignment_quality(x_attr_list, mean_traj=x_attr_mean, dmp_output=x_reproduced)

            # 저장
            save_name = out_path / f"dmp_{finger}_multi_{len(csv_files)}demos.pkl"
            dmp.save(save_name)
            print(f"  ✅ Saved model: {save_name}")
            
            finger_mean[finger] = x_attr_mean
            finger_reproduced[finger] = x_reproduced
            finger_raw_lists[finger] = x_attr_list

        # 시각화: 3D 궤적 플롯
        if len(finger_mean) > 0:
            print("\n📊 Visualizing multi-finger 3D trajectories ...")
            from mpl_toolkits.mplot3d import Axes3D
            
            fig = plt.figure(figsize=(15, 5*((len(finger_mean)+2)//3)))
            
            for idx, finger in enumerate(finger_mean.keys()):
                mean_traj = finger_mean[finger]
                reproduced = finger_reproduced[finger]
                raw_list = finger_raw_lists[finger]
                
                ax = fig.add_subplot((len(finger_mean)+2)//3, 3, idx+1, projection='3d')
                ax.set_title(f"{finger.upper()} - 3D (Ext: {finger_extensions[finger]*100:.1f}cm)", fontsize=12, fontweight='bold')
                
                # Plot raw demos (gray, transparent)
                for raw in raw_list:
                    ax.plot(raw[:, 0], raw[:, 1], raw[:, 2], color='gray', alpha=0.15, linewidth=1)
                
                # Plot mean trajectory (black dots)
                ax.plot(mean_traj[:, 0], mean_traj[:, 1], mean_traj[:, 2], 
                       'k.', alpha=0.4, markersize=1, label='Mean')
                
                # Plot DMP output (red thick line)
                ax.plot(reproduced[:, 0], reproduced[:, 1], reproduced[:, 2], 
                       'r-', linewidth=2.5, label='DMP Extended', alpha=0.9)
                
                # Mark start, original goal, and extended goal
                ax.scatter(*reproduced[0], color='green', s=100, marker='o', label='Start', zorder=10)
                ax.scatter(*mean_traj[-1], color='black', s=100, marker='x', label='Orig Goal', zorder=10)
                ax.scatter(*reproduced[-1], color='blue', s=100, marker='s', label='Extended Goal', zorder=10)
                
                ax.set_xlabel('X (m)', fontsize=9)
                ax.set_ylabel('Y (m)', fontsize=9)
                ax.set_zlabel('Z (m)', fontsize=9)
                ax.legend(fontsize=8, loc='upper right')
                ax.grid(True, alpha=0.3)
                
                # Set equal aspect ratio
                max_range = np.array([reproduced[:, 0].max()-reproduced[:, 0].min(),
                                     reproduced[:, 1].max()-reproduced[:, 1].min(),
                                     reproduced[:, 2].max()-reproduced[:, 2].min()]).max() / 2.0
                mid_x = (reproduced[:, 0].max()+reproduced[:, 0].min()) * 0.5
                mid_y = (reproduced[:, 1].max()+reproduced[:, 1].min()) * 0.5
                mid_z = (reproduced[:, 2].max()+reproduced[:, 2].min()) * 0.5
                ax.set_xlim(mid_x - max_range, mid_x + max_range)
                ax.set_ylim(mid_y - max_range, mid_y + max_range)
                ax.set_zlim(mid_z - max_range, mid_z + max_range)
            
            plt.tight_layout()
            
            # Save or show plot
            csv_basename = Path(csv_files[0]).stem
            match = re.search(r'(\d{8}_\d{6})', csv_basename)
            timestamp = match.group(1) if match else 'multi_demo'
            plot_path = Path(args.plot_dir) / f"dmp_global_3d_all_fingers_{timestamp}_{len(csv_files)}demos.png"
            
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"  💾 Global plot saved to: {plot_path}")
            
            if not args.save_plots:
                plt.show()  # Show interactively
            
            plt.close(fig)
        
        print("\n✅ Multi-demo multi-finger DMP training complete!")
        return
    
    # ========================================
    # Single CSV Mode
    # ========================================
    if not args.csv:
        print("❌ Error: Specify either --csv or --csv_pattern")
        parser.print_help()
        return

    print(f"\n[Single CSV Mode] {args.csv}")
    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 단일 CSV 모드: 모든 손가락 순회
    for finger in FINGERS:
        print(f"\n=== Single CSV Mode: Finger {finger} ===")
        try:
            x_demo, x_attr_raw = process_data(args.csv, finger, args.force_scale)
        except Exception as e:
            print(f"  ✗ Failed to load finger {finger}: {e}")
            continue
        
        if len(x_attr_raw) == 0:
            print(f"  ❌ Empty trajectory for finger {finger}, skip")
            continue

        # 목표 연장 계산 (손가락별)
        auto_offset = compute_auto_offset(x_attr_raw, finger_extensions[finger])
        # 최종 오프셋 = 자동 연장 + 전체 공통 오프셋 + 손가락별 축 오프셋
        final_offset = auto_offset + manual_offset + finger_offsets[finger]
        
        print(f"  🚀 Auto Extension: [{auto_offset[0]*100:.2f}, {auto_offset[1]*100:.2f}, {auto_offset[2]*100:.2f}] cm")
        print(f"  🎯 Finger-specific Offset: [{finger_offsets[finger][0]*100:.2f}, {finger_offsets[finger][1]*100:.2f}, {finger_offsets[finger][2]*100:.2f}] cm")
        print(f"  📍 Final Total Offset: [{final_offset[0]*100:.2f}, {final_offset[1]*100:.2f}, {final_offset[2]*100:.2f}] cm")
        print(f"  Training DMP (n_bfs={args.n_bfs}) with extended goal...")
        
        dmp = DiscreteDMP(n_bfs=args.n_bfs)
        dmp.train(x_attr_raw, dt=0.02, goal_offset=final_offset)
        save_name = out_path / f"dmp_{finger}_{Path(args.csv).stem}.pkl"
        dmp.save(save_name)
        print(f"  ✅ Saved model: {save_name}")

        x_reproduced = dmp.rollout(tau_scale=args.tau_scale, hold_time=args.hold_time,
                                  speed_profile=args.speed_profile, accel_factor=args.accel_factor)
        
        # 시각화: 3D 궤적
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.set_title(f"{finger.upper()} - 3D DMP (Ext: {finger_extensions[finger]*100:.1f}cm)", fontsize=14, fontweight='bold')
        
        # Demo trajectory (green dashed)
        ax.plot(x_demo[:, 0], x_demo[:, 1], x_demo[:, 2], 
               'g--', alpha=0.5, linewidth=2, label='Demo')
        
        # Target attractor (black dots)
        ax.plot(x_attr_raw[:, 0], x_attr_raw[:, 1], x_attr_raw[:, 2], 
               'k.', alpha=0.2, markersize=1, label='Target $x_{attr}$')
        
        # DMP output (red thick line)
        ax.plot(x_reproduced[:, 0], x_reproduced[:, 1], x_reproduced[:, 2], 
               'r-', linewidth=3, label='DMP Extended', alpha=0.9)
        
        # Mark start, original goal, and extended goal
        ax.scatter(*x_reproduced[0], color='green', s=150, marker='o', label='Start', zorder=10)
        ax.scatter(*x_attr_raw[-1], color='black', s=150, marker='x', label='Orig Goal', zorder=10)
        ax.scatter(*x_reproduced[-1], color='blue', s=150, marker='s', label='Extended Goal', zorder=10)
        
        ax.set_xlabel('X (m)', fontsize=11)
        ax.set_ylabel('Y (m)', fontsize=11)
        ax.set_zlabel('Z (m)', fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Equal aspect ratio
        max_range = np.array([x_reproduced[:, 0].max()-x_reproduced[:, 0].min(),
                             x_reproduced[:, 1].max()-x_reproduced[:, 1].min(),
                             x_reproduced[:, 2].max()-x_reproduced[:, 2].min()]).max() / 2.0
        mid_x = (x_reproduced[:, 0].max()+x_reproduced[:, 0].min()) * 0.5
        mid_y = (x_reproduced[:, 1].max()+x_reproduced[:, 1].min()) * 0.5
        mid_z = (x_reproduced[:, 2].max()+x_reproduced[:, 2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        plt.tight_layout()
        
        # Save or show plot
        os.makedirs(args.plot_dir, exist_ok=True)
        csv_stem = Path(args.csv).stem
        plot_path = Path(args.plot_dir) / f"dmp_global_3d_{finger}_{csv_stem}.png"
        
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"  💾 Plot saved to: {plot_path}")
        
        if not args.save_plots:
            plt.show()  # Show interactively
        
        plt.close()
    
    print("\n✅ Single CSV multi-finger processing complete.")


if __name__ == "__main__":
    main()
