#!/usr/bin/env python3
"""
Track 1: DMP Motion Learning
----------------------------
1. CSV 데이터 로드 (x_demo, F_demo, K_demo)
2. 가상 목표 궤적(x_attr) 역산: x_attr = x_demo + F / K
3. DMP 학습 (가중치 w 추출)
4. 결과 저장 (.pkl) 및 시각화
"""

import argparse
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import interp1d
from glob import glob

# 멀티 손가락 처리 대상 (엄지 th, 검지 if, 중지 mf)
FINGERS = ["th", "if", "mf"]

# ======================================================
# 0. Trajectory Alignment Functions
# ======================================================
def get_mean_trajectory_simple(demo_list, target_len=200):
    """
    [1단계: 선형 시간 정규화 - 필수]
    모든 데모를 0% ~ 100% 진행률로 치환해서 같은 길이로 맞춘 뒤 평균 냄.
    사람이 '비슷한 속도'로 움직였다면 이것만으로 충분함.
    
    Args:
        demo_list: List of trajectories, each (T_i, D) where T_i can vary
        target_len: 정규화할 표준 길이 (default: 200)
    
    Returns:
        mean_traj: (target_len, D) 평균 궤적
    """
    if len(demo_list) == 0:
        raise ValueError("demo_list is empty")
    
    interpolated_trajs = []
    
    for traj in demo_list:
        T = len(traj)
        if T < 2:
            print(f"⚠️  Warning: trajectory too short (len={T}), skipping...")
            continue
            
        # 0부터 1까지의 시간 축 생성
        x_old = np.linspace(0, 1, T)
        x_new = np.linspace(0, 1, target_len)
        
        # 보간 (Resampling) - 각 차원별로 선형 보간
        f = interp1d(x_old, traj, axis=0, kind='linear')
        traj_new = f(x_new)
        interpolated_trajs.append(traj_new)
    
    if len(interpolated_trajs) == 0:
        raise ValueError("No valid trajectories after filtering")
        
    # 단순 평균
    mean_traj = np.mean(np.stack(interpolated_trajs), axis=0)
    
    print(f"✅ Aligned {len(interpolated_trajs)} demos → target_len={target_len}")
    return mean_traj


def visualize_alignment_quality(demo_list, mean_traj=None, dmp_output=None):
    """
    [진단 도구] 여러 데모를 겹쳐 그려서 DTW 필요성 판단
    
    피크(꺾이는 점)들이 비슷한 x축 위치에 모여 있다 → DTW 불필요
    피크들이 중구난방으로 퍼져 있다 → DTW 필요
    
    각 축을 0~1로 정규화해서 형상 비교를 쉽게 함
    
    Args:
        demo_list: 원본 데모 궤적 리스트
        mean_traj: 평균 궤적 (선택)
        dmp_output: DMP 학습 후 출력 궤적 (선택)
    """
    plt.figure(figsize=(12, 4))
    dims = ['X', 'Y', 'Z']
    
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(f"Axis {dims[i]} - Alignment Check (Normalized)")
        
        # 전체 데모에서 min/max 찾기 (정규화 기준)
        all_vals = np.concatenate([traj[:, i] for traj in demo_list])
        if mean_traj is not None:
            all_vals = np.concatenate([all_vals, mean_traj[:, i]])
        if dmp_output is not None:
            all_vals = np.concatenate([all_vals, dmp_output[:, i]])
        v_min, v_max = all_vals.min(), all_vals.max()
        v_range = v_max - v_min if v_max > v_min else 1.0
        
        # 모든 데모를 정규화 후 겹쳐 그리기
        for idx, traj in enumerate(demo_list):
            progress = np.linspace(0, 1, len(traj))
            # MinMax 정규화: (x - min) / (max - min)
            normalized = (traj[:, i] - v_min) / v_range
            plt.plot(progress, normalized, 'b-', alpha=0.15, linewidth=1, label='Demos' if idx == 0 else None)
        
        # 평균 궤적 (있으면)
        if mean_traj is not None:
            progress_mean = np.linspace(0, 1, len(mean_traj))
            normalized_mean = (mean_traj[:, i] - v_min) / v_range
            plt.plot(progress_mean, normalized_mean, 'orange', linewidth=2.5, alpha=0.8, label='Mean', linestyle='--')
        
        # DMP 출력 (있으면)
        if dmp_output is not None:
            progress_dmp = np.linspace(0, 1, len(dmp_output))
            normalized_dmp = (dmp_output[:, i] - v_min) / v_range
            plt.plot(progress_dmp, normalized_dmp, 'r-', linewidth=3, label='DMP Output')
            
        plt.xlabel("Progress (0→1)")
        plt.ylabel("Normalized Position (0→1)")
        plt.ylim(-0.05, 1.05)
        plt.grid(True, alpha=0.3)
        if i == 0:
            plt.legend(loc='upper right', fontsize=9)
            # 원본 범위 표시
            plt.text(0.02, 0.98, f"Original: [{v_min:.3f}, {v_max:.3f}]", 
                    transform=plt.gca().transAxes, fontsize=8, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.suptitle("👀 Check: Are peaks aligned? (Yes→OK, No→Need DTW)", y=1.02)
    plt.show()


# ======================================================
# 1. DMP Class (이미 검증된 코드)
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
        """
        [학습 단계]
        입력: 계산된 x_attr 궤적 (TimeSteps, Dims)
        출력: 가중치 w 학습
        """
        n_steps, n_dims = trajectory.shape
        self.y0 = trajectory[0]
        self.goal = trajectory[-1]
        self.dt = dt
        self.tau = n_steps * dt

        # 미분 (속도, 가속도) 계산
        dy = np.gradient(trajectory, axis=0) / dt
        ddy = np.gradient(dy, axis=0) / dt

        # Canonical system (시간 s)
        x = np.exp(-self.a_x * np.linspace(0, 1, n_steps))
        
        # Target Force 계산 (f_target)
        # Transformation System 식을 뒤집어서 f를 구함
        f_target = ddy - self.alpha_y * (self.beta_y * (self.goal - trajectory) - dy)
        
        # Linear Regression으로 가중치 w 구하기
        psi = self._gaussian_basis(x)
        self.w = np.zeros((self.n_bfs, n_dims))
        for d in range(n_dims):
            X = psi * x[:, None]
            Y = f_target[:, d]
            # Ridge Regression (안정성 위해 1e-5 추가)
            self.w[:, d] = np.linalg.inv(X.T @ X + 1e-5 * np.eye(self.n_bfs)) @ (X.T @ Y)
        
        print(f"✅ DMP Training Done. Weights shape: {self.w.shape}")

    def save(self, path):
        """학습된 파라미터 저장"""
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

    def rollout(self, dt=None, tau=None):
        """검증용 재생성"""
        if dt is None: dt = self.dt
        if tau is None: tau = self.tau
        n_steps = int(tau / dt)
        y = self.y0.copy()
        dy = np.zeros_like(y)
        path = []
        x = 1.0
        for _ in range(n_steps):
            path.append(y.copy())
            x_next = x - self.a_x * x * (dt / tau)
            psi = self._gaussian_basis(np.array([x]))[0]
            f = np.dot(psi * x, self.w) / (np.sum(psi) + 1e-10)
            ddy = self.alpha_y * (self.beta_y * (self.goal - y) - dy) + f
            dy += ddy * (dt / tau)
            y += dy * (dt / tau)
            x = x_next
        return np.array(path)

# ======================================================
# 2. 데이터 로드 및 x_attr 계산 함수
# ======================================================
def process_data(csv_path, finger, force_scale=1.0):
    df = pd.read_csv(csv_path)
    
    # 컬럼 매핑 (데이터셋에 맞게 수정 필요)
    # 예: th -> s1 (엄지), if -> s2 (검지), mf -> s3 (중지)
    sensor_map = {'th': 's1', 'if': 's2', 'mf': 's3'}
    s_idx = sensor_map[finger]

    # 1. x_demo (현재 위치) 로드
    # 컬럼명 예시: ee_if_px, ee_if_py, ee_if_pz
    pos_cols = [f'ee_{finger}_px', f'ee_{finger}_py', f'ee_{finger}_pz']
    if not all(c in df.columns for c in pos_cols): # 구버전 호환
        pos_cols = ['ee_px', 'ee_py', 'ee_pz']
    x_demo = df[pos_cols].values

    # 2. F_demo (현재 힘) 로드
    force_cols = [f'{s_idx}_fx', f'{s_idx}_fy', f'{s_idx}_fz']
    F_demo = df[force_cols].values * force_scale

    # 3. K_demo (당시 강성) 로드
    stiff_cols = [f'{finger}_k1', f'{finger}_k2', f'{finger}_k3']
    K_demo = df[stiff_cols].values

    # 데이터 유효성 검사 (NaN 제거)
    valid_mask = np.isfinite(x_demo).all(axis=1) & \
                 np.isfinite(F_demo).all(axis=1) & \
                 np.isfinite(K_demo).all(axis=1)
    x_demo = x_demo[valid_mask]
    F_demo = F_demo[valid_mask]
    K_demo = K_demo[valid_mask]

    # ---------------------------------------------------------
    # ★ 핵심: 가상 궤적 역산 (Inverse Calculation) ★
    # x_attr = x_demo + F / K
    # ---------------------------------------------------------
    # K가 0이거나 너무 작으면 나눗셈 폭발하므로 안전장치(clip) 추가
    K_safe = np.maximum(K_demo, 1.0) 
    
    # 역산 수행
    x_attr_raw = x_demo + (F_demo / K_safe)

    return x_demo, x_attr_raw

# ======================================================
# 3. 메인 실행 함수
# ======================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, help='Path to single demo CSV (legacy mode)')
    parser.add_argument('--csv_pattern', type=str, help='Glob pattern for multiple CSVs (e.g., "outputs/*.csv")', default="/home/songwoo/ros2_ws/icra2025/src/hri_falcon_robot_bridge/outputs/stiffness_profiles_signaligned/*.csv")
    parser.add_argument('--n_bfs', type=int, default=50, help='Number of basis functions')
    parser.add_argument('--out_dir', type=str, default='/home/songwoo/ros2_ws/icra2025/src/hri_falcon_robot_bridge/scripts/legacy/dmp_models')
    parser.add_argument('--force_scale', type=float, default=1.0, help='Direction of force (1.0 or -1.0)')
    parser.add_argument('--target_len', type=int, default=1500, help='Target trajectory length for alignment')
    parser.add_argument('--visualize_alignment', action='store_true', help='Show alignment quality check plot', default=True)
    args = parser.parse_args()

    # ========================================
    # 다중 CSV 모드: 여러 데모 평균화
    # ========================================
    if args.csv_pattern:
        print(f"[Multi-Demo Mode] Pattern: {args.csv_pattern}")
        csv_files = sorted([f for f in glob(args.csv_pattern) if 'aug' not in Path(f).name])
        if len(csv_files) == 0:
            print(f"❌ No CSV files found matching pattern: {args.csv_pattern}")
            return
        print(f"📂 Found {len(csv_files)} demo files (excluding 'aug')")

        out_path = Path(args.out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        finger_mean = {}
        finger_dmp = {}
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

            print(f"  Computing mean trajectory (target_len={args.target_len}) ...")
            x_attr_mean = get_mean_trajectory_simple(x_attr_list, target_len=args.target_len)
            dmp = DiscreteDMP(n_bfs=args.n_bfs)
            dmp.train(x_attr_mean, dt=0.02)
            x_reproduced = dmp.rollout()

            if args.visualize_alignment:
                print("  [Alignment Check] plotting demos + mean + DMP output...")
                visualize_alignment_quality(x_attr_list, mean_traj=x_attr_mean, dmp_output=x_reproduced)

            save_name = out_path / f"dmp_{finger}_multi_{len(csv_files)}demos.pkl"
            dmp.save(save_name)
            print(f"  ✅ Saved model: {save_name}")

            finger_mean[finger] = x_attr_mean
            finger_dmp[finger] = dmp
            finger_reproduced[finger] = x_reproduced
            finger_raw_lists[finger] = x_attr_list

        # 시각화: 손가락별 3축 (3x3 subplot)
        if len(finger_mean) > 0:
            print("\n📊 Visualizing multi-finger results ...")
            fig, axes = plt.subplots(len(finger_mean), 3, figsize=(12, 4*len(finger_mean)))
            dims = ['X', 'Y', 'Z']
            for r, finger in enumerate(finger_mean.keys()):
                mean_traj = finger_mean[finger]
                reproduced = finger_reproduced[finger]
                raw_list = finger_raw_lists[finger]
                for c in range(3):
                    ax = axes[r, c] if len(finger_mean) > 1 else axes[c]
                    ax.set_title(f"{finger} - {dims[c]}")
                    for raw in raw_list:
                        ax.plot(raw[:, c], color='gray', alpha=0.08, linewidth=1)
                    ax.plot(mean_traj[:, c], 'k.', alpha=0.3, markersize=2, label='Mean')
                    ax.plot(reproduced[:, c], 'r-', linewidth=2, label='DMP')
                    ax.grid(alpha=0.3)
                    if r == 0 and c == 0:
                        ax.legend()
            plt.tight_layout()
            plt.show()
        print("\n✅ Multi-demo multi-finger DMP training complete!")
        return
    
    # ========================================
    # 단일 CSV 모드: 기존 방식 유지
    # ========================================
    if not args.csv:
        print("❌ Error: Specify either --csv or --csv_pattern")
        parser.print_help()
        return

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

        print(f"  Training DMP (n_bfs={args.n_bfs}) ...")
        dmp = DiscreteDMP(n_bfs=args.n_bfs)
        dmp.train(x_attr_raw, dt=0.02)
        save_name = out_path / f"dmp_{finger}_{Path(args.csv).stem}.pkl"
        dmp.save(save_name)
        print(f"  ✅ Saved model: {save_name}")

        x_reproduced = dmp.rollout()
        # 시각화
        plt.figure(figsize=(10, 4))
        dims = ['x', 'y', 'z']
        for i in range(3):
            plt.subplot(1, 3, i+1)
            plt.title(f"{finger} Axis {dims[i]}")
            plt.plot(x_demo[:, i], 'g--', alpha=0.3, label='Demo')
            plt.plot(x_attr_raw[:, i], 'k.', alpha=0.1, label='Target $x_{attr}$')
            plt.plot(x_reproduced[:, i], 'r-', linewidth=2, label='DMP')
            if i == 0:
                plt.legend()
            plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
    print("\n✅ Single CSV multi-finger processing complete.")

if __name__ == "__main__":
    main()