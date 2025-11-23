#!/usr/bin/env python3
"""
Virtual Attractor (K/F) vs DMP Trajectory 비교 스크립트

이 스크립트는 두 가지 방법으로 계산된 목표 궤적을 시각화합니다:
1. K/F Method: x_attr = x_demo + F/K (역산된 가상 위치)
2. DMP Method: K/F 궤적을 DMP로 학습하여 부드럽게 재생성한 궤적

사용 예시:
    python3 compare_dmp_kf.py path/to/demo.csv --finger if
    python3 compare_dmp_kf.py path/to/demo.csv --finger if --f-scale -1.0
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# ==============================================================================
# 1. DMP Implementation (Minimal Discrete DMP)
# ==============================================================================
class DiscreteDMP:
    def __init__(self, n_bfs=50, alpha_y=25.0, beta_y=6.25):
        self.n_bfs = n_bfs
        self.alpha_y = alpha_y
        self.beta_y = beta_y
        self.w = None
        self.a_x = 1.0  # Canonical system gain
        
    def _gaussian_basis(self, x):
        centers = np.exp(-self.a_x * np.linspace(0, 1, self.n_bfs))
        widths = (np.diff(centers)[0] if self.n_bfs > 1 else 1.0) ** 2
        h = np.exp(-((x[:, None] - centers[None, :]) ** 2) / (2 * widths))
        return h

    def train(self, trajectory, dt=0.02):
        """Fit DMP forcing term to the given trajectory."""
        n_steps, n_dims = trajectory.shape
        self.y0 = trajectory[0]
        self.goal = trajectory[-1]
        self.dt = dt
        self.tau = n_steps * dt

        # Compute derivatives
        dy = np.gradient(trajectory, axis=0) / dt
        ddy = np.gradient(dy, axis=0) / dt

        # Canonical system
        x = np.exp(-self.a_x * np.linspace(0, 1, n_steps))
        
        # Target forcing term
        # K(g - y) - D*dy + (g - y0)*x*w = ddy ... (Transformation system)
        # f_target = ddy - alpha*(beta*(g - y) - dy)
        f_target = ddy - self.alpha_y * (self.beta_y * (self.goal - trajectory) - dy)
        
        # Weighted Linear Regression for weights w
        psi = self._gaussian_basis(x)  # (N, n_bfs)
        # Solve w for each dimension: (s * psi) * w = f_target
        # s = x (scaling factor)
        
        self.w = np.zeros((self.n_bfs, n_dims))
        for d in range(n_dims):
            # Weighted least squares
            # Minimize sum_i psi_i * (f_target_i - w_i * x_i * psi_i)^2
            # Simplification: standard regression on basis features
            X = psi * x[:, None]
            Y = f_target[:, d]
            # Ridge regression
            self.w[:, d] = np.linalg.inv(X.T @ X + 1e-5 * np.eye(self.n_bfs)) @ (X.T @ Y)

    def rollout(self, dt=None, tau=None):
        """Generate trajectory from learned DMP."""
        if dt is None:
            dt = self.dt
        if tau is None:
            tau = self.tau
        
        n_steps = int(tau / dt)
        y = self.y0.copy()
        dy = np.zeros_like(self.y0)
        ddy = np.zeros_like(self.y0)
        
        path = []
        x = 1.0
        
        for _ in range(n_steps):
            path.append(y.copy())
            
            # Canonical system step
            x_next = x - self.a_x * x * (dt / tau)
            
            # Basis activation
            psi = self._gaussian_basis(np.array([x]))[0]
            f = np.dot(psi * x, self.w) / (np.sum(psi) + 1e-10)
            
            # Transformation system step
            # alpha * (beta * (g - y) - dy) + f
            ddy = self.alpha_y * (self.beta_y * (self.goal - y) - dy) + f
            dy += ddy * (dt / tau)
            y += dy * (dt / tau)
            
            x = x_next
            
        return np.array(path)


# ==============================================================================
# 2. Data Loading & Processing
# ==============================================================================
def load_and_process(csv_path, finger='if', force_scale=1.0):
    df = pd.read_csv(csv_path)
    
    # Select columns based on finger
    # Mapping: th -> s1, if -> s2, mf -> s3 (Adjust if your setup is different)
    sensor_map = {'th': 's1', 'if': 's2', 'mf': 's3'}
    sensor = sensor_map.get(finger, 's2')
    
    # Column names
    pos_cols = [f'ee_{finger}_px', f'ee_{finger}_py', f'ee_{finger}_pz']
    force_cols = [f'{sensor}_fx', f'{sensor}_fy', f'{sensor}_fz']
    stiff_cols = [f'{finger}_k1', f'{finger}_k2', f'{finger}_k3']
    
    # Fallback for legacy pos columns
    if not all(c in df.columns for c in pos_cols):
        pos_cols = ['ee_px', 'ee_py', 'ee_pz']
        
    # Extract data
    pos = df[pos_cols].values
    force = df[force_cols].values * force_scale  # Apply scale (e.g. for sign correction)
    stiff = df[stiff_cols].values
    
    # Filter NaNs
    valid = np.isfinite(pos).all(axis=1) & np.isfinite(force).all(axis=1) & np.isfinite(stiff).all(axis=1)
    pos = pos[valid]
    force = force[valid]
    stiff = stiff[valid]
    
    return pos, force, stiff


def calculate_virtual_attractor(pos, force, stiff):
    """
    Calculate x_attr = x_demo + K^-1 * F
    
    Note: Check signs! If F is 'force exerted by robot', x_attr should be BEYOND x_demo.
          If F is 'reaction force' (negative when pushing), x_attr = x + inv(K)*(-F_reaction)
          Assuming 'force' input is aligned such that positive force means 'pushing further'.
    """
    # Avoid division by zero
    stiff_safe = np.maximum(stiff, 1.0) 
    
    # Offset calculation: delta = F / K
    delta = force / stiff_safe
    
    x_attr = pos + delta
    return x_attr


# ==============================================================================
# 3. Main Visualization
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description='Compare Virtual Attractor (K/F) vs DMP trajectory')
    parser.add_argument('csv', type=str, help='Path to CSV file')
    parser.add_argument('--finger', type=str, default='if', choices=['th', 'if', 'mf'],
                        help='Finger to analyze (default: if)')
    parser.add_argument('--f-scale', type=float, default=1.0,
                        help='Force scale factor (use -1.0 if attractor goes wrong way)')
    parser.add_argument('--dmp-basis', type=int, default=50,
                        help='Number of DMP basis functions (default: 50)')
    parser.add_argument('--save', type=str, default='',
                        help='If set, save figure to this file instead of showing')
    args = parser.parse_args()
    
    # 1. Load Data
    print(f"[1/4] Loading {args.csv} for finger [{args.finger}]...")
    pos_demo, force, stiff = load_and_process(args.csv, args.finger, args.f_scale)
    print(f"      Loaded {len(pos_demo)} samples")
    
    # 2. Method 1: Raw Virtual Attractor (K/F)
    print("[2/4] Calculating Virtual Attractor (K/F)...")
    x_attr_raw = calculate_virtual_attractor(pos_demo, force, stiff)
    
    # 3. Method 2: DMP Fitting
    print(f"[3/4] Training DMP on Virtual Attractor ({args.dmp_basis} basis functions)...")
    dmp = DiscreteDMP(n_bfs=args.dmp_basis)
    dmp.train(x_attr_raw)
    x_attr_dmp = dmp.rollout()
    
    # Resample DMP to match length (just for plotting alignment if slight mismatch)
    if len(x_attr_dmp) != len(pos_demo):
        x = np.linspace(0, 1, len(x_attr_dmp))
        f = interp1d(x, x_attr_dmp, axis=0)
        x_new = np.linspace(0, 1, len(pos_demo))
        x_attr_dmp = f(x_new)

    # 4. Plotting
    print("[4/4] Generating visualization...")
    fig = plt.figure(figsize=(14, 8))
    
    # 3D Plot
    ax3d = fig.add_subplot(1, 2, 1, projection='3d')
    ax3d.plot(pos_demo[:, 0], pos_demo[:, 1], pos_demo[:, 2], 'b--', 
              label='Demo (Actual)', alpha=0.5, linewidth=1.5)
    ax3d.plot(x_attr_raw[:, 0], x_attr_raw[:, 1], x_attr_raw[:, 2], 'g-', 
              label='K/F Attractor (Raw)', alpha=0.3, linewidth=1)
    ax3d.plot(x_attr_dmp[:, 0], x_attr_dmp[:, 1], x_attr_dmp[:, 2], 'r-', 
              label='DMP Attractor (Smooth)', linewidth=2)
    ax3d.set_title(f"3D Trajectory Comparison ({args.finger.upper()})", fontweight='bold')
    ax3d.set_xlabel('X [m]')
    ax3d.set_ylabel('Y [m]')
    ax3d.set_zlabel('Z [m]')
    ax3d.legend()
    ax3d.grid(True, alpha=0.3)
    
    # 1D Plots (X, Y, Z)
    ax1d = fig.add_subplot(1, 2, 2)
    time = np.arange(len(pos_demo))
    
    labels = ['X', 'Y', 'Z']
    colors = ['r', 'g', 'b']
    
    for i in range(3):
        ax1d.plot(time, pos_demo[:, i], color=colors[i], linestyle='--', 
                  alpha=0.4, linewidth=1.5, label=f'Demo {labels[i]}')
        ax1d.plot(time, x_attr_raw[:, i], color=colors[i], alpha=0.3, 
                  linewidth=1, label=f'K/F {labels[i]}')
        ax1d.plot(time, x_attr_dmp[:, i], color=colors[i], linewidth=2, 
                  label=f'DMP {labels[i]}')
        
    ax1d.set_title("Per-Axis Comparison", fontweight='bold')
    ax1d.set_xlabel("Time steps")
    ax1d.set_ylabel("Position [m]")
    ax1d.legend(loc='best', fontsize=8, ncol=3)
    ax1d.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if args.save:
        plt.savefig(args.save, dpi=150, bbox_inches='tight')
        print(f"✓ Saved figure to: {args.save}")
    else:
        plt.show()
    
    print("\n✓ Done!")
    print("\n비교 포인트:")
    print("  1. 접촉 구간: Demo(파랑)은 멈췄지만 K/F(초록)는 더 안쪽으로 들어가는지 확인")
    print("  2. 노이즈 vs 스무딩: K/F(초록)는 떨리지만 DMP(빨강)는 부드러운지 확인")
    print("  3. 방향 체크: Attractor가 반대로 튀면 --f-scale -1.0 옵션 사용")


if __name__ == "__main__":
    main()
