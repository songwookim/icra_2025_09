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
import os
import re
from glob import glob
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
def process_csv(csv_path, dmp_basis=50, f_scale=1.0):
    """Process a single CSV file and return data for all fingers."""
    fingers = ['th', 'if', 'mf']
    data = {}
    
    for finger in fingers:
        pos_demo, force, stiff = load_and_process(csv_path, finger, f_scale)
        x_attr_raw = calculate_virtual_attractor(pos_demo, force, stiff)
        
        dmp = DiscreteDMP(n_bfs=dmp_basis)
        dmp.train(x_attr_raw)
        x_attr_dmp = dmp.rollout()
        
        # Resample DMP to match length
        if len(x_attr_dmp) != len(pos_demo):
            x = np.linspace(0, 1, len(x_attr_dmp))
            f = interp1d(x, x_attr_dmp, axis=0)
            x_new = np.linspace(0, 1, len(pos_demo))
            x_attr_dmp = f(x_new)
        
        data[finger] = {
            'pos_demo': pos_demo,
            'x_attr_raw': x_attr_raw,
            'x_attr_dmp': x_attr_dmp
        }
    
    return data


def plot_comparison(data, title="DMP vs K/F Comparison"):
    """Generate comparison plot for all fingers."""
    fig = plt.figure(figsize=(18, 12))
    finger_colors = {'th': 'red', 'if': 'green', 'mf': 'blue'}
    
    # 3D Plot - All three fingers
    ax3d = fig.add_subplot(2, 3, 1, projection='3d')
    fingers = ['th', 'if', 'mf']
    
    for finger in fingers:
        d = data[finger]
        color = finger_colors[finger]
        ax3d.plot(d['pos_demo'][:, 0], d['pos_demo'][:, 1], d['pos_demo'][:, 2], 
                  '--', color=color, label=f'{finger.upper()} Demo', alpha=0.4, linewidth=1.5)
        ax3d.plot(d['x_attr_dmp'][:, 0], d['x_attr_dmp'][:, 1], d['x_attr_dmp'][:, 2], 
                  '-', color=color, label=f'{finger.upper()} DMP', linewidth=2)
    
    ax3d.set_title(f"{title} - All Fingers", fontweight='bold')
    ax3d.set_xlabel('X [m]')
    ax3d.set_ylabel('Y [m]')
    ax3d.set_zlabel('Z [m]')
    ax3d.legend(fontsize=8)
    ax3d.grid(True, alpha=0.3)
    
    # Individual finger plots (2x3 grid)
    for idx, finger in enumerate(fingers):
        d = data[finger]
        time = np.arange(len(d['pos_demo']))
        color = finger_colors[finger]
        
        # 3D plot for individual finger
        ax = fig.add_subplot(2, 3, idx + 2, projection='3d')
        ax.plot(d['pos_demo'][:, 0], d['pos_demo'][:, 1], d['pos_demo'][:, 2], 
                'k--', label='Demo', alpha=0.5, linewidth=1.5)
        ax.plot(d['x_attr_raw'][:, 0], d['x_attr_raw'][:, 1], d['x_attr_raw'][:, 2], 
                '-', color=color, alpha=0.3, linewidth=1, label='K/F Raw')
        ax.plot(d['x_attr_dmp'][:, 0], d['x_attr_dmp'][:, 1], d['x_attr_dmp'][:, 2], 
                '-', color=color, linewidth=2, label='DMP')
        ax.set_title(f"{finger.upper()} - 3D", fontweight='bold')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        
        # 1D per-axis plot for individual finger
        ax1d = fig.add_subplot(2, 3, idx + 4)  # Bottom row: positions 4, 5, 6
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
    return fig


def main():
    parser = argparse.ArgumentParser(description='Compare Virtual Attractor (K/F) vs DMP trajectory')
    parser.add_argument('--csv', type=str, help='Path to CSV file or folder containing CSV files', 
                        default="/home/songwoo/ros2_ws/icra2025/src/hri_falcon_robot_bridge/outputs/stiffness_profiles_signaligned/")
    parser.add_argument('--finger', type=str, default='if', choices=['th', 'if', 'mf'],
                        help='Finger to analyze (default: if)')
    parser.add_argument('--f-scale', type=float, default=1.0,
                        help='Force scale factor (use -1.0 if attractor goes wrong way)')
    parser.add_argument('--dmp-basis', type=int, default=50,
                        help='Number of DMP basis functions (default: 50)')
    parser.add_argument('--save', type=str, default='/home/songwoo/ros2_ws/icra2025/src/hri_falcon_robot_bridge/scripts/legacy/plots',
                        help='Save path (folder or file). If folder, auto-generates filename from CSV.')
    args = parser.parse_args()
    
    # Determine if csv is file or folder
    csv_files = []
    if os.path.isdir(args.csv):
        # Find all CSV files in the directory (excluding 'aug' files)
        pattern = os.path.join(args.csv, "*.csv")
        csv_files = sorted([f for f in glob(pattern) if 'aug' not in os.path.basename(f)])
        if not csv_files:
            pattern = os.path.join(args.csv, "**/*.csv")
            csv_files = sorted([f for f in glob(pattern, recursive=True) if 'aug' not in os.path.basename(f)])
        print(f"[INFO] Found {len(csv_files)} CSV files in {args.csv} (excluding 'aug' files)")
    elif os.path.isfile(args.csv):
        csv_files = [args.csv]
    else:
        print(f"[ERROR] Path not found: {args.csv}")
        return
    
    if not csv_files:
        print(f"[ERROR] No CSV files found in {args.csv}")
        return
    
    # Process all CSV files
    for csv_idx, csv_path in enumerate(csv_files, 1):
        csv_basename = os.path.basename(csv_path)
        print(f"\n[{csv_idx}/{len(csv_files)}] Processing {csv_basename}...")
        
        try:
            # Process data
            print(f"  Loading data for all fingers...")
            data = process_csv(csv_path, args.dmp_basis, args.f_scale)
            
            # Generate plot
            print(f"  Generating visualization...")
            csv_basename = os.path.basename(csv_path)
            fig = plot_comparison(data, title=csv_basename.replace('.csv', ''))
            # Generate plot
            print(f"  Generating visualization...")
            csv_basename = os.path.basename(csv_path)
            fig = plot_comparison(data, title=csv_basename.replace('.csv', ''))
            
            # Save or show
            if args.save:
                # Check if save path is a directory
                save_path = args.save
                if os.path.isdir(save_path) or len(csv_files) > 1:
                    # Extract timestamp from CSV filename
                    match = re.search(r'(\d{8}_\d{6})', csv_basename)
                    timestamp = match.group(1) if match else os.path.splitext(csv_basename)[0]
                    
                    # Generate filename
                    filename = f"dmp_kf_comparison_all_fingers_{timestamp}.png"
                    save_path = os.path.join(args.save if os.path.isdir(args.save) else os.path.dirname(args.save), filename)
                
                # Ensure parent directory exists
                os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
                
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"  ✓ Saved to: {save_path}")
                plt.close(fig)
            else:
                # Only show if single file
                if len(csv_files) == 1:
                    plt.show()
                else:
                    plt.close(fig)
                    print(f"  ✓ Processed (use --save to save plots)")
        
        except Exception as e:
            print(f"  ✗ Error processing {csv_basename}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n✓ Done! Processed {len(csv_files)} file(s)")
    if args.save:
        print(f"  All plots saved to: {args.save}")


if __name__ == "__main__":
    main()
