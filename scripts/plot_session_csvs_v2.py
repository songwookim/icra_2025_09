#!/usr/bin/env python3
"""
Plot session CSVs in 4 separate windows:
1. Stiffness (3 fingers, smoothed)
2. Force (3 fingers, absolute values, smoothed)
3. Eccentricity (smoothed only)
4. EE Position (3D trajectory with time-based color gradient)
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import uniform_filter1d


def smooth_data(data, window_size=5):
    """Apply moving average smoothing."""
    return uniform_filter1d(data, size=window_size, mode='nearest')


def get_time_colors(n_points):
    """Generate time-based color gradient (dark red → bright red, 0→1 scale)."""
    # Use Reds colormap: lighter at start, darker at end
    import matplotlib.cm as mcm
    cmap = mcm.Reds
    colors = [cmap(0.3 + 0.7 * i / max(n_points - 1, 1)) for i in range(n_points)]
    return colors


def plot_stiffness(df, session_name):
    """Plot 1: Stiffness per finger (1x3 horizontal layout, X/Y/Z in red/green/blue)."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True)
    fig.suptitle(f'Stiffness - {session_name}', fontsize=14, fontweight='bold')
    
    t = df['time'].values
    finger_info = [
        ('th', 'Thumb'),
        ('if', 'Index'),
        ('mf', 'Middle')
    ]
    
    # X=Red, Y=Green, Z=Blue
    axis_colors = {'x': '#e74c3c', 'y': '#2ecc71', 'z': '#3498db'}
    axis_labels = {'x': 'X', 'y': 'Y', 'z': 'Z'}
    
    # First pass: find global y range across all fingers
    global_y_min, global_y_max = float('inf'), float('-inf')
    smoothed_data = {}
    for finger, _ in finger_info:
        smoothed_data[finger] = {}
        for axis in ['x', 'y', 'z']:
            col = f'{finger}_{axis}'
            if col in df.columns:
                smoothed = smooth_data(df[col].values, window_size=21)  # Stronger smoothing
                smoothed_data[finger][axis] = smoothed
                global_y_min = min(global_y_min, smoothed.min())
                global_y_max = max(global_y_max, smoothed.max())
    
    # Add 5% margin
    margin = (global_y_max - global_y_min) * 0.05
    y_lim = (max(0, global_y_min - margin), global_y_max + margin)
    
    # Plot each finger in a horizontal subplot
    for ax, (finger, title) in zip(axes, finger_info):
        for axis in ['x', 'y', 'z']:
            if axis in smoothed_data[finger]:
                ax.plot(t, smoothed_data[finger][axis], '-', 
                       color=axis_colors[axis], label=axis_labels[axis], linewidth=2)
        
        ax.set_ylim(y_lim)
        ax.set_xlabel('Time (s)', fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        if ax == axes[0]:
            ax.set_ylabel('Stiffness (N/m)', fontsize=11)
        if ax == axes[-1]:
            ax.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    return fig


def plot_force(df, session_name):
    """Plot 2: Force sensors per finger (1x3 horizontal layout, X/Y/Z in red/green/blue)."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True)
    fig.suptitle(f'Force Sensors (Absolute, Smoothed) - {session_name}', fontsize=14, fontweight='bold')
    
    t = df['time'].values
    finger_info = [
        ('th', 'Thumb'),
        ('if', 'Index'),
        ('mf', 'Middle')
    ]
    
    # X=Red, Y=Green, Z=Blue (FX, FY, FZ)
    axis_colors = {'fx': '#e74c3c', 'fy': '#2ecc71', 'fz': '#3498db'}
    axis_labels = {'fx': 'X', 'fy': 'Y', 'fz': 'Z'}
    smooth_window = 11  # Mild smoothing to reduce sensor noise
    
    # First pass: find global y max across all fingers (force is always >= 0)
    global_y_max = 0
    force_data = {}
    for finger, _ in finger_info:
        force_data[finger] = {}
        for axis in ['fx', 'fy', 'fz']:
            col = f'{finger}_{axis}'
            if col in df.columns:
                force_abs = np.abs(df[col].values)
                force_smoothed = smooth_data(force_abs, window_size=smooth_window)
                force_data[finger][axis] = force_smoothed
                global_y_max = max(global_y_max, force_smoothed.max())
    
    # y-axis: 0 to global max with 5% margin
    y_lim = (0, global_y_max * 1.05)
    
    # Plot each finger in a horizontal subplot
    for ax, (finger, title) in zip(axes, finger_info):
        for axis in ['fx', 'fy', 'fz']:
            if axis in force_data[finger]:
                ax.plot(t, force_data[finger][axis], '-', 
                       color=axis_colors[axis], label=axis_labels[axis], linewidth=2, alpha=0.9)
        
        ax.set_ylim(y_lim)
        ax.set_xlabel('Time (s)', fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        if ax == axes[0]:
            ax.set_ylabel('|Force| (N)', fontsize=11)
        if ax == axes[-1]:
            ax.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    return fig


def plot_eccentricity(df, session_name):
    """Plot 3: Eccentricity (smoothed only)."""
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.suptitle(f'Deformity Eccentricity - {session_name}', fontsize=12, fontweight='bold')
    
    t = df['time'].values
    
    if 'ecc_smoothed' in df.columns:
        ax.plot(t, df['ecc_smoothed'].values, color='#e74c3c', linewidth=2)
    elif 'ecc_raw' in df.columns:
        # Fallback to raw if smoothed not available
        ax.plot(t, df['ecc_raw'].values, color='#e74c3c', linewidth=2)
    
    ax.set_ylabel('Eccentricity')
    ax.set_xlabel('Time (s)')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    return fig


def plot_ee_position_3d(df, session_name):
    """Plot 4: EE Position 3D trajectory - All 3 fingers combined with time-based red gradient."""
    fig = plt.figure(figsize=(10, 8))
    fig.suptitle(f'EE Position Trajectory - {session_name}', fontsize=12, fontweight='bold')
    
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    
    finger_info = [
        ('th', 'Thumb (TH)', 'o'),
        ('if', 'Index (IF)', 's'),
        ('mf', 'Middle (MF)', '^')
    ]
    
    n_points = len(df)
    # Red gradient: light pink → dark red
    cmap = plt.get_cmap('Reds')
    colors = [cmap(0.3 + 0.7 * i / max(n_points - 1, 1)) for i in range(n_points)]
    
    for finger, label, marker in finger_info:
        # Get actual positions
        x_col = f'{finger}_actual_x'
        y_col = f'{finger}_actual_y'
        z_col = f'{finger}_actual_z'
        
        if all(col in df.columns for col in [x_col, y_col, z_col]):
            x = df[x_col].values * 1000  # Convert to mm
            y = df[y_col].values * 1000
            z = df[z_col].values * 1000
            
            # Plot trajectory with time-based colors (segment by segment)
            for i in range(len(x) - 1):
                ax.plot(x[i:i+2], y[i:i+2], z[i:i+2], 
                       color=colors[i], linewidth=1.5, alpha=0.8)
            
            # Mark start (light) and end (dark red)
            ax.scatter(x[0], y[0], z[0], color=colors[0], s=80, marker=marker, 
                      label=f'{label} Start', edgecolors='black', linewidths=0.5)
            ax.scatter(x[-1], y[-1], z[-1], color=colors[-1], s=120, marker=marker, 
                      edgecolors='black', linewidths=1)
    
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Add colorbar to show time scale
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, df['time'].max()))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label('Time (s)', fontsize=10)
    
    plt.tight_layout()
    return fig


def plot_combined(dfs, session_name):
    """Plot all data in a single figure: Stiffness (row 1), Force/Ecc/EE (row 2).
    
    Layout:
    - Row 1: Stiffness (3 fingers horizontally)
    - Row 2: Force (1 subplot) | Eccentricity | EE Position 3D
    """
    import matplotlib.gridspec as gridspec
    
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(f'Session Overview - {session_name}', fontsize=16, fontweight='bold')
    
    # Create GridSpec: 2 rows, 3 columns
    gs = gridspec.GridSpec(2, 3, figure=fig, height_ratios=[1, 1], hspace=0.3, wspace=0.25)
    
    # Colors
    axis_colors = {'x': '#e74c3c', 'y': '#2ecc71', 'z': '#3498db'}
    axis_labels = {'x': 'X', 'y': 'Y', 'z': 'Z'}
    finger_info = [('th', 'Thumb'), ('if', 'Index'), ('mf', 'Middle')]
    
    # === Row 1: Stiffness (3 subplots) ===
    if 'stiffness' in dfs:
        df = dfs['stiffness']
        t = df['time'].values
        
        # Find global y range
        global_y_min, global_y_max = float('inf'), float('-inf')
        smoothed_data = {}
        for finger, _ in finger_info:
            smoothed_data[finger] = {}
            for axis in ['x', 'y', 'z']:
                col = f'{finger}_{axis}'
                if col in df.columns:
                    smoothed = smooth_data(df[col].values, window_size=21)
                    smoothed_data[finger][axis] = smoothed
                    global_y_min = min(global_y_min, smoothed.min())
                    global_y_max = max(global_y_max, smoothed.max())
        
        margin = (global_y_max - global_y_min) * 0.05
        y_lim = (max(0, global_y_min - margin), global_y_max + margin)
        
        for i, (finger, title) in enumerate(finger_info):
            ax = fig.add_subplot(gs[0, i])
            for axis in ['x', 'y', 'z']:
                if axis in smoothed_data[finger]:
                    ax.plot(t, smoothed_data[finger][axis], '-', 
                           color=axis_colors[axis], label=axis_labels[axis], linewidth=1.5)
            ax.set_ylim(y_lim)
            ax.set_title(f'Stiffness - {title}', fontsize=11, fontweight='bold')
            ax.set_xlabel('Time (s)', fontsize=9)
            if i == 0:
                ax.set_ylabel('Stiffness (N/m)', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', fontsize=8)
    
    # === Row 2, Col 0: Force (all fingers combined) ===
    if 'force' in dfs:
        ax_force = fig.add_subplot(gs[1, 0])
        df = dfs['force']
        t = df['time'].values
        force_smooth_window = 11
        
        finger_colors = {'th': '#e74c3c', 'if': '#3498db', 'mf': '#2ecc71'}
        
        for finger, title in finger_info:
            # Sum of absolute forces (magnitude)
            force_cols = [f'{finger}_fx', f'{finger}_fy', f'{finger}_fz']
            if all(c in df.columns for c in force_cols):
                force_mag_raw = np.sqrt(sum(df[c].values**2 for c in force_cols))
                force_mag = smooth_data(force_mag_raw, window_size=force_smooth_window)
                ax_force.plot(t, force_mag, '-', color=finger_colors[finger], 
                             label=title, linewidth=1.5, alpha=0.9)
        
        ax_force.set_title('Force Magnitude (Smoothed)', fontsize=11, fontweight='bold')
        ax_force.set_xlabel('Time (s)', fontsize=9)
        ax_force.set_ylabel('|Force| (N)', fontsize=9)
        ax_force.grid(True, alpha=0.3)
        ax_force.legend(loc='upper right', fontsize=8)
    
    # === Row 2, Col 1: Eccentricity ===
    if 'eccentricity' in dfs:
        ax_ecc = fig.add_subplot(gs[1, 1])
        df = dfs['eccentricity']
        t = df['time'].values
        
        if 'ecc_smoothed' in df.columns:
            ax_ecc.plot(t, df['ecc_smoothed'].values, color='#e74c3c', linewidth=2)
        elif 'ecc_raw' in df.columns:
            ax_ecc.plot(t, df['ecc_raw'].values, color='#e74c3c', linewidth=2)
        
        ax_ecc.set_title('Eccentricity', fontsize=11, fontweight='bold')
        ax_ecc.set_xlabel('Time (s)', fontsize=9)
        ax_ecc.set_ylabel('Eccentricity', fontsize=9)
        ax_ecc.set_ylim(0, 1)
        ax_ecc.grid(True, alpha=0.3)
    
    # === Row 2, Col 2: EE Position 3D ===
    if 'ee_position' in dfs:
        ax_ee = fig.add_subplot(gs[1, 2], projection='3d')
        df = dfs['ee_position']
        
        n_points = len(df)
        cmap = plt.get_cmap('Reds')
        colors = [cmap(0.3 + 0.7 * i / max(n_points - 1, 1)) for i in range(n_points)]
        
        finger_markers = {'th': 'o', 'if': 's', 'mf': '^'}
        
        for finger, title in finger_info:
            x_col = f'{finger}_actual_x'
            y_col = f'{finger}_actual_y'
            z_col = f'{finger}_actual_z'
            
            if all(col in df.columns for col in [x_col, y_col, z_col]):
                x = df[x_col].values * 1000
                y = df[y_col].values * 1000
                z = df[z_col].values * 1000
                
                # Plot trajectory
                for i in range(len(x) - 1):
                    ax_ee.plot(x[i:i+2], y[i:i+2], z[i:i+2], 
                              color=colors[i], linewidth=1, alpha=0.7)
                
                # Mark start and end
                ax_ee.scatter(x[0], y[0], z[0], color=colors[0], s=40, 
                             marker=finger_markers[finger], edgecolors='black', linewidths=0.3)
                ax_ee.scatter(x[-1], y[-1], z[-1], color=colors[-1], s=60, 
                             marker=finger_markers[finger], edgecolors='black', linewidths=0.5)
        
        ax_ee.set_title('EE Position', fontsize=11, fontweight='bold')
        ax_ee.set_xlabel('X (mm)', fontsize=8)
        ax_ee.set_ylabel('Y (mm)', fontsize=8)
        ax_ee.set_zlabel('Z (mm)', fontsize=8)
        ax_ee.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_session_csvs_v2(session_dir: str):
    """Plot all CSVs from a session directory in 4 separate windows."""
    
    session_path = Path(session_dir)
    if not session_path.exists():
        print(f"Error: Directory not found: {session_dir}")
        return
    
    session_name = session_path.name
    
    # Load CSV files
    csv_files = {
        'stiffness': session_path / 'stiffness.csv',
        'eccentricity': session_path / 'eccentricity.csv',
        'force': session_path / 'force.csv',
        'ee_position': session_path / 'ee_position.csv',
    }
    
    dfs = {}
    for name, filepath in csv_files.items():
        if filepath.exists():
            dfs[name] = pd.read_csv(filepath)
            print(f"Loaded {name}: {len(dfs[name])} rows, time [{dfs[name]['time'].min():.3f}, {dfs[name]['time'].max():.3f}]")
        else:
            print(f"Warning: {name} not found at {filepath}")
    
    # Create 4 separate figures
    figs = []
    
    # Plot 1: Stiffness
    if 'stiffness' in dfs:
        fig1 = plot_stiffness(dfs['stiffness'], session_name)
        figs.append(('stiffness', fig1))
        fig1.savefig(session_path / 'plot_stiffness.png', dpi=150, bbox_inches='tight')
        print(f"Saved: {session_path / 'plot_stiffness.png'}")
    
    # Plot 2: Force
    if 'force' in dfs:
        fig2 = plot_force(dfs['force'], session_name)
        figs.append(('force', fig2))
        fig2.savefig(session_path / 'plot_force.png', dpi=150, bbox_inches='tight')
        print(f"Saved: {session_path / 'plot_force.png'}")
    
    # Plot 3: Eccentricity
    if 'eccentricity' in dfs:
        fig3 = plot_eccentricity(dfs['eccentricity'], session_name)
        figs.append(('eccentricity', fig3))
        fig3.savefig(session_path / 'plot_eccentricity.png', dpi=150, bbox_inches='tight')
        print(f"Saved: {session_path / 'plot_eccentricity.png'}")
    
    # Plot 4: EE Position 3D
    if 'ee_position' in dfs:
        fig4 = plot_ee_position_3d(dfs['ee_position'], session_name)
        figs.append(('ee_position_3d', fig4))
        fig4.savefig(session_path / 'plot_ee_position_3d.png', dpi=150, bbox_inches='tight')
        print(f"Saved: {session_path / 'plot_ee_position_3d.png'}")
    
    # Plot 5: Combined overview
    if dfs:
        fig5 = plot_combined(dfs, session_name)
        figs.append(('combined', fig5))
        fig5.savefig(session_path / 'plot_combined.png', dpi=150, bbox_inches='tight')
        print(f"Saved: {session_path / 'plot_combined.png'}")
    
    print(f"\n✅ All plots saved to: {session_path}")
    plt.show()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        # diffusion_t
        default_dir = '/home/songwoo/ros2_ws/icra2025/install/hri_falcon_robot_bridge/lib/outputs/stiffness_logs/session_20251209_025547_diffusion_t_seq16_h2_pid3174536_02'
        
        # lstm-gmm
        # default_dir = '/home/songwoo/ros2_ws/icra2025/install/hri_falcon_robot_bridge/lib/outputs/stiffness_logs/session_20251208_104004_pid1308665_01'
        
        # bc
        # default_dir = '/home/songwoo/ros2_ws/icra2025/install/hri_falcon_robot_bridge/lib/outputs/stiffness_logs/session_20251208_105937_pid1344409_01'
        
        # default_dir = '/home/songwoo/ros2_ws/icra2025/install/hri_falcon_robot_bridge/lib/outputs/stiffness_logs/session_20251208_110815_pid1362721_01'
        print(f"Usage: python {sys.argv[0]} <session_directory>")
        print(f"Using default: {default_dir}")
        session_dir = default_dir
    else:
        session_dir = sys.argv[1]
    
    plot_session_csvs_v2(session_dir)
