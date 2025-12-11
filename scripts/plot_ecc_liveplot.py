#!/usr/bin/env python3
"""
ECC Live Plot - Eccentricity CSV를 30Hz로 라이브 플롯하듯 재생
영상 저장 기능 포함
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
import argparse


def main():
    parser = argparse.ArgumentParser(description='ECC Live Plot')
    parser.add_argument('--csv', type=str, 
                        default='/home/songwoo/ros2_ws/icra2025/install/hri_falcon_robot_bridge/lib/outputs/stiffness_logs/session_20251208_131754_diffusion_t_seq4_h1_pid1588027_01/eccentricity.csv',
                        help='Path to eccentricity.csv')
    parser.add_argument('--fps', type=float, default=30.0, help='Playback FPS (default: 30)')
    parser.add_argument('--save', type=str, default='/home/songwoo/ros2_ws/icra2025/outputs/ecc_liveplot.mp4', help='Output video file path (e.g., output.mp4)')
    parser.add_argument('--speed', type=float, default=1.0, help='Playback speed multiplier')
    args = parser.parse_args()
    
    # Load CSV
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"❌ File not found: {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    print(f"✅ Loaded {len(df)} rows from {csv_path.name}")
    print(f"   Columns: {list(df.columns)}")
    
    time_col = df['time'].values
    ecc_raw = df['ecc_raw'].values
    
    # Figure setup
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # Initialize empty line
    line_raw, = ax.plot([], [], '-', color='#1f77b4', linewidth=1.5, label='ECC')
    
    # Current point marker
    point_marker, = ax.plot([], [], 'o', color='red', markersize=8, zorder=10)
    
    # Text annotations
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=11,
                        verticalalignment='top')
    value_text = ax.text(0.02, 0.88, '', transform=ax.transAxes, fontsize=10,
                         color='#1f77b4', verticalalignment='top')
    
    # Axis settings
    ax.set_xlim(0, time_col[-1] * 1.02)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.5, 1])
    
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Eccentricity', fontsize=12)
    ax.set_title('Eccentricity Live Plot', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=10)
    
    n_frames = len(df)
    
    def init():
        line_raw.set_data([], [])
        point_marker.set_data([], [])
        time_text.set_text('')
        value_text.set_text('')
        return line_raw, point_marker, time_text, value_text
    
    def animate(frame):
        idx = frame + 1
        
        # Update line (show data up to current frame)
        line_raw.set_data(time_col[:idx], ecc_raw[:idx])
        
        # Update current point marker
        point_marker.set_data([time_col[frame]], [ecc_raw[frame]])
        
        # Update text
        current_time = time_col[frame]
        current_value = ecc_raw[frame]
        time_text.set_text(f'Time: {current_time:.3f}s')
        value_text.set_text(f'ECC: {current_value:.4f}')
        
        return line_raw, point_marker, time_text, value_text
    
    # Calculate interval based on FPS and speed
    interval_ms = 1000.0 / (args.fps * args.speed)
    
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=n_frames, interval=interval_ms,
                                   blit=True, repeat=False)
    
    if args.save:
        # Save as video
        output_path = Path(args.save)
        print(f"🎬 Saving animation to: {output_path}")
        
        # Determine writer based on extension
        if output_path.suffix.lower() == '.gif':
            writer = animation.PillowWriter(fps=args.fps)
        else:
            try:
                writer = animation.FFMpegWriter(fps=args.fps, bitrate=2000,
                                                extra_args=['-vcodec', 'libx264'])
            except Exception as e:
                print(f"⚠️ FFMpeg not available, trying Pillow for GIF...")
                output_path = output_path.with_suffix('.gif')
                writer = animation.PillowWriter(fps=args.fps)
        
        anim.save(str(output_path), writer=writer, dpi=100)
        print(f"✅ Saved: {output_path}")
    else:
        # Show live
        print(f"▶️ Playing at {args.fps * args.speed:.1f} FPS (speed: {args.speed}x)")
        print("   Close the window to exit.")
        plt.tight_layout()
        plt.show()
    
    plt.close()


if __name__ == '__main__':
    main()
