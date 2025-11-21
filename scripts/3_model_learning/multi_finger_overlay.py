#!/usr/bin/env python3
"""Multi-finger 3D trajectory overlay (Matplotlib with Plotly fallback).

Loads a CSV containing columns like:
    ee_th_px, ee_th_py, ee_th_pz
    ee_if_px, ee_if_py, ee_if_pz
    ee_mf_px, ee_mf_py, ee_mf_pz
Optionally stiffness columns (per finger):
    th_k1, th_k2, th_k3 (etc.) used to derive a scalar color magnitude.

Features:
  - Automatically discovers available finger prefixes among: th, if, mf, rf, lf.
  - Optional centering & normalization of each finger trajectory.
  - Stiffness coloring (average or norm) if --stiffness-color specified.
  - Fallback to Plotly if Matplotlib 3D not available.

Usage examples:
    python3 multi_finger_overlay.py --csv path/to/profile.csv
    python3 multi_finger_overlay.py --csv path/to/profile.csv --center --normalize
    python3 multi_finger_overlay.py --csv path/to/profile.csv --stiffness-color --plotly
    python3 multi_finger_overlay.py --csv path/to/profile.csv --stiffness-color --stiffness-mode norm

Outputs:
    multi_finger_overlay_matplotlib.png
    multi_finger_overlay_plotly.html (if fallback or forced)

"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    pd = None  # type: ignore

FINGER_PREFIXES = ["th", "if", "mf"]

# --------------------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------------------

def load_finger_positions(csv_path: Path, center: bool, normalize: bool, scale: float) -> Dict[str, np.ndarray]:
    if not HAS_PANDAS:
        raise RuntimeError("pandas not installed: pip install pandas")
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)  # type: ignore
    result: Dict[str, np.ndarray] = {}
    for prefix in FINGER_PREFIXES:
        # Try standard naming first: ee_<prefix>_px/py/pz
        cols = [f"ee_{prefix}_px", f"ee_{prefix}_py", f"ee_{prefix}_pz"]
        # Special case: if uses ee_px/py/pz (no prefix)
        if prefix == "if" and not all(c in df.columns for c in cols):
            cols = ["ee_px", "ee_py", "ee_pz"]
        
        if all(c in df.columns for c in cols):
            arr = df[cols].to_numpy(dtype=float)  # type: ignore
            if center:
                arr = arr - np.mean(arr, axis=0, keepdims=True)
            if normalize:
                std = np.std(arr, axis=0, keepdims=True)
                std[std < 1e-9] = 1.0
                arr = arr / std
            arr = arr * scale
            result[prefix] = arr
    return result

def load_finger_stiffness(csv_path: Path, prefixes: List[str]) -> Dict[str, np.ndarray]:
    if not HAS_PANDAS:
        return {}
    if not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path)  # type: ignore
    stiff_map: Dict[str, np.ndarray] = {}
    for prefix in prefixes:
        # Try standard naming first: <prefix>_k1/k2/k3
        kcols = [f"{prefix}_k1", f"{prefix}_k2", f"{prefix}_k3"]
        # Special case: if might use k1/k2/k3 (no prefix)
        if prefix == "if" and not all(c in df.columns for c in kcols):
            kcols = ["k1", "k2", "k3"]
        
        if all(c in df.columns for c in kcols):
            karr = df[kcols].to_numpy(dtype=float)  # type: ignore
            stiff_map[prefix] = karr
    return stiff_map

# --------------------------------------------------------------------------------------
# Stiffness scalar derivation
# --------------------------------------------------------------------------------------

def stiffness_to_scalar(karr: np.ndarray, mode: str) -> np.ndarray:
    if karr.ndim != 2 or karr.shape[1] < 3:
        return np.zeros(karr.shape[0])
    if mode == "norm":
        return np.linalg.norm(karr[:, :3], axis=1)
    # default: average
    return np.mean(karr[:, :3], axis=1)

# --------------------------------------------------------------------------------------
# Matplotlib plotting
# --------------------------------------------------------------------------------------

def try_plot_matplotlib(traj_map: Dict[str, np.ndarray], stiff_map: Dict[str, np.ndarray], stiffness_color: bool, stiffness_mode: str, output_path: Path, max_points: int) -> bool:
    try:
        import matplotlib
        if not os.environ.get("DISPLAY"):
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: F401
        try:
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        except Exception as e:
            print(f"[matplotlib warn] Axes3D import failed: {e}")
            return False
        if not traj_map:
            print("[matplotlib warn] No finger trajectories found.")
            return False
        fig = plt.figure(figsize=(6.4, 5.6), dpi=110)
        ax = fig.add_subplot(111, projection="3d")
        ax.set_title("Multi-Finger Trajectories (Matplotlib)")
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_zlabel("Z [m]")

        # Color cycle fallback
        prop_cycle = plt.rcParams.get("axes.prop_cycle")
        color_cycle = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
        if prop_cycle is not None:
            try:
                extracted = prop_cycle.by_key().get("color", None)
                if extracted:
                    color_cycle = extracted
            except Exception:
                pass
        ranges_accum = []
        centers_accum = []

        # Explicit color mapping for visibility
        finger_colors = {"th": "#1f77b4", "if": "#ff7f0e", "mf": "#2ca02c", "rf": "#d62728", "lf": "#9467bd"}

        # Plot in reverse order so first fingers are on top
        for i, (prefix, arr) in enumerate(reversed(list(traj_map.items()))):
            data = arr[:max_points]
            print(f"[debug] {prefix}: {data.shape[0]} points, range X=[{data[:,0].min():.6f}, {data[:,0].max():.6f}], Y=[{data[:,1].min():.6f}, {data[:,1].max():.6f}], Z=[{data[:,2].min():.6f}, {data[:,2].max():.6f}]")
            if stiffness_color and prefix in stiff_map:
                sval = stiffness_to_scalar(stiff_map[prefix][:data.shape[0]], stiffness_mode)
                norm_s = (sval - sval.min()) / max(np.ptp(sval), 1e-9)
                sc = ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=norm_s, s=9, cmap="viridis", marker="o", label=f"{prefix} (stiffness)")
                cb = fig.colorbar(sc, pad=0.02, fraction=0.05)
                cb.set_label(f"{prefix} stiffness {stiffness_mode}")
            else:
                color = finger_colors.get(prefix, color_cycle[i % len(color_cycle)])
                # Pure line plot with explicit zorder (reverse i for proper layering)
                ax.plot(data[:, 0], data[:, 1], data[:, 2], lw=4.0, label=f"{prefix.upper()}", 
                        color=color, alpha=0.95, zorder=i)
                print(f"[plot] {prefix}: color={color}, zorder={i}, {data.shape[0]} points")
            ranges_accum.append(np.ptp(data, axis=0))
            centers_accum.append(np.mean(data, axis=0))

        # Aspect balancing across all fingers
        all_centers = np.mean(np.stack(centers_accum, axis=0), axis=0)
        max_range = np.max(np.max(np.stack(ranges_accum, axis=0), axis=0))
        
        # Use actual data aspect if range is very small (< 1cm)
        if max_range < 0.01:
            print(f"[info] Small range detected ({max_range*1000:.3f}mm); using 'auto' aspect mode for visibility")
            ax.set_box_aspect(None)  # auto aspect
        else:
            # Equal aspect for larger ranges
            for dim, c in enumerate(all_centers):
                getattr(ax, f"set_{'xyz'[dim]}lim")(c - max_range/2, c + max_range/2)

        ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
        fig.tight_layout()
        fig.savefig(str(output_path), bbox_inches="tight", dpi=150)
        print(f"[matplotlib ok] Saved: {output_path}")
        if matplotlib.get_backend().lower() not in ("agg",) and os.environ.get("DISPLAY"):
            plt.show()
        plt.close(fig)
        return True
    except Exception as e:
        print(f"[matplotlib error] {e}")
        return False

# --------------------------------------------------------------------------------------
# Plotly fallback
# --------------------------------------------------------------------------------------

def plot_plotly(traj_map: Dict[str, np.ndarray], stiff_map: Dict[str, np.ndarray], stiffness_color: bool, stiffness_mode: str, output_html: Path, max_points: int) -> None:
    import plotly.graph_objects as go
    
    # Explicit color mapping matching matplotlib
    finger_colors = {"th": "#1f77b4", "if": "#ff7f0e", "mf": "#2ca02c", "rf": "#d62728", "lf": "#9467bd"}
    
    fig = go.Figure()
    if not traj_map:
        print("[plotly warn] No trajectories found.")
    for prefix, arr in traj_map.items():
        data = arr[:max_points]
        color = finger_colors.get(prefix, "#888888")
        if stiffness_color and prefix in stiff_map:
            sval = stiffness_to_scalar(stiff_map[prefix][:data.shape[0]], stiffness_mode)
            norm_s = (sval - sval.min()) / max(np.ptp(sval), 1e-9)
            fig.add_trace(go.Scatter3d(x=data[:, 0], y=data[:, 1], z=data[:, 2], mode="markers", name=prefix.upper(),
                                       marker=dict(size=3.5, color=norm_s, colorscale="Viridis", showscale=True, colorbar=dict(title=f"{prefix} {stiffness_mode}"))))
        else:
            # Pure lines mode matching matplotlib
            fig.add_trace(go.Scatter3d(x=data[:, 0], y=data[:, 1], z=data[:, 2], mode="lines", name=prefix.upper(), 
                                       line=dict(width=6, color=color)))

    fig.update_layout(title="Multi-Finger Trajectories (Plotly)", scene=dict(aspectmode="data", xaxis_title="X [m]", yaxis_title="Y [m]", zaxis_title="Z [m]"))
    fig.write_html(str(output_html), include_plotlyjs="cdn")
    print(f"[plotly ok] Saved: {output_html}")

# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Overlay multiple finger trajectories in 3D")
    p.add_argument("--csv", type=Path, default=None, help="CSV containing ee_<finger>_px,ee_<finger>_py,ee_<finger>_pz columns (auto-search if omitted)")
    p.add_argument("--center", action="store_true", help="Center each trajectory by subtracting mean")
    p.add_argument("--normalize", action="store_true", help="Divide each trajectory by its std dev")
    p.add_argument("--scale", type=float, default=1.0, help="Scale factor applied after center/normalize")
    p.add_argument("--plotly", action="store_true", help="Force plotly even if matplotlib works")
    p.add_argument("--stiffness-color", action="store_true", help="Color points/markers by stiffness scalar per finger")
    p.add_argument("--stiffness-mode", type=str, default="avg", choices=["avg", "norm"], help="Scalar mode: avg or norm")
    p.add_argument("--max-points", type=int, default=5000, help="Truncate each finger to this many points for speed")
    p.add_argument("--synthetic", action="store_true", help="Generate synthetic trajectories if no CSV found")
    p.add_argument("--search-pattern", type=str, default="src/hri_falcon_robot_bridge/outputs/logs/success/*_synced.csv", help="Glob pattern for auto-search")
    p.add_argument("--matplotlib-only", action="store_true", help="Generate only matplotlib PNG (skip plotly HTML)")
    return p.parse_args()

# --------------------------------------------------------------------------------------
# Auto-search & synthetic fallback
# --------------------------------------------------------------------------------------

def auto_search_csv(pattern: str) -> Path | None:
    # Support both absolute and relative patterns
    if Path(pattern).is_absolute() or pattern.startswith('/'):
        search_path = Path(pattern).parent
        glob_pattern = Path(pattern).name
        candidates = list(search_path.glob(glob_pattern)) if search_path.exists() else []
    else:
        workspace = Path.cwd()
        candidates = list(workspace.glob(pattern))
    if not candidates:
        return None
    # Sort by modification time descending
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]

def generate_synthetic_trajectories(n: int) -> Dict[str, np.ndarray]:
    t = np.linspace(0, 4 * np.pi, n)
    result: Dict[str, np.ndarray] = {}
    offsets = {"th": [0.0, 0.0, 0.0], "if": [0.08, 0.0, 0.02], "mf": [0.04, 0.07, 0.01]}
    for prefix, offset in offsets.items():
        x = 0.05 * np.cos(t) * (1.0 + 0.3 * np.sin(0.5 * t)) + offset[0]
        y = 0.05 * np.sin(t) * (1.0 + 0.3 * np.sin(0.5 * t)) + offset[1]
        z = 0.02 * t / np.pi + 0.01 * np.sin(1.2 * t) + offset[2]
        result[prefix] = np.stack([x, y, z], axis=1)
    return result

# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def main():
    args = parse_args()
    
    csv_path = args.csv
    # Auto-search if not provided
    if csv_path is None:
        print(f"[info] No CSV provided; searching for pattern: {args.search_pattern}")
        csv_path = auto_search_csv(args.search_pattern)
        if csv_path:
            print(f"[info] Auto-selected: {csv_path}")
        else:
            print(f"[warn] No CSV found matching pattern {args.search_pattern}")
    
    # Try loading from CSV
    traj_map: Dict[str, np.ndarray] = {}
    stiff_map: Dict[str, np.ndarray] = {}
    if csv_path and csv_path.exists():
        try:
            traj_map = load_finger_positions(csv_path, args.center, args.normalize, args.scale)
            if traj_map:
                loaded_fingers = ', '.join(sorted(traj_map.keys()))
                missing_fingers = set(FINGER_PREFIXES) - set(traj_map.keys())
                print(f"[info] Loaded fingers: {loaded_fingers}")
                if missing_fingers:
                    print(f"[warn] Missing fingers in CSV: {', '.join(sorted(missing_fingers))}")
                if args.stiffness_color:
                    stiff_map = load_finger_stiffness(csv_path, list(traj_map.keys()))
        except Exception as e:
            print(f"[warn] Failed to load CSV {csv_path}: {e}")
    
    # Fallback to synthetic if needed
    if not traj_map:
        if args.synthetic or csv_path is None:
            print("[info] Generating synthetic trajectories...")
            traj_map = generate_synthetic_trajectories(args.max_points)
        else:
            raise RuntimeError(f"No usable finger trajectories found in {csv_path}. Use --synthetic to generate data.")

    out_png = Path("multi_finger_overlay_matplotlib.png")
    out_html = Path("multi_finger_overlay_plotly.html")

    # Generate matplotlib unless --plotly flag
    used_mpl = False
    if not args.plotly:
        used_mpl = try_plot_matplotlib(traj_map, stiff_map, args.stiffness_color, args.stiffness_mode, out_png, args.max_points)
    
    # Generate plotly unless --matplotlib-only flag
    if not args.matplotlib_only:
        plot_plotly(traj_map, stiff_map, args.stiffness_color, args.stiffness_mode, out_html, args.max_points)

if __name__ == "__main__":
    main()
