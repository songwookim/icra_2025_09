#!/usr/bin/env python3
"""Wrapper script to launch stiffness_force_control.launch.py without invoking ros2 CLI.

Usage:
  python3 launch_stiffness_force_control.py model_type:=bc rate_hz:=100.0 stiffness_scale:=0.8
Any key:=value tokens are forwarded as launch substitutions.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path
from launch import LaunchService
import importlib.util


def _ensure_ros_env(ws_root: Path):
    """Ensure minimal ROS2 environment variables so launch can locate packages when debug session didn't source setup.bash.

    We try to augment AMENT_PREFIX_PATH, COLCON_PREFIX_PATH, ROS_PACKAGE_PATH with the workspace install prefix.
    This is a best-effort fallback; sourcing setup.bash is still recommended for full env (typesupport, rmw, etc.).
    """
    install_root = ws_root / 'install'
    if not install_root.exists():
        print(f"[warn] install/ not found at {install_root}; build may be required (colcon build)")
        return

    # Add /opt/ros/humble to prefix paths if not already present
    humble_path = "/opt/ros/humble"
    pkg_install = install_root / 'hri_falcon_robot_bridge'
    
    # Prefix paths - must include individual package install directories
    for var in ["AMENT_PREFIX_PATH", "COLCON_PREFIX_PATH"]:
        existing = os.environ.get(var, "")
        paths = [str(pkg_install), str(install_root), humble_path] + ([p for p in existing.split(":") if p and p not in [str(pkg_install), str(install_root), humble_path]])
        os.environ[var] = ":".join(paths)

    # ROS_PACKAGE_PATH needs share subdirs for each package; add install root and its share/<pkg>
    share_pkg = install_root / 'share' / 'hri_falcon_robot_bridge'
    ros_pkg_path_entries = [str(share_pkg), str(install_root / 'share'), str(install_root)]
    existing_rpp = os.environ.get("ROS_PACKAGE_PATH", "")
    if existing_rpp:
        for p in existing_rpp.split(":"):
            if p and p not in ros_pkg_path_entries:
                ros_pkg_path_entries.append(p)
    os.environ["ROS_PACKAGE_PATH"] = ":".join(ros_pkg_path_entries)

    # Informative printout
    print(f"[env] AMENT_PREFIX_PATH={os.environ['AMENT_PREFIX_PATH']}")
    print(f"[env] ROS_PACKAGE_PATH={os.environ['ROS_PACKAGE_PATH']}")


def main() -> int:
    # Workspace root assumed 3 levels up from this script ( .../src/hri_falcon_robot_bridge/scripts )
    script_path = Path(__file__).resolve()
    ws_root = script_path.parents[3]
    _ensure_ros_env(ws_root)
    launch_path = ws_root / 'src' / 'hri_falcon_robot_bridge' / 'launch' / 'stiffness_force_control.launch.py'
    if not launch_path.exists():
        print(f"[error] Launch file not found: {launch_path}", file=sys.stderr)
        return 1

    # Forward argv tokens (e.g., model_type:=bc) directly; LaunchService parses them
    argv = sys.argv[1:]
    # Export artifact_dir to env as workaround if launch argument substitution fails
    for tok in argv:
        if tok.startswith('artifact_dir:='):
            val = tok.split(':=', 1)[1]
            os.environ['POLICY_ARTIFACT_DIR'] = val
            print(f"[env] POLICY_ARTIFACT_DIR={val}")
    print(f"[info] Launching stiffness pipeline via wrapper. Args: {argv}")

    # Use LaunchDescriptionSource get_launch_description directly
    # Dynamically import launch file and call generate_launch_description()
    spec = importlib.util.spec_from_file_location("stiffness_force_control_launch", str(launch_path))
    if spec is None or spec.loader is None:
        print("[error] Failed to load launch file spec", file=sys.stderr)
        return 1
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    if not hasattr(mod, "generate_launch_description"):
        print("[error] Launch file missing generate_launch_description()", file=sys.stderr)
        return 1
    launch_description = mod.generate_launch_description()
    ls = LaunchService(argv=argv)
    ls.include_launch_description(launch_description)
    return ls.run()

if __name__ == '__main__':
    raise SystemExit(main())
