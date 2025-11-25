#!/usr/bin/env python3
"""Wrapper script to launch full_stiffness_pipeline.launch.py without ros2 CLI.

Usage:
  python3 launch_full_stiffness_pipeline.py model_type:=bc allow_mock_missing:=true artifact_dir:=/path/to/artifacts

Any key:=value tokens are forwarded to LaunchService. If an artifact_dir:= token is present it is also exported as
POLICY_ARTIFACT_DIR for nodes that rely on the environment variable.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path
from launch import LaunchService
import importlib.util


def _ensure_ros_env(ws_root: Path):
    """Minimal ROS2 env augmentation - use src files directly for live development."""
    install_root = ws_root / 'install'
    src_pkg_root = ws_root / 'src' / 'hri_falcon_robot_bridge'
    
    # Add src package to PYTHONPATH (최우선 - 빌드 없이 직접 실행)
    # [FIX] dynamixel_control 등 하위 모듈도 import 가능하도록 내부 패키지 경로 추가
    import sys
    src_pkg_inner = str(src_pkg_root / 'hri_falcon_robot_bridge')
    if src_pkg_inner not in sys.path:
        sys.path.insert(0, src_pkg_inner)
        print(f"[env] PYTHONPATH prepended: {src_pkg_inner}")
    
    src_pkg_str = str(src_pkg_root)
    if src_pkg_str not in sys.path:
        sys.path.insert(0, src_pkg_str)
        print(f"[env] PYTHONPATH prepended: {src_pkg_str}")
    
    if not install_root.exists():
        print(f"[warn] install/ not found at {install_root}; using src directly")
        return

    humble_path = "/opt/ros/humble"
    pkg_install = install_root / 'hri_falcon_robot_bridge'

    for var in ["AMENT_PREFIX_PATH", "COLCON_PREFIX_PATH"]:
        existing = os.environ.get(var, "")
        paths = [str(pkg_install), str(install_root), humble_path] + [p for p in existing.split(":") if p and p not in [str(pkg_install), str(install_root), humble_path]]
        os.environ[var] = ":".join(paths)

    share_pkg = install_root / 'share' / 'hri_falcon_robot_bridge'
    ros_pkg_entries = [str(share_pkg), str(install_root / 'share'), str(install_root)]
    existing_rpp = os.environ.get("ROS_PACKAGE_PATH", "")
    if existing_rpp:
        for p in existing_rpp.split(":"):
            if p and p not in ros_pkg_entries:
                ros_pkg_entries.append(p)
    os.environ["ROS_PACKAGE_PATH"] = ":".join(ros_pkg_entries)
    print(f"[env] AMENT_PREFIX_PATH={os.environ['AMENT_PREFIX_PATH']}")
    print(f"[env] ROS_PACKAGE_PATH={os.environ['ROS_PACKAGE_PATH']}")


def main() -> int:
    script_path = Path(__file__).resolve()
    ws_root = script_path.parents[3]
    _ensure_ros_env(ws_root)
    launch_path = ws_root / 'src' / 'hri_falcon_robot_bridge' / 'launch' / 'full_stiffness_pipeline.launch.py'
    if not launch_path.exists():
        print(f"[error] Launch file not found: {launch_path}", file=sys.stderr)
        return 1

    argv = sys.argv[1:]
    for tok in argv:
        if tok.startswith('artifact_dir:='):
            val = tok.split(':=', 1)[1]
            os.environ['POLICY_ARTIFACT_DIR'] = val
            print(f"[env] POLICY_ARTIFACT_DIR={val}")

    print(f"[info] Launching full stiffness pipeline. Args: {argv}")

    spec = importlib.util.spec_from_file_location("full_stiffness_pipeline_launch", str(launch_path))
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
