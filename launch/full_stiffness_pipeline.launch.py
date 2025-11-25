#!/usr/bin/env python3
"""Full pipeline launch: policy + impedance + robot controller.

Adds robustness options:
    - allow_mock_missing: Run policy with zero-filled obs after timeout if sensors absent
    - mock_start_timeout_sec: Timeout before mock activation
    - run_mujoco: Flag passed to impedance controller (not auto-starting sense glove here)
    - start_key/manual_start for policy gating

Launch Arguments:
  model_type (str)                : bc|diffusion_c|diffusion_t|gmm|gmr|ibc
  artifact_dir (str)              : model artifact directory
  rate_hz (double)               : control loop / policy rate
  stiffness_scale (double)       : stiffness global scale
  allow_mock_missing (bool)      : enable mock observation fallback
  mock_start_timeout_sec (double): timeout seconds before mock activation
    run_mujoco (bool)              : pass-through to torque_impedance_controller_node
  manual_start (bool)            : require key press (start_key) to start policy
  start_key (str)                : start key char
  rc_use_force_control (bool)    : robot controller torque->current path
  rc_safe_mode (bool)            : robot controller safe mode
    enable_force_feedback (bool)   : impedance force feedback loop
  kp_force, ki_force (double)    : feedback gains
  max_torque (double)            : torque clamp
  max_current_units (int)        : current clamp (impedance & robot)

Example:
    ros2 launch hri_falcon_robot_bridge full_stiffness_pipeline.launch.py \
        model_type:=bc artifact_dir:=/path/to/artifacts rate_hz:=100.0 \
        allow_mock_missing:=true mock_start_timeout_sec:=2.0 \
        run_mujoco:=false rc_use_force_control:=true rc_safe_mode:=true stiffness_scale:=0.8
"""

import os
from pathlib import Path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, SetEnvironmentVariable, GroupAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    args = [
        DeclareLaunchArgument('model_type', default_value='bc'),
        DeclareLaunchArgument('artifact_dir', default_value='/home/songwoo/ros2_ws/icra2025/src/hri_falcon_robot_bridge/outputs/models/policy_learning_unified/artifacts/20251124_033503'),
        # DeclareLaunchArgument('artifact_dir', default_value='/home/songwoo/ros2_ws/icra2025/src/hri_falcon_robot_bridge/outputs/models/policy_learning_unified/artifacts/20251124_183103'),
        DeclareLaunchArgument('rate_hz', default_value='100.0'),
        DeclareLaunchArgument('stiffness_scale', default_value='1.0'),
        DeclareLaunchArgument('run_mujoco', default_value='true'),
        DeclareLaunchArgument('manual_start', default_value='false'),  # Auto-start policy
        DeclareLaunchArgument('start_key', default_value='p'),
        DeclareLaunchArgument('rc_use_force_control', default_value='true'),
        DeclareLaunchArgument('rc_safe_mode', default_value='false'),
        DeclareLaunchArgument('enable_force_feedback', default_value='false'),
        DeclareLaunchArgument('kp_force', default_value='0.4'),
        DeclareLaunchArgument('ki_force', default_value='0.0'),
        DeclareLaunchArgument('max_torque', default_value='20.'),
        DeclareLaunchArgument('max_current_units', default_value='15'),
        DeclareLaunchArgument('max_current_units_pos', default_value='5'),
        DeclareLaunchArgument('max_current_units_neg', default_value='30'),
        DeclareLaunchArgument('position_error_threshold', default_value='0.5'),  # 50mm default
        DeclareLaunchArgument('damping_ratio', default_value='2.'),
        DeclareLaunchArgument('virtual_mass', default_value='0.2'),
        DeclareLaunchArgument('torque_filter_alpha', default_value='0.5'),
        DeclareLaunchArgument('current_units_scale', default_value='[2.5, 3.5, 6.5, 2.5, 3.5, 9.5, 2.5, 3.5, 9.5]'),
    ]

    # Config substitutions
    model_type = LaunchConfiguration('model_type')
    artifact_dir = LaunchConfiguration('artifact_dir')
    rate_hz = LaunchConfiguration('rate_hz')
    stiffness_scale = LaunchConfiguration('stiffness_scale')
    run_mujoco = LaunchConfiguration('run_mujoco')
    manual_start = LaunchConfiguration('manual_start')
    start_key = LaunchConfiguration('start_key')
    rc_use_force_control = LaunchConfiguration('rc_use_force_control')
    rc_safe_mode = LaunchConfiguration('rc_safe_mode')
    enable_force_feedback = LaunchConfiguration('enable_force_feedback')
    kp_force = LaunchConfiguration('kp_force')
    ki_force = LaunchConfiguration('ki_force')
    max_torque = LaunchConfiguration('max_torque')
    max_current_units_pos = LaunchConfiguration('max_current_units_pos')
    max_current_units_neg = LaunchConfiguration('max_current_units_neg')
    position_error_threshold = LaunchConfiguration('position_error_threshold')
    damping_ratio = LaunchConfiguration('damping_ratio')
    virtual_mass = LaunchConfiguration('virtual_mass')
    torque_filter_alpha = LaunchConfiguration('torque_filter_alpha')
    current_units_scale = LaunchConfiguration('current_units_scale')

    # Ensure source package is in PYTHONPATH for dynamixel_control import
    launch_file_path = Path(__file__).resolve()
    ws_root = launch_file_path.parents[3]  # launch file is in src/pkg/launch/
    src_pkg_inner = str(ws_root / 'src' / 'hri_falcon_robot_bridge' / 'hri_falcon_robot_bridge')
    src_scripts = str(ws_root / 'src' / 'hri_falcon_robot_bridge' / 'scripts' / '3_model_learning')
    existing_pythonpath = os.environ.get('PYTHONPATH', '')
    # Add both package and scripts directories
    paths_to_add = [src_pkg_inner, src_scripts]
    new_paths = [p for p in paths_to_add if p not in existing_pythonpath.split(':')]
    if existing_pythonpath:
        new_paths.append(existing_pythonpath)
    new_pythonpath = ':'.join(new_paths)
    
    set_pythonpath = SetEnvironmentVariable(
        name='PYTHONPATH', value=new_pythonpath
    )

    # Environment export for artifact_dir (node already checks POLICY_ARTIFACT_DIR)
    set_artifact_env = SetEnvironmentVariable(
        name='POLICY_ARTIFACT_DIR', value=artifact_dir
    )

    # (Sensor / glove / deformity nodes intentionally omitted; user starts manually.)

    # Policy node
    policy_node = Node(
        package='hri_falcon_robot_bridge',
        executable='run_policy_node',
        name='run_policy_node',
        output='screen',
        parameters=[{
            'model_type': model_type,
            'artifact_dir': artifact_dir,
            'rate_hz': rate_hz,
            'stiffness_scale': stiffness_scale,
            'manual_start': manual_start,
            'start_key': start_key,
        }]
    )

    torque_node = Node(
        package='hri_falcon_robot_bridge',
        executable='torque_impedance_controller_node',
        name='torque_impedance_controller_node',
        output='screen',
        parameters=[{
            'rate_hz': rate_hz,
            'enable_force_feedback': enable_force_feedback,
            'kp_force': kp_force,
            'ki_force': ki_force,
            'max_torque': max_torque,
            'max_current_units_pos': max_current_units_pos,
            'max_current_units_neg': max_current_units_neg,
            'stiffness_scale': stiffness_scale,
            'use_mujoco': run_mujoco,
            'position_error_threshold': position_error_threshold,
            'damping_ratio': damping_ratio,
            'virtual_mass': virtual_mass,
            'torque_filter_alpha': torque_filter_alpha,
            'current_units_scale': current_units_scale,
        }]
    )

    robot_node = Node(
        package='hri_falcon_robot_bridge',
        executable='robot_controller_node',
        name='robot_controller_node',
        output='screen',
        parameters=[{
            'use_force_control': rc_use_force_control,
            'safe_mode': rc_safe_mode,
            'force_control_mode': 0,
            'torque_topic': '/impedance_control/computed_torques',
            'current_topic': '/dynamixel/goal_current',
            'defer_force_control_until_policy': True,
        }]
    )

    log_banner = LogInfo(msg=['[full_pipeline] model_type=', model_type, ' artifact_dir=', artifact_dir])

    group = GroupAction([
        set_pythonpath,
        set_artifact_env,
        policy_node,
        torque_node,
        robot_node,
    ])

    return LaunchDescription(args + [log_banner, group])
