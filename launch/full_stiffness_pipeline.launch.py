#!/usr/bin/env python3
"""Full pipeline launch: policy + impedance + robot controller.
Modified for stability: Added stiffness filtering parameters.
"""

import os
from pathlib import Path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, SetEnvironmentVariable, GroupAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    args = [
        DeclareLaunchArgument('model_type', default_value='diffusion_t_ddim'),  # Best: R²=0.96
        # DeclareLaunchArgument('model_type', default_value='bc'),
        DeclareLaunchArgument('artifact_dir', default_value='/home/songwoo/ros2_ws/icra2025/src/hri_falcon_robot_bridge/outputs/models/policy_learning_unified/artifacts/20251130_063538'),  # Latest with diffusion_t
        DeclareLaunchArgument('rate_hz', default_value='50.0'),
        # DeclareLaunchArgument('stiffness_scale', default_value='1.5'), # GOOD
        DeclareLaunchArgument('stiffness_scale', default_value='0.11'),  # Full scale (use with tau_scale=0.5 in demo_ee_player)
        DeclareLaunchArgument('run_mujoco', default_value='true'),
        DeclareLaunchArgument('manual_start', default_value='false'),
        DeclareLaunchArgument('start_key', default_value='p'),
        DeclareLaunchArgument('rc_use_force_control', default_value='true'),
        DeclareLaunchArgument('rc_safe_mode', default_value='false'),  # Set to true for zero torque output
        DeclareLaunchArgument('enable_force_feedback', default_value='false'),
        DeclareLaunchArgument('kp_force', default_value='0.3'),
        DeclareLaunchArgument('ki_force', default_value='0.1'),
        DeclareLaunchArgument('max_torque', default_value='100.'),  # [SAFETY] Reduced from 500
        DeclareLaunchArgument('max_current_units_pos', default_value='1'),
        DeclareLaunchArgument('max_current_units_neg', default_value='200'),  # Allow up to -200 current units for grasping
        DeclareLaunchArgument('max_pwm_limit', default_value='200'),  # Full PWM range (100%)
        DeclareLaunchArgument('position_error_threshold', default_value='500.'),
        DeclareLaunchArgument('damping_ratio', default_value='0.5'),
        DeclareLaunchArgument('virtual_mass', default_value='0.1'),
        
        # --- [수정] 필터 파라미터 ---
        DeclareLaunchArgument('torque_filter_alpha', default_value='0.3'),      # 토크 스무딩 (0.3 = 30% 새값)
        DeclareLaunchArgument('stiffness_filter_alpha', default_value='0.5'),   # Stiffness 스무딩 (0.5 = 50% 새값, 더 빠른 반응)
        DeclareLaunchArgument('max_stiffness_change', default_value='100.0'),   # K값 급발진 방지 (Rate Limit, 기존 50 -> 100)
        DeclareLaunchArgument('smooth_window', default_value='5'),
        
        # --- [추가] 시간 기반 stiffness 스케일링 (DISABLED: 처음부터 100%) ---
        DeclareLaunchArgument('time_ramp_duration', default_value='0.0'),  # 0 = ramp 비활성화 (즉시 100%)
        DeclareLaunchArgument('initial_stiffness_scale', default_value='1.0'),  # 처음부터 100%
        DeclareLaunchArgument('final_stiffness_scale', default_value='1.0'),  # 최종 100%
        # -----------------------------
        
        # DeclareLaunchArgument('current_units_scale', default_value='[2.5, 7.5, 7.5, 2.5, 11.5, 9.5, 2.5, 15.5, 7.5]'),
        # DeclareLaunchArgument('current_units_scale', default_value='[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]'),
        DeclareLaunchArgument('current_units_scale', default_value='[1.,1.25,1.,1.,1.,1.0,1.,1.,0.9]'),
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
    smooth_window = LaunchConfiguration('smooth_window')
    
    # [추가] 파라미터 바인딩
    stiffness_filter_alpha = LaunchConfiguration('stiffness_filter_alpha')
    max_stiffness_change = LaunchConfiguration('max_stiffness_change')
    max_pwm_limit = LaunchConfiguration('max_pwm_limit')
    
    # [추가] 시간 기반 스케일링 파라미터
    time_ramp_duration = LaunchConfiguration('time_ramp_duration')
    initial_stiffness_scale = LaunchConfiguration('initial_stiffness_scale')
    final_stiffness_scale = LaunchConfiguration('final_stiffness_scale')
    
    current_units_scale = LaunchConfiguration('current_units_scale')

    # PYTHONPATH Setup
    launch_file_path = Path(__file__).resolve()
    ws_root = launch_file_path.parents[3]
    src_pkg_inner = str(ws_root / 'src' / 'hri_falcon_robot_bridge' / 'hri_falcon_robot_bridge')
    src_scripts = str(ws_root / 'src' / 'hri_falcon_robot_bridge' / 'scripts' / '3_model_learning')
    existing_pythonpath = os.environ.get('PYTHONPATH', '')
    paths_to_add = [src_pkg_inner, src_scripts]
    new_paths = [p for p in paths_to_add if p not in existing_pythonpath.split(':')]
    if existing_pythonpath:
        new_paths.append(existing_pythonpath)
    new_pythonpath = ':'.join(new_paths)
    
    set_pythonpath = SetEnvironmentVariable(name='PYTHONPATH', value=new_pythonpath)
    set_artifact_env = SetEnvironmentVariable(name='POLICY_ARTIFACT_DIR', value=artifact_dir)

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
            'smooth_window': smooth_window,
            # 시간 기반 스케일링 파라미터
            'time_ramp_duration': time_ramp_duration,
            'initial_stiffness_scale': initial_stiffness_scale,
            'final_stiffness_scale': final_stiffness_scale,
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
            
            # [추가] 제어기 측 필터 파라미터 전달
            'stiffness_filter_alpha': stiffness_filter_alpha,
            'max_stiffness_change': max_stiffness_change,
            'max_pwm_limit': max_pwm_limit,  # For PWM plot visualization
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
            'max_current_units_pos': max_current_units_pos,
            'max_current_units_neg': max_current_units_neg,
            'max_pwm_limit': max_pwm_limit,  # [SAFETY] PWM limit
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