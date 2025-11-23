#!/usr/bin/env python3
"""Launch file: stiffness + (torque) impedance + robot controller pipeline.

Nodes:
  - run_policy_node (stiffness prediction)
  - torque_impedance_controller_node (Cartesian impedance -> joint torque & current)
  - robot_controller_node (current passthrough to Dynamixel)

Arguments (launch):
  model_type (str)                : 정책 모델 타입 (bc|diffusion_c|diffusion_t|gmm|gmr)
  rate_hz (double)               : 제어 루프 주파수 (기본 100.0)
  stiffness_scale (double)       : 정책 출력 전체 스케일
  enable_force_feedback (bool)   : 토크 노드 힘 피드백 활성 여부
  force_axis (str)               : 단일 힘 축 (x|y|z)
  kp_force (double)              : 힘 오차 P 게인
  ki_force (double)              : 힘 오차 I 게인
  max_torque (double)            : 토크 클램프
  max_current_units (int)        : 토크 노드 전류 클램프
  rc_use_force_control (bool)    : 로봇 컨트롤러가 토크→전류 변환 활성화
  rc_safe_mode (bool)            : 로봇 컨트롤러 안전 모드 (true면 실제 쓰기 안함)
    manual_start (bool)            : 키보드 입력으로 정책 시작 대기 (true면 'p'로 시작)
    start_key (str)                : 시작 키 (기본 'p')

Usage example:
  ros2 launch hri_falcon_robot_bridge stiffness_force_control.launch.py \
    model_type:=bc rate_hz:=100.0 stiffness_scale:=0.8 \
    enable_force_feedback:=false rc_use_force_control:=true rc_safe_mode:=false
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Launch arguments
    args = [
        DeclareLaunchArgument('model_type', default_value='bc'),
        DeclareLaunchArgument('rate_hz', default_value='100.0'),
        DeclareLaunchArgument('stiffness_scale', default_value='1.0'),
        DeclareLaunchArgument('artifact_dir', default_value=''),
        DeclareLaunchArgument('enable_force_feedback', default_value='false'),
        DeclareLaunchArgument('force_axis', default_value='z'),
        DeclareLaunchArgument('kp_force', default_value='0.4'),
        DeclareLaunchArgument('ki_force', default_value='0.0'),
        DeclareLaunchArgument('max_torque', default_value='1.2'),
        DeclareLaunchArgument('max_current_units', default_value='250'),
        DeclareLaunchArgument('rc_use_force_control', default_value='true'),
        DeclareLaunchArgument('rc_safe_mode', default_value='true'),
        DeclareLaunchArgument('force_sensor_rate_hz', default_value='500.0'),
        DeclareLaunchArgument('force_sensor_use_mock', default_value='false'),
        DeclareLaunchArgument('manual_start', default_value='true'),
        DeclareLaunchArgument('start_key', default_value='p'),
    ]

    # Substitutions
    model_type = LaunchConfiguration('model_type')
    rate_hz = LaunchConfiguration('rate_hz')
    stiffness_scale = LaunchConfiguration('stiffness_scale')
    enable_force_feedback = LaunchConfiguration('enable_force_feedback')
    artifact_dir = LaunchConfiguration('artifact_dir')
    force_axis = LaunchConfiguration('force_axis')
    kp_force = LaunchConfiguration('kp_force')
    ki_force = LaunchConfiguration('ki_force')
    max_torque = LaunchConfiguration('max_torque')
    max_current_units = LaunchConfiguration('max_current_units')
    rc_use_force_control = LaunchConfiguration('rc_use_force_control')
    rc_safe_mode = LaunchConfiguration('rc_safe_mode')
    manual_start = LaunchConfiguration('manual_start')
    start_key = LaunchConfiguration('start_key')

    # Nodes
    policy_node = Node(
        package='hri_falcon_robot_bridge',
        executable='run_policy_node',
        name='run_policy_node',
        output='screen',
        parameters=[{
            'model_type': model_type,
            'rate_hz': rate_hz,
            'stiffness_scale': stiffness_scale,
            'smooth_window': 5,
            'artifact_dir': artifact_dir,
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
            'force_axis': force_axis,
            'kp_force': kp_force,
            'ki_force': ki_force,
            'max_torque': max_torque,
            'max_current_units': max_current_units,
            'stiffness_scale': stiffness_scale,
            'use_mujoco': True,
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
            'max_current_units_fc': max_current_units,
            'torque_topic': '/impedance_control/computed_torques',
            'current_topic': '/dynamixel/goal_current',
        }]
    )

    # Full pipeline: include sensor, deformity, and pose bridge nodes
    # Debug: log artifact_dir resolution
    log_artifact = LogInfo(msg=['[launch] artifact_dir arg = ', artifact_dir])

    return LaunchDescription(args + [
        log_artifact,
        policy_node,
        torque_node,
        robot_node,
    ])
