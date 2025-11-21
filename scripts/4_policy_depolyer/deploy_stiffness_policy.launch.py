#!/usr/bin/env python3
"""
Launch file for Stiffness Policy Deployer

Example usage:
  # Launch with Diffusion model (Unified mode)
  ros2 launch hri_falcon_robot_bridge deploy_stiffness_policy.launch.py \
    model_type:=diffusion \
    artifact_dir:=/home/songwoo/ros2_ws/icra2025/outputs/policy_learning_global_tk_unified/artifacts/20251119_192314 \
    rate_hz:=100.0

  # Launch with BC model
  ros2 launch hri_falcon_robot_bridge deploy_stiffness_policy.launch.py \
    model_type:=bc \
    artifact_dir:=/path/to/artifacts \
    rate_hz:=50.0

  # Launch with Diffusion DDIM sampler
  ros2 launch hri_falcon_robot_bridge deploy_stiffness_policy.launch.py \
    model_type:=diffusion \
    artifact_dir:=/path/to/artifacts \
    diffusion_sampler:=ddim \
    diffusion_n_samples:=5
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        # Launch arguments
        DeclareLaunchArgument(
            'model_type',
            default_value='diffusion',
            description='Model type: bc, diffusion, lstm_gmm, gmm, gmr'
        ),
        DeclareLaunchArgument(
            'artifact_dir',
            default_value='',
            description='Path to model artifacts directory (REQUIRED)'
        ),
        DeclareLaunchArgument(
            'rate_hz',
            default_value='100.0',
            description='Prediction rate in Hz'
        ),
        DeclareLaunchArgument(
            'diffusion_sampler',
            default_value='ddpm',
            description='Diffusion sampler: ddpm or ddim'
        ),
        DeclareLaunchArgument(
            'diffusion_n_samples',
            default_value='1',
            description='Number of diffusion samples (1 for mean)'
        ),
        
        # Policy deployer node
        Node(
            package='hri_falcon_robot_bridge',
            executable='stiffness_policy_deployer_node.py',
            name='stiffness_policy_deployer',
            output='screen',
            parameters=[{
                'model_type': LaunchConfiguration('model_type'),
                'artifact_dir': LaunchConfiguration('artifact_dir'),
                'rate_hz': LaunchConfiguration('rate_hz'),
                'diffusion_sampler': LaunchConfiguration('diffusion_sampler'),
                'diffusion_n_samples': LaunchConfiguration('diffusion_n_samples'),
            }],
        ),
    ])
