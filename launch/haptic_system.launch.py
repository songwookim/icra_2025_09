#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Package directory
    pkg_dir = get_package_share_directory('hri_falcon_robot_bridge')
    
    # Launch arguments
    use_test_force = DeclareLaunchArgument(
        'use_test_force',
        default_value='true',
        description='Use continuous force test node instead of real force sensor'
    )
    
    # Set libnifalcon library path
    libnifalcon_path = "/home/songwoo/Desktop/work_dir/libnifalcon/build/lib"
    env_vars = {'LD_LIBRARY_PATH': f"{libnifalcon_path}:{os.environ.get('LD_LIBRARY_PATH', '')}"}
    
    # Falcon hardware node
    falcon_node = Node(
        package='hri_falcon_robot_bridge',
        executable='falcon_node',
        name='falcon_node',
        output='screen',
        additional_env=env_vars,
        parameters=[{
            'use_libnifalcon': True,
            'device_index': 0
        }]
    )
    
    
    # Continuous Force Test Node (instead of real force sensor)
    force_test_node = Node(
        package='hri_falcon_robot_bridge',
        executable='continuous_force_test_node.py',
        name='continuous_force_test_node',
        output='screen',
        condition=IfCondition(LaunchConfiguration('use_test_force'))
    )
    
    return LaunchDescription([
        use_test_force,
        
        # Start falcon node first
        falcon_node,
        
        # Start force test with additional delay
        TimerAction(
            period=5.0,
            actions=[force_test_node]
        ),
    ])
