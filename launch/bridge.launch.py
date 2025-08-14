from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Declare launch arguments for falcon_node parameters
    falcon_force_scale = DeclareLaunchArgument('falcon_force_scale', default_value='1500.0')
    falcon_publish_rate = DeclareLaunchArgument('falcon_publish_rate_hz', default_value='200.0')
    falcon_frame_id = DeclareLaunchArgument('falcon_frame_id', default_value='falcon_base')
    falcon_force_sensor_index = DeclareLaunchArgument('falcon_force_sensor_index', default_value='0')

    falcon_init_enable = DeclareLaunchArgument('falcon_init_posture_enable', default_value='true')
    falcon_init_target = DeclareLaunchArgument('falcon_init_enc_target', default_value='[-500,-500,-500]')
    falcon_init_kp = DeclareLaunchArgument('falcon_init_kp', default_value='100.0')
    falcon_init_kd = DeclareLaunchArgument('falcon_init_kd', default_value='0.1')
    falcon_init_force_limit = DeclareLaunchArgument('falcon_init_force_limit', default_value='1000')
    falcon_init_max_loops = DeclareLaunchArgument('falcon_init_max_loops', default_value='20000')
    falcon_init_stable_eps = DeclareLaunchArgument('falcon_init_stable_eps', default_value='5')
    falcon_init_stable_count = DeclareLaunchArgument('falcon_init_stable_count', default_value='0')

    return LaunchDescription([
        falcon_force_scale,
        falcon_publish_rate,
        falcon_frame_id,
        falcon_force_sensor_index,
        falcon_init_enable,
        falcon_init_target,
        falcon_init_kp,
        falcon_init_kd,
        falcon_init_force_limit,
        falcon_init_max_loops,
        falcon_init_stable_eps,
        falcon_init_stable_count,
        # Node 1: Force sensor publisher
        Node(
            package='hri_falcon_robot_bridge',
            executable='force_sensor_node.py',
            name='force_sensor_node',
            output='screen',
            parameters=[
                {'publish_rate_hz': 200.0},
                {'use_mock': True},                 # set False when real controller is available
                {'config_path': 'config.yaml'},
            ],
        ),
        # Node 3: Falcon bridge (C++)
        Node(
            package='hri_falcon_robot_bridge',
            executable='falcon_node',
            name='falcon_node',
            output='screen',
            parameters=[
                {'force_scale': LaunchConfiguration('falcon_force_scale')},            # N -> int units
                {'publish_rate_hz': LaunchConfiguration('falcon_publish_rate_hz')},
                {'frame_id': LaunchConfiguration('falcon_frame_id')},
                {'force_sensor_index': LaunchConfiguration('falcon_force_sensor_index')},
                # Initial posture PD
                {'init_posture_enable': LaunchConfiguration('falcon_init_posture_enable')},
                {'init_enc_target': LaunchConfiguration('falcon_init_enc_target')},
                {'init_kp': LaunchConfiguration('falcon_init_kp')},
                {'init_kd': LaunchConfiguration('falcon_init_kd')},
                {'init_force_limit': LaunchConfiguration('falcon_init_force_limit')},
                {'init_max_loops': LaunchConfiguration('falcon_init_max_loops')},
                {'init_stable_eps': LaunchConfiguration('falcon_init_stable_eps')},
                {'init_stable_count': LaunchConfiguration('falcon_init_stable_count')},
            ],
        ),
        # Node 2: Robot controller (Python)
        Node(
            package='hri_falcon_robot_bridge',
            executable='robot_controller_node.py',
            name='robot_controller_node',
            output='screen',
            parameters=[
                {'ids': [10, 20, 30]},
                {'mode': 3},
                {'scale': [1.0, 1.0, 1.0]},        # tune mapping here
                {'offset': [1000, 1000, 1000]},
                {'clip_min': [0, 0, 0]},
                {'clip_max': [4095, 4095, 4095]},
                {'use_encoders': True},            # True: use encoders; False: use position
            ],
        ),
    ])
