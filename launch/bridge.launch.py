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

    # Declare launch arguments for force_sensor_node parameters
    force_publish_rate = DeclareLaunchArgument('force_publish_rate_hz', default_value='200.0')
    force_use_mock = DeclareLaunchArgument('force_use_mock', default_value='true')
    force_config_path = DeclareLaunchArgument('force_config_path', default_value='config.yaml')
    force_num_sensors = DeclareLaunchArgument('force_num_sensors', default_value='3')
    force_use_temp = DeclareLaunchArgument('force_use_temp_controller', default_value='false')

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
        force_publish_rate,
        force_use_mock,
        force_config_path,
        force_num_sensors,
        force_use_temp,
        # Node 1: Force sensor publisher
        Node(
            package='hri_falcon_robot_bridge',
            executable='force_sensor_node.py',
            name='force_sensor_node',
            output='screen',
            parameters=[
                {'publish_rate_hz': LaunchConfiguration('force_publish_rate_hz')},
                {'use_mock': LaunchConfiguration('force_use_mock')},
                {'config_path': LaunchConfiguration('force_config_path')},
                {'num_sensors': LaunchConfiguration('force_num_sensors')},
                {'use_temp_controller': LaunchConfiguration('force_use_temp_controller')},
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
    ])
