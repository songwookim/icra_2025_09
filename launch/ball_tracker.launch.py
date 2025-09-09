from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    width = LaunchConfiguration('width')
    height = LaunchConfiguration('height')
    fps = LaunchConfiguration('fps')
    device_index = LaunchConfiguration('device_index')
    serial_number = LaunchConfiguration('serial_number')
    enable_cv_window = LaunchConfiguration('enable_cv_window')

    return LaunchDescription([
        DeclareLaunchArgument('width', default_value='640', description='Color stream width'),
        DeclareLaunchArgument('height', default_value='480', description='Color stream height'),
        DeclareLaunchArgument('fps', default_value='30', description='Color stream FPS'),
        DeclareLaunchArgument('device_index', default_value='-1', description='Device index (0,1,...) or -1 for auto'),
        DeclareLaunchArgument('serial_number', default_value='', description='Specific RealSense serial (overrides index)'),
        DeclareLaunchArgument('enable_cv_window', default_value='true', description='Enable OpenCV GUI window'),

        Node(
            package='hri_falcon_robot_bridge',
            executable='ball_tracker_node',
            name='ball_tracker',
            output='screen',
            parameters=[{
                'width': width,
                'height': height,
                'fps': fps,
                'device_index': device_index,
                'serial_number': serial_number,
                'enable_cv_window': enable_cv_window,
            }]
        )
    ])
