#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
	# Launch args
	run_tracker = DeclareLaunchArgument(
		'run_tracker', default_value='true',
		description='Start hand_tracker_node together with the controller')
	test_mode = DeclareLaunchArgument(
		'test_mode', default_value='true',
		description='If true, controller runs in dry-run logging mode')
	enable_cv = DeclareLaunchArgument(
		'enable_cv_window', default_value='true',
		description='Show OpenCV window (and request GUI env)')
	units_pub_en = DeclareLaunchArgument(
		'units_publish_enabled', default_value='true',
		description='Enable publishing targets_units to the robot')
	run_mj = DeclareLaunchArgument(
		'run_mujoco', default_value='true',
		description='Start MuJoCo passive viewer')
	mj_model = DeclareLaunchArgument(
		'mujoco_model_path',
		default_value='/home/songwoo/Desktop/work_dir/realsense_hand_retargetting/universal_robots_ur5e_with_dclaw/dclaw/dclaw3xh.xml',
		description='Path to MuJoCo DClaw model xml')

	controller = Node(
		package='hri_falcon_robot_bridge',
		executable='robot_controller_node',
		name='robot_controller_node',
		output='screen',
		parameters=[
			{'hand_units_topic': '/hand_tracker/targets_units'},
			{'hand_enable_topic': '/hand_tracker/enable'},
			{'keyboard_toggle_enable': True},
			{'keyboard_toggle_key': 'h'},
			{'test_mode': LaunchConfiguration('test_mode')},
		],
	)

	tracker = Node(
		package='hri_falcon_robot_bridge',
		executable='hand_tracker_node',
		name='hand_tracker_node',
		output='screen',
		condition=IfCondition(LaunchConfiguration('run_tracker')),
		# GUI를 확실히 띄우기 위해 환경변수 ENABLE_CV_WINDOW 전달 (코드가 이를 검사)
		additional_env={'ENABLE_CV_WINDOW': LaunchConfiguration('enable_cv_window')},
		parameters=[
			{'enable_cv_window': LaunchConfiguration('enable_cv_window')},
			{'units_publish_enabled': LaunchConfiguration('units_publish_enabled')},
			{'run_mujoco': LaunchConfiguration('run_mujoco')},
			{'mujoco_model_path': LaunchConfiguration('mujoco_model_path')},
		],
	)

	group = GroupAction(actions=[controller, tracker])

	return LaunchDescription([
		run_tracker,
		test_mode,
		enable_cv,
		units_pub_en,
		run_mj,
		mj_model,
		group,
	])
