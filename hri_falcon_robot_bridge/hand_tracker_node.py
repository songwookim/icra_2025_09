#!/usr/bin/env python3
"""RealSense + MediaPipe hand tracker node (simple ROS2-integrated script).

This script was adapted from user code and wrapped to run inside a ROS2 package.
It opens a RealSense camera, runs MediaPipe Hands, and displays annotated frames.
It also exposes a few ROS2 parameters for resolution and debug.
"""
from __future__ import annotations

import time
from collections import deque, defaultdict
import math
import numpy as np
import os
# Qt 없는 환경에서는 offscreen, 단 디스플레이가 있으면 기본적으로 GUI 사용
_ENABLE_CV_ENV = os.environ.get("ENABLE_CV_WINDOW", "").lower() in ("1", "true", "yes", "on")
_HAS_DISPLAY = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
_WANT_GUI = _ENABLE_CV_ENV or _HAS_DISPLAY
# 디스플레이가 전혀 없고, 사용자가 ENABLE_CV_WINDOW로 강제하지 않은 경우에만 offscreen 강제
if not _WANT_GUI:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
else:
    # GUI를 원하지만 offscreen이 강제되어 있으면 해제하여 창이 뜨도록 함
    if os.environ.get("QT_QPA_PLATFORM", "").lower() == "offscreen":
        try:
            del os.environ["QT_QPA_PLATFORM"]
        except Exception:
            pass
import cv2
import rclpy
from rclpy.node import Node
from typing import Any
import csv
import pathlib
import datetime
from sensor_msgs.msg import JointState
from std_msgs.msg import Int32MultiArray
from std_msgs.msg import String
from std_msgs.msg import Bool
from geometry_msgs.msg import PoseStamped

import mediapipe as mp

import pyrealsense2 as rs

# Optional MuJoCo import for local simulation visualization
import mujoco as mj  # type: ignore
from mujoco import viewer as mj_viewer  # type: ignore

# (Optional) dynamic usage; most calls already guarded with type:ignore

# Bring in the user's helper logic (trimmed and embedded)
FINGERS = {
    "THUMB":  [ (0,1,2,"CMC"), (1,2,3,"MCP"), (2,3,4,"IP") ],
    "INDEX":  [ (0,5,6,"MCP"), (5,6,7,"PIP"), (6,7,8,"DIP") ],
    "MIDDLE": [ (0,9,10,"MCP"), (9,10,11,"PIP"), (10,11,12,"DIP") ],
}

# DClaw joint names in MuJoCo model (index by finger)
DCLAW_JOINTS = {
    "THUMB":  ["THJ30", "THJ31", "THJ32"],
    "INDEX":  ["FFJ10", "FFJ11", "FFJ12"],
    "MIDDLE": ["MFJ20", "MFJ21", "MFJ22"],
}

# RealSense 카메라 설정
W, H, FPS = 640, 480, 30

# 사용할 손가락
USE_FINGERS = ["THUMB", "INDEX", "MIDDLE"]

# 표시할 손가락 (시각화용)
VISIBLE_FINGERS = {"THUMB", "INDEX", "MIDDLE"}

# MediaPipe 손 연결 구조 정의
HAND_CONNECTIONS = [
    # 손목에서 손가락 시작점
    (0, 1), (0, 5), (0, 9), (0, 13), (0, 17),
    # 엄지
    (1, 2), (2, 3), (3, 4),
    # 검지
    (5, 6), (6, 7), (7, 8),
    # 중지
    (9, 10), (10, 11), (11, 12),
    # 약지
    (13, 14), (14, 15), (15, 16),
    # 소지
    (17, 18), (18, 19), (19, 20),
    # 손바닥 연결
    (5, 9), (9, 13), (13, 17)
]


def angle_3d(a, b, c):
    if a is None or b is None or c is None:
        return None
    ba, bc = a - b, c - b
    nba, nbc = np.linalg.norm(ba), np.linalg.norm(bc)
    if nba == 0 or nbc == 0:
        return None
    cosang = np.clip(np.dot(ba, bc) / (nba * nbc), -1.0, 1.0)
    return math.degrees(math.acos(cosang))

def angle_2d(a, b, c):
    """2D에서 ∠ABC의 0~360도 각도 계산 (부호 포함).
    a,b,c는 2D 좌표(np.array-like)."""
    if a is None or b is None or c is None:
        return None
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    ba = a - b
    bc = c - b
    nba = np.linalg.norm(ba)
    nbc = np.linalg.norm(bc)
    if nba == 0 or nbc == 0:
        return None
    ba_n = ba / nba
    bc_n = bc / nbc
    dot = float(np.clip(np.dot(ba_n, bc_n), -1.0, 1.0))
    cross_z = float(ba_n[0] * bc_n[1] - ba_n[1] * bc_n[0])
    ang = math.degrees(math.atan2(cross_z, dot))
    if ang < 0:
        ang += 360.0
    return ang

def draw_angle_arc(img, pix, i, j, k, measured_angle, radius=30, color=(0, 255, 255), thickness=2, fill_alpha=0.25):
    """측정된 관절 각도(0~360)를 반영하여 부채꼴(호) 시각화.
    - measured_angle: angle_2d(or 3d)에서 나온 0~360 실각도
    - 실제 측정각이 작은 호보다 크면 큰 호(360-small)를 그려 과신전도 표현
    """
    if measured_angle is None:
        return
    try:
        center = pix[j]
        p_i = np.array(pix[i], dtype=float)
        p_j = np.array(pix[j], dtype=float)
        p_k = np.array(pix[k], dtype=float)
        v1 = p_i - p_j
        v2 = p_k - p_j
        if np.linalg.norm(v1) < 1e-6 or np.linalg.norm(v2) < 1e-6:
            return
        v1n = v1 / np.linalg.norm(v1)
        v2n = v2 / np.linalg.norm(v2)
        a1 = math.degrees(math.atan2(v1n[1], v1n[0])) % 360
        a2 = math.degrees(math.atan2(v2n[1], v2n[0])) % 360
        # 작은 호(<=180) 크기 계산
        raw_diff = (a2 - a1) % 360
        if raw_diff <= 0:
            raw_diff += 360
        if raw_diff > 180:
            small_arc = 360 - raw_diff
            small_from = a2
            small_to = a1  # a2 -> a1 경로가 작은 호
        else:
            small_arc = raw_diff
            small_from = a1
            small_to = a2  # a1 -> a2 경로가 작은 호
        # 실제 표시할 호 길이 결정
        use_large = measured_angle > small_arc + 1.0 and measured_angle > 180
        if use_large:
            arc_len = 360 - small_arc
            # 큰 호는 작은 호의 반대 방향: small_to -> small_from
            start_angle = small_to % 360
            end_angle = (start_angle + arc_len) % 360
        else:
            arc_len = small_arc
            start_angle = small_from % 360
            end_angle = (start_angle + arc_len) % 360

        def sample_arc(cx, cy, r, start_deg, length_deg, steps):
            angs = np.linspace(start_deg, start_deg + length_deg, steps)
            pts = []
            for a in angs:
                rad = math.radians(a % 360)
                x = int(cx + r * math.cos(rad))
                y = int(cy + r * math.sin(rad))
                pts.append([x, y])
            return np.array(pts, dtype=np.int32)

        steps = max(8, int(arc_len / 6))  # 약 6° 간격
        arc_pts = sample_arc(center[0], center[1], radius, start_angle, arc_len, steps)
        for idx in range(1, len(arc_pts)):
            cv2.line(img, tuple(arc_pts[idx - 1]), tuple(arc_pts[idx]), color, thickness)
        if arc_len > 5 and fill_alpha > 0:
            poly_pts = np.vstack([center, arc_pts])
            overlay = img.copy()
            cv2.fillPoly(overlay, [poly_pts], color)
            cv2.addWeighted(overlay, fill_alpha, img, 1 - fill_alpha, 0, img)
    except Exception:
        # 비주얼만이므로 오류는 무시
        return

class HandTrackerNode(Node):
    def __init__(self):
        super().__init__('hand_tracker_node')
        # base parameters
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 480)
        self.declare_parameter('fps', 30)
        self.declare_parameter('min_detection_confidence', 0.6)
        self.declare_parameter('min_tracking_confidence', 0.6)
        self.declare_parameter('publish_joint_state', False)
        self.declare_parameter('log_joint_state', False)
        self.declare_parameter('publish_empty_when_no_hand', True)
        # GUI 창 표시 여부 (Qt 문제 회피용). False면 imshow/waitKey 사용 안 함.
        self.declare_parameter('enable_cv_window', True)
        # 로봇으로 units Publish on/off (키보드로 토글 가능)
        self.declare_parameter('units_publish_enabled', False)
        # CSV
        self.declare_parameter('log_angles_csv_enable', False)
        self.declare_parameter('log_angles_csv_path', '')
        # 시간 오버레이는 항상 표시하되 파일 저장은 하지 않음 (파라미터 제거)
        # MuJoCo options
        self.declare_parameter('run_mujoco', True)
        # 기본 경로: 사용자가 제공한 DClaw XML 경로를 기본값으로 설정
        self.declare_parameter('mujoco_model_path', '/home/songwoo/Desktop/work_dir/realsense_hand_retargetting/universal_robots_ur5e_with_dclaw/dclaw/dclaw3xh.xml')
        # Optional EE pose topic (to display or log end-effector pose)
        # 기본 동작: /ee_pose_if 로 계속 퍼블리시
        self.declare_parameter('ee_pose_topic_if', '/ee_pose_if')
        # 추가: MF / TH 각 EE 토픽 병행 퍼블리시를 위한 토픽명
        self.declare_parameter('ee_pose_topic_mf', '/ee_pose_mf')
        self.declare_parameter('ee_pose_topic_th', '/ee_pose_th')
        # EE pose publish options
        self.declare_parameter('ee_pose_publish_enabled', True)
        self.declare_parameter('ee_pose_frame_id', 'camera_color_optical_frame')
        # vision 모드일 때만 사용: index_tip | thumb_tip | pinch_midpoint
        self.declare_parameter('ee_pose_mode', 'pinch_midpoint')
        # EE pose source: 'mujoco' | 'vision' (default: mujoco)
        self.declare_parameter('ee_pose_source', 'mujoco')
        # MuJoCo에서 EE 위치를 뽑을 대상 (site 우선, 없으면 body)
        self.declare_parameter('ee_pose_mj_site', 'MFtip')
        self.declare_parameter('ee_pose_mj_body', '')
        # 추가: THtip 별도 site 지원
        self.declare_parameter('ee_pose_mj_site_th', 'THtip')

        # read params
        self.width = int(self.get_parameter('width').get_parameter_value().integer_value)
        self.height = int(self.get_parameter('height').get_parameter_value().integer_value)
        self.fps = int(self.get_parameter('fps').get_parameter_value().integer_value)
        self.min_detection_confidence = float(self.get_parameter('min_detection_confidence').get_parameter_value().double_value)
        self.min_tracking_confidence = float(self.get_parameter('min_tracking_confidence').get_parameter_value().double_value)
        self.publish_joint_state = bool(self.get_parameter('publish_joint_state').get_parameter_value().bool_value)
        self.log_joint_state = bool(self.get_parameter('log_joint_state').get_parameter_value().bool_value)
        self.publish_empty_when_no_hand = bool(self.get_parameter('publish_empty_when_no_hand').get_parameter_value().bool_value)
        self.enable_cv_window = bool(self.get_parameter('enable_cv_window').get_parameter_value().bool_value)
        self.units_publish_enabled = bool(self.get_parameter('units_publish_enabled').get_parameter_value().bool_value)
        self.log_angles_csv_enable = bool(self.get_parameter('log_angles_csv_enable').get_parameter_value().bool_value)
        self.log_angles_csv_path = str(self.get_parameter('log_angles_csv_path').get_parameter_value().string_value)
        self.run_mujoco = bool(self.get_parameter('run_mujoco').get_parameter_value().bool_value)
        self.mujoco_model_path = str(self.get_parameter('mujoco_model_path').get_parameter_value().string_value)
        self.ee_pose_topic_if = str(self.get_parameter('ee_pose_topic_if').get_parameter_value().string_value)
        self.ee_pose_topic_mf = str(self.get_parameter('ee_pose_topic_mf').get_parameter_value().string_value)
        self.ee_pose_topic_th = str(self.get_parameter('ee_pose_topic_th').get_parameter_value().string_value)
        self.ee_pose_publish_enabled = bool(self.get_parameter('ee_pose_publish_enabled').get_parameter_value().bool_value)
        self.ee_pose_frame_id = str(self.get_parameter('ee_pose_frame_id').get_parameter_value().string_value)
        self.ee_pose_mode = str(self.get_parameter('ee_pose_mode').get_parameter_value().string_value).lower().strip()
        self.ee_pose_source = str(self.get_parameter('ee_pose_source').get_parameter_value().string_value).lower().strip()
        self.ee_pose_mj_site = str(self.get_parameter('ee_pose_mj_site').get_parameter_value().string_value)
        self.ee_pose_mj_body = str(self.get_parameter('ee_pose_mj_body').get_parameter_value().string_value)
        self.ee_pose_mj_site_th = str(self.get_parameter('ee_pose_mj_site_th').get_parameter_value().string_value)
        if not self.mujoco_model_path:
            self.mujoco_model_path = '/home/songwoo/Desktop/work_dir/realsense_hand_retargetting/universal_robots_ur5e_with_dclaw/dclaw_two_finger/dclaw3xh.xml'

        # ENABLE_CV_WINDOW 환경변수가 설정되면 GUI 파라미터를 자동 활성화
        if _ENABLE_CV_ENV and not self.enable_cv_window:
            self.enable_cv_window = True
            self.get_logger().info("ENABLE_CV_WINDOW 환경변수에 따라 GUI 창을 활성화합니다.")
        # DISPLAY/WAYLAND_DISPLAY가 있으면 자동으로 GUI를 활성화 (사용자 편의)
        elif not self.enable_cv_window and (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            self.enable_cv_window = True
            self.get_logger().info("DISPLAY/WAYLAND_DISPLAY 감지됨: GUI 창을 자동 활성화합니다.")

        # 안내 로그: env/param 불일치 시 사용자에게 힌트 제공
        if self.enable_cv_window:
            qpa = os.environ.get('QT_QPA_PLATFORM', '')
            if qpa == 'offscreen' and not _ENABLE_CV_ENV:
                self.get_logger().warn("enable_cv_window=True지만 QT_QPA_PLATFORM=offscreen 상태입니다. GUI가 표시되지 않을 수 있어요. 실행 전 ENABLE_CV_WINDOW=1 또는 QT_QPA_PLATFORM=xcb로 설정하세요.")

        # publishers
        if self.publish_joint_state:
            self.joint_pub = self.create_publisher(JointState, '/hand_tracker/joint_states', 10)
        self.key_pub = self.create_publisher(String, '/hand_tracker/key', 10)
        self.units_pub = self.create_publisher(Int32MultiArray, '/hand_tracker/targets_units', 10)
        # Subscribe logger state
        self._logger_active = None  # type: ignore[assignment]
        try:
            self.create_subscription(Bool, '/data_logger/logging_active', self._on_logger_state, 10)
        except Exception:
            pass
        # EE pose publish/subscribe
        self._ee_pose = None
        self.ee_pose_pub_if = None
        self.ee_pose_pub_mf = None
        self.ee_pose_pub_th = None
        if self.ee_pose_topic_if:
            # 퍼블리시를 원하면 퍼블리셔 생성
            if self.ee_pose_publish_enabled:
                try:
                    self.ee_pose_pub_if = self.create_publisher(PoseStamped, self.ee_pose_topic_if, 10)
                    self.get_logger().info(f"EE Pose 퍼블리시: {self.ee_pose_topic_if} (source={self.ee_pose_source})")
                except Exception as e:
                    self.get_logger().warn(f"EE pose publisher 생성 실패: {e}")
            # 동시에 외부 퍼블리셔도 볼 수 있도록 구독 유지(오버레이 표시용)
            try:
                self.create_subscription(PoseStamped, self.ee_pose_topic_if, self._on_ee_pose, 10)
            except Exception as e:
                self.get_logger().warn(f"EE pose subscribe 실패: {e}")
        # Dual publishers for MF / TH (optional)
        try:
            if self.ee_pose_publish_enabled and self.ee_pose_topic_mf:
                self.ee_pose_pub_mf = self.create_publisher(PoseStamped, self.ee_pose_topic_mf, 10)
                self.get_logger().info(f"EE Pose 퍼블리시(MF): {self.ee_pose_topic_mf}")
        except Exception as e:
            self.get_logger().warn(f"EE pose(MF) publisher 생성 실패: {e}")
        try:
            if self.ee_pose_publish_enabled and self.ee_pose_topic_th:
                self.ee_pose_pub_th = self.create_publisher(PoseStamped, self.ee_pose_topic_th, 10)
                self.get_logger().info(f"EE Pose 퍼블리시(TH): {self.ee_pose_topic_th}")
        except Exception as e:
            self.get_logger().warn(f"EE pose(TH) publisher 생성 실패: {e}")
        # MuJoCo EE 식별자
        self._ee_site_id = None
        self._ee_body_id = None
        self._ee_mj_warned = False
        # OpenCV window state
        self._cv_window_initialized = False

        # units mapping params
        self.declare_parameter('units_baseline', 2000.0)
        self.declare_parameter('units_per_degree', 4096.0/360.0)
        self.declare_parameter('units_motion_scale', 0.3)
        self.declare_parameter('units_min', 0.0)
        self.declare_parameter('units_max', 4095.0)
        self.declare_parameter('angle_min_deg', 160.0)
        self.declare_parameter('angle_max_deg', 210.0)
        self.units_baseline = float(self.get_parameter('units_baseline').get_parameter_value().double_value)
        self.units_per_degree = float(self.get_parameter('units_per_degree').get_parameter_value().double_value)
        self.units_motion_scale = float(self.get_parameter('units_motion_scale').get_parameter_value().double_value)
        self.units_min = float(self.get_parameter('units_min').get_parameter_value().double_value)
        self.units_max = float(self.get_parameter('units_max').get_parameter_value().double_value)
        self.angle_min_deg = float(self.get_parameter('angle_min_deg').get_parameter_value().double_value)
        self.angle_max_deg = float(self.get_parameter('angle_max_deg').get_parameter_value().double_value)

        # qpos mapping config
        self.joint_orientation = {
            'THUMB': [-1, -1, -1],
            'INDEX': [-1, -1, -1],
            'MIDDLE': [-1, -1, -1],
        }
        self.global_qpos_sign = -1
        self.qpos_gain = 0.5
        self.qpos_smooth_alpha = 0.5
        self.qpos_step_max = 0.05
        self.clamp_qpos_symm = True
        self.clamp_qpos_min = -1.57
        self.clamp_qpos_max = 1.57
        self.zero_qpos_ref = { 'THUMB': {}, 'INDEX': {}, 'MIDDLE': {} }
        self._prev_qpos_cmd = {}
        self.declare_parameter('units_per_rad', 4096.0 / (2.0 * math.pi))
        self.units_per_rad = float(self.get_parameter('units_per_rad').get_parameter_value().double_value)
        self.declare_parameter('units_motion_scale_qpos', 1.0)
        self.units_motion_scale_qpos = float(self.get_parameter('units_motion_scale_qpos').get_parameter_value().double_value)
        self._thumb_sign_patterns = [
            [1, 1, 1], [-1, -1, -1], [-1, 1, 1], [1, -1, 1], [1, 1, -1], [-1, -1, 1], [-1, 1, -1], [1, -1, -1],
        ]
        self._thumb_pattern_idx = 0

        # Optional MuJoCo init
        self._mj_enabled = False
        self._mj_model = None
        self._mj_data = None
        self._mj_viewer = None
        self._mj_qpos_adr = {}
        if self.run_mujoco and self.mujoco_model_path:
            try:
                self._mj_model = mj.MjModel.from_xml_path(self.mujoco_model_path)  # type: ignore[attr-defined]
                self._mj_data = mj.MjData(self._mj_model)  # type: ignore[attr-defined]
                # 모델/데이터 생성에 성공하면 뷰어 실패와 무관하게 기능 활성화
                self._mj_enabled = True
                for finger, names in DCLAW_JOINTS.items():
                    for name in names:
                        try:
                            jid = mj.mj_name2id(self._mj_model, mj.mjtObj.mjOBJ_JOINT, name)  # type: ignore[attr-defined]
                            if jid >= 0:
                                adr = int(self._mj_model.jnt_qposadr[jid])
                                self._mj_qpos_adr[name] = adr
                        except Exception:
                            pass
                # 뷰어는 선택 사항: 실패해도 시뮬레이션 데이터는 사용 가능
                try:
                    self._mj_viewer = mj_viewer.launch_passive(self._mj_model, self._mj_data, show_left_ui=False, show_right_ui=False)  # type: ignore[attr-defined]
                    self.get_logger().info('[MuJoCo] viewer started')
                except Exception as e:
                    self._mj_viewer = None
                    self.get_logger().warn(f'[MuJoCo] viewer start failed (continuing headless): {e}')
                # EE Pose from MuJoCo: resolve site/body ids
                try:
                    if self.ee_pose_mj_site:
                        sid = mj.mj_name2id(self._mj_model, mj.mjtObj.mjOBJ_SITE, self.ee_pose_mj_site)  # type: ignore[attr-defined]
                        if sid >= 0:
                            self._ee_site_id = int(sid)
                            self.get_logger().info(f"[EE] MuJoCo site 사용: {self.ee_pose_mj_site} (id={self._ee_site_id})")
                        else:
                            self.get_logger().warn(f"[EE] MuJoCo site 미발견: {self.ee_pose_mj_site}")
                    # 추가: THtip site 식별
                    self._ee_site_id_th = None
                    if self.ee_pose_mj_site_th:
                        sid_th = mj.mj_name2id(self._mj_model, mj.mjtObj.mjOBJ_SITE, self.ee_pose_mj_site_th)  # type: ignore[attr-defined]
                        if sid_th >= 0:
                            self._ee_site_id_th = int(sid_th)
                            self.get_logger().info(f"[EE] MuJoCo site 사용(TH): {self.ee_pose_mj_site_th} (id={self._ee_site_id_th})")
                        else:
                            self.get_logger().warn(f"[EE] MuJoCo site 미발견(TH): {self.ee_pose_mj_site_th}")
                    if self._ee_site_id is None and self.ee_pose_mj_body:
                        bid = mj.mj_name2id(self._mj_model, mj.mjtObj.mjOBJ_BODY, self.ee_pose_mj_body)  # type: ignore[attr-defined]
                        if bid >= 0:
                            self._ee_body_id = int(bid)
                            self.get_logger().info(f"[EE] MuJoCo body 사용: {self.ee_pose_mj_body} (id={self._ee_body_id})")
                        else:
                            self.get_logger().warn(f"[EE] MuJoCo body 미발견: {self.ee_pose_mj_body}")
                except Exception as e:
                    self.get_logger().warn(f"[EE] MuJoCo EE 식별자 확인 실패: {e}")
            except Exception as e:
                self.get_logger().warn(f'[MuJoCo] init failed: {e}')

        # RealSense pipeline (assume installed; let ImportError bubble otherwise)
        self.pipe = rs.pipeline()  # type: ignore[attr-defined]
        cfg = rs.config()  # type: ignore[attr-defined]
        cfg.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)  # type: ignore[attr-defined]
        cfg.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)  # type: ignore[attr-defined]
        self.profile = self.pipe.start(cfg)
        self.align = rs.align(rs.stream.color)  # type: ignore[attr-defined]

        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        color_prof = self.profile.get_stream(rs.stream.color).as_video_stream_profile()  # type: ignore[attr-defined]
        self.intr = color_prof.get_intrinsics()

        # MediaPipe hands
        self.mp_hands = mp.solutions.hands  # type: ignore[attr-defined]
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils  # type: ignore[attr-defined]
        # Visualization mapping options (affect display only)
        self._viz_thumb_complement = True
        # CSV init
        self._csv_fp = None
        self._csv_writer = None
        self._csv_active = False
        # 시간 기준 (로거 ON 시점). 파일 저장은 하지 않음
        self._time_start = None  # (sec, nsec) when logger became active
        if self.log_angles_csv_enable:
            try:
                if not self.log_angles_csv_path:
                    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                    self.log_angles_csv_path = str(pathlib.Path.cwd() / f"outpus/handangles/{ts}_handangles.csv")
                p = pathlib.Path(self.log_angles_csv_path)
                p.parent.mkdir(parents=True, exist_ok=True)
                need_header = (not p.exists()) or p.stat().st_size == 0
                self._csv_fp = open(p, 'a', newline='')
                self._csv_writer = csv.writer(self._csv_fp)
                if need_header:
                    header = ['t_sec','t_nanosec']
                    for finger, triples in FINGERS.items():
                        for (_i,_j,_k,jname) in triples:
                            header.append(f"{finger.lower()}_{jname.lower()}_raw_deg")
                    self._csv_writer.writerow(header)
                self.get_logger().info(f"Hand tracker CSV logging -> {p}")
                self._csv_active = True
            except Exception as e:
                self.get_logger().error(f"CSV 파일 열기 실패: {e}")

    def _on_logger_state(self, msg: Bool) -> None:
        # Follow data_logger active state for overlay elapsed time
        try:
            active = bool(msg.data)
        except Exception:
            active = False
        self._logger_active = active
        if active:
            try:
                now = self.get_clock().now().to_msg()
                self._time_start = (int(now.sec), int(now.nanosec))
            except Exception:
                self._time_start = None
        else:
            # reset baseline
            self._time_start = None

    def _on_ee_pose(self, msg: PoseStamped) -> None:
        try:
            self._ee_pose = (
                float(msg.pose.position.x),
                float(msg.pose.position.y),
                float(msg.pose.position.z),
            )
        except Exception:
            self._ee_pose = None


    def _print_keyboard_help(self) -> None:
        try:
            self.get_logger().info(
                "\n"
                "[Keyboard 단축키]\n"
                "  q : 종료 (quit)\n"
                "  r : (시각화) 엄지 보정 토글\n"
                "  s : Data Logger 저장 토글 (시작/정지)\n"
                "  h : 로봇으로 units Publish 토글 (ON/OFF)\n"
                "  c : 현재 손 자세를 zero 기준으로 설정\n"
                "  j : 현재 smoothed qpos 출력\n"
                "  t : 엄지 부호 패턴 순환\n"
                "  x : global_qpos_sign 반전(+/-)\n"
                "  a : 맵 모드 정보 (signed_angle)\n"
                "  g : qpos_gain 순환\n"
                "  참고: 키 입력은 OpenCV 창이 포커스일 때만 동작합니다.\n"
            )
        except Exception:
            pass

    def deproject(self, px, py, depth_img):
        if px < 0 or px >= depth_img.shape[1] or py < 0 or py >= depth_img.shape[0]:
            return None
        d = depth_img[py, px] * self.depth_scale
        if d <= 0:
            return None
        X, Y, Z = rs.rs2_deproject_pixel_to_point(self.intr, [float(px), float(py)], float(d))  # type: ignore[attr-defined]
        return np.array([X, Y, Z], dtype=np.float32)

    def _map_angle_to_qpos(self, finger: str, joint: str, raw_angle: float | None) -> float:
        """Map human joint angle (deg, 0~360 with 180 baseline) to qpos (rad).
        - signed-angle: deg' = raw-180, then to rad
        - apply joint orientation and global sign
        - scale by qpos_gain
        - clamp to symmetric limits if enabled
        Returns qpos in radians (float)."""
        if raw_angle is None or not (raw_angle == raw_angle):
            return 0.0
        raw_c = min(max(float(raw_angle), 0.0), 360.0)
        order = ["CMC","MCP","IP"] if finger == 'THUMB' else ["MCP","PIP","DIP"]
        try:
            jidx = order.index(joint)
        except ValueError:
            jidx = 0
        sdir_list = self.joint_orientation.get(finger, [1,1,1])
        sdir = sdir_list[jidx] if jidx < len(sdir_list) else 1
        deg = raw_c - 180.0
        qpos = math.radians(deg) * sdir * self.global_qpos_sign
        qpos *= self.qpos_gain
        if self.clamp_qpos_symm:
            if qpos < self.clamp_qpos_min:
                qpos = self.clamp_qpos_min
            elif qpos > self.clamp_qpos_max:
                qpos = self.clamp_qpos_max
        return qpos

    def run(self):
        try:
            while rclpy.ok():
                # 구독 콜백을 처리하여 logger 상태 및 EE pose 등이 즉시 반영되도록 함
                try:
                    rclpy.spin_once(self, timeout_sec=0.0)
                except Exception:
                    pass
                frames = self.pipe.wait_for_frames()
                frames = self.align.process(frames)
                depth = frames.get_depth_frame()
                color = frames.get_color_frame()
                if not depth or not color:
                    continue

                depth_img = np.asanyarray(depth.get_data())
                img = np.asanyarray(color.get_data())
                h, w = img.shape[:2]
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                res = self.hands.process(rgb)

                if res.multi_hand_landmarks:
                    lms = res.multi_hand_landmarks[0]
                    pix = [(int(min(max(lm.x,0),1)*w), int(min(max(lm.y,0),1)*h)) for lm in lms.landmark]
                    pts3d = [self.deproject(u, v, depth_img) for (u, v) in pix]

                    # Collect raw angles (0..360 preferred via 2D; fallback to 3D 0..180)
                    angles_raw: dict[str, dict[str, float | None]] = { 'THUMB': {}, 'INDEX': {}, 'MIDDLE': {} }
                    for finger, triples in FINGERS.items():
                        if finger not in angles_raw:
                            continue
                        for (i,j,k,name) in triples:
                            ang = angle_2d(np.array(pix[i]), np.array(pix[j]), np.array(pix[k]))
                            if ang is None:
                                ang = angle_3d(pts3d[i], pts3d[j], pts3d[k])
                            angles_raw[finger][name] = ang

                    def display_deg_for(finger: str, jname: str, deg: float | None) -> float | None:
                        if deg is None or not (deg == deg):
                            return None
                        # For thumb, optionally apply 360 - angle (complement) to mirror controller behavior
                        if self._viz_thumb_complement and finger.upper() == 'THUMB':
                            return max(0.0, min(360.0, 360.0 - float(deg)))
                        return max(0.0, min(360.0, float(deg)))

                    y0 = 26
                    for finger, triples in FINGERS.items():
                        parts = []
                        for (i,j,k,name) in triples:
                            # 우선 2D 0~360, 실패 시 3D 0~180
                            ang2d = angle_2d(np.array(pix[i]), np.array(pix[j]), np.array(pix[k]))
                            ang = ang2d if ang2d is not None else angle_3d(pts3d[i], pts3d[j], pts3d[k])
                            ddeg = display_deg_for(finger, name, ang)
                            if ddeg is not None:
                                parts.append(f"{name}:{ddeg:5.1f}°")
                        if parts:
                            cv2.putText(img, f"{finger}: " + " ".join(parts), (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
                            y0 += 20

                    def draw_label_near(idx, text, dy=-8, dx=6, color=(255,255,255)):
                        u,v = pix[idx]
                        cv2.putText(img, text, (u+dx, v+dy), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 2, cv2.LINE_AA)
                        cv2.putText(img, text, (u+dx, v+dy), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

                    for finger, triples in FINGERS.items():
                        for (i,j,k,name) in triples:
                            ang2d = angle_2d(np.array(pix[i]), np.array(pix[j]), np.array(pix[k]))
                            ang = ang2d if ang2d is not None else angle_3d(pts3d[i], pts3d[j], pts3d[k])
                            ddeg = display_deg_for(finger, name, ang)
                            if ddeg is not None:
                                draw_label_near(j, f"{name}:{ddeg:.0f}°")
                                # 부채꼴(호) 시각화 추가
                                try:
                                    # 손목과 관절 사이 픽셀 거리로 반지름 대략 조절
                                    p_i = np.array(pix[i], dtype=float)
                                    p_j = np.array(pix[j], dtype=float)
                                    dist = float(np.linalg.norm(p_i - p_j))
                                    base_r = int(max(18.0, min(40.0, dist * 0.6)))
                                    draw_angle_arc(img, pix, i, j, k, ddeg, radius=base_r)
                                except Exception:
                                    pass

                    # (삭제됨) EE Pose 퍼블리시 (vision 소스)

                    # Publish as JointState (angles in radians) in a stable name order
                    if self.publish_joint_state:
                        names = []
                        positions = []
                        for finger, triples in FINGERS.items():
                            for (i, j, k, jname) in triples:
                                joint_name = f"{finger.lower()}_{jname.lower()}"
                                ang_deg = angle_3d(pts3d[i], pts3d[j], pts3d[k])
                                ang_rad = math.radians(ang_deg) if ang_deg is not None else float('nan')
                                names.append(joint_name)
                                positions.append(ang_rad)

                        js = JointState()
                        js.header.stamp = self.get_clock().now().to_msg()
                        js.name = names
                        js.position = positions
                        if True:
                            # 주기적인 대량 로그를 피하기 위해 옵션으로만 출력
                            self.get_logger().info(f"publis6h positions len={len(js.position)}")
                        self.get_logger().info(f'{js.name}\n')
                            
                        self.joint_pub.publish(js)
                    # Publish 9-joint targets in units (0..4096) and drive MuJoCo if enabled
                    try:
                        order = [
                            ('THUMB','CMC'), ('THUMB','MCP'), ('THUMB','IP'),
                            ('INDEX','MCP'), ('INDEX','PIP'), ('INDEX','DIP'),
                            ('MIDDLE','MCP'), ('MIDDLE','PIP'), ('MIDDLE','DIP')
                        ]
                        out_units: list[int] = []
                        # keep a parallel map for smoothed qpos per joint
                        smoothed_qpos: dict[str, dict[str, float]] = { 'THUMB': {}, 'INDEX': {}, 'MIDDLE': {} }
                        for finger, jname in order:
                            raw_deg = angles_raw.get(finger, {}).get(jname)
                            # Apply same display transform (thumb complement) to control if enabled
                            if raw_deg is not None and self._viz_thumb_complement and finger == 'THUMB' and raw_deg > 0:
                                raw_deg = 360.0 - float(raw_deg)
                            # Map to qpos(rad)
                            qpos = self._map_angle_to_qpos(finger, jname, raw_deg)
                            # Apply zero offset
                            qpos -= self.zero_qpos_ref.get(finger, {}).get(jname, 0.0)
                            # Smooth + step limit + clamp
                            name_key = f"{finger}_{jname}"
                            prev = self._prev_qpos_cmd.get(name_key, 0.0)
                            smoothed = (1.0 - self.qpos_smooth_alpha) * prev + self.qpos_smooth_alpha * qpos
                            step = smoothed - prev
                            if step > self.qpos_step_max:
                                smoothed = prev + self.qpos_step_max
                            elif step < -self.qpos_step_max:
                                smoothed = prev - self.qpos_step_max
                            if self.clamp_qpos_symm:
                                if smoothed < self.clamp_qpos_min:
                                    smoothed = self.clamp_qpos_min
                                elif smoothed > self.clamp_qpos_max:
                                    smoothed = self.clamp_qpos_max
                            self._prev_qpos_cmd[name_key] = smoothed
                            # store for MuJoCo
                            smoothed_qpos.setdefault(finger, {})[jname] = smoothed
                            # Convert qpos(rad) -> units
                            units_f = self.units_baseline + smoothed * self.units_per_rad * self.units_motion_scale_qpos
                            if units_f < self.units_min:
                                units_f = self.units_min
                            elif units_f > self.units_max:
                                units_f = self.units_max
                            out_units.append(int(round(units_f)))
                        msg = Int32MultiArray()
                        msg.data = out_units
                        # print(msg)
                        if self.units_publish_enabled:
                            self.units_pub.publish(msg)
                        else:
                            # 발행 비활성 상태에서는 컨트롤러로 보내지 않음 (MuJoCo/시각화는 계속 동작)
                            pass
                        # Apply to MuJoCo
                        if self._mj_enabled and self._mj_model is not None and self._mj_data is not None:
                            try:
                                # Map to MuJoCo joint names
                                for finger, jlist in [('THUMB',["CMC","MCP","IP"]), ('INDEX',["MCP","PIP","DIP"]), ('MIDDLE',["MCP","PIP","DIP"])]:
                                    names = DCLAW_JOINTS.get(finger, [])
                                    for idx, jn in enumerate(jlist):
                                        if idx >= len(names):
                                            continue
                                        mj_name = names[idx]
                                        adr = self._mj_qpos_adr.get(mj_name)
                                        if adr is None:
                                            continue
                                        val = smoothed_qpos.get(finger, {}).get(jn)
                                        if val is None:
                                            continue
                                        self._mj_data.qpos[adr] = float(val)
                                mj.mj_forward(self._mj_model, self._mj_data)  # type: ignore[attr-defined]
                                if self._mj_viewer is not None and self._mj_viewer.is_running():  # type: ignore[attr-defined]
                                    self._mj_viewer.sync()  # type: ignore[attr-defined]
                                # Publish EE pose from MuJoCo (site_xpos or body xpos)
                                try:
                                    if (
                                        self.ee_pose_publish_enabled and self.ee_pose_pub_if is not None and
                                        self.ee_pose_source == 'mujoco'
                                    ):
                                        px = py = pz = None
                                        if self._ee_site_id is not None:
                                            try:
                                                pos = self._mj_data.site_xpos[self._ee_site_id]
                                                px, py, pz = float(pos[0]), float(pos[1]), float(pos[2])
                                            except Exception:
                                                px = py = pz = None
                                        elif self._ee_body_id is not None:
                                            try:
                                                pos = self._mj_data.xpos[self._ee_body_id]
                                                px, py, pz = float(pos[0]), float(pos[1]), float(pos[2])
                                            except Exception:
                                                px = py = pz = None
                                        if px is None or py is None or pz is None:
                                            if not self._ee_mj_warned:
                                                self.get_logger().warn('[EE] MuJoCo EE 위치를 가져오지 못했습니다 (site/body 미설정 또는 접근 실패).')
                                                self._ee_mj_warned = True
                                        else:
                                            msg = PoseStamped()
                                            msg.header.stamp = self.get_clock().now().to_msg()
                                            msg.header.frame_id = self.ee_pose_frame_id
                                            msg.pose.position.x = px
                                            msg.pose.position.y = py
                                            msg.pose.position.z = pz
                                            # Orientation은 사용하지 않으므로 항등 쿼터니언
                                            msg.pose.orientation.x = 0.0
                                            msg.pose.orientation.y = 0.0
                                            msg.pose.orientation.z = 0.0
                                            msg.pose.orientation.w = 1.0
                                            try:
                                                self.ee_pose_pub_if.publish(msg)
                                            except Exception:
                                                pass
                                            # Overlay 표시용 내부 상태도 업데이트
                                            self._ee_pose = (px, py, pz)
                                except Exception:
                                    pass
                                # 추가: MFtip / THtip 동시 퍼블리시 (MuJoCo 기준)
                                try:
                                    if self.ee_pose_publish_enabled and self.ee_pose_source == 'mujoco':
                                        now = self.get_clock().now().to_msg()
                                        # MF: 설정된 site/body 사용
                                        if self.ee_pose_pub_mf is not None:
                                            px_mf = py_mf = pz_mf = None
                                            if self._ee_site_id is not None:
                                                try:
                                                    p = self._mj_data.site_xpos[self._ee_site_id]
                                                    px_mf, py_mf, pz_mf = float(p[0]), float(p[1]), float(p[2])
                                                except Exception:
                                                    pass
                                            elif self._ee_body_id is not None:
                                                try:
                                                    p = self._mj_data.xpos[self._ee_body_id]
                                                    px_mf, py_mf, pz_mf = float(p[0]), float(p[1]), float(p[2])
                                                except Exception:
                                                    pass
                                            if px_mf is not None:
                                                m = PoseStamped()
                                                m.header.stamp = now
                                                m.header.frame_id = self.ee_pose_frame_id
                                                m.pose.position.x = px_mf; m.pose.position.y = py_mf; m.pose.position.z = pz_mf
                                                m.pose.orientation.w = 1.0
                                                try:
                                                    self.ee_pose_pub_mf.publish(m)
                                                except Exception:
                                                    pass
                                        # TH: 별도 site 사용
                                        if self.ee_pose_pub_th is not None and getattr(self, '_ee_site_id_th', None) is not None:
                                            try:
                                                p = self._mj_data.site_xpos[self._ee_site_id_th]
                                                m2 = PoseStamped()
                                                m2.header.stamp = now
                                                m2.header.frame_id = self.ee_pose_frame_id
                                                m2.pose.position.x = float(p[0])
                                                m2.pose.position.y = float(p[1])
                                                m2.pose.position.z = float(p[2])
                                                m2.pose.orientation.w = 1.0
                                                self.ee_pose_pub_th.publish(m2)
                                            except Exception:
                                                pass
                                except Exception:
                                    pass
                            except Exception as e:
                                # Avoid spamming logs
                                pass
                    except Exception as e:
                        self.get_logger().warn(f"units publish failed: {e}")
                    # CSV log per frame
                    if self.log_angles_csv_enable and self._csv_writer:
                        try:
                            now = self.get_clock().now().to_msg()
                            row = [now.sec, now.nanosec]
                            for finger, triples in FINGERS.items():
                                for (i, j, k, _jn) in triples:
                                    ang_deg = angle_3d(pts3d[i], pts3d[j], pts3d[k])
                                    row.append(f"{ang_deg:.4f}" if (ang_deg is not None and ang_deg == ang_deg) else '')
                            self._csv_writer.writerow(row)
                            if self._csv_fp:
                                self._csv_fp.flush()
                        except Exception as e:
                            self.get_logger().warn(f"CSV 로그 실패: {e}")
                    # 파일 저장은 하지 않음 (오버레이만 표시)

                    VISIBLE_IDX = {0,1,2,3,4,5,6,7,8,9,10,11,12}
                    VISIBLE_CONNS = [(a,b) for (a,b) in mp.solutions.hands.HAND_CONNECTIONS if a in VISIBLE_IDX and b in VISIBLE_IDX]  # type: ignore[attr-defined]
                    for i in VISIBLE_IDX:
                        u,v = pix[i]
                        cv2.circle(img, (u,v), 3, (0,255,0), -1)
                    for (a,b) in VISIBLE_CONNS:
                        ua,va = pix[a]; ub,vb = pix[b]
                        cv2.line(img, (ua,va), (ub,vb), (0,255,0), 1)
                else:
                    # No hand detected; optionally publish NaNs so the topic exists
                    if self.publish_joint_state and self.publish_empty_when_no_hand:
                        names = []
                        positions = []
                        for finger, triples in FINGERS.items():
                            for (_i, _j, _k, jname) in triples:
                                names.append(f"{finger.lower()}_{jname.lower()}")
                                positions.append(float('nan'))
                        js = JointState()
                        js.header.stamp = self.get_clock().now().to_msg()
                        js.name = names
                        js.position = positions
                        
                        self.joint_pub.publish(js)
                    # Optional: write empty row for time continuity
                    if self.log_angles_csv_enable and self._csv_writer:
                        try:
                            now = self.get_clock().now().to_msg()
                            row = [now.sec, now.nanosec]
                            for _finger, triples in FINGERS.items():
                                for _ in triples:
                                    row.append('')
                            self._csv_writer.writerow(row)
                            if self._csv_fp:
                                self._csv_fp.flush()
                        except Exception as e:
                            self.get_logger().warn(f"CSV 로그 실패(손 없음): {e}")
                # ---- Overlay: units_publish_enabled 상태 표시 (항상 표시) ----
                try:
                    # Units publish status
                    status_txt = f"UNITS PUBLISH (h): {'ON' if self.units_publish_enabled else 'OFF'}"
                    color_on = (0, 200, 0)
                    color_off = (0, 0, 255)
                    color = color_on if self.units_publish_enabled else color_off
                    cv2.putText(img, status_txt, (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
                    cv2.putText(img, status_txt, (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
                    # Data logger active state overlay (line below): 항상 표시
                    # 기본값은 OFF로 표기하여 혼동 최소화 (상태 수신 전)
                    if self._logger_active is None:
                        log_state_txt = 'OFF'
                        lcolor = (0, 0, 255)
                    else:
                        log_state_txt = 'ON' if self._logger_active else 'OFF'
                        lcolor = (0, 200, 0) if self._logger_active else (0, 0, 255)
                    log_txt = f"DATA LOGGER (s): {log_state_txt}"
                    cv2.putText(img, log_txt, (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
                    cv2.putText(img, log_txt, (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.55, lcolor, 1, cv2.LINE_AA)
                    # Elapsed time since logger start (sec)
                    try:
                        if self._logger_active and self._time_start is not None:
                            now = self.get_clock().now().to_msg()
                            s0, n0 = self._time_start
                            ds = int((int(now.sec) - int(s0)) + (int(now.nanosec) - int(n0)) / 1e9)
                            t_txt = f"TIME: {ds}s"
                            cv2.putText(img, t_txt, (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
                            cv2.putText(img, t_txt, (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 200, 255), 1, cv2.LINE_AA)
                    except Exception:
                        pass
                    # EE pose (optional) on next line
                    try:
                        if getattr(self, '_ee_pose', None) is not None:
                            # 지원: (px,py,pz) 또는 과거 호환 (px,py,pz,qx,qy,qz,qw)
                            if len(self._ee_pose) >= 7:  # type: ignore[arg-type]
                                px,py,pz,qx,qy,qz,qw = self._ee_pose  # type: ignore[misc]
                                ee_txt = f"EE: p=({px:+.3f},{py:+.3f},{pz:+.3f}) q=({qx:+.2f},{qy:+.2f},{qz:+.2f},{qw:+.2f})"
                            else:
                                px,py,pz = self._ee_pose  # type: ignore[misc]
                                ee_txt = f"EE: p=({px:+.3f},{py:+.3f},{pz:+.3f})"
                            cv2.putText(img, ee_txt, (10, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 0), 2, cv2.LINE_AA)
                            cv2.putText(img, ee_txt, (10, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 50), 1, cv2.LINE_AA)
                    except Exception:
                        pass
                    # Note: HAND CSV 상태는 s키로 제어하지 않으므로 별도 표시하지 않음
                except Exception:
                    pass

                key = 255
                if self.enable_cv_window:
                    if not self._cv_window_initialized:
                        try:
                            cv2.namedWindow('RealSense + MediaPipe (angles)', cv2.WINDOW_NORMAL)
                            cv2.resizeWindow('RealSense + MediaPipe (angles)', 960, 720)
                            try:
                                cv2.moveWindow('RealSense + MediaPipe (angles)', 40, 40)
                            except Exception:
                                pass
                            self._cv_window_initialized = True
                        except Exception as e:
                            # 창 생성 실패 시 비활성화하여 루프가 막히지 않도록
                            self.get_logger().warn(f"OpenCV 창 생성 실패: {e}. 비주얼 비활성화.")
                            self.enable_cv_window = False
                    if self.enable_cv_window:
                        cv2.imshow('RealSense + MediaPipe (angles)', img)
                        key = cv2.waitKey(1) & 0xFF
                    else:
                        key = 255
                if key != 255:
                    try:
                        ch = chr(key)
                    except Exception:
                        ch = ''
                    # 대소문자 구분 없이 처리
                    if ch:
                        ch = ch.lower()
                    # local handling
                    if ch == 'q':
                        # publish and quit
                        self.key_pub.publish(String(data='q'))
                        break
                    elif ch == 'r':
                        self._viz_thumb_complement = not self._viz_thumb_complement
                        self.get_logger().info(f"[Vis] complement (thumb) -> {self._viz_thumb_complement}")
                        self.key_pub.publish(String(data='r'))
                    elif ch == 's':
                        # 중앙 Data Logger 저장 토글 신호만 전달
                        try:
                            self.key_pub.publish(String(data='s'))
                            self.get_logger().info('[Key] Requested DataLogger toggle (s)')
                        except Exception:
                            pass
                    elif ch == 'h':
                        # 로봇으로의 Publish 토글 (로컬 게이트 + 컨트롤러에도 알림)
                        self.units_publish_enabled = not self.units_publish_enabled
                        self.get_logger().info(f"[Publish] units -> {'ON' if self.units_publish_enabled else 'OFF'}")
                        self.key_pub.publish(String(data='h'))
                    elif ch in ('c','j','t','x','a','g'):
                        # local mapping toggles + forward to controller for symmetry
                        if ch == 'c':
                            # set zero offsets from current raw angles (when available from last frame)
                            try:
                                # Use last computed angles_raw from this iteration scope
                                for finger, triples in FINGERS.items():
                                    joints = ['CMC','MCP','IP'] if finger == 'THUMB' else ['MCP','PIP','DIP']
                                    for jname in joints:
                                        rawv = None
                                        # best-effort: reuse last local var if in scope
                                        # (angles_raw defined only when hand is detected)
                                        try:
                                            rawv = angles_raw.get(finger, {}).get(jname)  # type: ignore[name-defined]
                                        except Exception:
                                            rawv = None
                                        if rawv is not None and self._viz_thumb_complement and finger == 'THUMB' and rawv > 0:
                                            rawv = 360.0 - float(rawv)
                                        q = self._map_angle_to_qpos(finger, jname, rawv)
                                        self.zero_qpos_ref.setdefault(finger, {})[jname] = q if q is not None else 0.0
                                self._prev_qpos_cmd.clear()
                                self.get_logger().info('[Zero] zero_qpos_ref updated from current hand pose')
                            except Exception as e:
                                self.get_logger().warn(f'[Zero] failed: {e}')
                        elif ch == 'j':
                            # print current smoothed qpos snapshot
                            try:
                                lines = []
                                groups = [('THUMB',['CMC','MCP','IP']), ('INDEX',['MCP','PIP','DIP']), ('MIDDLE',['MCP','PIP','DIP'])]
                                for finger, jlist in groups:
                                    segs = []
                                    for jn in jlist:
                                        name_key = f'{finger}_{jn}'
                                        v = self._prev_qpos_cmd.get(name_key, 0.0)
                                        segs.append(f"{jn}={v:+.3f}")
                                    lines.append(f"{finger}: " + ', '.join(segs))
                                self.get_logger().info("[QPOS]\n" + "\n".join(lines))
                            except Exception:
                                pass
                        elif ch == 't':
                            self._thumb_pattern_idx = (self._thumb_pattern_idx + 1) % len(self._thumb_sign_patterns)
                            self.joint_orientation['THUMB'] = self._thumb_sign_patterns[self._thumb_pattern_idx]
                            self.get_logger().info(f"[Thumb] sign pattern -> {self.joint_orientation['THUMB']}")
                        elif ch == 'x':
                            self.global_qpos_sign = -self.global_qpos_sign
                            self.get_logger().info(f"[Ctrl] global_qpos_sign -> {self.global_qpos_sign:+d}")
                        elif ch == 'a':
                            # Only signed_angle supported; keep for future extensibility
                            self.get_logger().info('[Map] signed_angle mode')
                        elif ch == 'g':
                            choices = [0.25, 0.5, 0.75, 1.0]
                            try:
                                i = choices.index(self.qpos_gain)
                            except ValueError:
                                i = 0
                            self.qpos_gain = choices[(i + 1) % len(choices)]
                            self.get_logger().info(f"[Ctrl] qpos_gain -> {self.qpos_gain:.2f}")
                        # always forward to controller to keep behaviors in sync
                        self.key_pub.publish(String(data=ch))
                # (삭제됨) EE Pose 퍼블리시 (MuJoCo 소스)
        finally:
            try:
                self.pipe.stop()
            except Exception:
                pass
            if getattr(self, 'enable_cv_window', False):
                try:
                    if self.enable_cv_window:
                        try:
                            cv2.destroyAllWindows()
                        except Exception:
                            pass
                except Exception:
                    pass
            # Close MuJoCo viewer if running
            try:
                if getattr(self, '_mj_viewer', None) is not None:
                    self._mj_viewer.close()  # type: ignore[attr-defined]
            except Exception:
                pass
            fp = getattr(self, '_csv_fp', None)
            if fp is not None:
                try:
                    fp.close()
                except Exception:
                    pass


def main(args=None):
    rclpy.init(args=args)
    node = HandTrackerNode()
    node.get_logger().info('Hand tracker node started')
    node._print_keyboard_help()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass

if __name__ == '__main__':
    main()
