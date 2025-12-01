#!/usr/bin/env python3
"""Torque-Based Impedance Controller Node (Clean Rebuild)

기능 개요:
 - 정책에서 받은 Cartesian stiffness (finger별 3축 -> 총 9D)를 이용해 스프링-댐퍼 힘 계산
 - Jacobian을 통해 joint torque로 변환 후 Dynamixel current 명령 퍼블리시
 - 선택적으로 MuJoCo 기반 kinematics 및 viewer 출력(render_mujoco)
 - 선택적 force feedback (힘 측정 토픽을 통한 간단 P+I 보정)
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple
import os
import yaml  # type: ignore
import math
from datetime import datetime
import threading
import sys
import select
import termios
import tty

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, TwistStamped, WrenchStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray, Int32MultiArray, Bool, UInt8, Float32

try:
    import mujoco as mj  # type: ignore
    MUJOCO_AVAILABLE = True
except ImportError:
    mj = None  # type: ignore
    MUJOCO_AVAILABLE = False

DEFAULT_MUJOCO_MODEL = "/home/songwoo/git/ur_dclaw/dclaw_finger_description/urdf/dclaw_finger_mjcf_final.xml"
SITE_NAMES = {"if": "FFtip", "mf": "MFtip", "th": "THtip"}
DCLAW_JOINTS = {
    "thumb": ["THJ30", "THJ31", "THJ32"],
    "index": ["FFJ10", "FFJ11", "FFJ12"],
    "middle": ["MFJ20", "MFJ21", "MFJ22"],
}

CURRENT_TO_TORQUE = 1.78e-3
CURRENT_UNIT = 2.69
MAX_CURRENT_HW = 1193


@dataclass
class FingerKinematicState:
    ee_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    ee_vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    jacobian: Optional[np.ndarray] = None
    has_desired_pos: bool = False
    has_desired_vel: bool = False


class TorqueImpedanceControllerNode(Node):
    def __init__(self) -> None:
        super().__init__("torque_impedance_controller_node")
        self._pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        default_stiffness_log_dir = os.path.join(self._pkg_root, "outputs", "stiffness_logs")
        # Parameters
        self.declare_parameter("rate_hz", 100.0)
        self.declare_parameter("use_mujoco", True)
        self.declare_parameter("render_mujoco", True)
        self.declare_parameter("mujoco_model_path", DEFAULT_MUJOCO_MODEL)
        self.declare_parameter("stiffness_scale", 1.0)
        self.declare_parameter("damping_ratio", 0.7)
        self.declare_parameter("virtual_mass", 0.1)
        self.declare_parameter("max_torque", 2.0)
        self.declare_parameter("max_current_units_pos", 500)
        self.declare_parameter("max_current_units_neg", 500)
        self.declare_parameter("torque_filter_alpha", 0.3)
        self.declare_parameter("velocity_window", 15)
        self.declare_parameter("enable_force_feedback", False)
        self.declare_parameter("force_sensor_map", ["if", "mf", "th"])
        self.declare_parameter("kp_force", 0.3)
        self.declare_parameter("ki_force", 0.0)
        self.declare_parameter("stiffness_filter_alpha", 0.01)
        self.declare_parameter("max_stiffness_change", 50.0)
        # New parameters (robot controller joint source + initial baseline + EE pose publish)
        self.declare_parameter("joint_state_topic", "/robot_controller/joint_state")
        self.declare_parameter("initial_qpos", [0.0]*9)
        self.declare_parameter("ee_pose_publish_enabled", True)
        self.declare_parameter("ee_pose_frame_id", "world")
        self.declare_parameter("ee_pose_topic_if", "/ee_pose_if")
        self.declare_parameter("ee_pose_topic_mf", "/ee_pose_mf")
        self.declare_parameter("ee_pose_topic_th", "/ee_pose_th")
        self.declare_parameter("position_error_threshold", 0.05)  # 5cm threshold for tracking lag warning
        self.declare_parameter("current_units_scale", [1.0] * 9)  # Per-joint scaling factor for current units output
        self.declare_parameter("stiffness_logging_enabled", True)
        self.declare_parameter("stiffness_log_dir", default_stiffness_log_dir)
        self.declare_parameter("max_pwm_limit", 400)  # PWM safety limit for plotting (matches robot_controller)

        def _p(name: str):
            return self.get_parameter(name).value
        self.rate_hz = float(_p("rate_hz") or 100.0)
        self.use_mujoco = bool(_p("use_mujoco") if _p("use_mujoco") is not None else True)
        self.render_mujoco = bool(_p("render_mujoco") if _p("render_mujoco") is not None else False)
        mp = _p("mujoco_model_path")
        self.model_path = str(mp) if mp else DEFAULT_MUJOCO_MODEL
        self.stiffness_scale = float(_p("stiffness_scale") or 1.0)
        self.damping_ratio = float(_p("damping_ratio") or 0.7)
        self.virtual_mass = float(_p("virtual_mass") or 0.1)
        self.max_torque = float(_p("max_torque") or 200.0)
        self.max_current_units_pos = int(_p("max_current_units_pos") or 500)
        self.max_current_units_neg = int(_p("max_current_units_neg") or 500)
        self.torque_filter_alpha = float(_p("torque_filter_alpha") or 0.3)
        self.velocity_window = int(_p("velocity_window") or 5)
        self.enable_force_fb = bool(_p("enable_force_feedback") or False)
        fm = _p("force_sensor_map")
        self.force_sensor_map = [str(x) for x in (fm if isinstance(fm, (list, tuple)) else ["if", "mf", "th"])]
        self.kp_force = float(_p("kp_force") or 0.3)
        self.ki_force = float(_p("ki_force") or 0.0)
        self.stiffness_alpha = float(_p("stiffness_filter_alpha") or 0.1)
        self.max_k_change = float(_p("max_stiffness_change") or 1.0)
        self.joint_state_topic = str(_p("joint_state_topic") or "/robot_controller/joint_state")
        self.pos_error_threshold = float(_p("position_error_threshold") or 0.05)
        self.stiffness_logging_enabled = bool(_p("stiffness_logging_enabled") if _p("stiffness_logging_enabled") is not None else True)
        self.max_pwm_limit = int(_p("max_pwm_limit") or 400)  # PWM limit for plotting
        log_dir_param = _p("stiffness_log_dir")
        self.stiffness_log_dir = str(log_dir_param) if log_dir_param else default_stiffness_log_dir
        if self.stiffness_logging_enabled:
            try:
                os.makedirs(self.stiffness_log_dir, exist_ok=True)
            except Exception as exc:
                self.get_logger().warning(f"[STIFFNESS_LOG] Failed to create log dir {self.stiffness_log_dir}: {exc}")
                self.stiffness_logging_enabled = False
        
        # Parse current_units_scale as array
        scale_param = _p("current_units_scale")
        if isinstance(scale_param, (list, tuple)):
            self.current_units_scale = np.array([float(x) for x in scale_param], dtype=float)
        else:
            self.current_units_scale = np.array([float(scale_param or 1.0)] * 9, dtype=float)
        
        # Load initial qpos: parameter first, then override by config.yaml if available
        raw_initial_qpos = _p("initial_qpos")
        self.initial_qpos = np.zeros(9)
        if isinstance(raw_initial_qpos, (list, tuple)) and len(raw_initial_qpos) >= 9:
            try:
                self.initial_qpos = np.array([float(x) for x in raw_initial_qpos[:9]], dtype=float)
            except Exception:
                pass
        cfg_qpos = self._load_initial_qpos_from_config()
        if cfg_qpos is not None:
            self.initial_qpos = cfg_qpos
            self.get_logger().info("initial_qpos overridden from config.yaml")
        self.ee_pose_publish_enabled = bool(_p("ee_pose_publish_enabled") if _p("ee_pose_publish_enabled") is not None else True)
        self.ee_pose_frame_id = str(_p("ee_pose_frame_id") or "world")
        self.ee_pose_topic_if = str(_p("ee_pose_topic_if") or "/ee_pose_if")
        self.ee_pose_topic_mf = str(_p("ee_pose_topic_mf") or "/ee_pose_mf")
        self.ee_pose_topic_th = str(_p("ee_pose_topic_th") or "/ee_pose_th")
        
        # Log critical parameters at startup
        self.get_logger().info(f"[PARAM] position_error_threshold={self.pos_error_threshold*1000:.1f}mm")
        self.get_logger().info(f"[PARAM] max_torque={self.max_torque:.3f} Nm, max_current_units=[+{self.max_current_units_pos}, -{self.max_current_units_neg}]")
        self.get_logger().info(f"[PARAM] current_units_scale=[{', '.join(f'{x:.3f}' for x in self.current_units_scale)}]")
        self.get_logger().info(f"[PARAM] stiffness_scale={self.stiffness_scale}, damping_ratio={self.damping_ratio}")
        self.get_logger().info(f"[PARAM] current_units_scale={self.current_units_scale}")

        # Log critical parameters
        self.get_logger().info(f"[INIT] position_error_threshold = {self.pos_error_threshold} meters ({self.pos_error_threshold*1000:.1f} mm)")
        self.get_logger().info(f"[INIT] max_torque = {self.max_torque}, max_current_units = [+{self.max_current_units_pos}, -{self.max_current_units_neg}]")
        self.get_logger().info(f"[INIT] rate_hz = {self.rate_hz}, damping_ratio = {self.damping_ratio}")

        # State
        self.target_stiffness = np.zeros(9)
        # Seed with initial_qpos until first real joint state arrives
        self.current_qpos = self.initial_qpos.copy()
        self.current_qvel = np.zeros(9)
        self.fingers: Dict[str, FingerKinematicState] = {"if": FingerKinematicState(), "mf": FingerKinematicState(), "th": FingerKinematicState()}
        self.desired_pos = {"if": np.zeros(3), "mf": np.zeros(3), "th": np.zeros(3)}
        self.desired_vel = {"if": np.zeros(3), "mf": np.zeros(3), "th": np.zeros(3)}
        self.ee_pos_history = {"if": [], "mf": [], "th": []}
        self.last_torques = np.zeros(9)
        self.has_stiffness = False
        self.has_qpos = False
        self.measured_force: Dict[str, float] = {"if": 0.0, "mf": 0.0, "th": 0.0}
        self.measured_force_vec: Dict[str, np.ndarray] = {"if": np.zeros(3), "mf": np.zeros(3), "th": np.zeros(3)}
        self.has_force: Dict[str, bool] = {"if": False, "mf": False, "th": False}
        self.force_int_err = np.zeros(9)
        self.filtered_stiffness = np.zeros(9)
        self._joint_state_sub_counter = 0
        self._error_throttle_counter = 0  # Manual throttle counter for error messages
        # Desired EE logging timestamps (per finger)
        self._desired_ee_first_log: Dict[str, bool] = {"if": False, "mf": False, "th": False}
        self._desired_ee_last_log: Dict[str, float] = {"if": 0.0, "mf": 0.0, "th": 0.0}
        # First-time warning flags (only log once during initialization)
        self._logged_no_stiffness_warning = False
        self._logged_no_desired_pos_warning = False
        # Flag to use current EE position as initial desired when no desired pos exists
        self._use_current_as_initial_desired = True
        # Demo playback status
        self.demo_playback_active = False
        self._playback_status_logged = False
        self.demo_playback_stage = 0
        self.stiffness_log_active = False
        self.stiffness_log_data: List[Tuple[float, np.ndarray]] = []
        self.eccentricity_log_data: List[Tuple[float, float]] = []
        self.eccentricity_smoothed_log_data: List[Tuple[float, float]] = []  # Smoothed eccentricity from run_policy_node
        self.force_log_data: List[Tuple[float, Dict[str, np.ndarray]]] = []  # Store fx,fy,fz for each finger
        self.ee_pos_log_data: List[Tuple[float, Dict[str, np.ndarray]]] = []  # Store actual EE positions
        self.desired_pos_log_data: List[Tuple[float, Dict[str, np.ndarray]]] = []  # Store desired EE positions
        self.torque_log_data: List[Tuple[float, np.ndarray]] = []  # Store computed torques
        self.current_units_log_data: List[Tuple[float, np.ndarray]] = []  # Store current units (clipped)
        self.current_units_raw_log_data: List[Tuple[float, np.ndarray]] = []  # Store raw current units (before clipping)
        self.pwm_log_data: List[Tuple[float, np.ndarray]] = []  # Store PWM values (9 joints)
        self.stiffness_log_start_time = 0.0
        self._stiffness_log_session_id = 0
        self.current_eccentricity = 0.0
        self.current_eccentricity_smoothed = 0.0
        self.has_eccentricity = False
        self.has_eccentricity_smoothed = False
        self.current_pwm: Optional[np.ndarray] = None  # Latest PWM values

        self.mj_model = None
        self.mj_data = None
        self.mj_qpos_adr: Dict[str, int] = {}
        self._mj_viewer = None  # passive viewer handle
        if self.use_mujoco:
            self._init_mujoco()

        # Subscriptions
        self.create_subscription(Float32MultiArray, "/impedance_control/target_stiffness", self.subscribe_stiffness, 10)
        self.create_subscription(Float32, "/deformity_tracker/eccentricity", self.subscribe_eccentricity, 10)
        self.create_subscription(Float32, "/deformity_tracker/eccentricity_smoothed", self.subscribe_eccentricity_smoothed, 10)
        # Subscribe to external joint state (robot controller) instead of hardcoded hand tracker
        self.create_subscription(JointState, self.joint_state_topic, self.subscribe_joint_state, 10)
        self.get_logger().info(f"[INIT] JointState subscriber CREATED: topic={self.joint_state_topic}")
        self.create_subscription(PoseStamped, "/ee_pose_desired_if", self.subscribe_desired_pose_if, 10)
        self.create_subscription(PoseStamped, "/ee_pose_desired_mf", self.subscribe_desired_pose_mf, 10)
        self.create_subscription(PoseStamped, "/ee_pose_desired_th", self.subscribe_desired_pose_th, 10)
        self.create_subscription(TwistStamped, "/ee_velocity_desired_if", self.subscribe_desired_velocity_if, 10)
        self.create_subscription(TwistStamped, "/ee_velocity_desired_mf", self.subscribe_desired_velocity_mf, 10)
        self.create_subscription(TwistStamped, "/ee_velocity_desired_th", self.subscribe_desired_velocity_th, 10)
        # Subscribe to demo playback status
        self.create_subscription(Bool, "/demo_playback_active", self.subscribe_playback_status, 10)
        self.get_logger().info("[INIT] Subscribed to /demo_playback_active")
        self.create_subscription(UInt8, "/demo_playback_stage", self.subscribe_playback_stage, 10)
        self.get_logger().info("[INIT] Subscribed to /demo_playback_stage")
        # Always subscribe to force sensors for logging (independent of enable_force_feedback)
        self.create_subscription(WrenchStamped, "/force_sensor/s1/wrench", self.subscribe_wrench_single_s1, 10)
        self.create_subscription(WrenchStamped, "/force_sensor/s2/wrench", self.subscribe_wrench_single_s2, 10)
        self.create_subscription(WrenchStamped, "/force_sensor/s3/wrench", self.subscribe_wrench_single_s3, 10)
        self.get_logger().info("[INIT] Force sensor subscribers created (s1, s2, s3)")
        # Subscribe to PWM topic for monitoring and logging
        self.create_subscription(Int32MultiArray, "/dynamixel/present_pwm", self.subscribe_pwm, 10)
        self.get_logger().info("[INIT] PWM subscriber created (/dynamixel/present_pwm)")

        # Publishers & timer
        self.current_pub = self.create_publisher(Int32MultiArray, "/dynamixel/goal_current", 10)
        self.torque_pub = self.create_publisher(Float32MultiArray, "/impedance_control/computed_torques", 10)
        self._log_counter = 0
        self.control_timer = self.create_timer(1.0 / self.rate_hz, self._control_loop)
        self.get_logger().info(f"TorqueImpedanceController started (rate={self.rate_hz}Hz, mujoco={'on' if self.use_mujoco else 'off'})")

        # EE pose publishers
        self.ee_pose_pub_if = None
        self.ee_pose_pub_mf = None
        self.ee_pose_pub_th = None
        if self.ee_pose_publish_enabled:
            if self.ee_pose_topic_if:
                self.ee_pose_pub_if = self.create_publisher(PoseStamped, self.ee_pose_topic_if, 10)
            if self.ee_pose_topic_mf:
                self.ee_pose_pub_mf = self.create_publisher(PoseStamped, self.ee_pose_topic_mf, 10)
            if self.ee_pose_topic_th:
                self.ee_pose_pub_th = self.create_publisher(PoseStamped, self.ee_pose_topic_th, 10)
            # Timer for continuous EE pose publishing (same rate as control loop)
            self.create_timer(1.0 / self.rate_hz, self.publish_ee_pose_message)
            self.get_logger().info(f"EE pose publish timer created: rate={self.rate_hz}Hz")

        # Goal position publisher for initial pose movement
        self._goal_position_pub = self.create_publisher(Int32MultiArray, "/dynamixel/goal_position", 10)
        
        # Keyboard listener thread (non-blocking)
        self._keyboard_enabled = True
        self._keyboard_thread = threading.Thread(target=self._keyboard_listener, daemon=True)
        self._keyboard_thread.start()
        self.get_logger().info("[KEYBOARD] Listener started - Press 'i' to move to initial pose, 'q' to quit keyboard listener")

    def _init_mujoco(self) -> None:
        if not MUJOCO_AVAILABLE:
            self.get_logger().warning("MuJoCo import 실패 -> kinematics 비활성화")
            self.use_mujoco = False
            return
        # 모델 로드
        self.get_logger().info(f"[MUJOCO] Loading model from: {self.model_path}")
        try:
            self.mj_model = mj.MjModel.from_xml_path(self.model_path)  # type: ignore
            self.mj_data = mj.MjData(self.mj_model)  # type: ignore
            self.get_logger().info("[MUJOCO] Model and data created successfully")
        except Exception as e:
            import traceback
            self.get_logger().error(f"[MUJOCO] Initialization failed: {e}")
            self.get_logger().error(f"[MUJOCO] Traceback:\n{traceback.format_exc()}")
            self.use_mujoco = False
            self.mj_model = None
            self.mj_data = None
            return
        # 조인트 address 매핑
        for finger, joint_names in DCLAW_JOINTS.items():
            for i, joint_name in enumerate(joint_names):
                try:
                    jid = mj.mj_name2id(self.mj_model, mj.mjtObj.mjOBJ_JOINT, joint_name)  # type: ignore
                    if jid >= 0:
                        adr = int(self.mj_model.jnt_qposadr[jid])  # type: ignore
                        self.mj_qpos_adr[f"{finger}_{i}"] = adr
                except Exception:
                    continue
        self.get_logger().info(f"MuJoCo 모델 로드 성공: {self.model_path}")
        self.get_logger().info(f"[MUJOCO] Joints mapped: {list(self.mj_qpos_adr.keys())}")
        # Apply initial pose from config to MuJoCo
        self._apply_initial_mujoco_pose()
        # Passive viewer (SenseGlove 방식 모방)
        if self.render_mujoco:
            try:
                from mujoco import viewer as mj_viewer  # type: ignore
            except Exception as e:
                self.get_logger().warning(f"mujoco.viewer import 실패 -> headless: {e}")
                return
            try:
                self._mj_viewer = mj_viewer.launch_passive(
                    self.mj_model,
                    self.mj_data,
                    show_left_ui=False,
                    show_right_ui=False,
                    key_callback=None,
                )  # type: ignore
                self.get_logger().info("MuJoCo viewer 시작")
            except Exception as e:
                self._mj_viewer = None
                self.get_logger().warning(f"Viewer 시작 실패(headless 진행): {e}")

    def _load_initial_qpos_from_config(self) -> Optional[np.ndarray]:
        """Load initial qpos from config.yaml as units (no conversion)."""
        try:
            pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            cfg_path = os.path.join(pkg_dir, "resource", "robot_parameter", "config.yaml")
        except Exception:
            return None
        if not os.path.exists(cfg_path):
            return None
        data = None
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except Exception:
            return None
        if not isinstance(data, dict):
            return None
        dyn = data.get("dynamixel")
        if not isinstance(dyn, dict):
            return None
        init_vals = dyn.get("initial_positions") or dyn.get("initial_position")
        if not isinstance(init_vals, (list, tuple)) or len(init_vals) < 9:
            return None
        
        # Store units directly (robot_controller now publishes units in JointState.position)
        qpos = np.zeros(9)
        for cmd_idx in range(9):
            if cmd_idx >= len(init_vals):
                break
            try:
                qpos[cmd_idx] = float(init_vals[cmd_idx])
            except Exception:
                continue
        
        return qpos

    def _apply_initial_mujoco_pose(self) -> None:
        """Apply initial qpos from config to MuJoCo (units → radians conversion)."""
        if not self.use_mujoco or self.mj_model is None or self.mj_data is None:
            return
        
        any_written = False
        finger_list = ["thumb", "index", "middle"]
        units_per_rad = 4096.0 / (2.0 * math.pi)
        
        for f_idx, finger in enumerate(finger_list):
            joint_names = DCLAW_JOINTS[finger]
            for j_idx, mj_joint in enumerate(joint_names):
                key = f"{finger}_{j_idx}"
                if key not in self.mj_qpos_adr:
                    continue
                adr = self.mj_qpos_adr[key]
                try:
                    cmd_idx = f_idx * 3 + j_idx
                    units_val = float(self.initial_qpos[cmd_idx])
                except (IndexError, TypeError, ValueError):
                    continue
                
                # Convert units → radians (subtract bias, divide by units_per_rad, add offset)
                bias = 1000.0 if cmd_idx in [0, 3, 6] else 2000.0
                offset = 1.57 if cmd_idx in [0, 3, 6] else 3.14
                rad_val = (units_val - bias) / units_per_rad + offset
                
                # MuJoCo expects radians without offset
                self.mj_data.qpos[adr] = rad_val - offset
                any_written = True
        
        if not any_written:
            return
        
        try:
            mj.mj_forward(self.mj_model, self.mj_data)  # type: ignore
        except Exception as e:
            self.get_logger().warning(f"[MuJoCo] forward failed after initial pose: {e}")
            return
        
        self.get_logger().info("[MuJoCo] qpos set to initial config values (units→radians)")
        
        # Compute initial EE positions from initial pose
        for finger, site_name in SITE_NAMES.items():
            try:
                sid = mj.mj_name2id(self.mj_model, mj.mjtObj.mjOBJ_SITE, site_name)  # type: ignore
                if sid >= 0:
                    pos = self.mj_data.site_xpos[sid].copy()  # type: ignore
                    self.fingers[finger].ee_pos = pos
                    self.get_logger().info(f"[MUJOCO] Initial EE position {finger}: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")
            except Exception as e:
                self.get_logger().warning(f"[MUJOCO] Failed to get initial EE for {finger}: {e}")

    def move_to_initial_pose(self) -> None:
        """Move robot to initial pose by publishing initial_qpos to /dynamixel/goal_position."""
        self.get_logger().info("[KEYBOARD] 'i' pressed -> Moving to INITIAL POSE")
        
        # Publish goal position (initial_qpos in units)
        try:
            goal_pos_msg = Int32MultiArray()
            goal_pos_msg.data = [int(q) for q in self.initial_qpos]
            
            # Create publisher if not exists
            if not hasattr(self, '_goal_position_pub') or self._goal_position_pub is None:
                self._goal_position_pub = self.create_publisher(Int32MultiArray, "/dynamixel/goal_position", 10)
            
            self._goal_position_pub.publish(goal_pos_msg)
            self.get_logger().info(f"[INITIAL POSE] Published goal_position: {goal_pos_msg.data}")
            
            # Reset desired positions so impedance controller doesn't fight
            for finger in ['th', 'if', 'mf']:
                self.fingers[finger].has_desired_pos = False
            
            # Also reset MuJoCo visualization
            self._apply_initial_mujoco_pose()
            
        except Exception as e:
            self.get_logger().error(f"[INITIAL POSE] Failed to move: {e}")

    def _keyboard_listener(self) -> None:
        """Non-blocking keyboard listener thread for initial pose command."""
        self.get_logger().info("[KEYBOARD] Listener thread running...")
        
        # Check if running in a terminal with stdin
        if not sys.stdin.isatty():
            self.get_logger().warning("[KEYBOARD] Not a TTY - keyboard listener disabled")
            return
        
        try:
            old_settings = termios.tcgetattr(sys.stdin)
        except Exception as e:
            self.get_logger().warning(f"[KEYBOARD] Failed to get terminal settings: {e}")
            return
        
        try:
            tty.setcbreak(sys.stdin.fileno())
            
            while self._keyboard_enabled and rclpy.ok():
                # Non-blocking check for key press
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    key = sys.stdin.read(1)
                    
                    if key == 'i' or key == 'I':
                        self.move_to_initial_pose()
                    elif key == 'q' or key == 'Q':
                        self.get_logger().info("[KEYBOARD] 'q' pressed -> Disabling keyboard listener")
                        self._keyboard_enabled = False
                        break
                        
        except Exception as e:
            self.get_logger().error(f"[KEYBOARD] Listener error: {e}")
        finally:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            except Exception:
                pass
            self.get_logger().info("[KEYBOARD] Listener thread ended")

    # Wrench callbacks
    def subscribe_wrench_single(self, idx: int, msg: WrenchStamped) -> None:
        try:
            if idx < len(self.force_sensor_map):
                finger = self.force_sensor_map[idx]
                fx = float(msg.wrench.force.x)
                fy = float(msg.wrench.force.y)
                fz = float(msg.wrench.force.z)
                if finger in self.measured_force:
                    # Store only fz for backward compatibility (used in force feedback)
                    self.measured_force[finger] = fz
                    self.has_force[finger] = True
                # Store full force vector for logging
                self.measured_force_vec[finger] = np.array([fx, fy, fz], dtype=float)
        except Exception as e:
            self.get_logger().warning(f"wrench 콜백 오류(idx={idx}): {e}")
    def subscribe_wrench_single_s1(self, msg: WrenchStamped) -> None:
        self.subscribe_wrench_single(0, msg)

    def subscribe_wrench_single_s2(self, msg: WrenchStamped) -> None:
        self.subscribe_wrench_single(1, msg)

    def subscribe_wrench_single_s3(self, msg: WrenchStamped) -> None:
        self.subscribe_wrench_single(2, msg)

    def subscribe_playback_status(self, msg: Bool) -> None:
        """Track demo playback status - reset desired positions when playback stops"""
        prev_status = self.demo_playback_active
        self.demo_playback_active = msg.data
        
        # When playback stops (True -> False), reset desired positions
        if prev_status and not self.demo_playback_active:
            for finger in ['th', 'if', 'mf']:
                self.fingers[finger].has_desired_pos = False
            self._logged_no_desired_pos_warning = False  # Allow new warning when restarted
            if not self._playback_status_logged:
                self.get_logger().info("[PLAYBACK] Demo stopped -> reset desired positions")
                self._playback_status_logged = True
            if self.stiffness_log_active:
                self._stop_stiffness_logging(reason="playback_inactive")
        elif not prev_status and self.demo_playback_active:
            if not self._playback_status_logged:
                self.get_logger().info("[PLAYBACK] Demo started")
            self._playback_status_logged = False
            if self.stiffness_logging_enabled and not self.stiffness_log_active:
                self._start_stiffness_logging()

    def subscribe_playback_stage(self, msg: UInt8) -> None:
        try:
            prev_stage = self.demo_playback_stage
            self.demo_playback_stage = int(msg.data)
            if not self.stiffness_logging_enabled:
                return
            if self.demo_playback_stage == 2 and not self.stiffness_log_active:
                self._start_stiffness_logging()
            elif self.demo_playback_stage != 2 and self.stiffness_log_active:
                self._stop_stiffness_logging(reason=f"stage_{prev_stage}_to_{self.demo_playback_stage}")
        except Exception as exc:
            self.get_logger().warning(f"[STAGE] subscribe error: {exc}")

    # Generic callbacks
    def subscribe_stiffness(self, msg: Float32MultiArray) -> None:
        try:
            if len(msg.data) >= 9:
                raw_k = np.array(msg.data[:9], dtype=float)

                # [DEBUG] Bypass all filtering - use raw value from policy node
                # Policy node already applies LP filter, so no additional filtering needed here
                # This helps debug whether step function comes from policy or controller
                self.filtered_stiffness = raw_k
                self.target_stiffness = raw_k
                self.has_stiffness = True
                
                # [ORIGINAL CODE - DISABLED FOR DEBUGGING]
                # The combination of rate limiting (max_k_change=1.0) and exponential
                # smoothing (alpha=0.1) was causing step-function behavior by:
                # 1. Limiting changes to 1 N/m per step (too restrictive)
                # 2. Only applying 10% of new value (too slow response)
                #
                # if not self.has_stiffness:
                #     self.filtered_stiffness = raw_k
                #     self.target_stiffness = raw_k
                #     self.has_stiffness = True
                #     return
                #
                # delta = raw_k - self.filtered_stiffness
                # delta = np.clip(delta, -self.max_k_change, self.max_k_change)
                # target_k_limited = self.filtered_stiffness + delta
                #
                # self.filtered_stiffness = (
                #     self.stiffness_alpha * target_k_limited
                #     + (1.0 - self.stiffness_alpha) * self.filtered_stiffness
                # )
                #
                # self.target_stiffness = self.filtered_stiffness
        except Exception as e:
            self.get_logger().warning(f"stiffness 콜백 오류: {e}")

    def subscribe_eccentricity(self, msg: Float32) -> None:
        try:
            self.current_eccentricity = float(msg.data)
            self.has_eccentricity = True
        except Exception as e:
            self.get_logger().warning(f"eccentricity 콜백 오류: {e}")
    
    def subscribe_eccentricity_smoothed(self, msg: Float32) -> None:
        try:
            self.current_eccentricity_smoothed = float(msg.data)
            self.has_eccentricity_smoothed = True
        except Exception as e:
            self.get_logger().warning(f"eccentricity_smoothed 콜백 오류: {e}")

    def subscribe_pwm(self, msg: Int32MultiArray) -> None:
        """Subscribe to PWM values from robot_controller_node"""
        try:
            if len(msg.data) >= 9:
                self.current_pwm = np.array(msg.data[:9], dtype=np.int32)
        except Exception as e:
            self.get_logger().warning(f"PWM 콜백 오류: {e}")

    def subscribe_joint_state(self, msg: JointState) -> None:
        try:
            n = min(len(msg.position), 9)
            if n > 0:
                for i in range(n):
                    self.current_qpos[i] = float(msg.position[i])
                if len(msg.velocity) >= n:
                    for i in range(n):
                        self.current_qvel[i] = float(msg.velocity[i])
                self.has_qpos = True
                self._joint_state_sub_counter += 1
                # print(f"[JointState subscribe #{self._joint_state_sub_counter}] pos={self.current_qpos[:3].tolist()} vel={self.current_qvel[:3].tolist()}")
                # self.get_logger().info(f"[JointState subscribe #{self._joint_state_sub_counter}] pos={self.current_qpos[:3].tolist()} vel={self.current_qvel[:3].tolist()}")
                if self.use_mujoco:
                    self._update_kinematics()
            else:
                self.get_logger().warning("JointState position 값이 비어있음!")
        except Exception as e:
            self.get_logger().warning(f"qpos 콜백 오류: {e}")

    def subscribe_desired_pose(self, finger: str, msg: PoseStamped) -> None:
        try:
            p = msg.pose.position
            self.desired_pos[finger] = np.array([p.x, p.y, p.z], dtype=float)
            self.fingers[finger].has_desired_pos = True
            
            # First reception log
            if not self._desired_ee_first_log[finger]:
                self._desired_ee_first_log[finger] = True
                self.get_logger().info(
                    f"[DESIRED_EE] First reception for {finger}: "
                    f"pos=[{p.x:.4f}, {p.y:.4f}, {p.z:.4f}]"
                )
            
            # # Periodic logging (every 5 seconds)
            # now = self.get_clock().now().nanoseconds / 1e9
            # if now - self._desired_ee_last_log[finger] >= 5.0:
            #     self._desired_ee_last_log[finger] = now
            #     current_ee = self.fingers[finger].ee_pos
            #     self.get_logger().info(
            #         f"[DESIRED_EE] {finger}: desired=[{p.x:.4f}, {p.y:.4f}, {p.z:.4f}] "
            #         f"current=[{current_ee[0]:.4f}, {current_ee[1]:.4f}, {current_ee[2]:.4f}]"
            #     )
        except Exception as e:
            self.get_logger().warning(f"desired pose 오류({finger}): {e}")
    def subscribe_desired_pose_if(self, msg: PoseStamped) -> None:
        self.subscribe_desired_pose("if", msg)
    def subscribe_desired_pose_mf(self, msg: PoseStamped) -> None:
        self.subscribe_desired_pose("mf", msg)
    def subscribe_desired_pose_th(self, msg: PoseStamped) -> None:
        self.subscribe_desired_pose("th", msg)
    def subscribe_desired_velocity(self, finger: str, msg: TwistStamped) -> None:
        try:
            v = msg.twist.linear
            self.desired_vel[finger] = np.array([v.x, v.y, v.z], dtype=float)
            self.fingers[finger].has_desired_vel = True
        except Exception as e:
            self.get_logger().warning(f"desired velocity 오류({finger}): {e}")
    def subscribe_desired_velocity_if(self, msg: TwistStamped) -> None:
        self.subscribe_desired_velocity("if", msg)
    def subscribe_desired_velocity_mf(self, msg: TwistStamped) -> None:
        self.subscribe_desired_velocity("mf", msg)
    def subscribe_desired_velocity_th(self, msg: TwistStamped) -> None:
        self.subscribe_desired_velocity("th", msg)
    def _update_kinematics(self) -> None:
        if not self.use_mujoco:
            return
        if self.mj_data is None or self.mj_model is None:
            if self._joint_state_sub_counter == 1:
                self.get_logger().error("[KINEMATICS] MuJoCo enabled but model/data is None! Check initialization logs.")
            return
        
        # Debug: log kinematics update call (first few times only)
        if self._joint_state_sub_counter <= 3:
            self.get_logger().info(
                f"[KINEMATICS] Update #{self._joint_state_sub_counter}: "
                f"qpos={self.current_qpos[:3]} (first 3 joints)"
            )
        
        try:
            units_per_rad = 4096.0 / (2.0 * math.pi)
            for f_idx, finger in enumerate(["thumb", "index", "middle"]):
                for j in range(3):
                    key = f"{finger}_{j}"
                    if key in self.mj_qpos_adr:
                        q_idx = f_idx * 3 + j
                        # Convert units → radians: (units - bias) / units_per_rad + offset, then subtract offset for MuJoCo
                        bias = 1000.0 if q_idx in [0, 3, 6] else 2000.0
                        offset = 1.57 if q_idx in [0, 3, 6] else 3.14
                        rad_val = (self.current_qpos[q_idx] - bias) / units_per_rad + offset
                        self.mj_data.qpos[self.mj_qpos_adr[key]] = rad_val - offset  # type: ignore
                        # Velocity: units/s → rad/s
                        self.mj_data.qvel[self.mj_qpos_adr[key]] = self.current_qvel[q_idx] / units_per_rad  # type: ignore
            
            # Debug: log qpos before mj_forward (first time only)
            if self._joint_state_sub_counter == 1:
                self.get_logger().info(
                    f"[KINEMATICS] Before mj_forward: mj_data.qpos={self.mj_data.qpos[:]}  "  # type: ignore
                    f"shape={self.mj_data.qpos.shape}"  # type: ignore
                )
            
            mj.mj_forward(self.mj_model, self.mj_data)  # type: ignore
            
            # Debug: log after mj_forward (first time only)
            if self._joint_state_sub_counter == 1:
                self.get_logger().info("[KINEMATICS] mj_forward completed successfully")
            for finger, site_name in SITE_NAMES.items():
                try:
                    sid = mj.mj_name2id(self.mj_model, mj.mjtObj.mjOBJ_SITE, site_name)  # type: ignore
                    if sid >= 0:
                        pos = self.mj_data.site_xpos[sid].copy()  # type: ignore
                        self.fingers[finger].ee_pos = pos
                        # Diagnostic: log first EE position update
                        if self._joint_state_sub_counter == 1:
                            self.get_logger().info(f"[EE_POS] {finger} first update: {pos}")
                        hist = self.ee_pos_history[finger]; hist.append(pos)
                        if len(hist) > self.velocity_window: hist.pop(0)
                        if len(hist) >= 2:
                            dt = 1.0 / self.rate_hz
                            vel_raw = (hist[-1] - hist[-2]) / dt
                            alpha = 0.3
                            self.fingers[finger].ee_vel = alpha * vel_raw + (1 - alpha) * self.fingers[finger].ee_vel
                        self.fingers[finger].jacobian = self._compute_jacobian(finger, sid)
                except Exception:
                    pass
            # Render desired EE positions as red spheres in viewer
            if self._mj_viewer is not None:
                try:
                    if self._mj_viewer.is_running():  # type: ignore[attr-defined]
                        # Reset user scene geometry count
                        self._mj_viewer.user_scn.ngeom = 0  # type: ignore[attr-defined]
                        # Add desired positions as visual markers
                        rendered_count = 0
                        # Finger-specific colors: mf=[0.3,1,0.3,1], th=[0.3,0.3,1,1], if=[1,0.3,0.3,1]
                        finger_colors = {
                            'mf': [0.3, 1.0, 0.3, 1.0],  # Green-ish
                            'th': [0.3, 0.3, 1.0, 1.0],  # Blue-ish
                            'if': [1.0, 0.3, 0.3, 1.0],  # Red-ish
                        }
                        for finger in ['th', 'if', 'mf']:
                            if self.fingers[finger].has_desired_pos:
                                pos = self.desired_pos[finger]
                                geom_idx = self._mj_viewer.user_scn.ngeom  # type: ignore[attr-defined]
                                if geom_idx < self._mj_viewer.user_scn.maxgeom:  # type: ignore[attr-defined]
                                    geom = self._mj_viewer.user_scn.geoms[geom_idx]  # type: ignore[attr-defined]
                                    geom.type = mj.mjtGeom.mjGEOM_SPHERE  # type: ignore
                                    geom.size[:] = [0.01, 0.01, 0.01]  # 3cm radius - much larger
                                    geom.pos[:] = pos
                                    geom.rgba[:] = finger_colors[finger]  # Finger-specific color
                                    geom.mat[:] = np.eye(3).reshape(3, 3)  # 3x3 identity matrix
                                    self._mj_viewer.user_scn.ngeom += 1  # type: ignore[attr-defined]
                                    rendered_count += 1
                        self._mj_viewer.sync()  # type: ignore[attr-defined]
                except Exception as e:
                    if self._joint_state_sub_counter % 100 == 1:
                        self.get_logger().warning(f"Viewer rendering error: {e}")
            # EE pose는 별도 타이머에서 publish됨
        except Exception as e:
            self.get_logger().warning(f"kinematics 업데이트 오류: {e}")
    def _compute_jacobian(self, finger: str, site_id: int) -> Optional[np.ndarray]:
        if not self.use_mujoco or self.mj_data is None: return None
        try:
            jacp = np.zeros((3, self.mj_model.nv))  # type: ignore
            jacr = np.zeros((3, self.mj_model.nv))  # type: ignore
            mj.mj_jacSite(self.mj_model, self.mj_data, jacp, jacr, site_id)  # type: ignore
            
            # Debug: log full jacp matrix (only first time per finger)
            if not hasattr(self, '_jacobian_logged'):
                self._jacobian_logged = set()
            if finger not in self._jacobian_logged:
                jacp_norm = np.linalg.norm(jacp)
                self.get_logger().info(
                    f"[JAC_FULL] {finger}: jacp_norm={jacp_norm:.6f}, "
                    f"jacp.shape={jacp.shape}, nv={self.mj_model.nv}\n"  # type: ignore
                    f"jacp=\n{jacp}"
                )
                self._jacobian_logged.add(finger)
            
            # FIX: MuJoCo joint order is [if, mf, th] not [th, if, mf]!
            # Based on actual jacp output:
            # - if (index): columns 0-2
            # - mf (middle): columns 3-5  
            # - th (thumb): columns 6-8
            fmap = {"if": 0, "mf": 1, "th": 2}
            start = fmap[finger] * 3; end = start + 3
            J_extracted = jacp[:, start:end]
            
            # Debug: log extracted Jacobian
            if finger not in getattr(self, '_jacobian_extract_logged', set()):
                if not hasattr(self, '_jacobian_extract_logged'):
                    self._jacobian_extract_logged = set()
                self.get_logger().info(
                    f"[JAC_EXTRACT] {finger}: start={start}, end={end}, "
                    f"J_extracted=\n{J_extracted}"
                )
                self._jacobian_extract_logged.add(finger)
            
            return J_extracted
        except Exception as e:
            self.get_logger().warning(f"Jacobian 계산 오류({finger}): {e}")
            return None
    def _compute_damping(self, K_diag: np.ndarray) -> np.ndarray:
        D = np.zeros_like(K_diag)
        for i, k in enumerate(K_diag):
            if k > 0: D[i] = 2.0 * self.damping_ratio * math.sqrt(self.virtual_mass * k)
        return D
    def _compute_torques(self) -> np.ndarray:
        torques = np.zeros(9)
        if not (self.has_stiffness and self.has_qpos):
            if not self._logged_no_stiffness_warning and self._log_counter > 0:
                self.get_logger().warning("[INIT] Waiting for stiffness and joint state data...")
                self._logged_no_stiffness_warning = True
            return torques
        
        # If no desired positions received yet, use current EE positions as desired (zero error = zero torque)
        if not any(f.has_desired_pos for f in self.fingers.values()):
            if self._use_current_as_initial_desired:
                # Use current position as desired -> position error = 0 -> torque = 0
                for finger in ['th', 'if', 'mf']:
                    if self.fingers[finger].ee_pos is not None:
                        self.desired_pos[finger] = self.fingers[finger].ee_pos.copy()
                if not self._logged_no_desired_pos_warning:
                    self.get_logger().info("[TORQUE] No desired EE positions -> using current positions (zero torque)")
                    self._logged_no_desired_pos_warning = True
            else:
                # Log once only with detailed status
                if not self._logged_no_desired_pos_warning:
                    missing = [f for f in ['th', 'if', 'mf'] if not self.fingers[f].has_desired_pos]
                    self.get_logger().warning(f"[INIT] Waiting for desired EE positions (missing: {missing})")
                    self._logged_no_desired_pos_warning = True
                return torques
        
        finger_index_map = {"th": 0, "if": 1, "mf": 2}
        for finger, f_idx in finger_index_map.items():
            if not self.fingers[finger].has_desired_pos:
                # Log missing desired position periodically
                if self._log_counter % int(self.rate_hz) == 0:
                    self.get_logger().warning(f"[TORQUE] Skipping {finger}: no desired_pos")
                continue
            k_start = f_idx * 3
            k_end = k_start + 3
            K_vec = self.target_stiffness[k_start:k_end] * self.stiffness_scale  # Apply stiffness scaling
            D_vec = self._compute_damping(K_vec)
            pos_err = self.desired_pos[finger] - self.fingers[finger].ee_pos
            vel_err = self.desired_vel[finger] - self.fingers[finger].ee_vel
            
            # # Debug: log position error and force calculation
            # if self._log_counter % int(self.rate_hz) == 0:
            #     pos_err_norm = np.linalg.norm(pos_err)
            #     self.get_logger().info(
            #         f"[TORQUE_DEBUG] {finger}: pos_err={pos_err_norm*1000:.1f}mm "
            #         f"K_vec=[{K_vec[0]:.1f},{K_vec[1]:.1f},{K_vec[2]:.1f}] "
            #         f"D_vec=[{D_vec[0]:.2f},{D_vec[1]:.2f},{D_vec[2]:.2f}]"
            #     )
            
            F = K_vec * pos_err + D_vec * vel_err
            
            # # Debug: log force vector
            # if self._log_counter % int(self.rate_hz) == 0:
            #     self.get_logger().info(
            #         f"[TORQUE_DEBUG] {finger}: F=[{F[0]:.3f},{F[1]:.3f},{F[2]:.3f}] "
            #         f"pos_err=[{pos_err[0]*1000:.1f},{pos_err[1]*1000:.1f},{pos_err[2]*1000:.1f}]mm"
            #     )
            J = self.fingers[finger].jacobian
            
            # Debug: check Jacobian
            if J is None:
                if self._log_counter % int(self.rate_hz) == 0:
                    self.get_logger().warning(f"[TORQUE] {finger}: Jacobian is None!")
                continue  # Skip this finger if no Jacobian
            
            # # Debug: log Jacobian values
            # if self._log_counter % int(self.rate_hz) == 0:
            #     J_norm = np.linalg.norm(J)
            #     self.get_logger().info(
            #         f"[TORQUE_DEBUG] {finger}: J_norm={J_norm:.6f}, "
            #         f"J=\n{J}"
            #     )
            
            if self.use_mujoco:
                tau = J.T @ F
                # tau = np.zeros_like(tau) + 1
                
                # # Debug: log tau after Jacobian transform
                # if self._log_counter % int(self.rate_hz) == 0:
                #     self.get_logger().info(
                #         f"[TORQUE_DEBUG] {finger}: tau_raw=[{tau[0]:.3f},{tau[1]:.3f},{tau[2]:.3f}] "
                #         f"J.shape={J.shape}"
                #     )
                
                if self.enable_force_fb and self.has_force.get(finger, False):
                    F_meas = np.zeros(3)
                    F_meas[2] = self.measured_force[finger] # FZ 값 (그래프상 음수)
                    tau_meas = J.T @ F_meas
                    
                    for i in range(3):
                        joint_idx = k_start + i
                        target_tau_i = tau[i]
                        
                        # ---------------------------------------------------------
                        # [최종 수정] Stable Constant Booster (상수 기반 쥐는 힘 강화)
                        # ---------------------------------------------------------
                        # 문제: 기존 (Force * Gain) 방식은 센서 노이즈를 증폭시켜 진동(발작)을 유발함.
                        # 해결: 힘이 감지되면 '계산'하지 말고 무조건 '상수(고정값)'를 더해서 꾹 누르게 함.
                        
                        # 1. 감지 (FZ는 쥘 때 음수)
                        current_force_z = F_meas[2]
                        is_grasping = (current_force_z < -0.15)  # 0.15N 이상 힘이 걸리면
                        is_closing = (target_tau_i < 0)           # 모터도 쥐는 방향이면

                        if is_grasping and is_closing:
                            # [핵심 변경 사항]
                            # 기존: squeeze_torque = F_meas[2] * 2.0 (센서 흔들리면 토크도 흔들림 -> 진동)
                            # 변경: 기본적으로 -0.5Nm를 깔고 감 (센서가 흔들려도 토크는 일정함 -> 안정적)
                            
                            base_squeeze = -0.5   # [튜닝 포인트] 이 값을 늘리면(-0.8 등) 더 꽉 쥡니다.
                            prop_squeeze = current_force_z * 0.2 # 비례항은 2.0 -> 0.2로 대폭 낮춤 (단순 보조용)
                            
                            # 최종 추가 토크 계산
                            squeeze_total = base_squeeze + prop_squeeze
                            
                            # 목표 토크(target_tau_i)에 추가 토크를 더함 (더 음수 쪽으로)
                            fake_target = target_tau_i + squeeze_total
                            
                            tau_err = fake_target - tau_meas[i]
                            
                            # 디버깅: 부스터가 얼마나 힘을 보태는지 확인 (1번 조인트만)
                            if self._log_counter % int(self.rate_hz) == 0 and i == 0:
                                self.get_logger().info(f"[GRIP] F={current_force_z:.2f} | Added={squeeze_total:.2f}")

                        else:
                            # 접촉 없거나 펴는 방향이면 원래대로 계산
                            tau_err = target_tau_i - tau_meas[i]
                        # ---------------------------------------------------------
                        
                        # 적분 제어 (I-term) 적용
                        self.force_int_err[joint_idx] += tau_err * (1.0 / self.rate_hz)
                        
                        # [안전장치] 적분항 클램핑 (-10.0 ~ 10.0)
                        self.force_int_err[joint_idx] = np.clip(self.force_int_err[joint_idx], -10.0, 10.0)
                        
                        # 최종 토크에 P+I 추가
                        tau[i] += self.kp_force * tau_err + self.ki_force * self.force_int_err[joint_idx]
                
                # Assign tau to torques array
                for i in range(3):
                    torques[k_start + i] = tau[i]
                    
                # # Debug: verify assignment per finger (루프 안에서 출력)
                # if self._log_counter % int(self.rate_hz) == 0:
                #     self.get_logger().info(
                #         f"[TORQUE_DEBUG] {finger} assigned: "
                #         f"[{torques[k_start]:.3f},{torques[k_start+1]:.3f},{torques[k_start+2]:.3f}]"
                #     )
        
        # 루프 종료 후 전체 torques 출력
        if self._log_counter % int(self.rate_hz) == 0:
            self.get_logger().info(
                f"[TORQUE_FULL] "
                f"th=[{torques[0]:.3f},{torques[1]:.3f},{torques[2]:.3f}] "
                f"if=[{torques[3]:.3f},{torques[4]:.3f},{torques[5]:.3f}] "
                f"mf=[{torques[6]:.3f},{torques[7]:.3f},{torques[8]:.3f}]"
            )
        return torques

    # def _torques_to_current_units(self, torques: np.ndarray) -> List[int]:
    #     currents_mA = np.where(CURRENT_TO_TORQUE > 0.0, torques / CURRENT_TO_TORQUE, 0.0)
    #     units = currents_mA / CURRENT_UNIT
    #     # Apply per-joint current_units_scale before clipping (element-wise multiplication)
    #     # units = units * self.current_units_scale
    #     # units = torques * self.current_units_scale
    #     # Apply asymmetric clipping: positive and negative limits can differ
    #     units = np.clip(units, -self.max_current_units_neg, self.max_current_units_pos)
    #     # self.get_logger().info(
    #     #     f"[TORQUE_DEBUG] current_units ,{self.current_units_scale} "
    #     #     f"th=[{currents_mA[0]:.3f},{currents_mA[1]:.3f},{currents_mA[2]:.3f}] "
    #     #     f"if=[{currents_mA[3]:.3f},{currents_mA[4]:.3f},{currents_mA[5]:.3f}] "
    #     #     f"mf=[{currents_mA[6]:.3f},{currents_mA[7]:.3f},{currents_mA[8]:.3f}]"
    #     # )
    #     return [int(round(u)) for u in units]
    def _torques_to_current_units(self, torques: np.ndarray) -> tuple[List[int], List[int]]:
        """
        Convert torques to current units.
        
        Returns:
            tuple: (clipped_units, raw_units_before_clipping) - both as List[int]
        """
        # 1) 토크 → 전류(mA)
        #    CURRENT_TO_TORQUE: N·m / mA (약 1.78e-3)
        #    torque [Nm] / CURRENT_TO_TORQUE [Nm/mA] = current [mA]
        currents_mA = np.where(
            CURRENT_TO_TORQUE > 0.0,
            torques / CURRENT_TO_TORQUE,
            0.0,
        )

        # 2) 전류(mA) → current unit (1 unit ≈ 2.69 mA)
        units = currents_mA / CURRENT_UNIT
        
        # 3) per-joint 튜닝 스케일 적용
        units = units * self.current_units_scale
        
        # Store raw units before clipping (as int list)
        units_raw = [int(round(u)) for u in units]

        # 4) 안전 클리핑
        units = np.clip(units, -self.max_current_units_neg, self.max_current_units_pos)

        return [int(round(u)) for u in units], units_raw

    def _publish_zero_current(self, reason: Optional[str] = None, level: str = "warn") -> None:
            """Send zero current command to all motors with optional throttled logging."""
            zero_msg = Int32MultiArray()
            zero_msg.data = [1] * 9 
            self.current_pub.publish(zero_msg)
            
            if reason and self._log_counter % int(self.rate_hz) == 0:
                # 수정된 부분: 동적 할당 대신 if-else로 물리적인 코드 라인을 분리함
                if level == "warn":
                    self.get_logger().warning(reason)
                else:
                    self.get_logger().info(reason)

    def _filter_torques(self, torques: np.ndarray) -> np.ndarray:
        filtered = self.torque_filter_alpha * torques + (1.0 - self.torque_filter_alpha) * self.last_torques
        self.last_torques = filtered.copy(); return filtered

    def _start_stiffness_logging(self) -> None:
        if not self.stiffness_logging_enabled:
            return
        self.stiffness_log_data = []
        self.eccentricity_log_data = []
        self.eccentricity_smoothed_log_data = []
        self.force_log_data = []
        self.ee_pos_log_data = []
        self.desired_pos_log_data = []
        self.torque_log_data = []
        self.current_units_log_data = []
        self.current_units_raw_log_data = []
        self.pwm_log_data = []  # Clear PWM log data
        self.stiffness_log_active = True
        self.stiffness_log_start_time = self.get_clock().now().nanoseconds / 1e9
        self._stiffness_log_session_id += 1
        self.get_logger().warning(
            f"[STIFFNESS_LOG] *** Recording STARTED *** (session #{self._stiffness_log_session_id}) - samples will be saved to {self.stiffness_log_dir}"
        )

    def _stop_stiffness_logging(self, reason: str = "stage_exit") -> None:
        if not self.stiffness_log_active:
            return
        self.stiffness_log_active = False
        sample_count = len(self.stiffness_log_data)
        ecc_count = len(self.eccentricity_log_data)
        force_count = len(self.force_log_data)
        ee_pos_count = len(self.ee_pos_log_data)
        torque_count = len(self.torque_log_data)
        pwm_count = len(self.pwm_log_data)
        if sample_count == 0 and ecc_count == 0 and force_count == 0 and ee_pos_count == 0 and torque_count == 0:
            self.get_logger().info(
                f"[STIFFNESS_LOG] Recording stopped ({reason}) but no samples captured"
            )
            return
        self.get_logger().info(
            f"[STIFFNESS_LOG] Recording stopped ({reason}), stiffness={sample_count}, ecc={ecc_count}, force={force_count}, ee_pos={ee_pos_count}, torque={torque_count}, pwm={pwm_count}"
        )
        
        # Create session folder with timestamp
        import os as _os_module
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pid = _os_module.getpid()
        session_folder = os.path.join(
            self.stiffness_log_dir,
            f"session_{timestamp}_pid{pid}_{self._stiffness_log_session_id:02d}"
        )
        os.makedirs(session_folder, exist_ok=True)
        self.get_logger().info(f"[STIFFNESS_LOG] Session folder created: {session_folder}")
        
        # Save all 5 plots and CSV data into session folder
        self._save_stiffness_eccentricity_plot(session_folder)
        self._save_force_plot(session_folder)
        self._save_ee_position_plot(session_folder)
        self._save_torque_current_plot(session_folder)
        self._save_pwm_plot(session_folder)
        self._save_all_csv_data(session_folder)
        
        # Clear log data
        self.stiffness_log_data = []
        self.eccentricity_log_data = []
        self.eccentricity_smoothed_log_data = []
        self.force_log_data = []
        self.ee_pos_log_data = []
        self.desired_pos_log_data = []
        self.torque_log_data = []
        self.current_units_log_data = []
        self.current_units_raw_log_data = []
        self.pwm_log_data = []

    def _save_stiffness_eccentricity_plot(self, session_folder: str) -> None:
        """Save stiffness + eccentricity plot (2 subplots)"""
        if not self.stiffness_log_data:
            return
        try:
            import matplotlib.pyplot as plt  # type: ignore
        except ImportError:
            self.get_logger().warning("[STIFFNESS_LOG] Matplotlib not available - skip plot")
            return
        
        times_stiff = np.array([entry[0] for entry in self.stiffness_log_data], dtype=float)
        values = np.stack([entry[1] for entry in self.stiffness_log_data], axis=0)
        component_labels = [
            "th_x", "th_y", "th_z",
            "if_x", "if_y", "if_z",
            "mf_x", "mf_y", "mf_z",
        ]
        
        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # Subplot 1: Stiffness
        for idx in range(values.shape[1]):
            ax1.plot(times_stiff, values[:, idx], label=component_labels[idx])
        ax1.set_ylabel("Stiffness (N/m)")
        ax1.set_title("stiffness P(k|obs) + eccentricity")
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="upper right", ncol=3, fontsize=8)
        
        # Subplot 2: Eccentricity (raw and smoothed)
        if self.eccentricity_log_data:
            times_ecc = np.array([entry[0] for entry in self.eccentricity_log_data], dtype=float)
            ecc_vals = np.array([entry[1] for entry in self.eccentricity_log_data], dtype=float)
            ax2.plot(times_ecc, ecc_vals, color='red', linewidth=1.0, alpha=0.6, label='Eccentricity (raw)')
        if self.eccentricity_smoothed_log_data:
            times_ecc_sm = np.array([entry[0] for entry in self.eccentricity_smoothed_log_data], dtype=float)
            ecc_vals_sm = np.array([entry[1] for entry in self.eccentricity_smoothed_log_data], dtype=float)
            ax2.plot(times_ecc_sm, ecc_vals_sm, color='blue', linewidth=1.5, label='Eccentricity (smoothed)')
        if self.eccentricity_log_data or self.eccentricity_smoothed_log_data:
            ax2.set_ylabel("Eccentricity")
            ax2.set_xlabel("Time (s)")
            ax2.grid(True, alpha=0.3)
            ax2.legend(loc="upper right")
        else:
            ax2.text(0.5, 0.5, "No eccentricity data", ha='center', va='center', transform=ax2.transAxes)
            ax2.set_ylabel("Eccentricity")
            ax2.set_xlabel("Time (s)")
        
        filename = os.path.join(session_folder, "01_stiffness_eccentricity.png")
        try:
            plt.tight_layout()
            plt.savefig(filename, dpi=200)
            self.get_logger().warning(
                f"[STIFFNESS_LOG] *** SAVED PLOT 1/5 *** Stiffness+Eccentricity ({len(times_stiff)} samples) -> {filename}"
            )
        except Exception as exc:
            self.get_logger().error(f"[STIFFNESS_LOG] Failed to save stiffness plot: {exc}")
            import traceback
            self.get_logger().error(f"[STIFFNESS_LOG] Traceback:\n{traceback.format_exc()}")
        finally:
            plt.close()

    def _save_force_plot(self, session_folder: str) -> None:
        """Save force sensor plot (3 fingers x 3 axes = 9 subplots in 3x3 grid)"""
        if not self.force_log_data:
            self.get_logger().info("[FORCE_LOG] No force data to plot")
            return
        try:
            import matplotlib.pyplot as plt  # type: ignore
        except ImportError:
            self.get_logger().warning("[FORCE_LOG] Matplotlib not available - skip plot")
            return
        
        times_force = np.array([entry[0] for entry in self.force_log_data], dtype=float)
        
        # Extract force data for each finger and axis
        force_data: Dict[str, Dict[str, List[float]]] = {
            'th': {'fx': [], 'fy': [], 'fz': []},
            'if': {'fx': [], 'fy': [], 'fz': []},
            'mf': {'fx': [], 'fy': [], 'fz': []}
        }
        
        for _, force_dict in self.force_log_data:
            for finger in ['th', 'if', 'mf']:
                if finger in force_dict:
                    vec = force_dict[finger]
                    force_data[finger]['fx'].append(vec[0])
                    force_data[finger]['fy'].append(vec[1])
                    force_data[finger]['fz'].append(vec[2])
                else:
                    force_data[finger]['fx'].append(0.0)
                    force_data[finger]['fy'].append(0.0)
                    force_data[finger]['fz'].append(0.0)
        
        # Convert to numpy arrays
        force_arrays: Dict[str, Dict[str, np.ndarray]] = {}
        for finger in ['th', 'if', 'mf']:
            force_arrays[finger] = {}
            for axis in ['fx', 'fy', 'fz']:
                force_arrays[finger][axis] = np.array(force_data[finger][axis], dtype=float)
        
        # Create 3x3 grid: rows=fingers, columns=axes
        fig, axes = plt.subplots(3, 3, figsize=(15, 10), sharex=True)
        fig.suptitle("Force Sensors (3 fingers x 3 axes)", fontsize=14)
        
        finger_names = {'th': 'Thumb', 'if': 'Index', 'mf': 'Middle'}
        axis_names = ['fx', 'fy', 'fz']
        colors = {'fx': 'blue', 'fy': 'green', 'fz': 'red'}
        
        for row, finger in enumerate(['th', 'if', 'mf']):
            for col, axis in enumerate(axis_names):
                ax = axes[row, col]
                ax.plot(times_force, force_arrays[finger][axis], color=colors[axis], linewidth=1.2)
                ax.set_ylabel(f"{finger_names[finger]} (N)")
                ax.grid(True, alpha=0.3)
                if row == 0:
                    ax.set_title(f"{axis.upper()}")
                if row == 2:
                    ax.set_xlabel("Time (s)")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        import os as _os_module
        pid = _os_module.getpid()
        filename = os.path.join(session_folder, "02_force_sensors.png")
        try:
            plt.tight_layout()
            plt.savefig(filename, dpi=200)
            self.get_logger().warning(
                f"[FORCE_LOG] *** SAVED PLOT 2/5 *** Force sensors ({len(times_force)} samples) -> {filename}"
            )
        except Exception as exc:
            self.get_logger().error(f"[FORCE_LOG] Failed to save force plot: {exc}")
            import traceback
            self.get_logger().error(f"[FORCE_LOG] Traceback:\n{traceback.format_exc()}")
        finally:
            plt.close()

    def _save_ee_position_plot(self, session_folder: str) -> None:
        """Save EE position tracking plot: desired vs actual (3 fingers x 3 axes = 9 subplots)"""
        if not self.ee_pos_log_data or not self.desired_pos_log_data:
            self.get_logger().info("[EE_POS_LOG] No EE position data to plot")
            return
        try:
            import matplotlib.pyplot as plt  # type: ignore
        except ImportError:
            self.get_logger().warning("[EE_POS_LOG] Matplotlib not available - skip plot")
            return
        
        times_ee = np.array([entry[0] for entry in self.ee_pos_log_data], dtype=float)
        times_desired = np.array([entry[0] for entry in self.desired_pos_log_data], dtype=float)
        
        # Extract position data for each finger and axis
        ee_data: Dict[str, Dict[str, List[float]]] = {
            'th': {'x': [], 'y': [], 'z': []},
            'if': {'x': [], 'y': [], 'z': []},
            'mf': {'x': [], 'y': [], 'z': []}
        }
        desired_data: Dict[str, Dict[str, List[float]]] = {
            'th': {'x': [], 'y': [], 'z': []},
            'if': {'x': [], 'y': [], 'z': []},
            'mf': {'x': [], 'y': [], 'z': []}
        }
        
        for _, pos_dict in self.ee_pos_log_data:
            for finger in ['th', 'if', 'mf']:
                if finger in pos_dict:
                    vec = pos_dict[finger]
                    ee_data[finger]['x'].append(vec[0])
                    ee_data[finger]['y'].append(vec[1])
                    ee_data[finger]['z'].append(vec[2])
                else:
                    ee_data[finger]['x'].append(0.0)
                    ee_data[finger]['y'].append(0.0)
                    ee_data[finger]['z'].append(0.0)
        
        for _, pos_dict in self.desired_pos_log_data:
            for finger in ['th', 'if', 'mf']:
                if finger in pos_dict:
                    vec = pos_dict[finger]
                    desired_data[finger]['x'].append(vec[0])
                    desired_data[finger]['y'].append(vec[1])
                    desired_data[finger]['z'].append(vec[2])
                else:
                    desired_data[finger]['x'].append(0.0)
                    desired_data[finger]['y'].append(0.0)
                    desired_data[finger]['z'].append(0.0)
        
        # Convert to numpy arrays
        ee_arrays: Dict[str, Dict[str, np.ndarray]] = {}
        desired_arrays: Dict[str, Dict[str, np.ndarray]] = {}
        for finger in ['th', 'if', 'mf']:
            ee_arrays[finger] = {}
            desired_arrays[finger] = {}
            for axis in ['x', 'y', 'z']:
                ee_arrays[finger][axis] = np.array(ee_data[finger][axis], dtype=float)
                desired_arrays[finger][axis] = np.array(desired_data[finger][axis], dtype=float)
        
        # Create 3x3 grid: rows=fingers, columns=axes
        fig, axes = plt.subplots(3, 3, figsize=(15, 10), sharex=True)
        fig.suptitle("EE Position Tracking: Desired vs Actual", fontsize=14)
        
        finger_names = {'th': 'Thumb', 'if': 'Index', 'mf': 'Middle'}
        axis_names = ['x', 'y', 'z']
        
        for row, finger in enumerate(['th', 'if', 'mf']):
            for col, axis in enumerate(axis_names):
                ax = axes[row, col]
                # Plot desired position (dashed line)
                ax.plot(times_desired, desired_arrays[finger][axis], 
                       color='red', linestyle='--', linewidth=1.5, label='Desired', alpha=0.7)
                # Plot actual position (solid line)
                ax.plot(times_ee, ee_arrays[finger][axis], 
                       color='blue', linestyle='-', linewidth=1.2, label='Actual')
                ax.set_ylabel(f"{finger_names[finger]} (m)")
                ax.grid(True, alpha=0.3)
                if row == 0:
                    ax.set_title(f"{axis.upper()}")
                if row == 2:
                    ax.set_xlabel("Time (s)")
                if row == 0 and col == 2:
                    ax.legend(loc='upper right', fontsize=8)
        
        filename = os.path.join(session_folder, "03_ee_position_tracking.png")
        try:
            plt.tight_layout()
            plt.savefig(filename, dpi=200)
            self.get_logger().warning(
                f"[EE_POS_LOG] *** SAVED PLOT 3/5 *** EE Position Tracking ({len(times_ee)} samples) -> {filename}"
            )
        except Exception as exc:
            self.get_logger().error(f"[EE_POS_LOG] Failed to save EE position plot: {exc}")
            import traceback
            self.get_logger().error(f"[EE_POS_LOG] Traceback:\n{traceback.format_exc()}")
        finally:
            plt.close()

    def _save_torque_current_plot(self, session_folder: str) -> None:
        """Save current units plot (PRIMARY) with torque as secondary reference.
        Current units are the actual motor commands - int values that matter for robot control."""
        if not self.torque_log_data or not self.current_units_log_data:
            self.get_logger().info("[TORQUE_LOG] No torque/current data to plot")
            return
        try:
            import matplotlib.pyplot as plt  # type: ignore
        except ImportError:
            self.get_logger().warning("[TORQUE_LOG] Matplotlib not available - skip plot")
            return
        
        times_torque = np.array([entry[0] for entry in self.torque_log_data], dtype=float)
        times_current = np.array([entry[0] for entry in self.current_units_log_data], dtype=float)
        times_current_raw = np.array([entry[0] for entry in self.current_units_raw_log_data], dtype=float) if self.current_units_raw_log_data else times_current
        
        # Extract torque and current data (9 joints each) - CURRENT AS INT!
        torques_array = np.stack([entry[1] for entry in self.torque_log_data], axis=0)  # shape: (N, 9)
        currents_array = np.stack([entry[1] for entry in self.current_units_log_data], axis=0).astype(int)  # shape: (N, 9), int!
        currents_raw_array = np.stack([entry[1] for entry in self.current_units_raw_log_data], axis=0).astype(int) if self.current_units_raw_log_data else currents_array
        
        # Log current stats
        self.get_logger().info(f"[CURRENT_STATS] Clipped: min={currents_array.min()}, max={currents_array.max()}, mean={currents_array.mean():.1f}")
        self.get_logger().info(f"[CURRENT_STATS] Raw: min={currents_raw_array.min()}, max={currents_raw_array.max()}, mean={currents_raw_array.mean():.1f}")
        
        # Create 3x3 grid for 9 joints - CURRENT UNITS AS PRIMARY AXIS
        fig, axes = plt.subplots(3, 3, figsize=(16, 11), sharex=True)
        fig.suptitle("Current Units (Motor Commands) - Integer Values\n"
                     "Red: Clipped Current (sent to motor) | Orange: Raw Current (before clipping) | Blue dashed: Torque (Nm)", 
                     fontsize=14, fontweight='bold')
        
        joint_labels = [
            'TH_J0', 'TH_J1', 'TH_J2',
            'IF_J0', 'IF_J1', 'IF_J2',
            'MF_J0', 'MF_J1', 'MF_J2'
        ]
        
        for joint_idx in range(9):
            row = joint_idx // 3
            col = joint_idx % 3
            ax = axes[row, col]
            
            # PRIMARY Y-AXIS: Current Units (int) - what actually goes to the motor!
            # Plot raw current (before clipping) - orange
            if self.current_units_raw_log_data:
                ax.plot(times_current_raw, currents_raw_array[:, joint_idx], 
                        color='orange', linewidth=1.5, label=f'Raw ({currents_raw_array[:, joint_idx].min()} ~ {currents_raw_array[:, joint_idx].max()})', alpha=0.7)
            # Plot clipped current - red solid (thicker, more prominent)
            ax.plot(times_current, currents_array[:, joint_idx], 
                   color='red', linewidth=2.0, label=f'Clipped ({currents_array[:, joint_idx].min()} ~ {currents_array[:, joint_idx].max()})')
            ax.set_ylabel("Current Units (int)", color='red', fontweight='bold')
            ax.tick_params(axis='y', labelcolor='red')
            from matplotlib.ticker import MaxNLocator
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))  # Force integer ticks
            ax.grid(True, alpha=0.3)
            
            # Add horizontal line at 0
            ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
            
            # SECONDARY Y-AXIS: Torque (for reference only)
            ax2 = ax.twinx()
            ax2.plot(times_torque, torques_array[:, joint_idx], 
                    color='blue', linewidth=1.0, linestyle='--', label='Torque', alpha=0.5)
            ax2.set_ylabel("Torque (Nm)", color='blue', alpha=0.6)
            ax2.tick_params(axis='y', labelcolor='blue', labelsize=8)
            
            # Title with current stats
            curr_mean = currents_array[:, joint_idx].mean()
            curr_std = currents_array[:, joint_idx].std()
            ax.set_title(f"{joint_labels[joint_idx]} | μ={curr_mean:.0f}, σ={curr_std:.0f}", fontsize=10, fontweight='bold')
            
            if row == 2:
                ax.set_xlabel("Time (s)")
            
            # Add legend to first subplot
            if joint_idx == 0:
                lines1, labels1 = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=7)
        
        filename = os.path.join(session_folder, "04_torque_current.png")
        try:
            plt.tight_layout()
            plt.savefig(filename, dpi=200)
            self.get_logger().warning(
                f"[TORQUE_LOG] *** SAVED PLOT 4/5 *** Torque vs Current ({len(times_torque)} samples) -> {filename}"
            )
        except Exception as exc:
            self.get_logger().error(f"[TORQUE_LOG] Failed to save torque/current plot: {exc}")
            import traceback
            self.get_logger().error(f"[TORQUE_LOG] Traceback:\n{traceback.format_exc()}")
        finally:
            plt.close()

    def _save_pwm_plot(self, session_folder: str) -> None:
        """Save PWM monitoring plot (9 joints in 3x3 grid)"""
        if not self.pwm_log_data:
            self.get_logger().info("[PWM_LOG] No PWM data to plot")
            return
        try:
            import matplotlib.pyplot as plt  # type: ignore
        except ImportError:
            self.get_logger().warning("[PWM_LOG] Matplotlib not available - skip plot")
            return
        
        times_pwm = np.array([entry[0] for entry in self.pwm_log_data], dtype=float)
        pwm_array = np.stack([entry[1] for entry in self.pwm_log_data], axis=0)  # shape: (N, 9)
        
        # Create 3x3 grid for 9 joints
        fig, axes = plt.subplots(3, 3, figsize=(15, 10), sharex=True)
        pwm_limit = self.max_pwm_limit  # Use actual limit from parameter
        fig.suptitle(f"PWM Monitoring (9 joints) | Limit=±{pwm_limit} | Max HW=±885", fontsize=14)
        
        joint_labels = [
            'TH_J0', 'TH_J1', 'TH_J2',
            'IF_J0', 'IF_J1', 'IF_J2',
            'MF_J0', 'MF_J1', 'MF_J2'
        ]
        
        # Dynamic y-axis limit based on pwm_limit
        y_max = max(pwm_limit * 1.5, 100)  # At least 100, or 1.5x the limit
        
        for joint_idx in range(9):
            row = joint_idx // 3
            col = joint_idx % 3
            ax = axes[row, col]
            
            # Plot PWM values
            ax.plot(times_pwm, pwm_array[:, joint_idx], color='purple', linewidth=1.2)
            ax.axhline(y=pwm_limit, color='orange', linestyle='--', alpha=0.7, label=f'Limit (±{pwm_limit})')
            ax.axhline(y=-pwm_limit, color='orange', linestyle='--', alpha=0.7)
            ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
            ax.set_ylabel("PWM Units")
            ax.set_ylim(-y_max, y_max)
            ax.grid(True, alpha=0.3)
            ax.set_title(joint_labels[joint_idx], fontsize=10)
            
            if row == 2:
                ax.set_xlabel("Time (s)")
            if joint_idx == 0:
                ax.legend(loc='upper right', fontsize=7)
        
        filename = os.path.join(session_folder, "05_pwm_monitoring.png")
        try:
            plt.tight_layout()
            plt.savefig(filename, dpi=200)
            self.get_logger().warning(
                f"[PWM_LOG] *** SAVED PLOT 5/5 *** PWM Monitoring ({len(times_pwm)} samples) -> {filename}"
            )
        except Exception as exc:
            self.get_logger().error(f"[PWM_LOG] Failed to save PWM plot: {exc}")
            import traceback
            self.get_logger().error(f"[PWM_LOG] Traceback:\n{traceback.format_exc()}")
        finally:
            plt.close()

    def _save_all_csv_data(self, session_folder: str) -> None:
        """Save all logged data as CSV files"""
        import csv
        
        # 1. Stiffness data
        if self.stiffness_log_data:
            filename = os.path.join(session_folder, "stiffness.csv")
            try:
                with open(filename, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['time', 'th_x', 'th_y', 'th_z', 'if_x', 'if_y', 'if_z', 'mf_x', 'mf_y', 'mf_z'])
                    for t, vals in self.stiffness_log_data:
                        writer.writerow([f'{t:.4f}'] + [f'{v:.4f}' for v in vals])
                self.get_logger().info(f"[CSV] Saved stiffness.csv ({len(self.stiffness_log_data)} rows)")
            except Exception as e:
                self.get_logger().error(f"[CSV] Failed to save stiffness.csv: {e}")
        
        # 2. Eccentricity data
        if self.eccentricity_log_data:
            filename = os.path.join(session_folder, "eccentricity.csv")
            try:
                with open(filename, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['time', 'ecc_raw', 'ecc_smoothed'])
                    ecc_smoothed_dict = {t: v for t, v in self.eccentricity_smoothed_log_data}
                    for t, raw in self.eccentricity_log_data:
                        smoothed = ecc_smoothed_dict.get(t, '')
                        writer.writerow([f'{t:.4f}', f'{raw:.6f}', f'{smoothed:.6f}' if smoothed else ''])
                self.get_logger().info(f"[CSV] Saved eccentricity.csv ({len(self.eccentricity_log_data)} rows)")
            except Exception as e:
                self.get_logger().error(f"[CSV] Failed to save eccentricity.csv: {e}")
        
        # 3. Force data
        if self.force_log_data:
            filename = os.path.join(session_folder, "force.csv")
            try:
                with open(filename, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['time', 'th_fx', 'th_fy', 'th_fz', 'if_fx', 'if_fy', 'if_fz', 'mf_fx', 'mf_fy', 'mf_fz'])
                    for t, force_dict in self.force_log_data:
                        row = [f'{t:.4f}']
                        for finger in ['th', 'if', 'mf']:
                            vec = force_dict.get(finger, np.zeros(3))
                            row.extend([f'{vec[0]:.4f}', f'{vec[1]:.4f}', f'{vec[2]:.4f}'])
                        writer.writerow(row)
                self.get_logger().info(f"[CSV] Saved force.csv ({len(self.force_log_data)} rows)")
            except Exception as e:
                self.get_logger().error(f"[CSV] Failed to save force.csv: {e}")
        
        # 4. EE position data
        if self.ee_pos_log_data:
            filename = os.path.join(session_folder, "ee_position.csv")
            try:
                with open(filename, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['time', 
                                   'th_actual_x', 'th_actual_y', 'th_actual_z',
                                   'if_actual_x', 'if_actual_y', 'if_actual_z',
                                   'mf_actual_x', 'mf_actual_y', 'mf_actual_z',
                                   'th_desired_x', 'th_desired_y', 'th_desired_z',
                                   'if_desired_x', 'if_desired_y', 'if_desired_z',
                                   'mf_desired_x', 'mf_desired_y', 'mf_desired_z'])
                    desired_dict = {t: d for t, d in self.desired_pos_log_data}
                    for t, ee_dict in self.ee_pos_log_data:
                        row = [f'{t:.4f}']
                        for finger in ['th', 'if', 'mf']:
                            vec = ee_dict.get(finger, np.zeros(3))
                            row.extend([f'{vec[0]:.6f}', f'{vec[1]:.6f}', f'{vec[2]:.6f}'])
                        desired = desired_dict.get(t, {})
                        for finger in ['th', 'if', 'mf']:
                            vec = desired.get(finger, np.zeros(3))
                            row.extend([f'{vec[0]:.6f}', f'{vec[1]:.6f}', f'{vec[2]:.6f}'])
                        writer.writerow(row)
                self.get_logger().info(f"[CSV] Saved ee_position.csv ({len(self.ee_pos_log_data)} rows)")
            except Exception as e:
                self.get_logger().error(f"[CSV] Failed to save ee_position.csv: {e}")
        
        # 5. Torque and current data (index-based matching since they're logged together)
        if self.torque_log_data and self.current_units_log_data:
            filename = os.path.join(session_folder, "torque_current.csv")
            try:
                with open(filename, 'w', newline='') as f:
                    writer = csv.writer(f)
                    header = ['time']
                    for j in range(9):
                        header.extend([f'tau_{j}', f'curr_{j}', f'curr_raw_{j}'])
                    writer.writerow(header)
                    
                    # Use index-based matching (torque, curr, curr_raw logged at same time)
                    n_samples = min(len(self.torque_log_data), len(self.current_units_log_data), 
                                    len(self.current_units_raw_log_data) if self.current_units_raw_log_data else len(self.current_units_log_data))
                    
                    for i in range(n_samples):
                        t, tau = self.torque_log_data[i]
                        _, curr = self.current_units_log_data[i]
                        _, curr_raw = self.current_units_raw_log_data[i] if i < len(self.current_units_raw_log_data) else (0, np.zeros(9))
                        row = [f'{t:.4f}']
                        for j in range(9):
                            row.extend([f'{tau[j]:.6f}', f'{int(curr[j])}', f'{int(curr_raw[j])}'])
                        writer.writerow(row)
                self.get_logger().info(f"[CSV] Saved torque_current.csv ({n_samples} rows)")
            except Exception as e:
                self.get_logger().error(f"[CSV] Failed to save torque_current.csv: {e}")
        
        # 6. PWM data
        if self.pwm_log_data:
            filename = os.path.join(session_folder, "pwm.csv")
            try:
                with open(filename, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['time'] + [f'pwm_{j}' for j in range(9)])
                    for t, pwm in self.pwm_log_data:
                        writer.writerow([f'{t:.4f}'] + [f'{int(p)}' for p in pwm])
                self.get_logger().info(f"[CSV] Saved pwm.csv ({len(self.pwm_log_data)} rows)")
            except Exception as e:
                self.get_logger().error(f"[CSV] Failed to save pwm.csv: {e}")
        
        self.get_logger().warning(f"[CSV] *** ALL CSV DATA SAVED *** to {session_folder}")

    def _control_loop(self) -> None:
        try:
            raw_tau = self._compute_torques()
            if self.stiffness_log_active:
                now_s = self.get_clock().now().nanoseconds / 1e9
                rel_t = now_s - self.stiffness_log_start_time
                if self.has_stiffness:
                    self.stiffness_log_data.append((rel_t, self.target_stiffness.copy()))
                if self.has_eccentricity:
                    self.eccentricity_log_data.append((rel_t, self.current_eccentricity))
                if self.has_eccentricity_smoothed:
                    self.eccentricity_smoothed_log_data.append((rel_t, self.current_eccentricity_smoothed))
                # Log force sensor data (all three fingers, all three axes)
                force_snapshot = {
                    'th': self.measured_force_vec['th'].copy(),
                    'if': self.measured_force_vec['if'].copy(),
                    'mf': self.measured_force_vec['mf'].copy()
                }
                self.force_log_data.append((rel_t, force_snapshot))
                # Log EE positions (actual and desired)
                ee_pos_snapshot = {
                    'th': self.fingers['th'].ee_pos.copy() if self.fingers['th'].ee_pos is not None else np.zeros(3),
                    'if': self.fingers['if'].ee_pos.copy() if self.fingers['if'].ee_pos is not None else np.zeros(3),
                    'mf': self.fingers['mf'].ee_pos.copy() if self.fingers['mf'].ee_pos is not None else np.zeros(3)
                }
                self.ee_pos_log_data.append((rel_t, ee_pos_snapshot))
                desired_pos_snapshot = {
                    'th': self.desired_pos['th'].copy(),
                    'if': self.desired_pos['if'].copy(),
                    'mf': self.desired_pos['mf'].copy()
                }
                self.desired_pos_log_data.append((rel_t, desired_pos_snapshot))
                # Log PWM values (torque and current logged after filtering below)
                if self.current_pwm is not None:
                    self.pwm_log_data.append((rel_t, self.current_pwm.copy()))
                # Debug: log every 100 samples
                # if len(self.stiffness_log_data) % 100 == 1:
                #     self.get_logger().info(
                #         f"[STIFFNESS_LOG] Recording... stiffness={len(self.stiffness_log_data)}, ecc={len(self.eccentricity_log_data)}, force={len(self.force_log_data)}, t={rel_t:.2f}s"
                #     )
            
            # # Debug: log raw_tau before safety check
            # if self._log_counter % int(self.rate_hz) == 0:
            #     tau_norm = np.linalg.norm(raw_tau)
            #     self.get_logger().info(
            #         f"[TORQUE_DEBUG] raw_tau norm={tau_norm:.3f}, "
            #         f"th=[{raw_tau[0]:.3f},{raw_tau[1]:.3f},{raw_tau[2]:.3f}] "
            #         f"if=[{raw_tau[3]:.3f},{raw_tau[4]:.3f},{raw_tau[5]:.3f}] "
            #         f"mf=[{raw_tau[6]:.3f},{raw_tau[7]:.3f},{raw_tau[8]:.3f}]"
            #     )
            
            # Position error safety check: zero torques if any finger exceeds threshold
            # safety_violation = False
            # violating_finger = None
            # max_err = 0.0
            # for finger in ['th', 'if', 'mf']:
            #     if self.fingers[finger].has_desired_pos:
            #         err = np.linalg.norm(self.desired_pos[finger] - self.fingers[finger].ee_pos)
            #         if err > max_err:
            #             max_err = err
            #             violating_finger = finger
            #         if err > self.pos_error_threshold:
            #             safety_violation = True
            #             # Don't break - continue to find max error for logging
            
            # if safety_violation:
            #     raw_tau = np.zeros(9)
            #     if self._log_counter % int(self.rate_hz) == 0:
            #         self.get_logger().warning(
            #             f"[SAFETY] Position error exceeded threshold - torques set to ZERO | "
            #             f"worst={violating_finger}: {max_err*1000:.1f}mm > {self.pos_error_threshold*1000:.1f}mm"
            #         )
            
            # Playback gating: keep torques/currents at zero until playback becomes active
            if not self.demo_playback_active:
                self._publish_zero_current("[PLAYBACK] Inactive -> holding zero torque", level="info")
                self._log_counter += 1
                return

            # Safety: if no stiffness input, send ZERO current immediately
            if not self.has_stiffness:
                self._publish_zero_current("[SAFETY] No stiffness input - sending ZERO current to all motors")
                self._log_counter += 1
                return
            
            raw_tau = np.clip(raw_tau, -self.max_torque, self.max_torque)
            
            # Apply torque filter for smoothing
            filt_tau = self._filter_torques(raw_tau)
            current_units, current_units_raw = self._torques_to_current_units(filt_tau)
            
            # Log torque AND current units together (same timestamp) if logging active
            if self.stiffness_log_active:
                now_s = self.get_clock().now().nanoseconds / 1e9
                rel_t = now_s - self.stiffness_log_start_time
                # Log all three together with same timestamp
                self.torque_log_data.append((rel_t, filt_tau.copy()))  # filtered torque
                self.current_units_log_data.append((rel_t, np.array(current_units, dtype=int)))
                self.current_units_raw_log_data.append((rel_t, np.array(current_units_raw, dtype=int)))
            
            # Debug: log filtered torque and current units
            if self._log_counter % int(self.rate_hz) == 0:
                self.get_logger().info(
                    f"[TORQUE_DEBUG] max_torque={self.max_torque}, "
                    f"th=[{filt_tau[0]:.3f},{filt_tau[1]:.3f},{filt_tau[2]:.3f}] "
                    f"if=[{filt_tau[3]:.3f},{filt_tau[4]:.3f},{filt_tau[5]:.3f}] "
                    f"mf=[{filt_tau[6]:.3f},{filt_tau[7]:.3f},{filt_tau[8]:.3f}]"
                )
                self.get_logger().info(
                    f"[TORQUE_DEBUG] current_units (clipped), "
                    f"th=[{current_units[0]},{current_units[1]},{current_units[2]}] "
                    f"if=[{current_units[3]},{current_units[4]},{current_units[5]}] "
                    f"mf=[{current_units[6]},{current_units[7]},{current_units[8]}]"
                )
                self.get_logger().info(
                    f"[TORQUE_DEBUG] current_units_raw (before clip), "
                    f"th=[{current_units_raw[0]},{current_units_raw[1]},{current_units_raw[2]}] "
                    f"if=[{current_units_raw[3]},{current_units_raw[4]},{current_units_raw[5]}] "
                    f"mf=[{current_units_raw[6]},{current_units_raw[7]},{current_units_raw[8]}]"
                )
            
            cur_msg = Int32MultiArray()
            cur_msg.data = current_units
            self.current_pub.publish(cur_msg)
            # tau_msg = Float32MultiArray(); tau_msg.data = filt_tau.tolist(); self.torque_pub.publish(tau_msg)
            self._log_counter += 1
            # Log K_rcv every ~30ms (like PWM monitor) instead of 1 second
            log_interval = max(2, int(self.rate_hz / 30))  # ~30Hz logging
            if self._log_counter % log_interval == 0:
                # Skip repetitive warnings during initialization
                if not self.has_qpos and self._log_counter == int(self.rate_hz):
                    self.get_logger().warning(f"[INIT] Waiting for joint state data...")
                if not self.has_stiffness and self._log_counter == int(self.rate_hz):
                    self.get_logger().warning(f"[INIT] Waiting for stiffness predictions...")
                
                # Only log status if we have basic data
                if self.has_qpos and self.has_stiffness:
                    # Desired EE status + position error check
                    desired_status = f"th={self.fingers['th'].has_desired_pos} if={self.fingers['if'].has_desired_pos} mf={self.fingers['mf'].has_desired_pos}"
                    pos_errors = []
                    for finger in ['th', 'if', 'mf']:
                        if self.fingers[finger].has_desired_pos:
                            err = np.linalg.norm(self.desired_pos[finger] - self.fingers[finger].ee_pos)
                            pos_errors.append(f"{finger}={err*1000:.1f}mm")
                            # if err > self.pos_error_threshold:
                            # self.get_logger().warning(
                            #     f"[TRACKING LAG] {finger}: error={err*1000:.1f}mm > threshold={self.pos_error_threshold*1000:.1f}mm"
                            # )
                    # error_str = " ".join(pos_errors) if pos_errors else "N/A"
                    # self.get_logger().info(f"[STATUS] desired_ee: {desired_status} | pos_err: {error_str}")
                    if self.has_stiffness:
                        k = self.target_stiffness
                        self.get_logger().info(
                            f"K_rcv: [TH: {k[0]:5.1f} {k[1]:5.1f} {k[2]:5.1f} | "
                            f"IF: {k[3]:5.1f} {k[4]:5.1f} {k[5]:5.1f} | "
                            f"MF: {k[6]:5.1f} {k[7]:5.1f} {k[8]:5.1f}]\n"
                            f"       τ_out: [TH: {filt_tau[0]:5.3f} {filt_tau[1]:5.3f} {filt_tau[2]:5.3f} | "
                            f"IF: {filt_tau[3]:5.3f} {filt_tau[4]:5.3f} {filt_tau[5]:5.3f} | "
                            f"MF: {filt_tau[6]:5.3f} {filt_tau[7]:5.3f} {filt_tau[8]:5.3f}] | "
                            f"I_avg={np.mean(np.abs(current_units)):.0f}"
                        )
        except Exception as e:
            # Manual throttling: log error only once per second
            self._error_throttle_counter += 1
            if self._error_throttle_counter >= int(self.rate_hz):
                self.get_logger().error(f"제어 루프 오류: {e}")
                self._error_throttle_counter = 0

    def destroy_node(self) -> None:
        if self.stiffness_log_active:
            self._stop_stiffness_logging(reason="shutdown")
        # Viewer 종료 처리
        if self._mj_viewer is not None:
            try:
                self._mj_viewer.close()  # type: ignore[attr-defined]
            except Exception:
                pass
            self._mj_viewer = None
        super().destroy_node()

    def publish_ee_pose_message(self) -> None:
        # Publish per-finger EE position (orientation left zero)
        now = self.get_clock().now().to_msg()
        def _mk_pose(pos: np.ndarray) -> PoseStamped:
            msg = PoseStamped()
            msg.header.stamp = now
            msg.header.frame_id = self.ee_pose_frame_id
            msg.pose.position.x = float(pos[0])
            msg.pose.position.y = float(pos[1])
            msg.pose.position.z = float(pos[2])
            # orientation left default (0,0,0,0) — can be extended later
            return msg
        try:
            if self.ee_pose_pub_if and self.fingers["if"].ee_pos is not None:
                self.ee_pose_pub_if.publish(_mk_pose(self.fingers["if"].ee_pos))
            if self.ee_pose_pub_mf and self.fingers["mf"].ee_pos is not None:
                self.ee_pose_pub_mf.publish(_mk_pose(self.fingers["mf"].ee_pos))
            if self.ee_pose_pub_th and self.fingers["th"].ee_pos is not None:
                self.ee_pose_pub_th.publish(_mk_pose(self.fingers["th"].ee_pos))
        except Exception:
            pass


def main(args=None) -> None:
    rclpy.init(args=args)
    try:
        node = TorqueImpedanceControllerNode(); rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok(): rclpy.shutdown()


if __name__ == "__main__":
    main()
