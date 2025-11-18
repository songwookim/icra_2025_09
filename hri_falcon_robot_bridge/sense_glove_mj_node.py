#!/usr/bin/env python3
"""SenseGlove-to-MuJoCo bridge that exports qpos-driven unit commands.

The node treats MuJoCo as the authoritative source of joint positions, polls
the simulator for qpos updates, and converts them into Dynamixel unit targets
while optionally mirroring the motion inside MuJoCo for visualization. The
behaviour mirrors the original hand_tracker_node.py implementation but without
any camera or MediaPipe dependencies.
"""

from __future__ import annotations

import math
import os
import select
import sys
import threading
from typing import IO, Any, Dict, List, Optional, Sequence, Tuple, cast

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.parameter import Parameter
from rcl_interfaces.msg import SetParametersResult

from sensor_msgs.msg import JointState
from std_msgs.msg import Int32MultiArray, String, Bool
from std_srvs.srv import Trigger
from geometry_msgs.msg import PoseStamped

try:
    import mujoco as mj  # type: ignore
    from mujoco import viewer as mj_viewer  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    mj = None  # type: ignore[assignment]
    mj_viewer = None  # type: ignore[assignment]

try:
    import termios
    import tty
except Exception:  # pragma: no cover - platform-specific
    termios = None  # type: ignore[assignment]
    tty = None  # type: ignore[assignment]

try:
    from omegaconf import OmegaConf  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    OmegaConf = None  # type: ignore[assignment]

FINGER_NAMES: Sequence[str] = ("THUMB", "INDEX", "MIDDLE")
JOINT_NAMES = {
    "THUMB": ("CMC", "MCP", "IP"),
    "INDEX": ("MCP", "PIP", "DIP"),
    "MIDDLE": ("MCP", "PIP", "DIP"),
}
COMMAND_ORDER: Sequence[Tuple[str, str]] = (
    ("THUMB", "CMC"),
    ("THUMB", "MCP"),
    ("THUMB", "IP"),
    ("INDEX", "MCP"),
    ("INDEX", "PIP"),
    ("INDEX", "DIP"),
    ("MIDDLE", "MCP"),
    ("MIDDLE", "PIP"),
    ("MIDDLE", "DIP"),
)

COMMAND_COUNT = len(COMMAND_ORDER)


def _load_initial_units_from_config() -> Optional[List[float]]:
    if OmegaConf is None:
        return None
    try:
        pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cfg_path = os.path.join(pkg_dir, "resource", "robot_parameter", "config.yaml")
    except Exception:
        return None
    if not os.path.exists(cfg_path):
        return None
    try:
        cfg = OmegaConf.load(cfg_path)
        cfg_data = OmegaConf.to_container(cfg, resolve=True)  # type: ignore[call-arg]
    except Exception:
        return None
    if not isinstance(cfg_data, dict):
        return None
    dynamixel_cfg = cfg_data.get("dynamixel")
    if not isinstance(dynamixel_cfg, dict):
        return None
    initial_vals = dynamixel_cfg.get("initial_positions")
    if not isinstance(initial_vals, (list, tuple)):
        return None
    collected: List[float] = []
    for value in initial_vals:
        try:
            collected.append(float(value))
        except (TypeError, ValueError):
            continue
    return collected if collected else None


_FALLBACK_UNITS_BASELINE: Sequence[float] = tuple(
    1000.0 if idx % 3 == 0 else 2000.0 for idx in range(COMMAND_COUNT)
)
_CONFIG_UNITS_BASELINE = _load_initial_units_from_config()
if _CONFIG_UNITS_BASELINE and len(_CONFIG_UNITS_BASELINE) >= COMMAND_COUNT:
    DEFAULT_UNITS_BASELINE: Sequence[float] = tuple(
        _CONFIG_UNITS_BASELINE[idx] for idx in range(COMMAND_COUNT)
    )
else:
    DEFAULT_UNITS_BASELINE = _FALLBACK_UNITS_BASELINE

# DClaw joint names used by the MuJoCo model
DCLAW_JOINTS: Dict[str, Sequence[str]] = {
    "THUMB": ("THJ30", "THJ31", "THJ32"),
    "INDEX": ("FFJ10", "FFJ11", "FFJ12"),
    "MIDDLE": ("MFJ20", "MFJ21", "MFJ22"),
}

# DEFAULT_MUJOCO_MODEL_PATH = "/home/songwoo/Desktop/work_dir/realsense_hand_retargetting/universal_robots_ur5e_with_dclaw/dclaw/dclaw3xh.xml"
DEFAULT_MUJOCO_MODEL_PATH = '/home/songwoo/git/ur_dclaw/dclaw_finger_description/urdf/dclaw_finger_mjcf_final.xml'


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


class SenseGloveMJNode(Node):
    def __init__(self) -> None:
        super().__init__("sense_glove_mj_node")

        # Topic configuration
        self.declare_parameter("joint_state_topic", "/hand_tracker/joint_state")
        self.declare_parameter("units_topic", "/hand_tracker/targets_units")
        self.declare_parameter("key_topic", "/hand_tracker/key")
        self.declare_parameter("units_state_topic", "/hand_tracker/units_enabled")
        self.declare_parameter("pose_log_interval_sec", 1.0)
        self.declare_parameter("pose_log_enabled", False)
        self.declare_parameter("log_clamp_events", True)

        # Unit / qpos mapping configuration (mirrors the C++ node defaults)
        self.declare_parameter("units_publish_enabled", False)
        self.declare_parameter("units_baseline", list(DEFAULT_UNITS_BASELINE))
        self.declare_parameter("units_per_rad", 4096.0 / (2.0 * math.pi))
        self.declare_parameter("units_motion_scale_qpos", 1.0)
        self.declare_parameter("units_min", 0.0)
        self.declare_parameter("units_max", 4095.0)
        self.declare_parameter("mujoco_qpos_poll_interval_sec", 0.01)

        self.declare_parameter("qpos_gain", 0.75)
        self.declare_parameter("qpos_smooth_alpha", 0.5)
        self.declare_parameter("qpos_step_max", 0.05)
        self.declare_parameter("clamp_qpos_symm", True)
        self.declare_parameter("clamp_qpos_min", -1.57)
        self.declare_parameter("clamp_qpos_max", 1.57)
        self.declare_parameter("global_qpos_sign", -1.0)
        self.declare_parameter("units_output_sign", -1.0)
    # Per-joint movement weights (follow COMMAND_ORDER; default 1.0)
        self.declare_parameter("joint_weights", [1.0] * COMMAND_COUNT)

        # MuJoCo / EE pose configuration (optional)
        self.declare_parameter("run_mujoco", True)
        self.declare_parameter("mujoco_model_path", DEFAULT_MUJOCO_MODEL_PATH)
        self.declare_parameter("ee_pose_publish_enabled", True)
        self.declare_parameter("ee_pose_topic_if", "/ee_pose_if")
        self.declare_parameter("ee_pose_topic_mf", "/ee_pose_mf")
        self.declare_parameter("ee_pose_topic_th", "/ee_pose_th")
        self.declare_parameter("ee_pose_frame_id", "world")
        self.declare_parameter("ee_pose_source", "mujoco")
        self.declare_parameter("ee_pose_mj_site", "MFtip")
        self.declare_parameter("ee_pose_mj_body", "")
        self.declare_parameter("ee_pose_mj_site_th", "THtip")
        self.declare_parameter("terminal_status_interval_sec", 1.0)

        self.joint_state_topic = self.get_parameter("joint_state_topic").get_parameter_value().string_value
        self.units_topic = self.get_parameter("units_topic").get_parameter_value().string_value
        self.key_topic = self.get_parameter("key_topic").get_parameter_value().string_value
        self.units_state_topic = self.get_parameter("units_state_topic").get_parameter_value().string_value
        self.pose_log_interval_sec = max(0.0, self.get_parameter("pose_log_interval_sec").get_parameter_value().double_value)
        self.pose_log_enabled = self.get_parameter("pose_log_enabled").get_parameter_value().bool_value
        self.log_clamp_events = self.get_parameter("log_clamp_events").get_parameter_value().bool_value

        self.units_publish_enabled = self.get_parameter("units_publish_enabled").get_parameter_value().bool_value
        raw_units_baseline = self.get_parameter("units_baseline").value
        self.units_baseline: List[float] = self._ensure_float_list(
            raw_units_baseline,
            COMMAND_COUNT,
            DEFAULT_UNITS_BASELINE,
        )
        self.units_per_rad = self.get_parameter("units_per_rad").get_parameter_value().double_value
        self.units_motion_scale_qpos = self.get_parameter("units_motion_scale_qpos").get_parameter_value().double_value
        self.units_min = self.get_parameter("units_min").get_parameter_value().double_value
        self.units_max = self.get_parameter("units_max").get_parameter_value().double_value
        self.units_output_sign = self.get_parameter("units_output_sign").get_parameter_value().double_value
        poll_interval_param = self.get_parameter("mujoco_qpos_poll_interval_sec").get_parameter_value().double_value
        if poll_interval_param <= 0.0:
            poll_interval_param = 0.01
        self.mujoco_qpos_poll_interval_sec = poll_interval_param

        self.qpos_gain = self.get_parameter("qpos_gain").get_parameter_value().double_value
        self.qpos_smooth_alpha = clamp(self.get_parameter("qpos_smooth_alpha").get_parameter_value().double_value, 0.0, 1.0)
        self.qpos_step_max = max(0.0, self.get_parameter("qpos_step_max").get_parameter_value().double_value)
        self.clamp_qpos_symm = self.get_parameter("clamp_qpos_symm").get_parameter_value().bool_value
        self.clamp_qpos_min = self.get_parameter("clamp_qpos_min").get_parameter_value().double_value
        self.clamp_qpos_max = self.get_parameter("clamp_qpos_max").get_parameter_value().double_value
        self.global_qpos_sign = self.get_parameter("global_qpos_sign").get_parameter_value().double_value

        self.run_mujoco = self.get_parameter("run_mujoco").get_parameter_value().bool_value
        self.mujoco_model_path = self.get_parameter("mujoco_model_path").get_parameter_value().string_value
        if not self.mujoco_model_path:
            self.mujoco_model_path = DEFAULT_MUJOCO_MODEL_PATH
        self.ee_pose_publish_enabled = self.get_parameter("ee_pose_publish_enabled").get_parameter_value().bool_value
        self.ee_pose_topic_if = self.get_parameter("ee_pose_topic_if").get_parameter_value().string_value
        self.ee_pose_topic_mf = self.get_parameter("ee_pose_topic_mf").get_parameter_value().string_value
        self.ee_pose_topic_th = self.get_parameter("ee_pose_topic_th").get_parameter_value().string_value
        self.ee_pose_frame_id = self.get_parameter("ee_pose_frame_id").get_parameter_value().string_value
        self.ee_pose_source = self.get_parameter("ee_pose_source").get_parameter_value().string_value.lower().strip()
        self.ee_pose_mj_site = self.get_parameter("ee_pose_mj_site").get_parameter_value().string_value
        self.ee_pose_mj_body = self.get_parameter("ee_pose_mj_body").get_parameter_value().string_value
        self.ee_pose_mj_site_th = self.get_parameter("ee_pose_mj_site_th").get_parameter_value().string_value
        self.terminal_status_interval_sec = max(0.0, self.get_parameter("terminal_status_interval_sec").get_parameter_value().double_value)

        # Parse per-joint movement weights (0.0..1.0)
        raw_joint_weights = self.get_parameter("joint_weights").value
        self.joint_weights: List[float] = [
            clamp(v, 0.0, 1.0) for v in self._ensure_float_list(raw_joint_weights, COMMAND_COUNT, [1.0] * COMMAND_COUNT)
        ]

        # Internal state
        # Per-finger orientation overrides (mirrors hand_tracker defaults)
        self.declare_parameter("joint_orientation_thumb", [-1., -1., -1.])
        self.declare_parameter("joint_orientation_index", [-1., -1., -1.])
        self.declare_parameter("joint_orientation_middle", [-1., -1., -1.])

        self._finger_index = {name: idx for idx, name in enumerate(FINGER_NAMES)}
        self._joint_orientation: List[List[float]] = [
            self._load_orientation_param("joint_orientation_thumb", (-1., -1., -1.)),
            self._load_orientation_param("joint_orientation_index", (-1., -1., -1.)),
            self._load_orientation_param("joint_orientation_middle", (-1., -1., -1.)),
        ]
        self._thumb_sign_patterns: List[List[float]] = [
            [1.0, 1.0, 1.0],
            [-1.0, -1.0, -1.0],
            [-1.0, 1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, 1.0, -1.0],
            [1.0, -1.0, -1.0],
        ]
        baseline_zero_qpos = self._build_zero_qpos_from_units(self.units_baseline)
        self._mujoco_qpos_poll_timer = None
        self._mj_poll_warned = False
        self._thumb_pattern_idx = 0
        self._viz_thumb_complement = False
        self._initial_zero_qpos_ref: List[List[float]] = [row[:] for row in baseline_zero_qpos]
        self._zero_qpos_ref: List[List[float]] = [row[:] for row in baseline_zero_qpos]
        self._prev_qpos_cmd: List[List[float]] = [[0.0] * 3 for _ in FINGER_NAMES]
        self._latest_smoothed_qpos: List[List[float]] = [[0.0] * 3 for _ in FINGER_NAMES]
        self._latest_raw_qpos: List[List[float]] = [row[:] for row in self._zero_qpos_ref]
        self._latest_qpos_valid: bool = False
        self._latest_raw_valid: bool = False
        self._first_pose_logged: bool = False
        self._last_pose_log_time: Optional[Time] = None
        self._clamp_notified: List[List[bool]] = [[False] * 3 for _ in FINGER_NAMES]
        self._logger_active: Optional[bool] = None
        self._last_ee_pose: Optional[Tuple[float, float, float]] = None
        self._latest_input_qpos: List[List[float]] = [[math.nan] * 3 for _ in FINGER_NAMES]
        self._latest_glove_positions: List[List[float]] = [[math.nan] * 3 for _ in FINGER_NAMES]
        # Track recently published key to avoid feedback when we also subscribe to the same topic
        self._last_published_key: Optional[str] = None
        self._last_published_time: Optional[Time] = None

        # ROS entities
        self.units_pub = self.create_publisher(Int32MultiArray, self.units_topic, 10)
        self.key_pub = self.create_publisher(String, self.key_topic, 10)
        self.units_state_pub = self.create_publisher(Bool, self.units_state_topic, 10)

        # qpos publisher 추가
        self.qpos_topic = "/hand_tracker/qpos"
        self.qpos_pub = self.create_publisher(JointState, self.qpos_topic, 10)

        self.ee_pose_pub = None
        self.ee_pose_pub_mf = None
        self.ee_pose_pub_th = None
        if self.ee_pose_publish_enabled and self.ee_pose_topic_if:
            self.ee_pose_pub = self.create_publisher(PoseStamped, self.ee_pose_topic_if, 10)
        if self.ee_pose_publish_enabled and self.ee_pose_topic_mf:
            self.ee_pose_pub_mf = self.create_publisher(PoseStamped, self.ee_pose_topic_mf, 10)
        if self.ee_pose_publish_enabled and self.ee_pose_topic_th:
            self.ee_pose_pub_th = self.create_publisher(PoseStamped, self.ee_pose_topic_th, 10)

        self.zero_srv = self.create_service(Trigger, "set_zero", self._handle_zero_request)
        self.get_logger().info("SenseGlove MJ bridge ready...")

        self.joint_state_sub = None

        zero_segments = []
        for finger_name, joint_name in COMMAND_ORDER:
            f_idx = self._finger_index[finger_name]
            j_idx = JOINT_NAMES[finger_name].index(joint_name)
            zero_segments.append(f"{finger_name}_{joint_name}={self._zero_qpos_ref[f_idx][j_idx]:.3f}rad")
        if zero_segments:
            self.get_logger().info("[Zero] baseline qpos (rad) loaded from config -> " + ", ".join(zero_segments))

        try:
            self.create_subscription(Bool, "/data_logger/logging_active", self._on_logger_state, 10)
        except Exception:
            pass

        # Subscribe to remote key topic (e.g., keys from deformity_tracker UI)
        try:
            self.create_subscription(String, self.key_topic, self._on_remote_key, 10)
        except Exception:
            pass

        # MuJoCo runtime placeholders
        self._mj_enabled = False
        self._mj_model = None
        self._mj_data = None
        self._mj_viewer = None
        self._mj_qpos_adr: Dict[str, int] = {}
        self._ee_site_id: Optional[int] = None
        self._ee_body_id: Optional[int] = None
        self._ee_site_id_th: Optional[int] = None
        self._ee_mj_warned = False
        self._mj_forward_warned = False

        # Terminal input handling
        self.declare_parameter("enable_terminal_input", True)
        self.enable_terminal_input: bool = bool(self.get_parameter("enable_terminal_input").value)
        self._terminal_stop_event = threading.Event()
        self._terminal_thread: Optional[threading.Thread] = None
        self._terminal_fd: Optional[int] = None
        self._terminal_old_attrs: Optional[List[Any]] = None
        self._terminal_stream: Optional[IO[str]] = None
        if self.enable_terminal_input:
            if self._setup_terminal_reader():
                self._terminal_thread = threading.Thread(target=self._terminal_input_loop, name="sg_mj_terminal", daemon=True)
                self._terminal_thread.start()
            else:
                self.get_logger().warn("터미널 키 입력 초기화 실패 -> 키 입력 비활성화")

        self._init_mujoco()
        self._apply_initial_mujoco_pose()
        self._configure_qpos_source()
        self._log_key_shortcuts()
        self._publish_units_state(initial=True)
        self._publish_baseline_units(initial=True)

        # Enable runtime parameter updates for joint_weights
        try:
            self.add_on_set_parameters_callback(self._on_set_parameters)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------
    def _log_key_shortcuts(self) -> None:
        shortcuts = [
            "\nh: toggle units publish\n",
            "s: toggle data logger\n",
            "c: capture zero reference\n",
            "t: cycle thumb orientatioe\n",
            "r: toggle thumb complement visuae\n",
            "o: flip index/middle orientatioe\n",
            "x: flip global qpos sige\n",
            "g: cycle qpos gaie\n",
            "j: print current qpoe\n",
        ]
        self.get_logger().info("[Keys] " + " | ".join(shortcuts))

    def _ensure_float_list(self, raw: Any, length: int, default: Sequence[float]) -> List[float]:
        default_list = list(default)
        values: List[float] = []
        if isinstance(raw, (list, tuple)):
            for element in raw:
                try:
                    values.append(float(element))
                except (TypeError, ValueError):
                    continue
        elif isinstance(raw, (int, float)):
            values = [float(raw)]

        if not values:
            values = [float(val) for val in default_list[:length]]

        if len(values) < length:
            fill_val = values[-1] if values else float(default_list[-1])
            values.extend([fill_val] * (length - len(values)))

        return values[:length]

    def _publish_units_state(self, initial: bool = False) -> None:
        if self.units_state_pub is None:
            return
        try:
            msg = Bool()
            msg.data = bool(self.units_publish_enabled)
            self.units_state_pub.publish(msg)
            if not initial:
                state = "ON" if msg.data else "OFF"
                self.get_logger().info(f"[Units] state broadcast -> {state}")
                if msg.data:
                    self._publish_baseline_units()
        except Exception:
            pass

    def _publish_baseline_units(self, initial: bool = False) -> None:
        if not self.units_publish_enabled:
            return
        if self.units_pub is None:
            return
        try:
            msg = Int32MultiArray()
            msg.data = [int(round(val)) for val in self.units_baseline]
            self.units_pub.publish(msg)
            if initial:
                self.get_logger().info("[Units] published baseline posture from config")
        except Exception:
            pass

    def _build_zero_qpos_from_units(self, units_values: Sequence[float]) -> List[List[float]]:
        zero_qpos = [[0.0] * 3 for _ in FINGER_NAMES]
        if not units_values:
            return zero_qpos

        denom = self.units_per_rad * self.units_motion_scale_qpos
        if not math.isfinite(denom) or abs(denom) < 1e-9:
            fallback = self.units_per_rad if math.isfinite(self.units_per_rad) and abs(self.units_per_rad) >= 1e-9 else 1.0
            denom = fallback

        for cmd_idx, (finger_name, joint_name) in enumerate(COMMAND_ORDER):
            if cmd_idx >= len(units_values):
                units_val = float(units_values[-1])
            else:
                units_val = float(units_values[cmd_idx])
            
            # Subtract the 12 offset that was added when converting qpos to units
            units_val_adjusted = units_val - 12.0
            
            # Subtract bias to get units_calc
            bias = 1000.0 if cmd_idx in (0, 3, 6) else 2000.0
            units_calc = units_val_adjusted - bias
            
            # Convert to radians
            rad_val = units_calc / denom
            
            # Add offset to get final qpos
            offset = 1.57 if cmd_idx in (0, 3, 6) else 3.14
            rad_val += offset
            
            f_idx = self._finger_index[finger_name]
            j_idx = JOINT_NAMES[finger_name].index(joint_name)
            zero_qpos[f_idx][j_idx] = rad_val
        return zero_qpos

    def _apply_initial_mujoco_pose(self) -> None:
        if not self._mj_enabled or self._mj_model is None or self._mj_data is None:
            return

        any_written = False
        for finger_name, joint_names in DCLAW_JOINTS.items():
            f_idx = self._finger_index.get(finger_name)
            if f_idx is None:
                continue
            for j_idx, mj_joint in enumerate(joint_names):
                adr = self._mj_qpos_adr.get(mj_joint)
                if adr is None:
                    continue
                try:
                    value = float(self._zero_qpos_ref[f_idx][j_idx])
                except (IndexError, TypeError, ValueError):
                    continue
                if adr in [0,3,6]:
                    value -= 1.57
                else :
                    value -= 3.14
                self._mj_data.qpos[adr] = value
                any_written = True

        if not any_written:
            return

        try:
            mj.mj_forward(self._mj_model, self._mj_data)  # type: ignore[attr-defined]
        except Exception as exc:
            if not self._mj_forward_warned:
                self._mj_forward_warned = True
                self.get_logger().warn(f"[MuJoCo] forward failed after baseline apply: {exc}")
            return

        if self._mj_viewer is not None:
            try:
                if self._mj_viewer.is_running():  # type: ignore[attr-defined]
                    self._mj_viewer.sync()  # type: ignore[attr-defined]
            except Exception:
                pass

        self.get_logger().info("[MuJoCo] qpos set to baseline zero reference from config")

    def _configure_qpos_source(self) -> None:
        def _start_mujoco_poll() -> None:
            if not self._mj_enabled or self._mj_data is None:
                self.get_logger().error("MuJoCo initialization failed; qpos polling cannot start.")
                return
            if self._mujoco_qpos_poll_timer is not None:
                return
            interval = max(0.001, self.mujoco_qpos_poll_interval_sec)
            self._mujoco_qpos_poll_timer = self.create_timer(interval, self._poll_mujoco_qpos)
            self.get_logger().info(f"[MuJoCo] qpos polling enabled (period={interval:.3f}s)")

        if self.joint_state_topic:
            try:
                if self.joint_state_sub is None:
                    self.joint_state_sub = self.create_subscription(
                        JointState,
                        self.joint_state_topic,
                        self._on_joint_state,
                        10,
                    )
                    self.get_logger().info(
                        f"[SenseGlove] joint state subscription attached -> {self.joint_state_topic}"
                    )
            except Exception as exc:
                self.get_logger().error(
                    f"Failed to subscribe joint state topic '{self.joint_state_topic}': {exc}. Falling back to MuJoCo polling."
                )
                _start_mujoco_poll()
                return

            if self._mujoco_qpos_poll_timer is not None:
                try:
                    self._mujoco_qpos_poll_timer.cancel()
                except Exception:
                    pass
                self._mujoco_qpos_poll_timer = None

            if self._mj_enabled and self.run_mujoco:
                self.get_logger().info("[MuJoCo] joint states will drive the simulation mirror.")
            return

        _start_mujoco_poll()

    def _poll_mujoco_qpos(self) -> None:
        if not self._mj_enabled or self._mj_data is None:
            return

        qpos_buffer = getattr(self._mj_data, "qpos", None)
        if qpos_buffer is None:
            if not self._mj_poll_warned:
                self._mj_poll_warned = True
                self.get_logger().warn("[MuJoCo] qpos buffer unavailable; skipping poll.")
            return

        qpos_values: List[List[float]] = [[math.nan] * 3 for _ in FINGER_NAMES]
        any_valid = False
        for finger_name, joint_names in DCLAW_JOINTS.items():
            f_idx = self._finger_index.get(finger_name)
            if f_idx is None:
                continue
            for j_idx, mj_joint in enumerate(joint_names):
                adr = self._mj_qpos_adr.get(mj_joint)
                if adr is None:
                    continue
                try:
                    value = float(qpos_buffer[adr])
                except Exception:
                    continue
                if not math.isfinite(value):
                    continue
                qpos_values[f_idx][j_idx] = value
                any_valid = True

        if any_valid:
            if self._mj_poll_warned:
                self._mj_poll_warned = False
            self._process_qpos(qpos_values)
        elif not self._mj_poll_warned:
            self._mj_poll_warned = True
            self.get_logger().warn("[MuJoCo] qpos polling yielded no valid joint values.")

    def _load_orientation_param(self, param_name: str, fallback: Sequence[float]) -> List[float]:
        try:
            raw_value = self.get_parameter(param_name).value
        except Exception:
            raw_value = fallback

        parsed: List[float] = []
        if isinstance(raw_value, (list, tuple)):
            for element in raw_value:
                if len(parsed) == 3:
                    break
                try:
                    parsed.append(float(element))
                except (TypeError, ValueError):
                    continue
        elif isinstance(raw_value, (int, float)):
            parsed = [float(raw_value)] * 3

        if len(parsed) < 3:
            fallback_list = list(fallback)
            parsed.extend(fallback_list[len(parsed):3])
        return parsed[:3]

    def _flip_non_thumb_orientation(self) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        index_idx = self._finger_index["INDEX"]
        middle_idx = self._finger_index["MIDDLE"]

        for finger_idx in (index_idx, middle_idx):
            self._joint_orientation[finger_idx] = [-val for val in self._joint_orientation[finger_idx]]
            for joint_idx in range(3):
                self._clamp_notified[finger_idx][joint_idx] = False

        return (
            cast(Tuple[float, float, float], tuple(self._joint_orientation[index_idx])),
            cast(Tuple[float, float, float], tuple(self._joint_orientation[middle_idx])),
        )

    # ------------------------------------------------------------------
    # ROS callbacks
    # ------------------------------------------------------------------
    def _on_remote_key(self, msg: String) -> None:
        try:
            incoming = (msg.data or "").strip().lower()
        except Exception:
            return
        if not incoming:
            return

        # Ignore our own recently published key to avoid feedback loop
        try:
            if self._last_published_key == incoming and self._last_published_time is not None:
                elapsed = self.get_clock().now() - self._last_published_time
                if elapsed.nanoseconds() <= int(0.3 * 1e9):
                    return
        except Exception:
            pass

        prefix = "[Remote/Key]"
        if incoming == "h":
            self.units_publish_enabled = not self.units_publish_enabled
            state = "ON" if self.units_publish_enabled else "OFF"
            self.get_logger().info(f"{prefix} [Units] publish -> {state}")
            self._publish_units_state()
        elif incoming == "c":
            if self._capture_zero_reference():
                self.get_logger().info(f"{prefix} [Zero] zero_qpos_ref updated from current glove pose")
            else:
                self.get_logger().warn(f"{prefix} [Zero] zero capture failed (no pose yet)")
        elif incoming == "s":
            # Logger toggle handled elsewhere; acknowledge receipt.
            self.get_logger().info(f"{prefix} [Logger] toggle request")
        else:
            # Unhandled remote key
            return

    def _on_joint_state(self, msg: JointState) -> None:
        incoming_qpos = [[math.nan] * 3 for _ in FINGER_NAMES]

        for name, position in zip(msg.name, msg.position):
            parts = name.split("_", 1)
            if len(parts) != 2:
                continue
            finger, joint = parts
            if finger not in self._finger_index:
                continue
            joint_list = JOINT_NAMES.get(finger)
            if joint_list is None or joint not in joint_list:
                continue
            f_idx = self._finger_index[finger]
            j_idx = joint_list.index(joint)
            baseline = 0.0
            try:
                baseline = float(self._zero_qpos_ref[f_idx][j_idx])
            except (IndexError, TypeError, ValueError):
                baseline = 0.0
            glove_val = float(position)
            self._latest_glove_positions[f_idx][j_idx] = glove_val
            incoming_qpos[f_idx][j_idx] = -glove_val + baseline

        self._process_qpos(incoming_qpos)

    def _handle_zero_request(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        if not self._capture_zero_reference():
            response.success = False
            response.message = "No SenseGlove joint data received yet."
            return response
        response.success = True
        response.message = "Zero reference captured from current glove pose."
        self.get_logger().info("Zero offsets updated from SenseGlove pose.")
        return response

    # ------------------------------------------------------------------
    # Processing helpers
    # ------------------------------------------------------------------
    def _process_qpos(self, qpos_values: List[List[float]]) -> None:
        self._latest_input_qpos = [row[:] for row in qpos_values]
        smoothed_qpos = [row[:] for row in self._latest_smoothed_qpos]
        raw_qpos = [row[:] for row in self._latest_raw_qpos]
        units_out: List[int] = []
        any_valid = False
        raw_all_valid = True

        for cmd_idx, (finger_name, joint_name) in enumerate(COMMAND_ORDER):
            baseline_units = self.units_baseline[cmd_idx]
            f_idx = self._finger_index[finger_name]
            j_idx = JOINT_NAMES[finger_name].index(joint_name)
            mapped_qpos = qpos_values[f_idx][j_idx]

            if math.isnan(mapped_qpos):
                raw_all_valid = False
                units_out.append(int(round(baseline_units)))
                continue

            any_valid = True
            # Apply per-joint weight around zero reference to reduce motion amplitude
            try:
                zero_ref = float(self._zero_qpos_ref[f_idx][j_idx])
            except (IndexError, TypeError, ValueError):
                zero_ref = 0.0
            try:
                weight = float(self.joint_weights[cmd_idx])
            except Exception:
                weight = 1.0
            weight = clamp(weight, 0.0, 1.0)
            final_qpos = zero_ref + weight * (mapped_qpos - zero_ref)
            raw_qpos[f_idx][j_idx] = final_qpos
        

            self._prev_qpos_cmd[f_idx][j_idx] = final_qpos
            smoothed_qpos[f_idx][j_idx] = final_qpos

            raw_angle = final_qpos
            offset = 1.57 if cmd_idx in (0, 3, 6) else 3.14
            adjusted = raw_angle - offset
            units_calc = adjusted * self.units_per_rad * self.units_motion_scale_qpos
            if not math.isfinite(units_calc):
                units_calc = 0.0
            units_calc = clamp(units_calc, -4096.0, 4096.0)
            bias = 1000.0 if cmd_idx in (0, 3, 6) else 2000.0
            units_val = clamp(units_calc + bias + 12.0, self.units_min, self.units_max)
            units_out.append(int(round(units_val)))

        if any_valid:
            self._latest_raw_qpos = raw_qpos
            self._latest_raw_valid = raw_all_valid
            self._latest_smoothed_qpos = smoothed_qpos
            self._latest_qpos_valid = True
            if self.run_mujoco:
                self._apply_to_mujoco(raw_qpos)
            self._maybe_log_qpos(smoothed_qpos)
            if self.units_publish_enabled:
                msg = Int32MultiArray()
                msg.data = units_out
                self.units_pub.publish(msg)
            
            # qpos publish - MuJoCo 시뮬레이터에서 읽어온 qpos 사용
            if self._mj_enabled and self._mj_data is not None:
                qpos_msg = JointState()
                qpos_msg.header.stamp = self.get_clock().now().to_msg()
                qpos_msg.name = [f"{finger}_{joint}" for finger, joint in COMMAND_ORDER]
                qpos_positions = []
                for finger_name, joint_name in COMMAND_ORDER:
                    f_idx = self._finger_index[finger_name]
                    j_idx = JOINT_NAMES[finger_name].index(joint_name)
                    mj_joint_names = DCLAW_JOINTS[finger_name]
                    mj_joint = mj_joint_names[j_idx]
                    adr = self._mj_qpos_adr.get(mj_joint)
                    if adr is not None:
                        try:
                            mj_qpos_val = float(self._mj_data.qpos[adr])
                            # MuJoCo에서 읽은 값에 offset을 더해 원래 qpos로 복원
                            qpos_positions.append(mj_qpos_val)
                        except Exception:
                            qpos_positions.append(0.0)
                    else:
                        qpos_positions.append(0.0)
                qpos_msg.position = qpos_positions
                self.qpos_pub.publish(qpos_msg)
        else:
            self._latest_qpos_valid = False
            self._latest_raw_valid = False
    def _maybe_log_qpos(self, qpos_values: List[List[float]]) -> None:
        if not self.pose_log_enabled:
            return

        now = self.get_clock().now()
        should_log = False

        if not self._first_pose_logged:
            should_log = True
        elif self.pose_log_interval_sec > 0.0 and self._last_pose_log_time is not None:
            elapsed = now - self._last_pose_log_time
            threshold_ns = int(self.pose_log_interval_sec * 1e9)
            if threshold_ns <= 0:
                threshold_ns = 0
            if elapsed.nanoseconds() >= threshold_ns:
                should_log = True

        if not should_log:
            return

        entries = []
        for finger_name, joint_name in COMMAND_ORDER:
            f_idx = self._finger_index[finger_name]
            j_idx = JOINT_NAMES[finger_name].index(joint_name)
            qpos = qpos_values[f_idx][j_idx]
            if math.isnan(qpos):
                continue
            entries.append(f"{finger_name}_{joint_name}={qpos:.3f}")

        if entries:
            self.get_logger().info("SenseGlove qpos(rad): " + ", ".join(entries))
            self._first_pose_logged = True
            self._last_pose_log_time = now

    def _on_set_parameters(self, params: List[Parameter]) -> SetParametersResult:
        """Handle runtime updates for selected parameters (e.g., joint_weights)."""
        success = True
        reason = ""
        try:
            for p in params:
                if p.name == "joint_weights":
                    values: List[float] = []
                    if isinstance(p.value, (list, tuple)):
                        for el in p.value:
                            try:
                                values.append(float(el))
                            except (TypeError, ValueError):
                                continue
                    elif isinstance(p.value, (int, float)):
                        values = [float(p.value)]
                    if not values:
                        values = [1.0] * COMMAND_COUNT
                    if len(values) < COMMAND_COUNT:
                        values.extend([values[-1]] * (COMMAND_COUNT - len(values)))
                    self.joint_weights = [clamp(v, 0.0, 1.0) for v in values[:COMMAND_COUNT]]
                    # Log a concise summary for quick verification
                    try:
                        thumb = tuple(round(v, 2) for v in self.joint_weights[0:3])
                        index = tuple(round(v, 2) for v in self.joint_weights[3:6])
                        middle = tuple(round(v, 2) for v in self.joint_weights[6:9])
                        self.get_logger().info(f"[Weights] THUMB={thumb} INDEX={index} MIDDLE={middle}")
                    except Exception:
                        pass
        except Exception as exc:
            success = False
            reason = str(exc)
        return SetParametersResult(successful=success, reason=reason)

    # ------------------------------------------------------------------
    # Key handling helpers
    # ------------------------------------------------------------------
    def _setup_terminal_reader(self) -> bool:
        if termios is None or tty is None:
            return False

        fd: Optional[int] = None
        stream: Optional[IO[str]] = None
        opened_stream = False

        try:
            if sys.stdin.isatty():
                stream = sys.stdin
                fd = sys.stdin.fileno()
            else:
                stream = open("/dev/tty")
                opened_stream = True
                fd = stream.fileno()
        except Exception as exc:
            if opened_stream and stream is not None:
                try:
                    stream.close()
                except Exception:
                    pass
            self.get_logger().warn(f"TTY 열기 실패: {exc}")
            return False

        if fd is None or stream is None:
            if opened_stream and stream is not None:
                try:
                    stream.close()
                except Exception:
                    pass
            return False

        try:
            old_attrs = termios.tcgetattr(fd)
            tty.setcbreak(fd)
        except Exception as exc:
            if opened_stream:
                try:
                    stream.close()
                except Exception:
                    pass
            self.get_logger().warn(f"TTY cbreak 설정 실패: {exc}")
            return False

        self._terminal_fd = fd
        self._terminal_old_attrs = list(old_attrs)
        self._terminal_stream = stream
        return True

    def _terminal_input_loop(self) -> None:
        prefix = "[Term/Key]"
        fd = self._terminal_fd
        stream = self._terminal_stream
        if fd is None or stream is None:
            return
        while not self._terminal_stop_event.is_set():
            try:
                ready, _, _ = select.select([fd], [], [], 0.25)
            except Exception:
                break
            if not ready:
                continue
            try:
                data = os.read(fd, 1)
            except Exception:
                break
            if not data:
                break
            try:
                ch = data.decode(errors="ignore")
            except Exception:
                continue
            if ch in ("\n", "\r"):
                continue
            if ch == "\x03":  # Ctrl-C
                continue
            self._process_key_command(ch, origin="terminal", prefix=prefix)

    def _process_key_command(self, ch: str, origin: str, prefix: Optional[str] = None) -> None:
        if not ch:
            return
        key = ch.lower()
        if not key.isprintable():
            return

        if prefix is None:
            prefix = "[MuJoCo/Key]" if origin == "viewer" else "[Term/Key]"

        publish = False

        if key == "q":
            publish = True
            self.get_logger().info(f"{prefix} requested 'q' (use Ctrl+C to stop node)")
        elif key == "r":
            publish = True
            self._viz_thumb_complement = not self._viz_thumb_complement
            self.get_logger().info(f"{prefix} complement (thumb) -> {self._viz_thumb_complement}")
        elif key == "s":
            publish = True
            self.get_logger().info(f"{prefix} [Logger] toggle request")
        elif key == "h":
            publish = True
            self.units_publish_enabled = not self.units_publish_enabled
            state = "ON" if self.units_publish_enabled else "OFF"
            self.get_logger().info(f"{prefix} [Units] publish -> {state}")
            self._publish_units_state()
        elif key == "c":
            publish = True
            if self._capture_zero_reference():
                self.get_logger().info(f"{prefix} [Zero] zero_qpos_ref updated from current glove pose")
            else:
                self.get_logger().warn(f"{prefix} [Zero] zero capture failed (no pose yet)")
        elif key == "j":
            publish = True
            if self._latest_qpos_valid:
                self._log_current_qpos()
            else:
                self.get_logger().warn(f"{prefix} no qpos data yet")
        elif key == "t":
            publish = True
            self._cycle_thumb_orientation(prefix)
        elif key == "o":
            publish = True
            index_orientation, middle_orientation = self._flip_non_thumb_orientation()
            self.get_logger().info(
                f"{prefix} [Orient] index/middle sign -> {index_orientation}, {middle_orientation}"
            )
        elif key == "x":
            publish = True
            self.global_qpos_sign *= -1.0
            self.get_logger().info(f"{prefix} global_qpos_sign -> {self.global_qpos_sign:+.0f}")
        elif key == "a":
            publish = True
            self.get_logger().info(f"{prefix} signed_angle mapping active")
        elif key == "g":
            publish = True
            choices = [0.25, 0.5, 0.75, 1.0]
            try:
                idx = choices.index(self.qpos_gain)
            except ValueError:
                idx = 0
            self.qpos_gain = choices[(idx + 1) % len(choices)]
            self.get_logger().info(f"{prefix} qpos_gain -> {self.qpos_gain:.2f}")
        else:
            return

        if publish:
            self._publish_key(key)

    def _publish_key(self, key: str) -> None:
        if not key:
            return
        try:
            # Mark last published key to avoid processing our own echo
            self._last_published_key = key.lower()
            self._last_published_time = self.get_clock().now()
            msg = String()
            msg.data = key
            self.key_pub.publish(msg)
        except Exception:
            pass

    def _cycle_thumb_orientation(self, prefix: str) -> None:
        thumb_idx = self._finger_index["THUMB"]
        self._thumb_pattern_idx = (self._thumb_pattern_idx + 1) % len(self._thumb_sign_patterns)
        new_pattern = list(self._thumb_sign_patterns[self._thumb_pattern_idx])
        self._joint_orientation[thumb_idx] = new_pattern

        for joint_idx in range(3):
            self._clamp_notified[thumb_idx][joint_idx] = False

        thumb_qpos = self._latest_smoothed_qpos[thumb_idx]
        angle_segments: List[str] = []
        for joint_idx, joint_name in enumerate(JOINT_NAMES["THUMB"]):
            value = thumb_qpos[joint_idx]
            if math.isfinite(value):
                angle_segments.append(f"{joint_name}={value:+.3f}rad")
            else:
                angle_segments.append(f"{joint_name}=--")
        angle_info = ", ".join(angle_segments)
        self.get_logger().info(
            f"{prefix} [Thumb] sign pattern -> {tuple(new_pattern)} | angles={angle_info}"
        )

    # ------------------------------------------------------------------
    # MuJoCo helpers
    # ------------------------------------------------------------------
    def _init_mujoco(self) -> None:
        if not self.run_mujoco:
            return
        if mj is None:
            self.get_logger().warn("MuJoCo Python bindings not available; run_mujoco disabled.")
            return
        if not self.mujoco_model_path:
            self.get_logger().warn("MuJoCo model path is empty; skipping MuJoCo setup.")
            return
        try:
            self._mj_model = mj.MjModel.from_xml_path(self.mujoco_model_path)  # type: ignore[attr-defined]
            self._mj_data = mj.MjData(self._mj_model)  # type: ignore[attr-defined]
            self._mj_enabled = True
        except Exception as exc:  # pragma: no cover - relies on external simulator
            self.get_logger().warn(f"[MuJoCo] init failed: {exc}")
            self._mj_model = None
            self._mj_data = None
            self._mj_enabled = False
            return
        # Resolve joint addresses for the DClaw model once during init
        for finger, names in DCLAW_JOINTS.items():
            for name in names:
                try:
                    joint_id = mj.mj_name2id(self._mj_model, mj.mjtObj.mjOBJ_JOINT, name)  # type: ignore[attr-defined]
                except Exception:
                    continue
                if joint_id < 0:
                    continue
                adr = int(self._mj_model.jnt_qposadr[joint_id])
                self._mj_qpos_adr[name] = adr
        # Attempt to launch the passive viewer (optional)
        if mj_viewer is not None:
            try:
                self._mj_viewer = mj_viewer.launch_passive(
                    self._mj_model,
                    self._mj_data,
                    show_left_ui=False,
                    show_right_ui=False,
                    key_callback=None,
                )  # type: ignore[attr-defined]
                self.get_logger().info("[MuJoCo] viewer started")
            except Exception as exc:
                self._mj_viewer = None
                self.get_logger().warn(f"[MuJoCo] viewer start failed, continuing headless: {exc}")
        # Identify EE pose handles if requested
        if self.ee_pose_publish_enabled:
            try:
                if self.ee_pose_mj_site:
                    sid = mj.mj_name2id(self._mj_model, mj.mjtObj.mjOBJ_SITE, self.ee_pose_mj_site)  # type: ignore[attr-defined]
                    if sid >= 0:
                        self._ee_site_id = int(sid)
                        self.get_logger().info(f"[EE] MuJoCo site: {self.ee_pose_mj_site} (id={self._ee_site_id})")
                if self._ee_site_id is None and self.ee_pose_mj_body:
                    bid = mj.mj_name2id(self._mj_model, mj.mjtObj.mjOBJ_BODY, self.ee_pose_mj_body)  # type: ignore[attr-defined]
                    if bid >= 0:
                        self._ee_body_id = int(bid)
                        self.get_logger().info(f"[EE] MuJoCo body: {self.ee_pose_mj_body} (id={self._ee_body_id})")
                if self.ee_pose_mj_site_th:
                    sid_th = mj.mj_name2id(self._mj_model, mj.mjtObj.mjOBJ_SITE, self.ee_pose_mj_site_th)  # type: ignore[attr-defined]
                    if sid_th >= 0:
                        self._ee_site_id_th = int(sid_th)
                        self.get_logger().info(f"[EE] MuJoCo site(TH): {self.ee_pose_mj_site_th} (id={self._ee_site_id_th})")
            except Exception as exc:
                self.get_logger().warn(f"[EE] MuJoCo identifiers lookup failed: {exc}")

    def _apply_to_mujoco(self, smoothed_qpos: List[List[float]]) -> None:
        if not self._mj_enabled or self._mj_model is None or self._mj_data is None:
            return

        for finger_name, joint_names in DCLAW_JOINTS.items():
            f_idx = self._finger_index.get(finger_name)
            if f_idx is None:
                continue
            for j_idx, mj_joint in enumerate(joint_names):
                if j_idx >= len(JOINT_NAMES[finger_name]):
                    continue
                adr = self._mj_qpos_adr.get(mj_joint)
                if adr is None:
                    continue
                try:
                    val = smoothed_qpos[f_idx][j_idx]
                except IndexError:
                    continue
                if not math.isfinite(val):
                    continue
                applied = float(val)
                if adr in [0, 3, 6]:
                    applied -= 1.57
                else:
                    applied -= 3.14
                self._mj_data.qpos[adr] = applied

        try:
            mj.mj_forward(self._mj_model, self._mj_data)  # type: ignore[attr-defined]
        except Exception as exc:
            if not self._mj_forward_warned:
                self.get_logger().warn(f"[MuJoCo] forward failed: {exc}")
                self._mj_forward_warned = True
            return

        if self._mj_viewer is not None:
            try:
                if self._mj_viewer.is_running():  # type: ignore[attr-defined]
                    self._mj_viewer.sync()  # type: ignore[attr-defined]
            except Exception:
                pass

        if self.ee_pose_publish_enabled and self.ee_pose_source == "mujoco":
            self._publish_ee_pose()

    def _capture_zero_reference(self) -> bool:
        updated_any = False
        missing: List[str] = []
        for finger_idx in range(len(FINGER_NAMES)):
            for joint_idx in range(3):
                try:
                    initial_val = self._initial_zero_qpos_ref[finger_idx][joint_idx]
                except IndexError:
                    initial_val = 0.0
                glove_val = self._latest_glove_positions[finger_idx][joint_idx]
                if not math.isfinite(glove_val):
                    finger_name = FINGER_NAMES[finger_idx]
                    joint_name = JOINT_NAMES[finger_name][joint_idx]
                    missing.append(f"{finger_name}_{joint_name}")
                    continue
                new_zero = initial_val + glove_val
                self._zero_qpos_ref[finger_idx][joint_idx] = new_zero
                self._latest_raw_qpos[finger_idx][joint_idx] = new_zero
                self._prev_qpos_cmd[finger_idx][joint_idx] = 0.0
                self._latest_smoothed_qpos[finger_idx][joint_idx] = 0.0
                self._clamp_notified[finger_idx][joint_idx] = False
                updated_any = True

        if not updated_any:
            return False
        if missing:
            detail = ", ".join(missing)
            self.get_logger().warn(
                f"[Zero] 일부 관절의 SenseGlove 측정값이 없어 기존 값을 유지합니다: {detail}"
            )
        return True

    def _log_current_qpos(self) -> None:
        lines = []
        for f_idx, finger_name in enumerate(FINGER_NAMES):
            joint_names = JOINT_NAMES[finger_name]
            segments = []
            for j_idx, joint_name in enumerate(joint_names):
                try:
                    val = self._latest_smoothed_qpos[f_idx][j_idx]
                except IndexError:
                    val = 0.0
                segments.append(f"{joint_name}={val:+.3f}")
            lines.append(f"{finger_name}: " + ", ".join(segments))
        if lines:
            self.get_logger().info("[QPOS]\n" + "\n".join(lines))
    def _publish_ee_pose(self) -> None:
        if self._mj_data is None:
            return

        timestamp = self.get_clock().now().to_msg()

        def _send_pose(pub: Optional[Any], position: Optional[Sequence[float]]) -> None:
            if pub is None or position is None:
                return
            try:
                msg = PoseStamped()
                msg.header.stamp = timestamp
                msg.header.frame_id = self.ee_pose_frame_id
                msg.pose.position.x = float(position[0])
                msg.pose.position.y = float(position[1])
                msg.pose.position.z = float(position[2])
                msg.pose.orientation.x = 0.0
                msg.pose.orientation.y = 0.0
                msg.pose.orientation.z = 0.0
                msg.pose.orientation.w = 1.0
                pub.publish(msg)
            except Exception:
                pass

        main_pose: Optional[Sequence[float]] = None
        if self._ee_site_id is not None:
            try:
                main_pose = self._mj_data.site_xpos[self._ee_site_id]
            except Exception:
                main_pose = None
        elif self._ee_body_id is not None:
            try:
                main_pose = self._mj_data.xpos[self._ee_body_id]
            except Exception:
                main_pose = None
        elif not self._ee_mj_warned:
            self.get_logger().warn("[EE] MuJoCo site/body not configured; skipping EE pose publish.")
            self._ee_mj_warned = True

        if main_pose is not None:
            try:
                self._last_ee_pose = (
                    float(main_pose[0]),
                    float(main_pose[1]),
                    float(main_pose[2])
                )
            except Exception:
                self._last_ee_pose = None
        else:
            self._last_ee_pose = None

        _send_pose(self.ee_pose_pub, main_pose)
        _send_pose(self.ee_pose_pub_mf, main_pose)

        thumb_pose: Optional[Sequence[float]] = None
        if self._ee_site_id_th is not None:
            try:
                thumb_pose = self._mj_data.site_xpos[self._ee_site_id_th]
            except Exception:
                thumb_pose = None
        _send_pose(self.ee_pose_pub_th, thumb_pose)

    def _on_logger_state(self, msg: Bool) -> None:
        try:
            active = bool(msg.data)
        except Exception:
            active = False
        previous = self._logger_active
        self._logger_active = active
        if previous is None or previous != active:
            state = "ON" if active else "OFF"
            self.get_logger().info(f"[Logger] state -> {state}")

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def destroy_node(self) -> None:
        if self._mujoco_qpos_poll_timer is not None:
            try:
                self._mujoco_qpos_poll_timer.cancel()
            except Exception:
                pass
            self._mujoco_qpos_poll_timer = None
        if self._terminal_thread is not None:
            self._terminal_stop_event.set()
            try:
                if self._terminal_thread.is_alive():
                    self._terminal_thread.join(timeout=0.5)
            except Exception:
                pass
            self._terminal_thread = None
        if self._terminal_fd is not None and self._terminal_old_attrs is not None and termios is not None:
            try:
                termios.tcsetattr(self._terminal_fd, termios.TCSADRAIN, self._terminal_old_attrs)
            except Exception:
                pass
        if self._terminal_stream is not None and self._terminal_stream is not sys.stdin:
            try:
                self._terminal_stream.close()
            except Exception:
                pass
        self._terminal_fd = None
        self._terminal_old_attrs = None
        self._terminal_stream = None
        if self._mj_viewer is not None:
            try:
                self._mj_viewer.close()  # type: ignore[attr-defined]
            except Exception:
                pass
            self._mj_viewer = None
        super().destroy_node()


def main(args: Optional[Sequence[str]] = None) -> None:
    init_args = list(args) if args is not None else None
    rclpy.init(args=init_args)
    node = SenseGloveMJNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
