#!/usr/bin/env python3
"""SenseGlove-to-MuJoCo bridge that consumes SenseGlove joint angles.

This node subscribes to the joint angles published by the C++ SenseGlove node
and maps them to qpos / unit commands for the downstream hardware interface or
simulator. It mirrors the behaviour of the original hand_tracker_node.py but
without any camera or MediaPipe dependencies.
"""

from __future__ import annotations

import math
import select
import sys
import threading
from typing import Any, Dict, List, Optional, Sequence, Tuple

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node

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

# DClaw joint names used by the MuJoCo model
DCLAW_JOINTS: Dict[str, Sequence[str]] = {
    "THUMB": ("THJ30", "THJ31", "THJ32"),
    "INDEX": ("FFJ10", "FFJ11", "FFJ12"),
    "MIDDLE": ("MFJ20", "MFJ21", "MFJ22"),
}

DEFAULT_MUJOCO_MODEL_PATH = "/home/songwoo/Desktop/work_dir/realsense_hand_retargetting/universal_robots_ur5e_with_dclaw/dclaw/dclaw3xh.xml"


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


class SenseGloveMJNode(Node):
    def __init__(self) -> None:
        super().__init__("sense_glove_mj_node")

        # Topic configuration
        self.declare_parameter("joint_state_topic", "/hand_tracker/joint_state")
        self.declare_parameter("republish_joint_state_topic", "/hand_tracker/joint_states")
        self.declare_parameter("units_topic", "/hand_tracker/targets_units")
        self.declare_parameter("key_topic", "/hand_tracker/key")
        self.declare_parameter("pose_log_interval_sec", 1.0)
        self.declare_parameter("pose_log_enabled", False)
        self.declare_parameter("log_clamp_events", True)

        # Unit / qpos mapping configuration (mirrors the C++ node defaults)
        self.declare_parameter("units_publish_enabled", True)
        self.declare_parameter("units_baseline", 2000.0)
        self.declare_parameter("units_per_rad", 4096.0 / (2.0 * math.pi))
        self.declare_parameter("units_motion_scale_qpos", 1.0)
        self.declare_parameter("units_min", 0.0)
        self.declare_parameter("units_max", 4095.0)

        self.declare_parameter("qpos_gain", 0.5)
        self.declare_parameter("qpos_smooth_alpha", 0.5)
        self.declare_parameter("qpos_step_max", 0.05)
        self.declare_parameter("clamp_qpos_symm", True)
        self.declare_parameter("clamp_qpos_min", -1.57)
        self.declare_parameter("clamp_qpos_max", 1.57)
        self.declare_parameter("global_qpos_sign", -1.0)

        # MuJoCo / EE pose configuration (optional)
        self.declare_parameter("run_mujoco", True)
        self.declare_parameter("mujoco_model_path", DEFAULT_MUJOCO_MODEL_PATH)
        self.declare_parameter("ee_pose_publish_enabled", False)
        self.declare_parameter("ee_pose_topic", "/ee_pose")
        self.declare_parameter("ee_pose_topic_mf", "/ee_pose_mf")
        self.declare_parameter("ee_pose_topic_th", "/ee_pose_th")
        self.declare_parameter("ee_pose_frame_id", "world")
        self.declare_parameter("ee_pose_source", "mujoco")
        self.declare_parameter("ee_pose_mj_site", "MFtip")
        self.declare_parameter("ee_pose_mj_body", "")
        self.declare_parameter("ee_pose_mj_site_th", "THtip")
        self.declare_parameter("terminal_status_interval_sec", 1.0)

        self.joint_state_topic: str = self.get_parameter("joint_state_topic").value
        self.republish_joint_state_topic: str = self.get_parameter("republish_joint_state_topic").value
        self.units_topic: str = self.get_parameter("units_topic").value
        self.key_topic: str = self.get_parameter("key_topic").value
        self.pose_log_interval_sec: float = max(0.0, float(self.get_parameter("pose_log_interval_sec").value))
        self.pose_log_enabled: bool = bool(self.get_parameter("pose_log_enabled").value)
        self.log_clamp_events: bool = bool(self.get_parameter("log_clamp_events").value)

        self.units_publish_enabled: bool = bool(self.get_parameter("units_publish_enabled").value)
        self.units_baseline: float = float(self.get_parameter("units_baseline").value)
        self.units_per_rad: float = float(self.get_parameter("units_per_rad").value)
        self.units_motion_scale_qpos: float = float(self.get_parameter("units_motion_scale_qpos").value)
        self.units_min: float = float(self.get_parameter("units_min").value)
        self.units_max: float = float(self.get_parameter("units_max").value)

        self.qpos_gain: float = float(self.get_parameter("qpos_gain").value)
        self.qpos_smooth_alpha: float = clamp(float(self.get_parameter("qpos_smooth_alpha").value), 0.0, 1.0)
        self.qpos_step_max: float = max(0.0, float(self.get_parameter("qpos_step_max").value))
        self.clamp_qpos_symm: bool = bool(self.get_parameter("clamp_qpos_symm").value)
        self.clamp_qpos_min: float = float(self.get_parameter("clamp_qpos_min").value)
        self.clamp_qpos_max: float = float(self.get_parameter("clamp_qpos_max").value)
        self.global_qpos_sign: float = float(self.get_parameter("global_qpos_sign").value)

        self.run_mujoco: bool = bool(self.get_parameter("run_mujoco").value)
        self.mujoco_model_path: str = str(self.get_parameter("mujoco_model_path").value)
        if not self.mujoco_model_path:
            self.mujoco_model_path = DEFAULT_MUJOCO_MODEL_PATH
        self.ee_pose_publish_enabled: bool = bool(self.get_parameter("ee_pose_publish_enabled").value)
        self.ee_pose_topic: str = str(self.get_parameter("ee_pose_topic").value)
        self.ee_pose_topic_mf: str = str(self.get_parameter("ee_pose_topic_mf").value)
        self.ee_pose_topic_th: str = str(self.get_parameter("ee_pose_topic_th").value)
        self.ee_pose_frame_id: str = str(self.get_parameter("ee_pose_frame_id").value)
        self.ee_pose_source: str = str(self.get_parameter("ee_pose_source").value).lower().strip()
        self.ee_pose_mj_site: str = str(self.get_parameter("ee_pose_mj_site").value)
        self.ee_pose_mj_body: str = str(self.get_parameter("ee_pose_mj_body").value)
        self.ee_pose_mj_site_th: str = str(self.get_parameter("ee_pose_mj_site_th").value)
        self.terminal_status_interval_sec: float = max(0.0, float(self.get_parameter("terminal_status_interval_sec").value))

        # Internal state
        # Per-finger orientation overrides (mirrors hand_tracker defaults)
        self.declare_parameter("joint_orientation_thumb", [1., 1., 1.])
        self.declare_parameter("joint_orientation_index", [1., 1., 1.])
        self.declare_parameter("joint_orientation_middle", [1., 1., 1.])

        self._finger_index = {name: idx for idx, name in enumerate(FINGER_NAMES)}
        self._joint_orientation: List[List[float]] = [
            self._load_orientation_param("joint_orientation_thumb", (1., 1., 1.)),
            self._load_orientation_param("joint_orientation_index", (1., 1., 1.)),
            self._load_orientation_param("joint_orientation_middle", (1., 1., 1.)),
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
        self._thumb_pattern_idx = 0
        self._viz_thumb_complement = False
        self._zero_qpos_ref: List[List[float]] = [[0.0] * 3 for _ in FINGER_NAMES]
        self._prev_qpos_cmd: List[List[float]] = [[0.0] * 3 for _ in FINGER_NAMES]
        self._latest_smoothed_qpos: List[List[float]] = [[0.0] * 3 for _ in FINGER_NAMES]
        self._latest_raw_qpos: List[List[float]] = [[0.0] * 3 for _ in FINGER_NAMES]
        self._latest_qpos_valid: bool = False
        self._latest_raw_valid: bool = False
        self._first_pose_logged: bool = False
        self._last_pose_log_time: Optional[rclpy.time.Time] = None
        self._clamp_notified: List[List[bool]] = [[False] * 3 for _ in FINGER_NAMES]
        self._logger_active: Optional[bool] = None
        self._time_start: Optional[Tuple[int, int]] = None
        self._last_ee_pose: Optional[Tuple[float, float, float]] = None
        self._latest_angles_deg: List[List[float]] = [[math.nan] * 3 for _ in FINGER_NAMES]
        self._last_status_overlay: Optional[Tuple[str, str]] = None

        # ROS entities
        self.units_pub = self.create_publisher(Int32MultiArray, self.units_topic, 10)
        self.key_pub = self.create_publisher(String, self.key_topic, 10)
        self.joint_state_sub = self.create_subscription(JointState, self.joint_state_topic, self._on_joint_state, 10)
        self.joint_state_repub = None
        if self.republish_joint_state_topic:
            self.joint_state_repub = self.create_publisher(JointState, self.republish_joint_state_topic, 10)

        self.ee_pose_pub = None
        self.ee_pose_pub_mf = None
        self.ee_pose_pub_th = None
        if self.ee_pose_publish_enabled and self.ee_pose_topic:
            self.ee_pose_pub = self.create_publisher(PoseStamped, self.ee_pose_topic, 10)
        if self.ee_pose_publish_enabled and self.ee_pose_topic_mf:
            self.ee_pose_pub_mf = self.create_publisher(PoseStamped, self.ee_pose_topic_mf, 10)
        if self.ee_pose_publish_enabled and self.ee_pose_topic_th:
            self.ee_pose_pub_th = self.create_publisher(PoseStamped, self.ee_pose_topic_th, 10)

        self.zero_srv = self.create_service(Trigger, "set_zero", self._handle_zero_request)
        self.get_logger().info("SenseGlove MJ bridge ready (waiting for joint states)...")

        try:
            self.create_subscription(Bool, "/data_logger/logging_active", self._on_logger_state, 10)
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
        if self.enable_terminal_input and sys.stdin.isatty():
            self._terminal_thread = threading.Thread(target=self._terminal_input_loop, name="sg_mj_terminal", daemon=True)
            self._terminal_thread.start()
        elif not sys.stdin.isatty():
            self.get_logger().warn("STDIN이 TTY가 아니라 터미널 키 입력을 사용할 수 없습니다. 필요시 ros2 run 명령을 직접 실행해 주세요.")

        self._init_mujoco()
        self._log_key_shortcuts()

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------
    def _log_key_shortcuts(self) -> None:
        shortcuts = [
            "h: toggle units publish",
            "s: toggle data logger",
            "c: capture zero reference",
            "t: cycle thumb orientation",
            "r: toggle thumb complement visual",
            "o: flip index/middle orientation",
            "x: flip global qpos sign",
            "g: cycle qpos gain",
            "j: print current qpos",
        ]
        self.get_logger().info("[Keys] " + " | ".join(shortcuts))

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
            tuple(self._joint_orientation[index_idx]),
            tuple(self._joint_orientation[middle_idx]),
        )

    # ------------------------------------------------------------------
    # ROS callbacks
    # ------------------------------------------------------------------
    def _on_joint_state(self, msg: JointState) -> None:
        if self.joint_state_repub is not None:
            self.joint_state_repub.publish(msg)

        angles_deg = [[math.nan] * 3 for _ in FINGER_NAMES]

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
            angles_deg[f_idx][j_idx] = math.degrees(position)

        self._process_angles(angles_deg)

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
    def _process_angles(self, angles_deg: List[List[float]]) -> None:
        self._latest_angles_deg = [row[:] for row in angles_deg]
        smoothed_qpos = [row[:] for row in self._latest_smoothed_qpos]
        raw_qpos = [row[:] for row in self._latest_raw_qpos]
        units_out: List[int] = []
        any_valid = False
        raw_all_valid = True

        for finger_name, joint_name in COMMAND_ORDER:
            f_idx = self._finger_index[finger_name]
            j_idx = JOINT_NAMES[finger_name].index(joint_name)
            angle_deg = angles_deg[f_idx][j_idx]
            mapped_qpos = self._map_angle_to_qpos(f_idx, j_idx, angle_deg)

            if math.isnan(angle_deg):
                raw_all_valid = False
                units_out.append(int(round(self.units_baseline)))
                continue

            any_valid = True
            raw_qpos[f_idx][j_idx] = mapped_qpos
            qpos = mapped_qpos - self._zero_qpos_ref[f_idx][j_idx]

            previous = self._prev_qpos_cmd[f_idx][j_idx]
            smoothed = (1.0 - self.qpos_smooth_alpha) * previous + self.qpos_smooth_alpha * qpos
            delta = smoothed - previous
            if delta > self.qpos_step_max:
                smoothed = previous + self.qpos_step_max
            elif delta < -self.qpos_step_max:
                smoothed = previous - self.qpos_step_max

            was_clamped = False
            if self.clamp_qpos_symm:
                unclamped = smoothed
                smoothed = clamp(smoothed, self.clamp_qpos_min, self.clamp_qpos_max)
                was_clamped = abs(smoothed - unclamped) > 1e-6

            if (
                was_clamped
                and self.log_clamp_events
                and not self._clamp_notified[f_idx][j_idx]
            ):
                self._clamp_notified[f_idx][j_idx] = True
                self.get_logger().warn(
                    "[Clamp] %s_%s saturated at %.3f rad (raw=%.3f rad). "
                    "Adjust clamp_qpos_min/max or qpos_gain if this is unintended.",
                    finger_name,
                    joint_name,
                    smoothed,
                    qpos,
                )

            self._prev_qpos_cmd[f_idx][j_idx] = smoothed
            smoothed_qpos[f_idx][j_idx] = smoothed

            units_val = self.units_baseline + smoothed * self.units_per_rad * self.units_motion_scale_qpos
            units_val = clamp(units_val, self.units_min, self.units_max)
            units_out.append(int(round(units_val)))

        if any_valid:
            self._latest_raw_qpos = raw_qpos
            self._latest_raw_valid = raw_all_valid
            self._latest_smoothed_qpos = smoothed_qpos
            self._latest_qpos_valid = True
            self._maybe_log_pose(angles_deg)
            if self.units_publish_enabled:
                msg = Int32MultiArray()
                msg.data = units_out
                self.units_pub.publish(msg)
            if self._mj_enabled:
                self._apply_to_mujoco(smoothed_qpos)
        else:
            self._latest_qpos_valid = False
            self._latest_raw_valid = False

        self._update_status_overlay()

    def _map_angle_to_qpos(self, f_idx: int, j_idx: int, angle_deg: float) -> float:
        direction = self._joint_orientation[f_idx][j_idx]
        return self._map_angle_with_direction(f_idx, j_idx, angle_deg, direction)

    def _map_angle_with_direction(
        self,
        f_idx: int,
        j_idx: int,
        angle_deg: float,
        direction: float,
        apply_clamp: bool = True,
    ) -> float:
        if math.isnan(angle_deg):
            return 0.0

        if -180.0 <= angle_deg <= 180.0:
            centered_deg = angle_deg
        else:
            centered_deg = clamp(angle_deg, 0.0, 360.0) - 180.0

        qpos = math.radians(centered_deg) * direction * self.global_qpos_sign
        qpos *= self.qpos_gain

        if apply_clamp and self.clamp_qpos_symm:
            qpos = clamp(qpos, self.clamp_qpos_min, self.clamp_qpos_max)
        return qpos

    def _maybe_log_pose(self, angles_deg: List[List[float]]) -> None:
        if not self.pose_log_enabled:
            return

        now = self.get_clock().now()
        should_log = False

        if not self._first_pose_logged:
            should_log = True
        elif self.pose_log_interval_sec > 0.0 and self._last_pose_log_time is not None:
            elapsed = now - self._last_pose_log_time
            if elapsed >= Duration(seconds=self.pose_log_interval_sec):
                should_log = True

        if not should_log:
            return

        entries = []
        for finger_name, joint_name in COMMAND_ORDER:
            f_idx = self._finger_index[finger_name]
            j_idx = JOINT_NAMES[finger_name].index(joint_name)
            angle_deg = angles_deg[f_idx][j_idx]
            if math.isnan(angle_deg):
                continue
            entries.append(f"{finger_name}_{joint_name}={angle_deg:.1f}")

        if entries:
            self.get_logger().info("SenseGlove angles(deg): " + ", ".join(entries))
            self._first_pose_logged = True
            self._last_pose_log_time = now

    def _update_status_overlay(self, force: bool = False) -> None:
        if mj_viewer is None or self._mj_viewer is None:
            self._last_status_overlay = None
            return

        units_state = "ON" if self.units_publish_enabled else "OFF"
        if self._logger_active is None:
            log_state = "WAIT"
        else:
            log_state = "ON" if self._logger_active else "OFF"

        line1 = f"Units (h): {units_state}"
        line2 = f"Logger (s): {log_state}"

        try:
            self._mj_viewer.add_overlay(mj_viewer.Overlay.GridTopLeft, line1, line2)  # type: ignore[attr-defined]
            self._last_status_overlay = (line1, line2)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Key handling helpers
    # ------------------------------------------------------------------
    def _terminal_input_loop(self) -> None:
        prefix = "[Term/Key]"
        while not self._terminal_stop_event.is_set():
            try:
                ready, _, _ = select.select([sys.stdin], [], [], 0.25)
            except Exception:
                break
            if not ready:
                continue
            try:
                ch = sys.stdin.read(1)
            except Exception:
                break
            if not ch:
                break
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
            self._update_status_overlay(force=True)
            self.get_logger().info(f"{prefix} [Logger] toggle request")
        elif key == "h":
            publish = True
            self.units_publish_enabled = not self.units_publish_enabled
            state = "ON" if self.units_publish_enabled else "OFF"
            self._update_status_overlay(force=True)
            self.get_logger().info(f"{prefix} [Units] publish -> {state}")
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

        thumb_angles = self._latest_angles_deg[thumb_idx]
        angle_segments: List[str] = []
        for joint_name, angle in zip(JOINT_NAMES["THUMB"], thumb_angles):
            if math.isfinite(angle):
                angle_segments.append(f"{joint_name}={angle:+.1f}deg")
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
                self.get_logger().info("[MuJoCo] 키 입력은 터미널에서만 처리됩니다 (viewer key callback 미사용)")
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

        self._update_status_overlay(force=True)

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
                self._mj_data.qpos[adr] = float(val)

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

        self._update_status_overlay()

    def _capture_zero_reference(self) -> bool:
        if not self._latest_qpos_valid:
            return False

        updated_any = False
        for finger_idx in range(len(FINGER_NAMES)):
            for joint_idx in range(3):
                value = self._latest_raw_qpos[finger_idx][joint_idx]
                if not math.isfinite(value):
                    continue
                self._zero_qpos_ref[finger_idx][joint_idx] = value
                self._prev_qpos_cmd[finger_idx][joint_idx] = 0.0
                self._latest_smoothed_qpos[finger_idx][joint_idx] = 0.0
                self._clamp_notified[finger_idx][joint_idx] = False
                updated_any = True

        if not updated_any:
            return False

        missing = []
        for finger_idx, finger_name in enumerate(FINGER_NAMES):
            for joint_idx, joint_name in enumerate(JOINT_NAMES[finger_name]):
                if not math.isfinite(self._latest_raw_qpos[finger_idx][joint_idx]):
                    missing.append(f"{finger_name}_{joint_name}")
        if missing:
            detail = ", ".join(missing)
            self.get_logger().warn(
                f"[Zero] 일부 관절 값이 유효하지 않아 이전 값을 유지합니다: {detail}"
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
        self._logger_active = active
        if active:
            try:
                now_msg = self.get_clock().now().to_msg()
                self._time_start = (int(now_msg.sec), int(now_msg.nanosec))
            except Exception:
                self._time_start = None
        else:
            self._time_start = None
        self._update_status_overlay(force=True)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def destroy_node(self) -> None:
        if self._terminal_thread is not None:
            self._terminal_stop_event.set()
            try:
                if self._terminal_thread.is_alive():
                    self._terminal_thread.join(timeout=0.5)
            except Exception:
                pass
            self._terminal_thread = None
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
