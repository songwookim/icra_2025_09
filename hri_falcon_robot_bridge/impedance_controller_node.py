#!/usr/bin/env python3
"""
Impedance Controller Node

Subscribes to:
- /impedance_control/target_stiffness (Float32MultiArray) - 9D stiffness from policy
- /hand_tracker/qpos (JointState) - current joint positions
- /ee_pose_desired_{if|mf|th} (PoseStamped) - desired end-effector positions

Publishes to:
- /hand_tracker/targets_units (Int32MultiArray) - joint commands for robot

Implements Cartesian impedance control: F = K * (x_des - x_cur), then maps to joint space via Jacobian.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray, Int32MultiArray

# Try importing mujoco
try:
    import mujoco as mj

    MUJOCO_AVAILABLE = True
except ImportError:
    MUJOCO_AVAILABLE = False
    mj = None

# Default paths
DEFAULT_MUJOCO_MODEL = "/home/songwoo/git/ur_dclaw/dclaw_finger_description/urdf/dclaw_finger_mjcf_final.xml"

# Joint order (matching robot_controller_node)
JOINT_ORDER = [
    ("thumb", "cmc"),
    ("thumb", "mcp"),
    ("thumb", "ip"),
    ("index", "mcp"),
    ("index", "pip"),
    ("index", "dip"),
    ("middle", "mcp"),
    ("middle", "pip"),
    ("middle", "dip"),
]

# MuJoCo joint names
DCLAW_JOINTS = {
    "thumb": ["THJ30", "THJ31", "THJ32"],
    "index": ["FFJ10", "FFJ11", "FFJ12"],
    "middle": ["MFJ20", "MFJ21", "MFJ22"],
}

# Default baselines
DEFAULT_UNITS_BASELINE = [
    1000.0, 2000.0, 2000.0,  # thumb
    1000.0, 2000.0, 2000.0,  # index
    1000.0, 2000.0, 2000.0,  # middle
]


class ImpedanceControllerNode(Node):
    """Impedance controller using learned stiffness."""

    def __init__(self):
        super().__init__("impedance_controller_node")

        # Parameters
        self.declare_parameter("rate_hz", 50.0)
        self.declare_parameter("use_mujoco", True)
        self.declare_parameter("mujoco_model_path", DEFAULT_MUJOCO_MODEL)
        self.declare_parameter("position_gain", 1.0)
        self.declare_parameter("damping_ratio", 0.7)
        self.declare_parameter("units_per_rad", 4096.0 / (2.0 * math.pi))
        self.declare_parameter("units_min", 0.0)
        self.declare_parameter("units_max", 4095.0)
        self.declare_parameter("units_baseline", list(DEFAULT_UNITS_BASELINE))
        self.declare_parameter("smooth_alpha", 0.3)
        self.declare_parameter("max_step_units", 50.0)

        # Get parameters
        self.rate_hz = float(self.get_parameter("rate_hz").value)
        self.use_mujoco = bool(self.get_parameter("use_mujoco").value)
        self.mujoco_model_path = str(self.get_parameter("mujoco_model_path").value)
        self.position_gain = float(self.get_parameter("position_gain").value)
        self.damping_ratio = float(self.get_parameter("damping_ratio").value)
        self.units_per_rad = float(self.get_parameter("units_per_rad").value)
        self.units_min = float(self.get_parameter("units_min").value)
        self.units_max = float(self.get_parameter("units_max").value)
        self.units_baseline = list(self.get_parameter("units_baseline").value)
        self.smooth_alpha = float(self.get_parameter("smooth_alpha").value)
        self.max_step_units = float(self.get_parameter("max_step_units").value)

        # State
        self.target_stiffness = np.zeros(9)
        self.current_qpos = np.zeros(9)
        self.desired_ee_pos = {"if": np.zeros(3), "mf": np.zeros(3), "th": np.zeros(3)}
        self.current_ee_pos = {"if": np.zeros(3), "mf": np.zeros(3), "th": np.zeros(3)}
        self.last_units_cmd = np.array(self.units_baseline, dtype=float)

        self.has_stiffness = False
        self.has_qpos = False
        self.has_desired_ee = {"if": False, "mf": False, "th": False}

        # MuJoCo
        self.mj_model = None
        self.mj_data = None
        self.mj_qpos_adr = {}

        if self.use_mujoco:
            self._init_mujoco()

        # Subscribers
        self.stiffness_sub = self.create_subscription(
            Float32MultiArray, "/impedance_control/target_stiffness", self._on_stiffness, 10
        )

        self.qpos_sub = self.create_subscription(JointState, "/hand_tracker/qpos", self._on_qpos, 10)

        self.ee_des_if_sub = self.create_subscription(
            PoseStamped, "/ee_pose_desired_if", self._on_desired_ee_if, 10
        )

        self.ee_des_mf_sub = self.create_subscription(
            PoseStamped, "/ee_pose_desired_mf", self._on_desired_ee_mf, 10
        )

        self.ee_des_th_sub = self.create_subscription(
            PoseStamped, "/ee_pose_desired_th", self._on_desired_ee_th, 10
        )

        # Publisher
        self.units_pub = self.create_publisher(Int32MultiArray, "/hand_tracker/targets_units", 10)

        # Timer
        period = 1.0 / self.rate_hz
        self.control_timer = self.create_timer(period, self._control_callback)

        self._log_counter = 0

        self.get_logger().info(
            f"ImpedanceController: rate={self.rate_hz}Hz, "
            f"mujoco={'enabled' if self.use_mujoco else 'disabled'}"
        )

    def _init_mujoco(self):
        """Initialize MuJoCo for forward kinematics."""
        if not MUJOCO_AVAILABLE:
            self.get_logger().warn("MuJoCo not available - disabled")
            self.use_mujoco = False
            return

        try:
            self.mj_model = mj.MjModel.from_xml_path(self.mujoco_model_path)
            self.mj_data = mj.MjData(self.mj_model)

            # Map joint names
            for finger, joint_names in DCLAW_JOINTS.items():
                for i, joint_name in enumerate(joint_names):
                    try:
                        joint_id = mj.mj_name2id(self.mj_model, mj.mjtObj.mjOBJ_JOINT, joint_name)
                        if joint_id >= 0:
                            adr = int(self.mj_model.jnt_qposadr[joint_id])
                            self.mj_qpos_adr[f"{finger}_{i}"] = adr
                    except Exception as e:
                        self.get_logger().warn(f"Joint mapping failed ({joint_name}): {e}")

            self.get_logger().info(f"MuJoCo loaded: {self.mujoco_model_path}")

        except Exception as e:
            self.get_logger().error(f"MuJoCo init failed: {e}")
            self.use_mujoco = False

    def _on_stiffness(self, msg: Float32MultiArray):
        """Callback for target stiffness."""
        try:
            if len(msg.data) >= 9:
                self.target_stiffness = np.array(msg.data[:9], dtype=float)
                self.has_stiffness = True
        except Exception as e:
            self.get_logger().warn(f"Stiffness callback error: {e}", throttle_duration_sec=1.0)

    def _on_qpos(self, msg: JointState):
        """Callback for joint positions."""
        try:
            if len(msg.position) >= 9:
                self.current_qpos = np.array(msg.position[:9], dtype=float)
                self.has_qpos = True

                if self.use_mujoco:
                    self._update_current_ee_positions()

        except Exception as e:
            self.get_logger().warn(f"Qpos callback error: {e}", throttle_duration_sec=1.0)

    def _on_desired_ee_if(self, msg: PoseStamped):
        """Callback for desired index finger EE position."""
        try:
            pos = msg.pose.position
            self.desired_ee_pos["if"] = np.array([pos.x, pos.y, pos.z], dtype=float)
            self.has_desired_ee["if"] = True
        except Exception as e:
            self.get_logger().warn(f"Desired EE (if) error: {e}", throttle_duration_sec=1.0)

    def _on_desired_ee_mf(self, msg: PoseStamped):
        """Callback for desired middle finger EE position."""
        try:
            pos = msg.pose.position
            self.desired_ee_pos["mf"] = np.array([pos.x, pos.y, pos.z], dtype=float)
            self.has_desired_ee["mf"] = True
        except Exception as e:
            self.get_logger().warn(f"Desired EE (mf) error: {e}", throttle_duration_sec=1.0)

    def _on_desired_ee_th(self, msg: PoseStamped):
        """Callback for desired thumb EE position."""
        try:
            pos = msg.pose.position
            self.desired_ee_pos["th"] = np.array([pos.x, pos.y, pos.z], dtype=float)
            self.has_desired_ee["th"] = True
        except Exception as e:
            self.get_logger().warn(f"Desired EE (th) error: {e}", throttle_duration_sec=1.0)

    def _update_current_ee_positions(self):
        """Update current EE positions via MuJoCo FK."""
        if not self.use_mujoco or self.mj_data is None:
            return

        try:
            # Set qpos
            for finger_idx, finger in enumerate(["thumb", "index", "middle"]):
                for joint_idx in range(3):
                    key = f"{finger}_{joint_idx}"
                    if key in self.mj_qpos_adr:
                        qpos_idx = finger_idx * 3 + joint_idx
                        offset = 1.57 if qpos_idx in [0, 3, 6] else 3.14
                        self.mj_data.qpos[self.mj_qpos_adr[key]] = self.current_qpos[qpos_idx] - offset

            # Forward kinematics
            mj.mj_forward(self.mj_model, self.mj_data)

            # Get EE positions
            site_names = {"if": "FFtip", "mf": "MFtip", "th": "THtip"}

            for finger, site_name in site_names.items():
                try:
                    site_id = mj.mj_name2id(self.mj_model, mj.mjtObj.mjOBJ_SITE, site_name)
                    if site_id >= 0:
                        self.current_ee_pos[finger] = self.mj_data.site_xpos[site_id].copy()
                except Exception:
                    pass

        except Exception as e:
            self.get_logger().warn(f"FK update error: {e}", throttle_duration_sec=5.0)

    def _compute_jacobian(self, finger: str) -> Optional[np.ndarray]:
        """Compute Jacobian for a finger."""
        if not self.use_mujoco or self.mj_data is None:
            return None

        try:
            site_names = {"if": "FFtip", "mf": "MFtip", "th": "THtip"}
            site_name = site_names.get(finger)
            if not site_name:
                return None

            site_id = mj.mj_name2id(self.mj_model, mj.mjtObj.mjOBJ_SITE, site_name)
            if site_id < 0:
                return None

            # Jacobian matrices
            jacp = np.zeros((3, self.mj_model.nv))
            jacr = np.zeros((3, self.mj_model.nv))

            mj.mj_jac(self.mj_model, self.mj_data, jacp, jacr, self.mj_data.site_xpos[site_id], site_id)

            # Extract columns for this finger
            finger_map = {"th": 0, "if": 1, "mf": 2}
            start_idx = finger_map[finger] * 3
            end_idx = start_idx + 3

            return jacp[:, start_idx:end_idx]

        except Exception as e:
            self.get_logger().warn(f"Jacobian error ({finger}): {e}", throttle_duration_sec=10.0)
            return None

    def _impedance_control_step(self) -> np.ndarray:
        """Compute impedance control command."""
        cmd_units = np.array(self.last_units_cmd, dtype=float)

        if not (self.has_stiffness and self.has_qpos):
            return cmd_units

        if not any(self.has_desired_ee.values()):
            return cmd_units

        # Control per finger
        finger_map = {"th": 0, "if": 1, "mf": 2}

        for finger, finger_idx in finger_map.items():
            if not self.has_desired_ee[finger]:
                continue

            # Stiffness for this finger
            k_start = finger_idx * 3
            k_end = k_start + 3
            K_finger = self.target_stiffness[k_start:k_end]

            # Cartesian error
            ee_error = self.desired_ee_pos[finger] - self.current_ee_pos[finger]

            if self.use_mujoco:
                # Jacobian-based control
                J = self._compute_jacobian(finger)
                # F = K * Δx
                F_cartesian = K_finger * ee_error
                # τ = J^T * F
                tau = J.T @ F_cartesian

                # Convert to position change
                delta_q = self.position_gain * tau

                # Update joints
                for i in range(3):
                    joint_idx = finger_idx * 3 + i
                    new_qpos = self.current_qpos[joint_idx] + delta_q[i]

                    # Convert to units
                    offset = 1.57 if joint_idx in [0, 3, 6] else 3.14
                    adjusted = new_qpos - offset
                    units_calc = adjusted * self.units_per_rad
                    bias = 1000.0 if joint_idx in [0, 3, 6] else 2000.0
                    units_val = units_calc + bias + 12.0
                    units_val = np.clip(units_val, self.units_min, self.units_max)
                    cmd_units[joint_idx] = units_val
        return cmd_units

    def _smooth_command(self, cmd: np.ndarray) -> np.ndarray:
        """Smooth and limit commands."""
        # Exponential smoothing
        smoothed = self.smooth_alpha * cmd + (1 - self.smooth_alpha) * self.last_units_cmd

        # Step limiting
        delta = smoothed - self.last_units_cmd
        delta_clipped = np.clip(delta, -self.max_step_units, self.max_step_units)
        limited = self.last_units_cmd + delta_clipped

        return limited

    def _control_callback(self):
        """Main control loop."""
        try:
            # Compute command
            raw_cmd = self._impedance_control_step()

            # Smooth
            cmd_units = self._smooth_command(raw_cmd)

            # Update
            self.last_units_cmd = cmd_units.copy()

            # Publish
            msg = Int32MultiArray()
            msg.data = [int(round(u)) for u in cmd_units]
            self.units_pub.publish(msg)

            # Log
            self._log_counter += 1
            if self._log_counter % int(self.rate_hz) == 0:
                if self.has_stiffness:
                    self.get_logger().info(
                        f"Impedance: K_avg={np.mean(self.target_stiffness):.1f}, "
                        f"Units: TH={cmd_units[0:3].mean():.0f} "
                        f"IF={cmd_units[3:6].mean():.0f} "
                        f"MF={cmd_units[6:9].mean():.0f}"
                    )

        except Exception as e:
            self.get_logger().error(f"Control error: {e}", throttle_duration_sec=1.0)


def main(args=None):
    rclpy.init(args=args)

    try:
        node = ImpedanceControllerNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback

        traceback.print_exc()
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
