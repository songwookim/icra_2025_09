#!/usr/bin/env python3
"""Torque-Based Impedance Controller Node

구현 목적:
 - 기존 position(admittance) 형태 대신 Dynamixel current control 모드를 활용한 실제 임피던스(스프링-댐퍼) 제어
 - 정책으로부터 전달받은 stiffness(K)를 Cartesian 공간에 적용 후 Jacobian을 통해 관절 토크(=모터 전류)로 변환

제어 법칙(단일 핑거):
    F = K * (x_des - x_cur) + D * (v_des - v_cur)
    τ = J^T * F
전류 변환:
    τ = Kt * I  =>  I(mA) = τ / Kt
    current_units = I(mA) / CURRENT_UNIT

토픽:
  Subscribes:
    - /impedance_control/target_stiffness (Float32MultiArray, 9D)
    - /hand_tracker/qpos (JointState, 9 joint positions & optional velocities)
    - /ee_pose_desired_{if|mf|th} (PoseStamped)
    - /ee_velocity_desired_{if|mf|th} (TwistStamped) [선택적]
  Publishes:
    - /dynamixel/goal_current (Int16MultiArray) : 각 관절 전류 명령 단위
    - /impedance_control/computed_torques (Float32MultiArray) : 모니터링용 계산된 토크

안전 요소:
  - max_torque, max_current_units 클램핑
  - 저역통과 필터(torque_filter_alpha)
  - velocity 추정 시 윈도우 기반 미분 + 필터

주의:
  - 모터별 실제 Torque constant(Kt), CURRENT_UNIT 값은 하드웨어 모델에 따라 다를 수 있음 (XM430 가정)
  - 필요 시 파라미터로 재조정 가능
"""

from __future__ import annotations

import math
from typing import Dict, Optional, List
from dataclasses import dataclass

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, TwistStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray, Int16MultiArray

# Try importing mujoco
try:
    import mujoco as mj  # type: ignore
    MUJOCO_AVAILABLE = True
except ImportError:
    mj = None  # type: ignore
    MUJOCO_AVAILABLE = False

DEFAULT_MUJOCO_MODEL = \
    "/home/songwoo/git/ur_dclaw/dclaw_finger_description/urdf/dclaw_finger_mjcf_final.xml"

# Jacobian 추출 대상 사이트 이름
SITE_NAMES = {"if": "FFtip", "mf": "MFtip", "th": "THtip"}

# MuJoCo 상의 조인트 이름 (3DOF x 3 fingers)
DCLAW_JOINTS = {
    "thumb": ["THJ30", "THJ31", "THJ32"],
    "index": ["FFJ10", "FFJ11", "FFJ12"],
    "middle": ["MFJ20", "MFJ21", "MFJ22"],
}

# Dynamixel (XM430 가정) 상수 (필요 시 파라미터화 가능)
CURRENT_TO_TORQUE = 1.78e-3  # Nm per mA (Torque constant)
CURRENT_UNIT = 2.69          # mA per raw unit
MAX_CURRENT_HW = 1193        # HW limit (±1193) XM430 기준


@dataclass
class FingerKinematicState:
    ee_pos: np.ndarray = None
    ee_vel: np.ndarray = None
    jacobian: np.ndarray = None
    has_desired_pos: bool = False
    has_desired_vel: bool = False

    def __post_init__(self):
        if self.ee_pos is None:
            self.ee_pos = np.zeros(3)
        if self.ee_vel is None:
            self.ee_vel = np.zeros(3)


class TorqueImpedanceControllerNode(Node):
    def __init__(self):
        super().__init__("torque_impedance_controller_node")

        # --- Parameters ---
        self.declare_parameter("rate_hz", 100.0)  # 토크 제어는 높은 주파수 권장
        self.declare_parameter("use_mujoco", True)
        self.declare_parameter("mujoco_model_path", DEFAULT_MUJOCO_MODEL)

        # 임피던스 관련 파라미터
        self.declare_parameter("stiffness_scale", 1.0)
        self.declare_parameter("damping_ratio", 0.7)  # ζ
        self.declare_parameter("virtual_mass", 0.1)    # M(kg)

        # 안전/필터링
        self.declare_parameter("max_torque", 2.0)            # Nm clamp
        self.declare_parameter("max_current_units", 500)     # ±500 (HW < 1193)
        self.declare_parameter("torque_filter_alpha", 0.3)   # 0~1 LPF 계수
        self.declare_parameter("velocity_window", 5)         # EE 속도 추정용 샘플수

        # Force feedback (optional, closed-loop force control)
        self.declare_parameter("enable_force_feedback", False)
        self.declare_parameter("force_axis", "z")            # 사용할 힘 성분 (x|y|z)
        self.declare_parameter("force_sensor_map", ["if", "mf", "th"])  # s1,s2,s3 -> finger 매핑
        self.declare_parameter("kp_force", 0.3)               # 힘 오차 P 게인 (토크 단위)
        self.declare_parameter("ki_force", 0.0)               # 힘 오차 I 게인 (토크 단위)

        # 파라미터 조회
        self.rate_hz = float(self.get_parameter("rate_hz").value)
        self.use_mujoco = bool(self.get_parameter("use_mujoco").value)
        self.model_path = str(self.get_parameter("mujoco_model_path").value)

        self.stiffness_scale = float(self.get_parameter("stiffness_scale").value)
        self.damping_ratio = float(self.get_parameter("damping_ratio").value)
        self.virtual_mass = float(self.get_parameter("virtual_mass").value)

        self.max_torque = float(self.get_parameter("max_torque").value)
        self.max_current_units = int(self.get_parameter("max_current_units").value)
        self.torque_filter_alpha = float(self.get_parameter("torque_filter_alpha").value)
        self.velocity_window = int(self.get_parameter("velocity_window").value)

        self.enable_force_fb = bool(self.get_parameter("enable_force_feedback").value)
        self.force_axis = str(self.get_parameter("force_axis").value)
        self.force_sensor_map = list(self.get_parameter("force_sensor_map").value)  # len=3 예상
        self.kp_force = float(self.get_parameter("kp_force").value)
        self.ki_force = float(self.get_parameter("ki_force").value)

        # --- State ---
        self.target_stiffness = np.zeros(9)  # 정책 출력 (finger별 3)
        self.current_qpos = np.zeros(9)
        self.current_qvel = np.zeros(9)

        self.fingers: Dict[str, FingerKinematicState] = {
            "if": FingerKinematicState(),
            "mf": FingerKinematicState(),
            "th": FingerKinematicState(),
        }
        self.desired_pos = {"if": np.zeros(3), "mf": np.zeros(3), "th": np.zeros(3)}
        self.desired_vel = {"if": np.zeros(3), "mf": np.zeros(3), "th": np.zeros(3)}

        self.ee_pos_history = {"if": [], "mf": [], "th": []}
        self.last_torques = np.zeros(9)

        self.has_stiffness = False
        self.has_qpos = False

        # Force feedback state (per finger -> scalar force along axis)
        self.measured_force: Dict[str, float] = {"if": 0.0, "mf": 0.0, "th": 0.0}
        self.has_force: Dict[str, bool] = {"if": False, "mf": False, "th": False}
        self.force_int_err = np.zeros(9)  # joint torque integral (mapped)

        # MuJoCo 모델/데이터
        self.mj_model = None
        self.mj_data = None
        self.mj_qpos_adr: Dict[str, int] = {}
        if self.use_mujoco:
            self._init_mujoco()

        # --- Subscribers ---
        self.create_subscription(
            Float32MultiArray,
            "/impedance_control/target_stiffness",
            self._on_stiffness,
            10,
        )
        self.create_subscription(JointState, "/hand_tracker/qpos", self._on_qpos, 10)
        # Desired pose
        self.create_subscription(PoseStamped, "/ee_pose_desired_if", lambda m: self._on_desired_pose("if", m), 10)
        self.create_subscription(PoseStamped, "/ee_pose_desired_mf", lambda m: self._on_desired_pose("mf", m), 10)
        self.create_subscription(PoseStamped, "/ee_pose_desired_th", lambda m: self._on_desired_pose("th", m), 10)
        # Desired velocity (optional)
        self.create_subscription(TwistStamped, "/ee_velocity_desired_if", lambda m: self._on_desired_velocity("if", m), 10)
        self.create_subscription(TwistStamped, "/ee_velocity_desired_mf", lambda m: self._on_desired_velocity("mf", m), 10)
        self.create_subscription(TwistStamped, "/ee_velocity_desired_th", lambda m: self._on_desired_velocity("th", m), 10)

        # Optional force sensor wrenches (s1,s2,s3) -> map to fingers (only force part used)
        if self.enable_force_fb:
            self.create_subscription(Float32MultiArray, "/force_sensor/flattened_forces", self._on_force_flattened, 10)
            # 또는 개별 /force_sensor/s{i}/wrench 사용 가능. 여기서는 단일 Flattened 입력 가정.
            # 개별 토픽 구독 예시 (주석):
            # self.create_subscription(WrenchStamped, "/force_sensor/s1/wrench", lambda m: self._on_wrench_single(0, m), 10)
            # self.create_subscription(WrenchStamped, "/force_sensor/s2/wrench", lambda m: self._on_wrench_single(1, m), 10)
            # self.create_subscription(WrenchStamped, "/force_sensor/s3/wrench", lambda m: self._on_wrench_single(2, m), 10)

        # --- Publishers ---
        self.current_pub = self.create_publisher(Int16MultiArray, "/dynamixel/goal_current", 10)
        self.torque_pub = self.create_publisher(Float32MultiArray, "/impedance_control/computed_torques", 10)

        # --- Timer ---
        self._log_counter = 0
        self.control_timer = self.create_timer(1.0 / self.rate_hz, self._control_loop)

        self.get_logger().info(
            f"TorqueImpedanceController started (rate={self.rate_hz}Hz, mujoco={'on' if self.use_mujoco else 'off'})"
        )

    # ---------------- MuJoCo 초기화 ----------------
    def _init_mujoco(self):
        if not MUJOCO_AVAILABLE:
            self.get_logger().warn("MuJoCo import 실패 -> kinematics 비활성화")
            self.use_mujoco = False
            return
        try:
            self.mj_model = mj.MjModel.from_xml_path(self.model_path)  # type: ignore
            self.mj_data = mj.MjData(self.mj_model)  # type: ignore
            # joint 이름 -> qpos address 매핑
            for finger, joint_names in DCLAW_JOINTS.items():
                for i, joint_name in enumerate(joint_names):
                    try:
                        jid = mj.mj_name2id(self.mj_model, mj.mjtObj.mjOBJ_JOINT, joint_name)  # type: ignore
                        if jid >= 0:
                            adr = int(self.mj_model.jnt_qposadr[jid])  # type: ignore
                            self.mj_qpos_adr[f"{finger}_{i}"] = adr
                    except Exception as e:  # noqa: BLE001
                        self.get_logger().warn(f"Joint map 실패 {joint_name}: {e}")
            self.get_logger().info(f"MuJoCo 모델 로드 성공: {self.model_path}")
        except Exception as e:  # noqa: BLE001
            self.get_logger().error(f"MuJoCo 초기화 실패: {e}")
            self.use_mujoco = False

    # ---------------- 콜백 ----------------
    def _on_stiffness(self, msg: Float32MultiArray):
        try:
            if len(msg.data) >= 9:
                self.target_stiffness = np.array(msg.data[:9], dtype=float) * self.stiffness_scale
                self.has_stiffness = True
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f"stiffness 콜백 오류: {e}", throttle_duration_sec=1.0)

    def _on_qpos(self, msg: JointState):
        try:
            if len(msg.position) >= 9:
                self.current_qpos = np.array(msg.position[:9], dtype=float)
                if len(msg.velocity) >= 9:
                    self.current_qvel = np.array(msg.velocity[:9], dtype=float)
                self.has_qpos = True
                if self.use_mujoco:
                    self._update_kinematics()
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f"qpos 콜백 오류: {e}", throttle_duration_sec=1.0)

    def _on_desired_pose(self, finger: str, msg: PoseStamped):
        try:
            p = msg.pose.position
            self.desired_pos[finger] = np.array([p.x, p.y, p.z], dtype=float)
            self.fingers[finger].has_desired_pos = True
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f"desired pose 오류({finger}): {e}", throttle_duration_sec=1.0)

    def _on_desired_velocity(self, finger: str, msg: TwistStamped):
        try:
            v = msg.twist.linear
            self.desired_vel[finger] = np.array([v.x, v.y, v.z], dtype=float)
            self.fingers[finger].has_desired_vel = True
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f"desired velocity 오류({finger}): {e}", throttle_duration_sec=1.0)

    # ---------------- Force Feedback 콜백 ----------------
    def _on_force_flattened(self, msg: Float32MultiArray):
        """예시: 9D = [s1_fx,s1_fy,s1_fz, s2_fx,..., s3_fz] 형태라 가정.
        force_sensor_map = ["if","mf","th"] 로 finger 매핑.
        선택된 force_axis 성분만 사용 (x|y|z)."""
        try:
            data = list(msg.data)
            if len(data) < 9:
                return
            axis_idx = {"x": 0, "y": 1, "z": 2}.get(self.force_axis, 2)
            for sensor_idx, finger in enumerate(self.force_sensor_map):
                base = sensor_idx * 3
                f_val = float(data[base + axis_idx])
                if finger in self.measured_force:
                    self.measured_force[finger] = f_val
                    self.has_force[finger] = True
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f"force 데이터 파싱 오류: {e}", throttle_duration_sec=1.0)

    # ---------------- Kinematics 업데이트 ----------------
    def _update_kinematics(self):
        if not self.use_mujoco or self.mj_data is None:
            return
        try:
            # qpos 설정 (모델 offset 보정)
            for f_idx, finger in enumerate(["thumb", "index", "middle"]):
                for j in range(3):
                    key = f"{finger}_{j}"
                    if key in self.mj_qpos_adr:
                        q_idx = f_idx * 3 + j
                        offset = 1.57 if q_idx in [0, 3, 6] else 3.14
                        self.mj_data.qpos[self.mj_qpos_adr[key]] = self.current_qpos[q_idx] - offset  # type: ignore
                        self.mj_data.qvel[self.mj_qpos_adr[key]] = self.current_qvel[q_idx]  # type: ignore
            mj.mj_forward(self.mj_model, self.mj_data)  # type: ignore

            # EE 위치 및 속도 추정
            for finger, site_name in SITE_NAMES.items():
                try:
                    sid = mj.mj_name2id(self.mj_model, mj.mjtObj.mjOBJ_SITE, site_name)  # type: ignore
                    if sid >= 0:
                        pos = self.mj_data.site_xpos[sid].copy()  # type: ignore
                        self.fingers[finger].ee_pos = pos
                        # 속도 추정 (history + 미분 + LPF)
                        hist = self.ee_pos_history[finger]
                        hist.append(pos)
                        if len(hist) > self.velocity_window:
                            hist.pop(0)
                        if len(hist) >= 2:
                            dt = 1.0 / self.rate_hz
                            vel_raw = (hist[-1] - hist[-2]) / dt
                            alpha = 0.3
                            self.fingers[finger].ee_vel = alpha * vel_raw + (1 - alpha) * self.fingers[finger].ee_vel
                        # Jacobian 계산
                        self.fingers[finger].jacobian = self._compute_jacobian(finger, sid)
                except Exception:
                    pass
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f"kinematics 업데이트 오류: {e}", throttle_duration_sec=5.0)

    def _compute_jacobian(self, finger: str, site_id: int) -> Optional[np.ndarray]:
        if not self.use_mujoco or self.mj_data is None:
            return None
        try:
            jacp = np.zeros((3, self.mj_model.nv))  # type: ignore
            jacr = np.zeros((3, self.mj_model.nv))  # type: ignore
            mj.mj_jac(self.mj_model, self.mj_data, jacp, jacr, self.mj_data.site_xpos[site_id], site_id)  # type: ignore
            fmap = {"th": 0, "if": 1, "mf": 2}
            start = fmap[finger] * 3
            end = start + 3
            return jacp[:, start:end]
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f"Jacobian 계산 오류({finger}): {e}", throttle_duration_sec=10.0)
            return None

    # ---------------- Impedance 계산 ----------------
    def _compute_damping(self, K_diag: np.ndarray) -> np.ndarray:
        # D_i = 2 * ζ * sqrt(M * K_i)
        D = np.zeros_like(K_diag)
        for i, k in enumerate(K_diag):
            if k > 0:
                D[i] = 2.0 * self.damping_ratio * math.sqrt(self.virtual_mass * k)
        return D

    def _compute_torques(self) -> np.ndarray:
        torques = np.zeros(9)
        if not (self.has_stiffness and self.has_qpos):
            return torques
        if not any(f.has_desired_pos for f in self.fingers.values()):
            return torques
        fmap = {"th": 0, "if": 1, "mf": 2}
        for finger, f_idx in fmap.items():
            if not self.fingers[finger].has_desired_pos:
                continue
            k_start = f_idx * 3
            k_end = k_start + 3
            K_vec = self.target_stiffness[k_start:k_end]
            D_vec = self._compute_damping(K_vec)
            pos_err = self.desired_pos[finger] - self.fingers[finger].ee_pos
            vel_err = self.desired_vel[finger] - self.fingers[finger].ee_vel
            F = K_vec * pos_err + D_vec * vel_err  # element-wise (diag 적용)
            J = self.fingers[finger].jacobian
            if J is not None and self.use_mujoco:
                tau = J.T @ F  # 3x3 * 3 -> 3  (desired joint torques)
                # Force feedback 적용: fingertip 측정 힘이 있을 경우 보정
                if self.enable_force_fb and self.has_force.get(finger, False):
                    # 측정 힘을 Cartesian scalar -> vector로 확장 (단일 축 가정)
                    axis_idx = {"x": 0, "y": 1, "z": 2}.get(self.force_axis, 2)
                    F_meas = np.zeros(3)
                    F_meas[axis_idx] = self.measured_force[finger]
                    tau_meas = J.T @ F_meas  # 측정 토크 근사
                    tau_err = tau - tau_meas
                    # P + I 보정 (I는 소량 권장)
                    for i in range(3):
                        joint_idx = k_start + i
                        self.force_int_err[joint_idx] += tau_err[i] * (1.0 / self.rate_hz)
                        tau[i] += self.kp_force * tau_err[i] + self.ki_force * self.force_int_err[joint_idx]
                for i in range(3):
                    torques[k_start + i] = tau[i]
            else:
                # Jacobian 없을 때 간단 근사 (비권장, fallback)
                for i in range(3):
                    torques[k_start + i] = K_vec[i] * pos_err[i] * 0.01
        return torques

    # ---------------- 토크 → 전류 변환 및 필터 ----------------
    def _torques_to_current_units(self, torques: np.ndarray) -> List[int]:
        # τ = Kt * I  =>  I(mA) = τ / Kt,  units = I(mA) / CURRENT_UNIT
        currents_mA = np.where(CURRENT_TO_TORQUE > 0.0, torques / CURRENT_TO_TORQUE, 0.0)
        units = currents_mA / CURRENT_UNIT
        units = np.clip(units, -self.max_current_units, self.max_current_units)
        return [int(round(u)) for u in units]

    def _filter_torques(self, torques: np.ndarray) -> np.ndarray:
        filtered = self.torque_filter_alpha * torques + (1.0 - self.torque_filter_alpha) * self.last_torques
        self.last_torques = filtered.copy()
        return filtered

    # ---------------- 메인 제어 루프 ----------------
    def _control_loop(self):
        try:
            raw_tau = self._compute_torques()
            raw_tau = np.clip(raw_tau, -self.max_torque, self.max_torque)
            filt_tau = self._filter_torques(raw_tau)
            current_units = self._torques_to_current_units(filt_tau)

            # Publish current commands
            cur_msg = Int16MultiArray()
            cur_msg.data = current_units
            self.current_pub.publish(cur_msg)

            # Publish torques for monitoring
            tau_msg = Float32MultiArray()
            tau_msg.data = filt_tau.tolist()
            self.torque_pub.publish(tau_msg)

            self._log_counter += 1
            if self._log_counter % int(self.rate_hz) == 0 and self.has_stiffness:
                self.get_logger().info(
                    "TorqueCtrl: K_avg=%.1f | τ_TH=%.3f τ_IF=%.3f τ_MF=%.3f | Ī=%.0f units" % (
                        float(np.mean(self.target_stiffness)),
                        float(np.mean(filt_tau[0:3])),
                        float(np.mean(filt_tau[3:6])),
                        float(np.mean(filt_tau[6:9])),
                        float(np.mean(np.abs(current_units))),
                    )
                )
        except Exception as e:  # noqa: BLE001
            self.get_logger().error(f"제어 루프 오류: {e}", throttle_duration_sec=1.0)


def main(args=None):
    rclpy.init(args=args)
    try:
        node = TorqueImpedanceControllerNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:  # noqa: BLE001
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
