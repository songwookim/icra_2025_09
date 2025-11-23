#!/usr/bin/env python3
from __future__ import annotations

"""
Robot Controller Node (unit passthrough).

- Subscribes: Int32MultiArray (9) on /hand_tracker/targets_units
- Commands: Dynamixel via DynamixelControl if available and safe_mode=False; otherwise dry-run logs
- Initial posture: fixed initial_val applied once at startup and used as baseline when disabled
"""

DEFAULT_UNITS_TOPIC = '/hand_tracker/targets_units'
DEFAULT_JOINT_ORDER = [
    'thumb_cmc', 'thumb_mcp', 'thumb_ip',
    'index_mcp', 'index_pip', 'index_dip',
    'middle_mcp', 'middle_pip', 'middle_dip',
]
DEFAULT_UNITS_BASELINE = [
    1000.0, 2000.0, 2000.0,
    1000.0, 2000.0, 2000.0,
    1000.0, 2000.0, 2000.0,
]

FALLBACK_INITIAL_POSITIONS = [1117, 1881, 1789, 1222, 1815, 1790, 1299, 1689, 1745]

import math
import os
import sys
from typing import List, Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32MultiArray, Bool, Float32MultiArray
import numpy as np

try:
    from .dynamixel_control import DynamixelControl  # type: ignore
except Exception:
    try:
        from dynamixel_control import DynamixelControl  # type: ignore
    except Exception:  # pragma: no cover
        DynamixelControl = None  # type: ignore

try:  # optional
    import hydra  # type: ignore
    from omegaconf import OmegaConf, DictConfig  # type: ignore
except Exception:  # pragma: no cover
    hydra = None  # type: ignore
    OmegaConf = None  # type: ignore
    DictConfig = None  # type: ignore


class RobotControllerNode(Node):
    def __init__(self, config: Optional['DictConfig'] = None):  # type: ignore[name-defined]
        super().__init__('robot_controller_node')

        # 설정 로드
        self.config = self._load_config()
        if self.config is None:
            # 기존에는 즉시 종료했지만, 기본값(Dry-run)으로 계속 진행하도록 변경
            self.get_logger().error("설정 로드 실패: config.yaml 없음 또는 파싱 실패 -> 기본값으로 계속 진행 (Dry-run 모드)")
            # cfg_dyn 은 None 으로 남아 Dry-run 경로를 타게 됨
        cfg_dyn = getattr(self.config, 'dynamixel', None) if self.config is not None else None
        default_ids = list(getattr(cfg_dyn, 'ids', [10, 11, 12, 20, 21, 22, 30, 31, 32]))
        default_mode = int(getattr(getattr(cfg_dyn, 'control_modes', {}), 'default_mode', 3))
        initial_positions_cfg = list(getattr(cfg_dyn, 'initial_positions', [])) if cfg_dyn is not None else []
        if initial_positions_cfg:
            initial_val_default = [int(x) for x in initial_positions_cfg]
        else:
            initial_val_default = list(FALLBACK_INITIAL_POSITIONS)

        # 파라미터 선언
        self.declare_parameter('ids', default_ids)
        self.declare_parameter('mode', default_mode)
        self.declare_parameter('units_baseline', list(DEFAULT_UNITS_BASELINE))
        self.declare_parameter('clip_min', [])
        self.declare_parameter('clip_max', [])
        self.declare_parameter('hand_joint_order', [
            'thumb_cmc', 'thumb_mcp', 'thumb_ip',
            'index_mcp', 'index_pip', 'index_dip',
            'middle_mcp', 'middle_pip', 'middle_dip'
        ])
        self.declare_parameter('hand_units_topic', DEFAULT_UNITS_TOPIC)
        self.declare_parameter('units_enabled_topic', '/hand_tracker/units_enabled')
        self.declare_parameter('max_step_units', 20.0)
        self.declare_parameter('safe_mode', True)
        # Force / current control extension params
        self.declare_parameter('use_force_control', False)  # true -> torque/current path 활성화
        self.declare_parameter('torque_topic', '/impedance_control/computed_torques')
        self.declare_parameter('current_topic', '/dynamixel/goal_current')  # 직접 current 명령 받아 Passthrough 가능
        self.declare_parameter('force_control_mode', 0)  # Dynamixel current control mode (XM430: 0)
        self.declare_parameter('current_to_torque', 1.78e-3)  # Nm per mA (Kt)
        self.declare_parameter('current_unit_mA', 2.69)  # mA per raw current unit
        self.declare_parameter('max_current_units_fc', 500)  # force control 시 클램프
        # hand enable/disable parameters removed (always enabled)
        self.declare_parameter('initial_val', initial_val_default)
        

        # 유틸
        def _ensure_list(val):
            if isinstance(val, (list, tuple)):
                return list(val)
            if val is None:
                return []
            return [val]

        def _fit_len(arr, n: int, fill):
            arr = list(arr)
            if len(arr) < n:
                arr.extend([fill] * (n - len(arr)))
            return arr[:n]

        # 파라미터 값 읽기
        ids_param = _ensure_list(self.get_parameter('ids').value) or default_ids
        self.ids = [int(x) for x in ids_param]
        self.mode = int(self.get_parameter('mode').value or default_mode)

        joint_param = _ensure_list(self.get_parameter('hand_joint_order').value)
        self.hand_joint_order = [str(x).lower() for x in (joint_param or DEFAULT_JOINT_ORDER)]
        if len(self.hand_joint_order) != len(self.ids):
            self.get_logger().warn(
                f"joint/order mismatch ids={len(self.ids)} joints={len(self.hand_joint_order)} -> min len 사용"
            )

        init_param = _ensure_list(self.get_parameter('initial_val').value) or initial_val_default
        init_fill = init_param[-1] if init_param else (initial_val_default[-1] if initial_val_default else 2000)
        self.initial_val = [int(x) for x in _fit_len(init_param, len(self.ids), init_fill)]
        self.base_positions = [float(v) for v in self.initial_val]

        baseline_param = _ensure_list(self.get_parameter('units_baseline').value)
        if not baseline_param:
            baseline_param = list(DEFAULT_UNITS_BASELINE)
        baseline_fill = baseline_param[-1]
        self.units_baseline = [float(x) for x in _fit_len(baseline_param, len(self.hand_joint_order), baseline_fill)]

        clip_min_param = _ensure_list(self.get_parameter('clip_min').value)
        clip_max_param = _ensure_list(self.get_parameter('clip_max').value)
        clip_min_source = clip_min_param if clip_min_param else [0]
        clip_max_source = clip_max_param if clip_max_param else [4095]
        clip_min_fill = clip_min_source[-1] if clip_min_source else 0
        clip_max_fill = clip_max_source[-1] if clip_max_source else 4095
        self.clip_min = [int(x) for x in _fit_len(clip_min_source, len(self.ids), clip_min_fill)]
        self.clip_max = [int(x) for x in _fit_len(clip_max_source, len(self.ids), clip_max_fill)]

        self.hand_units_topic = str(self.get_parameter('hand_units_topic').value or DEFAULT_UNITS_TOPIC)
        self.units_enabled_topic = str(self.get_parameter('units_enabled_topic').value or '/hand_tracker/units_enabled')
        self.max_step_units = float(self.get_parameter('max_step_units').value or 20.0)
        self.safe_mode = bool(self.get_parameter('safe_mode').value)
        self.use_force_control = bool(self.get_parameter('use_force_control').value)
        self.torque_topic = str(self.get_parameter('torque_topic').value)
        self.current_topic = str(self.get_parameter('current_topic').value)
        # None 방지 기본값 할당 (ROS param 획득 실패 대비)
        self.force_control_mode = int(self.get_parameter('force_control_mode').value or 0)
        self.current_to_torque = float(self.get_parameter('current_to_torque').value or 1.78e-3)
        self.current_unit_mA = float(self.get_parameter('current_unit_mA').value or 2.69)
        self.max_current_units_fc = int(self.get_parameter('max_current_units_fc').value or 500)
        self.hand_enabled = True

        # Backend 연결 (Dynamixel or Dry-run)
        self.controller = None
        if DynamixelControl is not None and cfg_dyn is not None:
            try:  # pragma: no cover
                self.controller = DynamixelControl(cfg_dyn)
                self.controller.connect()
                self.get_logger().info(f"Dynamixel connected (ids={self.ids}, mode={self.mode})")
                if self.use_force_control and self.mode != self.force_control_mode:
                    try:
                        self.get_logger().info(f"Switch operating mode -> current (mode={self.force_control_mode}) for force control")
                        self.controller.set_operating_mode_all(self.force_control_mode)
                        self.mode = self.force_control_mode
                    except Exception as e:
                        self.get_logger().warn(f"Operating mode switch 실패: {e}")
            except Exception as e:
                self.get_logger().error(f"Dynamixel init/connect 실패: {e} -> Dry-run")
        elif DynamixelControl is None:
            self.get_logger().warn('dynamixel_control 모듈 없음 -> Dry-run')
        else:
            self.get_logger().warn('dynamixel 설정 없음 -> Dry-run')

        self.get_logger().info(f"[SETUP] src=hand(units) test={self.safe_mode} ids={self.ids}")

        # 내부 상태 & CSV
        # removed _filt_deg (was only used for disable behavior)
        self._last_targets: Optional[List[int]] = None
        # keyboard toggle removed
        self._units_enabled_log_state: Optional[bool] = None

        # 구독 설정
        self.sub_units = self.create_subscription(Int32MultiArray, self.hand_units_topic, self.on_unit_targets, 10)
        self.units_enabled_sub = self.create_subscription(Bool, self.units_enabled_topic, self.on_units_enabled, 10)
        # Force control: subscribe to torque topic -> convert to current
        if self.use_force_control:
            self.sub_torque = self.create_subscription(Float32MultiArray, self.torque_topic, self.on_target_torques, 10)
            # Optional direct current passthrough (Int16/Int32 both 가능) if external node already converts
            self.sub_current_passthrough = self.create_subscription(Int32MultiArray, self.current_topic, self.on_direct_currents, 10)
        self.get_logger().info(f"Input source: {self.hand_units_topic} (units)")
        self.get_logger().info(f"Enable source: {self.units_enabled_topic}")
        if self.use_force_control:
            self.get_logger().info(f"Force control 활성화 torque_topic={self.torque_topic} current_topic={self.current_topic}")

        # 시작 시 초기 포즈로 세팅(한 번)
        try:
            self.get_logger().info(f"Apply initial posture: {self.initial_val}")
            self._send_targets(self.initial_val)
        except Exception as e:
            self.get_logger().warn(f"초기 포즈 적용 실패: {e}")
        # Torque log throttling counter
        self._torque_log_counter = 0

    # ============================== Utils
    def _load_config(self) -> Optional['DictConfig']:  # type: ignore[name-defined]
        try:
            pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            cfg_path = os.path.join(pkg_dir, 'resource', 'robot_parameter', 'config.yaml')
            if os.path.exists(cfg_path) and OmegaConf is not None:
                cfg = OmegaConf.load(cfg_path)
                self.get_logger().debug(f"Loaded config.yaml ({cfg_path})")
                return cfg
            if not os.path.exists(cfg_path):
                self.get_logger().error(f"Config 파일 없음: {cfg_path}")
            elif OmegaConf is None:
                self.get_logger().error("OmegaConf 미설치 -> config 사용 불가")
        except Exception as e:
            self.get_logger().warn(f"Config load 실패: {e}")

    # ============================== Hand Units (Int32MultiArray, 9 elems)
    def on_unit_targets(self, msg: Int32MultiArray):
        units_in = [int(x) for x in (msg.data or [])]
        if not units_in and self._last_targets is None:
            self.get_logger().info("No units yet -> apply initial posture once")
            self._send_targets(self.initial_val)
            return

        base = [float(v) for v in (self.base_positions or self.initial_val)]
        n = len(self.ids)

        if not self.hand_enabled:
            if self._last_targets is None:
                idle_targets = [self._clip_target(i, base[i]) for i in range(n)]
                self._send_targets(idle_targets)
            return

        final_targets: List[int] = []
        for i in range(n):
            if i < len(units_in):
                candidate = float(units_in[i])
            else:
                candidate = base[i]
            final_targets.append(self._clip_target(i, candidate))

        self._send_targets(final_targets)

    def on_units_enabled(self, msg: Bool):
        new_state = bool(msg.data)
        if self._units_enabled_log_state is None or new_state != self._units_enabled_log_state:
            status = 'ENABLED' if new_state else 'DISABLED'
            self.get_logger().info(f"Qpos stream -> {status}")
            self._units_enabled_log_state = new_state
        if self.hand_enabled != new_state:
            self.hand_enabled = new_state

    # ============================== Utilities (send)
    def _clip_target(self, i: int, val: float) -> int:
        return int(round(max(float(self.clip_min[i]), min(float(self.clip_max[i]), val))))

    def _safe_step_limit(self, targets: List[int]) -> List[int]:
        if self._last_targets is None:
            return targets
        out: List[int] = []
        limited = []
        for i, t in enumerate(targets):
            prev = self._last_targets[i] if i < len(self._last_targets) else t
            if abs(t - prev) > self.max_step_units:
                nt = int(round(prev + self.max_step_units * (1 if t > prev else -1)))
                out.append(nt)
                limited.append(f"ID{self.ids[i]} {prev}->{t} limited->{nt}")
            else:
                out.append(t)
        if limited:
            self.get_logger().debug('[step_limit] ' + '; '.join(limited))
        return out

    def _send_targets(self, targets: List[int]):
        targets = self._safe_step_limit(targets)
        self._last_targets = list(targets)
        self.get_logger().debug('final ' + ', '.join(f"ID{self.ids[i]}={targets[i]}" for i in range(len(targets))))
        if self.controller is None:
            self.get_logger().info(f"robot is not connected -> dry-run {targets}")
            return
        try:  # pragma: no cover
            if self.safe_mode:
                self.get_logger().info(f"[Safe Mode] {targets} ")
                return
            else :
                self.get_logger().info(f"[Safe Mode] {targets} ")
                self.controller.set_joint_positions(targets)
                
        except Exception as e:
            self.get_logger().error(f"set_joint_positions 실패: {e}")

    # ============================== Force Control (Torque -> Current) ==============================
    def on_target_torques(self, msg: Float32MultiArray):
        if not self.use_force_control:
            return
        torques = np.array(list(msg.data), dtype=float)
        if len(torques) < len(self.ids):
            self.get_logger().warn(f"torque length {len(torques)} < ids {len(self.ids)}")
            return
        # τ(Nm) -> current units: units = τ / (Kt * mA_per_unit)
        denom = self.current_to_torque * self.current_unit_mA if self.current_to_torque > 0 else 1e-6
        current_units = torques / denom
        current_units = np.clip(current_units, -self.max_current_units_fc, self.max_current_units_fc)
        int_currents = [int(round(c)) for c in current_units[:len(self.ids)]]
        log_needed = any(c != 0 for c in int_currents) or (self._torque_log_counter % 50 == 0)
        if self.controller is None:
            if log_needed:
                self.get_logger().info(f"[Dry-run torque->current] {int_currents}")
            self._torque_log_counter += 1
            return
        if self.safe_mode:
            if log_needed:
                self.get_logger().info(f"[Safe Mode torque] {int_currents}")
            self._torque_log_counter += 1
            return
        try:
            # test_torqueinputs(ids, input_torque)
            self.controller.test_torqueinputs(self.ids, int_currents, log=False)
        except Exception as e:
            self.get_logger().error(f"torque inputs 실패: {e}")
        self._torque_log_counter += 1

    def on_direct_currents(self, msg: Int32MultiArray):
        if not self.use_force_control:
            return
        currents = [int(x) for x in msg.data]
        if len(currents) < len(self.ids):
            return
        currents = currents[:len(self.ids)]
        currents = [int(max(-self.max_current_units_fc, min(self.max_current_units_fc, c))) for c in currents]
        if self.controller is None:
            self.get_logger().info(f"[Dry-run passthrough currents] {currents}")
            return
        if self.safe_mode:
            self.get_logger().info(f"[Safe Mode currents] {currents}")
            return
        try:
            self.controller.test_torqueinputs(self.ids, currents, log=False)
        except Exception as e:
            self.get_logger().error(f"current passthrough 실패: {e}")

    # enable/disable control removed (always enabled)

    # _keyboard_loop removed


# ============================== Entrypoints
def _run(node: RobotControllerNode):
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            # Ignore double-shutdown race (RCLError: rcl_shutdown already called)
            pass

def main():  # ROS only / hydra 우회
    rclpy.init()
    node = RobotControllerNode()
    _run(node)


if __name__ == '__main__':
    main()