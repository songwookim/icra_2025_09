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
from std_msgs.msg import Int32MultiArray, Bool, Float32MultiArray, UInt8
from sensor_msgs.msg import JointState
import numpy as np

try:
    from .dynamixel_control import DynamixelControl  # type: ignore
except ImportError:
    # Relative import fails when run as script (expected), try absolute
    try:
        from dynamixel_control import DynamixelControl  # type: ignore
    except ImportError as e:
        print(f"[ERROR] DynamixelControl module not found: {e}")
        print("[ERROR] Robot control will not be available - running in dry-run mode")
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
        self.declare_parameter('demo_playback_stage_topic', '/demo_playback_stage')

        # Force / current control extension params
        self.declare_parameter('use_force_control', False)  # true -> torque/current path 활성화
        self.declare_parameter('safe_mode', True)
        self.declare_parameter('torque_topic', '/impedance_control/computed_torques')
        self.declare_parameter('current_topic', '/dynamixel/goal_current')  # 직접 current 명령 받아 Passthrough 가능
        self.declare_parameter('force_control_mode', 0)  # Dynamixel current control mode (XM430: 0)
        self.declare_parameter('current_to_torque', 1.78e-3)  # Nm per mA (Kt)
        self.declare_parameter('current_unit_mA', 2.69)  # mA per raw current unit
        self.declare_parameter('max_current_units_pos', 500)  # test_torqueinputs 양수 리미트
        self.declare_parameter('max_current_units_neg', 500)  # test_torqueinputs 음수 리미트
        self.declare_parameter('max_pwm_limit', 500)  # [SAFETY] PWM limit - stop if exceeded (XM430: 885=100%)
        # 정책 준비 전까지 force control 입력을 지연할지 여부
        self.declare_parameter('defer_force_control_until_policy', True)
        # hand enable/disable parameters removed (always enabled)
        self.declare_parameter('initial_val', initial_val_default)
        # Joint state publish params
        self.declare_parameter('publish_joint_state', True)
        self.declare_parameter('joint_state_topic', '/robot_controller/joint_state')
        self.declare_parameter('units_per_rad', 4096.0 / (2.0 * math.pi))  # conversion scale
        self.declare_parameter('joint_state_rate_hz', 0.0)  # if >0 use timer, else publish on updates
        

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
        self.position_mode = self.mode

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
        self.max_current_units_pos = int(self.get_parameter('max_current_units_pos').value or 500)
        self.max_current_units_neg = int(self.get_parameter('max_current_units_neg').value or 500)
        self.max_pwm_limit = int(self.get_parameter('max_pwm_limit').value or 500)  # [SAFETY] PWM limit
        self._pwm_safety_triggered = False  # Flag to track if PWM limit was hit
        self.hand_enabled = True
        self.defer_force_control_until_policy = bool(self.get_parameter('defer_force_control_until_policy').value)
        self.publish_joint_state = bool(self.get_parameter('publish_joint_state').value)
        self.joint_state_topic = str(self.get_parameter('joint_state_topic').value or '/robot_controller/joint_state')
        self.units_per_rad = float(self.get_parameter('units_per_rad').value or (4096.0 / (2.0 * math.pi)))
        self.joint_state_rate_hz = float(self.get_parameter('joint_state_rate_hz').value or 0.0)
        self.demo_stage_topic = str(self.get_parameter('demo_playback_stage_topic').value or '/demo_playback_stage')
        self._joint_state_pub_counter = 0
        # 정책 예측값 수신 여부
        self._policy_ready: bool = False
        self._policy_wait_log_counter: int = 0
        self._demo_stage = 0
        self._force_mode_active = False
        self._torque_playback_enabled = not self.use_force_control

        # Backend 연결 (Dynamixel or Dry-run)
        self.controller = None
        if DynamixelControl is not None and cfg_dyn is not None:
            try:  # pragma: no cover
                self.controller = DynamixelControl(cfg_dyn, max_current_pos=self.max_current_units_pos, max_current_neg=self.max_current_units_neg)
                self.controller.connect()
                self.get_logger().info(f"Dynamixel connected (ids={self.ids}, initial_mode={self.mode})")
                
                # 초기화: Position control mode로 명시적 설정 및 torque enable
                self.get_logger().info("[INIT] Setting up position control mode for initialization...")
                position_mode = 3  # Extended Position Control Mode
                try:
                    self.controller.disable_torque()
                    self.controller.set_operating_mode_all(position_mode)
                    self.mode = position_mode
                    self.position_mode = position_mode
                    self.controller.enable_torque()
                    self.get_logger().info(f"[INIT] Position control mode set (mode={position_mode}), torque enabled")
                except Exception as e:
                    self.get_logger().error(f"Failed to set position mode: {e}")
                    
            except Exception as e:
                self.get_logger().error(f"Dynamixel init/connect 실패: {e} -> Dry-run")
                self.controller = None
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

        # 구독 설정: use_force_control에 따라 다른 입력 구독
        if self.use_force_control:
            # Force control mode: torque/current 입력 구독
            self.sub_torque = self.create_subscription(Float32MultiArray, self.torque_topic, self.subscribe_target_torques, 10)
            self.sub_current_passthrough = self.create_subscription(Int32MultiArray, self.current_topic, self.subscribe_direct_currents, 10)
            # 정책 출력 구독하여 첫 값 수신 후 force control 활성화
            if self.defer_force_control_until_policy:
                self.sub_policy_pred = self.create_subscription(Float32MultiArray, '/stiffness_policy/predicted', self.subscribe_policy_predicted, 10)
            self.get_logger().info(f"[MODE] Force control: torque_topic={self.torque_topic}, current_topic={self.current_topic}")
        else:
            # Position control mode: units 입력 구독
            self.sub_units = self.create_subscription(Int32MultiArray, self.hand_units_topic, self.subscribe_unit_targets, 10)
            self.units_enabled_sub = self.create_subscription(Bool, self.units_enabled_topic, self.subscribe_units_enabled, 10)
            self.get_logger().info(f"[MODE] Position control: units_topic={self.hand_units_topic}, enable_topic={self.units_enabled_topic}")

        self.demo_stage_sub = None
        if self.use_force_control and self.demo_stage_topic:
            self.demo_stage_sub = self.create_subscription(UInt8, self.demo_stage_topic, self.subscribe_demo_stage, 10)
            self.get_logger().info(f"[MODE] Demo playback stage subscribed: topic={self.demo_stage_topic}")
        elif self.use_force_control:
            self._torque_playback_enabled = True
            self.get_logger().warn("[MODE] demo_playback_stage_topic unset -> force control enabled immediately")

        # Joint state publisher & state buffers (must initialize before _send_targets)
        self.joint_state_pub = None
        self._last_qpos: Optional[List[float]] = None
        self._last_qpos_time: Optional[float] = None
        if self.publish_joint_state:
            self.joint_state_pub = self.create_publisher(JointState, self.joint_state_topic, 10)
            self.get_logger().info(f"[INIT] JointState publisher CREATED: topic={self.joint_state_topic}")
        else:
            self.get_logger().warn("[INIT] JointState publishing is DISABLED")

        # PWM publisher for monitoring and logging
        self.pwm_pub = self.create_publisher(Int32MultiArray, '/dynamixel/present_pwm', 10)
        self.get_logger().info("[INIT] PWM publisher CREATED: topic=/dynamixel/present_pwm")

        # Goal position subscriber for external position commands (e.g., 'i' key for initial pose)
        self.sub_goal_position = self.create_subscription(
            Int32MultiArray, '/dynamixel/goal_position', self.subscribe_goal_position, 10)
        self.get_logger().info("[INIT] Goal position subscriber CREATED: topic=/dynamixel/goal_position")

        # 시작 시 초기 포즈로 세팅 (항상 position mode에서 수행)
        # Controller 없어도 _send_targets 호출 (dry-run 로깅 위해)
        try:
            self.get_logger().info(f"[INIT] Applying initial posture: {self.initial_val[:3]}...")
            self._send_targets(self.initial_val)
            if self.controller is not None:
                self.get_logger().info("[INIT] Initial posture applied successfully")
                # 로봇이 초기 포즈로 이동할 시간을 주기 위해 2초 대기
                import time
                self.get_logger().info("[INIT] Waiting 2 seconds for robot to reach initial pose...")
                time.sleep(2.0)
                self.get_logger().info("[INIT] Initial pose settling time completed")
            else:
                self.get_logger().warn("[INIT] Dry-run mode - initial pose logged but not sent to hardware")
        except Exception as e:
            self.get_logger().error(f"초기 포즈 적용 실패: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())
        
        # Force control mode will be enabled when demo stage>=1 (or immediately if no stage topic)
        if self.use_force_control and not self.demo_stage_topic:
            self._switch_to_force_mode()
        
        # Torque log throttling counter
        self._torque_log_counter = 0
        
        # Timer to republish JointState at regular intervals (for continuous monitoring)
        if self.publish_joint_state:
            self._joint_state_republish_rate = 30.0  # Hz
            self.create_timer(1.0 / self._joint_state_republish_rate, self._republish_joint_state)
            self.get_logger().info(f"[INIT] JointState republish timer created: rate={self._joint_state_republish_rate}Hz")

        # 초기에는 정책이 올 때까지 force control 입력 무시
        if self.use_force_control and self.defer_force_control_until_policy:
            self.get_logger().info("Force control deferred until first policy prediction is received.")

    # ============================== Utils
    def _load_config(self) -> Optional['DictConfig']:  # type: ignore[name-defined]
        if OmegaConf is None:
            self.get_logger().error("OmegaConf 미설치 -> config 사용 불가")
            return None
            
        # Try ament_index first (for installed package)
        try:
            from ament_index_python.packages import get_package_share_directory
            pkg_share = get_package_share_directory('hri_falcon_robot_bridge')
            cfg_path = os.path.join(pkg_share, 'robot_parameter', 'config.yaml')
            self.get_logger().info(f"[Config] Trying ament_index path: {cfg_path}")
            
            if os.path.exists(cfg_path):
                cfg = OmegaConf.load(cfg_path)
                self.get_logger().info(f"[Config] Loaded from ament_index: {cfg_path}")
                return cfg
            else:
                self.get_logger().error(f"[Config] File not found: {cfg_path}")
        except Exception as e:
            self.get_logger().warn(f"[Config] ament_index failed: {e}")
        
        # Fallback to source directory (for development)
        try:
            pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            cfg_path = os.path.join(pkg_dir, 'resource', 'robot_parameter', 'config.yaml')
            self.get_logger().info(f"[Config] Trying source path: {cfg_path}")
            
            if os.path.exists(cfg_path):
                cfg = OmegaConf.load(cfg_path)
                self.get_logger().info(f"[Config] Loaded from source: {cfg_path}")
                return cfg
            else:
                self.get_logger().error(f"[Config] Source file not found: {cfg_path}")
        except Exception as e:
            self.get_logger().error(f"[Config] Source path load failed: {e}")
        
        return None

    # ============================== Hand Units (Int32MultiArray, 9 elems)
    def subscribe_unit_targets(self, msg: Int32MultiArray):
        units_in = [int(x) for x in (msg.data or [])]
        if not units_in and self._last_targets is None:
            self.get_logger().info("No units yet -> apply initial posture once")
            self._send_targets(self.initial_val)
            return

        base = [float(v) for v in (self.base_positions or self.initial_val)]
        n = len(self.ids)

        if not self.hand_enabled:
            if self._last_targets is None:
                self.get_logger().warn(f"[Hand DISABLED] Received units {units_in[:3]}... but hand_enabled=False. Press 'h' in sense_glove_mj_node to enable.")
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

    def subscribe_units_enabled(self, msg: Bool):
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
        """Position control: 목표 위치(units)를 Dynamixel에 전송"""
        self.get_logger().info(f"[_send_targets] Called with targets={targets[:3]}... (total {len(targets)} values)")
        targets = self._safe_step_limit(targets)
        self._last_targets = list(targets)
        self.get_logger().debug('final ' + ', '.join(f"ID{self.ids[i]}={targets[i]}" for i in range(len(targets))))
        
        if self.controller is None:
            self.get_logger().warn(f"[Dry-run Position] No controller - targets={targets[:3]}...")
            if self.publish_joint_state:
                self.publish_joint_state_message(targets)
            return
        
        try:
            self.get_logger().info(f"[Position Control] Calling set_joint_positions with targets: {targets[:3]}...")
            self.controller.set_joint_positions(targets)
            self.get_logger().info(f"[Position Control] set_joint_positions completed successfully")
            if self.publish_joint_state:
                self.publish_joint_state_message(targets)
        except Exception as e:
            self.get_logger().error(f"set_joint_positions 실패: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())
            if self.publish_joint_state:
                self.publish_joint_state_message(targets)

    def subscribe_goal_position(self, msg: Int32MultiArray) -> None:
        """Handle external goal position commands (e.g., from 'i' key initial pose)."""
        if len(msg.data) < len(self.ids):
            self.get_logger().warn(f"[Goal Position] Invalid data length: {len(msg.data)} < {len(self.ids)}")
            return
        
        targets = list(msg.data[:len(self.ids)])
        self.get_logger().info(f"[Goal Position] Received: {targets[:3]}... -> switching to position mode")
        
        # Temporarily switch to position mode to move robot
        if self.controller is not None and self._force_mode_active:
            try:
                import time
                self.controller.disable_torque()
                time.sleep(0.05)
                position_mode = 3  # Extended Position Control Mode
                self.controller.set_operating_mode_all(position_mode)
                time.sleep(0.05)
                self.controller.enable_torque()
                self.mode = position_mode
                self._force_mode_active = False
                self.get_logger().info("[Goal Position] Switched to position mode")
            except Exception as e:
                self.get_logger().error(f"[Goal Position] Mode switch failed: {e}")
                return
        
        # Send position command
        self._send_targets(targets)
        self.get_logger().info("[Goal Position] Initial pose command sent")

    # ============================== Demo Playback Stage (Position ↔ Force) ==============================
    def _switch_to_force_mode(self) -> None:
        if not self.use_force_control:
            return
        if self._force_mode_active and self.mode == self.force_control_mode:
            return
        if self.controller is None:
            self._force_mode_active = True
            self.mode = self.force_control_mode
            self.get_logger().info("[MODE] (Dry-run) Force control mode marked active")
            return
        try:
            import time
            self.get_logger().info(f"[MODE] Switching to force control (mode={self.force_control_mode})")
            self.controller.disable_torque()
            time.sleep(0.05)  # Allow Dynamixel to process disable command
            self.controller.set_operating_mode_all(self.force_control_mode)
            time.sleep(0.05)  # Allow mode change to settle
            # Retry enable_torque up to 3 times
            for attempt in range(3):
                try:
                    self.controller.enable_torque()
                    break
                except Exception as retry_e:
                    if attempt < 2:
                        self.get_logger().warn(f"[MODE] Torque enable retry {attempt+1}/3: {retry_e}")
                        time.sleep(0.1)
                    else:
                        raise
            self.mode = self.force_control_mode
            self._force_mode_active = True
            self.get_logger().info("[MODE] Force control mode active, torque re-enabled")
        except Exception as e:
            self.get_logger().error(f"Force control mode switch 실패: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())

    def _switch_to_position_mode(self) -> None:
        if not self._force_mode_active:
            return
        if self.controller is None:
            self._force_mode_active = False
            self.mode = self.position_mode
            self.get_logger().info("[MODE] (Dry-run) Position control mode restored")
            return
        try:
            import time
            self.get_logger().info(f"[MODE] Restoring position control (mode={self.position_mode})")
            self.controller.disable_torque()
            time.sleep(0.05)  # Allow Dynamixel to process disable command
            self.controller.set_operating_mode_all(self.position_mode)
            time.sleep(0.05)  # Allow mode change to settle
            # Retry enable_torque up to 3 times
            for attempt in range(3):
                try:
                    self.controller.enable_torque()
                    break
                except Exception as retry_e:
                    if attempt < 2:
                        self.get_logger().warn(f"[MODE] Torque enable retry {attempt+1}/3: {retry_e}")
                        time.sleep(0.1)
                    else:
                        raise
            self.mode = self.position_mode
            self._force_mode_active = False
            self.get_logger().info("[MODE] Position control mode active, torque re-enabled")
        except Exception as e:
            self.get_logger().error(f"Position control mode restore 실패: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())

    def subscribe_demo_stage(self, msg: UInt8):
        if not self.use_force_control:
            return
        new_stage = int(msg.data)
        if new_stage == self._demo_stage:
            return
        prev = self._demo_stage
        self._demo_stage = new_stage
        self.get_logger().info(f"[DEMO_STAGE] {prev} -> {new_stage}")

        if new_stage <= 0:
            self._torque_playback_enabled = False
            self._switch_to_position_mode()
        else:
            # Stage 1 (initial pose) and beyond should both run in force control mode so the impedance
            # controller can actively drive to the published pose before playback begins.
            self._switch_to_force_mode()
            self._torque_playback_enabled = True

    def _send_currents(self, currents: List[int]):
        """Force control: 목표 current(units)를 Dynamixel에 전송"""
        # Current는 step limit 없이 바로 적용 (임피던스 제어 특성상 급격한 변화 필요)
        currents = [int(max(-self.max_current_units_neg, min(self.max_current_units_pos, c))) for c in currents]
        
        if self.controller is None:
            # self.get_logger().info(f"[Dry-run Current] currents={currents[:3]}...")
            return
        
        # [PWM SAFETY CHECK] Read PWM every call and check limit
        try:
            present_pwm = self.controller.get_present_pwm()
            max_pwm = max(abs(p) for p in present_pwm)
            pwm_percent = max_pwm / 885.0 * 100
            
            # Publish PWM values for logging/monitoring
            pwm_msg = Int32MultiArray()
            pwm_msg.data = [int(p) for p in present_pwm]
            self.pwm_pub.publish(pwm_msg)
            
            # Log every 50 calls (~1 second at 50Hz)
            if self._torque_log_counter % 50 == 0:
                self.get_logger().info(
                    f"[PWM_MONITOR] PWM: th={present_pwm[:3]}, if={present_pwm[3:6]}, mf={present_pwm[6:9]} "
                    f"| MAX={max_pwm} ({pwm_percent:.1f}%) | Goal_current={currents[:3]}"
                )
            
            # [SAFETY] If PWM exceeds limit, scale down currents proportionally (don't stop!)
            if max_pwm > self.max_pwm_limit:
                # Calculate scale factor to bring PWM back under limit
                scale_factor = self.max_pwm_limit / max_pwm * 0.9  # 90% of limit for safety margin
                currents = [int(c * scale_factor) for c in currents]
                if self._torque_log_counter % 10 == 0:  # Log more frequently when limiting
                    self.get_logger().warn(
                        f"[PWM LIMIT] PWM {max_pwm} > limit {self.max_pwm_limit}! "
                        f"Scaling currents by {scale_factor:.2f} -> {currents[:3]}"
                    )
            
            # Warning if approaching limit (80%)
            elif max_pwm > self.max_pwm_limit * 0.8:
                if self._torque_log_counter % 50 == 0:
                    self.get_logger().warn(f"[PWM WARNING] High PWM: {max_pwm} ({pwm_percent:.1f}%) - approaching limit {self.max_pwm_limit}")
                
        except Exception as e:
            pass  # Ignore PWM read errors
        
        if self.safe_mode:
        #     # 주기적 로깅 (너무 noisy 방지)
        #     if self._torque_log_counter % 5000 == 0:
        #         self.get_logger().info(f"[SAFE MODE] Current blocked: {currents[:3]}...")
            return
        
        try:
            self.controller.test_torqueinputs(self.ids, currents, log=False)
        except Exception as e:
            self.get_logger().error(f"test_torqueinputs 실패: {e}")

    # ============================== Force Control (Torque -> Current) ==============================
    def subscribe_target_torques(self, msg: Float32MultiArray):
        """Torque 입력을 current로 변환하여 전송 (use_force_control=True일 때만 호출됨)"""
        if self.use_force_control and not self._torque_playback_enabled:
            return
        if self.defer_force_control_until_policy and not self._policy_ready:
            return
        
        torques = np.array(list(msg.data), dtype=float)
        if len(torques) < len(self.ids):
            self.get_logger().warn(f"torque length {len(torques)} < ids {len(self.ids)}")
            return
        
        # τ(Nm) -> current units: units = τ / (Kt * mA_per_unit)
        denom = self.current_to_torque * self.current_unit_mA if self.current_to_torque > 0 else 1e-6
        current_units = torques / denom
        int_currents = [int(round(c)) for c in current_units[:len(self.ids)]]
        
        self._send_currents(int_currents)
        self._torque_log_counter += 1

    def subscribe_direct_currents(self, msg: Int32MultiArray):
        """Current를 직접 입력 (use_force_control=True일 때만 호출됨)"""
        if self.use_force_control and not self._torque_playback_enabled:
            return
        # Safe mode만 체크, policy 대기 로직 제거
        currents = [int(x) for x in msg.data]
        if len(currents) < len(self.ids):
            return
        currents = currents[:len(self.ids)]
        
        self._send_currents(currents)

    # ============================== Joint State Publishing ==============================
    def publish_joint_state_message(self, targets_units: List[int]) -> None:
        if not self.publish_joint_state or self.joint_state_pub is None:
            return
        now = self.get_clock().now().nanoseconds / 1e9
        # Publish units directly as position
        qpos = [float(u) for u in targets_units]
        # Velocity estimation
        if self._last_qpos is not None and self._last_qpos_time is not None:
            dt = max(1e-6, now - self._last_qpos_time)
            qvel = [(qpos[i] - self._last_qpos[i]) / dt for i in range(len(qpos))]
        else:
            qvel = [0.0] * len(qpos)
        self._last_qpos = list(qpos)
        self._last_qpos_time = now
        self._joint_state_pub_counter += 1
        # 로그는 100번마다 한 번씩만 (30Hz면 약 3초마다)
        # if self._joint_state_pub_counter % 100 == 0:
        #     print(f"[JointState publish #{self._joint_state_pub_counter}] t={now:.6f} pos={qpos[:3]}... vel={qvel[:3]}")
        #     self.get_logger().info(f"[JointState publish #{self._joint_state_pub_counter}] pos={qpos[:3]}... vel={qvel[:3]}")
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.hand_joint_order[:len(qpos)]
        msg.position = qpos
        msg.velocity = qvel
        self.joint_state_pub.publish(msg)

    def _republish_joint_state(self) -> None:
        """Timer callback to republish last joint state at regular intervals."""
        # Force control mode에서는 실제 Dynamixel 위치를 읽어서 publish
        if self.controller is not None and self.use_force_control and self.mode == self.force_control_mode:
            try:
                # 실제 Dynamixel 현재 위치 읽기
                current_positions = self.controller.get_joint_positions()
                if current_positions:
                    self.publish_joint_state_message(current_positions)
            except Exception as e:
                # 읽기 실패 시 마지막 타겟값 사용
                if self._last_targets is not None:
                    self.publish_joint_state_message(self._last_targets)
        elif self._last_targets is not None:
            # Position control mode에서는 마지막 타겟값 사용
            self.publish_joint_state_message(self._last_targets)

    # joint_state_rate_hz 타이머 기반 퍼블리시 완전 제거

    # enable/disable control removed (always enabled)

    def subscribe_policy_predicted(self, msg: Float32MultiArray):
        """첫 정책 예측 수신 시 force control 활성화."""
        if self._policy_ready:
            return
        self._policy_ready = True
        self.get_logger().info("Policy prediction received -> enabling force control inputs now.")
        # 필요 시 여기서 추가 안전 초기화 혹은 모드 스위치 재시도 가능
        if self.controller is not None and self.use_force_control and self.mode != self.force_control_mode:
            try:
                self.get_logger().info(f"(Deferred) Switch operating mode -> current (mode={self.force_control_mode})")
                self.controller.set_operating_mode_all(self.force_control_mode)
                self.mode = self.force_control_mode
            except Exception as e:
                self.get_logger().warn(f"Deferred operating mode switch 실패: {e}")

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