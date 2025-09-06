#!/usr/bin/env python3
from __future__ import annotations

"""Robot Controller Node (정리/복구 + Hand Joint CSV Logging)

기능 요약:
    * (선택) Hydra/OmegaConf 설정 로드 (--ros-args 있으면 Hydra 비활성)
    * Hand JointState -> Dynamixel 목표치 (EMA, 클램프, 스텝 제한, baseline 기준 편차 -> units)
    * Falcon 3축 입력 레거시 매핑 (encoders / position)
    * 키보드(h) / Bool 토픽 hand enable 토글
    * 상세 DEBUG 로그 + (옵션) CSV 로 hand joint 변환 과정 저장
"""

import os
import csv
import pathlib
import datetime
from typing import List, Optional, Dict, Any

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32MultiArray, Bool
from geometry_msgs.msg import Vector3Stamped
from sensor_msgs.msg import JointState

try:  # Hydra/OmegaConf (optional)
    import hydra  # type: ignore
    from omegaconf import DictConfig, OmegaConf  # type: ignore
except Exception:  # pragma: no cover
    hydra = None  # type: ignore
    DictConfig = None  # type: ignore
    OmegaConf = None  # type: ignore

try:  # Dynamixel (optional)
    from dynamixel_control import DynamixelControl  # type: ignore
except Exception:  # pragma: no cover
    DynamixelControl = None  # type: ignore


class RobotControllerNode(Node):
    def __init__(self, config: Optional['DictConfig'] = None):  # type: ignore[name-defined]
        super().__init__('robot_controller_node')

        # ---------------------- 설정 로드
        self.config = config if config is not None else self._load_config()
        cfg_dyn = getattr(self.config, 'dynamixel', None) if self.config is not None else None
        default_ids = list(getattr(cfg_dyn, 'ids', [11, 12, 21, 22, 31, 32]))
        default_mode = int(getattr(getattr(cfg_dyn, 'control_modes', {}), 'default_mode', 3))

        # ---------------------- 파라미터 선언
        self.declare_parameter('ids', default_ids)
        self.declare_parameter('mode', default_mode)
        self.declare_parameter('scale', [1.0, 1.0, 1.0])
        self.declare_parameter('offset', [1000, 1000, 1000])
        self.declare_parameter('clip_min', [0, 0, 0])
        self.declare_parameter('clip_max', [4095, 4095, 4095])
        self.declare_parameter('input_source', getattr(self.config, 'input_source', 'hand'))
        self.declare_parameter('use_encoders', True)
        self.declare_parameter('hand_joint_order', ['thumb_mcp','thumb_ip','index_pip','index_dip','middle_pip','middle_dip'])
        self.declare_parameter('hand_input_deg_min', 160.0)
        self.declare_parameter('hand_input_deg_max', 210.0)
        self.declare_parameter('baseline_deg', 180.0)
        self.declare_parameter('servo_units_per_degree', 4096.0/360.0)
        self.declare_parameter('smooth_alpha', 0.3)
        self.declare_parameter('motion_scale', 0.3)
        self.declare_parameter('max_step_units', 20.0)
        self.declare_parameter('test_mode', getattr(self.config, 'test_mode', True))
        self.declare_parameter('arm', getattr(self.config, 'arm', False))
        self.declare_parameter('hand_enabled', True)
        self.declare_parameter('hand_enable_topic', '/hand_tracker/enable')
        self.declare_parameter('hand_baseline_units', 2000.0)
        self.declare_parameter('hand_output_units_min', 1800.0)
        self.declare_parameter('hand_output_units_max', 2200.0)
        self.declare_parameter('hand_disable_behavior', 'hold')  # hold|baseline
        self.declare_parameter('keyboard_toggle_enable', True)
        self.declare_parameter('keyboard_toggle_key', 'h')
        self.declare_parameter('log_hand_csv_enable', True)
        self.declare_parameter('log_hand_csv_path', '')

        # ---------------------- 파라미터 해석
        def _fit_len(arr, n: int, fill):
            arr = list(arr)
            if len(arr) < n:
                arr.extend([fill]*(n-len(arr)))
            return arr[:n]

        ids_val = self.get_parameter('ids').value or default_ids
        self.ids: List[int] = [int(x) for x in ids_val]
        self.mode: int = int(self.get_parameter('mode').value or default_mode)
        self.scale: List[float] = [float(x) for x in (self.get_parameter('scale').value or [1.0,1.0,1.0])]
        offset_param = list(self.get_parameter('offset').value or [1000,1000,1000])
        clip_min_param = list(self.get_parameter('clip_min').value or [0,0,0])
        clip_max_param = list(self.get_parameter('clip_max').value or [4095,4095,4095])
        self.offset: List[float] = [float(x) for x in _fit_len(offset_param, len(self.ids), 1000.0)]
        self.clip_min: List[int] = [int(x) for x in _fit_len(clip_min_param, len(self.ids), 0)]
        self.clip_max: List[int] = [int(x) for x in _fit_len(clip_max_param, len(self.ids), 4095)]
        self.input_source: str = str(self.get_parameter('input_source').value or getattr(self.config,'input_source','hand'))
        self.use_encoders: bool = bool(self.get_parameter('use_encoders').value)
        self.hand_joint_order: List[str] = [str(x).lower() for x in (self.get_parameter('hand_joint_order').value or [])]
        self.hand_input_deg_min: float = float(self.get_parameter('hand_input_deg_min').value or 160.0)
        self.hand_input_deg_max: float = float(self.get_parameter('hand_input_deg_max').value or 210.0)
        self.baseline_deg: float = float(self.get_parameter('baseline_deg').value or 180.0)
        self.units_per_deg: float = float(self.get_parameter('servo_units_per_degree').value or (4096.0/360.0))
        self.smooth_alpha: float = float(self.get_parameter('smooth_alpha').value or 0.3)
        self.motion_scale: float = float(self.get_parameter('motion_scale').value or 0.3)
        self.max_step_units: float = float(self.get_parameter('max_step_units').value or 20.0)
        self.test_mode: bool = bool(self.get_parameter('test_mode').value)
        self.arm: bool = bool(self.get_parameter('arm').value)
        self.hand_enabled: bool = bool(self.get_parameter('hand_enabled').value if self.get_parameter('hand_enabled').value is not None else True)
        self.hand_enable_topic: str = str(self.get_parameter('hand_enable_topic').value or '/hand_tracker/enable')
        self.hand_baseline_units: float = float(self.get_parameter('hand_baseline_units').value or 2000.0)
        self.hand_output_units_min: float = float(self.get_parameter('hand_output_units_min').value or 1800.0)
        self.hand_output_units_max: float = float(self.get_parameter('hand_output_units_max').value or 2200.0)
        self.hand_disable_behavior: str = str(self.get_parameter('hand_disable_behavior').value or 'hold').lower()
        self.keyboard_toggle_enable: bool = bool(self.get_parameter('keyboard_toggle_enable').value)
        self.keyboard_toggle_key: str = str(self.get_parameter('keyboard_toggle_key').value or 'h')
        self.log_hand_csv_enable: bool = bool(self.get_parameter('log_hand_csv_enable').value)
        self.log_hand_csv_path: str = str(self.get_parameter('log_hand_csv_path').value or '')
        if self.log_hand_csv_enable and self.input_source != 'hand':
            self.get_logger().warn('log_hand_csv_enable 활성화 되었지만 input_source != hand -> 비활성화')
            self.log_hand_csv_enable = False

        # ---------------------- Dynamixel 연결
        self.controller = None
        if DynamixelControl is not None and cfg_dyn is not None:
            try:  # pragma: no cover
                self.controller = DynamixelControl(cfg_dyn)
                self.controller.connect()
                self.get_logger().info(f"Dynamixel connected (ids={self.ids}, mode={self.mode})")
            except Exception as e:
                self.get_logger().error(f"Dynamixel init/connect 실패: {e} -> Dry-run")
        elif DynamixelControl is None:
            self.get_logger().warn('dynamixel_control 모듈 없음 -> Dry-run')
        else:
            self.get_logger().warn('dynamixel 설정 없음 -> Dry-run')

        self.get_logger().info(f"[SETUP] src={self.input_source} test={self.test_mode} arm={self.arm} ids={self.ids} motion_scale={self.motion_scale}")

        # ---------------------- 내부 상태 & CSV
        self._filt_deg: Dict[str, float] = {}
        self._last_targets: Optional[List[int]] = None
        self.base_positions: Optional[List[float]] = None
        self._keyboard_thread = None
        self._hand_csv_fp = None
        self._hand_csv_writer = None
        self._last_joint_debug: Optional[List[Dict[str, Any]]] = None
        if self.log_hand_csv_enable:
            try:
                if not self.log_hand_csv_path:
                    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                    self.log_hand_csv_path = str(pathlib.Path.cwd() / f"{ts}_handangles.csv")
                p = pathlib.Path(self.log_hand_csv_path)
                p.parent.mkdir(parents=True, exist_ok=True)
                need_header = (not p.exists()) or p.stat().st_size == 0
                self._hand_csv_fp = open(p, 'a', newline='')
                self._hand_csv_writer = csv.writer(self._hand_csv_fp)
                if need_header:
                    header = ['t_sec','t_nanosec']
                    # joint 별 raw_deg, target_units wide 포맷
                    for jn in self.hand_joint_order:
                        header.append(f"{jn}_raw_deg")
                        header.append(f"{jn}_target_units")
                    self._hand_csv_writer.writerow(header)
                self.get_logger().info(f"Hand CSV logging (wide) -> {p}")
            except Exception as e:
                self.get_logger().error(f"CSV 파일 열기 실패: {e}")
                self.log_hand_csv_enable = False

        # ---------------------- 구독 설정
        if self.input_source == 'hand':
            self.sub_js = self.create_subscription(JointState, '/hand_tracker/joint_states', self.on_joint_state, 10)
            self.get_logger().info('Input source: /hand_tracker/joint_states')
            if self.hand_enable_topic:
                self.sub_enable = self.create_subscription(Bool, self.hand_enable_topic, self.on_hand_enable, 10)
                self.get_logger().info(f"Hand enable topic: {self.hand_enable_topic} (start={self.hand_enabled})")
        else:
            if self.use_encoders:
                self.sub_enc = self.create_subscription(Int32MultiArray, '/falcon/encoders', self.on_encoders, 10)
                self.get_logger().info('Input source: /falcon/encoders')
            else:
                self.sub_pos = self.create_subscription(Vector3Stamped, '/falcon/position', self.on_position, 10)
                self.get_logger().info('Input source: /falcon/position')

        # ---------------------- 키보드 토글
        if self.input_source == 'hand' and self.keyboard_toggle_enable:
            try:  # pragma: no cover
                import sys, threading
                if sys.stdin.isatty():
                    self._keyboard_thread = threading.Thread(target=self._keyboard_loop, daemon=True)
                    self._keyboard_thread.start()
                    self.get_logger().info(f"Press '{self.keyboard_toggle_key}' to toggle hand input.")
                else:
                    self.get_logger().warn('STDIN TTY 아님 -> 키보드 토글 비활성')
            except Exception as e:
                self.get_logger().warn(f"Keyboard thread 실패: {e}")

    # ============================== Utils
    def _load_config(self) -> Optional['DictConfig']:  # type: ignore[name-defined]
        if OmegaConf is None:
            return None
        try:
            pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            cfg_path = os.path.join(pkg_dir, 'resource', 'robot_parameter', 'config.yaml')
            if os.path.exists(cfg_path):
                cfg = OmegaConf.load(cfg_path)
                self.get_logger().debug(f"Loaded config.yaml ({cfg_path})")
                return cfg
        except Exception as e:
            self.get_logger().warn(f"Config load 실패: {e}")
        return None

    def _ensure_base_positions(self):
        if self.base_positions is not None:
            return
        base: Optional[List[float]] = None
        if self.controller is not None:
            for name in ('get_joint_positions','get_current_positions','present_positions','read_positions'):
                try:  # pragma: no cover
                    fn = getattr(self.controller, name, None)
                    if callable(fn):
                        res = fn() if name != 'read_positions' else fn(self.ids)
                        if isinstance(res,(list,tuple)) and len(res)>=1:
                            base = [float(res[i]) if i < len(res) else 0.0 for i in range(len(self.ids))]
                            break
                except Exception:
                    pass
        if base is None:
            base = [float(self.offset[i]) if i < len(self.offset) else 0.0 for i in range(len(self.ids))]
        self.base_positions = base

    # ============================== Falcon mapping
    def map_and_send(self, vec3: List[float]):
        self._ensure_base_positions()
        base = self.base_positions or [2500.0 for _ in range(max(3, len(self.ids)))]
        cmd: List[int] = []
        n = min(3, len(self.ids))
        for i in range(n):
            x = float(vec3[i])
            delta = (x / 1600.0) * 200.0
            delta = max(-100.0, min(100.0, delta))
            target = base[i] - delta
            target = max(float(self.clip_min[i]), min(float(self.clip_max[i]), target))
            cmd.append(int(round(target)))
        if self.controller is not None and self.arm and not self.test_mode:
            try:  # pragma: no cover
                self.controller.set_joint_positions(cmd)
            except Exception as e:
                self.get_logger().error(f"Falcon set_joint_positions 실패: {e}")
        else:
            self.get_logger().info(f"[DRY falcon] {cmd} (input={vec3[:n]})")

    # ============================== Hand JointState
    def on_joint_state(self, msg: JointState):
        name_to_deg: Dict[str, float] = {}
        for name, pos in zip(msg.name, msg.position):
            try:
                if pos is None:
                    continue
                val = float(pos)
                if val != val:
                    continue
                name_to_deg[str(name).lower()] = val * 180.0 / 3.141592653589793
            except Exception:
                continue
        if name_to_deg:
            self.get_logger().debug('hand_joints ' + ', '.join(f"{k}:{v:.1f}" for k,v in name_to_deg.items()))
        targets = self._compute_hand_targets(name_to_deg)
        if targets:
            self._send_targets(targets)
            if self.log_hand_csv_enable and self._hand_csv_writer and self._last_joint_debug:
                try:
                    now = self.get_clock().now().to_msg()
                    row = [now.sec, now.nanosec]
                    jd_index = {d['joint']: d for d in self._last_joint_debug}
                    for jn in self.hand_joint_order:
                        d = jd_index.get(jn)
                        if d:
                            rv = d['raw_deg']
                            row.append(f"{rv:.4f}" if rv == rv else '')
                            row.append(int(d['target_units']))
                        else:
                            row.append('')
                            row.append('')
                    self._hand_csv_writer.writerow(row)
                    if self._hand_csv_fp:
                        self._hand_csv_fp.flush()
                except Exception as e:
                    self.get_logger().warn(f"CSV 로그 실패: {e}")

    def _compute_hand_targets(self, name_to_deg: Dict[str, float]) -> Optional[List[int]]:
        self._ensure_base_positions()
        base = self.base_positions or [2500.0 for _ in range(len(self.ids))]
        count = min(len(self.ids), len(self.hand_joint_order))
        if count == 0:
            return None
        if not self.hand_enabled and self.hand_disable_behavior == 'hold' and self._last_targets is not None:
            return list(self._last_targets)

        targets: List[int] = []
        joint_debug: List[Dict[str, Any]] = []
        for i in range(count):
            jn = self.hand_joint_order[i]
            raw = name_to_deg.get(jn)
            prev = self._filt_deg.get(jn)
            if raw is None:
                filt = prev if prev is not None else self.baseline_deg
            else:
                raw = max(self.hand_input_deg_min - 30.0, min(self.hand_input_deg_max + 10.0, raw))
                filt = raw if prev is None else (self.smooth_alpha*raw + (1-self.smooth_alpha)*prev)
            self._filt_deg[jn] = filt
            clamped = max(self.hand_input_deg_min, min(self.hand_input_deg_max, filt))
            delta = clamped - self.baseline_deg
            tgt = self.hand_baseline_units + delta * self.units_per_deg * self.motion_scale
            if tgt < self.hand_output_units_min:
                tgt = self.hand_output_units_min
            elif tgt > self.hand_output_units_max:
                tgt = self.hand_output_units_max
            units_int = self._clip_target(i, tgt)
            targets.append(units_int)
            joint_debug.append({  # type: ignore[arg-type]
                'joint': jn,
                'raw_deg': float(name_to_deg.get(jn, float('nan'))),
                'filt_deg': float(self._filt_deg.get(jn, self.baseline_deg)),
                'clamped_deg': float(clamped),
                'target_units': float(units_int),
            })

        if targets:
            parts = []
            for i in range(count):
                jn = self.hand_joint_order[i]
                rd = name_to_deg.get(jn, -1.0)
                fd = self._filt_deg.get(jn, self.baseline_deg)
                cp = max(self.hand_input_deg_min, min(self.hand_input_deg_max, fd))
                parts.append(f"{jn}:{rd:.1f}->{fd:.1f}->{cp:.1f} tgt={targets[i]}")
            self.get_logger().debug('txf ' + ' | '.join(parts))
            self._last_joint_debug = joint_debug

        if len(self.ids) > count:
            for i in range(count, len(self.ids)):
                targets.append(self._clip_target(i, base[i]))
        return targets

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
        if self.test_mode or not self.arm or self.controller is None:
            self.get_logger().info(f"[DRY hand] {targets} (arm={self.arm} test={self.test_mode})")
            return
        try:  # pragma: no cover
            self.controller.set_joint_positions(targets)
        except Exception as e:
            self.get_logger().error(f"set_joint_positions 실패: {e}")

    # ============================== Falcon inputs
    def on_encoders(self, msg: Int32MultiArray):
        if len(msg.data) >= 3:
            self.map_and_send([float(msg.data[0]), float(msg.data[1]), float(msg.data[2])])

    def on_position(self, msg: Vector3Stamped):
        self.map_and_send([msg.vector.x, msg.vector.y, msg.vector.z])

    # ============================== Enable control
    def on_hand_enable(self, msg: Bool):
        self._set_hand_enabled(bool(msg.data), source='topic')

    def _set_hand_enabled(self, enabled: bool, source: str = 'unknown'):
        old = self.hand_enabled
        self.hand_enabled = bool(enabled)
        if old != self.hand_enabled:
            self.get_logger().info(f"hand_enabled {old}->{self.hand_enabled} (source={source})")
            if not self.hand_enabled and self.hand_disable_behavior == 'baseline':
                self._filt_deg = {}

    # ============================== Keyboard loop
    def _keyboard_loop(self):  # pragma: no cover
        import sys, select, termios, tty
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while rclpy.ok():
                r, _, _ = select.select([fd], [], [], 0.2)
                if fd in r:
                    ch = sys.stdin.read(1)
                    if ch == self.keyboard_toggle_key:
                        self._set_hand_enabled(not self.hand_enabled, source='keyboard')
                    elif ch in ('q','\u0003'):
                        pass
        except Exception as e:
            self.get_logger().warn(f"keyboard loop error: {e}")
        finally:
            try:
                termios.tcsetattr(fd, termios.TCSADRAIN, old)
            except Exception:
                pass


# ============================== Entrypoints
def _run(node: RobotControllerNode):
    try:
        rclpy.spin(node)
    finally:
        try:
            if getattr(node, '_hand_csv_fp', None):
                node._hand_csv_fp.close()  # type: ignore
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


def _hydra_task(cfg: 'DictConfig'):  # type: ignore
    rclpy.init()
    node = RobotControllerNode(cfg)
    _run(node)


def main():  # ROS only / hydra 우회
    rclpy.init()
    node = RobotControllerNode()
    _run(node)


if __name__ == '__main__':
    import sys
    use_hydra = hydra is not None and DictConfig is not None and '--ros-args' not in sys.argv
    if use_hydra:
        cfg_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'resource', 'robot_parameter')
        hydra_app = hydra.main(version_base=None, config_path=cfg_path, config_name='config')(_hydra_task)  # type: ignore
        hydra_app()
    else:
        main()