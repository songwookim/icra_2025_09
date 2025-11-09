#!/usr/bin/env python3
from __future__ import annotations

"""
Robot Controller Node (units-only from hand_tracker) with optional CSV logging.

- Subscribes: Int32MultiArray (9) on /hand_tracker/targets_units
- Commands: Dynamixel via DynamixelControl if available and safe_mode=False; otherwise dry-run logs
- Initial posture: fixed initial_val applied once at startup and used as baseline when disabled
"""

import os
import sys
import csv
import datetime
import pathlib
from typing import List, Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32MultiArray

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
            self.get_logger().error("설정 로드 실패 -> 종료")
            rclpy.shutdown()
            sys.exit(1)
        cfg_dyn = getattr(self.config, 'dynamixel', None) if self.config is not None else None
        default_ids = list(getattr(cfg_dyn, 'ids', [11, 12, 21, 22, 31, 32]))
        default_mode = int(getattr(getattr(cfg_dyn, 'control_modes', {}), 'default_mode', 3))

        # 파라미터 선언
        self.declare_parameter('ids', default_ids)
        self.declare_parameter('mode', default_mode)
        self.declare_parameter('scale', [1.0, 1.0, 1.0])
        self.declare_parameter('offset', [1000, 1000, 1000])
        self.declare_parameter('clip_min', [0, 0, 0])
        self.declare_parameter('clip_max', [4095, 4095, 4095])
        self.declare_parameter('hand_joint_order', [
            'thumb_cmc','thumb_mcp','thumb_ip',
            'index_mcp','index_hpip','index_dip',
            'middle_mcp','middle_pip','middle_dip'
        ])
        self.declare_parameter('hand_units_topic', '/hand_tracker/targets_units')
        self.declare_parameter('max_step_units', 20.0)
        self.declare_parameter('safe_mode', True)
        # hand enable/disable parameters removed (always enabled)
        self.declare_parameter('log_hand_csv_enable', True)
        self.declare_parameter('log_hand_csv_path', '')
        # initial_val_default = [1240, 2187, 2103, 1240, 2187, 2103, 1240, 2187, 2103]
        initial_val_default = [1154, 2019, 1940, 1000, 1960, 1960, 1138, 2048, 1801]
        self.declare_parameter('initial_val', initial_val_default)
        

        # 유틸
        def _fit_len(arr, n: int, fill):
            arr = list(arr)
            if len(arr) < n:
                arr.extend([fill] * (n - len(arr)))
            return arr[:n]

        # 파라미터 값 읽기
        ids_val = self.get_parameter('ids').value or default_ids
        self.ids = [int(x) for x in ids_val]
        self.mode = int(self.get_parameter('mode').value or default_mode)
        self.scale = [float(x) for x in (self.get_parameter('scale').value or [1.0, 1.0, 1.0])]
        offset_param = list(self.get_parameter('offset').value or [1000, 1000, 1000])
        clip_min_param = list(self.get_parameter('clip_min').value or [0, 0, 0])
        clip_max_param = list(self.get_parameter('clip_max').value or [4095, 4095, 4095])
        self.offset = [float(x) for x in _fit_len(offset_param, len(self.ids), 1000.0)]
        self.clip_min = [int(x) for x in _fit_len(clip_min_param, len(self.ids), 0)]
        self.clip_max = [int(x) for x in _fit_len(clip_max_param, len(self.ids), 4095)]
        self.hand_joint_order = [str(x).lower() for x in (self.get_parameter('hand_joint_order').value or [])]
        self.hand_units_topic = str(self.get_parameter('hand_units_topic').value or '/hand_tracker/targets_units')
        self.max_step_units = float(self.get_parameter('max_step_units').value or 20.0)
        self.safe_mode = bool(self.get_parameter('safe_mode').value)
        # hand enable/disable logic removed -> always enabled
        self.hand_enabled = True
        self.log_hand_csv_enable = bool(self.get_parameter('log_hand_csv_enable').value)
        self.log_hand_csv_path = str(self.get_parameter('log_hand_csv_path').value or '')
        init_param = list(self.get_parameter('initial_val').value or [1240, 2187, 2103, 1240, 2187, 2103, 1240, 2187, 2103])
        self.initial_val = [int(x) for x in _fit_len(init_param, len(self.ids), 2000)]

        # Backend 연결 (Dynamixel or Dry-run)
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

        self.get_logger().info(f"[SETUP] src=hand(units) test={self.safe_mode} ids={self.ids}")

        # 내부 상태 & CSV
    # removed _filt_deg (was only used for disable behavior)
        self._last_targets: Optional[List[int]] = None
        self.base_positions = list(self.initial_val)
        # keyboard toggle removed
        self._hand_csv_fp = None
        self._hand_csv_writer = None
        self._last_joint_debug = None
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
                    header = ['t_sec', 't_nanosec']
                    for jn in self.hand_joint_order:
                        header.append(f"{jn}_raw_deg")
                        header.append(f"{jn}_target_units")
                    self._hand_csv_writer.writerow(header)
                self.get_logger().info(f"Hand CSV logging (wide) -> {p}")
            except Exception as e:
                self.get_logger().error(f"CSV 파일 열기 실패: {e}")
                self.log_hand_csv_enable = False

        # 구독 설정
        self.sub_units = self.create_subscription(Int32MultiArray, self.hand_units_topic, self.on_units_targets, 10)
        self.get_logger().info(f"Input source: {self.hand_units_topic} (units)")
        # enable topic removed (always enabled)

        # 시작 시 초기 포즈로 세팅(한 번)
        try:
            self.get_logger().info(f"Apply initial posture: {self.initial_val}")
            self._send_targets(self.initial_val)
        except Exception as e:
            self.get_logger().warn(f"초기 포즈 적용 실패: {e}")

    # keyboard toggle feature removed

    # ============================== Utils
    def _load_config(self) -> Optional['DictConfig']:  # type: ignore[name-defined]
        try:
            pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            cfg_path = os.path.join(pkg_dir, 'resource', 'robot_parameter', 'config.yaml')
            if os.path.exists(cfg_path):
                cfg = OmegaConf.load(cfg_path)
                self.get_logger().debug(f"Loaded config.yaml ({cfg_path})")
                return cfg
        except Exception as e:
            self.get_logger().warn(f"Config load 실패: {e}")

    # ============================== Hand Units (Int32MultiArray, 9 elems)
    def on_units_targets(self, msg: Int32MultiArray):
        data = list(msg.data or [])
        data[0] = data[0] - 1000  # thumb_cmc offset 
        data[3] = data[3] - 1000  # thumb_cmc offset 
        # data[6] = data[6] - 1000  # thumb_cmc offset 
        # 데이터가 아직 없고 첫 호출이면 초기 포즈 적용 후 종료
        if not data and self._last_targets is None:
            self.get_logger().info("No data yet -> apply initial posture once")
            self._send_targets(self.initial_val)
            return
        # base는 설정된 초기 포즈 사용
        base = [float(v) for v in (self.base_positions or self.initial_val)]
        # decide commanded targets depending on enabled state
        # build targets aligned to motor ids length (always enabled)
        n = min(len(self.ids), len(data))
        clipped = [self._clip_target(i, float(int(data[i]))) for i in range(n)]
        for i in range(n, len(self.ids)):
            clipped.append(self._clip_target(i, base[i]))
        final_targets = clipped
        # send to motors (or dry-run); step-limit is applied inside
        self._send_targets(final_targets)
        # CSV logging: record final commanded target_units per joint order; raw_deg empty
        if self.log_hand_csv_enable and self._hand_csv_writer:
            try:
                now = self.get_clock().now().to_msg()
                row = [now.sec, now.nanosec]
                sent = self._last_targets or []
                for i, _ in enumerate(self.hand_joint_order):
                    row.append('')  # raw_deg
                    tu = sent[i] if i < len(sent) else ''
                    row.append(int(tu) if tu != '' else '')
                self._hand_csv_writer.writerow(row)
                if self._hand_csv_fp:
                    self._hand_csv_fp.flush()
            except Exception as e:
                self.get_logger().warn(f"CSV 로그 실패(units): {e}")

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

    # enable/disable control removed (always enabled)

    # _keyboard_loop removed


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

def main():  # ROS only / hydra 우회
    rclpy.init()
    node = RobotControllerNode()
    _run(node)


if __name__ == '__main__':
    main()