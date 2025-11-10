#!/usr/bin/env python3
import os
import sys
from typing import Optional, List, Tuple, Any

import hydra
from omegaconf import DictConfig

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import WrenchStamped
from std_msgs.msg import String, Bool
import numpy as np
# Optional dependencies
try:
    from omegaconf import OmegaConf  # type: ignore
except Exception:
    OmegaConf = None  # type: ignore

# numpy is a required dependency for this node; already imported above.

try:
    from mms101_controller import MMS101Controller  # original controller
except Exception:
    MMS101Controller = None  # type: ignore

class ForceSensorNode(Node):
    def __init__(self) -> None:
        super().__init__('force_sensor_node')

        # Load config
        self.config = self._load_config()
        if self.config is None:
            self.get_logger().error("설정 로드 실패 -> 종료")
            rclpy.shutdown()
            sys.exit(1)

        # Parameters
        self.declare_parameter('publish_rate_hz', 1000.0)
        self.declare_parameter('use_mock', False)
        self.declare_parameter('config_path', 'config.yaml')  # currently unused

        self.rate = self.get_parameter('publish_rate_hz').get_parameter_value().double_value
        self.use_mock = self.get_parameter('use_mock').get_parameter_value().bool_value
        self.num_sensors = 3  # fixed

        # Controller init
        self.controller = None
        if not self.use_mock:
            try:
                if MMS101Controller is not None:
                    self.controller = MMS101Controller(self.config)
                    self.get_logger().info('Using Original MMS101Controller.')
                else:
                    raise RuntimeError('No MMS101Controller implementation available')
            except Exception as e:
                self.get_logger().error(f'Failed to init controller: {e}')
                self.use_mock = True
        if self.use_mock:
            self.get_logger().warn('Using mock force data.')

        # Per-sensor publishers
        self.pub_sensors = [
            self.create_publisher(WrenchStamped, f'/force_sensor/s{idx+1}/wrench', 10)
            for idx in range(self.num_sensors)
        ]

        # Calibration state publishers
        self._calib_state_pub = self.create_publisher(Bool, '/force_sensor/calibration_mode', 10)
        
        try:
            self.create_subscription(String, '/hand_tracker/key', self._on_key, 10)
            self.get_logger().info("Subscribed /hand_tracker/key for calibration trigger (press 'f')")
        except Exception as e:
            self.get_logger().warn(f"Key subscribe 실패: {e}")
        # State
        self.i = 0
        self.last_values_list = [
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0) for _ in range(self.num_sensors)
        ]

        # Timer
        self.timer = self.create_timer(1.0 / self.rate, self.on_timer)
        try:
            self._calib_state_pub.publish(Bool(data=False))
        except Exception:
            pass

    def _on_key(self, msg: String) -> None:
        if str(msg.data).lower() == 'f':
            # 컨트롤러에 캘리브레이션 모드 설정
            if self.controller is not None:
                setattr(self.controller, 'calibration_mode', True)

    def read_force(self) -> List[Tuple[float, float, float, float, float, float]]:
        # Returns list of (fx, fy, fz, tx, ty, tz) length == num_sensors
        if self.use_mock:
            import math
            # Mock data for testing
            # t 값의 증가량을 줄여서 mock 데이터의 변화 주기를 늦춤 (0.0005 -> 0.0001)
            t = self.i * 0.0001
            values: List[Tuple[float, float, float, float, float, float]] = []
            for s in range(self.num_sensors):
                phase = s * 0.1
                values.append((
                    2 * math.sin(t + phase),
                    0 * math.cos(0.5 * t + phase),
                    0.,
                    0.,
                    0.,
                    0.
                ))
            return values

        # Real controller path
        try:
            if self.controller is None:
                raise RuntimeError('Controller not initialized')
            raw = self.controller.run(self.i)
            
            rows: List[List[float]] = []
            arr = np.array(raw)
            if arr.ndim == 2 and arr.shape[1] >= 6:
                rows = arr[:, :6].astype(float).tolist()
            elif arr.ndim == 1 and arr.size >= 6:
                rows = [arr[:6].astype(float).tolist()]
            if not rows:
                raise ValueError('Unsupported data shape from controller')

            # Normalize to configured num_sensors: pad or truncate
            if len(rows) < self.num_sensors:
                last = rows[-1]
                while len(rows) < self.num_sensors:
                    rows.append(list(last))
            elif len(rows) > self.num_sensors:
                rows = rows[:self.num_sensors]

            values = [tuple(r[:6]) for r in rows]  # type: ignore
            self.last_values_list = values

            if (self.i % 50) == 0 and len(values) > 0:
                # Print up to first 3 sensors safely
                for idx in range(min(3, len(values))):
                    v = values[idx]
                    print(f"Controller values{idx+1}: {[f'{x:.2f}' for x in v]} sum : {sum(v):.2f}")
                print()
            return values
        except Exception as e:
            if (self.i % 200) == 0:
                self.get_logger().warn(f'Using last values due to error: {e}')
            # ensure length == num_sensors
            lv = list(self.last_values_list)
            if len(lv) < self.num_sensors:
                while len(lv) < self.num_sensors:
                    lv.append(lv[-1] if lv else (0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
            elif len(lv) > self.num_sensors:
                lv = lv[:self.num_sensors]
            return lv

    def on_timer(self) -> None:
        self.i += 1
        values = self.read_force()
        if self.controller.calibration_mode :
            pass
        self._calib_state_pub.publish(Bool(data=getattr(self.controller, 'calibration_mode', False)))

        # 각 센서 별 토픽 퍼블리시
        now = self.get_clock().now().to_msg()
        for idx, row in enumerate(values):
            if idx >= len(self.pub_sensors):
                break
            fx, fy, fz, tx, ty, tz = row
            msg = WrenchStamped()
            msg.header.stamp = now
            msg.header.frame_id = f'force_sensor/s{idx+1}'
            msg.wrench.force.x = float(fx)
            msg.wrench.force.y = float(fy)
            msg.wrench.force.z = float(fz)
            msg.wrench.torque.x = float(tx)
            msg.wrench.torque.y = float(ty)
            msg.wrench.torque.z = float(tz)
            self.pub_sensors[idx].publish(msg)

    def _load_config(self) -> Optional[Any]:
        try:
            pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            cfg_path = os.path.join(pkg_dir, 'resource', 'sensor_parameter', 'config.yaml')
            if os.path.exists(cfg_path):
                if OmegaConf is not None:
                    cfg = OmegaConf.load(cfg_path)
                    self.get_logger().debug(f"Loaded config.yaml ({cfg_path})")
                    return cfg
                else:
                    # Fallback to plain YAML if OmegaConf is unavailable
                    try:
                        import yaml  # type: ignore
                        with open(cfg_path, 'r') as f:
                            data = yaml.safe_load(f)
                        self.get_logger().debug(f"Loaded config.yaml without OmegaConf ({cfg_path})")
                        return data
                    except Exception:
                        pass
        except Exception as e:
            self.get_logger().warn(f"Config load 실패: {e}")
            quit(1)



def main() -> None:
    rclpy.init()
    node = ForceSensorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()