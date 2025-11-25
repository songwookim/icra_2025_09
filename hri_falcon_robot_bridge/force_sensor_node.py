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

        # Parameters (declare first so we can override use_mock if config missing)
        self.declare_parameter('publish_rate_hz', 1000.0)
        self.declare_parameter('use_mock', False)
        self.declare_parameter('config_path', 'config.yaml')  # currently unused

        # Load config; if unavailable we will NOT publish mock data (strict gating)
        self.config = self._load_config()
        if self.config is None:
            self.get_logger().warn("config.yaml 없음 -> 측정 불가 상태. mock 퍼블리시 비활성화 (토픽 미발행).")

        self.rate = self.get_parameter('publish_rate_hz').get_parameter_value().double_value
        # use_mock 파라미터는 이제 무시: 실제 측정 불가 시 퍼블리시 중단
        requested_mock = self.get_parameter('use_mock').get_parameter_value().bool_value
        self.use_mock = False  # 기존 sinusoid 기능 제거
        self.measurement_available = (self.config is not None)
        self.num_sensors = 3  # fixed

        # Controller init
        self.controller = None
        if self.measurement_available:
            try:
                if MMS101Controller is not None:
                    self.controller = MMS101Controller(self.config)
                    self.get_logger().info('Using Original MMS101Controller.')
                else:
                    raise RuntimeError('No MMS101Controller implementation available')
            except Exception as e:
                self.get_logger().error(f'컨트롤러 초기화 실패: {e} -> 측정 불가. 퍼블리시 중단.')
                self.measurement_available = False
                self.controller = None
        if not self.measurement_available:
            self.get_logger().warn('ForceSensorNode 측정 불가 상태: force 토픽을 발행하지 않습니다.')

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

    def read_force(self) -> Optional[List[Tuple[float, float, float, float, float, float]]]:
        """실측 값을 읽어 반환. 측정 불가 시 None 반환 (퍼블리시 스킵)."""
        if not self.measurement_available:
            return None

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

            values: List[Tuple[float, float, float, float, float, float]] = []
            for r in rows:
                fx, fy, fz, tx, ty, tz = r[:6]
                values.append((float(fx), float(fy), float(fz), float(tx), float(ty), float(tz)))
            self.last_values_list = values

            if (self.i % 50) == 0 and len(values) > 0:
                # Print up to first 3 sensors safely
                for idx in range(min(3, len(values))):
                    v = values[idx]
                    print(f"Controller values{idx+1}: {[f'{x:.2f}' for x in v]} sum : {sum(v):.2f}")
                print()
            return values
        except Exception as e:
            if (self.i % 50) == 0:
                self.get_logger().warn(f'측정 오류 발생: {e} -> 이번 주기 스킵')
            return None

    def on_timer(self) -> None:
        self.i += 1
        values = self.read_force()
        if values is None:
            # 측정 불가이거나 오류 -> 퍼블리시 건너뜀
            return
        # Publish calibration mode safely (controller may be None)
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
                    try:
                        import yaml  # type: ignore
                        with open(cfg_path, 'r') as f:
                            data = yaml.safe_load(f)
                        self.get_logger().debug(f"Loaded config.yaml without OmegaConf ({cfg_path})")
                        return data
                    except Exception as e:
                        self.get_logger().warn(f"YAML 파싱 실패: {e}")
                        return None
            return None
        except Exception as e:
            self.get_logger().warn(f"Config load 예외: {e}")
            return None



def main() -> None:
    rclpy.init()
    node = ForceSensorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()