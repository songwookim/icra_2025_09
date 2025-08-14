#!/usr/bin/env python3
from typing import Optional, List, Tuple, Any

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import WrenchStamped
from std_msgs.msg import Float64MultiArray, MultiArrayDimension
import numpy as np
# Optional dependencies
try:
    from omegaconf import OmegaConf  # type: ignore
except Exception:
    OmegaConf = None  # type: ignore

try:
    import numpy as np  # type: ignore
except Exception:
    np = None  # type: ignore

try:
    from mms101_controller import MMS101Controller  # placeholder import
except Exception:
    MMS101Controller = None  # type: ignore


class ForceSensorNode(Node):
    def __init__(self) -> None:
        super().__init__('force_sensor_node')
        # Publishers
        self.pub_array = self.create_publisher(Float64MultiArray, '/force_sensor/wrench_array', 10)
        self.pub_legacy = self.create_publisher(WrenchStamped, '/force_sensor/wrench', 10)  # first sensor only
        # Parameters
        self.declare_parameter('publish_rate_hz', 200.0)
        self.declare_parameter('use_mock', True)
        self.declare_parameter('config_path', 'config.yaml')
        self.declare_parameter('num_sensors', 3)
        self.rate = self.get_parameter('publish_rate_hz').get_parameter_value().double_value
        self.use_mock = self.get_parameter('use_mock').get_parameter_value().bool_value
        cfg_path = self.get_parameter('config_path').get_parameter_value().string_value
        self.num_sensors = int(self.get_parameter('num_sensors').get_parameter_value().integer_value)
        # Controller
        self.controller = None
        if not self.use_mock and MMS101Controller is not None:
            cfg = None
            if OmegaConf is not None:
                try:
                    cfg = OmegaConf.load(cfg_path)
                except Exception as e:
                    self.get_logger().warn(f'Failed to load config via OmegaConf: {e}')
            try:
                self.controller = MMS101Controller(cfg)
            except Exception as e:
                self.get_logger().error(f'Failed to init MMS101Controller: {e}')
                self.use_mock = True
        else:
            if not self.use_mock:
                self.get_logger().warn('MMS101Controller not found. Falling back to mock.')
            self.use_mock = True
        # Per-sensor publishers (/force_sensor/s{i}/wrench)
        self.pub_sensors = [
            self.create_publisher(WrenchStamped, f'/force_sensor/s{idx+1}/wrench', 10)
            for idx in range(self.num_sensors)
        ]
        # State
        self.i = 0
        self.last_values_list = [
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0) for _ in range(max(1, self.num_sensors))
        ]
        # Timer
        self.timer = self.create_timer(1.0 / self.rate, self.on_timer)

    def read_force(self) -> List[Tuple[float, float, float, float, float, float]]:
        # Returns list of (fx, fy, fz, tx, ty, tz) length == num_sensors
        if self.use_mock:
            import math
            t = self.i * 0.01
            values: List[Tuple[float, float, float, float, float, float]] = []
            for s in range(self.num_sensors):
                phase = s * 0.3
                values.append((
                    8.3 * math.sin(t + phase),
                    2.2 * math.cos(0.5 * t + phase),
                    -10.6 + 0.1 * math.sin(0.2 * t + phase),
                    0.02 + 0.001 * s,
                    -0.038 + 0.002 * s,
                    -0.0008 - 0.0002 * s
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
            
            if (self.i % 50) == 0:
                print(f"\n\n Controller values1: {[f'{v:.2f}' for v in values[0]]} sum : {sum(values[0]):.2f} \n")
                print(f"Controller values2: {[f'{v:.2f}' for v in values[1]]} sum : {sum(values[1]):.2f} \n")
                print(f"Controller values3: {[f'{v:.2f}' for v in values[2]]} sum : {sum(values[2]):.2f} \n\n")
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

        # Publish combined array: shape (num_sensors, 6), row-major flatten
        arr_msg = Float64MultiArray()
        d0 = MultiArrayDimension(label='sensor', size=len(values), stride=6)
        d1 = MultiArrayDimension(label='axis', size=6, stride=1)
        arr_msg.layout.dim = [d0, d1]
        arr_msg.data = [float(x) for row in values for x in row]
        self.pub_array.publish(arr_msg)

        # Publish first sensor on legacy topic for compatibility
        if values:
            fx, fy, fz, tx, ty, tz = values[0]
            msg = WrenchStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'force_sensor'
            msg.wrench.force.x = fx
            msg.wrench.force.y = fy
            msg.wrench.force.z = fz
            msg.wrench.torque.x = tx
            msg.wrench.torque.y = ty
            msg.wrench.torque.z = tz
            self.pub_legacy.publish(msg)

        # Publish each sensor on its own topic
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


def main() -> None:
    rclpy.init()
    node = ForceSensorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
