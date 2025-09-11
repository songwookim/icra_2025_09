#!/usr/bin/env python3
import os
import sys
from typing import Optional, List, Tuple, Any
import csv
from datetime import datetime
from pathlib import Path

import hydra
from omegaconf import DictConfig

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

# numpy is a required dependency for this node; already imported above.

try:
    from mms101_controller import MMS101Controller  # original controller
except Exception:
    MMS101Controller = None  # type: ignore
    

try:
    from mms101_controller_temp import MMS101Controller as TempMMS101Controller  # temp EMA controller
except Exception:
    TempMMS101Controller = None  # type: ignore


class ForceSensorNode(Node):
    def __init__(self) -> None:
        super().__init__('force_sensor_node')

        # Load config
        self.config = self._load_config()
        if self.config is None:
            self.get_logger().error("설정 로드 실패 -> 종료")
            rclpy.shutdown()
            sys.exit(1)

        # Publishers
        self.pub_array = self.create_publisher(Float64MultiArray, '/force_sensor/wrench_array', 10)

        # Parameters
        self.declare_parameter('publish_rate_hz', 1000.0)
        self.declare_parameter('use_mock', False)
        self.declare_parameter('config_path', 'config.yaml')  # currently unused
        self.declare_parameter('csv_enable', True)
        self.declare_parameter('csv_dir', '')

        self.rate = self.get_parameter('publish_rate_hz').get_parameter_value().double_value
        self.use_mock = self.get_parameter('use_mock').get_parameter_value().bool_value
        self.num_sensors = 3  # fixed
        self.csv_enable = self.get_parameter('csv_enable').get_parameter_value().bool_value
        self.csv_dir_param = self.get_parameter('csv_dir').get_parameter_value().string_value

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

        # State & CSV
        self.i = 0
        self.last_values_list = [
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0) for _ in range(self.num_sensors)
        ]
        self.csv_file = None
        self.csv_writer = None
        self.csv_path = None

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
        # Initialize CSV once we know we're running
        if self.csv_enable and self.csv_writer is None:
            self._setup_csv()

        # Publish combined array: shape (num_sensors, 6), row-major flatten
        arr_msg = Float64MultiArray()
        d0 = MultiArrayDimension(label='sensor', size=len(values), stride=6)
        d1 = MultiArrayDimension(label='axis', size=6, stride=1)
        arr_msg.layout.dim = [d0, d1]
        arr_msg.data = [float(x) for row in values for x in row]
        self.pub_array.publish(arr_msg)
        self.get_logger().info(f'sensor data : {arr_msg.data}')

        # Append to CSV
        if self.csv_enable and self.csv_writer is not None:
            try:
                t_sec = float(self.get_clock().now().nanoseconds) / 1e9
                flat = [float(x) for row in values for x in row]
                row = [f"{t_sec:.6f}", str(self.i)] + [f"{v:.6f}" for v in flat]
                self.csv_writer.writerow(row)
                if self.csv_file is not None:
                    self.csv_file.flush()
            except Exception as e:
                if (self.i % 200) == 0:
                    self.get_logger().warn(f'CSV write error: {e}')

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
    
    def _resolve_default_csv_dir(self) -> Path:
        # Default to <workspace>/outputs/force/YYYYMMDD
        pkg_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # .../src/hri_falcon_robot_bridge
        # workspace dir is two levels up from package dir
        workspace_dir = (pkg_dir / '..' / '..').resolve()
        date_dir = datetime.now().strftime('%Y%m%d')
        return workspace_dir / 'outputs' / 'force' / date_dir

    def _setup_csv(self) -> None:
        try:
            # Resolve directory
            base_dir = Path(self.csv_dir_param) if self.csv_dir_param else self._resolve_default_csv_dir()
            base_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.csv_path = base_dir / f'{ts}_force.csv'

            # Open and write header
            self.csv_file = open(self.csv_path, 'w', newline='')
            self.csv_writer = csv.writer(self.csv_file)
            headers: List[str] = ['t_sec', 'i']
            axes = ['fx', 'fy', 'fz', 'tx', 'ty', 'tz']
            for s in range(self.num_sensors):
                for a in axes:
                    headers.append(f's{s+1}_{a}')
            self.csv_writer.writerow(headers)
            self.get_logger().info(f'CSV logging -> {self.csv_path}')
        except Exception as e:
            self.get_logger().error(f'CSV init 실패: {e}')
            # Disable csv on failure
            self.csv_enable = False
            

def main() -> None:
    rclpy.init()
    node = ForceSensorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()