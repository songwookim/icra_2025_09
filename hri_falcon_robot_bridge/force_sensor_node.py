#!/usr/bin/env python3
import os
import sys
import threading
from collections import deque
from typing import Optional, List, Tuple, Any, Deque

import hydra
from omegaconf import DictConfig

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import WrenchStamped
from std_msgs.msg import String, Bool
import numpy as np

# Matplotlib with threading support
try:
    import matplotlib
    matplotlib.use('TkAgg')  # Thread-safe backend
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    PLT_AVAILABLE = True
except ImportError:
    plt = None  # type: ignore
    FuncAnimation = None  # type: ignore
    PLT_AVAILABLE = False
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
        self.declare_parameter('enable_live_plot', True)  # Live plot 활성화
        self.declare_parameter('plot_window_sec', 5.0)  # 플롯에 표시할 시간 범위 (초)

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

        # Live plot setup
        self.enable_live_plot = self.get_parameter('enable_live_plot').get_parameter_value().bool_value
        self.plot_window_sec = self.get_parameter('plot_window_sec').get_parameter_value().double_value
        self._plot_buffer_size = int(self.rate * self.plot_window_sec)  # samples to keep
        self._plot_buffers: List[Deque[Tuple[float, ...]]] = [
            deque(maxlen=self._plot_buffer_size) for _ in range(self.num_sensors)
        ]
        self._plot_lock = threading.Lock()
        self._plot_thread: Optional[threading.Thread] = None
        self._plot_running = False
        
        if self.enable_live_plot and PLT_AVAILABLE:
            self._start_live_plot()
        elif self.enable_live_plot and not PLT_AVAILABLE:
            self.get_logger().warn('matplotlib 미설치 -> live plot 비활성화')

        # Timer
        self.timer = self.create_timer(1.0 / self.rate, self.on_timer)
        try:
            self._calib_state_pub.publish(Bool(data=False))
        except Exception:
            pass

    def _on_key(self, msg: String) -> None:
        key = str(msg.data).lower()
        if key == 'f':
            # 컨트롤러에 캘리브레이션 모드 설정
            if self.controller is not None:
                setattr(self.controller, 'calibration_mode', True)
        elif key == 'p':
            # Toggle live plot
            if self._plot_running:
                self._stop_live_plot()
                self.get_logger().info("Live plot 중지")
            elif PLT_AVAILABLE:
                self._start_live_plot()
                self.get_logger().info("Live plot 시작")

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
            
            # Update plot buffers (thread-safe)
            if self._plot_running:
                with self._plot_lock:
                    for idx, v in enumerate(values):
                        if idx < len(self._plot_buffers):
                            self._plot_buffers[idx].append(v)

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

    def _start_live_plot(self) -> None:
        """별도 스레드에서 matplotlib live plot 시작."""
        if self._plot_running:
            return
        self._plot_running = True
        self._plot_thread = threading.Thread(target=self._run_plot_loop, daemon=True)
        self._plot_thread.start()
        self.get_logger().info(f"Live plot 활성화 (window={self.plot_window_sec}s, press 'p' to toggle)")

    def _stop_live_plot(self) -> None:
        """Live plot 중지."""
        self._plot_running = False
        # Thread will exit on next animation frame check

    def _run_plot_loop(self) -> None:
        """Matplotlib animation loop (runs in separate thread)."""
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        fig.suptitle('Force Sensor Live Plot (6-axis)', fontsize=12)
        sensor_names = ['Sensor 1 (TH)', 'Sensor 2 (IF)', 'Sensor 3 (MF)']
        axis_labels = ['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz']
        colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628']
        
        lines = []
        for ax_idx, ax in enumerate(axes):
            ax.set_ylabel(sensor_names[ax_idx])
            ax.set_ylim(-50, 50)  # 초기 범위, 자동 조정됨
            ax.grid(True, alpha=0.3)
            ax.legend(axis_labels, loc='upper right', ncol=6, fontsize=8)
            sensor_lines = []
            for ch_idx in range(6):
                line, = ax.plot([], [], color=colors[ch_idx], linewidth=0.8, label=axis_labels[ch_idx])
                sensor_lines.append(line)
            lines.append(sensor_lines)
            ax.legend(loc='upper right', ncol=6, fontsize=8)
        axes[-1].set_xlabel('Time (samples)')
        
        plt.tight_layout()
        
        def update(frame):
            if not self._plot_running:
                plt.close(fig)
                return []
            
            all_lines = []
            with self._plot_lock:
                for sensor_idx in range(self.num_sensors):
                    buf = self._plot_buffers[sensor_idx]
                    if len(buf) < 2:
                        continue
                    data = np.array(list(buf))  # (N, 6)
                    x = np.arange(len(data))
                    
                    for ch_idx in range(6):
                        lines[sensor_idx][ch_idx].set_data(x, data[:, ch_idx])
                    
                    # Auto-scale Y axis
                    if len(data) > 0:
                        y_min, y_max = data.min(), data.max()
                        margin = max(0.1 * (y_max - y_min), 1.0)
                        axes[sensor_idx].set_ylim(y_min - margin, y_max + margin)
                        axes[sensor_idx].set_xlim(0, len(data))
                    
                    all_lines.extend(lines[sensor_idx])
            return all_lines
        
        ani = FuncAnimation(fig, update, interval=100, blit=False, cache_frame_data=False)  # 10Hz update
        try:
            plt.show()
        except Exception:
            pass
        self._plot_running = False

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
        # Stop live plot gracefully
        node._stop_live_plot()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()