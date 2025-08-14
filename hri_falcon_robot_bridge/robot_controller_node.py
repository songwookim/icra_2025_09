#!/usr/bin/env python3
import os
from typing import List, Optional
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32MultiArray
from geometry_msgs.msg import Vector3Stamped

# For loading config
try:
    from omegaconf import OmegaConf
except Exception:
    OmegaConf = None

# Placeholder imports for the user's controller
try:
    # from hri_falcon_robot_bridge.dynamixel_control import DynamixelControl  # user-provided module
    from dynamixel_control import DynamixelControl  # user-provided module
except Exception as e:
    print(f"Failed to import DynamixelControl: {e}")
    DynamixelControl = None

class RobotControllerNode(Node):
    def __init__(self):
        super().__init__('robot_controller_node')

        # Load config file automatically
        self.config = self.load_config()

        # Parameters with config fallback
        default_ids = list(self.config.dynamixel.ids) if self.config else [12, 22, 32]
        default_mode = int(self.config.dynamixel.control_modes.default_mode) if self.config else 3

        self.declare_parameter('ids', default_ids)
        self.declare_parameter('mode', default_mode)
        self.declare_parameter('scale', [1.0, 1.0, 1.0])  # scale encoders->robot units
        self.declare_parameter('offset', [1000, 1000, 1000])  # base offsets for robot
        self.declare_parameter('clip_min', [0, 0, 0])
        self.declare_parameter('clip_max', [4095, 4095, 4095])
        self.declare_parameter('use_encoders', True)  # if False, use /falcon/position

        ids_val = self.get_parameter('ids').value or default_ids
        self.ids = [int(x) for x in ids_val]
        mode_val = self.get_parameter('mode').value
        self.mode = int(mode_val if mode_val is not None else default_mode)
        scale_val = self.get_parameter('scale').value or [1.0, 1.0, 1.0]
        self.scale = [float(x) for x in scale_val]
        offset_val = self.get_parameter('offset').value or [1000, 1000, 1000]
        self.offset = [float(x) for x in offset_val]
        clip_min_val = self.get_parameter('clip_min').value or [0, 0, 0]
        self.clip_min = [int(x) for x in clip_min_val]
        clip_max_val = self.get_parameter('clip_max').value or [4095, 4095, 4095]
        self.clip_max = [int(x) for x in clip_max_val]
        self.use_encoders = bool(self.get_parameter('use_encoders').value)

        self.controller = None
        if DynamixelControl is not None:
            try:
                # Pass the loaded config to DynamixelControl
                if self.config:
                    self.controller = DynamixelControl(self.config.dynamixel)
                else:
                    # Create a basic config namespace for fallback
                    from types import SimpleNamespace
                    fallback_config = SimpleNamespace(
                        device_name='/dev/ttyUSB0',
                        baudrate=1000000,
                        protocol_version=2.0,
                        ids=self.ids,
                        default_mode=self.mode,
                    )
                    self.controller = DynamixelControl(fallback_config)

                self.controller.connect()
                self.get_logger().info(f"Dynamixel connected. IDs={self.ids}, mode={self.mode}")
            except Exception as e:
                self.get_logger().error(f"Failed to init/connect DynamixelControl: {e}")
        else:
            self.get_logger().warn("DynamixelControl module not found. Running in dry-run mode.")

        if self.use_encoders:
            self.sub_enc = self.create_subscription(Int32MultiArray, '/falcon/encoders', self.on_encoders, 10)
        else:
            self.sub_pos = self.create_subscription(Vector3Stamped, '/falcon/position', self.on_position, 10)

        # State
        self.initialized = False
        self.base_positions = None  # cached base joint positions

    def _ensure_base_positions(self):
        """Fetch current robot joint positions once and cache as base.
        Fallback order: controller API -> offsets param -> zeros.
        """
        if self.base_positions is not None:
            return
        base: Optional[List[float]] = None
        if self.controller is not None:
            # Try common method names on the provided controller
            for method_name in (
                'get_joint_positions',
                'get_current_positions',
                'read_positions',
                'present_positions',
            ):
                try:
                    meth = getattr(self.controller, method_name, None)
                    if callable(meth):
                        res = meth() if method_name != 'read_positions' else meth(self.ids)
                        if isinstance(res, (list, tuple)) and len(res) >= 3:
                            base = [float(res[0]), float(res[1]), float(res[2])]
                            break
                except Exception:
                    pass
        if base is None:
            # Use configured offsets as a reasonable base
            try:
                base = [float(self.offset[0]), float(self.offset[1]), float(self.offset[2])]
            except Exception:
                base = [0.0, 0.0, 0.0]
        self.base_positions = base

    def load_config(self):
        """Load config.yaml from robot_parameter directory"""
        try:
            # Get the package share directory
            import ament_index_python.packages as packages
            package_share_directory = packages.get_package_share_directory('hri_falcon_robot_bridge')
            config_path = os.path.join(package_share_directory, 'robot_parameter', 'config.yaml')
            
            # If not found in install directory, try source directory
            if not os.path.exists(config_path):
                # Try to find in source directory
                current_dir = os.path.dirname(os.path.abspath(__file__))
                package_root = os.path.dirname(os.path.dirname(current_dir))
                config_path = os.path.join(package_root, 'robot_parameter', 'config.yaml')
            
            if OmegaConf is not None and os.path.exists(config_path):
                config = OmegaConf.load(config_path)
                self.get_logger().info(f"Successfully loaded config from: {config_path}")
                return config
            else:
                self.get_logger().warn(f"Config file not found at: {config_path}")
                return None
        except Exception as e:
            self.get_logger().warn(f"Failed to load config: {e}")
            return None

    def map_and_send(self, vec3: List[float]):
        # Map input in [-1600,1600] to delta [-100,100] and add to current base positions
        self._ensure_base_positions()
        base = self.base_positions if self.base_positions is not None else [0.0, 0.0, 0.0]
        cmd: List[int] = []
        for i in range(3):
            x = float(vec3[i])
            # Normalize and scale: [-1600,1600] -> [-100,100]
            delta = (x / 1600.0) * 100.0
            if delta > 100.0:
                delta = 100.0
            elif delta < -100.0:
                delta = -100.0
            target = float(base[i]) - delta
            # Clip to configured limits
            target = max(float(self.clip_min[i]), min(float(self.clip_max[i]), target))
            cmd.append(int(round(target)))
        if self.controller is not None:
            try:
                self.controller.set_joint_positions(cmd)
            except Exception as e:
                self.get_logger().error(f"Failed to set_joint_positions: {e}")
        self.get_logger().info(f"Commanded positions: {cmd} (base={base}, input={vec3})")

    def on_encoders(self, msg: Int32MultiArray):
        data = msg.data
        if len(data) >= 3:
            vec3 = [float(data[0]), float(data[1]), float(data[2])]
            self.map_and_send(vec3)

    def on_position(self, msg: Vector3Stamped):
        vec3 = [msg.vector.x, msg.vector.y, msg.vector.z]
        self.map_and_send(vec3)

def main():
    rclpy.init()
    node = RobotControllerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
