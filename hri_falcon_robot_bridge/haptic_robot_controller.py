#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import WrenchStamped, Vector3Stamped
from sensor_msgs.msg import JointState
import numpy as np
import time
import yaml
from omegaconf import OmegaConf
import os

# from hri_falcon_robot_bridge.dynamixel_control import DynamixelControl
from dynamixel_control import DynamixelControl

class HapticRobotController(Node):
    def __init__(self):
        super().__init__('haptic_robot_controller')
        
        # Load robot configuration
        self.load_config()
        
        # Initialize Dynamixel control
        self.dynamixel_control = DynamixelControl(self.config.dynamixel)
        
        # Connect to Dynamixel
        try:
            self.dynamixel_control.connect()
            self.get_logger().info("Dynamixel connected successfully")
        except Exception as e:
            self.get_logger().error(f"Failed to connect to Dynamixel: {e}")
            raise
        
        # Control parameters
        self.max_position_change = 50  # Maximum position change per update
        self.update_rate = 20  # Hz (적당한 업데이트 속도)
        
        # Force scaling: 3N -> 100 falcon units (약 33.3배)
        self.force_scale = 33.3  # Force sensor(3N) -> Falcon force(100)
        
        # Position scaling: Falcon(-1600~1600) -> Robot(0~4096)
        # Falcon 범위 3200 -> Robot 범위 4096, 비율 = 4096/3200 = 1.28
        # 하지만 작은 움직임을 위해 더 작게 설정
        self.position_scale = 0.5  # Falcon position change -> Robot position change
        
        # State variables
        self.last_falcon_position = np.array([0.0, 0.0, 0.0])
        self.current_joint_positions = None
        self.target_joint_positions = None
        self.last_update_time = time.time()
        
        # Target joint IDs (12, 22, 32)
        self.target_joint_ids = [12, 22, 32]
        
        # Publishers
        self.force_pub = self.create_publisher(
            WrenchStamped, 
            '/falcon/force', 
            10
        )
        
        # Subscribers
        self.force_sub = self.create_subscription(
            WrenchStamped,
            '/force_sensor/wrench',
            self.force_callback,
            10
        )
        
        self.falcon_position_sub = self.create_subscription(
            Vector3Stamped,
            '/falcon/position',
            self.falcon_position_callback,
            10
        )
        
        # Timer for smooth robot control
        self.control_timer = self.create_timer(1.0/self.update_rate, self.control_callback)
        
        # Get initial joint positions
        self.initialize_robot_positions()
        
        self.get_logger().info("Haptic Robot Controller initialized!")
        self.get_logger().info(f"Target joints: {self.target_joint_ids}")
        self.get_logger().info(f"Max position change: {self.max_position_change}")
        self.get_logger().info(f"Update rate: {self.update_rate} Hz")

    def load_config(self):
        """Load robot configuration from YAML file"""
        try:
            # Try share directory (installed package)
            package_share_dir = "/home/songwoo/ros2_ws/icra2025/install/hri_falcon_robot_bridge/share/hri_falcon_robot_bridge"
            config_path = os.path.join(package_share_dir, 'robot_parameter', 'config.yaml')
            
            if not os.path.exists(config_path):
                # Try src directory as fallback
                config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'resource', 'robot_parameter', 'config.yaml')
            
            if os.path.exists(config_path):
                self.config = OmegaConf.load(config_path)
                self.get_logger().info(f"Config loaded from: {config_path}")
            else:
                self.get_logger().error(f"Config file not found at: {config_path}")
                raise FileNotFoundError("config.yaml not found")
                
        except Exception as e:
            self.get_logger().error(f"Failed to load config: {e}")
            raise

    def initialize_robot_positions(self):
        """Get initial positions of target joints"""
        try:
            # Get all motor positions as a list
            all_positions_list = self.dynamixel_control.get_joint_positions()
            all_motor_ids = self.config.dynamixel.ids
            
            # Convert to dictionary mapping ID to position
            all_positions = {}
            for i, motor_id in enumerate(all_motor_ids):
                if i < len(all_positions_list):
                    all_positions[motor_id] = all_positions_list[i]
            
            self.current_joint_positions = {}
            self.target_joint_positions = {}
            
            for joint_id in self.target_joint_ids:
                if joint_id in all_positions:
                    self.current_joint_positions[joint_id] = all_positions[joint_id]
                    self.target_joint_positions[joint_id] = all_positions[joint_id]
                    self.get_logger().info(f"Joint {joint_id} initial position: {all_positions[joint_id]}")
                else:
                    self.get_logger().warning(f"Joint {joint_id} not found in robot")
                    
        except Exception as e:
            self.get_logger().error(f"Failed to initialize robot positions: {e}")

    def force_callback(self, msg):
        """Handle incoming force sensor data and provide haptic feedback"""
        try:
            # Extract force values
            force_x = msg.wrench.force.x
            force_y = msg.wrench.force.y  
            force_z = msg.wrench.force.z
            
            # Debug: Log received values BEFORE any processing
            self.get_logger().info(f"RECEIVED RAW: x={force_x:.2f}, y={force_y:.2f}, z={force_z:.2f}")
            
            # Check for invalid values (inf, nan)
            if not (np.isfinite(force_x) and np.isfinite(force_y) and np.isfinite(force_z)):
                self.get_logger().warn(f"Invalid force values detected: [{force_x}, {force_y}, {force_z}], ignoring...")
                return
            
            # Limit force magnitude to reasonable range (±10N)
            max_force = 10.0
            force_x = max(-max_force, min(max_force, force_x))
            force_y = max(-max_force, min(max_force, force_y))
            force_z = max(-max_force, min(max_force, force_z))
            
            # Scale forces for haptic feedback (3N -> 100 falcon units)
            scaled_force_x = force_x * self.force_scale
            scaled_force_y = force_y * self.force_scale
            scaled_force_z = force_z * self.force_scale
            
            # SAFETY: Limit falcon force output to prevent device damage
            # Maximum safe range: -1000 to 1000 (instead of hardware max 2500)
            max_falcon_force = 1000.0
            scaled_force_x = max(-max_falcon_force, min(max_falcon_force, scaled_force_x))
            scaled_force_y = max(-max_falcon_force, min(max_falcon_force, scaled_force_y))
            scaled_force_z = max(-max_falcon_force, min(max_falcon_force, scaled_force_z))
            
            # Create haptic feedback message
            haptic_msg = WrenchStamped()
            haptic_msg.header.stamp = self.get_clock().now().to_msg()
            haptic_msg.header.frame_id = "falcon"
            haptic_msg.wrench.force.x = scaled_force_x
            haptic_msg.wrench.force.y = scaled_force_y
            haptic_msg.wrench.force.z = scaled_force_z
            
            # Publish haptic feedback to falcon
            self.force_pub.publish(haptic_msg)
            
            # Log significant forces
            force_magnitude = (force_x**2 + force_y**2 + force_z**2)**0.5
            if force_magnitude > 0.1:  # Only log if force is significant
                self.get_logger().info(f"Force input: [{force_x:.2f}, {force_y:.2f}, {force_z:.2f}]N -> Falcon: [{scaled_force_x:.0f}, {scaled_force_y:.0f}, {scaled_force_z:.0f}]")
            
        except Exception as e:
            self.get_logger().error(f"Error in force callback: {e}")

    def falcon_position_callback(self, msg):
        """Handle falcon position changes and update robot target positions"""
        try:
            # Get falcon position
            current_position = np.array([msg.vector.x, msg.vector.y, msg.vector.z])
            
            # Calculate position change
            position_change = current_position - self.last_falcon_position
            
            # Apply position scaling (Falcon range -> Robot range with smooth movement)
            scaled_change = position_change * self.position_scale
            
            # Update target joint positions
            if self.target_joint_positions is not None:
                # Map X, Y, Z to joints 12, 22, 32
                joint_changes = {
                    12: scaled_change[0],  # X axis -> joint 12
                    22: scaled_change[1],  # Y axis -> joint 22  
                    32: scaled_change[2]   # Z axis -> joint 32
                }
                
                for joint_id, change in joint_changes.items():
                    if joint_id in self.target_joint_positions:
                        # Limit maximum change per update
                        limited_change = np.clip(change, -self.max_position_change, self.max_position_change)
                        self.target_joint_positions[joint_id] += limited_change
                        
                        # Ensure joint limits (0-4096 range)
                        self.target_joint_positions[joint_id] = np.clip(self.target_joint_positions[joint_id], 0, 4096)
                        
                        # Log significant movements
                        if abs(limited_change) > 1.0:
                            self.get_logger().info(f"Falcon pos: [{current_position[0]:.0f}, {current_position[1]:.0f}, {current_position[2]:.0f}] -> Joint {joint_id}: {self.target_joint_positions[joint_id]:.1f} (Δ{limited_change:.1f})")
            
            # Update last position
            self.last_falcon_position = current_position.copy()
            
        except Exception as e:
            self.get_logger().error(f"Error in falcon position callback: {e}")

    def set_specific_joint_positions(self, joint_dict):
        """Set positions for specific joints using dictionary"""
        try:
            # Get current positions of all motors
            current_positions_list = self.dynamixel_control.get_joint_positions()
            all_motor_ids = self.config.dynamixel.ids
            
            # Create new position list with updates
            new_positions = current_positions_list.copy()
            
            for joint_id, new_position in joint_dict.items():
                if joint_id in all_motor_ids:
                    index = all_motor_ids.index(joint_id)
                    new_positions[index] = int(new_position)
            
            # Send to all motors
            self.dynamixel_control.set_joint_positions(new_positions)
            
        except Exception as e:
            self.get_logger().error(f"Failed to set joint positions: {e}")

    def control_callback(self):
        """Smooth robot control timer callback"""
        try:
            current_time = time.time()
            dt = current_time - self.last_update_time
            
            if self.current_joint_positions is None or self.target_joint_positions is None:
                return
            
            # Smooth movement towards target positions
            positions_to_set = {}
            movement_occurred = False
            
            for joint_id in self.target_joint_ids:
                if joint_id in self.current_joint_positions and joint_id in self.target_joint_positions:
                    current_pos = self.current_joint_positions[joint_id]
                    target_pos = self.target_joint_positions[joint_id]
                    
                    # Calculate smooth movement (exponential smoothing)
                    error = target_pos - current_pos
                    
                    if abs(error) > 0.5:  # Only move if error is significant
                        # Smooth approach to target (부드러운 움직임)
                        movement_speed = 0.1  # Adjust for smoothness
                        new_pos = current_pos + error * movement_speed
                        
                        positions_to_set[joint_id] = new_pos
                        self.current_joint_positions[joint_id] = new_pos
                        movement_occurred = True
            
            # Send commands to robot if there's movement
            if movement_occurred and positions_to_set:
                self.set_specific_joint_positions(positions_to_set)
                
                # Log movement
                movements = [f"{jid}:{pos:.1f}" for jid, pos in positions_to_set.items()]
                self.get_logger().info(f"Robot moved: {', '.join(movements)}")
            
            self.last_update_time = current_time
            
        except Exception as e:
            self.get_logger().error(f"Error in control callback: {e}")

def main(args=None):
    rclpy.init(args=args)
    
    try:
        controller = HapticRobotController()
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()
