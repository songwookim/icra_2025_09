#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import WrenchStamped
from std_msgs.msg import Float64MultiArray
import numpy as np
import time

class HapticRobotController(Node):
    """Minimal logger node: subscribe to sensor_node topics and print.

    - No publishers
    - No Dynamixel control
    - No timers/closed-loop control
    """

    def __init__(self):
        super().__init__('haptic_robot_controller')

        # Subscribers: force sensor topics
        self.force_sub = self.create_subscription(
            WrenchStamped,
            '/force_sensor/wrench',
            self.force_callback,
            10,
        )
        self.force_array_sub = self.create_subscription(
            Float64MultiArray,
            '/force_sensor/wrench_array',
            self.force_array_callback,
            10,
        )

        self.get_logger().info("Haptic Robot Controller (logger-only) initialized")
        self.get_logger().info("Subscribing: /force_sensor/wrench, /force_sensor/wrench_array")

    # Removed: configuration/Dynamixel initialization (not needed for logging)

    # Removed: initial robot positions (not needed for logging)

    def force_callback(self, msg: WrenchStamped):
        """Print force sensor vector from sensor_node."""
        try:
            fx = float(msg.wrench.force.x)
            fy = float(msg.wrench.force.y)
            fz = float(msg.wrench.force.z)
            if not (np.isfinite(fx) and np.isfinite(fy) and np.isfinite(fz)):
                self.get_logger().warn(f"/force_sensor/wrench invalid: [{fx}, {fy}, {fz}]")
                return
            self.get_logger().info(f"/force_sensor/wrench -> F[N]: x={fx:.3f}, y={fy:.3f}, z={fz:.3f}")
        except Exception as e:
            self.get_logger().error(f"Error in force_callback: {e}")

    def force_array_callback(self, msg: Float64MultiArray):
        """Print compact stats for the wrench array (if provided by sensor_node)."""
        try:
            data = msg.data
            n = len(data)
            if n == 0:
                self.get_logger().info("/force_sensor/wrench_array -> empty")
                return
            cols = 6  # fx, fy, fz, tx, ty, tz
            rows = n // cols
            # Print first row succinctly, and counts
            fx, fy, fz = data[0], data[1], data[2]
            self.get_logger().info(
                f"/force_sensor/wrench_array -> rows={rows}, first: fx={fx:.3f}, fy={fy:.3f}, fz={fz:.3f}"
            )
        except Exception as e:
            self.get_logger().error(f"Error in force_array_callback: {e}")

    # Removed: falcon position handling and robot target updates (user will add control later)

    # def set_specific_joint_positions(self, joint_dict):
    #     """Set positions for specific joints using dictionary"""
    #     try:
    #         # Get current positions of all motors
    #         current_positions_list = self.dynamixel_control.get_joint_positions()
    #         all_motor_ids = self.config.dynamixel.ids
            
    #         # Create new position list with updates
    #         new_positions = current_positions_list.copy()
            
    #         for joint_id, new_position in joint_dict.items():
    #             if joint_id in all_motor_ids:
    #                 index = all_motor_ids.index(joint_id)
    #                 new_positions[index] = int(new_position)
            
    #         # Send to all motors
    #         self.dynamixel_control.set_joint_positions(new_positions)
            
    #     except Exception as e:
    #         self.get_logger().error(f"Failed to set joint positions: {e}")

    # Removed: control loop (no publishing / actuation in logger mode)

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
