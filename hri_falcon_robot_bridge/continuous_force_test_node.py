#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import WrenchStamped, Vector3Stamped
import time
import math

class ContinuousForceTestNode(Node):
    def __init__(self):
        super().__init__('continuous_force_test_node')
        
        # Publishers
        self.force_pub = self.create_publisher(WrenchStamped, '/force_sensor/wrench', 10)
        
        # Subscribers - falcon position feedback으로 실제 움직임 확인
        self.falcon_position_sub = self.create_subscription(
            Vector3Stamped,
            '/falcon/position', 
            self.falcon_position_callback,
            10
        )
        
        # Timer for continuous force input
        self.timer = self.create_timer(0.1, self.continuous_force_callback)  # 10Hz
        self.start_time = time.time()
        self.last_falcon_position = None
        
        self.get_logger().info("Continuous Force Test Node started")
        self.get_logger().info("Sending continuous 3N force to test falcon movement...")

    def continuous_force_callback(self):
        """Continuously send 3N force to test falcon movement"""
        
        current_time_stamp = self.get_clock().now()
        elapsed_time = time.time() - self.start_time
        
        # Create force message with 3N continuous force
        force_msg = WrenchStamped()
        force_msg.header.stamp = current_time_stamp.to_msg()
        force_msg.header.frame_id = "force_sensor"
        
        # Different continuous force patterns to test movement
        if elapsed_time < 5.0:
            # Pattern 1: Constant 3N in X direction
            force_msg.wrench.force.x = 3.0
            force_msg.wrench.force.y = 0.0
            force_msg.wrench.force.z = 0.0
            pattern = "X-axis 3N"
            
        elif elapsed_time < 10.0:
            # Pattern 2: Constant 3N in Y direction  
            force_msg.wrench.force.x = 0.0
            force_msg.wrench.force.y = 3.0
            force_msg.wrench.force.z = 0.0
            pattern = "Y-axis 3N"
            
        elif elapsed_time < 15.0:
            # Pattern 3: Constant 3N in Z direction
            force_msg.wrench.force.x = 0.0
            force_msg.wrench.force.y = 0.0
            force_msg.wrench.force.z = 3.0
            pattern = "Z-axis 3N"
            
        elif elapsed_time < 20.0:
            # Pattern 4: Combined 3N forces
            force_msg.wrench.force.x = 2.0
            force_msg.wrench.force.y = 2.0
            force_msg.wrench.force.z = 1.0
            pattern = "Combined [2,2,1]N"
            
        elif elapsed_time < 25.0:
            # Pattern 5: Oscillating forces around 3N
            t = elapsed_time - 20.0
            force_msg.wrench.force.x = 3.0 + 1.0 * math.sin(t * 2)
            force_msg.wrench.force.y = 3.0 + 1.0 * math.cos(t * 2)
            force_msg.wrench.force.z = 3.0 + 0.5 * math.sin(t * 4)
            pattern = "Oscillating ~3N"
            
        else:
            # Stop after 25 seconds
            force_msg.wrench.force.x = 0.0
            force_msg.wrench.force.y = 0.0
            force_msg.wrench.force.z = 0.0
            pattern = "Stop"
            
            if elapsed_time > 26.0:
                self.get_logger().info("Test completed! Shutting down...")
                self.timer.cancel()
                return
        
        # Publish force
        self.force_pub.publish(force_msg)
        
        # Debug: Log every force value being sent
        self.get_logger().info(f"SENDING: x={force_msg.wrench.force.x:.1f}, y={force_msg.wrench.force.y:.1f}, z={force_msg.wrench.force.z:.1f}")
        
        # Log every 1 second
        if int(elapsed_time) != int(elapsed_time - 0.1):
            expected_falcon_x = force_msg.wrench.force.x * 33.3
            expected_falcon_y = force_msg.wrench.force.y * 33.3
            expected_falcon_z = force_msg.wrench.force.z * 33.3
            
            self.get_logger().info(f"[{elapsed_time:.1f}s] {pattern}: "
                                 f"Force=[{force_msg.wrench.force.x:.2f}, {force_msg.wrench.force.y:.2f}, {force_msg.wrench.force.z:.2f}]N -> "
                                 f"Expected Falcon=[{expected_falcon_x:.0f}, {expected_falcon_y:.0f}, {expected_falcon_z:.0f}]")

    def falcon_position_callback(self, msg):
        """Monitor falcon position changes to verify movement"""
        current_position = [msg.vector.x, msg.vector.y, msg.vector.z]
        
        if self.last_falcon_position is not None:
            # Calculate movement
            dx = current_position[0] - self.last_falcon_position[0]
            dy = current_position[1] - self.last_falcon_position[1] 
            dz = current_position[2] - self.last_falcon_position[2]
            
            # Log significant movements
            movement_magnitude = (dx**2 + dy**2 + dz**2)**0.5
            if movement_magnitude > 10.0:  # Only log significant movements
                self.get_logger().info(f"Falcon moved: Pos=[{current_position[0]:.0f}, {current_position[1]:.0f}, {current_position[2]:.0f}], "
                                     f"Movement=[{dx:.0f}, {dy:.0f}, {dz:.0f}], Magnitude={movement_magnitude:.1f}")
        
        self.last_falcon_position = current_position

def main(args=None):
    rclpy.init(args=args)
    
    try:
        test_node = ContinuousForceTestNode()
        rclpy.spin(test_node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()
