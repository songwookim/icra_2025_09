#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import WrenchStamped, Vector3Stamped
import time
import math

class HapticTestNode(Node):
    def __init__(self):
        super().__init__('haptic_test_node')
        
        # Publishers
        self.force_pub = self.create_publisher(WrenchStamped, '/force_sensor/wrench', 10)
        self.position_pub = self.create_publisher(Vector3Stamped, '/falcon/position', 10)
        
        # Timer for testing
        self.timer = self.create_timer(2.0 , self.test_callback)
        self.test_count = 0
        
        self.get_logger().info("Haptic Test Node started")

    def test_callback(self):
        """Test both force feedback and position control"""
        
        current_time = self.get_clock().now()
        
        if self.test_count < 3:
            # Test Force Feedback (3N 범위)
            force_msg = WrenchStamped()
            force_msg.header.stamp = current_time.to_msg()
            force_msg.header.frame_id = "force_sensor"
            
            # Different force patterns (3N 정도가 적당)
            if self.test_count == 0:
                force_msg.wrench.force.x = 1.5  # 1.5N
                force_msg.wrench.force.y = 1.0  # 1.0N
                force_msg.wrench.force.z = -0.8  # -0.8N
                self.get_logger().info("Sending force: [1.5, 1.0, -0.8]N -> Expected Falcon: [500, 333, -267]")
            elif self.test_count == 1:
                force_msg.wrench.force.x = -2.0  # -2.0N
                force_msg.wrench.force.y = 2.5   # 2.5N
                force_msg.wrench.force.z = 1.2   # 1.2N
                self.get_logger().info("Sending force: [-2.0, 2.5, 1.2]N -> Expected Falcon: [-667, 833, 400]")
            else:
                force_msg.wrench.force.x = 0.0
                force_msg.wrench.force.y = 0.0
                force_msg.wrench.force.z = 0.0
                self.get_logger().info("Sending force: [0.0, 0.0, 0.0]N (stop)")
            
            self.force_pub.publish(force_msg)
            
        elif self.test_count < 8:
            # Test Position Control (Falcon -1600~1600 범위)
            pos_msg = Vector3Stamped()
            pos_msg.header.stamp = current_time.to_msg()
            pos_msg.header.frame_id = "falcon"
            
            # Falcon position changes within realistic range
            t = (self.test_count - 3) * 0.5
            pos_msg.vector.x = 800.0 * math.sin(t)      # -800 ~ 800 range
            pos_msg.vector.y = 600.0 * math.cos(t)      # -600 ~ 600 range  
            pos_msg.vector.z = 400.0 * math.sin(t*2)    # -400 ~ 400 range
            
            self.position_pub.publish(pos_msg)
            self.get_logger().info(f"Sending falcon position: [{pos_msg.vector.x:.0f}, {pos_msg.vector.y:.0f}, {pos_msg.vector.z:.0f}] (falcon units)")
            
        else:
            # Stop test
            self.get_logger().info("Test completed!")
            self.timer.cancel()
            
        self.test_count += 1

def main(args=None):
    rclpy.init(args=args)
    
    try:
        test_node = HapticTestNode()
        rclpy.spin(test_node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()
