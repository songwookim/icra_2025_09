#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import WrenchStamped
import time
import math

class ForceTestNode(Node):
    def __init__(self):
        super().__init__('force_test_node')
        self.publisher = self.create_publisher(WrenchStamped, '/force_sensor/wrench', 10)
        self.timer = self.create_timer(0.1, self.timer_callback)  # 10 Hz
        self.counter = 0
        
    def timer_callback(self):
        msg = WrenchStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'force_sensor'
        
        # Generate different force patterns for testing
        t = self.counter * 0.1  # time in seconds
        
        if self.counter < 50:  # First 5 seconds: small constant force
            msg.wrench.force.x = 0.5
            msg.wrench.force.y = 0.3
            msg.wrench.force.z = -0.2
            pattern = "Constant small force"
        elif self.counter < 100:  # Next 5 seconds: sine wave
            msg.wrench.force.x = 1.0 * math.sin(t)
            msg.wrench.force.y = 0.5 * math.cos(t)
            msg.wrench.force.z = 0.3 * math.sin(2*t)
            pattern = "Sine wave pattern"
        elif self.counter < 150:  # Next 5 seconds: step function
            msg.wrench.force.x = 1.5 if (self.counter // 10) % 2 == 0 else -1.5
            msg.wrench.force.y = 1.0 if (self.counter // 15) % 2 == 0 else -1.0
            msg.wrench.force.z = 0.5 if (self.counter // 20) % 2 == 0 else -0.5
            pattern = "Step function"
        else:  # Reset
            self.counter = 0
            return
            
        # Always zero torque for this test
        msg.wrench.torque.x = 0.0
        msg.wrench.torque.y = 0.0
        msg.wrench.torque.z = 0.0
        
        self.publisher.publish(msg)
        
        if self.counter % 20 == 0:  # Log every 2 seconds
            self.get_logger().info(f'{pattern}: Force=[{msg.wrench.force.x:.2f}, {msg.wrench.force.y:.2f}, {msg.wrench.force.z:.2f}]')
        
        self.counter += 1

def main(args=None):
    rclpy.init(args=args)
    node = ForceTestNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
