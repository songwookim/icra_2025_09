#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3Stamped
import numpy as np

class FalconPositionReader(Node):
    def __init__(self):
        super().__init__('falcon_position_reader')
        
        # Last known position
        self.last_position = np.array([0.0, 0.0, 0.0])
        self.position_initialized = False
        
        # Subscribe to falcon position
        self.falcon_position_sub = self.create_subscription(
            Vector3Stamped,
            '/falcon/position',
            self.falcon_position_callback,
            10
        )
        
        self.get_logger().info("Falcon Position Reader started!")
        self.get_logger().info("Move the falcon device to see position changes...")

    def falcon_position_callback(self, msg):
        """Handle incoming falcon position data"""
        try:
            # Extract position values
            pos_x = msg.vector.x
            pos_y = msg.vector.y  
            pos_z = msg.vector.z
            
            current_position = np.array([pos_x, pos_y, pos_z])
            
            if not self.position_initialized:
                self.last_position = current_position
                self.position_initialized = True
                self.get_logger().info(f"FALCON INITIAL POSITION: x={pos_x:.1f}, y={pos_y:.1f}, z={pos_z:.1f}")
                return
            
            # Calculate position change
            position_delta = current_position - self.last_position
            movement_magnitude = np.linalg.norm(position_delta)
            
            # Only log if there's significant movement (> 5 units)
            if movement_magnitude > 5.0:
                self.get_logger().info(f"FALCON MOVEMENT: x={pos_x:.1f}, y={pos_y:.1f}, z={pos_z:.1f} | Delta: dx={position_delta[0]:.1f}, dy={position_delta[1]:.1f}, dz={position_delta[2]:.1f} | Magnitude: {movement_magnitude:.1f}")
                self.last_position = current_position
                
        except Exception as e:
            self.get_logger().error(f"Error processing falcon position: {e}")

def main(args=None):
    rclpy.init(args=args)
    
    falcon_reader = FalconPositionReader()
    
    try:
        rclpy.spin(falcon_reader)
    except KeyboardInterrupt:
        pass
    finally:
        falcon_reader.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
