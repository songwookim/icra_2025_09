#!/usr/bin/env python3
"""EE DMP Player Node

학습된 DMP 모델(.pkl)을 로드해 /ee_pose_desired_{th,if,mf} 토픽으로 재생.

특징:
 - 손가락별 독립 DMP 모델 로드 (th, if, mf)
 - Progress 기반 궤적 생성 (tau, dt 조절 가능)
 - Loop/Once 모드, 재생 속도 조절
 - 시작 트리거 서비스 제공

사용 예:
 ros2 run hri_falcon_robot_bridge ee_dmp_player_node --ros-args \
   -p model_dir:="./dmp_models" -p tau:=10.0 -p loop:=true
"""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_srvs.srv import Trigger

import numpy as np
import pickle
from pathlib import Path

class EEDMPPlayerNode(Node):
    def __init__(self):
        super().__init__('ee_dmp_player_node')
        
        # Parameters
        self.declare_parameter('model_dir', './dmp_models')
        self.declare_parameter('model_prefix', 'dmp')  # dmp_{finger}_multi_Ndemos.pkl
        self.declare_parameter('model_pattern', 'multi')  # 'multi' or 'single' or specific filename
        self.declare_parameter('rate_hz', 100.0)
        self.declare_parameter('tau', 10.0)  # Total duration (seconds)
        self.declare_parameter('dt', 0.01)   # Time step (seconds)
        self.declare_parameter('loop', True)
        self.declare_parameter('auto_start', True)
        self.declare_parameter('frame_id', 'world')
        
        p = lambda name: self.get_parameter(name).value
        self.model_dir = Path(str(p('model_dir')))
        self.model_prefix = str(p('model_prefix'))
        self.model_pattern = str(p('model_pattern'))
        self.rate_hz = float(p('rate_hz') or 100.0)
        self.tau = float(p('tau') or 10.0)
        self.dt = float(p('dt') or 0.01)
        self.loop = bool(p('loop'))
        self.auto_start = bool(p('auto_start'))
        self.frame_id = str(p('frame_id') or 'world')
        
        # State
        self.dmps = {}  # finger -> DMP model
        self.trajectories = {}  # finger -> rollout trajectory
        self.playing = self.auto_start
        self.idx = 0
        self.n_steps = 0
        
        # Publishers
        self.pubs = {
            'th': self.create_publisher(PoseStamped, '/ee_pose_desired_th', 10),
            'if': self.create_publisher(PoseStamped, '/ee_pose_desired_if', 10),
            'mf': self.create_publisher(PoseStamped, '/ee_pose_desired_mf', 10),
        }
        
        # Services
        self.srv_start = self.create_service(Trigger, '~/start', self.handle_start)
        self.srv_stop = self.create_service(Trigger, '~/stop', self.handle_stop)
        self.srv_reset = self.create_service(Trigger, '~/reset', self.handle_reset)
        
        # Load models
        self._load_models()
        
        # Timer
        self.timer = self.create_timer(1.0 / self.rate_hz, self._on_timer)
        self.get_logger().info(f"EE DMP Player started (tau={self.tau}s, dt={self.dt}s, loop={self.loop}, auto_start={self.auto_start})")
    
    def _load_models(self):
        """Load DMP models for all fingers"""
        fingers = ['th', 'if', 'mf']
        
        for finger in fingers:
            # Find model file
            pattern = f"{self.model_prefix}_{finger}*{self.model_pattern}*.pkl"
            matches = list(self.model_dir.glob(pattern))
            
            if len(matches) == 0:
                self.get_logger().warn(f"No model found for {finger} (pattern: {pattern})")
                continue
            
            # Use first match (or latest if multiple)
            model_path = sorted(matches)[-1]
            
            try:
                with open(model_path, 'rb') as f:
                    data = pickle.load(f)
                
                # Create DMP instance and restore
                dmp = DiscreteDMP()
                dmp.w = data['w']
                dmp.y0 = data['y0']
                dmp.goal = data['goal']
                dmp.dt = data['dt']
                dmp.tau = data['tau']
                dmp.n_bfs = data['n_bfs']
                dmp.alpha_y = data['alpha_y']
                dmp.beta_y = data['beta_y']
                
                self.dmps[finger] = dmp
                
                # Pre-generate trajectory
                traj = dmp.rollout(dt=self.dt, tau=self.tau)
                self.trajectories[finger] = traj
                
                self.get_logger().info(f"✓ Loaded {finger}: {model_path.name} (len={len(traj)})")
                
            except Exception as e:
                self.get_logger().error(f"✗ Failed to load {finger} from {model_path}: {e}")
                continue
        
        if len(self.dmps) == 0:
            self.get_logger().error(f"No models loaded from {self.model_dir}")
            self.n_steps = 0
        else:
            # Use max length across fingers
            self.n_steps = max(len(t) for t in self.trajectories.values())
            self.get_logger().info(f"Trajectories ready: {list(self.dmps.keys())} (n_steps={self.n_steps})")
    
    def _on_timer(self):
        if not self.playing or self.n_steps == 0:
            return
        
        # Publish current pose for each finger
        now = self.get_clock().now().to_msg()
        
        for finger, traj in self.trajectories.items():
            if self.idx >= len(traj):
                continue
            
            pos = traj[self.idx]
            msg = PoseStamped()
            msg.header.stamp = now
            msg.header.frame_id = self.frame_id
            msg.pose.position.x = float(pos[0])
            msg.pose.position.y = float(pos[1])
            msg.pose.position.z = float(pos[2])
            # Orientation left at zero (can extend later)
            msg.pose.orientation.w = 1.0
            
            self.pubs[finger].publish(msg)
        
        # Advance index
        self.idx += 1
        
        if self.idx >= self.n_steps:
            if self.loop:
                self.idx = 0
                self.get_logger().info("Trajectory loop restart")
            else:
                self.playing = False
                self.get_logger().info("Trajectory playback finished (loop disabled)")
    
    def handle_start(self, request, response):
        """Start/resume playback"""
        self.playing = True
        response.success = True
        response.message = f"Playback started (idx={self.idx})"
        self.get_logger().info(response.message)
        return response
    
    def handle_stop(self, request, response):
        """Pause playback"""
        self.playing = False
        response.success = True
        response.message = f"Playback stopped (idx={self.idx})"
        self.get_logger().info(response.message)
        return response
    
    def handle_reset(self, request, response):
        """Reset to beginning"""
        self.idx = 0
        response.success = True
        response.message = "Playback reset to start"
        self.get_logger().info(response.message)
        return response


class DiscreteDMP:
    """Minimal DMP class for rollout only (no training)"""
    def __init__(self):
        self.n_bfs = 50
        self.alpha_y = 25.0
        self.beta_y = 6.25
        self.w = None
        self.y0 = None
        self.goal = None
        self.dt = 0.02
        self.tau = 1.0
        self.a_x = 1.0
    
    def _gaussian_basis(self, x):
        centers = np.exp(-self.a_x * np.linspace(0, 1, self.n_bfs))
        widths = (np.diff(centers)[0] if self.n_bfs > 1 else 1.0) ** 2
        h = np.exp(-((x[:, None] - centers[None, :]) ** 2) / (2 * widths))
        return h
    
    def rollout(self, dt=None, tau=None):
        """Generate trajectory from learned weights"""
        if self.w is None:
            raise RuntimeError("DMP not trained (w is None)")
        
        if dt is None:
            dt = self.dt
        if tau is None:
            tau = self.tau
        
        n_steps = int(tau / dt)
        y = self.y0.copy()
        dy = np.zeros_like(y)
        path = []
        x = 1.0
        
        for _ in range(n_steps):
            path.append(y.copy())
            x_next = x - self.a_x * x * (dt / tau)
            psi = self._gaussian_basis(np.array([x]))[0]
            f = np.dot(psi * x, self.w) / (np.sum(psi) + 1e-10)
            ddy = self.alpha_y * (self.beta_y * (self.goal - y) - dy) + f
            dy += ddy * (dt / tau)
            y += dy * (dt / tau)
            x = x_next
        
        return np.array(path)


def main(args=None):
    rclpy.init(args=args)
    node = EEDMPPlayerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
