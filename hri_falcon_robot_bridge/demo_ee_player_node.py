#!/usr/bin/env python3
"""Demo EE Pose Player Node

CSV 데모 궤적 또는 DMP PKL 파일을 읽어 모든 손가락(th, if, mf)의 /ee_pose_desired_* 토픽으로 재생.

특징:
 - 모든 손가락에 대해 동시 퍼블리시 (th, if, mf)
 - DMP PKL 파일 우선 로딩 (연장된 목표점 자동 반영)
 - CSV 백업 지원 (glob 패턴)
 - 다중 CSV인 경우 선형 시간 정규화 후 평균 궤적 생성 (DTW 미적용)
 - 재생 속도(scale) 조절, 반복(loop) 옵션
 - 컬럼 자동 탐색: ee_{finger}_px 형태

필수 컬럼 (CSV):
 ee_th_px, ee_th_py, ee_th_pz
 ee_if_px, ee_if_py, ee_if_pz
 ee_mf_px, ee_mf_py, ee_mf_pz

사용 예:
 # DMP PKL 사용 (권장 - 연장된 목표점 반영)
 ros2 run hri_falcon_robot_bridge demo_ee_player_node --ros-args \
   -p pkl_pattern:="dmp_models/dmp_*_multi_10demos.pkl" -p loop:=true
   
 # CSV 사용 (백업)
 ros2 run hri_falcon_robot_bridge demo_ee_player_node --ros-args \
   -p csv_pattern:="outputs/demo_*.csv" -p loop:=true
"""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, TwistStamped
from std_msgs.msg import Bool, UInt8

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from glob import glob
from scipy.interpolate import interp1d
import sys
import select
import termios
import tty

class DemoEEPlayerNode(Node):
    def __init__(self):
        super().__init__('demo_ee_player_node')
        # Parameters
        self.declare_parameter('pkl_pattern', '')  # DMP PKL 파일 패턴 (우선순위)
        self.declare_parameter('csv', '')
        self.declare_parameter('csv_pattern', '')
        self.declare_parameter('rate_hz', 100.0)
        self.declare_parameter('target_len', 0)  # 0이면 원본 길이 유지(단일), 다중은 자동 설정
        self.declare_parameter('play_speed', 1.0)  # 1.0 = 실시간, 2.0 = 두배 속도
        self.declare_parameter('loop', True)
        self.declare_parameter('tau_scale', 1.0)  # DMP 속도 조절: 0.5 = 2배 빠름, 2.0 = 2배 느림
        self.declare_parameter('hold_time', 0.0)  # 끝 위치 유지 시간 (초)
        self.declare_parameter('speed_profile', 'constant')  # constant, accelerating, decelerating, sigmoid
        self.declare_parameter('accel_factor', 2.0)  # 가속/감속 강도 (1.0=선형, 2.0=제곱, 3.0=세제곱)
        self.declare_parameter('frame_id', 'world')
        self.declare_parameter('exclude_aug', True)
        self.declare_parameter('publish_topics_suffix', '_desired')  # "_desired" → /ee_pose_desired_*
        self.declare_parameter('manual_start', True)  # 키보드 트리거 대기 여부
        self.declare_parameter('start_key', 'p')  # 시작 키
        self.declare_parameter('stop_key', 'k')  # 긴급 정지 키

        p = lambda name: self.get_parameter(name).value
        self.fingers = ['th', 'if', 'mf']
        self.rate_hz = float(p('rate_hz') or 100.0)
        self.play_speed = float(p('play_speed') or 1.0)
        self.loop = bool(p('loop'))
        self.frame_id = str(p('frame_id') or 'world')
        self.exclude_aug = bool(p('exclude_aug'))
        self.align_multi = True  # 항상 시간 정규화 후 평균
        self.publish_suffix = str(p('publish_topics_suffix') or '_desired')
        self.manual_start = bool(p('manual_start') if p('manual_start') is not None else True)
        self.start_key = str(p('start_key') or 'p')
        self.stop_key = str(p('stop_key') or 'k')
        self.tau_scale = float(p('tau_scale') or 1.0)  # DMP 속도 조절
        self.hold_time = float(p('hold_time') or 0.0)  # 끝 유지 시간
        self.speed_profile = str(p('speed_profile') or 'constant')  # 속도 프로파일
        self.accel_factor = float(p('accel_factor') or 2.0)  # 가속 강도
        
        self.is_running = not self.manual_start  # manual_start=False면 즉시 시작
        self.initial_pos_sent = not self.manual_start  # 초기 위치 전송 완료 여부
        self.prev_pos = {}  # velocity 계산용 이전 위치

        pkl_pattern = str(p('pkl_pattern'))
        csv = str(p('csv'))
        csv_pattern = str(p('csv_pattern'))
        self.target_len = int(p('target_len') or 0)

        # Publishers (모든 손가락 - position + velocity)
        self.pubs = {}
        self.vel_pubs = {}
        for finger in self.fingers:
            # Position
            pos_topic = f"/ee_pose{self.publish_suffix}_{finger}"
            self.pubs[finger] = self.create_publisher(PoseStamped, pos_topic, 10)
            self.get_logger().info(f"[INIT] Publish topic={pos_topic}")
            # Velocity
            vel_topic = f"/ee_velocity{self.publish_suffix}_{finger}"
            self.vel_pubs[finger] = self.create_publisher(TwistStamped, vel_topic, 10)
            self.get_logger().info(f"[INIT] Publish velocity topic={vel_topic}")
        
        # Playback status + stage publishers
        self.playback_status_pub = self.create_publisher(Bool, "/demo_playback_active", 10)
        self.get_logger().info("[INIT] Publish playback status -> /demo_playback_active")
        self.playback_stage_pub = self.create_publisher(UInt8, "/demo_playback_stage", 10)
        self._playback_stage = -1
        init_stage = 0 if self.manual_start else 2
        self._set_playback_stage(init_stage)

        # Load trajectories (모든 손가락)
        # 우선순위: PKL > CSV
        self.trajs = {}
        self.N = 0
        
        if pkl_pattern:
            self.get_logger().info(f"[PKL MODE] Loading DMP models from: {pkl_pattern}")
            for finger in self.fingers:
                traj = self._load_dmp_trajectory(pkl_pattern, finger)
                self.trajs[finger] = traj
                if len(traj) > 0:
                    if self.N == 0:
                        self.N = len(traj)
                    elif self.N != len(traj):
                        self.get_logger().warn(f"Trajectory length mismatch for {finger}: {len(traj)} vs {self.N}")
                        self.N = min(self.N, len(traj))
                self.get_logger().info(f"  Loaded DMP trajectory for {finger}: len={len(traj)}")
        else:
            self.get_logger().info("[CSV MODE] Loading trajectories from CSV files")
            for finger in self.fingers:
                traj = self._load_trajectory(csv, csv_pattern, finger)
                self.trajs[finger] = traj
                if len(traj) > 0:
                    if self.N == 0:
                        self.N = len(traj)
                    elif self.N != len(traj):
                        self.get_logger().warn(f"Trajectory length mismatch for {finger}: {len(traj)} vs {self.N}")
                        self.N = min(self.N, len(traj))
                self.get_logger().info(f"Loaded trajectory for {finger}: len={len(traj)}")
        
        if self.N == 0:
            self.get_logger().error("No trajectory data loaded. Node idle.")
            self.idx = -1
        else:
            self.idx = 0
            self.get_logger().info(f"Loaded trajectories len={self.N} for all fingers")
            # Initialize prev_pos for velocity calculation
            for finger in self.fingers:
                if len(self.trajs[finger]) > 0:
                    self.prev_pos[finger] = self.trajs[finger][0].copy()

        # Keyboard setup for manual start
        if self.manual_start:
            self.orig_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
            self.get_logger().info(f"[MANUAL START] Press '{self.start_key}' to begin playback")
        else:
            self.orig_settings = None
            self.get_logger().info("[AUTO START] Playback started immediately")

        # Timer
        self.timer = self.create_timer(1.0 / self.rate_hz, self._on_timer)
        # Keyboard check timer (10Hz)
        if self.manual_start:
            self.key_timer = self.create_timer(0.1, self._check_keyboard)

    def _load_dmp_trajectory(self, pkl_pattern: str, finger: str) -> np.ndarray:
        """Load DMP model from PKL file and generate trajectory
        
        Args:
            pkl_pattern: Glob pattern for PKL files (e.g., "dmp_models/dmp_*_multi_10demos.pkl")
            finger: Finger identifier (th, if, mf)
        
        Returns:
            np.ndarray: Generated trajectory from DMP rollout (N, 3)
        """
        # Find matching PKL file for this finger
        pattern = pkl_pattern.replace('*', finger).replace('{finger}', finger)
        if '*' in pattern:
            # Pattern still has wildcards - search for files
            matching_files = sorted(glob(pattern))
            # Filter for exact finger match
            pkl_files = [f for f in matching_files if f"dmp_{finger}_" in Path(f).name]
        else:
            # Exact path provided
            pkl_files = [pattern] if Path(pattern).exists() else []
        
        if len(pkl_files) == 0:
            self.get_logger().warn(f"No PKL file found for {finger} with pattern: {pkl_pattern}")
            return np.zeros((0, 3))
        
        # Use the first matching file (or most recent if multiple)
        pkl_path = pkl_files[-1]  # Last in sorted list
        self.get_logger().info(f"Loading DMP for {finger}: {Path(pkl_path).name}")
        
        try:
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            
            # Extract DMP parameters
            y0 = data['y0']
            goal = data['goal']  # This is the EXTENDED goal (연장된 목표점)
            w = data['w']
            dt = data.get('dt', 0.02)
            tau = data.get('tau', 30.0) * self.tau_scale  # Apply speed scaling
            n_bfs = data.get('n_bfs', 50)
            alpha_y = data.get('alpha_y', 25.0)
            beta_y = data.get('beta_y', 6.25)
            a_x = 1.0
            
            # DMP rollout (same logic as DiscreteDMP.rollout)
            n_steps = int(tau / dt)
            y = y0.copy()
            dy = np.zeros_like(y)
            path = []
            x = 1.0
            
            def gaussian_basis(x_val, n_bfs, a_x):
                centers = np.exp(-a_x * np.linspace(0, 1, n_bfs))
                widths = (np.diff(centers)[0] if n_bfs > 1 else 1.0) ** 2
                h = np.exp(-((x_val - centers) ** 2) / (2 * widths))
                return h
            
            for step_idx in range(n_steps):
                path.append(y.copy())
                
                # 진행도 계산 (0.0 ~ 1.0)
                progress = step_idx / max(n_steps - 1, 1)
                
                # 속도 프로파일에 따른 동적 tau 조절
                if self.speed_profile == 'accelerating':
                    # 초반 느림(tau 큼) → 후반 빠름(tau 작음)
                    tau_dynamic = tau * (1.0 + (1.0 - progress**self.accel_factor) * 2.0)
                elif self.speed_profile == 'decelerating':
                    # 초반 빠름(tau 작음) → 후반 느림(tau 큼)
                    tau_dynamic = tau * (1.0 + progress**self.accel_factor * 2.0)
                elif self.speed_profile == 'sigmoid':
                    # S자 곡선: 중간에 가장 빠름
                    sigmoid = 1.0 / (1.0 + np.exp(-10 * (progress - 0.5)))
                    tau_dynamic = tau * (2.0 - sigmoid)
                else:  # 'constant'
                    tau_dynamic = tau
                
                x_next = x - a_x * x * (dt / tau_dynamic)
                psi = gaussian_basis(x, n_bfs, a_x)
                f = np.dot(psi * x, w) / (np.sum(psi) + 1e-10)
                ddy = alpha_y * (beta_y * (goal - y) - dy) + f
                dy += ddy * (dt / tau_dynamic)
                y += dy * (dt / tau_dynamic)
                x = x_next
            
            traj = np.array(path)
            
            # Hold at final position if hold_time > 0
            if self.hold_time > 0.0:
                hold_steps = int(self.hold_time / dt)
                final_pos = traj[-1].copy()
                hold_traj = np.tile(final_pos, (hold_steps, 1))
                traj = np.vstack([traj, hold_traj])
                self.get_logger().info(f"  ✅ DMP rollout complete: len={len(traj)} (motion={n_steps}, hold={hold_steps}), tau_scale={self.tau_scale:.2f}, profile={self.speed_profile}, accel={self.accel_factor:.1f}, goal={goal}")
            else:
                self.get_logger().info(f"  ✅ DMP rollout complete: len={len(traj)}, tau_scale={self.tau_scale:.2f}, profile={self.speed_profile}, accel={self.accel_factor:.1f}, goal={goal}")
            
            return traj
            
        except Exception as e:
            self.get_logger().error(f"Failed to load DMP for {finger}: {e}")
            return np.zeros((0, 3))

    def _load_trajectory(self, csv: str, csv_pattern: str, finger: str) -> np.ndarray:
        files = []
        if csv_pattern:
            all_files = sorted(glob(csv_pattern))
            for f in all_files:
                if self.exclude_aug and 'aug' in Path(f).name:
                    continue
                files.append(f)
            self.get_logger().info(f"Pattern matched {len(all_files)} files → using {len(files)} (exclude_aug={self.exclude_aug})")
        elif csv:
            if Path(csv).exists():
                files.append(csv)
            else:
                self.get_logger().warn(f"Single CSV not found: {csv}")
        else:
            self.get_logger().warn("Neither csv nor csv_pattern provided")

        if len(files) == 0:
            return np.zeros((0, 3))

        # Load per file positions
        trajs = []
        for fp in files:
            try:
                df = pd.read_csv(fp)
            except Exception as e:
                self.get_logger().warn(f"Read fail {fp}: {e}")
                continue
            pos_cols = [f'ee_{finger}_px', f'ee_{finger}_py', f'ee_{finger}_pz']
            if not all(c in df.columns for c in pos_cols):
                self.get_logger().warn(f"Missing position columns for {finger} in {fp}. Skip.")
                continue
            arr = df[pos_cols].to_numpy(dtype=float)
            # Filter NaN rows
            mask = np.isfinite(arr).all(axis=1)
            arr = arr[mask]
            if len(arr) < 5:
                self.get_logger().warn(f"Trajectory too short after filtering in {fp}. Skip.")
                continue
            trajs.append(arr)
            self.get_logger().info(f"Loaded {Path(fp).name} len={len(arr)}")

        if len(trajs) == 0:
            return np.zeros((0, 3))

        if len(trajs) == 1:
            return trajs[0]

        # Multi-demo mode: alignment & mean
        if self.align_multi:
            target_len = self.target_len if self.target_len > 0 else max(len(t) for t in trajs)
            aligned = []
            for t in trajs:
                x_old = np.linspace(0, 1, len(t))
                x_new = np.linspace(0, 1, target_len)
                f = interp1d(x_old, t, axis=0, kind='linear')
                aligned.append(f(x_new))
            mean_traj = np.mean(np.stack(aligned, axis=0), axis=0)
            self.get_logger().info(f"Multi-demo alignment→mean: demos={len(trajs)} target_len={target_len}")
            return mean_traj
        else:
            # Concatenate sequentially (no alignment)
            concat = np.concatenate(trajs, axis=0)
            self.get_logger().info(f"Multi-demo concat: total_len={len(concat)}")
            return concat

    def _publish_initial_positions(self):
        """Publish initial (first) positions for all fingers once"""
        for finger in self.fingers:
            if len(self.trajs[finger]) == 0:
                continue
            pos = self.trajs[finger][0]  # First position
            
            # Position message
            msg = PoseStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = self.frame_id
            msg.pose.position.x = float(pos[0])
            msg.pose.position.y = float(pos[1])
            msg.pose.position.z = float(pos[2])
            self.pubs[finger].publish(msg)
            
            # Velocity = 0 (stationary at start)
            vel_msg = TwistStamped()
            vel_msg.header.stamp = msg.header.stamp
            vel_msg.header.frame_id = self.frame_id
            vel_msg.twist.linear.x = 0.0
            vel_msg.twist.linear.y = 0.0
            vel_msg.twist.linear.z = 0.0
            self.vel_pubs[finger].publish(vel_msg)
        
        self.get_logger().info("[INIT_POS] Published initial positions for all fingers")

    def _set_playback_stage(self, stage: int) -> None:
        """Publish discrete playback stage for downstream controllers."""
        stage_val = max(0, int(stage))
        if self._playback_stage == stage_val:
            return
        self._playback_stage = stage_val
        msg = UInt8()
        msg.data = stage_val
        self.playback_stage_pub.publish(msg)
        labels = {0: "position", 1: "force-ready", 2: "playback"}
        label = labels.get(stage_val, "custom")
        self.get_logger().info(f"[STAGE] demo_playback_stage -> {stage_val} ({label})")

    def _check_keyboard(self):
        """Check for keyboard input to start/restart playback
        
        Two-step activation:
        1st 'p': Publish initial pose and hold (zero torque - playback_active=False)
        2nd 'p': Start actual playback trajectory (torque active - playback_active=True)
        """
        if select.select([sys.stdin], [], [], 0)[0]:
            key = sys.stdin.read(1)
            if key.lower() == self.stop_key.lower():
                self._handle_emergency_stop()
                return
            if key.lower() == self.start_key.lower():
                if not self.initial_pos_sent:
                    # STEP 1: First 'p' press - send initial position only
                    self.idx = 0
                    for finger in self.fingers:
                        if len(self.trajs[finger]) > 0:
                            self.prev_pos[finger] = self.trajs[finger][0].copy()
                    
                    self._publish_initial_positions()
                    self.initial_pos_sent = True
                    self.get_logger().info(f"[STEP 1] Initial position sent (ZERO TORQUE). Press '{self.start_key}' again to start playback")
                    # Status stays False - no torque output
                    self.playback_status_pub.publish(Bool(data=False))
                    self._set_playback_stage(1)
                    
                elif not self.is_running:
                    # STEP 2: Second 'p' press - start actual playback
                    self.is_running = True
                    self.get_logger().info(f"[STEP 2] Playback started (key='{key}')")
                    # Now activate playback and torque
                    self.playback_status_pub.publish(Bool(data=True))
                    self._set_playback_stage(2)
                    
                elif self.idx < 0:
                    # Restart after completion: reset to STEP 1
                    self.is_running = False
                    self.initial_pos_sent = False
                    self.idx = 0
                    for finger in self.fingers:
                        if len(self.trajs[finger]) > 0:
                            self.prev_pos[finger] = self.trajs[finger][0].copy()
                    
                    self._publish_initial_positions()
                    self.initial_pos_sent = True
                    self.get_logger().info(f"[RESTART-STEP1] Initial position sent (ZERO TORQUE). Press '{self.start_key}' to start")
                    self.playback_status_pub.publish(Bool(data=False))
                    self._set_playback_stage(1)

    def _on_timer(self):
        if self.idx < 0 or self.N == 0:
            return
        
        # Publish playback status (only True when actually running)
        status_msg = Bool()
        status_msg.data = self.is_running
        self.playback_status_pub.publish(status_msg)
        
        # If not running yet (waiting for 'p' key), do NOT publish any position
        # This prevents torque controller from receiving desired positions before playback starts
        if not self.is_running:
            return
        
        # Normal playback after 'p' key pressed
        # Playback step advancement considering play_speed
        step_advance = max(1, int(round(self.play_speed)))
        dt = step_advance / self.rate_hz

        # Publish current pose + velocity for all fingers
        for finger in self.fingers:
            if len(self.trajs[finger]) == 0:
                continue
            pos = self.trajs[finger][self.idx]
            
            # Position message
            msg = PoseStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = self.frame_id
            msg.pose.position.x = float(pos[0])
            msg.pose.position.y = float(pos[1])
            msg.pose.position.z = float(pos[2])
            self.pubs[finger].publish(msg)
            
            # Velocity message (feedforward)
            vel = (pos - self.prev_pos[finger]) / dt
            vel_msg = TwistStamped()
            vel_msg.header.stamp = msg.header.stamp
            vel_msg.header.frame_id = self.frame_id
            vel_msg.twist.linear.x = float(vel[0])
            vel_msg.twist.linear.y = float(vel[1])
            vel_msg.twist.linear.z = float(vel[2])
            self.vel_pubs[finger].publish(vel_msg)
            
            # Update prev_pos for next iteration
            self.prev_pos[finger] = pos.copy()

        self.idx += step_advance
        if self.idx >= self.N:
            if self.loop:
                self.idx = 0
                # Reset prev_pos on loop restart
                for finger in self.fingers:
                    if len(self.trajs[finger]) > 0:
                        self.prev_pos[finger] = self.trajs[finger][0].copy()
            else:
                self.get_logger().info(f"Playback finished (loop disabled, press '{self.start_key}' to replay)")
                self.idx = -1  # Mark as finished, can be restarted with 'p' key
                self.is_running = False
                self.initial_pos_sent = False
                # CRITICAL: Stop playback and reset stage when finished
                self.playback_status_pub.publish(Bool(data=False))
                self._set_playback_stage(0)

    def _handle_emergency_stop(self) -> None:
        """Immediate torque cut: publish inactive status and reset stage."""
        self.is_running = False
        self.initial_pos_sent = False
        if self.N > 0:
            self.idx = 0
        self.playback_status_pub.publish(Bool(data=False))
        self._set_playback_stage(0)
        self.get_logger().warn(
            f"[E-STOP] '{self.stop_key}' pressed -> playback halted and torque zeroed"
        )


    def destroy_node(self):
        """Restore terminal settings on shutdown"""
        if self.orig_settings is not None:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.orig_settings)
            except Exception:
                pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = DemoEEPlayerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
