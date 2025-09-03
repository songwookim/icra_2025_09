#!/usr/bin/env python3
"""RealSense + MediaPipe hand tracker node (simple ROS2-integrated script).

This script was adapted from user code and wrapped to run inside a ROS2 package.
It opens a RealSense camera, runs MediaPipe Hands, and displays annotated frames.
It also exposes a few ROS2 parameters for resolution and debug.
"""
from __future__ import annotations

import time
from collections import deque, defaultdict
import math
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from typing import Any, cast
from sensor_msgs.msg import JointState

try:
    import mediapipe as mp  # type: ignore
except Exception:
    mp = None  # type: ignore[assignment]
try:
    import pyrealsense2 as rs  # type: ignore
except Exception:
    rs = None  # type: ignore[assignment]

# Make type checker treat these as dynamic modules
mp = cast(Any, mp)
rs = cast(Any, rs)

# Bring in the user's helper logic (trimmed and embedded)
FINGERS = {
  "THUMB":  [ (0,1,2,"CMC"), (1,2,3,"MCP"), (2,3,4,"IP") ],
  "INDEX":  [ (0,5,6,"MCP"), (5,6,7,"PIP"), (6,7,8,"DIP") ],
  "MIDDLE": [ (0,9,10,"MCP"), (9,10,11,"PIP"), (10,11,12,"DIP") ],
}

def angle_3d(a, b, c):
    if a is None or b is None or c is None:
        return None
    ba, bc = a - b, c - b
    nba, nbc = np.linalg.norm(ba), np.linalg.norm(bc)
    if nba == 0 or nbc == 0:
        return None
    cosang = np.clip(np.dot(ba, bc) / (nba * nbc), -1.0, 1.0)
    return math.degrees(math.acos(cosang))

def tip_distance(pts3d, tip_idx):
    if pts3d[0] is None or pts3d[tip_idx] is None: return None
    return float(np.linalg.norm(pts3d[tip_idx] - pts3d[0]))

class HandTrackerNode(Node):
    def __init__(self):
        super().__init__('hand_tracker_node')
        # parameters
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 480)
        self.declare_parameter('fps', 30)
        self.declare_parameter('min_detection_confidence', 0.6)
        self.declare_parameter('min_tracking_confidence', 0.6)
        self.declare_parameter('publish_joint_state', True)
        self.declare_parameter('log_joint_state', False)
        self.declare_parameter('publish_empty_when_no_hand', True)

        self.width = int(self.get_parameter('width').get_parameter_value().integer_value)
        self.height = int(self.get_parameter('height').get_parameter_value().integer_value)
        self.fps = int(self.get_parameter('fps').get_parameter_value().integer_value)
        self.min_detection_confidence = float(self.get_parameter('min_detection_confidence').get_parameter_value().double_value)
        self.min_tracking_confidence = float(self.get_parameter('min_tracking_confidence').get_parameter_value().double_value)
        self.publish_joint_state = bool(self.get_parameter('publish_joint_state').get_parameter_value().bool_value)
        self.log_joint_state = bool(self.get_parameter('log_joint_state').get_parameter_value().bool_value)
        self.publish_empty_when_no_hand = bool(self.get_parameter('publish_empty_when_no_hand').get_parameter_value().bool_value)

        # publishers
        if self.publish_joint_state:
            self.joint_pub = self.create_publisher(JointState, '/hand_tracker/joint_states', 10)

        if mp is None or rs is None:
            self.get_logger().error('Missing optional dependency: mediapipe or pyrealsense2 not installed')
            raise RuntimeError('Missing mediapipe or pyrealsense2')

        # RealSense pipeline
        self.pipe = rs.pipeline()  # type: ignore[attr-defined]
        cfg = rs.config()  # type: ignore[attr-defined]
        cfg.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)  # type: ignore[attr-defined]
        cfg.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)  # type: ignore[attr-defined]
        self.profile = self.pipe.start(cfg)
        self.align = rs.align(rs.stream.color)  # type: ignore[attr-defined]

        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        color_prof = self.profile.get_stream(rs.stream.color).as_video_stream_profile()  # type: ignore[attr-defined]
        self.intr = color_prof.get_intrinsics()

        # MediaPipe hands
        self.mp_hands = mp.solutions.hands  # type: ignore[attr-defined]
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils  # type: ignore[attr-defined]

    def deproject(self, px, py, depth_img):
        if px < 0 or px >= depth_img.shape[1] or py < 0 or py >= depth_img.shape[0]:
            return None
        d = depth_img[py, px] * self.depth_scale
        if d <= 0:
            return None
        X, Y, Z = rs.rs2_deproject_pixel_to_point(self.intr, [float(px), float(py)], float(d))  # type: ignore[attr-defined]
        return np.array([X, Y, Z], dtype=np.float32)

    def run(self):
        try:
            while rclpy.ok():
                frames = self.pipe.wait_for_frames()
                frames = self.align.process(frames)
                depth = frames.get_depth_frame()
                color = frames.get_color_frame()
                if not depth or not color:
                    continue

                depth_img = np.asanyarray(depth.get_data())
                img = np.asanyarray(color.get_data())
                h, w = img.shape[:2]
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                res = self.hands.process(rgb)

                if res.multi_hand_landmarks:
                    lms = res.multi_hand_landmarks[0]
                    pix = [(int(min(max(lm.x,0),1)*w), int(min(max(lm.y,0),1)*h)) for lm in lms.landmark]
                    pts3d = [self.deproject(u, v, depth_img) for (u, v) in pix]

                    y0 = 26
                    for finger, triples in FINGERS.items():
                        parts = []
                        for (i,j,k,name) in triples:
                            ang = angle_3d(pts3d[i], pts3d[j], pts3d[k])
                            if ang is not None:
                                parts.append(f"{name}:{ang:5.1f}°")
                        if parts:
                            cv2.putText(img, f"{finger}: " + " ".join(parts), (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
                            y0 += 20

                    def draw_label_near(idx, text, dy=-8, dx=6, color=(255,255,255)):
                        u,v = pix[idx]
                        cv2.putText(img, text, (u+dx, v+dy), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 2, cv2.LINE_AA)
                        cv2.putText(img, text, (u+dx, v+dy), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

                    for finger, triples in FINGERS.items():
                        for (i,j,k,name) in triples:
                            ang = angle_3d(pts3d[i], pts3d[j], pts3d[k])
                            if ang is not None:
                                draw_label_near(j, f"{name}:{ang:.0f}°")

                    # Publish as JointState (angles in radians) in a stable name order
                    if self.publish_joint_state:
                        names = []
                        positions = []
                        for finger, triples in FINGERS.items():
                            for (i, j, k, jname) in triples:
                                joint_name = f"{finger.lower()}_{jname.lower()}"
                                ang_deg = angle_3d(pts3d[i], pts3d[j], pts3d[k])
                                ang_rad = math.radians(ang_deg) if ang_deg is not None else float('nan')
                                names.append(joint_name)
                                positions.append(ang_rad)

                        js = JointState()
                        js.header.stamp = self.get_clock().now().to_msg()
                        js.name = names
                        js.position = positions
                        if self.log_joint_state:
                            # 주기적인 대량 로그를 피하기 위해 옵션으로만 출력
                            self.get_logger().info(f"publish positions len={len(js.position)}")
                        self.joint_pub.publish(js)

                    VISIBLE_IDX = {0,1,2,3,4,5,6,7,8,9,10,11,12}
                    VISIBLE_CONNS = [(a,b) for (a,b) in mp.solutions.hands.HAND_CONNECTIONS if a in VISIBLE_IDX and b in VISIBLE_IDX]  # type: ignore[attr-defined]
                    for i in VISIBLE_IDX:
                        u,v = pix[i]
                        cv2.circle(img, (u,v), 3, (0,255,0), -1)
                    for (a,b) in VISIBLE_CONNS:
                        ua,va = pix[a]; ub,vb = pix[b]
                        cv2.line(img, (ua,va), (ub,vb), (0,255,0), 1)
                else:
                    # No hand detected; optionally publish NaNs so the topic exists
                    if self.publish_joint_state and self.publish_empty_when_no_hand:
                        names = []
                        positions = []
                        for finger, triples in FINGERS.items():
                            for (_i, _j, _k, jname) in triples:
                                names.append(f"{finger.lower()}_{jname.lower()}")
                                positions.append(float('nan'))
                        js = JointState()
                        js.header.stamp = self.get_clock().now().to_msg()
                        js.name = names
                        js.position = positions
                        self.joint_pub.publish(js)

                cv2.imshow('RealSense + MediaPipe (angles)', img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            try:
                self.pipe.stop()
            except Exception:
                pass
            cv2.destroyAllWindows()


def main(args=None):
    rclpy.init(args=args)
    node = HandTrackerNode()
    node.get_logger().info('Hand tracker node started')
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        # 이미 다른 곳에서 종료되었을 수 있으므로 안전하게 처리
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass

if __name__ == '__main__':
    main()
