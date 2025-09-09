#!/usr/bin/env python3
"""
RealSense 컬러 스트림 + HSV 마스킹 기반 공 추적 (독립 ROS2 노드)
- HSV 마스크 → 컨투어 최대값 → 원 근사 및 중심/반지름 계산
- GUI 윈도우로 시각화, 's'로 CSV 로깅 토글, 'q' 종료
- 다른 노드(hand_tracker)와 동시에 사용하도록 별도 파이프라인/장치 선택 지원

ROS2 파라미터
- device_index (int, 기본 -1): 연결된 카메라 인덱스 선택(0,1,...)  -1이면 자동
- serial_number (str): 특정 시리얼 카메라 선택(우선순위 높음)
- width/height/fps (int): 컬러 스트림 해상도/프레임레이트
- enable_cv_window (bool): GUI 창 사용 여부 (기본 True, DISPLAY 있으면 자동)
"""
from __future__ import annotations

import os
import time
import csv
from typing import Optional, Tuple

import numpy as np
import cv2
import rclpy
from rclpy.node import Node

try:
    import pyrealsense2 as rs  # type: ignore
except Exception as e:  # pragma: no cover
    rs = None  # type: ignore
    raise

try:
    import imutils  # type: ignore
    HAS_IMUTILS = True
except Exception:
    imutils = None  # type: ignore
    HAS_IMUTILS = False


# GUI 환경 설정: 디스플레이가 없으면 offscreen, 있으면 GUI 허용
_ENABLE_CV_ENV = os.environ.get("ENABLE_CV_WINDOW", "").lower() in ("1","true","yes","on")
_HAS_DISPLAY = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
_WANT_GUI = _ENABLE_CV_ENV or _HAS_DISPLAY
if not _WANT_GUI:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
else:
    if os.environ.get("QT_QPA_PLATFORM", "").lower() == "offscreen":
        try:
            del os.environ["QT_QPA_PLATFORM"]
        except Exception:
            pass


# HSV 색범위 프리셋
COLOR_RANGES = {
    'yellow': [(np.array([20, 100, 100], np.uint8), np.array([35, 255, 255], np.uint8))],
    'blue':   [(np.array([90, 100, 60],  np.uint8), np.array([130, 255, 255], np.uint8))],
    'red':    [
        (np.array([0, 120, 70],  np.uint8), np.array([10, 255, 255], np.uint8)),
        (np.array([170, 120, 70], np.uint8), np.array([179, 255, 255], np.uint8)),
    ],
}
ACTIVE_COLORS = {  # on/off 제어: {'yellow'}, {'yellow','blue'}, {'yellow','blue','red'} 등
    'red'
}

class BallTrackerNode(Node):
    def __init__(self) -> None:
        super().__init__('ball_tracker_node')
        # 파라미터
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 480)
        self.declare_parameter('fps', 30)
        self.declare_parameter('device_index', -1)
        self.declare_parameter('serial_number', '')
        self.declare_parameter('enable_cv_window', True)

        self.width = int(self.get_parameter('width').get_parameter_value().integer_value)
        self.height = int(self.get_parameter('height').get_parameter_value().integer_value)
        self.fps = int(self.get_parameter('fps').get_parameter_value().integer_value)
        self.device_index = int(self.get_parameter('device_index').get_parameter_value().integer_value)
        self.serial_number = str(self.get_parameter('serial_number').get_parameter_value().string_value)
        self.enable_cv_window = bool(self.get_parameter('enable_cv_window').get_parameter_value().bool_value)
        if _WANT_GUI and not self.enable_cv_window:
            self.enable_cv_window = True
            self.get_logger().info("GUI 환경 감지됨: CV 창 자동 활성화")

        # RealSense 파이프라인 초기화
        self.pipe = None
        try:
            self.pipe = self._open_realsense(self.serial_number, self.device_index)
        except Exception as e:
            self.get_logger().error(f"RealSense 초기화 실패: {e}")
            raise

        # CSV 로깅 상태
        self._csv_enabled = False
        self._csv_f = None
        self._csv_writer = None

        # FPS 측정
        self._fps_t = time.time()
        self._fps_c = 0
        self._fps_val = 0.0

        # OpenCV 창
        self._cv_inited = False

        self.get_logger().info("Ball tracker node started")

    def _list_devices(self):
        ctx = rs.context()  # type: ignore[attr-defined]
        return [d for d in ctx.devices]

    def _open_realsense(self, serial: str, index: int):
        pipe = rs.pipeline()  # type: ignore[attr-defined]
        cfg = rs.config()  # type: ignore[attr-defined]
        if serial:
            cfg.enable_device(serial)
            self.get_logger().info(f"Using RealSense device serial={serial}")
        elif index >= 0:
            devs = self._list_devices()
            if len(devs) == 0:
                raise RuntimeError("No RealSense devices found")
            if index >= len(devs):
                raise RuntimeError(f"device_index {index} out of range (found {len(devs)})")
            chosen = devs[index]
            cfg.enable_device(chosen.get_info(rs.camera_info.serial_number))  # type: ignore[attr-defined]
            self.get_logger().info(f"Using RealSense device index={index}")
        else:
            self.get_logger().info("Using default RealSense device")
        cfg.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)  # type: ignore[attr-defined]
        pipe.start(cfg)
        return pipe

    def _track_ball_hsv_multi(self, frame_bgr: np.ndarray):
        blurred = cv2.GaussianBlur(frame_bgr, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        masks_by_color = {}
        for name, ranges in COLOR_RANGES.items():
            acc = None
            for lo, hi in ranges:
                mr = cv2.inRange(hsv, lo, hi)
                acc = mr if acc is None else cv2.bitwise_or(acc, mr)
            if acc is not None:
                masks_by_color[name] = acc
                mask = cv2.bitwise_or(mask, acc)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=2)
        mask = cv2.dilate(mask, kernel, iterations=2)

        contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if HAS_IMUTILS:
            cnts = imutils.grab_contours(contours)  # type: ignore[attr-defined]
        else:
            cnts = contours[0] if len(contours) == 2 else contours[1]

        center = None
        radius = 0.0
        circularity = 0.0
        color_label = None
        if len(cnts) > 0:
            c = max(cnts, key=cv2.contourArea)
            ((x, y), r) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            if M["m00"] > 0:
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            radius = float(r)
            area = float(cv2.contourArea(c))
            peri = float(cv2.arcLength(c, True))
            if peri > 1e-6:
                circularity = float(4.0 * np.pi * area / (peri * peri))
            cmask = np.zeros(mask.shape, dtype=np.uint8)
            cv2.drawContours(cmask, [c], -1, (255, 255, 255), -1)
            best_name, best_overlap = None, -1
            for name, m in masks_by_color.items():
                overlap = int(cv2.countNonZero(cv2.bitwise_and(cmask, m)))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_name = name
            color_label = best_name

        return center, radius, mask, circularity, color_label

    def _update_fps(self):
        self._fps_c += 1
        now = time.time()
        if now - self._fps_t >= 1.0:
            self._fps_val = self._fps_c / (now - self._fps_t)
            self._fps_t = now
            self._fps_c = 0

    def spin(self):
        assert self.pipe is not None
        try:
            while rclpy.ok():
                frames = self.pipe.wait_for_frames()
                color = frames.get_color_frame()
                if not color:
                    continue
                frame = np.asanyarray(color.get_data())
                vis = frame.copy()

                center, radius, mask, circ, color_label = self._track_ball_hsv_multi(frame)

                if center is not None and radius >= 10.0:
                    cv2.circle(vis, (int(center[0]), int(center[1])), int(radius), (0, 255, 0), 2)
                    cv2.circle(vis, center, 5, (0, 0, 255), -1)

                self._update_fps()
                prefix = f"[{color_label}] " if color_label else ""
                info = f"{prefix}radius:{int(radius)} px  circ:{circ:.3f}  FPS:{self._fps_val:.1f}"
                cv2.putText(vis, info, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(vis, info, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

                key = 255
                if self.enable_cv_window:
                    if not self._cv_inited:
                        try:
                            cv2.namedWindow('Ball Tracking (RealSense Color)', cv2.WINDOW_NORMAL)
                            cv2.resizeWindow('Ball Tracking (RealSense Color)', 960, 720)
                            cv2.namedWindow('Mask', cv2.WINDOW_NORMAL)
                            cv2.resizeWindow('Mask', 480, 360)
                            self._cv_inited = True
                        except Exception as e:
                            self.get_logger().warn(f"OpenCV 창 생성 실패: {e}")
                            self.enable_cv_window = False
                    if self.enable_cv_window:
                        cv2.imshow('Ball Tracking (RealSense Color)', vis)
                        cv2.imshow('Mask', mask)
                        key = cv2.waitKey(1) & 0xFF

                # CSV 로깅
                if self._csv_enabled and self._csv_writer is not None:
                    t = time.time()
                    if center is not None and radius >= 10.0:
                        self._csv_writer.writerow({'t': f"{t:.3f}", 'cx': center[0], 'cy': center[1], 'r_px': int(radius), 'circ': f"{circ:.3f}"})
                    else:
                        self._csv_writer.writerow({'t': f"{t:.3f}", 'cx': '', 'cy': '', 'r_px': '', 'circ': ''})

                if key != 255:
                    if key == ord('q'):
                        break
                    elif key == ord('s'):
                        if not self._csv_enabled:
                            try:
                                self._csv_f = open('ball_track_log.csv', 'w', newline='')
                                # type: ignore[arg-type]
                                self._csv_writer = csv.DictWriter(self._csv_f, fieldnames=['t','cx','cy','r_px','circ'])
                                self._csv_writer.writeheader()
                                self._csv_enabled = True
                                self.get_logger().info('[CSV] ball_track_log.csv START')
                            except Exception as e:
                                self.get_logger().warn(f'[CSV] start failed: {e}')
                        else:
                            try:
                                if self._csv_f:
                                    self._csv_f.close()
                                self._csv_f = None
                                self._csv_writer = None
                                self._csv_enabled = False
                                self.get_logger().info('[CSV] STOP & saved')
                            except Exception:
                                pass
        finally:
            try:
                if self.pipe is not None:
                    self.pipe.stop()
            except Exception:
                pass
            try:
                if self.enable_cv_window:
                    cv2.destroyAllWindows()
            except Exception:
                pass


def main(args=None):
    rclpy.init(args=args)
    node = BallTrackerNode()
    try:
        node.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
