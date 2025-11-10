#!/usr/bin/env python3
"""
Multi-topic synchronized CSV logger node.

- Collects at a fixed rate and writes one row per tick with a common ROS time (when logging is active):
- Force sensors: `/force_sensor/s{1..3}/wrench` (geometry_msgs/WrenchStamped)
- EMG: `/emg/raw` (std_msgs/Float32MultiArray)
- Deformity metrics: `/deformity_tracker/circularity` & `/deformity_tracker/eccentricity` (std_msgs/Float32)
- End-effector pose: geometry_msgs/PoseStamped (ee_pose_topic_if) 또는 nav_msgs/Odometry (ee_pose_odometry_topic) 구독 지원 (optional)

Control:
- Subscribes to `/hand_tracker/key` (std_msgs/String). When message data == 's', toggles logging:
    - OFF -> ON: starts logging and opens a new CSV file
    - ON -> OFF: stops logging and closes the current CSV file
- Parameter `start_immediately` (bool, default False) controls whether logging starts active.
- When inactive, timer still runs but no CSV rows are appended until first activation.

Parameters:
- `rate_hz` (float, default 100.0): logging frequency
- `csv_dir` (string, default <workspace>/outputs/logs/YYYYMMDD)
- `ee_pose_topic_if` (string, optional): geometry_msgs/PoseStamped topic
- `ee_pose_odometry_topic` (string, optional): nav_msgs/Odometry topic (pose.pose를 사용)
- `start_immediately` (bool, default True): begin logging upon node start
 - `emg_log_every_n` (int, default 1): EMG 컬럼을 N번째 행마다 한 번만 기록 (1이면 매 행 기록)

Outputs CSV columns:
    t_sec, t_nanosec,
        s1_fx, ..., s3_tz, s3_stamp_sec, s3_stamp_nsec,
        ee_px, ee_py, ee_pz, ee_stamp_sec, ee_stamp_nsec,
  deform_circ, deform_circ_stamp_sec, deform_circ_stamp_nsec,
  deform_ecc, deform_ecc_stamp_sec, deform_ecc_stamp_nsec,
  emg_ch1..emg_ch8, emg_stamp_sec, emg_stamp_nsec
"""
from __future__ import annotations

import csv
import datetime
import os
from pathlib import Path
from typing import Optional, Tuple, List, Any

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import WrenchStamped, PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32, Float32MultiArray, String, Bool


class DataLoggerNode(Node):
    def __init__(self) -> None:
        super().__init__('data_logger_node')

        # Parameters
        self.declare_parameter('rate_hz', 200.0)
        self.declare_parameter('csv_dir', '')
        self.declare_parameter('ee_pose_topic_if', '/ee_pose_if')
        # 추가: 양손가락 EE 토픽 (MF/TH) 병행 구독 지원
        self.declare_parameter('ee_pose_topic_mf', '/ee_pose_mf')
        self.declare_parameter('ee_pose_topic_th', '/ee_pose_th')
        self.declare_parameter('ee_pose_odometry_topic', '')
        # EMG topic을 파라미터로 노출(기본값은 /emg/raw)
        self.declare_parameter('emg_topic', '/emg/raw')
        # EMG 저장 간격 (N행마다 한 번 기록)
        self.declare_parameter('emg_log_every_n', 1)
        # EMG 기록 모드: 'hold'(마지막 값 유지) | 'blank'(업데이트 없으면 공란) — 기본을 'blank'로 설정
        self.declare_parameter('emg_write_mode', 'hold')
        # EMG 경고 지연시간(초). 초기 연결 지연으로 인한 오경고를 줄이기 위해 기본 2.0초 대기
        self.declare_parameter('emg_warn_after_sec', 2.0)
        # CSV flush 주기(틱). 1이면 매 행 flush, 10이면 10행마다 flush -> I/O 부하 감소로 콜백 스타베이션 방지
        self.declare_parameter('csv_flush_every_n', 10)
        self.declare_parameter('start_immediately', False)

        self.rate_hz = float(self.get_parameter('rate_hz').get_parameter_value().double_value)
        self.csv_dir_param = str(self.get_parameter('csv_dir').get_parameter_value().string_value)
        self.ee_pose_topic_if = str(self.get_parameter('ee_pose_topic_if').get_parameter_value().string_value)
        self.ee_pose_topic_mf = str(self.get_parameter('ee_pose_topic_mf').get_parameter_value().string_value)
        self.ee_pose_topic_th = str(self.get_parameter('ee_pose_topic_th').get_parameter_value().string_value)
        self.ee_pose_odom_topic = str(self.get_parameter('ee_pose_odometry_topic').get_parameter_value().string_value)
        self.start_immediately = bool(self.get_parameter('start_immediately').get_parameter_value().bool_value)
        self.emg_topic = str(self.get_parameter('emg_topic').get_parameter_value().string_value)
        try:
            self.emg_log_every_n = int(self.get_parameter('emg_log_every_n').get_parameter_value().integer_value)
        except Exception:
            self.emg_log_every_n = 1
        if self.emg_log_every_n <= 0:
            self.emg_log_every_n = 1
        self.emg_write_mode = str(self.get_parameter('emg_write_mode').get_parameter_value().string_value).strip().lower()
        if self.emg_write_mode not in {'hold','blank'}:
            self.emg_write_mode = 'hold'
        try:
            self.emg_warn_after_sec = float(self.get_parameter('emg_warn_after_sec').get_parameter_value().double_value)
        except Exception:
            self.emg_warn_after_sec = 2.0
        if self.emg_warn_after_sec < 0.0:
            self.emg_warn_after_sec = 0.0
        try:
            self.csv_flush_every_n = int(self.get_parameter('csv_flush_every_n').get_parameter_value().integer_value)
        except Exception:
            self.csv_flush_every_n = 10
        if self.csv_flush_every_n <= 0:
            self.csv_flush_every_n = 1
        # Environment diagnostics
        try:
            dom = os.environ.get('ROS_DOMAIN_ID', '(unset)')
            rmw = os.environ.get('RMW_IMPLEMENTATION', '(default)')
            self.get_logger().info(f"Env: ROS_DOMAIN_ID={dom}, RMW_IMPLEMENTATION={rmw}, emg_topic='{self.emg_topic}'")
        except Exception:
            pass

        # State holders (latest) - 타임스탬프는 기록 시점에 생성하므로 값만 저장
        self._force: List[Optional[Tuple[float,float,float,float,float,float]]] = [None, None, None]
        self._deform_circ: Optional[float] = None
        self._deform_ecc: Optional[float] = None
        self._emg: Optional[List[float]] = None
        self._emg_recv_count: int = 0
        self._emg_warned: bool = False
        self._emg_last_written_count: int = 0
        self._ee: Optional[Tuple[float,float,float]] = None
        self._ee_mf: Optional[Tuple[float,float,float]] = None
        self._ee_th: Optional[Tuple[float,float,float]] = None
        self._ee_recv_count: int = 0
        self._ee_warned: bool = False
        # EMG 저장 주기 카운터
        self._emg_log_counter: int = 0

        # Logging state
        self._logging_active = self.start_immediately
        self._flush_counter = 0

        # Publisher: logging active state + human-readable status + current file path
        self._state_pub = self.create_publisher(Bool, '/data_logger/logging_active', 10)
        self._status_pub = self.create_publisher(String, '/data_logger/status', 10)
        self._file_pub = self.create_publisher(String, '/data_logger/current_file', 10)

        # Track current and last CSV file path (for user feedback)
        self._current_csv_path: Optional[str] = None
        self._last_closed_csv_path: Optional[str] = None

        # Subscriptions: force sensors 2..3 (s1은 저장하지 않음)
        self._force_topics = (
            ('/force_sensor/s1/wrench', 0),
            ('/force_sensor/s2/wrench', 1),
            ('/force_sensor/s3/wrench', 2),
        )
        for topic, idx in self._force_topics:
            self.create_subscription(WrenchStamped, topic, lambda msg, i=idx: self._on_force(i, msg), 50)

        # Deformity metrics
        self.create_subscription(Float32, '/deformity_tracker/circularity', self._on_deform_circ, 10)
        self.create_subscription(Float32, '/deformity_tracker/eccentricity', self._on_deform_ecc, 10)

        # EMG
        try:
            self.create_subscription(Float32MultiArray, self.emg_topic, self._on_emg, 50)
            self.get_logger().info(f"EMG 구독: {self.emg_topic}")
        except Exception as e:
            self.get_logger().warn(f"EMG subscribe 실패({self.emg_topic}): {e}")

        # Optional EE pose topic(s)
        if self.ee_pose_topic_if:
            try:
                self.create_subscription(PoseStamped, self.ee_pose_topic_if, self._on_ee_pose, 20)
                self.get_logger().info(f"EE Pose 구독(PoseStamped): {self.ee_pose_topic_if}")
            except Exception as e:
                self.get_logger().warn(f"EE pose topic(PoseStamped) subscribe 실패: {e}")
        if self.ee_pose_topic_mf:
            try:
                self.create_subscription(PoseStamped, self.ee_pose_topic_mf, self._on_ee_pose_mf, 20)
                self.get_logger().info(f"EE Pose 구독(MF): {self.ee_pose_topic_mf}")
            except Exception as e:
                self.get_logger().warn(f"EE pose(MF) subscribe 실패: {e}")
        if self.ee_pose_topic_th:
            try:
                self.create_subscription(PoseStamped, self.ee_pose_topic_th, self._on_ee_pose_th, 20)
                self.get_logger().info(f"EE Pose 구독(TH): {self.ee_pose_topic_th}")
            except Exception as e:
                self.get_logger().warn(f"EE pose(TH) subscribe 실패: {e}")
        if self.ee_pose_odom_topic:
            try:
                self.create_subscription(Odometry, self.ee_pose_odom_topic, self._on_ee_odom, 20)
                self.get_logger().info(f"EE Pose 구독(Odometry): {self.ee_pose_odom_topic}")
            except Exception as e:
                self.get_logger().warn(f"EE pose topic(Odometry) subscribe 실패: {e}")

        # Hand tracker key input (toggle logging on 's')
        self.create_subscription(String, '/hand_tracker/key', self._on_key, 10)

        # CSV init
        self._csv_fp = None
        self._csv_writer = None
        # Open CSV immediately only if starting active; otherwise wait for first 's'
        if self._logging_active:
            self._init_csv()

        # Timer
        period = max(0.001, 1.0 / max(1e-6, self.rate_hz))
        self.timer = self.create_timer(period, self._on_timer)
        state_txt = 'ACTIVE' if self._logging_active else 'IDLE'
        self.get_logger().info(f'DataLogger 시작 (rate={self.rate_hz:.1f}Hz, state={state_txt})')
        # Initial state publish
        try:
            self._state_pub.publish(Bool(data=self._logging_active))
        except Exception:
            pass
        # Also publish initial status line
        try:
            state_txt = 'ACTIVE' if self._logging_active else 'IDLE'
            self._status_pub.publish(String(data=f'DataLogger {state_txt}'))
            if self._current_csv_path:
                self._file_pub.publish(String(data=self._current_csv_path))
        except Exception:
            pass

    def _on_key(self, msg: String) -> None:
        try:
            if str(msg.data).lower() == 's':
                # Toggle ON/OFF: start -> open new CSV, stop -> close CSV
                if not self._logging_active:
                    self._logging_active = True
                    try:
                        self._init_csv()
                        self.get_logger().info('[Start] Logging ACTIVATED and CSV opened via hand_tracker (s)')
                        try:
                            self._status_pub.publish(String(data='DataLogger ACTIVE'))
                            if self._current_csv_path:
                                self._file_pub.publish(String(data=self._current_csv_path))
                        except Exception:
                            pass
                    except Exception as e:
                        self.get_logger().warn(f'CSV start 실패: {e}')
                    try:
                        self._state_pub.publish(Bool(data=True))
                    except Exception:
                        pass
                else:
                    self._logging_active = False
                    try:
                        self._close_csv()
                        self.get_logger().info('[Stop] Logging DEACTIVATED and CSV closed via hand_tracker (s)')
                        try:
                            self._status_pub.publish(String(data='DataLogger IDLE'))
                            # Publish last file path so user knows where it was saved
                            if self._last_closed_csv_path:
                                self._file_pub.publish(String(data=self._last_closed_csv_path))
                        except Exception:
                            pass
                    except Exception as e:
                        self.get_logger().warn(f'CSV close 실패: {e}')
                    try:
                        self._state_pub.publish(Bool(data=False))
                    except Exception:
                        pass
        except Exception:
            pass

    def _resolve_default_csv_dir(self) -> Path:
        # 기본 CSV 디렉터리를 패키지 하위 outputs/logs/YYYYMMDD로 설정
        # __file__ = <pkg_dir>/hri_falcon_robot_bridge/data_logger_node.py
        pkg_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        date_dir = datetime.datetime.now().strftime('%Y%m%d')
        return pkg_dir / 'outputs' / 'logs' / date_dir

    def _init_csv(self) -> None:
        try:
            base_dir = Path(self.csv_dir_param) if self.csv_dir_param else self._resolve_default_csv_dir()
            base_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            path = base_dir / f'{ts}_synced.csv'
            # If something was left open, close it first
            self._close_csv()
            self._csv_fp = open(path, 'w', newline='')
            self._csv_writer = csv.writer(self._csv_fp)
            self._current_csv_path = str(path)
            # reset EMG schedule counter when starting a new file
            try:
                self._emg_log_counter = 0
                # Force EMG to be considered "updated" for the first eligible row
                self._emg_last_written_count = -1
            except Exception:
                pass
            header = ['t_sec','t_nanosec']
            for label in ('s1', 's2', 's3'):
                header += [
                    f'{label}_fx', f'{label}_fy', f'{label}_fz',
                    f'{label}_tx', f'{label}_ty', f'{label}_tz',
                    f'{label}_stamp_sec', f'{label}_stamp_nsec'
                ]
            header += ['ee_px','ee_py','ee_pz','ee_stamp_sec','ee_stamp_nsec']
            # 추가: MF / TH EE 위치
            header += ['ee_mf_px','ee_mf_py','ee_mf_pz','ee_mf_stamp_sec','ee_mf_stamp_nsec']
            header += ['ee_th_px','ee_th_py','ee_th_pz','ee_th_stamp_sec','ee_th_stamp_nsec']
            header += ['deform_circ','deform_circ_stamp_sec','deform_circ_stamp_nsec']
            header += ['deform_ecc','deform_ecc_stamp_sec','deform_ecc_stamp_nsec']
            header += [f'emg_ch{i+1}' for i in range(8)] + ['emg_stamp_sec','emg_stamp_nsec']
            self._csv_writer.writerow(header)
            # Sanity: ensure no malformed placeholders remain
            if any('{si}' in h for h in header):
                self.get_logger().warn('CSV header contains malformed placeholders. Please report this issue.')
            self.get_logger().info(f'CSV logging -> {path}')
            # Broadcast current file path for UI/overlays
            try:
                self._file_pub.publish(String(data=str(path)))
                self._status_pub.publish(String(data='DataLogger ACTIVE'))
            except Exception:
                pass
        except Exception as e:
            self.get_logger().error(f'CSV 초기화 실패: {e}')

    def _close_csv(self) -> None:
        try:
            if self._csv_fp is not None:
                try:
                    self._csv_fp.flush()
                except Exception:
                    pass
                self._csv_fp.close()
                # Remember last closed path for user feedback
                try:
                    self._last_closed_csv_path = self._current_csv_path  # type: ignore[attr-defined]
                except Exception:
                    pass
        except Exception:
            pass
        finally:
            self._csv_fp = None
            self._csv_writer = None
            self._current_csv_path = None

    # =============== Callbacks (타임스탬프 없이 값만 저장, 기록 시점에 타임스탬프 생성)
    def _on_force(self, idx: int, msg: Any) -> None:
        try:
            w = getattr(msg, 'wrench', None)
            self._force[idx] = (
                float(getattr(getattr(w, 'force', None), 'x', 0.0)),
                float(getattr(getattr(w, 'force', None), 'y', 0.0)),
                float(getattr(getattr(w, 'force', None), 'z', 0.0)),
                float(getattr(getattr(w, 'torque', None), 'x', 0.0)),
                float(getattr(getattr(w, 'torque', None), 'y', 0.0)),
                float(getattr(getattr(w, 'torque', None), 'z', 0.0))
            )
        except Exception:
            pass

    def _on_deform_circ(self, msg: Float32) -> None:
        try:
            self._deform_circ = float(msg.data)
        except Exception:
            pass

    def _on_deform_ecc(self, msg: Float32) -> None:
        try:
            self._deform_ecc = float(msg.data)
        except Exception:
            pass

    def _on_emg(self, msg: Float32MultiArray) -> None:
        try:
            data = [float(x) for x in list(msg.data)[:8]]
            self._emg = data
            self._emg_recv_count += 1
            # Debug: Log first and every 50th EMG packet to verify delivery
            if self._emg_recv_count == 1 or (self._emg_recv_count % 50 == 0):
                try:
                    preview = ", ".join(f"{v:.1f}" for v in data)
                    self.get_logger().info(f"[EMG] recv #{self._emg_recv_count}: {preview}")
                except Exception:
                    pass
        except Exception:
            pass

    def _on_ee_pose(self, msg: PoseStamped) -> None:
        try:
            p = msg.pose.position
            self._ee = (float(p.x), float(p.y), float(p.z))
            self._ee_recv_count += 1
        except Exception:
            pass

    def _on_ee_pose_mf(self, msg: PoseStamped) -> None:
        try:
            p = msg.pose.position
            self._ee_mf = (float(p.x), float(p.y), float(p.z))
        except Exception:
            pass

    def _on_ee_pose_th(self, msg: PoseStamped) -> None:
        try:
            p = msg.pose.position
            self._ee_th = (float(p.x), float(p.y), float(p.z))
        except Exception:
            pass

    def _on_ee_odom(self, msg: Odometry) -> None:
        try:
            p = msg.pose.pose.position
            self._ee = (float(p.x), float(p.y), float(p.z))
            self._ee_recv_count += 1
        except Exception:
            pass

    # =============== Timer write
    def _on_timer(self) -> None:
        if self._csv_writer is None:
            return
        if not self._logging_active:
            return  # paused
        # Warn once if EE is configured but not received
        if not self._ee_warned:
            if (self.ee_pose_topic_if or self.ee_pose_odom_topic) and self._ee_recv_count == 0:
                # Warn only after a short grace period (~1s)
                # Using rate_hz, wait ~1s worth of ticks
                ticks_for_1s = int(max(1.0, self.rate_hz))
                if not hasattr(self, '_tick_count'):
                    self._tick_count = 0
                self._tick_count += 1
                if self._tick_count >= ticks_for_1s:
                    self.get_logger().warn(
                        f"EE Pose 미수신: ee_pose_topic_if='{self.ee_pose_topic_if}', ee_pose_odometry_topic='{self.ee_pose_odom_topic}'. "
                        "토픽명이 맞는지, 퍼블리셔가 작동 중인지 확인하세요.")
                    self._ee_warned = True
        # Warn once if EMG is not received
        if not self._emg_warned:
            if self._emg_recv_count == 0:
                ticks_for_delay = int(max(1.0, self.rate_hz * max(0.0, self.emg_warn_after_sec)))
                if not hasattr(self, '_tick_count_emg'):
                    self._tick_count_emg = 0
                self._tick_count_emg += 1
                if self._tick_count_emg >= ticks_for_delay:
                    self.get_logger().warn(
                        f"EMG 미수신: emg_topic='{self.emg_topic}'. 토픽명이 맞는지, EMG 노드가 퍼블리시 중인지 확인하세요.")
                    self._emg_warned = True
        now_msg = self.get_clock().now().to_msg()
        row: List[str] = [str(now_msg.sec), str(now_msg.nanosec)]
        # Optional: light debug every ~1s
        try:
            if not hasattr(self, '_dbg_tick'):
                self._dbg_tick = 0
            self._dbg_tick += 1
            if self._dbg_tick % int(max(1.0, self.rate_hz)) == 0:
                self.get_logger().debug(f'EMG recv count={self._emg_recv_count}, last_written={self._emg_last_written_count}')
        except Exception:
            pass
        # Force sensors s1..s3 (타임스탬프는 현재 시각 사용)
        for i in range(3):
            f = self._force[i]
            if f is None:
                # 6 values + 2 stamps = 8 columns
                row += ['','','','','','','','']
            else:
                fx,fy,fz,tx,ty,tz = f
                row += [f"{fx:.6f}", f"{fy:.6f}", f"{fz:.6f}", f"{tx:.6f}", f"{ty:.6f}", f"{tz:.6f}", str(now_msg.sec), str(now_msg.nanosec)]
        # EE pose (position only, 타임스탬프는 현재 시각)
        if self._ee is None:
            row += ['','','','','']
        else:
            px,py,pz = self._ee
            row += [f"{px:.6f}", f"{py:.6f}", f"{pz:.6f}", str(now_msg.sec), str(now_msg.nanosec)]
        # EE pose (MF, 타임스탬프는 현재 시각)
        if self._ee_mf is None:
            row += ['','','','','']
        else:
            px,py,pz = self._ee_mf
            row += [f"{px:.6f}", f"{py:.6f}", f"{pz:.6f}", str(now_msg.sec), str(now_msg.nanosec)]
        # EE pose (TH, 타임스탬프는 현재 시각)
        if self._ee_th is None:
            row += ['','','','','']
        else:
            px,py,pz = self._ee_th
            row += [f"{px:.6f}", f"{py:.6f}", f"{pz:.6f}", str(now_msg.sec), str(now_msg.nanosec)]
        # Deform circ (타임스탬프는 현재 시각)
        if self._deform_circ is None:
            row += ['','','']
        else:
            circ = self._deform_circ
            row += [f"{circ:.6f}", str(now_msg.sec), str(now_msg.nanosec)]
        # Deform ecc (타임스탬프는 현재 시각)
        if self._deform_ecc is None:
            row += ['','','']
        else:
            ecc = self._deform_ecc
            row += [f"{ecc:.6f}", str(now_msg.sec), str(now_msg.nanosec)]
        # EMG (저장 간격 적용: emg_log_every_n, 타임스탬프는 현재 시각)
        self._emg_log_counter += 1
        if (self._emg_log_counter % self.emg_log_every_n) != 0:
            # 이번 행에서는 EMG 생략
            row += ['']*8 + ['','']
        else:
            if self._emg is None:
                row += ['']*8 + ['','']
            else:
                # 업데이트 여부 확인: _emg_recv_count 변화가 없으면 이전 값 반복 방지 옵션
                updated = (self._emg_recv_count != self._emg_last_written_count)
                if (self.emg_write_mode == 'blank') and (not updated):
                    row += ['']*8 + ['','']
                else:
                    vals = self._emg
                    vals = (vals + [0.0]*8)[:8]
                    row += [f"{v:.6f}" for v in vals] + [str(now_msg.sec), str(now_msg.nanosec)]
                    self._emg_last_written_count = self._emg_recv_count
        try:
            self._csv_writer.writerow(row)
            self._flush_counter += 1
            if self._csv_fp and (self._flush_counter % self.csv_flush_every_n == 0):
                self._csv_fp.flush()
        except Exception as e:
            self.get_logger().warn(f'CSV write 실패: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = DataLoggerNode()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
