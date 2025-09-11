#!/usr/bin/env python3
"""간단 EMG ROS2 노드 (pyomyo + matplotlib)
 - 별도 프로세스에서 Myo PREPROCESSED (없으면 RAW) EMG 8채널 수집
 - /emg/raw Float32MultiArray 퍼블리시
 - 옵션: enable_plot 파라미터로 실시간 플롯
 - 요구: try / except 사용 최소 (오류 시 그냥 종료)
"""

import multiprocessing as mp
from collections import deque
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, MultiArrayDimension

from pyomyo import Myo, emg_mode  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from matplotlib import animation  # type: ignore
from matplotlib.cm import get_cmap  # type: ignore

EMG_CHANNELS = 8
QUEUE_MAX = 2048
PLOT_HISTORY = 1000  # plot 에 표시할 최대 샘플 수


def emg_worker(q: mp.Queue):
    mode = getattr(emg_mode, 'PREPROCESSED', None) or getattr(emg_mode, 'RAW', None) or 1
    m = Myo(mode=mode)  # type: ignore
    m.connect()
    if hasattr(m, 'set_leds'):
        m.set_leds([128, 0, 0], [128, 0, 0])  # type: ignore
    if hasattr(m, 'vibrate'):
        m.vibrate(1)  # type: ignore

    def on_emg(sample, moving=None):  # pyomyo 콜백 시그니처 호환
        # 큐가 가득 차면 가장 오래된 하나 제거 후 삽입
        if q.full():
            q.get()
        q.put(sample)

    if hasattr(m, 'add_emg_handler'):
        m.add_emg_handler(on_emg)  # type: ignore
    else:
        m.set_emg_handler(on_emg)  # type: ignore

    # 메인 루프 (차단형 폴링)
    while True:
        m.run()  # type: ignore[attr-defined]


class EMGNode(Node):
    def __init__(self):
        super().__init__('emg_node')
        self.declare_parameter('publish_rate_hz', 20.0)
        self.declare_parameter('enable_plot', True)
        self.rate = self.get_parameter('publish_rate_hz').get_parameter_value().double_value
        self.enable_plot = self.get_parameter('enable_plot').get_parameter_value().bool_value

        # Publisher
        self.pub = self.create_publisher(Float32MultiArray, '/emg/raw', 10)

        # Worker & Queue
        self.q: mp.Queue = mp.Queue(maxsize=QUEUE_MAX)
        self.proc = mp.Process(target=emg_worker, args=(self.q,), daemon=True)
        self.proc.start()
        self.get_logger().info('Myo worker 시작')

        self.last_sample = [0.0] * EMG_CHANNELS
        self._i = 0

        if self.enable_plot:
            self._init_plot()

        self.timer = self.create_timer(1.0 / self.rate, self.on_timer)
        self.get_logger().info(f'EMGNode 시작 (rate={self.rate:.1f}Hz, plot={self.enable_plot})')

    def _drain_latest(self):
        latest = None
        # empty() 체크 -> 최신만 유지
        while not self.q.empty():
            latest = self.q.get()
        return latest

    def on_timer(self):
        self._i += 1
        s = self._drain_latest()
        if s is not None:
            # EMG 라이브러리가 int 튜플을 줄 수 있으므로 float32 배열 요구사항 충족 위해 변환
            try:
                self.last_sample = [float(x) for x in list(s)[:EMG_CHANNELS]]
            except (TypeError, ValueError):
                # 잘못된 샘플이면 이전 값 유지
                pass
        msg = Float32MultiArray()
        # stride: 1차원에서는 전체 요소 수
        msg.layout.dim = [MultiArrayDimension(label='channel', size=EMG_CHANNELS, stride=EMG_CHANNELS)]
        msg.data = self.last_sample  # 이미 float 리스트
        self.pub.publish(msg)

    # 플롯 초기화
    def _init_plot(self):
        self.buf = [deque(maxlen=PLOT_HISTORY) for _ in range(EMG_CHANNELS)]
        plt.rcParams['figure.figsize'] = (5, 9)
        fig, axs = plt.subplots(EMG_CHANNELS, 1, sharex=True)
        if EMG_CHANNELS == 1:
            axs = [axs]
        else:
            axs = list(axs)
        cmap = get_cmap('tab10')
        colors = [cmap(i / 10.0) for i in range(10)]
        self.lines = []
        x_init = list(range(PLOT_HISTORY))
        zeros = [0.0] * PLOT_HISTORY
        for i, ax in enumerate(axs):
            line, = ax.plot(x_init, zeros, color=colors[i % len(colors)], lw=1.0)
            ax.set_xlim(0, PLOT_HISTORY)
            ax.set_ylim(0, 1024)
            ax.set_ylabel(f'ch{i+1}', rotation=0, labelpad=18, ha='right')
            ax.grid(True, alpha=0.3)
            self.lines.append(line)
        axs[-1].set_xlabel('samples')
        mgr = getattr(fig.canvas, 'manager', None)
        if mgr and hasattr(mgr, 'set_window_title'):
            mgr.set_window_title('EMG Live Plot')
        fig.tight_layout()
        self.fig = fig

        def animate(_frame):
            # 새 샘플 모두 수집 -> 버퍼 기록
            got = True
            while got:
                latest = self._drain_latest()
                if latest is None:
                    got = False
                else:
                    for ch in range(EMG_CHANNELS):
                        self.buf[ch].append(latest[ch])
            # 갱신
            for ch in range(EMG_CHANNELS):
                data = list(self.buf[ch])
                if not data:
                    continue
                x = list(range(len(data)))
                self.lines[ch].set_data(x, data)
                m = max(10, max(data))
                ax = self.lines[ch].axes
                ax.set_ylim(0, m)
                ax.set_xlim(max(0, len(data) - PLOT_HISTORY), len(data))
            return self.lines

        self.ani = animation.FuncAnimation(self.fig, animate, interval=10, blit=False)
        # GUI 이벤트 펌프 (ROS 스핀과 병행)
        self.gui_timer = self.create_timer(0.03, lambda: plt.pause(0.001))
        plt.ion()
        self.fig.show()

    def destroy_node(self):  # type: ignore
        if hasattr(self, 'proc') and self.proc.is_alive():
            self.proc.terminate()
        return super().destroy_node()


def main():
    rclpy.init()
    node = EMGNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
