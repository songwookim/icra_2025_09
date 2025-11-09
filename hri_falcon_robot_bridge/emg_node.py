#!/usr/bin/env python3
"""간단 EMG ROS2 노드 (pyomyo + matplotlib)
 - 별도 프로세스에서 Myo PREPROCESSED (없으면 RAW) EMG 8채널 수집
 - /emg/raw Float32MultiArray 퍼블리시
 - 옵션: enable_plot 또는 plot_mode 파라미터로 실시간 플롯
 - 플롯 창이 포커스를 가져가도 키 입력을 /hand_tracker/key 로 포워딩하여 다른 노드가 반응하도록 함
 - 요구: try / except 사용 최소 (오류 시 그냥 종료)
"""

import multiprocessing as mp
import threading
import queue
from collections import deque
import time
import os

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup  # type: ignore
from rclpy.executors import MultiThreadedExecutor  # type: ignore
from rcl_interfaces.msg import ParameterDescriptor  # type: ignore
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, String

from pyomyo import Myo, emg_mode  # type: ignore

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
        # Parameters
        self.declare_parameter('publish_rate_hz', 200.0)
        self.declare_parameter('enable_plot', False)  # 기본 비활성화 (Gdk 경고 회피)
        # 문자열/불리언 모두 허용 (동적 타이핑)
        self.declare_parameter('plot_mode', 'off', ParameterDescriptor(dynamic_typing=True))  # 'on','off','auto' 또는 bool
        self.declare_parameter('plot_animation_interval_ms', 10)
        # 백엔드 선택(고급): '', 'Qt5Agg', 'TkAgg', 'Agg' 등
        self.declare_parameter('plot_backend', '')
        self.declare_parameter('log_publish_debug', False)  # 퍼블리시 값 로그 토글
        self.rate = self.get_parameter('publish_rate_hz').get_parameter_value().double_value
        # Plot mode resolution (plot_mode overrides enable_plot if provided)
        self.enable_plot = self.get_parameter('enable_plot').get_parameter_value().bool_value
        # 안전 파싱: bool 또는 문자열 모두 처리
        _pm_val = self.get_parameter('plot_mode').value
        if isinstance(_pm_val, bool):
            mode = 'on' if _pm_val else 'off'
        else:
            try:
                mode = str(_pm_val).strip().lower()
            except Exception:
                mode = 'off'
        self.anim_interval_ms = int(self.get_parameter('plot_animation_interval_ms').get_parameter_value().integer_value)
        if mode == 'on':
            self.enable_plot = True
        elif mode == 'off':
            self.enable_plot = False
        elif mode == 'auto':
            # auto: 디스플레이가 없으면 비활성화
            self.enable_plot = bool(os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY'))

        # Callback groups: 퍼블리시 타이머와 GUI 펌프를 분리하여 블로킹 방지
        self.cb_pub = ReentrantCallbackGroup()
        self.cb_gui = ReentrantCallbackGroup()

        # Publisher
        self.pub = self.create_publisher(Float32MultiArray, '/emg/raw', 10)
        # Forward keys when plot window has focus
        self.key_pub = self.create_publisher(String, '/hand_tracker/key', 10)

        # Worker & Queue
        self.q = mp.Queue(maxsize=QUEUE_MAX)
        self.proc = mp.Process(target=emg_worker, args=(self.q,), daemon=True)
        self.proc.start()
        self.get_logger().info('Myo worker 시작')

        self.last_sample = [0.0] * EMG_CHANNELS
        self._i = 0

        # Background reader thread drains queue and updates last_sample
        self._stop_reader = False
        self.reader = threading.Thread(target=self._reader_loop, daemon=True)
        self.reader.start()

        if self.enable_plot:
            self._init_plot()

        # 주기 퍼블리시 타이머 (퍼블리시 전용 콜백 그룹)
        self.timer = self.create_timer(1.0 / self.rate, self.on_timer, callback_group=self.cb_pub)
        try:
            dom = os.environ.get('ROS_DOMAIN_ID', '(unset)')
            rmw = os.environ.get('RMW_IMPLEMENTATION', '(default)')
            self.get_logger().info(f'EMGNode 시작 (rate={self.rate:.1f}Hz, plot={self.enable_plot}, ROS_DOMAIN_ID={dom}, RMW={rmw})')
        except Exception:
            self.get_logger().info(f'EMGNode 시작 (rate={self.rate:.1f}Hz, plot={self.enable_plot})')

    def _reader_loop(self):
        # Continuously drain queue and keep only the latest sample
        while not self._stop_reader:
            try:
                # Wait for at least one sample (short timeout to allow clean shutdown)
                sample = self.q.get(timeout=0.1)
                # Flush remaining quickly to keep the newest
                while True:
                    try:
                        sample = self.q.get_nowait()
                    except queue.Empty:
                        break
                # Convert and assign
                try:
                    self.last_sample = [float(x) for x in list(sample)[:EMG_CHANNELS]]
                except (TypeError, ValueError):
                    # ignore malformed
                    pass
            except queue.Empty:
                # no data; loop again
                continue

    def on_timer(self):
        self._i += 1
        msg = Float32MultiArray()
        # stride: 1차원에서는 전체 요소 수
        msg.layout.dim = [MultiArrayDimension(label='channel', size=EMG_CHANNELS, stride=EMG_CHANNELS)]
        msg.data = self.last_sample  # 이미 float 리스트
        self.pub.publish(msg)
        if self.get_parameter('log_publish_debug').get_parameter_value().bool_value:
            self.get_logger().debug('pub ' + ', '.join(f'{v:.1f}' for v in self.last_sample))

    # 플롯 초기화
    def _init_plot(self):
        # 지연 임포트: 필요할 때만 matplotlib 로드 (GTK/Gdk 경고 회피)
        import matplotlib
        # 백엔드 결정
        want_display = bool(os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY'))
        backend_param = str(self.get_parameter('plot_backend').get_parameter_value().string_value).strip()
        backend_tried = None
        try:
            if not want_display:
                matplotlib.use('Agg', force=True)
                backend_tried = 'Agg'
            else:
                if backend_param:
                    matplotlib.use(backend_param, force=True)
                    backend_tried = backend_param
                else:
                    # 우선순위: Qt5Agg → TkAgg → Agg
                    for cand in ('Qt5Agg', 'TkAgg', 'Agg'):
                        try:
                            matplotlib.use(cand, force=True)
                            backend_tried = cand
                            break
                        except Exception:
                            continue
        except Exception:
            # 최후 수단
            try:
                matplotlib.use('Agg', force=True)
                backend_tried = 'Agg'
            except Exception:
                pass
        try:
            self.get_logger().info(f"Matplotlib backend: {matplotlib.get_backend()} (requested='{backend_param or 'auto'}', tried='{backend_tried}')")
        except Exception:
            pass
        # 환경 주의: Qt 백엔드 + offscreen이면 검은 화면이 될 수 있음
        try:
            be = str(matplotlib.get_backend()).lower()
            qpa = os.environ.get('QT_QPA_PLATFORM', '').lower()
            if ('qt' in be) and (qpa == 'offscreen'):
                self.get_logger().warn("QT_QPA_PLATFORM=offscreen 상태입니다. 플롯 표시를 위해 `export QT_QPA_PLATFORM=xcb` 권장")
        except Exception:
            pass
        import matplotlib.pyplot as plt  # type: ignore
        from matplotlib.cm import get_cmap  # type: ignore
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

        # 갱신 함수: 최신 샘플을 라인에 반영
        def _update_lines():
            s = self.last_sample
            for ch in range(EMG_CHANNELS):
                self.buf[ch].append(s[ch])
            for ch in range(EMG_CHANNELS):
                data = list(self.buf[ch])
                if not data:
                    continue
                x = list(range(len(data)))
                self.lines[ch].set_data(x, data)
                vmax = max(10.0, max(data))
                ax = self.lines[ch].axes
                ax.set_ylim(0, vmax)
                ax.set_xlim(max(0, len(data) - PLOT_HISTORY), len(data))
            try:
                self.fig.canvas.draw_idle()
            except Exception:
                pass
        self._mpl_manual_update = _update_lines
        # 가능하면 GUI 백엔드 타이머를 사용하여 메인 스레드에서 갱신
        self._canvas_timer_started = False
        try:
            be = __import__('matplotlib').get_backend().lower()
            if any(k in be for k in ('qt', 'tk', 'wx')):
                t = self.fig.canvas.new_timer(interval=max(1, self.anim_interval_ms))
                t.add_callback(self._mpl_manual_update)
                t.start()
                self._canvas_timer = t
                self._canvas_timer_started = True
                self.get_logger().info(f'GUI canvas timer started for backend={be}')
        except Exception:
            self._canvas_timer_started = False
        # 키 입력을 hand_tracker/key 로 포워딩
        def _on_mpl_key(evt):
            try:
                k = getattr(evt, 'key', None)
                if not k:
                    return
                # matplotlib는 'ctrl+s', 'left' 등 복합 키도 줌. 한 글자만 전달.
                if len(k) == 1:
                    ch = k.lower()
                    if ch in {'q','r','s','h','c','j','t','x','a','g'}:
                        self.key_pub.publish(String(data=ch))
            except Exception:
                pass
        try:
            self.fig.canvas.mpl_connect('key_press_event', _on_mpl_key)
        except Exception:
            pass

        # GUI 이벤트 펌프 (ROS 스핀과 병행). 백엔드 타이머가 없는 경우에만 백업 경로로 동작
        def _gui_tick():
            try:
                # 백엔드 GUI 타이머가 이미 갱신을 처리 중이면 건드리지 않음
                if getattr(self, '_canvas_timer_started', False):
                    return
                # 수동 갱신 + draw (백업 경로)
                if hasattr(self, '_mpl_manual_update'):
                    self._mpl_manual_update()
                try:
                    self.fig.canvas.draw_idle()
                except Exception:
                    pass
                # flush_events는 일부 백엔드에서만 필요
                try:
                    if hasattr(self.fig.canvas, 'flush_events'):
                        self.fig.canvas.flush_events()
                except Exception:
                    pass
            except Exception:
                pass

        pump_dt = max(0.005, float(self.anim_interval_ms) / 1000.0)
        self.gui_timer = self.create_timer(pump_dt, _gui_tick, callback_group=self.cb_gui)
        # 주: 창 생성/표시는 main()에서 plt.show(block=True)로 처리한다.

    def destroy_node(self):  # type: ignore
        self._stop_reader = True
        # No join needed for daemon thread; give it a moment by draining exception handling on shutdown
        if hasattr(self, 'proc') and self.proc.is_alive():
            self.proc.terminate()
        return super().destroy_node()


def main():
    rclpy.init()
    node = EMGNode()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        if getattr(node, 'enable_plot', False):
            # GUI는 메인 스레드에서, ROS 스핀은 백그라운드 스레드에서
            spin_thread = threading.Thread(target=executor.spin, daemon=True)
            spin_thread.start()
            try:
                import matplotlib.pyplot as plt  # type: ignore
                node.get_logger().info('Opening plot window (blocking show)')
                plt.show(block=True)  # 메인 스레드에서 GUI 이벤트 루프 실행
            except KeyboardInterrupt:
                pass
            finally:
                try:
                    executor.shutdown()
                except Exception:
                    pass
        else:
            executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
