#!/usr/bin/env python3
"""RealSense 기반 2D 타원 피팅 & 이심률 실시간 측정

기능:
    - RealSense 컬러 스트림 수신
    - HSV 마스크로 객체(예: 공) 추출
    - 최대 컨투어에 대해 OpenCV fitEllipse 수행
    - aspect=b/a, eccentricity=√(1-(b/a)^2), circularity=4πA/P^2 계산
    - 이동평균(--smooth), CSV 로깅(--csv), 마스크 표시(--show-mask)
    - 멀티 디바이스 선택(--serial/--auto-free/--list-devices, 재시도 --retry)
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from collections import deque
from typing import Iterable

import cv2
import numpy as np
import pyrealsense2 as rs  # type: ignore


# --------------------------------------------------
# 디바이스 열거
# --------------------------------------------------
def list_devices() -> list[tuple[str, str]]:
        ctx = rs.context()  # type: ignore[attr-defined]
        out: list[tuple[str, str]] = []
        for dev in ctx.query_devices():  # type: ignore[attr-defined]
                try:
                        sn = dev.get_info(rs.camera_info.serial_number)  # type: ignore[attr-defined]
                except Exception:
                        sn = "UNKNOWN"
                try:
                        name = dev.get_info(rs.camera_info.name)  # type: ignore[attr-defined]
                except Exception:
                        name = "RealSense"
                out.append((sn, name))
        return out


# --------------------------------------------------
# RealSense 초기화
# --------------------------------------------------
def open_realsense(
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        serial: str | None = None,
        auto_free: bool = False,
        retry: int = 3,
        retry_delay: float = 0.8,
) -> tuple[rs.pipeline, str]:  # type: ignore[type-arg]
        """컬러 스트림 전용 RealSense 파이프라인을 연다."""
        devs = list_devices()
        if not devs:
                raise RuntimeError("연결된 RealSense 디바이스가 없습니다. USB 연결/권한을 확인하세요.")

        # 후보 시리얼 목록 결정
        if serial:
                candidate_serials: list[str] = [serial]
        elif auto_free:
                candidate_serials = [sn for sn, _ in devs]
        else:
                # 인터랙티브 선택
                if sys.stdin.isatty():
                        print("[INFO] 사용할 RealSense 디바이스를 선택하세요:")
                        for idx, (sn, name) in enumerate(devs):
                                print(f"  [{idx}] serial={sn} name={name}")
                        while True:
                                sel = input("번호 입력 (Enter=0): ").strip()
                                if sel == "":
                                        choice = 0
                                        break
                                if sel.isdigit() and 0 <= int(sel) < len(devs):
                                        choice = int(sel)
                                        break
                                print("  잘못된 입력입니다. 다시 시도하세요.")
                        candidate_serials = [devs[choice][0]]
                else:
                        candidate_serials = [devs[0][0]]
                        if len(devs) > 1:
                                print(f"[INFO] 비인터랙티브: 첫 디바이스(serial={candidate_serials[0]}) 자동 선택")

        last_err: Exception | None = None

        for sn in candidate_serials:
                for attempt in range(1, retry + 1):
                        pipe = rs.pipeline()  # type: ignore[attr-defined]
                        cfg = rs.config()  # type: ignore[attr-defined]
                        try:
                                cfg.enable_device(sn)  # type: ignore[attr-defined]
                                cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)  # type: ignore[attr-defined]
                                pipe.start(cfg)
                                print(f"[INFO] RealSense 열기 성공 serial={sn} (attempt {attempt})")
                                return pipe, sn
                        except RuntimeError as e:
                                last_err = e
                                msg = str(e)
                                if "busy" in msg.lower():
                                        print(f"[WARN] 디바이스 busy serial={sn} attempt={attempt}/{retry}")
                                else:
                                        print(f"[WARN] 열기 실패 serial={sn} attempt={attempt}/{retry}: {msg}")
                                try:
                                        pipe.stop()
                                except Exception:
                                        pass
                                if attempt < retry:
                                        time.sleep(retry_delay)
                        except Exception as e:
                                last_err = e
                                print(f"[WARN] 예외 serial={sn} attempt={attempt}/{retry}: {e}")
                                try:
                                        pipe.stop()
                                except Exception:
                                        pass
                                if attempt < retry:
                                        time.sleep(retry_delay)

        if last_err:
                raise RuntimeError(
                        "모든 후보 디바이스 열기 실패. --list-devices 확인 후 --serial 지정 또는 다른 프로세스 점유 여부 확인."
                ) from last_err
        raise RuntimeError("디바이스 열기 실패(알 수 없는 상태).")


# --------------------------------------------------
# 타원/이심률 계산 유틸
# --------------------------------------------------
def compute_metrics_from_contour(cnt: np.ndarray) -> dict[str, float] | None:
        area = cv2.contourArea(cnt)
        if area < 1e-3:
                return None

        peri = cv2.arcLength(cnt, True)
        circularity = 4 * math.pi * area / (peri * peri) if peri > 1e-6 else 0.0

        if len(cnt) < 5:
                return None

        (cx, cy), (d1, d2), angle = cv2.fitEllipse(cnt)
        a_raw, b_raw = d1 / 2.0, d2 / 2.0  # 반축(semiaxes)
        if a_raw < b_raw:
                a_raw, b_raw = b_raw, a_raw
        a, b = a_raw, b_raw
        if a <= 1e-6:
                return None

        aspect = b / a
        ecc = math.sqrt(max(0.0, 1.0 - (aspect * aspect)))  # sqrt(1 - (b/a)^2)

        return {
                "cx": float(cx),
                "cy": float(cy),
                "a": float(a),
                "b": float(b),
                "aspect": float(aspect),
                "eccentricity": float(ecc),
                "circularity": float(circularity),
                "angle": float(angle),
                "area": float(area),
        }


def average_metrics(items: Iterable[dict[str, float]]) -> dict[str, float]:
        arr = list(items)
        if not arr:
                return {}
        keys = ["a", "b", "aspect", "eccentricity", "circularity", "area"]
        avg = {k: float(np.mean([d[k] for d in arr])) for k in keys}
        # 대표 center/angle은 최신값 사용
        last = arr[-1]
        avg["cx"], avg["cy"], avg["angle"] = last["cx"], last["cy"], last["angle"]
        return avg


def build_mask(hsv: np.ndarray, r1: list[int], r2: list[int] | None) -> np.ndarray:
        h1, s1, v1, h2, s2, v2 = r1
        lo1 = np.array([h1, s1, v1], np.uint8)
        hi1 = np.array([h2, s2, v2], np.uint8)
        mask = cv2.inRange(hsv, lo1, hi1)
        if r2:
                hb1, sb1, vb1, hb2, sb2, vb2 = r2
                lo2 = np.array([hb1, sb1, vb1], np.uint8)
                hi2 = np.array([hb2, sb2, vb2], np.uint8)
                mask2 = cv2.inRange(hsv, lo2, hi2)
                mask = cv2.bitwise_or(mask, mask2)
        return mask


def put_text(img: np.ndarray, y: int, line: str) -> int:
        cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
        cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        return y + 22


# --------------------------------------------------
# 메인 루프
# --------------------------------------------------
def run(args: argparse.Namespace) -> None:
        pipe, chosen_serial = open_realsense(
                args.width,
                args.height,
                args.fps,
                serial=args.serial,
                auto_free=args.auto_free,
                retry=args.retry,
                retry_delay=args.retry_delay,
        )

        smooth_buf: deque[dict[str, float]] = deque(maxlen=max(1, args.smooth))

        csv_file = open(args.csv, "w", newline="") if args.csv else None
        csv_writer = csv.writer(csv_file) if csv_file else None
        if csv_writer:
                csv_writer.writerow(["time", "a", "b", "aspect", "eccentricity", "circularity", "area"])

        try:
                last_t = time.time()
                frame_count = 0

                while True:
                        frames = pipe.wait_for_frames()
                        color_frame = frames.get_color_frame()
                        if not color_frame:
                                continue

                        color = np.asanyarray(color_frame.get_data())
                        disp = color.copy()

                        hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
                        mask = build_mask(hsv, args.hsv_range, args.hsv_range2)

                        if args.erode > 0:
                                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.erode, args.erode))
                                mask = cv2.erode(mask, k)
                        if args.dilate > 0:
                                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.dilate, args.dilate))
                                mask = cv2.dilate(mask, k)

                        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                        metrics: dict[str, float] | None = None
                        if cnts:
                                cnt = max(cnts, key=cv2.contourArea)
                                if cv2.contourArea(cnt) >= args.min_area:
                                        m = compute_metrics_from_contour(cnt)
                                        if m:
                                                smooth_buf.append(m)
                                                metrics = average_metrics(smooth_buf)
                                                # 타원 그리기(최근 측정값 사용)
                                                cv2.ellipse(
                                                        disp,
                                                        ((m["cx"], m["cy"]), (m["a"] * 2, m["b"] * 2), m["angle"]),
                                                        (0, 255, 0),
                                                        2,
                                                )
                                                cv2.drawContours(disp, [cnt], -1, (0, 128, 255), 1)

                        y = 24
                        y = put_text(disp, y, f"Ellipse Eccentricity Demo (q quit) serial={chosen_serial}")
                        if metrics:
                                y = put_text(disp, y, f"a={metrics['a']:.1f}px b={metrics['b']:.1f}px aspect={metrics['aspect']:.3f}")
                                y = put_text(
                                        disp,
                                        y,
                                        f"eccentricity={metrics['eccentricity']:.3f} circularity={metrics['circularity']:.3f}",
                                )
                        else:
                                y = put_text(disp, y, "No valid contour")

                        frame_count += 1
                        now = time.time()
                        if now - last_t >= 1.0:
                                fps = frame_count / (now - last_t)
                                y = put_text(disp, y, f"FPS={fps:.1f}")
                                last_t = now
                                frame_count = 0

                        cv2.imshow("Eccentricity", disp)
                        if args.show_mask:
                                cv2.imshow("Mask", mask)

                        if metrics and csv_writer:
                                csv_writer.writerow(
                                        [
                                                time.time(),
                                                metrics["a"],
                                                metrics["b"],
                                                metrics["aspect"],
                                                metrics["eccentricity"],
                                                metrics["circularity"],
                                                metrics["area"],
                                        ]
                                )

                        if cv2.waitKey(1) & 0xFF == ord("q"):
                                break
        finally:
                try:
                        pipe.stop()
                except Exception:
                        pass
                if csv_file:
                        csv_file.close()
                cv2.destroyAllWindows()


# --------------------------------------------------
# 인자 파싱
# --------------------------------------------------
def parse_args() -> argparse.Namespace:
        ap = argparse.ArgumentParser()
        ap.add_argument("--width", type=int, default=640)
        ap.add_argument("--height", type=int, default=480)
        ap.add_argument("--fps", type=int, default=30)
        ap.add_argument(
                "--hsv-range",
                nargs=6,
                type=int,
                default=[20, 100, 100, 35, 255, 255],
                metavar=("H1", "S1", "V1", "H2", "S2", "V2"),
        )
        ap.add_argument(
                "--hsv-range2",
                nargs=6,
                type=int,
                default=None,
                metavar=("H1b", "S1b", "V1b", "H2b", "S2b", "V2b"),
                help="두 번째 HSV 범위 (빨간색 등 Hue wrap 용)",
        )
        ap.add_argument("--erode", type=int, default=0)
        ap.add_argument("--dilate", type=int, default=0)
        ap.add_argument("--min-area", type=float, default=150.0, help="컨투어 최소 면적 (px)")
        ap.add_argument("--smooth", type=int, default=1, help="이동평균 윈도 크기 (1=비활성)")
        ap.add_argument("--show-mask", action="store_true")
        ap.add_argument("--csv", type=str, default=None, help="CSV 파일 이름")

        # 다중 디바이스 옵션
        ap.add_argument("--serial", type=str, default=None, help="사용할 RealSense 디바이스 시리얼")
        ap.add_argument("--list-devices", action="store_true", help="연결된 RealSense 디바이스 나열 후 종료")
        ap.add_argument("--auto-free", action="store_true", help="--serial 미지정 시 사용 가능한 첫 디바이스 자동 선택")
        ap.add_argument("--retry", type=int, default=3, help="열기 실패 재시도 횟수")
        ap.add_argument("--retry-delay", type=float, default=0.8, help="재시도 사이 대기 시간(초)")
        return ap.parse_args()


def main() -> None:
        args = parse_args()
        if args.list_devices:
                devs = list_devices()
                if not devs:
                        print("[INFO] 연결된 RealSense 디바이스가 없습니다.")
                else:
                        print("[INFO] 연결된 디바이스 목록:")
                        for sn, name in devs:
                                print(f"  - serial={sn} name={name}")
                return
        run(args)


if __name__ == "__main__":
        main()
