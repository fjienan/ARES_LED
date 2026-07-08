#!/usr/bin/env python3
"""USB RGB 2 三段四色灯带实时检测：不依赖 ROS2。

用途：
- 从外接 USB 摄像头实时读取画面；
- 复用 usb_rgb_2 的单段检测器与三段组合逻辑；
- 实时显示三段识别结果；
- 识别到三段阳性时保存原图，连续阳性按间隔保存，编码变化立即保存。
"""

import argparse
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
import time

import cv2

from capture_usb_rgb import camera_name, open_camera, read_camera_frame
from detect_usb_rgb_2_three_segment import (
    REPO_ROOT,
    R2_PACKAGE,
    load_detector,
)
from rgb_camera_receiver.three_segment import (
    annotate_three_segments,
    detect_single_segments,
    detect_three_segments,
    merge_strong_and_weak_segments,
    weak_segments_from_masks,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run the usb_rgb_2 three-segment detector on a live camera.')
    parser.add_argument(
        '--device', default='auto',
        help='auto、/dev/videoX 或 /dev/v4l/by-id/...；默认 auto')
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--fps', type=float, default=30.0)
    parser.add_argument(
        '--processing-scale', type=float,
        help='检测前缩放比例；默认读取 detector.yaml 的 processing.scale')
    parser.add_argument(
        '--preview-scale', type=float, default=0.6,
        help='显示窗口缩放比例；默认 0.6')
    parser.add_argument(
        '--output',
        default=str(
            REPO_ROOT / 'camera_capture_results' / 'positive_usb_rgb_2_combined'),
        help='阳性原图保存目录；默认 camera_capture_results/positive_usb_rgb_2_combined')
    parser.add_argument(
        '--interval', type=float, default=1.0,
        help='连续识别到同一编码时的最小保存间隔秒数；默认 1.0')
    parser.add_argument('--jpeg-quality', type=int, default=95)
    parser.add_argument(
        '--no-save', action='store_true',
        help='只实时显示，不保存阳性图片')
    parser.add_argument(
        '--no-preview', action='store_true',
        help='不打开显示窗口，只打印和保存')
    parser.add_argument(
        '--config',
        default=str(
            R2_PACKAGE / 'config' / 'cameras' / 'usb_rgb_2' / 'detector.yaml'),
        help='usb_rgb_2 detector.yaml 路径')

    # 三段组合参数。默认值保持与离线脚本一致。
    parser.add_argument('--max-single-candidates', type=int, default=30)
    parser.add_argument('--max-results', type=int, default=12)
    parser.add_argument('--min-three-score', type=float, default=0.05)
    parser.add_argument('--winner-margin', type=float, default=1.2)
    parser.add_argument('--max-angle-degrees', type=float, default=18.0)
    parser.add_argument('--max-cross-distance', type=float, default=45.0)
    parser.add_argument('--min-center-distance-ratio', type=float, default=0.35)
    parser.add_argument('--max-center-distance-ratio', type=float, default=4.0)
    parser.add_argument('--max-gap-ratio', type=float, default=3.2)
    return parser.parse_args()


def detect_frame(frame, classifier, config, processing_scale: float, args):
    strong_candidates = detect_single_segments(
        frame, classifier, config, processing_scale)
    weak_candidates = []
    single_candidates = list(strong_candidates)
    three_candidates = detect_three_segments(single_candidates, args)
    if not three_candidates:
        strong_colors = {item.color for item in strong_candidates}
        allowed_colors = {
            item.name for item in config.colors
            if item.name not in strong_colors
        }
        weak_candidates = weak_segments_from_masks(
            frame, classifier, config, processing_scale, allowed_colors)
        single_candidates = merge_strong_and_weak_segments(
            strong_candidates, weak_candidates)
        three_candidates = detect_three_segments(single_candidates, args)
    return strong_candidates, weak_candidates, single_candidates, three_candidates


def save_positive_frame(
        output_dir: Path,
        frame,
        symbols: str,
        score: float,
        now: float,
        jpeg_quality: int):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
    filename = f'{symbols.lower()}_{timestamp}_score{score:.3f}.jpg'
    path = output_dir / filename
    ok = cv2.imwrite(
        str(path), frame,
        [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    if not ok:
        raise RuntimeError(f'保存失败：{path}')
    print(f'saved positive: {path}')
    return now


def main():
    args = parse_args()
    if not 1 <= args.jpeg_quality <= 100:
        raise SystemExit('--jpeg-quality must be in 1..100')
    if args.interval < 0:
        raise SystemExit('--interval must be >= 0')
    if args.preview_scale <= 0:
        raise SystemExit('--preview-scale must be > 0')

    config_path = Path(args.config).expanduser().resolve()
    if not config_path.is_file():
        raise SystemExit(f'配置文件不存在：{config_path}')

    classifier, config = load_detector(config_path)
    processing_scale = (
        float(args.processing_scale)
        if args.processing_scale is not None
        else float(getattr(config, 'processing_scale', 1.0))
    )
    processing_scale = min(1.0, max(processing_scale, 0.1))

    # 只把三段检测需要的字段传给离线识别核心，避免后续 argparse 改动互相影响。
    three_args = SimpleNamespace(
        max_single_candidates=args.max_single_candidates,
        max_results=args.max_results,
        min_three_score=args.min_three_score,
        winner_margin=args.winner_margin,
        max_angle_degrees=args.max_angle_degrees,
        max_cross_distance=args.max_cross_distance,
        min_center_distance_ratio=args.min_center_distance_ratio,
        max_center_distance_ratio=args.max_center_distance_ratio,
        max_gap_ratio=args.max_gap_ratio,
    )

    output_dir = Path(args.output).expanduser().resolve()
    if not args.no_save:
        output_dir.mkdir(parents=True, exist_ok=True)

    capture, selected_device = open_camera(
        args.device, args.width, args.height, args.fps)
    print('profile: usb_rgb_2')
    print(f'detector: {config_path}')
    print(f'device: {selected_device} ({camera_name(selected_device)})')
    print(f'processing_scale: {processing_scale:g}')
    print(f'min_three_score: {three_args.min_three_score:g}')
    if args.no_save:
        print('positive saving: disabled')
    else:
        print(f'positive output: {output_dir}')
        print(f'positive interval: {args.interval:g}s')
    print('press q/Esc/Ctrl+C to stop')

    if not args.no_preview:
        cv2.namedWindow('USB RGB 2 three-segment detector', cv2.WINDOW_NORMAL)

    last_save_time = None
    positive_active = False
    last_symbols = None
    last_label = None
    try:
        while True:
            ok, frame = read_camera_frame(capture)
            if not ok or frame is None:
                print('warning: failed to read a camera frame')
                time.sleep(0.05)
                continue

            started = time.perf_counter()
            strong, weak, singles, triples = detect_frame(
                frame, classifier, config, processing_scale, three_args)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            winner = triples[0] if triples else None
            now = time.monotonic()

            if winner is None:
                positive_active = False
                last_symbols = None
                label = (
                    f'NONE singles={len(singles)} '
                    f'(strong={len(strong)} weak={len(weak)}) '
                    f'time={elapsed_ms:.1f}ms')
            else:
                symbols = ''.join(symbol[0] for symbol in winner.symbols)
                changed = positive_active and symbols != last_symbols
                label = (
                    f'{symbols} score={winner.score:.3f} '
                    f'conf={winner.confidence:.2f} '
                    f'triples={len(triples)} time={elapsed_ms:.1f}ms')
                if not args.no_save:
                    should_save = (
                        (not positive_active)
                        or changed
                        or args.interval == 0
                        or last_save_time is None
                        or now - last_save_time >= args.interval
                    )
                    positive_active = True
                    last_symbols = symbols
                    if should_save:
                        last_save_time = save_positive_frame(
                            output_dir, frame, symbols, winner.score, now,
                            args.jpeg_quality)
                else:
                    positive_active = True
                    last_symbols = symbols

            if label != last_label:
                print(label)
                last_label = label

            if not args.no_preview:
                rendered = annotate_three_segments(frame, singles, triples)
                if args.preview_scale != 1.0:
                    rendered = cv2.resize(
                        rendered, None,
                        fx=args.preview_scale, fy=args.preview_scale,
                        interpolation=cv2.INTER_AREA)
                cv2.imshow('USB RGB 2 three-segment detector', rendered)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        capture.release()
        if not args.no_preview:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
