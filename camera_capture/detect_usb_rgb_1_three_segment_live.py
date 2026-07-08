#!/usr/bin/env python3
"""USB RGB 1 三段四色灯带实时检测：不依赖 ROS2。"""

import argparse
from datetime import datetime
from pathlib import Path
import sys
import time

import cv2

from capture_usb_rgb import camera_name, open_camera, read_camera_frame
from detect_usb_rgb_1_three_segment import (
    REPO_ROOT,
    R2_PACKAGE,
    annotate_three_segments,
    detect_frame,
    load_detector,
    resolve_processing_scale,
    three_segment_args,
)

sys.path.insert(0, str(R2_PACKAGE))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run the usb_rgb_1 three-segment detector on a live camera.')
    parser.add_argument(
        '--device', default='auto',
        help='auto、/dev/videoX 或 /dev/v4l/by-id/...；默认 auto')
    parser.add_argument('--width', type=int, default=2560)
    parser.add_argument('--height', type=int, default=1440)
    parser.add_argument('--fps', type=float, default=30.0)
    parser.add_argument(
        '--processing-scale', type=float,
        help='检测前缩放比例；默认读取 detector.yaml 的 processing.scale')
    parser.add_argument(
        '--preview-scale', type=float, default=0.45,
        help='显示窗口缩放比例；默认 0.45')
    parser.add_argument(
        '--output',
        default=str(
            REPO_ROOT / 'camera_capture_results' / 'positive_usb_rgb_1_combined'),
        help='阳性原图保存目录；默认 camera_capture_results/positive_usb_rgb_1_combined')
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
            R2_PACKAGE / 'config' / 'cameras' / 'usb_rgb_1' / 'detector.yaml'),
        help='usb_rgb_1 detector.yaml 路径')

    parser.add_argument('--max-single-candidates', type=int)
    parser.add_argument('--max-results', type=int)
    parser.add_argument('--min-three-score', type=float)
    parser.add_argument('--winner-margin', type=float)
    parser.add_argument('--max-angle-degrees', type=float)
    parser.add_argument('--max-cross-distance', type=float)
    parser.add_argument('--min-center-distance-ratio', type=float)
    parser.add_argument('--max-center-distance-ratio', type=float)
    parser.add_argument('--max-gap-ratio', type=float)
    parser.add_argument('--fast-dilate-pixels', type=int)
    parser.add_argument('--fast-min-component-area', type=int)
    parser.add_argument('--fast-max-component-area', type=int)
    parser.add_argument('--fast-min-color-pixels', type=int)
    parser.add_argument('--fast-min-segment-length', type=float)
    parser.add_argument('--fast-min-union-aspect', type=float)
    parser.add_argument('--fast-max-segments-per-color', type=int)
    return parser.parse_args()


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


def save_slow_frame(
        output_dir: Path,
        frame,
        elapsed_ms: float,
        symbols: str,
        jpeg_quality: int):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
    filename = (
        f'slow_{symbols.lower()}_{timestamp}_time{elapsed_ms:.1f}ms.jpg')
    path = output_dir / filename
    ok = cv2.imwrite(
        str(path), frame,
        [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    if not ok:
        raise RuntimeError(f'保存失败：{path}')
    print(f'saved slow frame: {path}')


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
    processing_scale = resolve_processing_scale(args, config, config_path)
    combo_args = three_segment_args(args, config_path)

    output_dir = Path(args.output).expanduser().resolve()
    if not args.no_save:
        output_dir.mkdir(parents=True, exist_ok=True)

    capture, selected_device = open_camera(
        args.device, args.width, args.height, args.fps)
    print('profile: usb_rgb_1')
    print(f'detector: {config_path}')
    print(f'device: {selected_device} ({camera_name(selected_device)})')
    print(f'processing_scale: {processing_scale:g}')
    print(f'min_command_score: {combo_args.min_three_score:g}')
    if args.no_save:
        print('positive saving: disabled')
    else:
        print(f'positive output: {output_dir}')
        print(f'positive interval: {args.interval:g}s')
        print('slow frame output: enabled when detection time > 100ms')
    print('press q/Esc/Ctrl+C to stop')

    if not args.no_preview:
        cv2.namedWindow('USB RGB 1 three-segment detector', cv2.WINDOW_NORMAL)

    last_save_time = None
    positive_active = False
    last_label = None
    try:
        while True:
            ok, frame = read_camera_frame(capture)
            if not ok or frame is None:
                print('warning: failed to read a camera frame')
                time.sleep(0.05)
                continue

            started = time.perf_counter()
            candidates, protocol_candidates, protocol_winner = detect_frame(
                frame, classifier, config, processing_scale, combo_args)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            now = time.monotonic()
            slow_symbols = 'none'

            if protocol_winner is None:
                positive_active = False
                label = (
                    f'NONE strip_candidates={len(candidates)} '
                    f'protocol_candidates={len(protocol_candidates)} '
                    f'time={elapsed_ms:.1f}ms')
            else:
                symbols = ''.join(symbol[0] for symbol in protocol_winner.symbols)
                slow_symbols = symbols
                label = (
                    f'id={protocol_winner.command_id} {symbols} '
                    f'score={protocol_winner.score:.3f} '
                    f'conf={protocol_winner.confidence:.2f} '
                    f'protocol_candidates={len(protocol_candidates)} '
                    f'time={elapsed_ms:.1f}ms')
                if not args.no_save:
                    should_save = (
                        (not positive_active)
                        or args.interval == 0
                        or last_save_time is None
                        or now - last_save_time >= args.interval
                    )
                    positive_active = True
                    if should_save:
                        last_save_time = save_positive_frame(
                            output_dir, frame, symbols,
                            protocol_winner.score, now, args.jpeg_quality)
                else:
                    positive_active = True

            if not args.no_save and elapsed_ms > 100.0:
                save_slow_frame(
                    output_dir, frame, elapsed_ms, slow_symbols,
                    args.jpeg_quality)

            if label != last_label:
                print(label)
                last_label = label

            if not args.no_preview:
                rendered = annotate_three_segments(frame, [], protocol_candidates)
                if args.preview_scale != 1.0:
                    rendered = cv2.resize(
                        rendered, None,
                        fx=args.preview_scale, fy=args.preview_scale,
                        interpolation=cv2.INTER_AREA)
                cv2.imshow('USB RGB 1 three-segment detector', rendered)
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
