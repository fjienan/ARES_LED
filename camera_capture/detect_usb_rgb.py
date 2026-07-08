#!/usr/bin/env python3
"""USB RGB 摄像头实时单色检测：显示检测结果，并保存阳性原图。"""

import argparse
from datetime import datetime
from pathlib import Path
import sys
import time

import cv2

from capture_usb_rgb import camera_name, open_camera


REPO_ROOT = Path(__file__).resolve().parents[1]
R2_PACKAGE = REPO_ROOT / 'r2_ws' / 'src' / 'rgb_camera_receiver'
sys.path.insert(0, str(R2_PACKAGE))

from rgb_camera_receiver.classifier import classifier_for_profile  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run the current single-strip detector on a USB RGB camera.')
    parser.add_argument(
        '--camera', '--camera-id', dest='camera_id', type=int, choices=(1, 2),
        default=1,
        help='选择摄像头 profile：1 对应 usb_rgb_1，2 对应 usb_rgb_2；默认 1')
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
        help='阳性原图保存目录；默认 <仓库>/camera_capture_positive_usb_rgb_<camera>')
    parser.add_argument(
        '--interval', type=float, default=1.0,
        help='连续阳性时的最小保存间隔秒数；默认 1.0')
    parser.add_argument('--jpeg-quality', type=int, default=95)
    parser.add_argument(
        '--no-save', action='store_true',
        help='只实时显示，不保存阳性图片')
    parser.add_argument(
        '--no-preview', action='store_true',
        help='不打开显示窗口，只打印和保存')
    return parser.parse_args()


def detector_config_path(profile: str) -> Path:
    return (
        R2_PACKAGE / 'config' / 'cameras' / profile / 'detector.yaml'
    )


def load_detector(profile: str):
    path = detector_config_path(profile)
    if not path.is_file():
        raise RuntimeError(f'找不到 detector 配置：{path}')
    classifier = classifier_for_profile(profile)
    config = classifier.load_config(str(path))
    return classifier, config, path


def scaled_frame(frame, scale: float):
    if scale >= 0.999:
        return frame
    return cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def detect_frame(frame, classifier, config, processing_scale: float):
    work = scaled_frame(frame, processing_scale)
    candidates = classifier.detect_candidates(work, config)
    if processing_scale < 0.999:
        inverse = 1.0 / processing_scale
        candidates = [item.scaled(inverse) for item in candidates]
    winner = classifier.select_winner(candidates, config)
    return candidates, winner


def save_positive_frame(
        output_dir: Path,
        frame,
        winner,
        now: float,
        last_save_time,
        interval: float,
        jpeg_quality: int):
    if last_save_time is not None and now - last_save_time < interval:
        return last_save_time
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
    filename = (
        f'{winner.color.lower()}_{timestamp}_'
        f'conf{winner.confidence:.3f}_score{winner.score:.3f}.jpg'
    )
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

    profile = f'usb_rgb_{args.camera_id}'
    classifier, config, config_path = load_detector(profile)
    processing_scale = (
        float(args.processing_scale)
        if args.processing_scale is not None
        else float(getattr(config, 'processing_scale', 1.0))
    )
    processing_scale = min(1.0, max(processing_scale, 0.1))

    output_dir = (
        Path(args.output).expanduser().resolve()
        if args.output
        else REPO_ROOT / f'camera_capture_positive_{profile}'
    )
    if not args.no_save:
        output_dir.mkdir(parents=True, exist_ok=True)

    capture, selected_device = open_camera(
        args.device, args.width, args.height, args.fps)
    print(f'profile: {profile}')
    print(f'detector: {config_path}')
    print(f'device: {selected_device} ({camera_name(selected_device)})')
    print(f'processing_scale: {processing_scale:g}')
    print(f'min_score: {getattr(config, "min_score", "unknown")}')
    if args.no_save:
        print('positive saving: disabled')
    else:
        print(f'positive output: {output_dir}')
        print(f'positive interval: {args.interval:g}s')
    print('press q/Esc/Ctrl+C to stop')

    if not args.no_preview:
        cv2.namedWindow('USB RGB single-color detector', cv2.WINDOW_NORMAL)

    last_save_time = None
    positive_active = False
    last_positive_color = None
    last_label = None
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                print('warning: failed to read a camera frame')
                time.sleep(0.05)
                continue

            candidates, winner = detect_frame(
                frame, classifier, config, processing_scale)
            now = time.monotonic()

            if winner is None:
                positive_active = False
                last_positive_color = None
                label = f'NONE candidates={len(candidates)}'
            else:
                color_changed = (
                    positive_active and last_positive_color != winner.color
                )
                label = (
                    f'{winner.color} score={winner.score:.3f} '
                    f'conf={winner.confidence:.3f} candidates={len(candidates)}')
                if not args.no_save:
                    should_save = (
                        (not positive_active)
                        or color_changed
                        or args.interval == 0
                    )
                    if not should_save and last_save_time is not None:
                        should_save = now - last_save_time >= args.interval
                    positive_active = True
                    last_positive_color = winner.color
                    if should_save:
                        last_save_time = save_positive_frame(
                            output_dir, frame, winner, now, last_save_time,
                            args.interval, args.jpeg_quality)
                else:
                    positive_active = True
                    last_positive_color = winner.color

            if label != last_label:
                print(label)
                last_label = label

            if not args.no_preview:
                rendered = classifier.annotate(frame, candidates, winner)
                if args.preview_scale != 1.0:
                    rendered = cv2.resize(
                        rendered, None,
                        fx=args.preview_scale, fy=args.preview_scale,
                        interpolation=cv2.INTER_AREA)
                cv2.imshow('USB RGB single-color detector', rendered)
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
