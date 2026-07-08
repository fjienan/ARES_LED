#!/usr/bin/env python3
"""USB RGB 摄像头采集：自动选择外接摄像头，实时预览，并按固定间隔保存 JPG 原图。"""

import argparse
from datetime import datetime
import glob
import os
from pathlib import Path
import subprocess
import time

import cv2


INTEGRATED_CAMERA_KEYWORDS = (
    'integrated',
    'internal camera',
    'internal',
    'built-in',
    'builtin',
    'facetime',
    'laptop',
)


def camera_name(device: str) -> str:
    video_name = os.path.basename(os.path.realpath(device))
    name_path = Path('/sys/class/video4linux') / video_name / 'name'
    try:
        return name_path.read_text(encoding='utf-8').strip()
    except OSError:
        return ''


def is_integrated_camera(device: str) -> bool:
    name = camera_name(device).lower()
    return any(keyword in name for keyword in INTEGRATED_CAMERA_KEYWORDS)


def is_video_capture_device(device: str) -> bool:
    try:
        result = subprocess.run(
            ['v4l2-ctl', '--device', device, '--all'],
            check=False, capture_output=True, text=True, timeout=1.0)
    except (OSError, subprocess.TimeoutExpired):
        return True

    if result.returncode != 0:
        return True

    capability_lines = []
    in_device_caps = False
    for line in result.stdout.splitlines():
        if 'Device Caps' in line:
            in_device_caps = True
            capability_lines = []
            continue
        if in_device_caps:
            if line[:1].isspace():
                stripped = line.strip()
                if stripped:
                    capability_lines.append(stripped)
            elif capability_lines:
                break

    capabilities = capability_lines or result.stdout.splitlines()
    has_video = any(
        item in ('Video Capture', 'Video Capture Multiplanar')
        for item in capabilities)
    has_metadata = any(item == 'Metadata Capture' for item in capabilities)
    return has_video and not has_metadata


def device_label(device: str) -> str:
    name = camera_name(device) or 'unknown'
    real = os.path.realpath(device)
    return f'{device} -> {real} ({name})'


def read_camera_frame(capture):
    """读取一帧，并屏蔽部分 USB 摄像头 MJPEG 流触发的 libjpeg 噪声。"""
    stderr_fd = 2
    saved_stderr_fd = os.dup(stderr_fd)
    try:
        with open(os.devnull, 'w', encoding='utf-8') as devnull:
            os.dup2(devnull.fileno(), stderr_fd)
            return capture.read()
    finally:
        os.dup2(saved_stderr_fd, stderr_fd)
        os.close(saved_stderr_fd)


def camera_candidates(device: str):
    if device.lower() != 'auto':
        if is_integrated_camera(device):
            raise RuntimeError(
                f'refusing integrated camera {device} ({camera_name(device)})')
        return [device]

    # 优先使用 by-id 中的 index0。这样能避开同一摄像头暴露出的元数据节点，
    # 也比 /dev/videoN 更稳定。
    paths = sorted(glob.glob('/dev/v4l/by-id/*-video-index0'))
    paths += sorted(glob.glob('/dev/video*'))
    result = []
    real_devices = set()
    for path in paths:
        real = os.path.realpath(path)
        if is_integrated_camera(path):
            continue
        if not is_video_capture_device(path):
            continue
        if path in result or real in real_devices:
            continue
        result.append(path)
        real_devices.add(real)
    return result


def open_camera(device: str, width: int, height: int, fps: float):
    candidates = camera_candidates(device)
    failed = []
    opened = []
    for candidate in candidates:
        source = int(candidate) if candidate.isdigit() else candidate
        capture = cv2.VideoCapture(source, cv2.CAP_V4L2)
        if not capture.isOpened():
            capture.release()
            failed.append(candidate)
            continue

        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        capture.set(cv2.CAP_PROP_FPS, fps)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ok, frame = read_camera_frame(capture)
        if ok and frame is not None:
            if device.lower() != 'auto':
                return capture, candidate
            opened.append((candidate, capture))
            continue
        capture.release()
        failed.append(candidate)

    if device.lower() == 'auto':
        if len(opened) == 1:
            return opened[0][1], opened[0][0]
        if len(opened) > 1:
            for _, capture in opened:
                capture.release()
            lines = '\n  '.join(device_label(candidate) for candidate, _ in opened)
            raise RuntimeError(
                'found more than one readable non-integrated USB camera; '
                'keep only one connected or pass --device explicitly:\n  ' + lines)

    raise RuntimeError(
        f'cannot open USB RGB camera {device}; '
        f'failed candidates: {failed or ["none"]}')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Preview the USB RGB camera and save raw JPG images.')
    parser.add_argument(
        '--camera', '--camera-id', dest='camera_id', type=int, choices=(1, 2),
        default=1,
        help='camera dataset number: 1 saves to usb_rgb_1, 2 saves to usb_rgb_2; default: 1')
    parser.add_argument(
        '--device', default='auto',
        help='auto, /dev/video0, or a /dev/v4l/by-id path; default: auto')
    parser.add_argument(
        '--output',
        help='directory in which JPG images are saved; '
             'default: <repo>/camera_data/usb_rgb_<camera>/raw')
    parser.add_argument(
        '--prefix',
        help='saved file prefix; default: usb_rgb_<camera>')
    parser.add_argument(
        '--interval', type=float, default=2.0,
        help='save interval in seconds, default: 2')
    parser.add_argument(
        '--width', type=int,
        help='requested frame width; default: 2560 for camera 1, 1280 for camera 2')
    parser.add_argument(
        '--height', type=int,
        help='requested frame height; default: 1440 for camera 1, 720 for camera 2')
    parser.add_argument('--fps', type=float, default=30.0)
    parser.add_argument('--jpeg-quality', type=int, default=95)
    parser.add_argument(
        '--no-preview', action='store_true',
        help='save images without opening an OpenCV preview window')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.interval <= 0:
        raise SystemExit('--interval must be greater than zero')
    if not 1 <= args.jpeg_quality <= 100:
        raise SystemExit('--jpeg-quality must be in 1..100')

    camera_key = f'usb_rgb_{args.camera_id}'
    default_width = 2560 if args.camera_id == 1 else 1280
    default_height = 1440 if args.camera_id == 1 else 720
    width = args.width if args.width is not None else default_width
    height = args.height if args.height is not None else default_height
    default_output = (
        Path(__file__).resolve().parents[1] /
        'camera_data' / camera_key / 'raw')
    output_dir = Path(args.output).expanduser().resolve() if args.output else default_output
    prefix = args.prefix if args.prefix else camera_key
    output_dir.mkdir(parents=True, exist_ok=True)
    capture, selected_device = open_camera(
        args.device, width, height, args.fps)

    actual_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = capture.get(cv2.CAP_PROP_FPS)
    print(f'camera id: {args.camera_id}')
    print(f'device: {selected_device} ({camera_name(selected_device)})')
    print(f'format: {actual_width}x{actual_height} @ {actual_fps:.1f} Hz')
    print(f'output: {output_dir}')
    print(f'saving every {args.interval:g} seconds; press q/Esc/Ctrl+C to stop')

    if not args.no_preview:
        cv2.namedWindow('USB RGB capture', cv2.WINDOW_NORMAL)

    count = 0
    next_save = time.monotonic()
    try:
        while True:
            ok, frame = read_camera_frame(capture)
            if not ok or frame is None:
                print('warning: failed to read a camera frame')
                time.sleep(0.05)
                continue

            now = time.monotonic()
            if now >= next_save:
                stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
                path = output_dir / f'{prefix}_{stamp}_{count:06d}.jpg'
                saved = cv2.imwrite(
                    str(path), frame,
                    [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
                if not saved:
                    raise RuntimeError(f'failed to save {path}')
                count += 1
                next_save = now + args.interval
                print(f'[{count}] saved {path}')

            if not args.no_preview:
                preview = frame.copy()
                remaining = max(0.0, next_save - time.monotonic())
                cv2.putText(
                    preview,
                    f'USB RGB  saved: {count}  next: {remaining:.1f}s  q/Esc: quit',
                    (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
                    cv2.LINE_AA)
                cv2.imshow('USB RGB capture', preview)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        capture.release()
        if not args.no_preview:
            cv2.destroyAllWindows()
        print(f'stopped; saved {count} images in {output_dir}')


if __name__ == '__main__':
    main()
