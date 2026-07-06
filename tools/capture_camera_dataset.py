#!/usr/bin/env python3
"""按相机配置采集并预览灯带训练图片。"""

import argparse
import glob
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Dict, Iterable

import cv2
import yaml


UNCLASSIFIED_DIR = 'UNCLASSIFIED'
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_ROOT = (
    ROOT / 'r2_ws/src/rgb_camera_receiver/config/cameras')
DEFAULT_DATA_ROOT = ROOT / 'camera_data'


def load_profile(config_root: Path, profile: str) -> Dict:
    path = config_root / profile / 'capture.yaml'
    if not path.is_file():
        raise RuntimeError(f'找不到相机采集配置：{path}')
    with path.open('r', encoding='utf-8') as stream:
        config = yaml.safe_load(stream) or {}
    if config.get('source_type') not in {'v4l2', 'ros_topic'}:
        raise RuntimeError(f'{path} 中的 source_type 无效')
    return config


def camera_name(device: str) -> str:
    video_name = os.path.basename(os.path.realpath(device))
    path = Path('/sys/class/video4linux') / video_name / 'name'
    try:
        return path.read_text(encoding='utf-8').strip()
    except OSError:
        return ''


def camera_candidates(device: str, reject_keywords: Iterable[str]):
    if device.lower() != 'auto':
        return [device]
    paths = sorted(glob.glob('/dev/v4l/by-id/*-video-index0'))
    paths += sorted(glob.glob('/dev/v4l/by-id/*'))
    paths += sorted(glob.glob('/dev/video*'))
    rejected = tuple(str(item).lower() for item in reject_keywords)
    result = []
    seen = set()
    for path in paths:
        real = os.path.realpath(path)
        if real in seen:
            continue
        if any(word in camera_name(path).lower() for word in rejected):
            continue
        seen.add(real)
        result.append(path)
    return result


def open_v4l2(config: Dict):
    failed = []
    device = str(config.get('device', 'auto'))
    width = int(config.get('width', 1280))
    height = int(config.get('height', 720))
    fps = float(config.get('fps', 30.0))
    for candidate in camera_candidates(
            device, config.get('reject_keywords', [])):
        capture = cv2.VideoCapture(candidate, cv2.CAP_V4L2)
        if not capture.isOpened():
            capture.release()
            failed.append(candidate)
            continue
        capture.set(
            cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        capture.set(cv2.CAP_PROP_FPS, fps)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        apply_v4l2_controls(
            candidate, str(config.get('v4l2_controls', '')))
        ok, frame = capture.read()
        if ok and frame is not None:
            return capture, candidate
        capture.release()
        failed.append(candidate)
    raise RuntimeError(
        f'无法打开相机 {device}；尝试过：{failed or ["无候选设备"]}')


def apply_v4l2_controls(device: str, controls: str) -> None:
    if not controls.strip():
        return
    executable = Path('/usr/bin/v4l2-ctl')
    if not executable.exists():
        raise RuntimeError('采集配置包含 v4l2_controls，但未安装 v4l2-ctl')
    for assignment in controls.split(','):
        assignment = assignment.strip()
        if not assignment:
            continue
        result = subprocess.run(
            [str(executable), '--device', device, '--set-ctrl', assignment],
            text=True, capture_output=True, check=False)
        if result.returncode != 0:
            message = (result.stderr or result.stdout).strip()
            raise RuntimeError(
                f'设置相机参数 {assignment} 失败：{message}')


def next_index(output: Path) -> int:
    pattern = re.compile(r'^capture_(\d+)\.jpg$')
    indices = []
    for path in output.glob('capture_*.jpg'):
        match = pattern.match(path.name)
        if match:
            indices.append(int(match.group(1)))
    return max(indices, default=0) + 1


def save_frame(
        output: Path, index: int, frame, jpeg_quality: int) -> Path:
    path = output / f'capture_{index:04d}.jpg'
    if not cv2.imwrite(
            str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]):
        raise RuntimeError(f'无法保存图片：{path}')
    return path


def show_preview(frame, profile: str, count: int) -> bool:
    preview = frame.copy()
    cv2.putText(
        preview, f'{profile} / unclassified  saved: {count}  q/Esc: quit',
        (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
        cv2.LINE_AA)
    cv2.imshow('LED camera dataset capture', preview)
    return (cv2.waitKey(1) & 0xFF) in (ord('q'), 27)


def capture_v4l2(
        config: Dict, output: Path, profile: str,
        interval: float, jpeg_quality: int) -> int:
    capture, device = open_v4l2(config)
    print(f'相机：{device}（{camera_name(device)}）')
    count = 0
    index = next_index(output)
    next_save = time.monotonic()
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                time.sleep(0.02)
                continue
            now = time.monotonic()
            if now >= next_save:
                path = save_frame(output, index, frame, jpeg_quality)
                print(f'已保存：{path}')
                count += 1
                index += 1
                next_save = now + interval
            if show_preview(frame, profile, count):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()
    return count


def capture_ros_topic(
        config: Dict, output: Path, profile: str,
        interval: float, jpeg_quality: int) -> int:
    try:
        import rclpy
        from cv_bridge import CvBridge
        from rclpy.node import Node
        from sensor_msgs.msg import Image
    except ImportError as error:
        raise RuntimeError(
            'Odin1 采集需要已 source 的 ROS 2 环境以及 cv_bridge') from error

    topic = str(config.get('topic', '/odin1/image'))
    encoding = str(config.get('encoding', 'bgr8'))

    class ImageCollector(Node):
        def __init__(self):
            super().__init__('odin1_dataset_capture')
            self.bridge = CvBridge()
            self.frame = None
            self.subscription = self.create_subscription(
                Image, topic, self.on_image, 1)

        def on_image(self, message):
            self.frame = self.bridge.imgmsg_to_cv2(
                message, desired_encoding=encoding)

    rclpy.init()
    node = ImageCollector()
    count = 0
    index = next_index(output)
    next_save = time.monotonic()
    print(f'ROS 图像话题：{topic}，等待第一帧……')
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
            if node.frame is None:
                continue
            frame = node.frame
            now = time.monotonic()
            if now >= next_save:
                path = save_frame(output, index, frame, jpeg_quality)
                print(f'已保存：{path}')
                count += 1
                index += 1
                next_save = now + interval
            if show_preview(frame, profile, count):
                break
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        cv2.destroyAllWindows()
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='采集指定相机的灯带训练图片')
    parser.add_argument('--profile', required=True, choices=('usb_rgb', 'odin1'))
    parser.add_argument('--config-root', type=Path, default=DEFAULT_CONFIG_ROOT)
    parser.add_argument('--data-root', type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument('--interval', type=float)
    parser.add_argument('--jpeg-quality', type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_profile(args.config_root.resolve(), args.profile)
    interval = (
        args.interval if args.interval is not None
        else float(config.get('interval_sec', 3.0)))
    quality = (
        args.jpeg_quality if args.jpeg_quality is not None
        else int(config.get('jpeg_quality', 95)))
    if interval <= 0:
        raise RuntimeError('保存间隔必须大于 0')
    if not 1 <= quality <= 100:
        raise RuntimeError('JPEG 质量必须在 1 到 100 之间')
    output = (
        args.data_root.expanduser().resolve()
        / args.profile / UNCLASSIFIED_DIR)
    output.mkdir(parents=True, exist_ok=True)
    print(f'输出目录：{output}')
    source = config['source_type']
    if source == 'v4l2':
        count = capture_v4l2(
            config, output, args.profile, interval, quality)
    else:
        count = capture_ros_topic(
            config, output, args.profile, interval, quality)
    print(f'采集结束，共保存 {count} 张')


if __name__ == '__main__':
    main()
