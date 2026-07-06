"""R2 图像输入抽象。

当前检测只启用 V4L2 实现。Odin1 使用 ROS Image topic，其采集工具已经可用；
完成 Odin1 标定后可在此增加 RosTopicImageSource，而不修改检测算法。
"""

import glob
import os
from pathlib import Path
import subprocess
from typing import List, Sequence

import cv2


class OpenCvImageSource:
    """通过 OpenCV/V4L2 提供最新 USB 摄像头画面。"""

    def __init__(
            self, requested: str, width: int, height: int, fps: float,
            reject_keywords: Sequence[str], controls: str = ''):
        self.reject_keywords = tuple(
            item.strip().lower() for item in reject_keywords if item.strip())
        self.warnings = []
        self.capture, self.description = self._open(
            requested, width, height, fps)
        self.warnings.extend(self._apply_controls(
            self.description, controls))

    def read(self):
        return self.capture.read()

    def close(self) -> None:
        self.capture.release()

    def _open(self, requested: str, width: int, height: int, fps: float):
        failures: List[str] = []
        for device in self._candidates(requested):
            capture = cv2.VideoCapture(device, cv2.CAP_V4L2)
            if not capture.isOpened():
                failures.append(device)
                capture.release()
                continue
            capture.set(
                cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            capture.set(cv2.CAP_PROP_FPS, fps)
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            ok, frame = capture.read()
            if (
                    ok and frame is not None and frame.ndim == 3 and
                    frame.shape[2] == 3):
                return capture, device
            failures.append(device)
            capture.release()
        raise RuntimeError(
            f'cannot open camera {requested}; attempted={failures}')

    def _candidates(self, requested: str) -> List[str]:
        if requested != 'auto':
            return [requested]
        devices = (
            sorted(glob.glob('/dev/v4l/by-id/*-video-index0')) +
            sorted(glob.glob('/dev/v4l/by-id/*')) +
            sorted(glob.glob('/dev/video*')))
        result = []
        seen = set()
        for device in devices:
            resolved = os.path.realpath(device)
            base = os.path.basename(resolved)
            name_path = Path('/sys/class/video4linux') / base / 'name'
            name = (
                name_path.read_text(errors='replace').strip()
                if name_path.exists() else '')
            if any(word in name.lower() for word in self.reject_keywords):
                continue
            if resolved in seen:
                continue
            seen.add(resolved)
            result.append(device)
        return result

    @staticmethod
    def _apply_controls(device: str, controls: str):
        warnings = []
        if not controls.strip():
            return warnings
        executable = Path('/usr/bin/v4l2-ctl')
        if not executable.exists():
            return ['v4l2_controls 已设置，但未安装 v4l2-ctl']
        for assignment in controls.split(','):
            assignment = assignment.strip()
            if not assignment:
                continue
            result = subprocess.run(
                [str(executable), '--device', device, '--set-ctrl', assignment],
                text=True, capture_output=True, check=False)
            if result.returncode != 0:
                message = (result.stderr or result.stdout).strip()
                warnings.append(
                    f'camera control {assignment} failed: {message}')
        return warnings
