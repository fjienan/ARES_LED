import glob
import os
import subprocess
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Deque, List, Optional, Tuple

import cv2
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32

from rgb_comm_protocol import FixedColorProtocol

from .classifier import classifier_for_profile
from .protocol_decoder import (
    decode_protocol_candidates,
    load_pairing_config,
    protocol_color_symbols,
    scaled_candidate_crop_area,
    select_protocol_winner,
)
from .profiles import (
    DEFAULT_CAMERA_PROFILE,
    detector_config_path,
    require_calibrated_detector,
    validate_camera_profile,
)
from .three_segment import (
    annotate_three_segments,
    detect_three_segment_frame_old,
    load_three_segment_config,
)
from .three_segment_protocol import (
    protocol_candidates_from_triples,
    protocol_detection_from_three_segment,
    protocol_winner_from_triples,
)


def _read_camera_frame(capture):
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


def _preview_window_title(node_name: str, namespace: str, output_topic: str,
                          profile: str) -> str:
    for item in (node_name, namespace, output_topic):
        for part in reversed(item.strip('/').split('/')):
            if part.startswith('rgb_camera_receiver_'):
                suffix = part.removeprefix('rgb_camera_receiver_')
                if suffix:
                    return f'R2 LED strip detector - {suffix} ({profile})'
            if part.startswith('camera_'):
                return f'R2 LED strip detector - {part} ({profile})'
    return f'R2 LED strip detector - {profile}'


class LedStripReceiver(Node):
    """与协议无关的灯带检测器实时摄像头前端。"""

    def __init__(self) -> None:
        super().__init__('rgb_camera_receiver')
        profile = validate_camera_profile(str(self.declare_parameter(
            'camera_profile', DEFAULT_CAMERA_PROFILE).value))
        self.profile = profile
        self.classifier = classifier_for_profile(profile)
        self.requested_device = str(self.declare_parameter('camera_device', 'auto').value)
        self.frame_width = int(self.declare_parameter('frame_width', 1280).value)
        self.frame_height = int(self.declare_parameter('frame_height', 720).value)
        self.scan_rate_hz = max(float(
            self.declare_parameter('scan_rate_hz', 30.0).value), 1.0)
        self.camera_fps = max(float(
            self.declare_parameter('camera_fps', 30.0).value), 1.0)
        self.camera_fourcc = str(
            self.declare_parameter('camera_fourcc', '').value).strip().upper()
        self.camera_buffer_size = int(
            self.declare_parameter('camera_buffer_size', 0).value)
        self.camera_required = bool(
            self.declare_parameter('camera_required', True).value)
        self.camera_retry_period = max(float(self.declare_parameter(
            'camera_retry_period_sec', 1.0).value), 0.1)
        self.last_camera_retry_time = 0.0
        self.preview = bool(self.declare_parameter('show_preview', True).value) and bool(
            os.environ.get('DISPLAY'))
        self.preview_scale = max(float(
            self.declare_parameter('preview_scale', 0.5).value), 0.05)
        processing_scale_param = float(
            self.declare_parameter('processing_scale', 0.0).value)
        self.save_positive_images = bool(
            self.declare_parameter('save_positive_images', True).value)
        capture_dir = str(self.declare_parameter(
            'positive_capture_dir',
            '~/Desktop/LED/camera_capture_positive').value)
        self.positive_capture_dir = Path(capture_dir).expanduser()
        self.positive_save_interval = max(float(self.declare_parameter(
            'positive_save_interval_sec', 1.0).value), 0.0)
        self.positive_active = False
        self.last_positive_save_time: Optional[float] = None
        if self.save_positive_images:
            self.positive_capture_dir.mkdir(parents=True, exist_ok=True)
        reject = str(self.declare_parameter(
            'camera_reject_keywords', 'Integrated,Chicony').value)
        self.reject_keywords = tuple(
            item.strip().lower() for item in reject.split(',') if item.strip())
        self.controls = str(self.declare_parameter('v4l2_controls', '').value)
        default_detector = str(detector_config_path(profile))
        configured_detector = str(self.declare_parameter(
            'detector_config', default_detector).value).strip()
        config_path = str(require_calibrated_detector(
            Path(configured_detector or default_detector).expanduser(),
            profile))
        self.config = self.classifier.load_config(config_path)
        # processing_scale <= 0 表示跟随 detector.yaml，保证实时节点、离线评估、
        # 非 ROS 实时脚本使用同一套检测尺度；显式传入正数时才覆盖。
        self.processing_scale = (
            min(1.0, max(processing_scale_param, 0.1))
            if processing_scale_param > 0.0
            else min(1.0, max(float(self.config.processing_scale), 0.1))
        )
        protocol_config = str(self.declare_parameter(
            'protocol_config', '').value)
        self.protocol = FixedColorProtocol(
            config_path=protocol_config if protocol_config else None)
        self.pairing_config = load_pairing_config(config_path)
        self.three_segment_config = (
            load_three_segment_config(config_path)
            if self.profile == 'usb_rgb_2'
            else None
        )
        self.protocol_margin = max(float(self.declare_parameter(
            'protocol_winner_margin', 1.2).value), 1.0)
        output_topic = str(self.declare_parameter(
            'output_topic', '/aruco_comm/rx_id').value)
        self.preview_window_title = _preview_window_title(
            self.get_name(), self.get_namespace(), output_topic, self.profile)
        self.publisher = self.create_publisher(Int32, output_topic, 10)
        self.reset_command_id = int(self.declare_parameter(
            'reset_command_id', 0).value)
        self.publish_reset_commands = bool(self.declare_parameter(
            'publish_reset_commands', False).value)
        self.confirmation_window = max(int(self.declare_parameter(
            'confirmation_window', 2).value), 1)
        self.confirmation_required = max(int(self.declare_parameter(
            'confirmation_required', 2).value), 1)
        self.confirmation_required = min(
            self.confirmation_required, self.confirmation_window)
        self.max_confirm_latency_sec = max(float(self.declare_parameter(
            'max_confirm_latency_sec', 0.20).value), 0.0)
        self.history: Deque[Tuple[float, Optional[int]]] = deque(
            maxlen=self.confirmation_window)
        self.last_emitted_id: Optional[int] = None
        self.capture = None
        self.device = ''
        self.capture_lock = threading.Lock()
        self.capture_stop = threading.Event()
        self.capture_thread: Optional[threading.Thread] = None
        self.latest_frame = None
        self.latest_frame_seq = 0
        self.last_processed_frame_seq = 0
        self._ensure_camera(force=True)
        self.last_label = ''
        self.create_timer(1.0 / self.scan_rate_hz, self._scan)
        self.get_logger().info(
            f'R2 LED vision ready: profile={profile}, camera={self.device or "not-open"}, '
            f'config={config_path}, '
            f'camera_fps={self.camera_fps:g}, scan_rate_hz={self.scan_rate_hz:g}, '
            f'fourcc={self.camera_fourcc or "default"}, '
            f'buffer={self.camera_buffer_size if self.camera_buffer_size > 0 else "default"}, '
            f'processing_scale={self.processing_scale:g}, '
            f'colors={[item.name for item in self.config.colors]}, '
            f'output={output_topic}, confirm={self.confirmation_required}/'
            f'{self.confirmation_window}, max_latency={self.max_confirm_latency_sec:g}s')

    def _scan(self) -> None:
        if not self._ensure_camera():
            return
        frame, frame_seq = self._latest_frame_snapshot()
        if frame is None:
            return
        if frame_seq == self.last_processed_frame_seq:
            return
        self.last_processed_frame_seq = frame_seq
        triples = []
        if self.profile == 'usb_rgb_2':
            assert self.three_segment_config is not None
            strong, weak, candidates, triples = detect_three_segment_frame_old(
                frame,
                self.classifier,
                self.config,
                self.processing_scale,
                self.three_segment_config)
            protocol_candidates = self._protocol_candidates_from_triples(triples)
            protocol_winner = self._protocol_winner_from_triples(triples)
            candidate_count_label = (
                f'strip_candidates={len(candidates)} '
                f'three_candidates={len(triples)} '
                f'strong={len(strong)} weak={len(weak)}')
        else:
            work = frame
            scale = self.processing_scale
            if scale < 1.0:
                work = cv2.resize(
                    frame, None, fx=scale, fy=scale,
                    interpolation=cv2.INTER_AREA)
            candidates = self.classifier.detect_protocol_candidates(
                work,
                self.config,
                protocol_color_symbols(self.protocol),
                self.pairing_config.min_coarse_colors,
                scaled_candidate_crop_area(self.pairing_config, scale))
            if scale < 1.0:
                inverse = 1.0 / scale
                candidates = [item.scaled(inverse) for item in candidates]
            protocol_candidates = decode_protocol_candidates(
                candidates, self.protocol, self.pairing_config)
            protocol_winner = select_protocol_winner(
                protocol_candidates, self.protocol_margin)
            candidate_count_label = f'strip_candidates={len(candidates)}'
        confirmed_count = self._update_protocol_state(protocol_winner)
        self._save_positive_frame(frame, protocol_winner)
        if protocol_winner is None:
            label = 'WAIT:NONE'
        else:
            label = f'WAIT:{protocol_winner.command_id}:{confirmed_count}'
        if label != self.last_label:
            if protocol_winner is None:
                self.get_logger().info(
                    f'command=NONE {candidate_count_label} '
                    f'protocol_candidates={len(protocol_candidates)} '
                    f'state=WAIT')
            else:
                different = next(
                    (item for item in protocol_candidates[1:]
                     if item.command_id != protocol_winner.command_id),
                    None)
                second = different.score if different is not None else 0.0
                margin = (
                    protocol_winner.score / max(second, 1e-9)
                    if second else float('inf'))
                self.get_logger().info(
                    f'command={protocol_winner.command_id} '
                    f'symbols={protocol_winner.symbols} '
                    f'confidence={protocol_winner.confidence:.3f} '
                    f'score={protocol_winner.score:.3f} margin={margin:.3f} '
                    f'confirm={confirmed_count}/{self.confirmation_required} '
                    f'state=WAIT')
            self.last_label = label
        if self.preview:
            if self.profile == 'usb_rgb_2':
                rendered = annotate_three_segments(frame, candidates, triples)
            else:
                rendered = annotate_three_segments(frame, [], protocol_candidates)
            if self.preview_scale != 1.0:
                rendered = cv2.resize(
                    rendered, None, fx=self.preview_scale, fy=self.preview_scale,
                    interpolation=cv2.INTER_AREA)
            cv2.imshow(self.preview_window_title, rendered)
            cv2.waitKey(1)

    def _protocol_detection_from_three_segment(self, item):
        return protocol_detection_from_three_segment(self.protocol, item)

    def _protocol_candidates_from_triples(self, triples):
        return protocol_candidates_from_triples(self.protocol, triples)

    def _protocol_winner_from_triples(self, triples):
        return protocol_winner_from_triples(self.protocol, triples)

    def _update_protocol_state(self, winner) -> int:
        now = time.monotonic()
        command_id = None if winner is None else int(winner.command_id)
        if command_id is not None:
            existing = {value for _, value in self.history if value is not None}
            if existing and existing != {command_id}:
                self.history.clear()
        self.history.append((now, command_id))
        if self.max_confirm_latency_sec > 0.0:
            cutoff = now - self.max_confirm_latency_sec
            while self.history and self.history[0][0] < cutoff:
                self.history.popleft()
        if winner is None:
            return 0

        confirmed_count = sum(1 for _, value in self.history if value == command_id)
        if (
                confirmed_count >= self.confirmation_required and
                command_id != self.last_emitted_id):
            if (
                    command_id == self.reset_command_id and
                    not self.publish_reset_commands):
                self.last_emitted_id = command_id
                self.history.clear()
                self.get_logger().info(
                    f'confirmed reset command {command_id}; state cleared locally')
                return confirmed_count
            message = Int32()
            message.data = command_id
            self.publisher.publish(message)
            self.last_emitted_id = command_id
            self.history.clear()
            self.get_logger().info(
                f'published confirmed command {command_id}: '
                f'{winner.symbols}')
        return confirmed_count

    def _save_positive_frame(self, frame, winner) -> None:
        if winner is None:
            self.positive_active = False
            return
        if not self.save_positive_images:
            return
        now = time.monotonic()
        should_save = (
            not self.positive_active or
            self.last_positive_save_time is None or
            now - self.last_positive_save_time >= self.positive_save_interval
        )
        self.positive_active = True
        if not should_save:
            return
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        symbols = ''.join(symbol[0] for symbol in winner.symbols)
        filename = (
            f'{symbols.lower()}_{timestamp}_'
            f'conf{winner.confidence:.3f}_score{winner.score:.3f}.jpg'
        )
        path = self.positive_capture_dir / filename
        if cv2.imwrite(str(path), frame):
            self.last_positive_save_time = now
            self.get_logger().info(f'saved positive frame: {path}')
        else:
            self.get_logger().error(f'failed to save positive frame: {path}')

    def _latest_frame_snapshot(self):
        with self.capture_lock:
            return self.latest_frame, self.latest_frame_seq

    def _open_camera(self, requested: str, width: int, height: int, fps: float):
        failures: List[str] = []
        for device in self._camera_candidates(requested):
            capture = cv2.VideoCapture(device, cv2.CAP_V4L2)
            if not capture.isOpened():
                failures.append(device)
                capture.release()
                continue
            if self.camera_fourcc:
                if len(self.camera_fourcc) == 4:
                    capture.set(
                        cv2.CAP_PROP_FOURCC,
                        cv2.VideoWriter_fourcc(*self.camera_fourcc))
                else:
                    self.get_logger().warning(
                        f'ignoring invalid camera_fourcc={self.camera_fourcc!r}')
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            capture.set(cv2.CAP_PROP_FPS, fps)
            if self.camera_buffer_size > 0:
                capture.set(cv2.CAP_PROP_BUFFERSIZE, self.camera_buffer_size)
            ok, frame = _read_camera_frame(capture)
            if ok and frame is not None and frame.ndim == 3 and frame.shape[2] == 3:
                return capture, device, frame
            failures.append(device)
            capture.release()
        raise RuntimeError(
            f'cannot open camera {requested}; attempted={failures}')

    def _ensure_camera(self, force: bool = False) -> bool:
        with self.capture_lock:
            capture_open = self.capture is not None
        if capture_open:
            self._start_capture_thread()
            return True
        now = time.monotonic()
        if not force and now - self.last_camera_retry_time < self.camera_retry_period:
            return False
        self.last_camera_retry_time = now
        try:
            capture, device, frame = self._open_camera(
                self.requested_device, self.frame_width, self.frame_height,
                self.camera_fps)
            self._apply_controls(device, self.controls)
            with self.capture_lock:
                self.capture = capture
                self.device = device
                self.latest_frame = frame
                self.latest_frame_seq += 1
            self._start_capture_thread()
            self.get_logger().info(
                f'camera opened: {device} camera_fps={self.camera_fps:g}')
            return True
        except RuntimeError as exc:
            if self.camera_required:
                raise
            self.get_logger().warning(f'{exc}; will retry')
            with self.capture_lock:
                self.capture = None
                self.device = ''
                self.latest_frame = None
            return False

    def _start_capture_thread(self) -> None:
        thread = self.capture_thread
        if thread is not None and thread.is_alive():
            return
        self.capture_stop.clear()
        self.capture_thread = threading.Thread(
            target=self._capture_loop,
            name=f'{self.get_name()}_capture',
            daemon=True)
        self.capture_thread.start()

    def _capture_loop(self) -> None:
        current = threading.current_thread()
        try:
            while not self.capture_stop.is_set():
                with self.capture_lock:
                    capture = self.capture
                if capture is None:
                    return
                ok, frame = _read_camera_frame(capture)
                if self.capture_stop.is_set():
                    return
                if not ok or frame is None or frame.ndim != 3 or frame.shape[2] != 3:
                    with self.capture_lock:
                        active = self.capture is capture
                        if active:
                            self.capture = None
                            self.device = ''
                            self.latest_frame = None
                    if active:
                        capture.release()
                        self.get_logger().warning(
                            'camera frame unavailable; will reopen')
                    return
                with self.capture_lock:
                    if self.capture is not capture:
                        return
                    self.latest_frame = frame
                    self.latest_frame_seq += 1
        finally:
            with self.capture_lock:
                if self.capture_thread is current:
                    self.capture_thread = None

    def _camera_candidates(self, requested: str) -> List[str]:
        if requested != 'auto':
            return [requested]
        devices = sorted(glob.glob('/dev/v4l/by-id/*')) + sorted(glob.glob('/dev/video*'))
        result = []
        seen = set()
        for device in devices:
            resolved = os.path.realpath(device)
            base = os.path.basename(resolved)
            name_path = Path('/sys/class/video4linux') / base / 'name'
            name = name_path.read_text(errors='replace').strip() if name_path.exists() else ''
            if any(keyword in name.lower() for keyword in self.reject_keywords):
                continue
            if resolved in seen:
                continue
            seen.add(resolved)
            result.append(device)
        return result

    def _apply_controls(self, device: str, controls: str) -> None:
        if not controls.strip():
            return
        executable = '/usr/bin/v4l2-ctl' if Path('/usr/bin/v4l2-ctl').exists() else None
        if executable is None:
            self.get_logger().warning('v4l2-ctl not installed; camera controls skipped')
            return
        for assignment in controls.split(','):
            assignment = assignment.strip()
            if not assignment:
                continue
            result = subprocess.run(
                [executable, '--device', device, '--set-ctrl', assignment],
                text=True, capture_output=True, check=False)
            if result.returncode != 0:
                message = (result.stderr or result.stdout).strip()
                self.get_logger().warning(
                    f'camera control {assignment} failed: {message}')

    def destroy_node(self):
        self.capture_stop.set()
        thread = self.capture_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
        with self.capture_lock:
            capture = self.capture
            self.capture = None
            self.device = ''
            self.latest_frame = None
        if capture is not None:
            capture.release()
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
            if thread.is_alive():
                self.get_logger().warning('capture thread did not stop cleanly')
        if self.preview:
            cv2.destroyAllWindows()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node: Optional[LedStripReceiver] = None
    try:
        node = LedStripReceiver()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
