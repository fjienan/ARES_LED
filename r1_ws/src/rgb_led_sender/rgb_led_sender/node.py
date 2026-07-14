import glob
import os
import select
import termios
from typing import Optional

import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from std_msgs.msg import Int32

from rgb_comm_protocol import FixedColorProtocol

from .mapping import (
    build_triplet_segment_specs,
    build_wled_idle_effect_json,
    build_wled_state_json,
)


_BAUD_RATES = {
    9600: termios.B9600,
    19200: termios.B19200,
    38400: termios.B38400,
    57600: termios.B57600,
    115200: termios.B115200,
    230400: termios.B230400,
    460800: termios.B460800,
    921600: termios.B921600,
}

_IDLE_COMMAND_ID = 0


class LineSerial:
    def __init__(self, device: str, baudrate: int, timeout_sec: float) -> None:
        self.device = device
        self.baudrate = baudrate
        self.timeout_sec = max(timeout_sec, 0.0)
        self.fd: Optional[int] = None

    def open(self) -> None:
        if self.fd is not None:
            return
        if self.baudrate not in _BAUD_RATES:
            raise ValueError(f'unsupported serial baudrate: {self.baudrate}')
        fd = os.open(self.device, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
        try:
            attrs = termios.tcgetattr(fd)
            attrs[0] = 0
            attrs[1] = 0
            attrs[2] = termios.CS8 | termios.CREAD | termios.CLOCAL
            attrs[3] = 0
            attrs[4] = _BAUD_RATES[self.baudrate]
            attrs[5] = _BAUD_RATES[self.baudrate]
            attrs[6][termios.VMIN] = 0
            attrs[6][termios.VTIME] = 0
            termios.tcsetattr(fd, termios.TCSANOW, attrs)
            termios.tcflush(fd, termios.TCIOFLUSH)
        except Exception:
            os.close(fd)
            raise
        self.fd = fd

    def close(self) -> None:
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

    def write_line(self, line: str) -> str:
        self.open()
        assert self.fd is not None
        os.write(self.fd, line.encode('ascii'))
        if self.timeout_sec <= 0.0:
            return ''

        ready, _, _ = select.select([self.fd], [], [], self.timeout_sec)
        if not ready:
            return ''
        data = os.read(self.fd, 256)
        return data.decode('ascii', errors='replace').strip()


def _find_serial_device(configured: str) -> Optional[str]:
    if configured and configured != 'auto':
        return configured
    patterns = [
        '/dev/serial/by-id/*',
        '/dev/ttyACM*',
        '/dev/ttyUSB*',
    ]
    preferred_keywords = ('wled', 'lolin', 'wemos', 'esp32', 'usb-dev')
    rejected_keywords = ('daplink', 'cmsis-dap')
    candidates = []
    for pattern in patterns:
        for device in sorted(glob.glob(pattern)):
            lower = device.lower()
            if any(keyword in lower for keyword in rejected_keywords):
                continue
            candidates.append(device)
    for device in candidates:
        lower = device.lower()
        if any(keyword in lower for keyword in preferred_keywords):
            return device
    if candidates:
        return candidates[0]
    return None


class RgbLedSender(Node):
    def __init__(self) -> None:
        super().__init__('rgb_led_sender')
        topic = self.declare_parameter('input_topic', '/aruco_comm/tx_id').value
        self.transport = str(self.declare_parameter('transport', 'serial').value).lower()
        self.pixel_count = int(self.declare_parameter('pixel_count', 6).value)
        retry_period = float(self.declare_parameter('retry_period_sec', 0.5).value)
        if self.pixel_count <= 0:
            raise ValueError('pixel_count must be positive')

        default_colors = os.path.join(
            get_package_share_directory('rgb_led_sender'), 'config', 'colors.yaml')
        colors_path = str(self.declare_parameter(
            'colors_config', default_colors).value) or default_colors
        self.protocol = FixedColorProtocol(colors_path=colors_path)
        self.low_segments = [
            int(v) for v in self.declare_parameter('low_segments', [0, 1, 2]).value]
        self.high_segments = [
            int(v) for v in self.declare_parameter('high_segments', [3, 4, 5]).value]
        self.low_brightness = float(
            self.declare_parameter('low_brightness', 6.0).value)
        self.high_brightness = float(
            self.declare_parameter('high_brightness', 60.0).value)
        self.low_reverse_order = bool(
            self.declare_parameter('low_reverse_order', False).value)
        self.high_reverse_order = bool(
            self.declare_parameter('high_reverse_order', False).value)
        self.segment_starts = [
            int(v) for v in self.declare_parameter(
                'segment_starts', list(range(self.pixel_count))).value]
        self.segment_stops = [
            int(v) for v in self.declare_parameter(
                'segment_stops', list(range(1, self.pixel_count + 1))).value]
        self.brightness_mode = str(self.declare_parameter(
            'brightness_mode', 'segment_bri').value)
        self.wled_master_brightness = float(
            self.declare_parameter('wled_master_brightness', 255.0).value)
        self.initial_command_id = int(
            self.declare_parameter('initial_command_id', -1).value)
        self.idle_effect_enabled = bool(
            self.declare_parameter('idle_effect_enabled', True).value)
        self.idle_effect_brightness = float(
            self.declare_parameter('idle_effect_brightness', 20.0).value)
        self.idle_effect_fx = int(
            self.declare_parameter('idle_effect_fx', 8).value)
        self.idle_effect_speed = int(
            self.declare_parameter('idle_effect_speed', 64).value)
        self.idle_effect_intensity = int(
            self.declare_parameter('idle_effect_intensity', 128).value)
        self.idle_command_delay_sec = max(0.0, float(
            self.declare_parameter('idle_command_delay_sec', 0.1).value))
        self.display_segments = build_triplet_segment_specs(
            self.protocol.code_length,
            self.low_segments,
            self.low_brightness,
            self.low_reverse_order,
            self.high_segments,
            self.high_brightness,
            self.high_reverse_order,
            self.segment_starts,
            self.segment_stops,
        )
        self.pending_id: Optional[int] = None
        self.active_command: Optional[int] = None
        self.idle_effect_payload = ''
        if self.idle_effect_enabled:
            self.idle_effect_payload = build_wled_idle_effect_json(
                self.display_segments,
                self.pixel_count,
                self.idle_effect_brightness,
                self.idle_effect_fx,
                self.idle_effect_speed,
                self.idle_effect_intensity,
                self.wled_master_brightness,
            )

        self.serial: Optional[LineSerial] = None
        self.serial_device_config = ''
        self.serial_baudrate = 115200
        self.serial_timeout_sec = 0.05

        if self.transport != 'serial':
            raise ValueError("transport must be 'serial'")
        self._init_serial_transport()
        self.create_subscription(Int32, topic, self._on_command, 10)
        self.create_timer(max(retry_period, 0.05), self._dispatch)
        self._idle_timer = self.create_timer(
            max(self.idle_command_delay_sec, 0.001), self._dispatch_delayed_idle)
        self._idle_timer.cancel()
        self.idle_delay_pending = False
        self._queue_initial_command()
        self.get_logger().info(
            f'RGB LED sender ready: topic={topic}, transport={self.transport}, '
            f'low_segments={self.low_segments}, high_segments={self.high_segments}, '
            f'low_reverse={self.low_reverse_order}, '
            f'high_reverse={self.high_reverse_order}, pixel_count={self.pixel_count}, '
            f'idle_effect={self.idle_effect_enabled}, idle_command={_IDLE_COMMAND_ID}, '
            f'idle_delay={self.idle_command_delay_sec:g}s')

    def _init_serial_transport(self) -> None:
        self.serial_device_config = str(
            self.declare_parameter('serial_device', 'auto').value)
        self.serial_baudrate = int(
            self.declare_parameter('serial_baudrate', 115200).value)
        self.serial_timeout_sec = float(
            self.declare_parameter('serial_timeout_sec', 0.05).value)

    def _queue_initial_command(self) -> None:
        if self.initial_command_id < 0:
            if self.idle_effect_enabled:
                self._queue_idle_command()
            return
        if self.initial_command_id == _IDLE_COMMAND_ID:
            if self.idle_effect_enabled:
                self._queue_idle_command()
            else:
                self.get_logger().warning(
                    'initial command 0 requests the idle effect, but idle_effect_enabled is false')
            return
        if self.protocol.encode_symbols(self.initial_command_id) is None:
            self.get_logger().warning(
                f'initial command ID {self.initial_command_id} is not in RGB protocol; skipped')
            return
        self.pending_id = self.initial_command_id

    def _on_command(self, message: Int32) -> None:
        command = int(message.data)
        if command == _IDLE_COMMAND_ID:
            if not self.idle_effect_enabled:
                self.get_logger().warning(
                    'command 0 requests the idle effect, but idle_effect_enabled is false; ignored')
                return
            if command == self.pending_id or command == self.active_command:
                return
            self._queue_idle_command()
            return
        elif self.protocol.encode_symbols(command) is None:
            self.get_logger().warning(f'command ID {command} is not in RGB protocol; ignored')
            return

        self._cancel_idle_delay()
        if command == self.pending_id:
            return
        if command == self.active_command:
            self.pending_id = None
            return
        self.pending_id = command
        self._dispatch()

    def _queue_idle_command(self) -> None:
        self.pending_id = _IDLE_COMMAND_ID
        if self.idle_command_delay_sec <= 0.0:
            self.idle_delay_pending = False
            self._dispatch()
            return
        self.idle_delay_pending = True
        self._idle_timer.reset()

    def _cancel_idle_delay(self) -> None:
        if self.idle_delay_pending:
            self._idle_timer.cancel()
            self.idle_delay_pending = False

    def _dispatch_delayed_idle(self) -> None:
        self._idle_timer.cancel()
        if self.pending_id != _IDLE_COMMAND_ID:
            self.idle_delay_pending = False
            return
        self.idle_delay_pending = False
        self._dispatch()

    def _dispatch(self) -> None:
        self._dispatch_serial()

    def _dispatch_serial(self) -> None:
        if self.pending_id is None:
            return

        command = self.pending_id
        if command == _IDLE_COMMAND_ID:
            if self.idle_delay_pending:
                return
            self._dispatch_idle_serial()
            return
        code = self.protocol.encode_symbols(command)
        rgb = self.protocol.encode_rgb(command)
        if code is None or rgb is None:
            self.pending_id = None
            return
        payload = build_wled_state_json(
            rgb,
            self.display_segments,
            self.pixel_count,
            self.brightness_mode,
            self.wled_master_brightness,
        )
        response = self._write_serial_payload(payload)
        if response is None:
            return

        self.pending_id = None
        self.active_command = command
        if response:
            self.get_logger().info(f'sent command {command}: {code}; WLED replied: {response}')
        else:
            self.get_logger().info(f'sent command {command}: {code}')

    def _dispatch_idle_serial(self) -> None:
        if self.pending_id != _IDLE_COMMAND_ID or not self.idle_effect_payload:
            return
        response = self._write_serial_payload(self.idle_effect_payload)
        if response is None:
            return

        self.pending_id = None
        self.active_command = _IDLE_COMMAND_ID
        if response:
            self.get_logger().info(
                f'sent command {_IDLE_COMMAND_ID}: idle WLED effect; WLED replied: {response}')
        else:
            self.get_logger().info(f'sent command {_IDLE_COMMAND_ID}: idle WLED effect')

    def _write_serial_payload(self, payload: str) -> Optional[str]:
        device = _find_serial_device(self.serial_device_config)
        if device is None:
            self.get_logger().warning('WLED serial device not found; retrying')
            return None
        if self.serial is None or self.serial.device != device:
            if self.serial is not None:
                self.serial.close()
            self.serial = LineSerial(device, self.serial_baudrate, self.serial_timeout_sec)

        try:
            response = self.serial.write_line(f'{payload}\n')
        except OSError as exc:
            self.get_logger().warning(f'failed to write WLED serial {device}: {exc}; retrying')
            if self.serial is not None:
                self.serial.close()
            return None
        return response

    def destroy_node(self) -> bool:
        if self.serial is not None:
            self.serial.close()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RgbLedSender()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
