import time
from threading import Lock
from typing import List, Sequence, Tuple

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_srvs.srv import Trigger

from led_controller.action import LedPattern
from led_controller_nodes.group_utils import resolve_groups, validate_groups

Color = Tuple[int, int, int]


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def clamp_color(values: Sequence[int], fallback: Color) -> Color:
    if len(values) < 3:
        return fallback
    return tuple(int(clamp(v, 0, 255)) for v in values[:3])


def scale_color(color: Color, brightness: float) -> Color:
    return tuple(int(channel * brightness) for channel in color)


class LedBackend:
    def __init__(self, led_count: int, brightness: float) -> None:
        self.led_count = led_count
        self.brightness = brightness
        self.name = 'unknown'

    def show(self, pixels: Sequence[Color]) -> None:
        raise NotImplementedError

    def clear(self) -> None:
        self.show([(0, 0, 0)] * self.led_count)


class SimulatedBackend(LedBackend):
    def __init__(self, led_count: int, brightness: float, logger) -> None:
        super().__init__(led_count, brightness)
        self.name = 'simulate'
        self._logger = logger
        self._frame = 0

    def show(self, pixels: Sequence[Color]) -> None:
        self._frame += 1
        if self._frame == 1 or self._frame % 40 == 0:
            lit = sum(1 for pixel in pixels if pixel != (0, 0, 0))
            preview = list(pixels[: min(8, len(pixels))])
            self._logger.info(f'sim frame={self._frame} lit={lit}/{self.led_count} preview={preview}')


class Ft232hSpiBackend(LedBackend):
    def __init__(self, led_count: int, brightness: float, pixel_order: str) -> None:
        super().__init__(led_count, brightness)
        self.name = 'hardware'
        import board
        import neopixel_spi

        order = self._resolve_pixel_order(pixel_order.upper())
        self._pixels = neopixel_spi.NeoPixel_SPI(
            board.SPI(),
            led_count,
            pixel_order=order,
            auto_write=False,
        )
        # Per-goal brightness is already applied by scale_color(). Keep the
        # driver's global multiplier neutral so hardware and simulation use
        # the same brightness semantics.
        self._pixels.brightness = 1.0

    def show(self, pixels: Sequence[Color]) -> None:
        for index, color in enumerate(pixels[: self.led_count]):
            self._pixels[index] = color
        self._pixels.show()

    @staticmethod
    def _resolve_pixel_order(pixel_order: str):
        channel_index = {'R': 0, 'G': 1, 'B': 2, 'W': 3}
        if set(pixel_order).issubset(channel_index) and len(pixel_order) in (3, 4):
            return tuple(channel_index[channel] for channel in pixel_order)
        return tuple(channel_index[channel] for channel in 'BRG')


class LedActionServer(Node):
    def __init__(self) -> None:
        super().__init__('led_action_server')
        self.declare_parameter('led_count', 60)
        self.declare_parameter('active_group_count', 0)
        self.declare_parameter('backend', 'simulate')
        self.declare_parameter('pixel_order', 'BRG')
        self.declare_parameter('default_brightness', 0.4)
        self.declare_parameter('action_name', 'led_pattern')
        self.declare_parameter('cancel_service_name', 'cancel_led_pattern')
        self.declare_parameter('frame_rate', 40.0)

        self.led_count = int(self.get_parameter('led_count').value)
        requested_group_count = int(self.get_parameter('active_group_count').value)
        max_group_count = max(1, self.led_count)
        self.group_count = (
            max_group_count
            if requested_group_count <= 0
            else max(1, min(requested_group_count, max_group_count))
        )
        self.frame_rate = float(self.get_parameter('frame_rate').value)
        self.default_brightness = float(self.get_parameter('default_brightness').value)
        backend_name = str(self.get_parameter('backend').value)
        pixel_order = str(self.get_parameter('pixel_order').value)
        action_name = str(self.get_parameter('action_name').value)
        cancel_service_name = str(self.get_parameter('cancel_service_name').value)

        self._backend = self._make_backend(backend_name, pixel_order)
        self._active_goal = False
        self._active_goal_lock = Lock()
        self._stop_requested = False
        self._action_callback_group = ReentrantCallbackGroup()
        self._action_server = ActionServer(
            self,
            LedPattern,
            action_name,
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=self._action_callback_group,
        )
        self._cancel_service = self.create_service(
            Trigger,
            cancel_service_name,
            self.cancel_pattern_callback,
        )
        self.get_logger().info(
            f'LED group controller ready: action=/{action_name}, '
            f'cancel_service=/{cancel_service_name}, backend={self._backend.name}, '
            f'leds={self.led_count}, active_groups={self.group_count}'
        )

    def destroy_node(self) -> bool:
        self._backend.clear()
        self._action_server.destroy()
        return super().destroy_node()

    def _make_backend(self, backend_name: str, pixel_order: str) -> LedBackend:
        brightness = clamp(self.default_brightness, 0.0, 1.0)
        if backend_name == 'hardware':
            try:
                return Ft232hSpiBackend(self.led_count, brightness, pixel_order)
            except Exception as exc:
                self.get_logger().error(f'failed to initialize FT232H backend: {exc}')
                self.get_logger().warning('falling back to simulate backend')
        return SimulatedBackend(self.led_count, brightness, self.get_logger())

    def goal_callback(self, goal_request) -> GoalResponse:
        group_error = validate_groups(goal_request.groups, self.group_count)
        if group_error is not None:
            self.get_logger().warning(f'rejecting LED goal: {group_error}')
            return GoalResponse.REJECT

        with self._active_goal_lock:
            if self._active_goal:
                self.get_logger().warning('rejecting goal because another LED command is active')
                return GoalResponse.REJECT
            # Reserve the controller before returning ACCEPT. With a reentrant
            # callback group, another goal request may otherwise arrive before
            # execute_callback starts and be accepted concurrently.
            self._active_goal = True
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle) -> CancelResponse:
        del goal_handle
        self.get_logger().info('cancel requested')
        return CancelResponse.ACCEPT

    def cancel_pattern_callback(self, request, response):
        del request
        self._stop_requested = True
        self._backend.clear()
        response.success = True
        response.message = 'cancel requested for current LED command'
        return response

    def execute_callback(self, goal_handle):
        try:
            return self._execute_goal(goal_handle)
        finally:
            with self._active_goal_lock:
                self._active_goal = False

    def _execute_goal(self, goal_handle):
        request = goal_handle.request
        groups = self._resolve_groups(request.groups)
        colors = self._resolve_colors(request.colors, len(groups))
        brightnesses = self._resolve_brightnesses(request.brightnesses, len(groups))
        blink_hz = self._resolve_blink_hz(request.blink_hz, len(groups))
        duration = max(float(request.duration), 0.0)

        result = LedPattern.Result()
        if not groups:
            goal_handle.abort()
            self._backend.clear()
            result.success = False
            result.message = 'no valid groups requested'
            return result

        self._stop_requested = False
        started = self.get_clock().now()
        step = 0
        frame_delay = 1.0 / max(self.frame_rate, 1.0)
        periods = [1.0 / frequency if frequency > 0.0 else 0.0 for frequency in blink_hz]
        scaled_colors = [
            scale_color(color, brightness)
            for color, brightness in zip(colors, brightnesses)
        ]

        self.get_logger().info(
            f'goal accepted groups={groups} colors={colors} brightnesses={brightnesses} '
            f'blink_hz={blink_hz} duration={duration:.2f}s'
        )

        try:
            while rclpy.ok():
                if goal_handle.is_cancel_requested or self._stop_requested:
                    goal_handle.canceled()
                    self._backend.clear()
                    result.success = False
                    result.message = 'LED command canceled'
                    return result

                elapsed = (self.get_clock().now() - started).nanoseconds / 1_000_000_000.0
                if duration > 0.0 and elapsed >= duration:
                    goal_handle.succeed()
                    self._backend.clear()
                    result.success = True
                    result.message = 'LED command completed'
                    return result

                pixels = [(0, 0, 0)] * self.led_count
                any_on = False
                for group, color, period in zip(groups, scaled_colors, periods):
                    on = True if period <= 0.0 else (elapsed % period) < (period / 2.0)
                    if on:
                        pixels[group] = color
                        any_on = True
                self._backend.show(pixels)

                if step % max(1, int(self.frame_rate / 4.0)) == 0:
                    feedback = LedPattern.Feedback()
                    feedback.percent_complete = 0.0 if duration <= 0.0 else elapsed / duration * 100.0
                    feedback.current_step = step
                    feedback.message = self._feedback_message(groups, any_on, blink_hz)
                    goal_handle.publish_feedback(feedback)

                step += 1
                time.sleep(frame_delay)
        except Exception as exc:
            self.get_logger().error(f'LED command failed: {exc}')
            goal_handle.abort()
            self._backend.clear()
            result.success = False
            result.message = f'LED command failed: {exc}'
            return result

        # The while loop only exits here when rclpy shut down mid-goal. Return a
        # proper Result so execute_callback never hands None back to the action
        # server (which would otherwise log an error and leave the goal hanging).
        goal_handle.abort()
        self._backend.clear()
        result.success = False
        result.message = 'LED command aborted during shutdown'
        return result

    def _resolve_groups(self, requested_groups: Sequence[int]) -> List[int]:
        return resolve_groups(requested_groups, self.group_count)

    def _resolve_colors(self, requested_colors: Sequence[int], group_count: int) -> List[Color]:
        if group_count <= 0:
            return []
        if len(requested_colors) < 3:
            return [(255, 255, 255)] * group_count
        if len(requested_colors) < group_count * 3:
            color = clamp_color(requested_colors, (255, 255, 255))
            return [color] * group_count

        colors = []
        for index in range(group_count):
            start = index * 3
            colors.append(clamp_color(requested_colors[start:start + 3], (255, 255, 255)))
        return colors

    def _resolve_brightnesses(self, requested_brightnesses: Sequence[float], group_count: int) -> List[float]:
        if group_count <= 0:
            return []
        if len(requested_brightnesses) == 0:
            return [clamp(self.default_brightness, 0.0, 1.0)] * group_count
        if len(requested_brightnesses) < group_count:
            brightness = clamp(float(requested_brightnesses[0]), 0.0, 1.0)
            return [brightness] * group_count
        return [
            clamp(float(brightness), 0.0, 1.0)
            for brightness in requested_brightnesses[:group_count]
        ]

    @staticmethod
    def _resolve_blink_hz(requested_blink_hz: Sequence[float], group_count: int) -> List[float]:
        if group_count <= 0:
            return []
        if len(requested_blink_hz) == 0:
            return [0.0] * group_count
        if len(requested_blink_hz) < group_count:
            return [max(float(requested_blink_hz[0]), 0.0)] * group_count
        return [max(float(frequency), 0.0) for frequency in requested_blink_hz[:group_count]]

    @staticmethod
    def _feedback_message(groups: Sequence[int], any_on: bool, blink_hz: Sequence[float]) -> str:
        group_text = 'all groups' if len(groups) == 0 else f'groups={list(groups)}'
        has_blink = any(frequency > 0.0 for frequency in blink_hz)
        mode = 'solid' if not has_blink else f'blink {"on" if any_on else "off"}'
        return f'{mode} {group_text}'


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LedActionServer()
    # execute_callback may run indefinitely. A second worker is required so
    # action cancellation can be processed by the reentrant callback group.
    executor = MultiThreadedExecutor(num_threads=2)
    try:
        rclpy.spin(node, executor=executor)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
