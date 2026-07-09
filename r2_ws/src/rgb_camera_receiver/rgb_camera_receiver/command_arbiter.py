from typing import Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32


class CommandArbiter(Node):
    """合并多路摄像头确认结果，并处理 0 号内部重置命令。"""

    def __init__(self) -> None:
        super().__init__('rgb_command_arbiter')
        input_topics = [
            str(item) for item in self.declare_parameter(
                'input_topics',
                [
                    '/rgb_camera_receiver/camera_1/confirmed_id',
                    '/rgb_camera_receiver/camera_2/confirmed_id',
                ],
            ).value
        ]
        output_topic = str(self.declare_parameter(
            'output_topic', '/aruco_comm/rx_id').value)
        self.reset_command_id = int(self.declare_parameter(
            'reset_command_id', 0).value)
        self.last_published_id: Optional[int] = None
        self.publisher = self.create_publisher(Int32, output_topic, 10)
        self._command_subscriptions = []
        for topic in input_topics:
            self._command_subscriptions.append(
                self.create_subscription(
                    Int32,
                    topic,
                    lambda message, source=topic: self._on_command(message, source),
                    10,
                )
            )
        self.get_logger().info(
            f'command arbiter ready: inputs={input_topics}, output={output_topic}, '
            f'reset={self.reset_command_id}')

    def _on_command(self, message: Int32, source: str) -> None:
        command_id = int(message.data)
        if command_id == self.reset_command_id:
            self.last_published_id = None
            self.get_logger().info(f'reset command received from {source}; state cleared')
            return
        if command_id == self.last_published_id:
            self.get_logger().info(
                f'duplicate command {command_id} from {source}; suppressed')
            return
        output = Int32()
        output.data = command_id
        self.publisher.publish(output)
        self.last_published_id = command_id
        self.get_logger().info(f'published command {command_id} from {source}')


def main(args=None) -> None:
    rclpy.init(args=args)
    node: Optional[CommandArbiter] = None
    try:
        node = CommandArbiter()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
