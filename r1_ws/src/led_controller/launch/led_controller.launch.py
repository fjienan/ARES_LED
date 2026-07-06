from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from pathlib import Path


def generate_launch_description():
    config = Path(get_package_share_directory('led_controller')) / 'config' / 'led_controller.yaml'

    return LaunchDescription([
        Node(
            package='led_controller',
            executable='led_action_server',
            name='led_action_server',
            output='screen',
            parameters=[str(config)],
        )
    ])
