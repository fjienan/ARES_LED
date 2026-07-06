from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    sender_share = Path(get_package_share_directory('rgb_led_sender'))
    return LaunchDescription([
        # Legacy FT232H/action path only. The default R1 path uses serial WLED JSON.
        DeclareLaunchArgument('start_led_controller', default_value='false'),
        Node(
            package='led_controller',
            executable='led_action_server',
            name='led_action_server',
            output='screen',
            condition=IfCondition(LaunchConfiguration('start_led_controller')),
        ),
        Node(
            package='rgb_led_sender',
            executable='rgb_led_sender',
            name='rgb_led_sender',
            output='screen',
            parameters=[str(sender_share / 'config/sender.yaml')],
        ),
    ])
