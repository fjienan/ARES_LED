from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    share = Path(get_package_share_directory('rgb_camera_receiver'))
    return LaunchDescription([
        Node(
            package='rgb_camera_receiver',
            executable='rgb_camera_receiver',
            name='rgb_camera_receiver',
            output='screen',
            parameters=[str(share / 'config' / 'receiver.yaml')],
        ),
    ])
