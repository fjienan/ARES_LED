from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'camera_profile',
            default_value='usb_rgb_1',
            description='R2 摄像头 profile：usb_rgb_1 或 usb_rgb_2。'),
        Node(
            package='rgb_camera_receiver',
            executable='rgb_camera_receiver',
            name='rgb_camera_receiver',
            output='screen',
            parameters=[
                PathJoinSubstitution([
                    FindPackageShare('rgb_camera_receiver'),
                    'config',
                    'cameras',
                    LaunchConfiguration('camera_profile'),
                    'receiver.yaml',
                ]),
                {'camera_profile': LaunchConfiguration('camera_profile')},
            ],
        ),
    ])
