from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node


def generate_launch_description():
    profile = LaunchConfiguration('camera_profile')
    receiver_config = PathJoinSubstitution([
        FindPackageShare('rgb_camera_receiver'),
        'config', 'cameras', profile, 'receiver.yaml',
    ])
    return LaunchDescription([
        DeclareLaunchArgument(
            'camera_profile',
            default_value='usb_rgb',
            description='相机配置名称；当前检测仅支持 usb_rgb'),
        Node(
            package='rgb_camera_receiver',
            executable='rgb_camera_receiver',
            name='rgb_camera_receiver',
            output='screen',
            parameters=[receiver_config],
        ),
    ])
