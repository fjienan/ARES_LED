from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _as_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {'1', 'true', 'yes', 'on'}


def _launch_setup(context):
    package_share = Path(get_package_share_directory('rgb_camera_receiver'))
    config_path = Path(LaunchConfiguration('dual_config').perform(context)).expanduser()
    if not config_path.is_absolute():
        config_path = package_share / config_path
    with config_path.open('r', encoding='utf-8') as stream:
        config = yaml.safe_load(stream) or {}

    actions = []
    input_topics = []
    camera_slots = config.get('camera_slots', {})
    for name, slot in camera_slots.items():
        if not _as_bool(slot.get('enabled', True), True):
            continue
        profile = str(slot['profile'])
        topic = f'/rgb_camera_receiver/{name}/confirmed_id'
        input_topics.append(topic)
        params = {
            'camera_profile': profile,
            'camera_device': str(slot['device']),
            'camera_required': _as_bool(slot.get('required', False), False),
            'show_preview': _as_bool(slot.get('show_preview', True), True),
            'output_topic': topic,
            'publish_reset_commands': True,
            'positive_capture_dir': (
                f'~/Desktop/LED/camera_capture_positive_{profile}_protocol'),
        }
        if 'frame_width' in slot:
            params['frame_width'] = int(slot['frame_width'])
        if 'frame_height' in slot:
            params['frame_height'] = int(slot['frame_height'])
        if 'scan_rate_hz' in slot:
            params['scan_rate_hz'] = float(slot['scan_rate_hz'])
        if 'processing_scale' in slot:
            params['processing_scale'] = float(slot['processing_scale'])
        actions.append(Node(
            package='rgb_camera_receiver',
            executable='rgb_camera_receiver',
            name=f'rgb_camera_receiver_{name}',
            output='screen',
            parameters=[
                str(package_share / 'config' / 'cameras' / profile / 'receiver.yaml'),
                params,
            ],
        ))

    arbiter = config.get('arbiter', {})
    actions.append(Node(
        package='rgb_camera_receiver',
        executable='rgb_command_arbiter',
        name='rgb_command_arbiter',
        output='screen',
        parameters=[{
            'input_topics': input_topics,
            'output_topic': str(arbiter.get('output_topic', '/aruco_comm/rx_id')),
            'reset_command_id': int(arbiter.get('reset_command_id', 0)),
        }],
    ))
    return actions


def generate_launch_description():
    default_config = str(Path(
        get_package_share_directory('rgb_camera_receiver')) / 'config' /
        'dual_receiver.yaml')
    return LaunchDescription([
        DeclareLaunchArgument(
            'dual_config',
            default_value=default_config,
            description='双摄像头接收配置 YAML。'),
        OpaqueFunction(function=_launch_setup),
    ])
