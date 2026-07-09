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


def _receiver_params(package_share: Path, profile: str):
    receiver_path = package_share / 'config' / 'cameras' / profile / 'receiver.yaml'
    with receiver_path.open('r', encoding='utf-8') as stream:
        raw = yaml.safe_load(stream) or {}
    return dict(
        (raw.get('rgb_camera_receiver', {}) or {}).get('ros__parameters', {}) or {})


def _copy_slot_overrides(params, slot):
    int_keys = {
        'frame_width',
        'frame_height',
        'camera_buffer_size',
        'reset_command_id',
        'confirmation_window',
        'confirmation_required',
    }
    float_keys = {
        'scan_rate_hz',
        'camera_fps',
        'processing_scale',
        'preview_scale',
        'positive_save_interval_sec',
        'max_confirm_latency_sec',
        'protocol_winner_margin',
    }
    bool_keys = {
        'show_preview',
        'save_positive_images',
    }
    string_keys = {
        'camera_fourcc',
        'positive_capture_dir',
        'v4l2_controls',
        'detector_config',
        'protocol_config',
    }
    for key in int_keys:
        if key in slot:
            params[key] = int(slot[key])
    for key in float_keys:
        if key in slot:
            params[key] = float(slot[key])
    for key in bool_keys:
        if key in slot:
            params[key] = _as_bool(slot[key], bool(params.get(key, False)))
    for key in string_keys:
        if key in slot:
            params[key] = str(slot[key])


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
        params = _receiver_params(package_share, profile)
        params.update({
            'camera_profile': profile,
            'camera_device': str(slot['device']),
            'camera_required': _as_bool(slot.get('required', False), False),
            'output_topic': topic,
            'publish_reset_commands': True,
        })
        _copy_slot_overrides(params, slot)
        actions.append(Node(
            package='rgb_camera_receiver',
            executable='rgb_camera_receiver',
            name=f'rgb_camera_receiver_{name}',
            output='screen',
            parameters=[params],
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
