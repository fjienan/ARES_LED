"""相机配置路径和标定状态检查。"""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
import yaml


CAMERA_PROFILES = ('usb_rgb', 'odin1')


def config_root() -> Path:
    return (
        Path(get_package_share_directory('rgb_camera_receiver')) /
        'config' / 'cameras')


def detector_config_path(profile: str) -> Path:
    if profile not in CAMERA_PROFILES:
        raise ValueError(
            f'unknown camera profile {profile}; expected one of {CAMERA_PROFILES}')
    path = config_root() / profile / 'detector.yaml'
    if not path.is_file():
        raise FileNotFoundError(f'detector config not found: {path}')
    return path


def require_calibrated_detector(path: Path, profile: str):
    with path.open('r', encoding='utf-8') as stream:
        raw = yaml.safe_load(stream) or {}
    if not bool(raw.get('calibrated', False)):
        raise RuntimeError(
            f'camera profile {profile} has no calibrated detector; '
            f'collect and label its dataset first')
    return raw
