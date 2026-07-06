from pathlib import Path

import pytest
import yaml

from rgb_camera_receiver.detectors import create_detector_backend


PACKAGE = Path(__file__).resolve().parents[1]
CAMERAS = PACKAGE / 'config' / 'cameras'


def load_yaml(path: Path):
    with path.open('r', encoding='utf-8') as stream:
        return yaml.safe_load(stream)


def test_camera_profiles_have_isolated_configs():
    for profile in ('usb_rgb', 'odin1'):
        root = CAMERAS / profile
        assert (root / 'capture.yaml').is_file()
        assert (root / 'receiver.yaml').is_file()
        assert (root / 'detector.yaml').is_file()


def test_only_usb_detector_is_currently_calibrated():
    assert load_yaml(CAMERAS / 'usb_rgb' / 'detector.yaml')['calibrated'] is True
    assert load_yaml(CAMERAS / 'odin1' / 'detector.yaml')['calibrated'] is False


def test_unknown_detector_backend_is_rejected():
    with pytest.raises(RuntimeError, match='unsupported detector backend'):
        create_detector_backend(
            'future_odin_algorithm',
            CAMERAS / 'odin1' / 'detector.yaml')
