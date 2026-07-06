from pathlib import Path

import cv2
import numpy as np

from rgb_camera_receiver.classifier import (
    annotate,
    detect_candidates,
    load_config,
    select_winner,
)


PACKAGE = Path(__file__).resolve().parents[1]
CONFIG = load_config(str(PACKAGE / 'config' / 'detector.yaml'))


def color_bgr(name: str, saturation: int = 220, value: int = 180):
    model = next(item for item in CONFIG.colors if item.name == name)
    pixel = np.uint8([[[int(round(model.hue_center)), saturation, value]]])
    return tuple(int(x) for x in cv2.cvtColor(pixel, cv2.COLOR_HSV2BGR)[0, 0])


def dotted_strip(bgr=None):
    if bgr is None:
        bgr = color_bgr('BLUE')
    image = np.zeros((240, 360, 3), dtype=np.uint8)
    for index in range(24):
        cv2.circle(image, (45 + index * 10, 80 + index // 8), 3, bgr, -1)
    return image


def test_detects_single_blue_dot_train():
    candidates = detect_candidates(dotted_strip(), CONFIG)
    winner = select_winner(candidates, CONFIG)
    assert winner is not None
    assert winner.color == 'BLUE'


def test_rejects_short_colored_bar():
    image = np.zeros((240, 360, 3), dtype=np.uint8)
    cv2.rectangle(image, (35, 100), (95, 108), color_bgr('GREEN'), -1)
    assert detect_candidates(image, CONFIG) == []


def test_accepts_long_merged_led_strip():
    image = np.zeros((240, 360, 3), dtype=np.uint8)
    cv2.line(image, (45, 130), (300, 125), color_bgr('GREEN'), 4)
    candidates = detect_candidates(image, CONFIG)
    winner = select_winner(candidates, CONFIG)
    assert winner is not None
    assert winner.color == 'GREEN'


def test_annotation_keeps_original_shape():
    image = dotted_strip(color_bgr('RED'))
    candidates = detect_candidates(image, CONFIG)
    rendered = annotate(image, candidates, select_winner(candidates, CONFIG))
    assert rendered.shape == image.shape


def test_complete_calibration_dataset():
    dataset = PACKAGE.parents[2] / 'camera_capture'
    if not dataset.exists():
        return
    counts = 0
    for expected in ('BLUE', 'CYAN', 'GREEN', 'PURPLE', 'RED', 'NONE'):
        for path in sorted((dataset / expected).glob('*.jpg')):
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            candidates = detect_candidates(image, CONFIG)
            winner = select_winner(candidates, CONFIG)
            if expected == 'NONE':
                assert candidates == [], path
            else:
                assert winner is not None, path
                assert winner.color == expected, path
                wrong = [item for item in candidates if item.color != expected]
                assert wrong == [], path
            counts += 1
    assert counts == 192
