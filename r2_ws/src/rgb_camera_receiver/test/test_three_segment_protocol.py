from pathlib import Path

import numpy as np

from rgb_comm_protocol import FixedColorProtocol

from rgb_camera_receiver.classifier import StripDetection
from rgb_camera_receiver.three_segment_protocol import (
    protocol_candidates_from_triples,
    protocol_winner_from_triples,
)
from rgb_camera_receiver.three_segment import ThreeSegmentDetection
from rgb_camera_receiver.three_segment import ThreeSegmentConfig, detect_three_segments


PACKAGE = Path(__file__).resolve().parents[1]
PROTOCOL = FixedColorProtocol(
    config_path=str(PACKAGE.parents[2] / 'shared' / 'src' /
                    'rgb_comm_protocol' / 'config' / 'rgb_protocol.yaml'))


def strip(
        color: str,
        center_x: float,
        center_y: float = 100.0,
        reverse_axis: bool = False,
        vertical: bool = False) -> StripDetection:
    length = 80.0
    width = 8.0
    if vertical:
        left_to_right = np.array([
            [center_x - width / 2, center_y - length / 2],
            [center_x - width / 2, center_y + length / 2],
            [center_x + width / 2, center_y + length / 2],
            [center_x + width / 2, center_y - length / 2],
        ], dtype=np.float32)
        right_to_left = np.array([
            [center_x + width / 2, center_y + length / 2],
            [center_x + width / 2, center_y - length / 2],
            [center_x - width / 2, center_y - length / 2],
            [center_x - width / 2, center_y + length / 2],
        ], dtype=np.float32)
    else:
        left_to_right = np.array([
            [center_x - length / 2, center_y - width / 2],
            [center_x + length / 2, center_y - width / 2],
            [center_x + length / 2, center_y + width / 2],
            [center_x - length / 2, center_y + width / 2],
        ], dtype=np.float32)
        right_to_left = np.array([
            [center_x + length / 2, center_y + width / 2],
            [center_x - length / 2, center_y + width / 2],
            [center_x - length / 2, center_y - width / 2],
            [center_x + length / 2, center_y - width / 2],
        ], dtype=np.float32)
    corners = right_to_left if reverse_axis else left_to_right
    return StripDetection(
        color=color,
        confidence=0.9,
        score=0.20,
        corners=corners,
        dot_count=8,
        length=length,
        residual=1.0,
        spacing_cv=0.1,
    )


def triple(symbols, *, ambiguous=False) -> ThreeSegmentDetection:
    segments = tuple(
        strip(color, 100.0 + index * 90.0)
        for index, color in enumerate(symbols)
    )
    return ThreeSegmentDetection(
        symbols=tuple(symbols),
        segments=segments,
        score=0.20,
        confidence=0.80,
        geometry_quality=0.90,
        angle_degrees=2.0,
        cross_distance=1.0,
        gap_ratio=1.0,
        length_ratio=1.0,
        center_distance_ratio=1.1,
        ambiguous=ambiguous,
    )


def test_non_ambiguous_three_segment_decodes_to_command():
    winner = protocol_winner_from_triples(
        PROTOCOL,
        [triple(('BLUE', 'RED', 'GREEN'))])

    assert winner is not None
    assert winner.command_id == 1
    assert winner.symbols == ('BLUE', 'RED', 'GREEN')


def test_three_segment_detector_orders_by_image_centers():
    triples = detect_three_segments(
        [
            strip('GREEN', 280.0, reverse_axis=True),
            strip('BLUE', 100.0, reverse_axis=True),
            strip('RED', 190.0, reverse_axis=True),
        ],
        ThreeSegmentConfig(min_three_score=0.01, winner_margin=1.0))

    assert triples
    assert triples[0].symbols == ('BLUE', 'RED', 'GREEN')


def test_three_segment_detector_uses_y_order_when_x_is_close():
    triples = detect_three_segments(
        [
            strip('GREEN', 104.0, 280.0, vertical=True),
            strip('RED', 98.0, 190.0, vertical=True),
            strip('BLUE', 100.0, 100.0, vertical=True),
        ],
        ThreeSegmentConfig(min_three_score=0.01, winner_margin=1.0))

    assert triples
    assert triples[0].symbols == ('BLUE', 'RED', 'GREEN')


def test_ambiguous_three_segment_is_not_publishable():
    winner = protocol_winner_from_triples(
        PROTOCOL,
        [triple(('BLUE', 'RED', 'GREEN'), ambiguous=True)])

    assert winner is None


def test_unknown_three_segment_is_not_publishable():
    winner = protocol_winner_from_triples(
        PROTOCOL,
        [triple(('RED', 'BLUE', 'GREEN'))])

    assert winner is None


def test_top_ambiguous_does_not_publish_second_candidate():
    triples = [
        triple(('BLUE', 'RED', 'GREEN'), ambiguous=True),
        triple(('RED', 'GREEN', 'BLUE')),
    ]

    assert protocol_candidates_from_triples(PROTOCOL, triples)[0].command_id == 3
    assert protocol_winner_from_triples(PROTOCOL, triples) is None
