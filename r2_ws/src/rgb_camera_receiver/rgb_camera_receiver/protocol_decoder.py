import math
from dataclasses import dataclass
from functools import cmp_to_key
from itertools import combinations
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml

from rgb_comm_protocol import FixedColorProtocol

from .classifier import StripDetection


IMAGE_ORDER_X_TOLERANCE = 12.0


@dataclass(frozen=True)
class ProtocolGeometryConfig:
    min_command_score: float = 0.04
    max_angle_degrees: float = 18.0
    max_cross_distance_pixels: float = 35.0
    min_center_distance_ratio: float = 0.35
    max_center_distance_ratio: float = 3.0
    min_length_ratio: float = 0.35
    max_length_ratio: float = 2.8
    max_gap_ratio: float = 4.0
    max_candidates: int = 20
    min_coarse_colors: int = 3
    max_candidate_crop_area_pixels: float = 45000.0


# 保留旧名称，兼容已有测试和外部导入。
PairingConfig = ProtocolGeometryConfig


@dataclass(frozen=True)
class ProtocolDetection:
    command_id: int
    symbols: Tuple[str, ...]
    segments: Tuple[StripDetection, ...]
    score: float
    confidence: float
    geometry_quality: float
    angle_degrees: float
    cross_distance: float
    center_distance_ratio: float
    gap_ratio: float

    @property
    def first(self) -> StripDetection:
        return self.segments[0]

    @property
    def second(self) -> StripDetection:
        return self.segments[1]


def load_pairing_config(path: str) -> ProtocolGeometryConfig:
    with Path(path).open('r', encoding='utf-8') as stream:
        raw = yaml.safe_load(stream) or {}

    row = raw.get('protocol_geometry', {})
    if not row:
        three = raw.get('three_segment', {})
        if three:
            row = {
                'min_command_score': three.get('min_three_score', 0.04),
                'max_angle_degrees': three.get('max_angle_degrees', 18.0),
                'max_cross_distance_pixels': three.get(
                    'max_cross_distance', 35.0),
                'min_center_distance_ratio': three.get(
                    'min_center_distance_ratio', 0.35),
                'max_center_distance_ratio': three.get(
                    'max_center_distance_ratio', 3.0),
                'max_gap_ratio': three.get('max_gap_ratio', 4.0),
                'max_candidates': three.get('max_results', 20),
                'min_coarse_colors': three.get('min_coarse_colors', 3),
                'max_candidate_crop_area_pixels': three.get(
                    'max_candidate_crop_area_pixels', 45000.0),
            }
        else:
            pairing = raw.get('pairing', {})
            row = {
                'min_command_score': pairing.get(
                    'min_pair_score', pairing.get('min_command_score', 0.04)),
                'max_angle_degrees': pairing.get('max_angle_degrees', 18.0),
                'max_cross_distance_pixels': pairing.get(
                    'max_cross_distance_pixels', 35.0),
                'min_center_distance_ratio': pairing.get(
                    'min_center_distance_ratio', 0.35),
                'max_center_distance_ratio': pairing.get(
                    'max_center_distance_ratio', 3.0),
                'min_length_ratio': pairing.get('min_length_ratio', 0.35),
                'max_length_ratio': pairing.get('max_length_ratio', 2.8),
                'max_candidates': pairing.get('max_pairs', 20),
                'min_coarse_colors': pairing.get('min_coarse_colors', 3),
                'max_candidate_crop_area_pixels': pairing.get(
                    'max_candidate_crop_area_pixels', 45000.0),
            }

    return ProtocolGeometryConfig(
        min_command_score=float(row.get('min_command_score', 0.04)),
        max_angle_degrees=float(row.get('max_angle_degrees', 18.0)),
        max_cross_distance_pixels=float(
            row.get('max_cross_distance_pixels', 35.0)),
        min_center_distance_ratio=float(
            row.get('min_center_distance_ratio', 0.35)),
        max_center_distance_ratio=float(
            row.get('max_center_distance_ratio', 3.0)),
        min_length_ratio=float(row.get('min_length_ratio', 0.35)),
        max_length_ratio=float(row.get('max_length_ratio', 2.8)),
        max_gap_ratio=float(row.get('max_gap_ratio', 4.0)),
        max_candidates=int(row.get('max_candidates', 20)),
        min_coarse_colors=int(row.get('min_coarse_colors', 3)),
        max_candidate_crop_area_pixels=float(
            row.get('max_candidate_crop_area_pixels', 45000.0)),
    )


def protocol_color_symbols(protocol: FixedColorProtocol) -> Tuple[str, ...]:
    symbols = {
        symbol
        for code in protocol.commands.values()
        for symbol in code
    }
    return tuple(sorted(symbols))


def scaled_candidate_crop_area(
        config: ProtocolGeometryConfig,
        processing_scale: float) -> float:
    max_area = float(config.max_candidate_crop_area_pixels)
    if max_area <= 0.0:
        return max_area
    scale = min(1.0, max(float(processing_scale), 0.1))
    return max_area * scale * scale


def _axis(candidate: StripDetection) -> Tuple[np.ndarray, np.ndarray]:
    start = (candidate.corners[0] + candidate.corners[3]) * 0.5
    end = (candidate.corners[1] + candidate.corners[2]) * 0.5
    direction = end - start
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-6:
        direction = np.array((1.0, 0.0), dtype=np.float32)
    else:
        direction = direction / norm
    center = (start + end) * 0.5
    return center.astype(np.float32), direction.astype(np.float32)


def _aligned_average_axis(axes: Sequence[np.ndarray]) -> np.ndarray:
    base = axes[0]
    total = np.zeros(2, dtype=np.float32)
    for axis in axes:
        if float(np.dot(axis, base)) < 0.0:
            axis = -axis
        total += axis
    norm = float(np.linalg.norm(total))
    if norm <= 1e-6:
        return base
    return total / norm


def _angle_spread_degrees(axes: Sequence[np.ndarray]) -> float:
    max_angle = 0.0
    for left, right in combinations(axes, 2):
        dot = abs(float(np.dot(left, right)))
        dot = max(-1.0, min(1.0, dot))
        max_angle = max(max_angle, math.degrees(math.acos(dot)))
    return max_angle


def _image_order_indices(
        centers: np.ndarray,
        x_tolerance: float = IMAGE_ORDER_X_TOLERANCE):
    def compare(left_index: int, right_index: int) -> int:
        left = centers[left_index]
        right = centers[right_index]
        if abs(float(left[0] - right[0])) <= x_tolerance:
            if float(left[1]) < float(right[1]):
                return -1
            if float(left[1]) > float(right[1]):
                return 1
        else:
            if float(left[0]) < float(right[0]):
                return -1
            if float(left[0]) > float(right[0]):
                return 1
        return int(left_index) - int(right_index)

    return np.array(
        sorted(range(len(centers)), key=cmp_to_key(compare)),
        dtype=np.int64)


def _sequence_geometry(
        raw_segments: Sequence[StripDetection],
        config: ProtocolGeometryConfig
) -> Optional[Tuple[Tuple[StripDetection, ...], float, float, float, float, float]]:
    axis_rows = [_axis(item) for item in raw_segments]
    centers = np.stack([item[0] for item in axis_rows], axis=0)
    axes = [item[1] for item in axis_rows]
    angle = _angle_spread_degrees(axes)
    if angle > config.max_angle_degrees:
        return None

    common_axis = _aligned_average_axis(axes)
    origin = np.mean(centers, axis=0)
    projections = (centers - origin) @ common_axis
    order = _image_order_indices(centers)
    ordered = tuple(raw_segments[index] for index in order)
    ordered_centers = centers[order]
    ordered_proj = projections[order]

    normal = np.array((-common_axis[1], common_axis[0]), dtype=np.float32)
    cross_distances = np.abs((ordered_centers - origin) @ normal)
    cross_distance = float(np.max(cross_distances))
    if cross_distance > config.max_cross_distance_pixels:
        return None

    gaps = np.abs(np.diff(ordered_proj))
    if len(gaps) == 0 or np.any(gaps <= 1e-6):
        return None

    lengths = np.array([item.length for item in ordered], dtype=np.float32)
    median_length = max(float(np.median(lengths)), 1.0)
    center_distance_ratios = gaps / median_length
    if np.any(center_distance_ratios < config.min_center_distance_ratio):
        return None
    if np.any(center_distance_ratios > config.max_center_distance_ratio):
        return None

    center_distance_ratio = float(np.mean(center_distance_ratios))
    gap_ratio = float(np.max(gaps) / max(float(np.min(gaps)), 1e-6))
    if gap_ratio > config.max_gap_ratio:
        return None

    length_ratio = float(np.max(lengths) / max(float(np.min(lengths)), 1e-6))
    if not (config.min_length_ratio <= length_ratio <= config.max_length_ratio):
        return None

    angle_quality = max(0.0, 1.0 - angle / max(config.max_angle_degrees, 1e-6))
    cross_quality = max(
        0.0, 1.0 - cross_distance / max(config.max_cross_distance_pixels, 1e-6))
    gap_quality = max(
        0.0, 1.0 - (gap_ratio - 1.0) / max(config.max_gap_ratio - 1.0, 1e-6))
    length_quality = float(math.exp(-abs(math.log(max(length_ratio, 1e-6)))))
    distance_quality = float(np.mean([
        math.exp(-abs(math.log(max(value, 1e-6))))
        for value in center_distance_ratios
    ]))
    geometry_quality = float(np.clip(
        0.30 * angle_quality +
        0.25 * cross_quality +
        0.20 * gap_quality +
        0.15 * length_quality +
        0.10 * distance_quality,
        0.0, 1.0))
    return (
        ordered,
        geometry_quality,
        angle,
        cross_distance,
        center_distance_ratio,
        gap_ratio,
    )


def decode_protocol_candidates(
        strip_candidates: Sequence[StripDetection],
        protocol: FixedColorProtocol,
        config: ProtocolGeometryConfig) -> List[ProtocolDetection]:
    code_length = max(int(protocol.code_length), 1)
    if len(strip_candidates) < code_length:
        return []

    detections: List[ProtocolDetection] = []
    for raw_segments in combinations(strip_candidates, code_length):
        if len({item.color for item in raw_segments}) != code_length:
            continue
        geometry = _sequence_geometry(raw_segments, config)
        if geometry is None:
            continue
        (
            ordered_segments,
            geometry_quality,
            angle,
            cross_distance,
            center_distance_ratio,
            gap_ratio,
        ) = geometry
        symbols = tuple(item.color for item in ordered_segments)
        command_id = protocol.decode(symbols)
        if command_id is None:
            continue
        segment_score = float(np.prod([
            max(item.score, 0.0) for item in ordered_segments
        ]) ** (1.0 / code_length))
        score = segment_score * max(geometry_quality, 0.0)
        if score < config.min_command_score:
            continue
        confidence = float(np.prod([
            max(item.confidence, 0.0) for item in ordered_segments
        ] + [max(geometry_quality, 0.0)]) ** (1.0 / (code_length + 1)))
        detections.append(ProtocolDetection(
            command_id=int(command_id),
            symbols=symbols,
            segments=ordered_segments,
            score=float(score),
            confidence=confidence,
            geometry_quality=geometry_quality,
            angle_degrees=angle,
            cross_distance=cross_distance,
            center_distance_ratio=center_distance_ratio,
            gap_ratio=gap_ratio,
        ))
    detections.sort(key=lambda item: item.score, reverse=True)
    return detections[:max(config.max_candidates, 1)]


def select_protocol_winner(
        candidates: Sequence[ProtocolDetection],
        margin: float) -> Optional[ProtocolDetection]:
    if not candidates:
        return None
    different = next(
        (item for item in candidates[1:]
         if item.command_id != candidates[0].command_id),
        None)
    if different is not None:
        ratio = candidates[0].score / max(different.score, 1e-9)
        if ratio < margin:
            return None
    return candidates[0]


def annotate_protocol(
        bgr: np.ndarray,
        protocol_candidates: Sequence[ProtocolDetection],
        winner: Optional[ProtocolDetection],
        state: str,
        confirmed_count: int,
        required_count: int) -> np.ndarray:
    output = bgr.copy()
    for rank, item in enumerate(protocol_candidates, 1):
        selected = item is winner
        color = (0, 255, 255) if selected else (0, 165, 255)
        centers = []
        for segment in item.segments:
            corners = np.round(segment.corners).astype(np.int32)
            cv2.polylines(output, [corners], True, color, 2)
            center, _ = _axis(segment)
            centers.append(center)
        for left, right in zip(centers, centers[1:]):
            cv2.line(
                output,
                tuple(np.round(left).astype(np.int32)),
                tuple(np.round(right).astype(np.int32)),
                color,
                2)
        anchor = tuple(np.round(
            np.mean(np.stack(centers, axis=0), axis=0)).astype(np.int32))
        text = (
            f'#{rank} id={item.command_id} {"-".join(item.symbols)} '
            f'score={item.score:.3f} geo={item.geometry_quality:.2f}')
        cv2.putText(
            output, text, anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2)

    if winner is None:
        title = f'{state}: no valid command'
    else:
        title = (
            f'{state}: id={winner.command_id} '
            f'{"-".join(winner.symbols)} '
            f'{confirmed_count}/{required_count} '
            f'score={winner.score:.3f}')
    cv2.putText(
        output, title, (12, 58), cv2.FONT_HERSHEY_SIMPLEX,
        0.72, (255, 255, 255), 2)
    return output
