"""Legacy three-segment detector used by the old usb_rgb_2 ROS path."""

from dataclasses import dataclass
from functools import cmp_to_key
from itertools import combinations
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml


IMAGE_ORDER_X_TOLERANCE = 12.0


@dataclass(frozen=True)
class ThreeSegmentConfig:
    max_single_candidates: int = 30
    max_results: int = 12
    min_three_score: float = 0.05
    winner_margin: float = 1.2
    max_angle_degrees: float = 18.0
    max_cross_distance: float = 45.0
    min_center_distance_ratio: float = 0.35
    max_center_distance_ratio: float = 4.0
    max_gap_ratio: float = 3.2


@dataclass(frozen=True)
class SegmentAxis:
    center: np.ndarray
    axis: np.ndarray
    start: np.ndarray
    end: np.ndarray


@dataclass(frozen=True)
class ThreeSegmentDetection:
    symbols: Tuple[str, str, str]
    segments: Tuple[object, object, object]
    score: float
    confidence: float
    geometry_quality: float
    angle_degrees: float
    cross_distance: float
    gap_ratio: float
    length_ratio: float
    center_distance_ratio: float
    ambiguous: bool = False


@dataclass(frozen=True)
class WeakStripDetection:
    color: str
    confidence: float
    score: float
    corners: np.ndarray
    dot_count: int
    length: float
    residual: float
    spacing_cv: float
    line_quality: float = 0.0
    dot_quality: float = 0.0
    periodic_quality: float = 0.0
    color_quality: float = 0.0
    valley_quality: float = 0.0
    peak_centers: Optional[np.ndarray] = None
    mode: str = 'weak'


def load_three_segment_config(path: str) -> ThreeSegmentConfig:
    with Path(path).open('r', encoding='utf-8') as stream:
        raw = yaml.safe_load(stream) or {}
    row = raw.get('three_segment', {}) or {}
    return ThreeSegmentConfig(
        max_single_candidates=int(row.get('max_single_candidates', 30)),
        max_results=int(row.get('max_results', 12)),
        min_three_score=float(row.get('min_three_score', 0.05)),
        winner_margin=float(row.get('winner_margin', 1.2)),
        max_angle_degrees=float(row.get('max_angle_degrees', 18.0)),
        max_cross_distance=float(row.get('max_cross_distance', 45.0)),
        min_center_distance_ratio=float(
            row.get('min_center_distance_ratio', 0.35)),
        max_center_distance_ratio=float(
            row.get('max_center_distance_ratio', 4.0)),
        max_gap_ratio=float(row.get('max_gap_ratio', 3.2)),
    )


def scaled_frame(frame, scale: float):
    if scale >= 0.999:
        return frame
    return cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def detect_single_segments(frame, classifier, config, processing_scale: float):
    work = scaled_frame(frame, processing_scale)
    candidates = classifier.detect_candidates(work, config)
    if processing_scale < 0.999:
        inverse = 1.0 / processing_scale
        candidates = [item.scaled(inverse) for item in candidates]
    return sorted(candidates, key=lambda item: item.score, reverse=True)


def color_blob_points(work, classifier, config):
    hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)
    masks = classifier.color_masks(hsv, work, config)
    rows = {}
    for color, mask in masks.items():
        count, labels, stats, centers = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), 8)
        points = []
        for index in range(1, count):
            area = int(stats[index, cv2.CC_STAT_AREA])
            x, y, width, height = stats[index, :4]
            if not (1 <= area <= max(config.max_blob_area, 1)):
                continue
            if max(width, height) > 24:
                continue
            radius = max(1.0, 0.5 * float(max(width, height)))
            points.append((
                np.array(centers[index], dtype=np.float32),
                radius,
                float(area),
            ))
        rows[color] = points
    return rows


def weak_line_from_points(color, selected, scale: float) -> Optional[WeakStripDetection]:
    if len(selected) < 4:
        return None
    points = np.stack([item[0] for item in selected], axis=0)
    center = np.mean(points, axis=0)
    centered = points - center
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    axis = vt[0].astype(np.float32)
    if float(np.linalg.norm(axis)) <= 1e-6:
        return None
    axis = axis / float(np.linalg.norm(axis))
    normal = np.array((-axis[1], axis[0]), dtype=np.float32)
    projection = centered @ axis
    cross = centered @ normal
    order = np.argsort(projection)
    projection = projection[order]
    points = points[order]
    cross = cross[order]
    length = float(projection[-1] - projection[0])
    if length <= 1e-6:
        return None
    residual = float(np.sqrt(np.mean(np.square(cross))))
    gaps = np.diff(projection)
    if len(gaps) == 0 or np.any(gaps <= 1e-6):
        return None
    gap_ratio = float(np.max(gaps) / max(float(np.min(gaps)), 1e-6))
    if gap_ratio > 3.8:
        return None
    spacing_cv = float(np.std(gaps) / max(float(np.mean(gaps)), 1e-6))
    if spacing_cv > 0.45:
        return None

    width = max(4.0, 2.8 * max(item[1] for item in selected))
    start = center + axis * projection[0]
    end = center + axis * projection[-1]
    half = normal * width
    corners = np.stack([start - half, end - half, end + half, start + half], axis=0)
    inverse = 1.0 / scale

    line_quality = float(np.exp(-np.square(residual / 4.0)))
    dot_quality = min(1.0, len(selected) / 6.0)
    periodic_quality = float(np.clip(1.0 - spacing_cv / 0.45, 0.0, 1.0))
    color_quality = min(1.0, np.mean([item[2] for item in selected]) / 12.0)
    score = float(np.clip(
        0.72 * line_quality *
        (0.35 + 0.65 * dot_quality) *
        (0.35 + 0.65 * periodic_quality) *
        (0.40 + 0.60 * color_quality),
        0.0, 1.0))
    if score < 0.18:
        return None

    return WeakStripDetection(
        color=color,
        confidence=float(np.sqrt(score)),
        score=score,
        corners=corners * inverse,
        dot_count=len(selected),
        length=length * inverse,
        residual=residual * inverse,
        spacing_cv=spacing_cv,
        line_quality=line_quality,
        dot_quality=dot_quality,
        periodic_quality=periodic_quality,
        color_quality=color_quality,
        valley_quality=0.0,
        peak_centers=points * inverse,
    )


def weak_segments_from_masks(
        frame,
        classifier,
        config,
        processing_scale: float,
        allowed_colors: Optional[set] = None):
    work = scaled_frame(frame, processing_scale)
    rows = color_blob_points(work, classifier, config)
    weak = []
    for color, points in rows.items():
        if allowed_colors is not None and color not in allowed_colors:
            continue
        if len(points) < 4:
            continue
        points = sorted(points, key=lambda item: item[2], reverse=True)[:50]
        color_candidates = []
        centers = np.stack([item[0] for item in points], axis=0)
        for left, right in combinations(range(len(points)), 2):
            start = centers[left]
            end = centers[right]
            vector = end - start
            length = float(np.linalg.norm(vector))
            if length < 18.0:
                continue
            axis = vector / length
            normal = np.array((-axis[1], axis[0]), dtype=np.float32)
            delta = centers - start
            projection = delta @ axis
            cross = np.abs(delta @ normal)
            indexes = np.where(
                (cross <= 4.5) &
                (projection >= -4.5) &
                (projection <= length + 4.5))[0]
            if len(indexes) < 4:
                continue
            candidate = weak_line_from_points(
                color,
                [points[index] for index in indexes],
                processing_scale)
            if candidate is not None:
                color_candidates.append(candidate)
        color_candidates = deduplicate_single_segments(color_candidates)
        weak.extend(color_candidates[:4])
    return sorted(weak, key=lambda item: item.score, reverse=True)


def deduplicate_single_segments(candidates: Sequence[object]) -> List[object]:
    kept = []
    for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
        if any(overlap_ratio(candidate, existing) >= 0.55 for existing in kept):
            continue
        kept.append(candidate)
    return kept


def merge_strong_and_weak_segments(strong: Sequence[object], weak: Sequence[object]):
    merged = list(strong)
    for candidate in weak:
        overlaps_strong = [
            existing for existing in strong
            if overlap_ratio(candidate, existing) >= 0.45
        ]
        if any(existing.color != candidate.color for existing in overlaps_strong):
            continue
        if any(
                existing.color == candidate.color and
                overlap_ratio(candidate, existing) >= 0.35
                for existing in merged):
            continue
        merged.append(candidate)
    return sorted(merged, key=lambda item: item.score, reverse=True)


def segment_axis(candidate) -> SegmentAxis:
    start = (candidate.corners[0] + candidate.corners[3]) * 0.5
    end = (candidate.corners[1] + candidate.corners[2]) * 0.5
    direction = end - start
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-6:
        direction = np.array((1.0, 0.0), dtype=np.float32)
    else:
        direction = direction / norm
    center = (start + end) * 0.5
    return SegmentAxis(
        center=center.astype(np.float32),
        axis=direction.astype(np.float32),
        start=start.astype(np.float32),
        end=end.astype(np.float32),
    )


def aligned_average_axis(axes: Sequence[SegmentAxis]) -> np.ndarray:
    base = axes[0].axis
    total = np.zeros(2, dtype=np.float32)
    for item in axes:
        axis = item.axis
        if float(np.dot(axis, base)) < 0.0:
            axis = -axis
        total += axis
    norm = float(np.linalg.norm(total))
    if norm <= 1e-6:
        return base
    return total / norm


def angle_spread_degrees(axes: Sequence[SegmentAxis]) -> float:
    max_angle = 0.0
    for left, right in combinations(axes, 2):
        dot = abs(float(np.dot(left.axis, right.axis)))
        dot = max(-1.0, min(1.0, dot))
        max_angle = max(max_angle, float(np.degrees(np.arccos(dot))))
    return max_angle


def image_order_indices(centers: np.ndarray, x_tolerance: float = IMAGE_ORDER_X_TOLERANCE):
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


def build_three_segment_candidate(
        raw_segments: Tuple[object, object, object],
        config: ThreeSegmentConfig) -> Optional[ThreeSegmentDetection]:
    axes = [segment_axis(item) for item in raw_segments]
    angle = angle_spread_degrees(axes)
    if angle > config.max_angle_degrees:
        return None

    common_axis = aligned_average_axis(axes)
    normal = np.array((-common_axis[1], common_axis[0]), dtype=np.float32)
    centers = np.stack([item.center for item in axes], axis=0)
    origin = np.mean(centers, axis=0)
    projections = (centers - origin) @ common_axis
    order = image_order_indices(centers)
    segments = tuple(raw_segments[index] for index in order)
    ordered_proj = projections[order]

    symbols = tuple(item.color for item in segments)
    if symbols[0] == symbols[1] or symbols[1] == symbols[2]:
        return None

    gaps = np.abs(np.diff(ordered_proj))
    if np.any(gaps <= 1e-6):
        return None

    lengths = np.array([item.length for item in segments], dtype=np.float32)
    median_length = max(float(np.median(lengths)), 1.0)
    gap_ratios_to_length = gaps / median_length
    if np.any(gap_ratios_to_length < config.min_center_distance_ratio):
        return None
    if np.any(gap_ratios_to_length > config.max_center_distance_ratio):
        return None

    gap_ratio = float(np.max(gaps) / max(float(np.min(gaps)), 1e-6))
    if gap_ratio > config.max_gap_ratio:
        return None

    cross_distances = np.abs((centers - origin) @ normal)
    cross_distance = float(np.max(cross_distances))
    if cross_distance > config.max_cross_distance:
        return None

    length_ratio = float(np.max(lengths) / max(float(np.min(lengths)), 1e-6))
    if length_ratio > 3.0:
        return None

    angle_quality = max(0.0, 1.0 - angle / max(config.max_angle_degrees, 1e-6))
    cross_quality = max(
        0.0, 1.0 - cross_distance / max(config.max_cross_distance, 1e-6))
    gap_quality = max(
        0.0, 1.0 - (gap_ratio - 1.0) / max(config.max_gap_ratio - 1.0, 1e-6))
    length_quality = float(np.exp(-abs(np.log(max(length_ratio, 1e-6)))))
    distance_quality = float(np.mean([
        np.exp(-abs(np.log(max(value, 1e-6))))
        for value in gap_ratios_to_length
    ]))
    geometry_quality = float(np.clip(
        0.30 * angle_quality +
        0.25 * cross_quality +
        0.20 * gap_quality +
        0.15 * length_quality +
        0.10 * distance_quality,
        0.0, 1.0))

    segment_score = float(np.cbrt(
        max(segments[0].score, 0.0) *
        max(segments[1].score, 0.0) *
        max(segments[2].score, 0.0)))
    score = segment_score * geometry_quality
    if score < config.min_three_score:
        return None

    confidence = float(np.cbrt(
        max(segments[0].confidence, 0.0) *
        max(segments[1].confidence, 0.0) *
        max(segments[2].confidence, 0.0) *
        max(geometry_quality, 0.0)))

    return ThreeSegmentDetection(
        symbols=symbols,
        segments=segments,
        score=float(score),
        confidence=confidence,
        geometry_quality=geometry_quality,
        angle_degrees=angle,
        cross_distance=cross_distance,
        gap_ratio=gap_ratio,
        length_ratio=length_ratio,
        center_distance_ratio=float(np.mean(gap_ratios_to_length)),
    )


def overlap_ratio(left, right) -> float:
    rect_left = cv2.minAreaRect(left.corners.astype(np.float32))
    rect_right = cv2.minAreaRect(right.corners.astype(np.float32))
    result = cv2.rotatedRectangleIntersection(rect_left, rect_right)
    if result[0] == cv2.INTERSECT_NONE or result[1] is None:
        return 0.0
    area = abs(float(cv2.contourArea(result[1])))
    min_area = min(
        abs(float(cv2.contourArea(left.corners.astype(np.float32)))),
        abs(float(cv2.contourArea(right.corners.astype(np.float32)))))
    return area / max(min_area, 1e-6)


def deduplicate_three_segment_candidates(
        candidates: Sequence[ThreeSegmentDetection]) -> List[ThreeSegmentDetection]:
    kept: List[ThreeSegmentDetection] = []
    for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
        duplicate = False
        for existing in kept:
            same_symbols = candidate.symbols == existing.symbols
            overlaps = [
                overlap_ratio(left, right)
                for left, right in zip(candidate.segments, existing.segments)
            ]
            if same_symbols and min(overlaps) >= 0.35:
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
    return kept


def detect_three_segments(
        single_candidates: Sequence[object],
        config: ThreeSegmentConfig):
    limited = list(single_candidates[:max(config.max_single_candidates, 3)])
    raw = []
    for triple in combinations(limited, 3):
        candidate = build_three_segment_candidate(triple, config)
        if candidate is not None:
            raw.append(candidate)
    candidates = deduplicate_three_segment_candidates(raw)
    candidates = candidates[:max(config.max_results, 1)]
    if len(candidates) >= 2:
        margin = candidates[0].score / max(candidates[1].score, 1e-9)
        if margin < config.winner_margin:
            winner = ThreeSegmentDetection(
                symbols=candidates[0].symbols,
                segments=candidates[0].segments,
                score=candidates[0].score,
                confidence=candidates[0].confidence,
                geometry_quality=candidates[0].geometry_quality,
                angle_degrees=candidates[0].angle_degrees,
                cross_distance=candidates[0].cross_distance,
                gap_ratio=candidates[0].gap_ratio,
                length_ratio=candidates[0].length_ratio,
                center_distance_ratio=candidates[0].center_distance_ratio,
                ambiguous=True,
            )
            candidates[0] = winner
    return candidates


def detect_three_segment_frame_old(
        frame,
        classifier,
        detector_config,
        processing_scale: float,
        three_segment_config: ThreeSegmentConfig):
    strong_candidates = detect_single_segments(
        frame, classifier, detector_config, processing_scale)
    weak_candidates = []
    single_candidates = list(strong_candidates)
    three_candidates = detect_three_segments(single_candidates, three_segment_config)
    if not three_candidates:
        strong_colors = {item.color for item in strong_candidates}
        allowed_colors = {
            item.name for item in detector_config.colors
            if item.name not in strong_colors
        }
        weak_candidates = weak_segments_from_masks(
            frame, classifier, detector_config, processing_scale, allowed_colors)
        single_candidates = merge_strong_and_weak_segments(
            strong_candidates, weak_candidates)
        three_candidates = detect_three_segments(
            single_candidates, three_segment_config)
    return strong_candidates, weak_candidates, single_candidates, three_candidates


def draw_label(output, text: str, anchor, color, scale=0.52, thickness=2):
    x, y = int(anchor[0]), int(anchor[1])
    y = max(24, y)
    cv2.putText(
        output, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
        scale, (0, 0, 0), thickness + 2)
    cv2.putText(
        output, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
        scale, color, thickness)


def annotate_three_segments(
        image,
        single_candidates: Sequence[object],
        three_candidates: Sequence[ThreeSegmentDetection]):
    output = image.copy()

    for rank, candidate in enumerate(single_candidates[:20], 1):
        corners = np.round(candidate.corners).astype(np.int32)
        cv2.polylines(output, [corners], True, (120, 120, 120), 1)
        anchor = corners[np.argmin(corners[:, 1])]
        draw_label(
            output,
            f's{rank}:{candidate.color[0]} {candidate.score:.2f}',
            anchor,
            (180, 180, 180),
            scale=0.42,
            thickness=1)

    for rank, item in enumerate(three_candidates, 1):
        selected = rank == 1
        color = (0, 255, 255) if selected else (0, 165, 255)
        thickness = 3 if selected else 2
        centers = []
        for segment_index, segment in enumerate(item.segments):
            corners = np.round(segment.corners).astype(np.int32)
            cv2.polylines(output, [corners], True, color, thickness)
            axis = segment_axis(segment)
            centers.append(axis.center)
            cv2.circle(
                output,
                tuple(np.round(axis.center).astype(np.int32)),
                5,
                color,
                -1)
            draw_label(
                output,
                f'{segment_index + 1}:{segment.color[0]}',
                np.round(axis.center + np.array((6, -8))).astype(np.int32),
                color,
                scale=0.55,
                thickness=2)
        for left, right in zip(centers[:-1], centers[1:]):
            cv2.line(
                output,
                tuple(np.round(left).astype(np.int32)),
                tuple(np.round(right).astype(np.int32)),
                color,
                thickness)
        middle = np.mean(np.stack(centers, axis=0), axis=0)
        ambiguous = bool(getattr(item, 'ambiguous', False))
        status = 'AMBIG ' if ambiguous else ''
        text = (
            f'#{rank} {status}{"-".join(s[0] for s in item.symbols)} '
            f'score={item.score:.3f} conf={item.confidence:.2f} '
            f'geo={item.geometry_quality:.2f}')
        draw_label(
            output,
            text,
            np.round(middle + np.array((10, 18 * rank))).astype(np.int32),
            color,
            scale=0.55,
            thickness=2)

    if not three_candidates:
        title = f'NO THREE-SEGMENT DETECTION; singles={len(single_candidates)}'
    else:
        winner = three_candidates[0]
        reverse = '-'.join(symbol[0] for symbol in winner.symbols[::-1])
        title = (
            f'SELECTED {"-".join(s[0] for s in winner.symbols)} '
            f'rev={reverse} score={winner.score:.3f}')
        if bool(getattr(winner, 'ambiguous', False)):
            title = 'AMBIGUOUS ' + title
    draw_label(output, title, (12, 32), (255, 255, 255), scale=0.72, thickness=2)
    return output
