"""与协议无关的单色可寻址 LED 灯带检测。

灯带位置只由亮度、灯珠形状和排列规律决定，颜色随后使用当前相机独立标定的色度模型
分类。这样既不会把 R1 的发送 RGB 值误当成摄像头观测值，也能对未知颜色作明确拒绝。
"""

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml


_FEATURE_ARRAY_CACHE: Dict[int, np.ndarray] = {}


@dataclass(frozen=True)
class ColorModel:
    name: str
    chroma_center: Tuple[float, float]
    chroma_covariance: Tuple[Tuple[float, float], Tuple[float, float]]
    max_distance: float
    hue_center: Optional[float] = None
    hue_tolerance: float = 18.0
    feature_centers: Tuple[Tuple[float, float, float], ...] = ()
    feature_max_distance: Optional[float] = None

    @property
    def inverse_covariance(self) -> np.ndarray:
        covariance = np.asarray(self.chroma_covariance, dtype=np.float32)
        return np.linalg.inv(covariance)


@dataclass(frozen=True)
class DetectorConfig:
    colors: Tuple[ColorModel, ...]
    min_saturation: float = 60.0
    min_value: float = 55.0
    min_colored_fraction: float = 0.15
    min_color_support: float = 0.70
    min_class_margin: float = 1.25
    min_blob_area: int = 1
    max_blob_area: int = 300
    max_blob_aspect: float = 3.5
    min_blob_compactness: float = 0.12
    dog_sigma_small: float = 0.8
    dog_sigma_large: float = 2.6
    min_dog_response: float = 5.0
    max_points_per_color: int = 160
    max_pair_hypotheses: int = 2000
    max_line_hypotheses: int = 20
    fallback_max_pair_hypotheses: int = 10000
    fallback_max_line_hypotheses: int = 40
    min_dots: int = 6
    min_length_pixels: float = 35.0
    line_distance_pixels: float = 3.5
    max_gap_ratio: float = 3.2
    max_spacing_trend_error: float = 0.32
    min_coverage: float = 0.55
    min_valley_contrast: float = 0.10
    min_periodic_dot_quality: float = 0.38
    min_periodic_color_quality: float = 0.45
    min_geometry_score: float = 0.0
    min_score: float = 0.04
    winner_margin: float = 1.0
    fast_resize_width: int = 640
    fast_max_points_per_color: int = 80
    fast_enable_fallback: bool = False
    fast_global_search: bool = False
    fast_check_valley: bool = False


@dataclass(frozen=True)
class StripDetection:
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
    mode: str = 'periodic'
    geometry_score: float = 0.0
    class_distance: float = float('inf')
    class_margin: float = 0.0
    color_support: float = 0.0
    chroma: Optional[Tuple[float, float]] = None

    def scaled(self, factor: float) -> 'StripDetection':
        peaks = None if self.peak_centers is None else self.peak_centers * factor
        return StripDetection(
            color=self.color,
            confidence=self.confidence,
            score=self.score,
            corners=self.corners * factor,
            dot_count=self.dot_count,
            length=self.length * factor,
            residual=self.residual * factor,
            spacing_cv=self.spacing_cv,
            line_quality=self.line_quality,
            dot_quality=self.dot_quality,
            periodic_quality=self.periodic_quality,
            color_quality=self.color_quality,
            valley_quality=self.valley_quality,
            peak_centers=peaks,
            mode=self.mode,
            geometry_score=self.geometry_score,
            class_distance=self.class_distance,
            class_margin=self.class_margin,
            color_support=self.color_support,
            chroma=self.chroma,
        )


@dataclass(frozen=True)
class _LightPoint:
    center: np.ndarray
    radius: float
    response: float
    shape_quality: float
    color_quality: float
    chroma: Tuple[float, float]
    hue: Optional[float] = None

    @property
    def quality(self) -> float:
        response_quality = max(0.0, min(self.response / 28.0, 1.0))
        return float(np.sqrt(
            max(self.shape_quality, 0.0) *
            max(response_quality, 0.0) *
            max(self.color_quality, 0.0)))


def load_config(path: str) -> DetectorConfig:
    with Path(path).open('r', encoding='utf-8') as stream:
        raw = yaml.safe_load(stream) or {}
    color_rows = raw.get('colors', {})
    colors = tuple(
        ColorModel(
            name=str(name).upper(),
            chroma_center=tuple(
                float(value) for value in values['chroma_center']),
            chroma_covariance=tuple(
                tuple(float(value) for value in row)
                for row in values['chroma_covariance']),
            max_distance=float(values.get('max_distance', 3.0)),
            hue_center=(
                None if values.get('hue_center') is None
                else float(values['hue_center'])),
            hue_tolerance=float(values.get('hue_tolerance', 18.0)),
            feature_centers=tuple(
                tuple(float(item) for item in row)
                for row in values.get('feature_centers', ())),
            feature_max_distance=(
                None if values.get('feature_max_distance') is None
                else float(values['feature_max_distance'])),
        )
        for name, values in color_rows.items()
    )
    if not colors:
        raise ValueError('detector config must define at least one color')
    dots = raw.get('dots', {})
    geometry = raw.get('geometry', {})
    color = raw.get('color_classification', {})
    selection = raw.get('selection', {})
    fast = raw.get('fast_detection', {})
    return DetectorConfig(
        colors=colors,
        min_saturation=float(color.get('min_saturation', 60)),
        min_value=float(color.get('min_value', 55)),
        min_colored_fraction=float(
            color.get('min_colored_fraction', 0.15)),
        min_color_support=float(color.get('min_color_support', 0.70)),
        min_class_margin=float(color.get('min_class_margin', 1.25)),
        min_blob_area=int(dots.get('min_blob_area', 1)),
        max_blob_area=int(dots.get('max_blob_area', 300)),
        max_blob_aspect=float(dots.get('max_blob_aspect', 3.5)),
        min_blob_compactness=float(dots.get('min_blob_compactness', 0.12)),
        dog_sigma_small=float(dots.get('dog_sigma_small', 0.8)),
        dog_sigma_large=float(dots.get('dog_sigma_large', 2.6)),
        min_dog_response=float(dots.get('min_dog_response', 5.0)),
        max_points_per_color=int(dots.get('max_points_per_color', 160)),
        max_pair_hypotheses=int(geometry.get('max_pair_hypotheses', 2000)),
        max_line_hypotheses=int(geometry.get('max_line_hypotheses', 20)),
        fallback_max_pair_hypotheses=int(
            geometry.get('fallback_max_pair_hypotheses', 10000)),
        fallback_max_line_hypotheses=int(
            geometry.get('fallback_max_line_hypotheses', 40)),
        min_dots=int(geometry.get('min_dots', 6)),
        min_length_pixels=float(geometry.get('min_length_pixels', 35)),
        line_distance_pixels=float(geometry.get('line_distance_pixels', 3.5)),
        max_gap_ratio=float(geometry.get('max_gap_ratio', 3.2)),
        max_spacing_trend_error=float(
            geometry.get('max_spacing_trend_error', 0.32)),
        min_coverage=float(geometry.get('min_coverage', 0.55)),
        min_valley_contrast=float(
            geometry.get('min_valley_contrast', 0.10)),
        min_periodic_dot_quality=float(
            geometry.get('min_periodic_dot_quality', 0.38)),
        min_periodic_color_quality=float(
            geometry.get('min_periodic_color_quality', 0.45)),
        min_geometry_score=float(
            geometry.get('min_geometry_score', 0.0)),
        min_score=float(selection.get('min_score', 0.04)),
        winner_margin=float(selection.get('winner_margin', 1.0)),
        fast_resize_width=int(fast.get('resize_width', 640)),
        fast_max_points_per_color=int(
            fast.get('max_points_per_color', 80)),
        fast_enable_fallback=bool(fast.get('enable_fallback', False)),
        fast_global_search=bool(fast.get('global_search', False)),
        fast_check_valley=bool(fast.get('check_valley', False)),
    )


def _hue_distance(first: float, second: float) -> float:
    diff = abs((first - second) % 180.0)
    return min(diff, 180.0 - diff)


def _mean_hue(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    radians = np.asarray(values, dtype=np.float32) * np.pi / 90.0
    angle = float(np.arctan2(np.sin(radians).mean(), np.cos(radians).mean()))
    return (angle * 90.0 / np.pi) % 180.0


def _point_chroma(
        bgr: np.ndarray,
        hsv: np.ndarray,
        center: np.ndarray,
        config: DetectorConfig,
        radius: int) -> Optional[Tuple[Tuple[float, float], float]]:
    height, width = hsv.shape[:2]
    x, y = np.round(center).astype(int)
    radius = max(2, min(radius, 8))
    x0, x1 = max(0, x - radius), min(width, x + radius + 1)
    y0, y1 = max(0, y - radius), min(height, y + radius + 1)
    patch = hsv[y0:y1, x0:x1]
    if patch.size == 0:
        return 0.0
    yy, xx = np.ogrid[y0 - y:y1 - y, x0 - x:x1 - x]
    disk = xx * xx + yy * yy <= radius * radius
    pixels = patch[disk]
    bright = pixels[pixels[:, 2] >= config.min_value]
    if len(bright) == 0:
        return None
    saturated_mask = (
        (pixels[:, 1] >= config.min_saturation) &
        (pixels[:, 2] >= config.min_value))
    saturated = pixels[saturated_mask]
    if len(saturated) == 0:
        return None
    colored_fraction = len(saturated) / max(len(bright), 1)
    if colored_fraction < config.min_colored_fraction:
        return None
    saturated_values = saturated[:, 2]
    value_cutoff = float(np.percentile(saturated_values, 65))
    core_mask = saturated_mask & (pixels[:, 2] >= value_cutoff)
    bgr_patch = bgr[y0:y1, x0:x1][disk][core_mask].astype(np.float32)
    channel_sum = np.maximum(np.sum(bgr_patch, axis=1), 1.0)
    normalized = bgr_patch / channel_sum[:, None]
    chroma = np.median(normalized[:, :2], axis=0)
    saturation_quality = min(
        1.0, float(np.median(saturated[:, 1])) / 180.0)
    quality = float(np.clip(
        np.sqrt(colored_fraction * saturation_quality), 0.0, 1.0))
    return (float(chroma[0]), float(chroma[1])), quality


def _extract_light_points(
        bgr: np.ndarray,
        hsv: np.ndarray,
        mask: np.ndarray,
        dog: np.ndarray,
        config: DetectorConfig) -> List[_LightPoint]:
    response_mask = mask & (dog >= config.min_dog_response)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        response_mask.astype(np.uint8), 8)
    points: List[_LightPoint] = []
    for label in range(1, count):
        x, y, width, height, area = stats[label]
        if area < config.min_blob_area or area > config.max_blob_area:
            continue
        aspect = max(width, height) / max(min(width, height), 1)
        if aspect > config.max_blob_aspect:
            continue
        ys, xs = np.nonzero(labels[y:y + height, x:x + width] == label)
        if len(xs) == 0:
            continue
        xs = xs.astype(np.float32) + x
        ys = ys.astype(np.float32) + y
        weights = np.maximum(dog[ys.astype(int), xs.astype(int)], 1e-3)
        center = np.array([
            np.average(xs, weights=weights),
            np.average(ys, weights=weights),
        ], dtype=np.float32)
        if len(xs) >= 3:
            coordinates = np.column_stack((xs, ys))
            covariance = np.cov(coordinates, rowvar=False, aweights=weights)
            eigenvalues = np.linalg.eigvalsh(covariance)
            compactness = float(
                (eigenvalues[0] + 0.35) / (eigenvalues[-1] + 0.35))
        else:
            compactness = 1.0 / aspect
        if compactness < config.min_blob_compactness:
            continue
        radius = max(1.0, float(np.sqrt(area / np.pi)))
        response = float(np.percentile(weights, 80))
        color_result = _point_chroma(
            bgr, hsv, center, config, int(np.ceil(radius + 2.0)))
        if color_result is None:
            continue
        chroma, color_quality = color_result
        points.append(_LightPoint(
            center=center,
            radius=radius,
            response=response,
            shape_quality=float(np.sqrt(compactness)),
            color_quality=color_quality,
            chroma=chroma,
        ))
    points.sort(key=lambda item: item.quality, reverse=True)
    unique: List[_LightPoint] = []
    for point in points:
        if any(
                np.linalg.norm(point.center - old.center) <
                min(24.0, max(
                    4.0, max(point.radius, old.radius) * 2.0 + 3.0))
                for old in unique):
            continue
        unique.append(point)
        if len(unique) >= config.max_points_per_color:
            break
    return unique


def _extract_saturated_components(
        bgr: np.ndarray,
        hsv: np.ndarray,
        dog: np.ndarray,
        config: DetectorConfig) -> List[_LightPoint]:
    """将完整彩色灯珠作为单个连通区域，补偿环形光斑被 DoG 切碎的情况。"""
    mask = (
        (hsv[:, :, 1] >= config.min_saturation) &
        (hsv[:, :, 2] >= config.min_value))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), 8)
    points: List[_LightPoint] = []
    for label in range(1, count):
        x, y, width, height, area = stats[label]
        if area < 3 or area > max(config.max_blob_area * 4, 600):
            continue
        aspect = max(width, height) / max(min(width, height), 1)
        if aspect > config.max_blob_aspect:
            continue
        ys, xs = np.nonzero(labels[y:y + height, x:x + width] == label)
        if len(xs) < 3:
            continue
        xs = xs.astype(np.float32) + x
        ys = ys.astype(np.float32) + y
        values = hsv[ys.astype(int), xs.astype(int), 2].astype(np.float32)
        center = np.array([
            np.average(xs, weights=np.maximum(values, 1.0)),
            np.average(ys, weights=np.maximum(values, 1.0)),
        ], dtype=np.float32)
        coordinates = np.column_stack((xs, ys))
        covariance = np.cov(coordinates, rowvar=False)
        eigenvalues = np.linalg.eigvalsh(covariance)
        compactness = float(
            (eigenvalues[0] + 0.35) / (eigenvalues[-1] + 0.35))
        if compactness < config.min_blob_compactness:
            continue
        radius = max(1.0, float(np.sqrt(area / np.pi)))
        color_result = _point_chroma(
            bgr, hsv, center, config, int(np.ceil(radius + 2.0)))
        if color_result is None:
            continue
        chroma, color_quality = color_result
        response = float(np.percentile(
            dog[ys.astype(int), xs.astype(int)], 90))
        if response < config.min_dog_response * 0.55:
            continue
        points.append(_LightPoint(
            center=center,
            radius=radius,
            response=response,
            shape_quality=float(np.sqrt(compactness)),
            color_quality=color_quality,
            chroma=chroma,
        ))
    return points


def _line_hypotheses(
        points: Sequence[_LightPoint],
        config: DetectorConfig) -> List[np.ndarray]:
    if len(points) < config.min_dots:
        return []
    centers = np.asarray([item.center for item in points], dtype=np.float32)
    first, second = np.triu_indices(len(centers), 1)
    vectors = centers[second] - centers[first]
    lengths = np.linalg.norm(vectors, axis=1)
    valid = np.flatnonzero(lengths >= config.min_length_pixels)
    if len(valid) == 0:
        return []
    # 较长的点对更可能覆盖完整灯带。固定计算预算可使杂乱场景的结果保持确定，
    # 并避免旧版无上限的 O(n²) 循环。
    point_quality = np.asarray(
        [item.quality for item in points], dtype=np.float32)
    pair_quality = np.sqrt(
        point_quality[first] * point_quality[second])
    pair_rank = lengths * (0.35 + 0.65 * pair_quality)
    order = valid[np.argsort(pair_rank[valid])[::-1]]
    order = order[:config.max_pair_hypotheses]
    # 旧实现逐条在 Python 中计算点到直线的距离，杂乱画面会执行数千次循环。
    # 这里分批交给 NumPy 计算，只对排名靠前的少量直线恢复点索引。
    rank_scores = np.full(len(order), -np.inf, dtype=np.float32)
    batch_size = 256
    for offset in range(0, len(order), batch_size):
        batch_pairs = order[offset:offset + batch_size]
        batch_first = first[batch_pairs]
        batch_axes = (
            vectors[batch_pairs] /
            np.maximum(lengths[batch_pairs, None], 1e-6))
        batch_normals = np.column_stack(
            (-batch_axes[:, 1], batch_axes[:, 0]))
        relative = (
            centers[None, :, :] -
            centers[batch_first][:, None, :])
        projection = np.einsum(
            'mni,mi->mn', relative, batch_axes)
        distance = np.abs(np.einsum(
            'mni,mi->mn', relative, batch_normals))
        selected_mask = (
            (distance <= config.line_distance_pixels) &
            (projection >= -config.line_distance_pixels) &
            (projection <= (
                lengths[batch_pairs, None] +
                config.line_distance_pixels)))
        counts = np.count_nonzero(selected_mask, axis=1)
        quality_sum = selected_mask.astype(np.float32) @ point_quality
        mean_quality = quality_sum / np.maximum(counts, 1)
        density = counts / np.maximum(lengths[batch_pairs], 1.0)
        scores = counts * mean_quality * np.sqrt(density)
        scores[counts < config.min_dots] = -np.inf
        rank_scores[offset:offset + len(batch_pairs)] = scores

    candidate_count = min(
        len(order), max(config.max_line_hypotheses * 8, 40))
    ranked_indexes = np.argsort(rank_scores)[::-1][:candidate_count]
    hypotheses: List[np.ndarray] = []
    seen = set()
    for ranked_index in ranked_indexes:
        if not np.isfinite(rank_scores[ranked_index]):
            continue
        pair_index = order[ranked_index]
        start_index = first[pair_index]
        axis = vectors[pair_index] / max(lengths[pair_index], 1e-6)
        normal = np.array((-axis[1], axis[0]), dtype=np.float32)
        relative = centers - centers[start_index]
        projection = relative @ axis
        distance = np.abs(relative @ normal)
        selected = np.flatnonzero(
            (distance <= config.line_distance_pixels) &
            (projection >= -config.line_distance_pixels) &
            (projection <= lengths[pair_index] +
             config.line_distance_pixels))
        key = tuple(selected.tolist())
        if key in seen:
            continue
        seen.add(key)
        hypotheses.append(selected)
        if len(hypotheses) >= config.max_line_hypotheses:
            break
    return hypotheses


def _regular_chain(
        projections: np.ndarray,
        points: Sequence[_LightPoint],
        config: DetectorConfig) -> Optional[np.ndarray]:
    """从含噪直线中选择符合透视规律的清晰子序列。"""
    count = len(projections)
    if count < config.min_dots:
        return None
    best: Optional[Tuple[Tuple[float, ...], np.ndarray]] = None
    point_qualities = np.asarray(
        [item.quality for item in points], dtype=np.float32)
    # 量化后的短程差值能够显现重复的物理间距，无需尝试把每个可能点对都作为链起点。
    gap_histogram: Dict[float, int] = {}
    max_seed_gap = max(config.min_length_pixels * 1.8, 36.0)
    for offset in range(1, min(5, count)):
        for gap in projections[offset:] - projections[:-offset]:
            gap = float(gap)
            # 这里限制的是相邻灯珠候选间距，不能直接使用整条灯带的最小长度。
            # 缩小图中真实灯珠间距可能略大于 min_length_pixels。
            if gap < 2.0 or gap > max_seed_gap:
                continue
            quantized = round(gap * 2.0) / 2.0
            gap_histogram[quantized] = gap_histogram.get(quantized, 0) + 1
    seed_gaps = [
        row[0] for row in sorted(
            gap_histogram.items(), key=lambda row: (row[1], -row[0]),
            reverse=True)[:10]
    ]
    for initial_gap in seed_gaps:
        for first in range(count):
            chain = [first]
            gaps: List[float] = []
            last = first
            expected = initial_gap
            while last + 1 < count:
                if len(gaps) >= 2:
                    recent = gaps[-5:]
                    expected = float(
                        recent[-1] +
                        (recent[-1] - recent[0]) / max(len(recent) - 1, 1))
                expected = max(expected, 1.0)
                target = projections[last] + expected
                insertion = int(np.searchsorted(
                    projections, target, side='left'))
                choices = []
                for following in range(
                        max(last + 1, insertion - 3),
                        min(count, insertion + 4)):
                    gap = float(projections[following] - projections[last])
                    normalized_error = abs(gap - expected) / expected
                    if normalized_error > 0.46:
                        continue
                    choices.append((
                        normalized_error - 0.08 * point_qualities[following],
                        following,
                        gap,
                    ))
                if not choices:
                    break
                _, following, gap = min(choices)
                chain.append(following)
                gaps.append(gap)
                last = following
            if len(chain) < config.min_dots:
                continue
            chain_array = np.asarray(chain, dtype=np.int32)
            chain_gaps = np.diff(projections[chain_array])
            span = float(
                projections[chain_array[-1]] - projections[chain_array[0]])
            if span < config.min_length_pixels:
                continue
            if best is not None:
                if len(chain_array) < best[0][0]:
                    continue
            gap_axis = np.arange(len(chain_gaps), dtype=np.float32)
            centered_axis = gap_axis - float(np.mean(gap_axis))
            slope = float(
                (centered_axis @ chain_gaps) /
                max(centered_axis @ centered_axis, 1e-6))
            intercept = float(np.mean(chain_gaps) - slope * np.mean(gap_axis))
            predicted = intercept + slope * gap_axis
            if np.any(predicted <= 0.5):
                continue
            error = float(
                np.sqrt(np.mean(np.square(chain_gaps - predicted))) /
                max(float(np.median(chain_gaps)), 1e-6))
            mean_quality = float(np.mean(point_qualities[chain_array]))
            rank = (
                float(len(chain_array)),
                -error,
                span,
                mean_quality,
            )
            row = (rank, chain_array)
            if best is None or row[0] > best[0]:
                best = row
    if best is None:
        return None
    return best[1]


def _sample_image(image: np.ndarray, positions: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    rounded = np.round(positions).astype(int)
    rounded[:, 0] = np.clip(rounded[:, 0], 0, width - 1)
    rounded[:, 1] = np.clip(rounded[:, 1], 0, height - 1)
    return image[rounded[:, 1], rounded[:, 0]].astype(np.float32)


def _sample_local_max(
        image: np.ndarray,
        positions: np.ndarray,
        radius: int = 6) -> np.ndarray:
    height, width = image.shape[:2]
    values = []
    for x_value, y_value in np.round(positions).astype(int):
        x0, x1 = max(0, x_value - radius), min(
            width, x_value + radius + 1)
        y0, y1 = max(0, y_value - radius), min(
            height, y_value + radius + 1)
        patch = image[y0:y1, x0:x1]
        values.append(float(np.max(patch)) if patch.size else 0.0)
    return np.asarray(values, dtype=np.float32)


def _chroma_distance(
        chroma: Sequence[float],
        model: ColorModel) -> float:
    delta = np.asarray(chroma, dtype=np.float32) - np.asarray(
        model.chroma_center, dtype=np.float32)
    squared = float(delta @ model.inverse_covariance @ delta)
    return float(np.sqrt(max(squared, 0.0)))


def _feature_array(model: ColorModel) -> np.ndarray:
    cached = _FEATURE_ARRAY_CACHE.get(id(model))
    if cached is None:
        cached = np.asarray(model.feature_centers, dtype=np.float32)
        _FEATURE_ARRAY_CACHE[id(model)] = cached
    return cached


def _model_distance(
        chroma: Sequence[float],
        model: ColorModel,
        hue: Optional[float] = None) -> float:
    if hue is not None and model.feature_centers:
        features = _feature_array(model)
        blue = (float(chroma[0]) - features[:, 0]) / 0.08
        green = (float(chroma[1]) - features[:, 1]) / 0.08
        hue_delta = np.abs((float(hue) - features[:, 2]) % 180.0)
        hue_term = np.minimum(hue_delta, 180.0 - hue_delta) / 14.0
        squared = blue * blue + green * green + hue_term * hue_term
        return float(np.sqrt(float(np.min(squared))))
    chroma_distance = _chroma_distance(chroma, model)
    if hue is None or model.hue_center is None:
        return chroma_distance
    hue_term = (
        _hue_distance(hue, model.hue_center) /
        max(model.hue_tolerance, 1e-6))
    return float(np.sqrt(chroma_distance ** 2 + (hue_term * 2.0) ** 2))


def _model_limit(model: ColorModel) -> float:
    if model.feature_centers and model.feature_max_distance is not None:
        return model.feature_max_distance
    return model.max_distance


def _classify_chroma(
        chroma: Sequence[float],
        config: DetectorConfig,
        hue: Optional[float] = None) -> Tuple[
            Optional[ColorModel], float, float]:
    best_model: Optional[ColorModel] = None
    best_distance = float('inf')
    second_distance = float('inf')
    for model in config.colors:
        distance = _model_distance(chroma, model, hue)
        if distance < best_distance:
            second_distance = best_distance
            best_distance = distance
            best_model = model
        elif distance < second_distance:
            second_distance = distance
    if best_model is None:
        return None, float('inf'), 0.0
    margin = second_distance / max(best_distance, 1e-6)
    if (
            best_distance > _model_limit(best_model) or
            margin < config.min_class_margin):
        return None, best_distance, margin
    return best_model, best_distance, margin


def _fit_candidate(
        points: Sequence[_LightPoint],
        indexes: np.ndarray,
        dog: np.ndarray,
        config: DetectorConfig,
        model: Optional[ColorModel]) -> Optional[StripDetection]:
    selected_points = [points[index] for index in indexes]
    selected = np.asarray(
        [item.center for item in selected_points], dtype=np.float32)
    center = np.mean(selected, axis=0)
    centered = selected - center
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    axis = vt[0].astype(np.float32)
    if axis[0] < 0 or (abs(axis[0]) < 1e-6 and axis[1] < 0):
        axis = -axis
    normal = np.array((-axis[1], axis[0]), dtype=np.float32)
    projection = centered @ axis
    residuals = np.abs(centered @ normal)
    order = np.argsort(projection)
    projection = projection[order]
    selected = selected[order]
    selected_points = [selected_points[index] for index in order]
    raw_gaps = np.diff(projection)
    raw_consistent = False
    if len(raw_gaps) >= config.min_dots - 1 and np.all(raw_gaps > 0.75):
        raw_axis = np.arange(len(raw_gaps), dtype=np.float32)
        centered_raw_axis = raw_axis - float(np.mean(raw_axis))
        raw_slope = float(
            (centered_raw_axis @ raw_gaps) /
            max(centered_raw_axis @ centered_raw_axis, 1e-6))
        raw_intercept = float(
            np.mean(raw_gaps) - raw_slope * np.mean(raw_axis))
        raw_predicted = raw_intercept + raw_slope * raw_axis
        raw_error = float(
            np.sqrt(np.mean(np.square(raw_gaps - raw_predicted))) /
            max(float(np.median(raw_gaps)), 1e-6))
        raw_consistent = (
            np.all(raw_predicted > 0.5) and
            float(np.max(raw_gaps) / np.min(raw_gaps)) <=
            config.max_gap_ratio and
            raw_error <= config.max_spacing_trend_error
        )
    if not raw_consistent:
        chain = _regular_chain(projection, selected_points, config)
        if chain is None:
            return None
        projection = projection[chain]
        selected = selected[chain]
        selected_points = [selected_points[index] for index in chain]
    center = np.mean(selected, axis=0)
    centered = selected - center
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    axis = vt[0].astype(np.float32)
    if axis[0] < 0 or (abs(axis[0]) < 1e-6 and axis[1] < 0):
        axis = -axis
    normal = np.array((-axis[1], axis[0]), dtype=np.float32)
    projection = centered @ axis
    order = np.argsort(projection)
    projection = projection[order]
    selected = selected[order]
    selected_points = [selected_points[index] for index in order]
    residuals = np.abs(centered @ normal)
    residual = float(np.percentile(residuals, 90))
    length = float(projection[-1] - projection[0])
    if length < config.min_length_pixels:
        return None
    if residual > config.line_distance_pixels * 1.25:
        return None
    gaps = np.diff(projection)
    if len(gaps) < config.min_dots - 1 or np.any(gaps <= 0.75):
        return None
    median_gap = float(np.median(gaps))
    if float(np.max(gaps) / max(np.min(gaps), 1e-6)) > config.max_gap_ratio:
        return None
    gap_indexes = np.arange(len(gaps), dtype=np.float32)
    centered_gap_indexes = gap_indexes - float(np.mean(gap_indexes))
    slope = float(
        (centered_gap_indexes @ gaps) /
        max(centered_gap_indexes @ centered_gap_indexes, 1e-6))
    intercept = float(np.mean(gaps) - slope * np.mean(gap_indexes))
    predicted_gaps = intercept + slope * gap_indexes
    if np.any(predicted_gaps <= 0.5):
        return None
    trend_error = float(
        np.sqrt(np.mean(np.square(gaps - predicted_gaps))) /
        max(median_gap, 1e-6))
    if trend_error > config.max_spacing_trend_error:
        return None
    spacing_cv = float(np.std(gaps) / max(np.mean(gaps), 1e-6))
    expected = max(length / max(float(np.mean(gaps)), 1.0) + 1.0, 1.0)
    coverage = min(1.0, len(selected) / expected)
    if coverage < config.min_coverage:
        return None

    peak_response = _sample_local_max(dog, selected)
    midpoints = (selected[:-1] + selected[1:]) * 0.5
    valley_response = _sample_image(dog, midpoints)
    pair_peaks = np.minimum(peak_response[:-1], peak_response[1:])
    contrasts = (pair_peaks - valley_response) / np.maximum(pair_peaks, 1.0)
    valley_contrast = float(np.median(np.clip(contrasts, 0.0, 1.0)))
    if valley_contrast < config.min_valley_contrast:
        return None

    line_quality = float(np.exp(
        -np.square(residual / max(config.line_distance_pixels, 1e-6))))
    trend_quality = float(np.exp(
        -np.square(trend_error /
                   max(config.max_spacing_trend_error, 1e-6))))
    periodic_quality = float(np.clip(
        trend_quality * np.sqrt(coverage), 0.0, 1.0))
    dot_quality = float(np.median(
        [item.quality for item in selected_points]))
    color_values = np.asarray(
        [item.color_quality for item in selected_points], dtype=np.float32)
    color_quality = float(np.median(color_values))
    color_consistency = float(
        np.count_nonzero(color_values >= color_quality * 0.65) /
        len(color_values))
    color_quality *= color_consistency
    valley_quality = float(np.clip(
        (valley_contrast - config.min_valley_contrast) /
        max(0.55 - config.min_valley_contrast, 1e-6),
        0.0, 1.0))
    # 此处有意使用乘法：没有独立灯珠的彩色线，或只有彩色边缘的周期性白色结构，
    # 其得分必须保持较低。
    geometry_score = float(
        line_quality * dot_quality * periodic_quality *
        color_quality * valley_quality)
    chroma_values = np.asarray(
        [item.chroma for item in selected_points], dtype=np.float32)
    candidate_chroma = np.median(chroma_values, axis=0)
    class_distance = float('inf')
    class_margin = 0.0
    color_support = 1.0
    if model is None:
        color = 'UNKNOWN'
        score = geometry_score
        confidence = float(np.sqrt(max(
            color_quality * periodic_quality, 0.0)))
    else:
        classified, class_distance, class_margin = _classify_chroma(
            candidate_chroma, config)
        if classified is None or classified.name != model.name:
            return None
        point_distances = np.asarray([
            _chroma_distance(item.chroma, model)
            for item in selected_points
        ], dtype=np.float32)
        color_support = float(np.count_nonzero(
            point_distances <= model.max_distance) / len(point_distances))
        if color_support < config.min_color_support:
            return None
        normalized_distance = class_distance / max(model.max_distance, 1e-6)
        class_likelihood = float(np.exp(-0.5 * normalized_distance ** 2))
        confidence = float(np.clip(
            class_likelihood * np.sqrt(color_support), 0.0, 1.0))
        score = geometry_score * confidence
        color = model.name
    half_width = max(
        3.0,
        residual + float(np.median(
            [item.radius for item in selected_points])) + 1.5)
    low, high = float(projection[0]), float(projection[-1])
    corners = np.array([
        center + axis * low - normal * half_width,
        center + axis * high - normal * half_width,
        center + axis * high + normal * half_width,
        center + axis * low + normal * half_width,
    ], dtype=np.float32)
    return StripDetection(
        color=color,
        confidence=confidence,
        score=score,
        corners=corners,
        dot_count=len(selected),
        length=length,
        residual=residual,
        spacing_cv=spacing_cv,
        line_quality=line_quality,
        dot_quality=dot_quality,
        periodic_quality=periodic_quality,
        color_quality=color_quality,
        valley_quality=valley_quality,
        peak_centers=selected,
        mode='periodic',
        geometry_score=geometry_score,
        class_distance=class_distance,
        class_margin=class_margin,
        color_support=color_support,
        chroma=(float(candidate_chroma[0]), float(candidate_chroma[1])),
    )


def _overlap(first: StripDetection, second: StripDetection) -> float:
    box_a = cv2.boundingRect(np.round(first.corners).astype(np.int32))
    box_b = cv2.boundingRect(np.round(second.corners).astype(np.int32))
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    intersection = max(0, min(ax + aw, bx + bw) - max(ax, bx)) * max(
        0, min(ay + ah, by + bh) - max(ay, by))
    return intersection / max(min(aw * ah, bw * bh), 1)


def _same_physical_strip(first: StripDetection, second: StripDetection) -> bool:
    def endpoints(item: StripDetection) -> np.ndarray:
        return np.array([
            (item.corners[0] + item.corners[3]) * 0.5,
            (item.corners[1] + item.corners[2]) * 0.5,
        ])

    first_ends = endpoints(first)
    second_ends = endpoints(second)
    first_axis = first_ends[1] - first_ends[0]
    second_axis = second_ends[1] - second_ends[0]
    first_axis /= max(float(np.linalg.norm(first_axis)), 1e-6)
    second_axis /= max(float(np.linalg.norm(second_axis)), 1e-6)
    if abs(float(first_axis @ second_axis)) < float(np.cos(np.deg2rad(20.0))):
        return False
    normal = np.array((-first_axis[1], first_axis[0]), dtype=np.float32)
    perpendicular = max(
        abs(float((second_ends[0] - first_ends[0]) @ normal)),
        abs(float((second_ends[1] - first_ends[0]) @ normal)),
    )
    if perpendicular > max(9.0, min(first.length, second.length) * 0.14):
        return False
    first_projection = np.sort((first_ends - first_ends[0]) @ first_axis)
    second_projection = np.sort((second_ends - first_ends[0]) @ first_axis)
    overlap = min(first_projection[1], second_projection[1]) - max(
        first_projection[0], second_projection[0])
    if overlap >= min(first.length, second.length) * 0.25:
        return True
    gap = max(first_projection[0], second_projection[0]) - min(
        first_projection[1], second_projection[1])
    # 曝光和白平衡可能把一条物理单色灯带拆成相邻的色相区域。
    # 若共线间隙不超过较短区域的大约一倍，则视为同一灯带，并仅保留更强的颜色类别。
    return gap <= max(25.0, min(first.length, second.length) * 2.50)


def _prepare_light_points(
        bgr: np.ndarray,
        config: DetectorConfig) -> Tuple[np.ndarray, List[_LightPoint]]:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    value = hsv[:, :, 2].astype(np.float32)
    smooth = cv2.GaussianBlur(
        value, (0, 0), config.dog_sigma_small)
    broad = cv2.GaussianBlur(
        value, (0, 0), config.dog_sigma_large)
    dog = smooth - broad
    bright_mask = hsv[:, :, 2] >= config.min_value
    points = _extract_light_points(
        bgr, hsv, bright_mask, dog, config)
    points.extend(_extract_saturated_components(bgr, hsv, dog, config))
    points.sort(key=lambda item: item.quality, reverse=True)
    unique: List[_LightPoint] = []
    for point in points:
        if any(
                np.linalg.norm(point.center - old.center) <
                min(24.0, max(
                    4.0, max(point.radius, old.radius) * 2.0 + 3.0))
                for old in unique):
            continue
        unique.append(point)
        if len(unique) >= config.max_points_per_color:
            break
    return dog, unique


def _suppress_duplicate_proposals(
        raw: Sequence[StripDetection]) -> List[StripDetection]:
    ordered = sorted(raw, key=lambda item: item.score, reverse=True)
    kept: List[StripDetection] = []
    for candidate in ordered:
        if any(
                _overlap(candidate, existing) >= 0.55 or
                _same_physical_strip(candidate, existing) or
                _same_physical_strip(existing, candidate)
                for existing in kept):
            continue
        kept.append(candidate)
    return kept


def sample_candidate_chroma(
        bgr: np.ndarray,
        candidate: StripDetection,
        config: DetectorConfig) -> Optional[
            Tuple[Tuple[float, float], float]]:
    """在候选灯带内部提取去亮度化色度，并返回彩色像素比例。"""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    region = np.zeros(hsv.shape[:2], dtype=np.uint8)
    corners = np.round(candidate.corners).astype(np.int32)
    cv2.fillConvexPoly(region, corners, 1)
    region = cv2.dilate(
        region,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
    inside = region.astype(bool)
    bright = inside & (hsv[:, :, 2] >= config.min_value)
    bright_count = int(np.count_nonzero(bright))
    if bright_count == 0:
        return None
    colored = (
        bright &
        (hsv[:, :, 1] >= config.min_saturation))
    colored_count = int(np.count_nonzero(colored))
    colored_fraction = colored_count / bright_count
    if colored_count == 0 or colored_fraction < config.min_colored_fraction:
        return None
    values = hsv[:, :, 2][colored]
    # 低亮度彩色光晕受曝光和背景影响较大；最高亮的一成像素更接近 LED 发光核心，
    # 与整幅标定图使用的特征保持一致。
    cutoff = float(np.percentile(values, 90))
    selected = colored & (hsv[:, :, 2] >= cutoff)
    pixels = bgr[selected].astype(np.float32)
    normalized = pixels / np.maximum(
        np.sum(pixels, axis=1, keepdims=True), 1.0)
    chroma = np.median(normalized[:, :2], axis=0)
    return (
        (float(chroma[0]), float(chroma[1])),
        float(colored_fraction),
    )


def _classify_geometry_candidate(
        bgr: np.ndarray,
        candidate: StripDetection,
        config: DetectorConfig) -> Optional[StripDetection]:
    sampled = sample_candidate_chroma(bgr, candidate, config)
    if sampled is None:
        return None
    chroma, colored_fraction = sampled
    model, distance, margin = _classify_chroma(chroma, config)
    if model is None:
        return None
    normalized_distance = distance / max(model.max_distance, 1e-6)
    confidence = float(np.exp(-0.5 * normalized_distance ** 2))
    return replace(
        candidate,
        color=model.name,
        confidence=confidence,
        score=candidate.geometry_score * confidence,
        class_distance=distance,
        class_margin=margin,
        color_support=colored_fraction,
        chroma=chroma,
    )


def _scaled_fast_config(
        config: DetectorConfig,
        scale: float) -> DetectorConfig:
    """把按原图调好的几何阈值映射到快速检测用的缩小图。"""
    if scale >= 0.999:
        return config
    area_scale = scale * scale
    return replace(
        config,
        min_blob_area=max(1, int(round(config.min_blob_area * area_scale))),
        max_blob_area=max(8, int(round(config.max_blob_area * area_scale))),
        min_length_pixels=max(
            12.0, config.min_length_pixels * scale),
        line_distance_pixels=max(
            1.8, config.line_distance_pixels * scale),
        min_dog_response=config.min_dog_response * scale,
    )


def _resize_for_fast_path(
        bgr: np.ndarray,
        config: DetectorConfig) -> Tuple[np.ndarray, float]:
    """固定快速路径的处理宽度，避免 720p 图像上做全量几何搜索。"""
    height, width = bgr.shape[:2]
    target_width = max(1, config.fast_resize_width)
    if width <= target_width:
        return bgr, 1.0
    scale = target_width / float(width)
    resized = cv2.resize(
        bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return resized, scale


def _extract_fast_light_points(
        bgr: np.ndarray,
        config: DetectorConfig) -> List[_LightPoint]:
    """直接从高饱和小光斑提取灯珠中心，避免旧版 DoG 全图搜索。"""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = (
        (hsv[:, :, 1] >= config.min_saturation) &
        (hsv[:, :, 2] >= config.min_value))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), 8)
    points: List[_LightPoint] = []

    def add_point(
            absolute_x: np.ndarray,
            absolute_y: np.ndarray,
            area: int,
            width: int,
            height: int,
            fill: float,
            allowed_colors: Optional[set] = None) -> bool:
        if len(absolute_x) < 2:
            return False
        values = hsv[
            absolute_y.astype(np.int32),
            absolute_x.astype(np.int32),
            2,
        ].astype(np.float32)
        weights = np.maximum(values, 1.0)
        center = np.array([
            np.average(absolute_x, weights=weights),
            np.average(absolute_y, weights=weights),
        ], dtype=np.float32)
        saturation = hsv[
            absolute_y.astype(np.int32),
            absolute_x.astype(np.int32),
            1,
        ].astype(np.float32)
        saturation_max = float(np.max(saturation))
        value_max = float(np.max(values))
        saturation_cutoff = max(
            config.min_saturation, saturation_max * 0.55)
        value_low = max(config.min_value, value_max * 0.35)
        # LED 核心容易过曝偏白；取高饱和中高亮区域，避免逐点分位数计算造成卡顿。
        core = (
            (saturation >= saturation_cutoff) &
            (values >= value_low))
        if np.count_nonzero(core) < 2:
            core = saturation >= max(
                config.min_saturation, saturation_max * 0.45)
        if np.count_nonzero(core) < 2:
            core = np.ones(len(absolute_x), dtype=bool)
        pixels = bgr[
            absolute_y[core].astype(np.int32),
            absolute_x[core].astype(np.int32),
        ].astype(np.float32)
        if len(pixels) == 0:
            return False
        hue = _mean_hue(
            hsv[
                absolute_y[core].astype(np.int32),
                absolute_x[core].astype(np.int32),
                0,
            ].astype(np.float32).tolist())
        normalized = pixels / np.maximum(
            np.sum(pixels, axis=1, keepdims=True), 1.0)
        chroma = np.mean(normalized[:, :2], axis=0)
        if allowed_colors is not None:
            model, _, _ = _classify_chroma(chroma, config, hue)
            if model is None or model.name not in allowed_colors:
                return False
        box_compactness = min(width, height) / max(max(width, height), 1)
        shape_quality = float(np.sqrt(np.clip(
            box_compactness * min(fill / 0.55, 1.0), 0.0, 1.0)))
        color_quality = float(np.clip(
            np.sqrt(
                min(float(np.mean(saturation)) / 180.0, 1.0) *
                min(float(np.mean(values)) / 180.0, 1.0)),
            0.0,
            1.0,
        ))
        points.append(_LightPoint(
            center=center,
            radius=max(1.0, float(np.sqrt(area / np.pi))),
            response=value_max,
            shape_quality=shape_quality,
            color_quality=color_quality,
            chroma=(float(chroma[0]), float(chroma[1])),
            hue=hue,
        ))
        return True

    for label in range(1, count):
        x, y, width, height, area = stats[label]
        if area < max(config.min_blob_area, 1):
            continue
        aspect = max(width, height) / max(min(width, height), 1)
        fill = area / max(width * height, 1)
        local = labels[y:y + height, x:x + width] == label
        ys, xs = np.nonzero(local)
        if len(xs) < 2:
            continue
        oversized = (
            area > max(config.max_blob_area * 3, 24) or
            aspect > config.max_blob_aspect or
            fill < 0.16)
        if oversized:
            values_patch = hsv[y:y + height, x:x + width, 2]
            saturation_patch = hsv[y:y + height, x:x + width, 1]
            value_data = values_patch[local].astype(np.float32)
            saturation_data = saturation_patch[local].astype(np.float32)
            if len(value_data) == 0:
                continue
            # 光晕可能把一串灯珠连成一个大组件；用较亮的局部峰值重新切分。
            value_cutoff = max(
                config.min_value,
                float(np.mean(value_data)) +
                (float(np.max(value_data)) - float(np.mean(value_data))) *
                0.30)
            saturation_cutoff = max(
                config.min_saturation,
                float(np.max(saturation_data)) * 0.45)
            peak_mask = (
                local &
                (values_patch >= value_cutoff) &
                (saturation_patch >= saturation_cutoff))
            sub_count, sub_labels, sub_stats, _ = (
                cv2.connectedComponentsWithStats(
                    peak_mask.astype(np.uint8), 8))
            added = 0
            for sub_label in range(1, sub_count):
                sx, sy, sw, sh, sub_area = sub_stats[sub_label]
                if sub_area < 2 or sub_area > max(config.max_blob_area, 32):
                    continue
                sub_aspect = max(sw, sh) / max(min(sw, sh), 1)
                if sub_aspect > config.max_blob_aspect:
                    continue
                sub_local = sub_labels[
                    sy:sy + sh, sx:sx + sw] == sub_label
                sub_y, sub_x = np.nonzero(sub_local)
                sub_fill = sub_area / max(sw * sh, 1)
                if add_point(
                        sub_x.astype(np.float32) + x + sx,
                        sub_y.astype(np.float32) + y + sy,
                        int(sub_area),
                        int(sw),
                        int(sh),
                        float(sub_fill),
                        {'BLUE', 'PURPLE'}):
                    added += 1
            if added:
                continue
            continue
        absolute_x = xs.astype(np.float32) + x
        absolute_y = ys.astype(np.float32) + y
        add_point(
            absolute_x,
            absolute_y,
            int(area),
            int(width),
            int(height),
            float(fill))
    points.sort(key=lambda item: item.quality, reverse=True)
    return points


def _fit_fast_candidate(
        points: Sequence[_LightPoint],
        indexes: np.ndarray,
        config: DetectorConfig,
        model: Optional[ColorModel],
        value_image: Optional[np.ndarray] = None) -> Optional[StripDetection]:
    """在同色灯珠集合中快速拟合一条独立、等间距的灯带。"""
    selected_points = [points[index] for index in indexes]
    if len(selected_points) < config.min_dots:
        return None
    selected = np.asarray(
        [item.center for item in selected_points], dtype=np.float32)
    center = np.mean(selected, axis=0)
    centered = selected - center
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    axis = vt[0].astype(np.float32)
    if axis[0] < 0 or (abs(axis[0]) < 1e-6 and axis[1] < 0):
        axis = -axis
    normal = np.array((-axis[1], axis[0]), dtype=np.float32)
    projection = centered @ axis
    order = np.argsort(projection)
    projection = projection[order]
    selected = selected[order]
    selected_points = [selected_points[index] for index in order]

    raw_gaps = np.diff(projection)
    raw_consistent = False
    if len(raw_gaps) >= config.min_dots - 1 and np.all(raw_gaps > 0.75):
        gap_axis = np.arange(len(raw_gaps), dtype=np.float32)
        centered_gap_axis = gap_axis - float(np.mean(gap_axis))
        slope = float(
            (centered_gap_axis @ raw_gaps) /
            max(centered_gap_axis @ centered_gap_axis, 1e-6))
        intercept = float(np.mean(raw_gaps) - slope * np.mean(gap_axis))
        predicted = intercept + slope * gap_axis
        trend_error = float(
            np.sqrt(np.mean(np.square(raw_gaps - predicted))) /
            max(float(np.median(raw_gaps)), 1e-6))
        raw_consistent = (
            np.all(predicted > 0.5) and
            float(np.max(raw_gaps) / max(np.min(raw_gaps), 1e-6)) <=
            config.max_gap_ratio and
            trend_error <= config.max_spacing_trend_error
        )
    if not raw_consistent:
        chain = _regular_chain(projection, selected_points, config)
        if chain is None:
            return None
        projection = projection[chain]
        selected = selected[chain]
        selected_points = [selected_points[index] for index in chain]

    center = np.mean(selected, axis=0)
    centered = selected - center
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    axis = vt[0].astype(np.float32)
    if axis[0] < 0 or (abs(axis[0]) < 1e-6 and axis[1] < 0):
        axis = -axis
    normal = np.array((-axis[1], axis[0]), dtype=np.float32)
    projection = centered @ axis
    order = np.argsort(projection)
    projection = projection[order]
    selected = selected[order]
    selected_points = [selected_points[index] for index in order]
    residuals = np.abs(centered @ normal)
    residual = float(np.percentile(residuals, 90))
    length = float(projection[-1] - projection[0])
    if length < config.min_length_pixels:
        return None
    if residual > config.line_distance_pixels * 1.35:
        return None
    gaps = np.diff(projection)
    if len(gaps) < config.min_dots - 1 or np.any(gaps <= 0.75):
        return None
    median_gap = float(np.median(gaps))
    if float(np.max(gaps) / max(np.min(gaps), 1e-6)) > config.max_gap_ratio:
        return None
    gap_axis = np.arange(len(gaps), dtype=np.float32)
    centered_gap_axis = gap_axis - float(np.mean(gap_axis))
    slope = float(
        (centered_gap_axis @ gaps) /
        max(centered_gap_axis @ centered_gap_axis, 1e-6))
    intercept = float(np.mean(gaps) - slope * np.mean(gap_axis))
    predicted_gaps = intercept + slope * gap_axis
    if np.any(predicted_gaps <= 0.5):
        return None
    trend_error = float(
        np.sqrt(np.mean(np.square(gaps - predicted_gaps))) /
        max(median_gap, 1e-6))
    if trend_error > config.max_spacing_trend_error:
        return None
    expected = max(length / max(float(np.mean(gaps)), 1.0) + 1.0, 1.0)
    coverage = min(1.0, len(selected) / expected)
    if coverage < config.min_coverage:
        return None

    radii = np.asarray([item.radius for item in selected_points], dtype=np.float32)
    radius_quality = float(np.clip(
        (median_gap - float(np.median(radii)) * 1.4) /
        max(median_gap, 1e-6),
        0.0,
        1.0,
    ))
    line_quality = float(np.exp(
        -np.square(residual / max(config.line_distance_pixels, 1e-6))))
    trend_quality = float(np.exp(
        -np.square(trend_error /
                   max(config.max_spacing_trend_error, 1e-6))))
    periodic_quality = float(np.clip(
        trend_quality * np.sqrt(coverage) * np.sqrt(radius_quality),
        0.0,
        1.0,
    ))
    valley_quality = 1.0
    reported_valley_quality = radius_quality
    if value_image is not None and len(selected) >= 2:
        sample_radius = max(2, int(round(float(np.median(radii)) * 1.3)))
        peak_response = _sample_local_max(
            value_image, selected, radius=sample_radius)
        midpoints = (selected[:-1] + selected[1:]) * 0.5
        valley_response = _sample_local_max(
            value_image, midpoints, radius=max(1, sample_radius // 2))
        pair_peaks = np.minimum(peak_response[:-1], peak_response[1:])
        contrasts = (
            (pair_peaks - valley_response) /
            np.maximum(pair_peaks, 1.0))
        valley_contrast = float(np.median(np.clip(contrasts, 0.0, 1.0)))
        if valley_contrast < config.min_valley_contrast:
            return None
        valley_quality = float(np.clip(
            (valley_contrast - config.min_valley_contrast) /
            max(0.55 - config.min_valley_contrast, 1e-6),
            0.0,
            1.0,
        ))
        reported_valley_quality = valley_quality
    dot_quality = float(np.median(
        [item.quality for item in selected_points]))
    color_quality = float(np.median(
        [item.color_quality for item in selected_points]))
    chroma_values = np.asarray(
        [item.chroma for item in selected_points], dtype=np.float32)
    candidate_chroma = np.median(chroma_values, axis=0)
    candidate_hue = _mean_hue([
        item.hue for item in selected_points if item.hue is not None])
    classified, class_distance, class_margin = _classify_chroma(
        candidate_chroma, config, candidate_hue)
    if classified is None:
        return None
    if model is not None and classified.name != model.name:
        return None
    model = classified
    point_distances = np.asarray([
        _model_distance(item.chroma, model, item.hue)
        for item in selected_points
    ], dtype=np.float32)
    color_support = float(np.count_nonzero(
        point_distances <= _model_limit(model)) / len(point_distances))
    if color_support < config.min_color_support:
        return None
    normalized_distance = class_distance / max(_model_limit(model), 1e-6)
    confidence = float(np.clip(
        np.exp(-0.5 * normalized_distance ** 2) *
        np.sqrt(color_support),
        0.0,
        1.0,
    ))
    geometry_score = float(
        line_quality * dot_quality * periodic_quality *
        color_quality * valley_quality)
    if geometry_score < config.min_geometry_score:
        return None
    score = geometry_score * confidence
    spacing_cv = float(np.std(gaps) / max(np.mean(gaps), 1e-6))
    half_width = max(
        2.5,
        residual + float(np.median(radii)) + 1.5)
    low, high = float(projection[0]), float(projection[-1])
    corners = np.array([
        center + axis * low - normal * half_width,
        center + axis * high - normal * half_width,
        center + axis * high + normal * half_width,
        center + axis * low + normal * half_width,
    ], dtype=np.float32)
    return StripDetection(
        color=model.name,
        confidence=confidence,
        score=score,
        corners=corners,
        dot_count=len(selected),
        length=length,
        residual=residual,
        spacing_cv=spacing_cv,
        line_quality=line_quality,
        dot_quality=dot_quality,
        periodic_quality=periodic_quality,
        color_quality=color_quality,
        valley_quality=reported_valley_quality,
        peak_centers=selected,
        mode='fast',
        geometry_score=geometry_score,
        class_distance=class_distance,
        class_margin=class_margin,
        color_support=color_support,
        chroma=(float(candidate_chroma[0]), float(candidate_chroma[1])),
    )


def _detect_fast_proposals(
        bgr: np.ndarray,
        config: DetectorConfig) -> List[StripDetection]:
    work, scale = _resize_for_fast_path(bgr, config)
    fast_config = _scaled_fast_config(config, scale)
    value_image = None
    if fast_config.fast_check_valley:
        value_image = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)[:, :, 2].astype(
            np.float32)
    points = _extract_fast_light_points(work, fast_config)
    groups: Dict[str, List[_LightPoint]] = {
        model.name: [] for model in fast_config.colors}
    for point in points:
        model, _, _ = _classify_chroma(point.chroma, fast_config, point.hue)
        if model is not None:
            groups[model.name].append(point)
    raw: List[StripDetection] = []
    if fast_config.fast_global_search:
        # 默认关闭：天花板灯阵等 NONE 场景也可能形成颜色无关的共线亮点。
        for indexes in _line_hypotheses(points, fast_config):
            candidate = _fit_fast_candidate(
                points, indexes, fast_config, None, value_image)
            if candidate is not None:
                raw.append(candidate)
    for model in fast_config.colors:
        color_points = groups[model.name][:fast_config.fast_max_points_per_color]
        if len(color_points) < fast_config.min_dots:
            continue
        for indexes in _line_hypotheses(color_points, fast_config):
            candidate = _fit_fast_candidate(
                color_points, indexes, fast_config, model, value_image)
            if candidate is not None:
                raw.append(candidate)
    proposals = _suppress_duplicate_proposals(raw)
    if scale < 0.999:
        inverse = 1.0 / scale
        proposals = [item.scaled(inverse) for item in proposals]
    return proposals


def _search_configs(
        config: DetectorConfig,
        allow_fallback: bool = True) -> Sequence[DetectorConfig]:
    rows = [config]
    if allow_fallback and (
            config.fallback_max_pair_hypotheses >
            config.max_pair_hypotheses or
            config.fallback_max_line_hypotheses >
            config.max_line_hypotheses):
        rows.append(replace(
            config,
            max_pair_hypotheses=max(
                config.max_pair_hypotheses,
                config.fallback_max_pair_hypotheses),
            max_line_hypotheses=max(
                config.max_line_hypotheses,
                config.fallback_max_line_hypotheses),
        ))
    return rows


def detect_geometry_proposals(
        bgr: np.ndarray,
        config: DetectorConfig) -> List[StripDetection]:
    """返回不依赖颜色类别的灯珠排列候选，供标定流程使用。"""
    if bgr is None or bgr.size == 0:
        return []
    dog, points = _prepare_light_points(bgr, config)
    allow_fallback = sum(
        point.quality >= 0.50 for point in points) >= 24
    raw: List[StripDetection] = []
    for search_config in _search_configs(config, allow_fallback):
        attempt: List[StripDetection] = []
        for indexes in _line_hypotheses(points, search_config):
            candidate = _fit_candidate(
                points, indexes, dog, config, None)
            if candidate is None:
                continue
            if candidate.dot_quality < config.min_periodic_dot_quality:
                continue
            if candidate.color_quality < config.min_periodic_color_quality:
                continue
            if candidate.geometry_score < config.min_geometry_score:
                continue
            attempt.append(candidate)
        if attempt:
            raw.extend(attempt)
            break
    return _suppress_duplicate_proposals(raw)


def detect_proposals(
        bgr: np.ndarray,
        config: DetectorConfig) -> List[StripDetection]:
    """返回应用最终分数阈值前、颜色和结构均有效的候选。"""
    if bgr is None or bgr.size == 0:
        return []
    fast = _detect_fast_proposals(bgr, config)
    if fast or not config.fast_enable_fallback:
        return fast

    dog, points = _prepare_light_points(bgr, config)
    allow_fallback = sum(
        point.quality >= 0.50 for point in points) >= 24
    groups: Dict[str, List[_LightPoint]] = {
        model.name: [] for model in config.colors}
    for point in points:
        model, _, _ = _classify_chroma(point.chroma, config)
        if model is not None:
            groups[model.name].append(point)

    raw: List[StripDetection] = []
    # 单色训练画面和被局部过曝切碎的灯带，先按完整几何链拟合，再整体分类。
    # 若真实画面包含两段不同颜色，合并链的色度通常落在类别间隙并会被开放集规则拒绝。
    for search_config in _search_configs(config, allow_fallback):
        attempt: List[StripDetection] = []
        for indexes in _line_hypotheses(points, search_config):
            geometry = _fit_candidate(
                points, indexes, dog, config, None)
            if geometry is None:
                continue
            if geometry.dot_quality < config.min_periodic_dot_quality:
                continue
            if geometry.color_quality < config.min_periodic_color_quality:
                continue
            if geometry.geometry_score < config.min_geometry_score:
                continue
            classified = _classify_geometry_candidate(
                bgr, geometry, config)
            if classified is not None:
                attempt.append(classified)
        if attempt:
            raw.extend(attempt)
            break

    # 同时按逐灯珠类别建立候选，保证双色通信中的两段不会被合并。
    for model in config.colors:
        color_points = groups[model.name]
        for search_config in _search_configs(config, allow_fallback):
            attempt = []
            for indexes in _line_hypotheses(color_points, search_config):
                candidate = _fit_candidate(
                    color_points, indexes, dog, config, model)
                if candidate is None:
                    continue
                if candidate.dot_quality < config.min_periodic_dot_quality:
                    continue
                if candidate.color_quality < config.min_periodic_color_quality:
                    continue
                if candidate.geometry_score < config.min_geometry_score:
                    continue
                attempt.append(candidate)
            if attempt:
                raw.extend(attempt)
                break
    return _suppress_duplicate_proposals(raw)


def detect_candidates(
        bgr: np.ndarray,
        config: DetectorConfig) -> List[StripDetection]:
    return [
        item for item in detect_proposals(bgr, config)
        if item.score >= config.min_score
    ]


def select_winner(
        candidates: Sequence[StripDetection],
        config: DetectorConfig) -> Optional[StripDetection]:
    if not candidates:
        return None
    if len(candidates) > 1:
        # 同色反光不会造成颜色类别判定歧义。
        different = next(
            (item for item in candidates[1:]
             if item.color != candidates[0].color),
            None)
        if different is not None:
            ratio = candidates[0].score / max(different.score, 1e-9)
            if ratio < config.winner_margin:
                return None
    return candidates[0]


def detection_metrics(item: StripDetection) -> Mapping[str, float]:
    return {
        'line': item.line_quality,
        'dots': item.dot_quality,
        'periodic': item.periodic_quality,
        'color': item.color_quality,
        'valley': item.valley_quality,
        'geometry': item.geometry_score,
        'class_distance': item.class_distance,
        'class_margin': item.class_margin,
        'color_support': item.color_support,
    }


def annotate(
        bgr: np.ndarray,
        candidates: Sequence[StripDetection],
        winner: Optional[StripDetection]) -> np.ndarray:
    output = bgr.copy()
    for rank, candidate in enumerate(candidates, 1):
        selected = candidate is winner
        draw_color = (0, 255, 255) if selected else (255, 180, 0)
        corners = np.round(candidate.corners).astype(np.int32)
        cv2.polylines(output, [corners], True, draw_color, 2)
        if candidate.peak_centers is not None:
            for point in np.round(candidate.peak_centers).astype(np.int32):
                cv2.circle(output, tuple(point), 3, draw_color, 1)
        anchor = tuple(corners[np.argmin(corners[:, 1])])
        text = (
            f'#{rank} {candidate.color} score={candidate.score:.3f} '
            f'conf={candidate.confidence:.3f} dots={candidate.dot_count} '
            f'{candidate.mode[0].upper()} '
            f'D={candidate.class_distance:.2f} '
            f'M={candidate.class_margin:.2f} '
            f'G={candidate.geometry_score:.2f}')
        cv2.putText(
            output, text, anchor, cv2.FONT_HERSHEY_SIMPLEX,
            0.48, draw_color, 2)
    if winner is None:
        title = f'NO DETECTION ({len(candidates)} candidates)'
    else:
        different = next(
            (item for item in candidates[1:] if item.color != winner.color),
            None)
        margin = (
            winner.score / max(different.score, 1e-9)
            if different is not None else float('inf'))
        title = (
            f'SELECTED {winner.color} confidence={winner.confidence:.3f} '
            f'margin={margin:.2f}')
    cv2.putText(
        output, title, (12, 30), cv2.FONT_HERSHEY_SIMPLEX,
        0.72, (255, 255, 255), 2)
    return output
