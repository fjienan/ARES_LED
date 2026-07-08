#!/usr/bin/env python3
"""USB RGB 1 三段四色灯带离线识别：不依赖 ROS2。"""

import argparse
from dataclasses import replace
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace
import sys
import time
from typing import List, Optional

import cv2
import numpy as np
import yaml

from detect_usb_rgb_2_three_segment import (
    REPO_ROOT,
    R2_PACKAGE,
    WeakStripDetection,
    annotate_three_segments,
    build_three_segment_candidate,
    deduplicate_three_segment_candidates,
    image_paths,
)

sys.path.insert(0, str(R2_PACKAGE))

from rgb_camera_receiver.classifier import classifier_for_profile  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description='Detect three-color LED-strip codes in usb_rgb_1 images.')
    parser.add_argument(
        '--input',
        default=str(REPO_ROOT / 'camera_data' / 'usb_rgb_1' / 'combined'),
        help='输入图片目录；默认 camera_data/usb_rgb_1/combined')
    parser.add_argument(
        '--output',
        default=str(
            REPO_ROOT / 'camera_capture_results' / 'usb_rgb_1' / 'combined'),
        help='标注图片输出目录；默认 camera_capture_results/usb_rgb_1/combined')
    parser.add_argument(
        '--config',
        default=str(
            R2_PACKAGE / 'config' / 'cameras' / 'usb_rgb_1' / 'detector.yaml'),
        help='usb_rgb_1 detector.yaml 路径')
    parser.add_argument(
        '--processing-scale', type=float,
        help='检测前缩放比例；默认读取 detector.yaml 的 processing.scale')
    parser.add_argument('--max-single-candidates', type=int)
    parser.add_argument('--max-results', type=int)
    parser.add_argument('--min-three-score', type=float)
    parser.add_argument('--winner-margin', type=float)
    parser.add_argument('--max-angle-degrees', type=float)
    parser.add_argument('--max-cross-distance', type=float)
    parser.add_argument('--min-center-distance-ratio', type=float)
    parser.add_argument('--max-center-distance-ratio', type=float)
    parser.add_argument('--max-gap-ratio', type=float)
    parser.add_argument('--fast-dilate-pixels', type=int)
    parser.add_argument('--fast-min-component-area', type=int)
    parser.add_argument('--fast-max-component-area', type=int)
    parser.add_argument('--fast-min-color-pixels', type=int)
    parser.add_argument('--fast-min-segment-length', type=float)
    parser.add_argument('--fast-min-union-aspect', type=float)
    parser.add_argument('--fast-max-segments-per-color', type=int)
    parser.add_argument('--jpeg-quality', type=int, default=95)
    return parser.parse_args()


def load_detector(config_path: Path):
    classifier = classifier_for_profile('usb_rgb_1')
    config = classifier.load_config(str(config_path))
    return classifier, config


def scaled_frame(frame, scale: float):
    if scale >= 0.999:
        return frame
    return cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def read_three_segment_config(config_path: Path):
    with config_path.open('r', encoding='utf-8') as stream:
        raw = yaml.safe_load(stream) or {}
    return raw.get('three_segment', {}) or {}


def config_value(args, config_rows, name: str, default):
    value = getattr(args, name, None)
    if value is not None:
        return value
    return config_rows.get(name, default)


def line_segment_from_color_pixels(
        color: str,
        mask: np.ndarray,
        component_mask: np.ndarray,
        scale: float,
        args) -> Optional[WeakStripDetection]:
    selected = cv2.bitwise_and(mask.astype(np.uint8), component_mask)
    count, labels, stats, centers = cv2.connectedComponentsWithStats(selected, 8)
    if count <= 1:
        return None

    components = []
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area < 1:
            continue
        components.append((area, index))
    if not components:
        return None
    pixel_count = sum(area for area, _ in components)
    if pixel_count < args.fast_min_color_pixels:
        return None

    points = []
    component_centers = []
    for area, index in components:
        ys, xs = np.nonzero(labels == index)
        if len(xs) == 0:
            continue
        points.append(np.column_stack([xs, ys]).astype(np.float32))
        if area >= 2:
            component_centers.append(centers[index].astype(np.float32))
    if not points:
        return None
    points = np.concatenate(points, axis=0)
    if len(points) < args.fast_min_color_pixels:
        return None

    center = points.mean(axis=0)
    centered = points - center
    if len(points) >= 2:
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        axis = vt[0].astype(np.float32)
        axis /= max(float(np.linalg.norm(axis)), 1e-6)
        normal = np.array((-axis[1], axis[0]), dtype=np.float32)
        projection = centered @ axis
        cross = centered @ normal
        residual = float(np.sqrt(np.mean(np.square(cross))))
    else:
        axis = np.array((1.0, 0.0), dtype=np.float32)
        normal = np.array((0.0, 1.0), dtype=np.float32)
        projection = np.zeros((len(points),), dtype=np.float32)
        cross = np.zeros((len(points),), dtype=np.float32)
        residual = 0.0
    length = float(np.max(projection) - np.min(projection))
    if length < args.fast_min_segment_length:
        return None
    cross_width = float(np.percentile(np.abs(cross), 90) * 2.0 + 3.0)
    thickness = max(cross_width, 3.0)
    start = center + axis * float(np.min(projection))
    end = center + axis * float(np.max(projection))
    half_width = normal * (thickness * 0.5)
    corners = np.stack([
        start - half_width,
        end - half_width,
        end + half_width,
        start + half_width,
    ], axis=0).astype(np.float32)
    aspect = length / thickness
    aspect_quality = float(np.clip((aspect - 1.2) / 5.0, 0.0, 1.0))
    pixel_quality = float(np.clip(pixel_count / 120.0, 0.0, 1.0))
    length_quality = float(np.clip(length / 70.0, 0.0, 1.0))
    score = float(np.clip(
        0.28 + 0.72 *
        (0.35 + 0.65 * aspect_quality) *
        (0.35 + 0.65 * pixel_quality) *
        (0.35 + 0.65 * length_quality),
        0.0, 1.0))

    inverse = 1.0 / scale
    peaks = None
    if component_centers:
        peaks = np.stack(component_centers, axis=0) * inverse
    return WeakStripDetection(
        color=color,
        confidence=float(np.sqrt(score)),
        score=score,
        corners=corners * inverse,
        dot_count=len(component_centers),
        length=length * inverse,
        residual=residual * inverse,
        spacing_cv=0.0,
        line_quality=aspect_quality,
        dot_quality=float(np.clip(len(component_centers) / 6.0, 0.0, 1.0)),
        periodic_quality=length_quality,
        color_quality=pixel_quality,
        valley_quality=0.0,
        peak_centers=peaks,
        mode='three_fast',
    )


def segments_from_color_components(
        masks,
        processing_scale: float,
        args) -> List[WeakStripDetection]:
    """分别从每种颜色的连通区域生成单段候选，不要求三段彼此贴近。"""
    kernel_size = max(1, int(args.fast_dilate_pixels))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    candidates: List[WeakStripDetection] = []

    for color, mask in masks.items():
        if int(mask.sum()) < args.fast_min_color_pixels:
            continue
        grouped = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(grouped, 8)
        color_candidates: List[WeakStripDetection] = []
        for index in range(1, count):
            area = int(stats[index, cv2.CC_STAT_AREA])
            if area < args.fast_min_component_area:
                continue
            if (
                    args.fast_max_component_area > 0 and
                    area > args.fast_max_component_area):
                continue
            x, y, width, height = stats[index, :4]
            aspect = max(width, height) / max(float(min(width, height)), 1.0)
            if aspect < args.fast_min_union_aspect:
                continue
            component_mask = (labels == index).astype(np.uint8)
            segment = line_segment_from_color_pixels(
                color, mask, component_mask, processing_scale, args)
            if segment is not None:
                color_candidates.append(segment)
        color_candidates.sort(key=lambda item: item.score, reverse=True)
        limit = max(1, int(args.fast_max_segments_per_color))
        candidates.extend(color_candidates[:limit])
    candidates.sort(key=lambda item: item.score, reverse=True)
    return candidates


def fast_single_segments(image, classifier, config, processing_scale: float, args):
    work = scaled_frame(image, processing_scale)
    hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)
    masks = {
        color: mask.astype(np.uint8)
        for color, mask in classifier.color_masks(hsv, work, config).items()
    }
    active_colors = [
        color for color, mask in masks.items()
        if int(mask.sum()) >= args.fast_min_color_pixels
    ]
    if len(active_colors) < 3:
        return []
    if sum(int(mask.sum()) for mask in masks.values()) < args.fast_min_color_pixels * 3:
        return []
    candidates = segments_from_color_components(
        masks, processing_scale, args)
    candidates.sort(key=lambda item: item.score, reverse=True)
    return candidates[:max(args.max_single_candidates, 3)]


def detect_frame(image, classifier, config, processing_scale: float, args):
    single_candidates = fast_single_segments(
        image, classifier, config, processing_scale, args)
    three_candidates = detect_three_segments(single_candidates, args)
    return single_candidates, [], single_candidates, three_candidates


def detect_three_segments(single_candidates, args):
    """组合三段候选；usb_rgb_1 当前协议要求三段颜色全不同，排除 ABA。"""
    limited = list(single_candidates[:max(args.max_single_candidates, 3)])
    raw = []
    for triple in combinations(limited, 3):
        if len({item.color for item in triple}) != 3:
            continue
        candidate = build_three_segment_candidate(triple, args)
        if candidate is not None:
            raw.append(candidate)

    candidates = deduplicate_three_segment_candidates(raw)
    candidates = candidates[:max(args.max_results, 1)]
    if len(candidates) >= 2:
        margin = candidates[0].score / max(candidates[1].score, 1e-9)
        if margin < args.winner_margin:
            candidates[0] = replace(candidates[0], ambiguous=True)
    return candidates


def three_segment_args(args, config_path: Path):
    rows = read_three_segment_config(config_path)
    return SimpleNamespace(
        max_single_candidates=int(config_value(
            args, rows, 'max_single_candidates', 30)),
        max_results=int(config_value(args, rows, 'max_results', 8)),
        min_three_score=float(config_value(
            args, rows, 'min_three_score', 0.08)),
        winner_margin=float(config_value(args, rows, 'winner_margin', 1.2)),
        max_angle_degrees=float(config_value(
            args, rows, 'max_angle_degrees', 18.0)),
        max_cross_distance=float(config_value(
            args, rows, 'max_cross_distance', 45.0)),
        min_center_distance_ratio=float(config_value(
            args, rows, 'min_center_distance_ratio', 0.25)),
        max_center_distance_ratio=float(config_value(
            args, rows, 'max_center_distance_ratio', 4.5)),
        max_gap_ratio=float(config_value(args, rows, 'max_gap_ratio', 3.4)),
        fast_dilate_pixels=int(config_value(
            args, rows, 'fast_dilate_pixels', 5)),
        fast_min_component_area=int(config_value(
            args, rows, 'fast_min_component_area', 70)),
        fast_max_component_area=int(config_value(
            args, rows, 'fast_max_component_area', 8000)),
        fast_min_color_pixels=int(config_value(
            args, rows, 'fast_min_color_pixels', 20)),
        fast_min_segment_length=float(config_value(
            args, rows, 'fast_min_segment_length', 14.0)),
        fast_min_union_aspect=float(config_value(
            args, rows, 'fast_min_union_aspect', 1.35)),
        fast_max_segments_per_color=int(config_value(
            args, rows, 'fast_max_segments_per_color', 2)),
    )


def resolve_processing_scale(args, config, config_path: Path) -> float:
    if args.processing_scale is not None:
        value = float(args.processing_scale)
    else:
        rows = read_three_segment_config(config_path)
        value = float(rows.get(
            'processing_scale',
            getattr(config, 'processing_scale', 1.0)))
    return min(1.0, max(value, 0.1))


def main():
    args = parse_args()
    input_dir = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()

    if not input_dir.is_dir():
        raise SystemExit(f'输入目录不存在：{input_dir}')
    if not config_path.is_file():
        raise SystemExit(f'配置文件不存在：{config_path}')
    if not 1 <= args.jpeg_quality <= 100:
        raise SystemExit('--jpeg-quality must be in 1..100')

    classifier, config = load_detector(config_path)
    processing_scale = resolve_processing_scale(args, config, config_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = image_paths(input_dir)
    if not paths:
        raise SystemExit(f'输入目录没有图片：{input_dir}')

    combo_args = three_segment_args(args, config_path)
    timings = []
    detected = 0
    ambiguous = 0
    rows = []

    print(f'profile: usb_rgb_1')
    print(f'input: {input_dir}')
    print(f'output: {output_dir}')
    print(f'config: {config_path}')
    print(f'processing_scale: {processing_scale:g}')

    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            print(f'SKIP unreadable: {path.name}')
            continue

        started = time.perf_counter()
        strong, weak, singles, triples = detect_frame(
            image, classifier, config, processing_scale, combo_args)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        timings.append(elapsed_ms)

        winner = triples[0] if triples else None
        if winner is not None:
            detected += 1
            ambiguous += int(winner.ambiguous)
            label = '-'.join(winner.symbols)
            status = 'AMBIG' if winner.ambiguous else 'OK'
            score = winner.score
        else:
            label = 'NONE'
            status = 'NONE'
            score = 0.0

        output = annotate_three_segments(image, singles, triples)
        out_path = output_dir / path.name
        ok = cv2.imwrite(
            str(out_path), output,
            [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
        if not ok:
            raise RuntimeError(f'保存失败：{out_path}')

        rows.append(
            f'{path.name},{status},{label},{score:.6f},'
            f'{len(singles)},{len(triples)},{elapsed_ms:.3f}')
        print(
            f'{path.name}: {status} {label} score={score:.3f} '
            f'singles={len(singles)} '
            f'(strong={len(strong)} weak={len(weak)}) '
            f'triples={len(triples)} '
            f'time={elapsed_ms:.1f}ms')

    summary_path = output_dir / 'summary.csv'
    summary_path.write_text(
        'file,status,symbols,score,single_candidates,three_candidates,time_ms\n'
        + '\n'.join(rows) + '\n',
        encoding='utf-8')

    if timings:
        values = np.array(timings, dtype=np.float32)
        print(
            f'summary: images={len(timings)} detected={detected} '
            f'ambiguous={ambiguous} mean={float(np.mean(values)):.1f}ms '
            f'p95={float(np.percentile(values, 95)):.1f}ms')
        print(f'summary_csv: {summary_path}')


if __name__ == '__main__':
    main()
