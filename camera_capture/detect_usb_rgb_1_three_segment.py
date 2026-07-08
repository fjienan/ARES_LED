#!/usr/bin/env python3
"""USB RGB 1 三段四色灯带离线识别：不依赖 ROS2。"""

import argparse
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import sys
import time

import cv2
import numpy as np
import yaml

from detect_usb_rgb_2_three_segment import (
    REPO_ROOT,
    R2_PACKAGE,
    image_paths,
)

SHARED_PACKAGE = REPO_ROOT / 'shared' / 'src' / 'rgb_comm_protocol'
sys.path.insert(0, str(SHARED_PACKAGE))
sys.path.insert(0, str(R2_PACKAGE))

from rgb_comm_protocol import FixedColorProtocol  # noqa: E402
from rgb_camera_receiver.classifier import classifier_for_profile  # noqa: E402
from rgb_camera_receiver.protocol_decoder import (  # noqa: E402
    annotate_protocol,
    decode_protocol_candidates,
    load_pairing_config,
    protocol_color_symbols,
    scaled_candidate_crop_area,
    select_protocol_winner,
)
from rgb_camera_receiver.three_segment import annotate_three_segments  # noqa: E402


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


def detect_frame(image, classifier, config, processing_scale: float, args):
    work = scaled_frame(image, processing_scale)
    candidates = classifier.detect_protocol_candidates(
        work,
        config,
        protocol_color_symbols(args.protocol),
        args.pairing_config.min_coarse_colors,
        scaled_candidate_crop_area(args.pairing_config, processing_scale))
    if processing_scale < 0.999:
        inverse = 1.0 / processing_scale
        candidates = [item.scaled(inverse) for item in candidates]
    protocol_candidates = decode_protocol_candidates(
        candidates, args.protocol, args.pairing_config)
    protocol_winner = select_protocol_winner(
        protocol_candidates, args.winner_margin)
    return candidates, protocol_candidates, protocol_winner


def three_segment_args(args, config_path: Path):
    pairing_config = load_pairing_config(str(config_path))
    overrides = {}
    if args.min_three_score is not None:
        overrides['min_command_score'] = float(args.min_three_score)
    if args.max_angle_degrees is not None:
        overrides['max_angle_degrees'] = float(args.max_angle_degrees)
    if args.max_cross_distance is not None:
        overrides['max_cross_distance_pixels'] = float(args.max_cross_distance)
    if args.min_center_distance_ratio is not None:
        overrides['min_center_distance_ratio'] = float(
            args.min_center_distance_ratio)
    if args.max_center_distance_ratio is not None:
        overrides['max_center_distance_ratio'] = float(
            args.max_center_distance_ratio)
    if args.max_gap_ratio is not None:
        overrides['max_gap_ratio'] = float(args.max_gap_ratio)
    if args.max_results is not None:
        overrides['max_candidates'] = int(args.max_results)
    if overrides:
        pairing_config = replace(pairing_config, **overrides)
    return SimpleNamespace(
        protocol=FixedColorProtocol(),
        pairing_config=pairing_config,
        winner_margin=float(
            args.winner_margin
            if args.winner_margin is not None
            else 1.2),
        min_three_score=pairing_config.min_command_score,
    )


def resolve_processing_scale(args, config, config_path: Path) -> float:
    if args.processing_scale is not None:
        value = float(args.processing_scale)
    else:
        value = float(getattr(config, 'processing_scale', 1.0))
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
        candidates, protocol_candidates, protocol_winner = detect_frame(
            image, classifier, config, processing_scale, combo_args)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        timings.append(elapsed_ms)

        if protocol_winner is not None:
            detected += 1
            label = '-'.join(protocol_winner.symbols)
            status = f'OK:{protocol_winner.command_id}'
            score = protocol_winner.score
        else:
            label = 'NONE'
            status = 'NONE'
            score = 0.0

        single_winner = classifier.select_winner(candidates, config)
        output = classifier.annotate(image, candidates, single_winner)
        output = annotate_protocol(
            output,
            protocol_candidates,
            protocol_winner,
            'OFFLINE',
            1 if protocol_winner is not None else 0,
            1,
        )
        out_path = output_dir / path.name
        ok = cv2.imwrite(
            str(out_path), output,
            [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
        if not ok:
            raise RuntimeError(f'保存失败：{out_path}')

        rows.append(
            f'{path.name},{status},{label},{score:.6f},'
            f'{len(candidates)},{len(protocol_candidates)},{elapsed_ms:.3f}')
        print(
            f'{path.name}: {status} {label} score={score:.3f} '
            f'strip_candidates={len(candidates)} '
            f'protocol_candidates={len(protocol_candidates)} '
            f'time={elapsed_ms:.1f}ms')

    summary_path = output_dir / 'summary.csv'
    summary_path.write_text(
        'file,status,symbols,score,strip_candidates,protocol_candidates,time_ms\n'
        + '\n'.join(rows) + '\n',
        encoding='utf-8')

    if timings:
        values = np.array(timings, dtype=np.float32)
        print(
            f'summary: images={len(timings)} detected={detected} '
            f'mean={float(np.mean(values)):.1f}ms '
            f'p95={float(np.percentile(values, 95)):.1f}ms')
        print(f'summary_csv: {summary_path}')


if __name__ == '__main__':
    main()
