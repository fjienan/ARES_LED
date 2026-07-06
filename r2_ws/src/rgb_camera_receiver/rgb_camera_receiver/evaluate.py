import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Dict, List

import cv2

from .classifier import (
    annotate,
    detect_proposals,
    detection_metrics,
    load_config,
    select_winner,
)
from .profiles import (
    CAMERA_PROFILES,
    detector_config_path,
    require_calibrated_detector,
)


CLASSES = ('RED', 'GREEN', 'BLUE', 'YELLOW', 'PURPLE', 'NONE')
TRAIN_FRACTION = 0.75
REQUIRED_SCORE_SEPARATION = 3.0


def _different_color_margin(candidates) -> float:
    if not candidates:
        return math.inf
    different = next(
        (item for item in candidates[1:]
         if item.color != candidates[0].color),
        None)
    if different is None:
        return math.inf
    return candidates[0].score / max(different.score, 1e-9)


def evaluate(dataset: Path, output: Path, config_path: Path) -> int:
    config = load_config(str(config_path))
    records: List[Dict] = []
    durations = []
    for expected in CLASSES:
        image_paths = sorted((dataset / expected).glob('*.jpg'))
        train_count = int(math.floor(len(image_paths) * TRAIN_FRACTION))
        for image_index, image_path in enumerate(image_paths):
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f'cannot read {image_path}')
            started = time.perf_counter()
            proposals = detect_proposals(image, config)
            durations.append(time.perf_counter() - started)
            candidates = [
                item for item in proposals if item.score >= config.min_score
            ]
            winner = select_winner(candidates, config)
            records.append({
                'path': image_path,
                'expected': expected,
                'proposals': proposals,
                'candidates': candidates,
                'winner': winner,
                'margin': _different_color_margin(candidates),
                'split': (
                    'train' if image_index < train_count else 'validation'),
            })

    true_scores = [
        row['winner'].score for row in records
        if row['expected'] != 'NONE' and row['winner'] is not None and
        row['winner'].color == row['expected']
    ]
    false_scores = [
        item.score
        for row in records
        for item in row['proposals']
        if (
            row['expected'] == 'NONE' or
            item.color != row['expected']
        )
    ]
    min_true_score = min(true_scores) if true_scores else 0.0
    max_false_score = max(false_scores) if false_scores else 0.0
    score_separation = (
        min_true_score / max_false_score
        if max_false_score > 0.0 else math.inf)
    separation_passed = score_separation >= REQUIRED_SCORE_SEPARATION

    validation_rows = [
        row for row in records if row['split'] == 'validation']
    validation_true_scores = [
        row['winner'].score for row in validation_rows
        if row['expected'] != 'NONE' and row['winner'] is not None and
        row['winner'].color == row['expected']
    ]
    validation_false_scores = [
        item.score
        for row in validation_rows
        for item in row['proposals']
        if row['expected'] == 'NONE' or item.color != row['expected']
    ]
    validation_min_true = (
        min(validation_true_scores) if validation_true_scores else 0.0)
    validation_max_false = (
        max(validation_false_scores) if validation_false_scores else 0.0)
    validation_separation = (
        validation_min_true / validation_max_false
        if validation_max_false > 0.0 else math.inf)
    validation_separation_passed = (
        validation_separation >= REQUIRED_SCORE_SEPARATION)

    output.mkdir(parents=True, exist_ok=True)
    rows = []
    failures = 0
    for record in records:
        expected = record['expected']
        candidates = record['candidates']
        winner = record['winner']
        actual = winner.color if winner is not None else 'NONE'
        wrong_accepted = [
            item for item in candidates
            if expected != 'NONE' and item.color != expected
        ]
        passed = (
            (expected == 'NONE' and not candidates) or
            (expected != 'NONE' and actual == expected and
             not wrong_accepted)
        )
        failures += int(not passed)
        target_dir = output / expected
        target_dir.mkdir(parents=True, exist_ok=True)
        image = cv2.imread(str(record['path']), cv2.IMREAD_COLOR)
        rendered = annotate(image, record['proposals'], winner)
        cv2.imwrite(str(target_dir / record['path'].name), rendered)
        proposal_data = []
        for index, item in enumerate(record['proposals'], 1):
            metrics = {
                name: round(value, 6)
                for name, value in detection_metrics(item).items()
            }
            proposal_data.append({
                'rank': index,
                'accepted': item.score >= config.min_score,
                'color': item.color,
                'confidence': round(item.confidence, 6),
                'score': round(item.score, 6),
                'dots': item.dot_count,
                'length': round(item.length, 3),
                'residual': round(item.residual, 3),
                'spacing_cv': round(item.spacing_cv, 6),
                'metrics': metrics,
            })
        rows.append({
            'image': str(record['path'].relative_to(dataset)),
            'expected': expected,
            'actual': actual,
            'passed': passed,
            'candidate_count': len(candidates),
            'proposal_count': len(record['proposals']),
            'winner_margin': record['margin'],
            'split': record['split'],
            'proposals': proposal_data,
        })

    sorted_durations = sorted(durations)
    median_ms = (
        sorted_durations[len(sorted_durations) // 2] * 1000.0
        if sorted_durations else 0.0)
    p95_index = max(0, math.ceil(len(sorted_durations) * 0.95) - 1)
    p95_ms = (
        sorted_durations[p95_index] * 1000.0
        if sorted_durations else 0.0)
    summary = {
        'dataset': str(dataset),
        'images': len(rows),
        'passed': len(rows) - failures,
        'failed': failures,
        'min_true_score': min_true_score,
        'max_false_score': max_false_score,
        'score_separation': score_separation,
        'required_score_separation': REQUIRED_SCORE_SEPARATION,
        'separation_passed': separation_passed,
        'validation_min_true_score': validation_min_true,
        'validation_max_false_score': validation_max_false,
        'validation_score_separation': validation_separation,
        'validation_separation_passed': validation_separation_passed,
        'median_processing_ms': median_ms,
        'p95_processing_ms': p95_ms,
        'results': rows,
    }
    with (output / 'results.json').open('w', encoding='utf-8') as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
    with (output / 'results.csv').open(
            'w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            'image', 'expected', 'actual', 'passed', 'candidate_count',
            'proposal_count', 'winner_margin', 'split', 'proposals'))
        writer.writeheader()
        for row in rows:
            flat = dict(row)
            flat['proposals'] = json.dumps(
                row['proposals'], ensure_ascii=False)
            writer.writerow(flat)
    print(
        f'evaluated={len(rows)} passed={len(rows)-failures} failed={failures} '
        f'min_true={min_true_score:.6f} max_false={max_false_score:.6f} '
        f'separation={score_separation:.3f} separation_ok={separation_passed} '
        f'validation_separation={validation_separation:.3f} '
        f'validation_ok={validation_separation_passed} '
        f'median_ms={median_ms:.1f} p95_ms={p95_ms:.1f} output={output}')
    return (
        0 if failures == 0 and separation_passed and
        validation_separation_passed else 1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Evaluate and annotate the R2 LED dataset')
    parser.add_argument(
        '--camera-profile', choices=CAMERA_PROFILES, default='usb_rgb')
    parser.add_argument(
        '--dataset', type=Path)
    parser.add_argument(
        '--output', type=Path)
    parser.add_argument(
        '--config', type=Path)
    args = parser.parse_args()
    dataset = args.dataset or (
        Path('~/Desktop/LED/camera_data').expanduser() /
        args.camera_profile)
    output = args.output or (
        Path('~/Desktop/LED/camera_results').expanduser() /
        args.camera_profile)
    config = args.config or detector_config_path(args.camera_profile)
    metadata = require_calibrated_detector(
        config.resolve(), args.camera_profile)
    if metadata.get('algorithm') != 'classical':
        raise RuntimeError(
            'evaluate_led_dataset currently supports the classical backend only')
    raise SystemExit(evaluate(
        dataset.resolve(),
        output.resolve(),
        config.resolve()))


if __name__ == '__main__':
    main()
