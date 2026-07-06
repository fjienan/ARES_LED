"""Generate an R2-only hue model from labelled single-colour captures."""

import argparse
from dataclasses import replace
import math
from pathlib import Path

import cv2
import numpy as np
import yaml

from .classifier import ColorModel, detect_candidates, load_config, select_winner


CLASSES = ('RED', 'GREEN', 'CYAN', 'BLUE', 'PURPLE')


def hue_distance(values, center):
    delta = np.abs(values.astype(np.float32) - center)
    return np.minimum(delta, 180.0 - delta)


def calibrate(dataset: Path, base_config: Path, output: Path) -> None:
    with base_config.open('r', encoding='utf-8') as stream:
        config = yaml.safe_load(stream)
    calibrated = {}
    for name in CLASSES:
        hue_rows = []
        weight_rows = []
        for path in sorted((dataset / name).glob('*.jpg')):
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f'cannot read {path}')
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            mask = (hsv[:, :, 1] >= 80) & (hsv[:, :, 2] >= 55)
            hue_rows.append(hsv[:, :, 0][mask])
            weight_rows.append(
                hsv[:, :, 1][mask].astype(np.float32) *
                hsv[:, :, 2][mask].astype(np.float32))
        hues = np.concatenate(hue_rows)
        weights = np.concatenate(weight_rows)
        histogram = np.bincount(hues, weights=weights, minlength=180)
        peak = int(np.argmax(histogram))
        local = hue_distance(hues, peak) <= 22.0
        angles = hues[local].astype(np.float32) * (2.0 * np.pi / 180.0)
        local_weights = weights[local]
        angle = np.arctan2(
            np.sum(np.sin(angles) * local_weights),
            np.sum(np.cos(angles) * local_weights))
        center = float((angle * 180.0 / (2.0 * np.pi)) % 180.0)
        previous = config['colors'][name]
        # Labelled frames still contain unrelated background colours. Never
        # broaden an already validated class boundary from these pixels.
        radius = float(min(
            float(previous['hue_radius']),
            np.clip(np.percentile(hue_distance(hues[local], center), 98), 7.0, 18.0)))
        calibrated[name] = {
            'hue_center': round(center, 2),
            'hue_radius': round(radius, 2),
            'min_saturation': previous['min_saturation'],
            'min_value': previous['min_value'],
        }
    base = load_config(str(base_config))
    proposed = replace(base, colors=tuple(
        ColorModel(name=name, **{
            key: float(value) for key, value in calibrated[name].items()
            if key != 'name'
        }) for name in CLASSES))
    valid = True
    for expected in (*CLASSES, 'NONE'):
        for path in sorted((dataset / expected).glob('*.jpg')):
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            candidates = detect_candidates(image, proposed)
            winner = select_winner(candidates, proposed)
            if expected == 'NONE':
                valid = not candidates
            else:
                margin = math.inf
                if len(candidates) > 1:
                    margin = candidates[0].score / max(candidates[1].score, 1e-9)
                valid = (
                    winner is not None and winner.color == expected and
                    all(item.color == expected for item in candidates) and
                    margin >= 1.10
                )
            if not valid:
                print(
                    f'proposed model rejected by {path}; retaining validated base colors')
                break
        if not valid:
            break
    if not valid:
        calibrated = {
            model.name: {
                'hue_center': model.hue_center,
                'hue_radius': model.hue_radius,
                'min_saturation': model.min_saturation,
                'min_value': model.min_value,
            }
            for model in base.colors
        }
    config['colors'] = calibrated
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', encoding='utf-8') as stream:
        yaml.safe_dump(config, stream, sort_keys=False, allow_unicode=True)
    print(f'wrote camera color model: {output}')


def main() -> None:
    parser = argparse.ArgumentParser(description='Calibrate R2 camera colour models')
    package = Path(__file__).resolve().parents[1]
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--base-config', type=Path, default=package / 'config' / 'detector.yaml')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    calibrate(args.dataset.resolve(), args.base_config.resolve(), args.output.resolve())


if __name__ == '__main__':
    main()
