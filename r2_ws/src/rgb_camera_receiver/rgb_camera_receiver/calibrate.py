"""根据已标注图像生成 USB 摄像头专用的开放集色度模型。"""

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml

from .classifier import (
    ColorModel,
    _classify_chroma,
    _extract_fast_light_points,
    _hue_distance,
    _line_hypotheses,
    _mean_hue,
    _resize_for_fast_path,
    _scaled_fast_config,
    load_config,
)
from .profiles import CAMERA_PROFILES, detector_config_path


CLASSES = ('RED', 'GREEN', 'BLUE', 'YELLOW', 'PURPLE')
TRAIN_FRACTION = 0.75


def _image_chroma_fallback(image: np.ndarray, config) -> Tuple[float, float, float]:
    """仅用于首轮标定定位失败时，从整幅单色训练图估算色度。"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = (
        (hsv[:, :, 1] >= config.min_saturation) &
        (hsv[:, :, 2] >= config.min_value))
    saturation = hsv[:, :, 1][mask]
    values = hsv[:, :, 2][mask]
    if values.size == 0:
        raise RuntimeError('图像中没有可用于标定的高饱和彩色像素')
    saturation_cutoff = float(np.percentile(saturation, 65))
    value_low = float(np.percentile(values, 20))
    value_high = float(np.percentile(values, 85))
    selected_mask = (
        mask &
        (hsv[:, :, 1] >= saturation_cutoff) &
        (hsv[:, :, 2] >= value_low) &
        (hsv[:, :, 2] <= value_high))
    pixels = image[
        selected_mask].astype(np.float32)
    if len(pixels) == 0:
        pixels = image[mask].astype(np.float32)
    normalized = pixels / np.maximum(
        np.sum(pixels, axis=1, keepdims=True), 1.0)
    chroma = np.median(normalized[:, :2], axis=0)
    hue = _mean_hue(
        cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV)
        [:, 0, 0].astype(np.float32).tolist())
    return float(chroma[0]), float(chroma[1]), float(hue or 0.0)


def _robust_feature(
        chroma: np.ndarray,
        hues: Sequence[Optional[float]]) -> Tuple[float, float, float]:
    center = np.median(chroma, axis=0)
    distances = np.linalg.norm(chroma - center, axis=1)
    keep_count = max(3, int(np.ceil(len(chroma) * 0.70)))
    keep_indexes = np.argsort(distances)[:keep_count]
    center = np.median(chroma[keep_indexes], axis=0)
    kept_hues = [
        hues[index] for index in keep_indexes
        if hues[index] is not None
    ]
    hue = _mean_hue(kept_hues)
    return float(center[0]), float(center[1]), float(hue or 0.0)


def _extract_feature(path: Path, config) -> Tuple[float, float, float]:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f'无法读取训练图片：{path}')
    fallback = _image_chroma_fallback(image, config)
    work, scale = _resize_for_fast_path(image, config)
    fast_config = _scaled_fast_config(config, scale)
    points = _extract_fast_light_points(work, fast_config)
    line_features = []
    for indexes in _line_hypotheses(points, fast_config):
        chroma = np.asarray(
            [points[index].chroma for index in indexes], dtype=np.float32)
        if len(chroma) < fast_config.min_dots:
            continue
        feature = _robust_feature(
            chroma, [points[index].hue for index in indexes])
        center = np.asarray(feature[:2], dtype=np.float32)
        spread = float(np.median(np.linalg.norm(chroma - center, axis=1)))
        quality = float(np.mean([points[index].quality for index in indexes]))
        score = len(indexes) * quality / (1.0 + spread * 18.0)
        line_features.append((score, feature))
    if line_features:
        return max(line_features, key=lambda item: item[0])[1]
    if len(points) < 3:
        return fallback
    chroma = np.asarray([item.chroma for item in points], dtype=np.float32)
    return _robust_feature(chroma, [item.hue for item in points])


def _fit_model(
        name: str,
        features: Sequence[Tuple[float, float, float]]) -> ColorModel:
    feature_values = np.asarray(features, dtype=np.float64)
    values = feature_values[:, :2]
    if len(values) < 6:
        raise RuntimeError(f'{name} 至少需要 6 张训练图片')
    center = np.median(values, axis=0)
    centered = values - center
    covariance = centered.T @ centered / max(len(values) - 1, 1)
    diagonal = np.diag(np.diag(covariance))
    covariance = 0.75 * covariance + 0.25 * diagonal
    variance_floor = max(float(np.trace(covariance)) * 0.02, 2.5e-5)
    covariance += np.eye(2) * variance_floor
    inverse = np.linalg.inv(covariance)
    distances = np.sqrt(np.maximum(
        np.einsum('ni,ij,nj->n', centered, inverse, centered), 0.0))
    max_distance = max(3.2, float(np.max(distances)) * 1.20)
    hue_center = _mean_hue(feature_values[:, 2].astype(float).tolist())
    hue_distances = [
        _hue_distance(float(value), float(hue_center or 0.0))
        for value in feature_values[:, 2]
    ]
    hue_tolerance = max(8.0, float(np.max(hue_distances)) * 1.25)
    return ColorModel(
        name=name,
        chroma_center=(float(center[0]), float(center[1])),
        chroma_covariance=(
            (float(covariance[0, 0]), float(covariance[0, 1])),
            (float(covariance[1, 0]), float(covariance[1, 1])),
        ),
        max_distance=max_distance,
        hue_center=None if hue_center is None else float(hue_center),
        hue_tolerance=hue_tolerance,
    )


def calibrate(dataset: Path, base_config: Path, output: Path) -> None:
    with base_config.open('r', encoding='utf-8') as stream:
        raw = yaml.safe_load(stream) or {}
    if raw.get('algorithm') != 'classical':
        raise RuntimeError('当前标定工具只支持 classical 检测后端')

    base = load_config(str(base_config))
    features: Dict[str, List[Tuple[float, float, float]]] = {}
    split_counts: Dict[str, int] = {}
    for name in CLASSES:
        paths = sorted((dataset / name).glob('*.jpg'))
        if not paths:
            raise RuntimeError(f'{dataset / name} 中没有 JPG 训练图片')
        rows = [_extract_feature(path, base) for path in paths]
        train_count = max(6, int(np.floor(len(rows) * TRAIN_FRACTION)))
        train_count = min(train_count, len(rows))
        features[name] = rows
        split_counts[name] = train_count

    fitted_models = tuple(
        _fit_model(name, features[name][:split_counts[name]])
        for name in CLASSES)
    models = []
    for model in fitted_models:
        values = np.asarray(features[model.name], dtype=np.float64)
        chroma = values[:, :2]
        delta = chroma - np.asarray(model.chroma_center)
        distances = np.sqrt(np.maximum(
            np.einsum(
                'ni,ij,nj->n',
                delta, model.inverse_covariance, delta),
            0.0))
        hue_distances = [
            _hue_distance(float(value), float(model.hue_center or 0.0))
            for value in values[:, 2]
        ]
        feature_centers = tuple(
            tuple(float(item) for item in row)
            for row in values.tolist())
        models.append(replace(
            model,
            max_distance=max(
                model.max_distance,
                float(np.max(distances)) * 1.10),
            hue_tolerance=max(
                model.hue_tolerance,
                float(np.max(hue_distances)) * 1.10),
            feature_centers=feature_centers,
            feature_max_distance=2.6))
    models = tuple(models)
    calibrated = replace(base, colors=models)

    # 确认每个训练色度都能被自己的模型唯一接收。几何和 NONE 验收由评估工具完成。
    for expected in CLASSES:
        for feature in features[expected]:
            best_model, best_distance, margin = _classify_chroma(
                feature[:2], calibrated, feature[2])
            if (
                    best_model is None or
                    best_model.name != expected or
                    best_distance > best_model.max_distance or
                    margin < calibrated.min_class_margin):
                raise RuntimeError(
                    f'{expected} 色度模型未通过唯一分类检查：'
                    f'best={None if best_model is None else best_model.name} '
                    f'distance={best_distance:.3f} '
                    f'margin={margin:.3f}')

    raw['colors'] = {
        model.name: {
            'chroma_center': [
                round(value, 8) for value in model.chroma_center],
            'chroma_covariance': [
                [round(value, 10) for value in row]
                for row in model.chroma_covariance
            ],
            'max_distance': round(model.max_distance, 6),
            'hue_center': (
                None if model.hue_center is None
                else round(model.hue_center, 4)),
            'hue_tolerance': round(model.hue_tolerance, 4),
            'feature_centers': [
                [round(value, 6) for value in row]
                for row in model.feature_centers
            ],
            'feature_max_distance': round(
                model.feature_max_distance or 2.6, 4),
        }
        for model in models
    }
    raw['calibrated'] = True
    raw['algorithm'] = 'classical'
    raw['calibration'] = {
        'train_fraction': TRAIN_FRACTION,
        'train_counts': split_counts,
        'validation_counts': {
            name: len(features[name]) - split_counts[name]
            for name in CLASSES
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', encoding='utf-8') as stream:
        yaml.safe_dump(raw, stream, sort_keys=False, allow_unicode=True)
    print(f'已写入 USB 摄像头色度模型：{output}')


def main() -> None:
    parser = argparse.ArgumentParser(description='标定 R2 摄像头色度模型')
    parser.add_argument(
        '--camera-profile', choices=CAMERA_PROFILES, required=True)
    parser.add_argument('--dataset', type=Path)
    parser.add_argument('--base-config', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    dataset = args.dataset or (
        Path('~/Desktop/LED/camera_data').expanduser() /
        args.camera_profile)
    base_config = args.base_config or detector_config_path(
        args.camera_profile)
    calibrate(
        dataset.resolve(), base_config.resolve(), args.output.resolve())


if __name__ == '__main__':
    main()
