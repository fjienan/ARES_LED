import json
from dataclasses import dataclass
from typing import Sequence, Tuple

Color = Tuple[int, int, int]


@dataclass(frozen=True)
class WledSegmentSpec:
    """WLED 物理段配置。"""

    segment_id: int
    start: int
    stop: int
    source_index: int
    brightness: int


def _clamp_brightness(value: float) -> int:
    return int(max(0, min(round(value), 255)))


def _clamp_byte(value: float) -> int:
    return int(max(0, min(round(value), 255)))


def build_triplet_segment_specs(
        code_length: int,
        low_segments: Sequence[int],
        low_brightness: float,
        low_reverse_order: bool,
        high_segments: Sequence[int],
        high_brightness: float,
        high_reverse_order: bool,
        segment_starts: Sequence[int],
        segment_stops: Sequence[int]) -> Tuple[WledSegmentSpec, ...]:
    if code_length != 3:
        raise ValueError('R1 WLED sender currently requires a three-symbol protocol')
    rows = []
    for name, segments, brightness, reverse in (
            ('low_segments', low_segments, low_brightness, low_reverse_order),
            ('high_segments', high_segments, high_brightness, high_reverse_order)):
        if len(segments) != code_length:
            raise ValueError(f'{name} must contain exactly {code_length} physical segments')
        source_order = list(range(code_length))
        if reverse:
            source_order.reverse()
        for physical_segment, source_index in zip(segments, source_order):
            physical_segment = int(physical_segment)
            if physical_segment < 0:
                raise ValueError(f'{name} must contain non-negative physical segment indexes')
            if physical_segment >= len(segment_starts) or physical_segment >= len(segment_stops):
                raise ValueError(
                    f'physical segment {physical_segment} has no start/stop range')
            start = int(segment_starts[physical_segment])
            stop = int(segment_stops[physical_segment])
            if start < 0 or stop <= start:
                raise ValueError(
                    f'invalid WLED range for physical segment {physical_segment}: '
                    f'{start}..{stop}')
            rows.append(WledSegmentSpec(
                segment_id=physical_segment,
                start=start,
                stop=stop,
                source_index=int(source_index),
                brightness=_clamp_brightness(brightness),
            ))
    ids = [item.segment_id for item in rows]
    if len(set(ids)) != len(ids):
        raise ValueError('low_segments and high_segments must not overlap')
    rows.sort(key=lambda item: item.segment_id)
    return tuple(rows)


def build_wled_state_json(
        colors: Sequence[Color],
        display_segments: Sequence[WledSegmentSpec],
        pixel_count: int,
        brightness_mode: str = 'segment_bri',
        master_brightness: float = 255.0) -> str:
    if len(colors) != 3:
        raise ValueError('WLED serial transport requires exactly three colors')
    if not display_segments:
        raise ValueError('display_segments must not be empty')
    if pixel_count <= 0:
        raise ValueError('pixel_count must be positive')
    mode = brightness_mode.strip().lower()
    if mode not in {'segment_bri', 'rgb_scale'}:
        raise ValueError("brightness_mode must be 'segment_bri' or 'rgb_scale'")
    brightness_value = _clamp_brightness(master_brightness)
    segments = []
    max_stop = 0
    for spec in sorted(display_segments, key=lambda item: item.segment_id):
        if spec.source_index < 0 or spec.source_index >= len(colors):
            raise ValueError(f'invalid source_index: {spec.source_index}')
        if spec.start < 0 or spec.stop <= spec.start or spec.stop > pixel_count:
            raise ValueError(
                f'invalid WLED range for segment {spec.segment_id}: '
                f'{spec.start}..{spec.stop}')
        rgb = tuple(int(channel) for channel in colors[spec.source_index])
        row = {
            'id': int(spec.segment_id),
            'start': int(spec.start),
            'stop': int(spec.stop),
            'col': [[int(channel) for channel in rgb]],
            'fx': 0,
        }
        if mode == 'segment_bri':
            row['bri'] = int(spec.brightness)
        else:
            scale = int(spec.brightness) / 255.0
            row['col'] = [[int(round(channel * scale)) for channel in rgb]]
        segments.append(row)
        max_stop = max(max_stop, int(spec.stop))
    if pixel_count > max_stop:
        segments.append({
            'id': max(item.segment_id for item in display_segments) + 1,
            'start': int(max_stop),
            'stop': int(pixel_count),
            'col': [[0, 0, 0]],
            'fx': 0,
        })
    return json.dumps(
        {'on': True, 'bri': brightness_value, 'seg': segments},
        separators=(',', ':'))


def build_wled_idle_effect_json(
        pixel_count: int,
        color: Sequence[int],
        brightness: float,
        effect: int,
        speed: int,
        intensity: int,
        palette: int) -> str:
    if pixel_count <= 0:
        raise ValueError('pixel_count must be positive')
    if len(color) != 3:
        raise ValueError('idle effect color must contain exactly three channels')
    rgb = [_clamp_byte(channel) for channel in color]
    segment = {
        'id': 0,
        'start': 0,
        'stop': int(pixel_count),
        'col': [rgb, [0, 0, 0]],
        'fx': int(effect),
        'sx': _clamp_byte(speed),
        'ix': _clamp_byte(intensity),
        'pal': int(palette),
    }
    return json.dumps(
        {'on': True, 'bri': _clamp_brightness(brightness), 'seg': [segment]},
        separators=(',', ':'))
