import json
from typing import Sequence, Tuple

Color = Tuple[int, int, int]


def build_wled_state_json(
        colors: Sequence[Color],
        pixel_count: int) -> str:
    if len(colors) != 2:
        raise ValueError('WLED serial transport requires exactly two colors')
    if pixel_count < 2:
        raise ValueError('pixel_count must be at least 2')
    segments = [
        {
            'id': 0,
            'start': 0,
            'stop': 1,
            'col': [[int(channel) for channel in colors[0]]],
            'fx': 0,
        },
        {
            'id': 1,
            'start': 1,
            'stop': 2,
            'col': [[int(channel) for channel in colors[1]]],
            'fx': 0,
        },
    ]
    if pixel_count > 2:
        segments.append({
            'id': 2,
            'start': 2,
            'stop': int(pixel_count),
            'col': [[0, 0, 0]],
            'fx': 0,
        })
    return json.dumps(
        {'on': True, 'seg': segments},
        separators=(',', ':'))


def build_wled_brightness_json(brightness: int) -> str:
    """生成 WLED 独立亮度命令，例如 {"bri":20}。"""
    if not 0 <= brightness <= 255:
        raise ValueError('brightness must be in range 0..255')
    return json.dumps({'bri': brightness}, separators=(',', ':'))
