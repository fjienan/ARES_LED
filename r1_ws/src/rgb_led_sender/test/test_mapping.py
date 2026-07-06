import pytest

from rgb_led_sender.mapping import order_group_colors
from rgb_led_sender.mapping import build_wled_state_json


def test_nonascending_groups_keep_their_colors():
    groups, colors = order_group_colors(
        [2, 0, 3, 1],
        [(255, 0, 0), (0, 255, 0), (255, 255, 0), (0, 0, 255)],
    )

    assert groups == [0, 1, 2, 3]
    assert colors == [
        0, 255, 0,
        0, 0, 255,
        255, 0, 0,
        255, 255, 0,
    ]


def test_group_and_color_lengths_must_match():
    with pytest.raises(ValueError):
        order_group_colors([0, 1, 2, 3], [(255, 0, 0)])


def test_build_wled_state_json_lights_two_segments_and_clears_rest():
    payload = build_wled_state_json(
        [(255, 0, 0), (0, 0, 255)],
        brightness=40.0,
        pixel_count=11,
    )

    assert payload == (
        '{"on":true,"bri":40,"seg":['
        '{"id":0,"start":0,"stop":1,"col":[[255,0,0]],"fx":0},'
        '{"id":1,"start":1,"stop":2,"col":[[0,0,255]],"fx":0},'
        '{"id":2,"start":2,"stop":11,"col":[[0,0,0]],"fx":0}'
        ']}'
    )


def test_build_wled_state_json_rejects_wrong_shape():
    with pytest.raises(ValueError):
        build_wled_state_json([(255, 0, 0)], brightness=40.0, pixel_count=11)
    with pytest.raises(ValueError):
        build_wled_state_json(
            [(255, 0, 0), (0, 0, 255)],
            brightness=40.0,
            pixel_count=1,
        )
