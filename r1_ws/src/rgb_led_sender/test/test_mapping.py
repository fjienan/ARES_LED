import pytest

from rgb_led_sender.mapping import (
    WledSegmentSpec,
    build_triplet_segment_specs,
    build_wled_idle_effect_json,
    build_wled_state_json,
)


def test_build_wled_state_json_lights_six_segments():
    specs = build_triplet_segment_specs(
        code_length=3,
        low_segments=[0, 1, 2],
        low_brightness=6.0,
        low_reverse_order=False,
        high_segments=[3, 4, 5],
        high_brightness=60.0,
        high_reverse_order=False,
        segment_starts=[0, 1, 2, 3, 4, 5],
        segment_stops=[1, 2, 3, 4, 5, 6],
    )
    payload = build_wled_state_json(
        [(0, 0, 255), (255, 0, 255), (255, 0, 0)],
        specs,
        pixel_count=6,
    )

    assert payload == (
        '{"on":true,"bri":255,"seg":['
        '{"id":0,"start":0,"stop":1,"col":[[0,0,255]],"fx":0,"bri":6},'
        '{"id":1,"start":1,"stop":2,"col":[[255,0,255]],"fx":0,"bri":6},'
        '{"id":2,"start":2,"stop":3,"col":[[255,0,0]],"fx":0,"bri":6},'
        '{"id":3,"start":3,"stop":4,"col":[[0,0,255]],"fx":0,"bri":60},'
        '{"id":4,"start":4,"stop":5,"col":[[255,0,255]],"fx":0,"bri":60},'
        '{"id":5,"start":5,"stop":6,"col":[[255,0,0]],"fx":0,"bri":60}'
        ']}'
    )


def test_build_wled_idle_effect_json_uses_six_segment_orange_yellow_effect():
    payload = build_wled_idle_effect_json(
        pixel_count=6,
        color=[220, 0, 120],
        brightness=20.0,
        effect=28,
        speed=160,
        intensity=120,
        palette=0,
    )

    assert payload == (
        '{"on":true,"bri":20,"seg":['
        '{"id":0,"start":0,"stop":6,'
        '"col":[[220,0,120],[0,0,0]],'
        '"fx":28,"sx":160,"ix":120,"pal":0}'
        ']}'
    )


def test_build_triplet_segment_specs_can_reverse_each_group_independently():
    specs = build_triplet_segment_specs(
        code_length=3,
        low_segments=[0, 1, 2],
        low_brightness=6.0,
        low_reverse_order=True,
        high_segments=[3, 4, 5],
        high_brightness=60.0,
        high_reverse_order=False,
        segment_starts=[0, 1, 2, 3, 4, 5],
        segment_stops=[1, 2, 3, 4, 5, 6],
    )
    assert [item.source_index for item in specs] == [2, 1, 0, 0, 1, 2]


def test_rgb_scale_mode_scales_segment_colors():
    payload = build_wled_state_json(
        [(100, 50, 0), (0, 100, 50), (50, 0, 100)],
        [WledSegmentSpec(0, 0, 1, 0, 51)],
        pixel_count=1,
        brightness_mode='rgb_scale',
    )
    assert payload == (
        '{"on":true,"bri":255,"seg":['
        '{"id":0,"start":0,"stop":1,"col":[[20,10,0]],"fx":0}'
        ']}'
    )


def test_build_wled_state_json_rejects_wrong_shape():
    with pytest.raises(ValueError):
        build_wled_state_json(
            [(255, 0, 0)],
            [WledSegmentSpec(0, 0, 1, 0, 6)],
            pixel_count=11)
    with pytest.raises(ValueError):
        build_wled_state_json(
            [(255, 0, 0), (0, 0, 255), (0, 255, 0)],
            [WledSegmentSpec(0, 0, 2, 0, 6)],
            pixel_count=1,
        )
