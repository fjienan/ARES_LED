"""按摄像头 profile 选择对应的灯带检测算法。

实际检测实现分别放在独立模块中：

* ``classifier_usb_rgb_1``：第一台 USB RGB 摄像头。
* ``classifier_usb_rgb_2``：第二台 USB RGB 摄像头。

本文件保留 ``usb_rgb_1`` 的公开符号导出，兼容已有测试和外部导入；
新代码应通过 ``classifier_for_profile`` 显式选择算法。
"""

from importlib import import_module

from .classifier_usb_rgb_1 import *  # noqa: F401,F403
from .classifier_usb_rgb_1 import _deduplicate_candidates  # noqa: F401


def classifier_for_profile(profile: str):
    """返回指定摄像头 profile 的检测算法模块。"""
    normalized = profile.strip().lower()
    if normalized not in {'usb_rgb_1', 'usb_rgb_2'}:
        raise ValueError(f'unknown camera profile for classifier: {profile!r}')
    return import_module(f'.classifier_{normalized}', __package__)
