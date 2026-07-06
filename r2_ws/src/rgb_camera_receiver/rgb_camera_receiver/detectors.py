"""按相机配置选择检测算法的稳定入口。"""

from pathlib import Path

from .classifier import annotate, detect_candidates, load_config, select_winner


class ClassicalDetectorBackend:
    """当前 USB RGB 数据集使用的传统图像检测算法。"""

    def __init__(self, config_path: Path):
        self.config = load_config(str(config_path))

    def detect(self, frame):
        return detect_candidates(frame, self.config)

    def select(self, candidates):
        return select_winner(candidates, self.config)

    def render(self, frame, candidates, winner):
        return annotate(frame, candidates, winner)


def create_detector_backend(name: str, config_path: Path):
    """创建相机指定的检测算法；未知算法必须显式实现后才能启用。"""
    if name == 'classical':
        return ClassicalDetectorBackend(config_path)
    raise RuntimeError(f'unsupported detector backend: {name}')
