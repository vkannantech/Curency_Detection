"""Runtime configuration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
DATASET_CONFIG = ROOT_DIR / "dataset" / "data.yaml"


@dataclass(slots=True)
class CameraConfig:
    weights: str
    source: str = "0"
    imgsz: int = 640
    conf: float = 0.5
    iou: float = 0.45
    device: str = "cpu"
    speak: bool = False
    headless: bool = False
    max_fps: float = 12.0
    cooldown: float = 2.5
    resolution_width: int = 1280
    resolution_height: int = 720
    profile: str = "desktop"
    preprocess: bool = True
    use_tta: bool = False
    smoothing_window: int = 12
    stable_ratio: float = 0.5
    min_confidence_sum: float = 1.2
