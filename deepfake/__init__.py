"""Wan-only visual deepfake detection (no RGB or audio branch)."""

from .config import DetectorConfig
from .model import WanOnlyDetector, DetectorHeads
from .losses import detector_loss

__all__ = ['DetectorConfig', 'WanOnlyDetector', 'DetectorHeads', 'detector_loss']
