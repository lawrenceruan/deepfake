"""Frozen-Wan visual deepfake detection, with an optional SAM RGB map branch."""

from .config import DetectorConfig
from .model import WanOnlyDetector, WanRGBDetector, DetectorHeads, build_detector
from .losses import detector_loss

__all__ = ['DetectorConfig', 'WanOnlyDetector', 'WanRGBDetector', 'DetectorHeads', 'build_detector', 'detector_loss']
