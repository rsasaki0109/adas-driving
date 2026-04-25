from typing import Any

from .lane import LaneDetector
from .lane_segmentation import LaneSegmentationDetector
from .objects import TorchVisionObjectDetector, UltralyticsObjectDetector, create_object_detector
from .signs import ColorSignDetector
from .traffic_lights import ColorTrafficLightDetector


def create_lane_detector(config: dict[str, Any]):
    """Pick the lane detector backend.

    `lane.backend: cv` (default) returns the OpenCV/Hough/polynomial-fit
    detector. `lane.backend: segmentation` returns the ONNX-based
    `LaneSegmentationDetector` (requires onnxruntime + a model file).
    """
    backend = str(config.get("backend", "cv")).lower()
    if backend in {"cv", "opencv", "hough"}:
        return LaneDetector(config)
    if backend in {"segmentation", "seg", "onnx"}:
        return LaneSegmentationDetector(config)
    raise ValueError(f"Unsupported lane.backend: {backend}. Use 'cv' or 'segmentation'.")


__all__ = [
    "LaneDetector",
    "LaneSegmentationDetector",
    "create_lane_detector",
    "TorchVisionObjectDetector",
    "UltralyticsObjectDetector",
    "create_object_detector",
    "ColorSignDetector",
    "ColorTrafficLightDetector",
]
