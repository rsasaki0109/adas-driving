from __future__ import annotations

import numpy as np

from adas_perception.traffic_light_state import TrafficLightStateClassifier
from adas_perception.types import Box, Detection


def _frame_with_color(bgr: tuple[int, int, int]) -> np.ndarray:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    frame[40:200, 120:200] = bgr
    return frame


def test_traffic_light_classifier_detects_red():
    frame = _frame_with_color((0, 0, 220))
    classifier = TrafficLightStateClassifier({"enabled": True, "min_box_pixels": 16})
    detection = Detection(
        kind="traffic_light",
        label="traffic light",
        confidence=0.9,
        box=Box(x1=120, y1=40, x2=200, y2=200),
        source="test",
    )
    results = classifier.classify(frame, [detection])
    assert results[0].state == "red"


def test_onnx_method_falls_back_to_hsv_when_model_missing():
    frame = _frame_with_color((0, 0, 220))
    classifier = TrafficLightStateClassifier(
        {
            "enabled": True,
            "min_box_pixels": 16,
            "method": "onnx",
            "onnx_model": "outputs/models/does_not_exist.onnx",
        }
    )
    detection = Detection(
        kind="traffic_light",
        label="traffic light",
        confidence=0.9,
        box=Box(x1=120, y1=40, x2=200, y2=200),
        source="test",
    )
    results = classifier.classify(frame, [detection])
    assert results[0].state == "red"  # HSV fallback still classifies
    # Second call must not retry the failed load path differently.
    assert classifier.classify(frame, [detection])[0].state == "red"


def test_traffic_light_classifier_skips_non_traffic_light():
    frame = _frame_with_color((0, 220, 0))
    classifier = TrafficLightStateClassifier({"enabled": True})
    detection = Detection(
        kind="vehicle",
        label="car",
        confidence=0.9,
        box=Box(x1=120, y1=40, x2=200, y2=200),
        source="test",
    )
    results = classifier.classify(frame, [detection])
    assert results[0].state is None
