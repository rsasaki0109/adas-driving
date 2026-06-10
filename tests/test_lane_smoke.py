from __future__ import annotations

import numpy as np
import pytest

from adas_perception.detectors.lane import LaneDetector


def test_lane_detector_smoke_on_synthetic_road():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[350:480, 180:260] = (255, 255, 255)
    frame[350:480, 380:460] = (255, 255, 255)

    detector = LaneDetector({"enabled": True})
    result = detector.detect(frame)
    assert isinstance(result.lines, list)


@pytest.mark.slow
def test_lane_segmentation_detector_smoke_if_model_present():
    from pathlib import Path

    import onnxruntime  # noqa: F401

    from adas_perception.detectors.lane_segmentation import LaneSegmentationDetector

    model_path = Path("outputs/models/twinlitenet_lane.onnx")
    if not model_path.is_file():
        pytest.skip("TwinLiteNet ONNX not downloaded")

    detector = LaneSegmentationDetector(
        {
            "segmentation": {
                "model_path": str(model_path),
                "input_size": [360, 640],
                "output_name": "ll",
                "lane_channel": 1,
                "normalize": {"mean": [0, 0, 0], "std": [1, 1, 1], "scale": 255.0},
            }
        }
    )
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    result = detector.detect(frame)
    assert isinstance(result.lines, list)
