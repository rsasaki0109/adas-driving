from __future__ import annotations

from adas_perception.tracking import SimpleTracker, _centroid_distance, _diagonal, _iou
from adas_perception.types import Box, Detection


def _vehicle(x1: int, y1: int, x2: int, y2: int, confidence: float = 0.9) -> Detection:
    return Detection(
        kind="vehicle",
        label="car",
        confidence=confidence,
        box=Box(x1=x1, y1=y1, x2=x2, y2=y2),
        source="test",
    )


def test_tracker_assigns_stable_id_with_motion_prediction():
    tracker = SimpleTracker(
        {
            "enabled": True,
            "motion_prediction": True,
            "centroid_distance_fraction": 0.0,
            "two_stage": False,
        }
    )
    first = tracker.update([_vehicle(100, 100, 200, 200)])
    assert first[0].track_id == 1
    shifted = tracker.update([_vehicle(110, 100, 210, 200)])
    assert shifted[0].track_id == 1


def test_tracker_centroid_fallback_recovers_low_iou_match():
    tracker = SimpleTracker(
        {
            "enabled": True,
            "iou_threshold": 0.90,
            "centroid_distance_fraction": 0.75,
            "motion_prediction": False,
        }
    )
    tracker.update([_vehicle(100, 100, 200, 200)])
    recovered = tracker.update([_vehicle(130, 100, 230, 200)])
    assert recovered[0].track_id == 1


def test_iou_and_centroid_helpers():
    box_a = Box(0, 0, 100, 100)
    box_b = Box(50, 50, 150, 150)
    assert 0.0 < _iou(box_a, box_b) < 1.0
    assert _centroid_distance(box_a, box_b) > 0.0
    assert _diagonal(box_a) > 0.0
