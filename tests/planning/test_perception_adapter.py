from __future__ import annotations

import json

from adas_planning.io.perception_adapter import adapt_perception_document, adapt_perception_frame
from adas_planning.io.planning_json import planning_result_from_dict, planning_result_to_dict
from adas_planning.types import Behavior


def test_adapts_schema_less_image_json():
    payload = {
        "image": {"width": 1280, "height": 720},
        "result": {
            "lanes": {
                "lines": [
                    {"side": "left", "points": [[500, 700], [560, 400]], "confidence": 0.8},
                    {"side": "right", "points": [[780, 700], [720, 400]], "confidence": 0.75},
                ],
                "polygon": [[500, 700], [780, 700], [720, 400], [560, 400]],
            },
            "detections": [
                {
                    "kind": "vehicle",
                    "label": "car",
                    "confidence": 0.9,
                    "box": {"x1": 600, "y1": 500, "x2": 700, "y2": 600, "width": 100, "height": 100},
                    "track_id": 3,
                    "distance_m": 10.5,
                }
            ],
        },
    }
    inputs = adapt_perception_document(payload)
    assert len(inputs) == 1
    assert inputs[0].image_width == 1280
    assert inputs[0].schema_version == "perception.v0.1"
    assert len(inputs[0].lanes) == 2
    assert inputs[0].detections[0].distance_m == 10.5


def test_adapts_video_json_frames():
    payload = {
        "schema_version": "0.1",
        "video": {"width": 640, "height": 360, "fps": 10.0},
        "frames": [
            {
                "frame_index": 0,
                "timestamp_ms": 0.0,
                "lanes": {"lines": [], "polygon": []},
                "detections": [],
            }
        ],
    }
    inputs = adapt_perception_document(payload)
    assert len(inputs) == 1
    assert inputs[0].frame_id == 0
    assert inputs[0].timestamp_s == 0.0


def test_empty_lane_and_detections_do_not_crash():
    payload = {"image": {"width": 640, "height": 360}, "result": {"lanes": {"lines": []}, "detections": []}}
    inputs = adapt_perception_document(payload)
    assert inputs[0].lanes == []
    assert inputs[0].detections == []


def test_planning_json_round_trip():
    from adas_planning.types import PlanningResult, Warning

    result = PlanningResult(
        frame_id=1,
        timestamp_s=0.5,
        behavior=Behavior.KEEP_LANE,
        confidence=0.8,
        warnings=[Warning(code="LOW_CONFIDENCE", message="test")],
        target_path_px=[(640, 700), (640, 500)],
    )
    restored = planning_result_from_dict(planning_result_to_dict(result))
    assert restored.frame_id == 1
    assert restored.behavior == Behavior.KEEP_LANE
    assert restored.target_path_px == [(640, 700), (640, 500)]


def test_adapt_perception_frame_from_video_frame_dict():
    frame = {
        "frame_index": 4,
        "timestamp_ms": 133.3,
        "lanes": {"lines": [], "polygon": []},
        "detections": [],
    }
    planning_input = adapt_perception_frame(frame, image_width=1280, image_height=720, fps=30.0)
    assert planning_input.frame_id == 4
    assert planning_input.timestamp_s == 0.1333
    assert planning_input.image_width == 1280
