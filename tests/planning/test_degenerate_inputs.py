from __future__ import annotations

from adas_planning.io.perception_adapter import adapt_perception_document
from adas_planning.pipeline import PlanningPipeline


def test_degenerate_inputs_produce_valid_results():
    payload = {
        "video": {"width": 640, "height": 360, "fps": 10.0},
        "frames": [
            {"frame_index": 0, "timestamp_ms": 0, "lanes": {"lines": []}, "detections": []},
            {
                "frame_index": 1,
                "timestamp_ms": 100,
                "lanes": {
                    "lines": [
                        {"side": "left", "points": [[100, 300], [120, 150]], "confidence": 0.05},
                        {"side": "right", "points": [[500, 300], [480, 150]], "confidence": 0.05},
                    ]
                },
                "detections": [
                    {
                        "kind": "traffic_light",
                        "label": "traffic light",
                        "confidence": 0.1,
                        "box": {"x1": 300, "y1": 50, "x2": 320, "y2": 90, "width": 20, "height": 40},
                        "state": "red",
                    }
                ],
            },
        ],
    }
    pipeline = PlanningPipeline({"memory": {}, "lane_target": {}, "traffic_light": {}})
    for planning_input in adapt_perception_document(payload):
        result = pipeline.plan(planning_input)
        assert result.frame_id == planning_input.frame_id
        assert 0.0 <= result.confidence <= 1.0
