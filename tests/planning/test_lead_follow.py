from __future__ import annotations

from adas_planning.memory.track_history import TrackHistory
from adas_planning.planners.lead_follow import compute_lead_follow
from adas_planning.types import DetectionInput, PlanningInput


def test_lead_follow_selects_closest_vehicle():
    planning_input = PlanningInput(
        frame_id=0,
        timestamp_s=0.0,
        image_width=1280,
        image_height=720,
        detections=[
            DetectionInput(
                kind="vehicle",
                label="car",
                confidence=0.9,
                box={"x1": 600, "y1": 500, "x2": 700, "y2": 600, "width": 100, "height": 100},
                track_id=1,
                distance_m=20.0,
            ),
            DetectionInput(
                kind="vehicle",
                label="car",
                confidence=0.85,
                box={"x1": 620, "y1": 520, "x2": 710, "y2": 610, "width": 90, "height": 90},
                track_id=2,
                distance_m=8.0,
            ),
        ],
    )
    config = {"lead_follow": {"warning_distance_m": 12.0, "critical_distance_m": 6.0}}
    proposal = compute_lead_follow(planning_input, config, TrackHistory(config))
    assert proposal is not None
    assert proposal.lead_object_id == 2
    assert proposal.warnings


def test_lead_follow_handles_id_switch_without_crash():
    history = TrackHistory({"lead_follow": {"track_history_frames": 5}})
    for frame_id, track_id, distance in [(0, 1, 15.0), (1, 99, 14.0), (2, 99, 13.5)]:
        planning_input = PlanningInput(
            frame_id=frame_id,
            timestamp_s=float(frame_id) / 10.0,
            image_width=1280,
            image_height=720,
            detections=[
                DetectionInput(
                    kind="vehicle",
                    label="car",
                    confidence=0.9,
                    box={"x1": 640, "y1": 500, "x2": 740, "y2": 600, "width": 100, "height": 100},
                    track_id=track_id,
                    distance_m=distance,
                )
            ],
        )
        proposal = compute_lead_follow(planning_input, {"lead_follow": {}}, history)
        assert proposal is not None
