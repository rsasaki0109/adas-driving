from __future__ import annotations

from adas_planning.planners.lane_target import compute_lane_target
from adas_planning.types import Behavior, LaneInput, PlanningInput


def _sample_input() -> PlanningInput:
    return PlanningInput(
        frame_id=0,
        timestamp_s=0.0,
        image_width=1280,
        image_height=720,
        lanes=[
            LaneInput(side="left", points_px=((500, 700), (560, 400)), confidence=0.8),
            LaneInput(side="right", points_px=((780, 700), (720, 400)), confidence=0.75),
        ],
    )


def test_lane_target_builds_centerline():
    proposal = compute_lane_target(_sample_input(), {})
    assert proposal.behavior == Behavior.KEEP_LANE
    assert len(proposal.target_path_px) >= 8
    assert all(620 <= x <= 660 for x, _y in proposal.target_path_px)


def test_missing_lane_returns_caution():
    planning_input = PlanningInput(frame_id=0, timestamp_s=0.0, image_width=640, image_height=360, lanes=[])
    proposal = compute_lane_target(planning_input, {})
    assert proposal.behavior == Behavior.CAUTION
