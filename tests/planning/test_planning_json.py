from __future__ import annotations

from adas_planning.io.planning_json import planning_result_from_dict, planning_result_to_dict
from adas_planning.types import Behavior, PlanningResult


def test_planning_result_round_trip_preserves_behavior():
    original = PlanningResult(
        frame_id=7,
        timestamp_s=1.25,
        behavior=Behavior.STOP_FOR_RED,
        confidence=0.66,
        stop_reason="red_traffic_light",
    )
    payload = planning_result_to_dict(original)
    restored = planning_result_from_dict(payload)
    assert restored.behavior == Behavior.STOP_FOR_RED
    assert restored.stop_reason == "red_traffic_light"
    assert restored.confidence == 0.66
