from __future__ import annotations

from dataclasses import replace

from adas_planning.planners.traffic_light import TrafficLightFSM, compute_traffic_light
from adas_planning.types import Behavior, DetectionInput, PlanningInput


def _light(state: str, confidence: float = 0.9) -> DetectionInput:
    return DetectionInput(
        kind="traffic_light",
        label="traffic light",
        confidence=confidence,
        box={"x1": 600, "y1": 80, "x2": 640, "y2": 140, "width": 40, "height": 60},
        state=state,
    )


def test_traffic_light_fsm_debounces_red():
    fsm = TrafficLightFSM(red_enter_frames=2, green_exit_frames=3, unknown_hold_frames=2)
    planning_input = PlanningInput(frame_id=0, timestamp_s=0.0, image_width=1280, image_height=720)
    config = {"traffic_light": {}}

    assert compute_traffic_light(
        PlanningInput(
            frame_id=0,
            timestamp_s=0.0,
            image_width=1280,
            image_height=720,
            detections=[_light("red")],
        ),
        config,
        fsm,
    ) is None
    proposal = compute_traffic_light(
        PlanningInput(
            frame_id=1,
            timestamp_s=0.1,
            image_width=1280,
            image_height=720,
            detections=[_light("red")],
        ),
        config,
        fsm,
    )
    assert proposal is not None
    assert proposal.behavior == Behavior.STOP_FOR_RED


def test_traffic_light_flicker_hold():
    fsm = TrafficLightFSM(red_enter_frames=2, green_exit_frames=3, unknown_hold_frames=3)
    config = {"traffic_light": {}}
    red_input = PlanningInput(
        frame_id=0,
        timestamp_s=0.0,
        image_width=1280,
        image_height=720,
        detections=[_light("red")],
    )
    compute_traffic_light(red_input, config, fsm)
    compute_traffic_light(replace(red_input, frame_id=1, timestamp_s=0.1), config, fsm)
    held = compute_traffic_light(
        PlanningInput(frame_id=2, timestamp_s=0.2, image_width=1280, image_height=720, detections=[]),
        config,
        fsm,
    )
    assert held is not None
    assert held.behavior == Behavior.STOP_FOR_RED
