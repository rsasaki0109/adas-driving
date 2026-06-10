from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from adas_planning.types import Behavior, DetectionInput, PlannerProposal, PlanningInput, Warning


@dataclass
class TrafficLightFSM:
    red_enter_frames: int = 2
    green_exit_frames: int = 3
    unknown_hold_frames: int = 2
    min_confidence: float = 0.25

    active_state: str = "unknown"
    debounced_state: str = "unknown"
    candidate_state: str = "unknown"
    candidate_count: int = 0
    unknown_hold_remaining: int = 0

    def observe(self, raw_state: str | None) -> str:
        observed = raw_state or "unknown"
        if observed == self.candidate_state:
            self.candidate_count += 1
        else:
            self.candidate_state = observed
            self.candidate_count = 1

        if observed == "red" and self.candidate_count >= self.red_enter_frames:
            self.debounced_state = "red"
            self.active_state = "red"
            self.unknown_hold_remaining = 0
        elif observed == "green" and self.candidate_count >= self.green_exit_frames:
            self.debounced_state = "green"
            self.active_state = "green"
            self.unknown_hold_remaining = 0
        elif observed == "unknown":
            if self.debounced_state != "unknown" and self.unknown_hold_remaining <= 0:
                self.unknown_hold_remaining = self.unknown_hold_frames
            elif self.unknown_hold_remaining > 0:
                self.unknown_hold_remaining -= 1
            if self.unknown_hold_remaining <= 0:
                self.debounced_state = "unknown"
                self.active_state = "unknown"
        elif observed == "yellow":
            self.debounced_state = observed
            self.active_state = observed
            self.unknown_hold_remaining = self.unknown_hold_frames

        return self.debounced_state


def _select_traffic_light(planning_input: PlanningInput) -> DetectionInput | None:
    lights = [
        det
        for det in planning_input.detections
        if det.kind == "traffic_light" and det.confidence > 0.0
    ]
    if not lights:
        return None
    width = max(1, planning_input.image_width)
    height = max(1, planning_input.image_height)

    def score(det: DetectionInput) -> tuple[float, float, float]:
        center_x = (det.box["x1"] + det.box["x2"]) / 2.0 / width
        center_y = (det.box["y1"] + det.box["y2"]) / 2.0 / height
        upper_center = 1.0 - abs(center_x - 0.5) + (1.0 - center_y)
        return (det.confidence, upper_center, -center_y)

    lights.sort(key=score, reverse=True)
    return lights[0]


def compute_traffic_light(
    planning_input: PlanningInput,
    config: dict[str, Any],
    fsm: TrafficLightFSM,
) -> PlannerProposal | None:
    tl_cfg = config.get("traffic_light", {})
    selected = _select_traffic_light(planning_input)
    if selected is None:
        state = fsm.observe("unknown")
        if state == "unknown":
            return None
    else:
        if selected.confidence < float(tl_cfg.get("min_confidence", fsm.min_confidence)):
            state = fsm.observe("unknown")
        else:
            state = fsm.observe(selected.state or "unknown")

    if state == "red":
        return PlannerProposal(
            behavior=Behavior.STOP_FOR_RED,
            warnings=[
                Warning(
                    code="STOP_RECOMMENDATION",
                    message="Stop recommendation for red traffic light",
                    severity="warning",
                )
            ],
            confidence=min(1.0, selected.confidence if selected else 0.5),
            stop_reason="red_traffic_light",
            priority=100,
            debug={"traffic_light_state": state, "debounced": True},
        )

    if state == "green":
        return PlannerProposal(
            behavior=Behavior.GO_CAUTION,
            confidence=min(1.0, selected.confidence if selected else 0.4),
            priority=5,
            debug={"traffic_light_state": state, "debounced": True},
        )

    if state == "yellow":
        return PlannerProposal(
            behavior=Behavior.GO_CAUTION,
            warnings=[
                Warning(
                    code="STOP_RECOMMENDATION",
                    message="Caution for yellow traffic light",
                    severity="info",
                )
            ],
            confidence=min(1.0, selected.confidence if selected else 0.35),
            priority=40,
            debug={"traffic_light_state": state, "debounced": True},
        )
    return None
