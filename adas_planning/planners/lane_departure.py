from __future__ import annotations

from typing import Any

from adas_planning.planners.lane_target import _interpolate_x, _lane_by_side
from adas_planning.types import Behavior, PlannerProposal, PlanningInput, Warning


class LaneDeparturePlanner:
    def __init__(self) -> None:
        self._active = False

    def compute(self, planning_input: PlanningInput, config: dict[str, Any]) -> PlannerProposal | None:
        dep_cfg = config.get("lane_departure", {})
        enter_offset_ratio = float(dep_cfg.get("enter_offset_ratio", 0.18))
        exit_offset_ratio = float(dep_cfg.get("exit_offset_ratio", 0.12))
        ego_x_ratio = float(dep_cfg.get("ego_x_ratio", 0.5))

        left = _lane_by_side(planning_input.lanes, "left")
        right = _lane_by_side(planning_input.lanes, "right")
        if left is None or right is None:
            return None

        ego_y = int(planning_input.image_height * float(dep_cfg.get("ego_y_ratio", 0.92)))
        left_x = _interpolate_x(left.points_px, ego_y)
        right_x = _interpolate_x(right.points_px, ego_y)
        if left_x is None or right_x is None:
            return None

        lane_center_x = (left_x + right_x) / 2.0
        lane_width = max(1.0, abs(right_x - left_x))
        ego_x = planning_input.image_width * ego_x_ratio
        lateral_offset_ratio = abs(ego_x - lane_center_x) / lane_width

        threshold = exit_offset_ratio if self._active else enter_offset_ratio
        triggered = lateral_offset_ratio > threshold
        self._active = triggered
        if not triggered:
            return None

        return PlannerProposal(
            behavior=Behavior.LANE_DEPARTURE,
            warnings=[
                Warning(
                    code="LANE_DEPARTURE",
                    message=f"Lateral offset {lateral_offset_ratio:.2f} lane widths",
                    severity="warning",
                    value=lateral_offset_ratio,
                )
            ],
            confidence=min(left.confidence, right.confidence),
            priority=15,
            debug={"lateral_offset_ratio": lateral_offset_ratio},
        )
