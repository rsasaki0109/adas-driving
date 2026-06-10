from __future__ import annotations

from typing import Any

from adas_planning.types import Behavior, LaneInput, PathPoint, PlannerProposal, PlanningInput


def _lane_by_side(lanes: list[LaneInput], side: str) -> LaneInput | None:
    for lane in lanes:
        if lane.side == side:
            return lane
    return None


def _interpolate_x(points: tuple[tuple[int, int], ...], y: float) -> float | None:
    ordered = sorted(points, key=lambda point: point[1], reverse=True)
    if len(ordered) < 2:
        return None
    for idx in range(len(ordered) - 1):
        y_top, x_top = ordered[idx][1], ordered[idx][0]
        y_bottom, x_bottom = ordered[idx + 1][1], ordered[idx + 1][0]
        if y_top == y_bottom:
            continue
        if y_top >= y >= y_bottom or y_bottom >= y >= y_top:
            ratio = (y - y_bottom) / (y_top - y_bottom)
            return x_bottom + ratio * (x_top - x_bottom)
    return float(ordered[0][0])


def compute_lane_target(planning_input: PlanningInput, config: dict[str, Any]) -> PlannerProposal:
    lane_cfg = config.get("lane_target", {})
    num_samples = int(lane_cfg.get("num_samples", 16))
    min_confidence = float(lane_cfg.get("min_lane_confidence", 0.15))

    left = _lane_by_side(planning_input.lanes, "left")
    right = _lane_by_side(planning_input.lanes, "right")
    if left is None or right is None:
        return PlannerProposal(
            behavior=Behavior.CAUTION,
            confidence=0.1,
            debug={"reason": "missing_lane_side"},
        )

    lane_confidence = min(left.confidence, right.confidence)
    if lane_confidence < min_confidence:
        return PlannerProposal(
            behavior=Behavior.CAUTION,
            confidence=max(0.05, lane_confidence),
            debug={"reason": "low_lane_confidence"},
        )

    y_start = int(planning_input.image_height * float(lane_cfg.get("path_y_start_ratio", 0.95)))
    y_end = int(planning_input.image_height * float(lane_cfg.get("path_y_end_ratio", 0.55)))
    if y_end >= y_start:
        y_end = max(0, y_start - 1)

    target_path: list[PathPoint] = []
    target_path_px: list[tuple[int, int]] = []
    for idx in range(num_samples):
        if num_samples == 1:
            y = y_start
        else:
            y = y_start + (y_end - y_start) * idx / (num_samples - 1)
        left_x = _interpolate_x(left.points_px, y)
        right_x = _interpolate_x(right.points_px, y)
        if left_x is None or right_x is None:
            continue
        center_x = int(round((left_x + right_x) / 2.0))
        y_px = int(round(y))
        target_path.append(PathPoint(x_px=center_x, y_px=y_px))
        target_path_px.append((center_x, y_px))

    if not target_path:
        return PlannerProposal(
            behavior=Behavior.CAUTION,
            confidence=max(0.05, lane_confidence * 0.5),
            debug={"reason": "empty_target_path"},
        )

    return PlannerProposal(
        behavior=Behavior.KEEP_LANE,
        target_path=target_path,
        target_path_px=target_path_px,
        confidence=lane_confidence,
        priority=10,
        debug={"source": "lane_centerline"},
    )
