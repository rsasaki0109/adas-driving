from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from adas_perception.types import LaneLine, LaneResult, Point


@dataclass
class _LaneState:
    line: LaneLine
    missed: int = 0


class LaneSmoother:
    """Small temporal smoother for video lane overlays."""

    def __init__(self, config: dict[str, Any]):
        self.enabled = bool(config.get("enabled", False))
        self.alpha = float(config.get("alpha", 0.65))
        self.max_missed = int(config.get("max_missed", 3))
        self._state: dict[str, _LaneState] = {}

    def reset(self) -> None:
        self._state.clear()

    def smooth(self, lanes: LaneResult) -> LaneResult:
        if not self.enabled:
            return lanes

        current_by_side = {line.side: line for line in lanes.lines}
        output_lines: list[LaneLine] = []

        for side, line in current_by_side.items():
            previous = self._state.get(side)
            if previous is None:
                smoothed = line
            else:
                smoothed = _blend_lines(previous.line, line, self.alpha)
            self._state[side] = _LaneState(line=smoothed, missed=0)
            output_lines.append(smoothed)

        for side, state in list(self._state.items()):
            if side in current_by_side:
                continue
            state.missed += 1
            if state.missed > self.max_missed:
                del self._state[side]
                continue
            faded = LaneLine(
                side=state.line.side,
                points=state.line.points,
                confidence=max(0.0, state.line.confidence * (1.0 - state.missed / (self.max_missed + 1))),
                polyline=list(state.line.polyline),
            )
            state.line = faded
            output_lines.append(faded)

        output_lines.sort(key=lambda line: line.side)
        return LaneResult(
            lines=output_lines,
            raw_segments=lanes.raw_segments,
            polygon=_lane_polygon(output_lines),
        )


def _blend_lines(previous: LaneLine, current: LaneLine, alpha: float) -> LaneLine:
    start = _blend_points(previous.points[0], current.points[0], alpha)
    end = _blend_points(previous.points[1], current.points[1], alpha)
    confidence = alpha * previous.confidence + (1.0 - alpha) * current.confidence
    polyline = _blend_polylines(previous.polyline, current.polyline, alpha)
    return LaneLine(side=current.side, points=(start, end), confidence=confidence, polyline=polyline)


def _blend_polylines(previous: list[Point], current: list[Point], alpha: float) -> list[Point]:
    """Blend two polylines element-wise. Falls back to whichever side has data
    when only one is populated, and skips smoothing if lengths differ (a length
    change usually means the fitter switched between linear and polynomial)."""
    if not current:
        return list(previous)
    if not previous:
        return list(current)
    if len(previous) != len(current):
        return list(current)
    return [_blend_points(prev, cur, alpha) for prev, cur in zip(previous, current)]


def _blend_points(previous: Point, current: Point, alpha: float) -> Point:
    x = int(round(alpha * previous[0] + (1.0 - alpha) * current[0]))
    y = int(round(alpha * previous[1] + (1.0 - alpha) * current[1]))
    return (x, y)


def _lane_polygon(lines: list[LaneLine]) -> list[Point]:
    by_side = {line.side: line for line in lines}
    left = by_side.get("left")
    right = by_side.get("right")
    if not left or not right:
        return []
    left_pts = list(left.polyline) if left.polyline else list(left.points)
    right_pts = list(right.polyline) if right.polyline else list(right.points)
    if not left_pts or not right_pts:
        return []
    left_pts.sort(key=lambda p: -p[1])  # bottom -> top
    right_pts.sort(key=lambda p: -p[1])
    return left_pts + list(reversed(right_pts))

