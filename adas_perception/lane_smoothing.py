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
    return LaneLine(side=current.side, points=(start, end), confidence=confidence)


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
    left_bottom, left_top = left.points
    right_bottom, right_top = right.points
    return [left_bottom, left_top, right_top, right_bottom]

