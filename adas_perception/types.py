from __future__ import annotations

from dataclasses import dataclass, field


Point = tuple[int, int]


@dataclass(frozen=True)
class Box:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def area(self) -> int:
        return self.width * self.height

    def clamp(self, width: int, height: int) -> "Box":
        return Box(
            x1=max(0, min(width - 1, self.x1)),
            y1=max(0, min(height - 1, self.y1)),
            x2=max(0, min(width - 1, self.x2)),
            y2=max(0, min(height - 1, self.y2)),
        )


@dataclass(frozen=True)
class Detection:
    kind: str
    label: str
    confidence: float
    box: Box
    source: str
    track_id: int | None = None
    distance_m: float | None = None


@dataclass(frozen=True)
class LaneLine:
    side: str
    points: tuple[Point, Point]
    confidence: float


@dataclass(frozen=True)
class LaneResult:
    lines: list[LaneLine] = field(default_factory=list)
    raw_segments: list[tuple[Point, Point]] = field(default_factory=list)
    polygon: list[Point] = field(default_factory=list)


@dataclass(frozen=True)
class PerceptionResult:
    lanes: LaneResult
    detections: list[Detection]

    def count_by_kind(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for detection in self.detections:
            counts[detection.kind] = counts.get(detection.kind, 0) + 1
        if self.lanes.lines:
            counts["lane"] = len(self.lanes.lines)
        return counts
