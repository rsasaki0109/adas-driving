from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


PointPx = tuple[int, int]
PointGround = tuple[float, float]  # (lateral x_m, forward z_m)


class Behavior(str, Enum):
    KEEP_LANE = "KEEP_LANE"
    FOLLOW_LEAD = "FOLLOW_LEAD"
    STOP_FOR_RED = "STOP_FOR_RED"
    GO_CAUTION = "GO_CAUTION"
    YIELD_VRU = "YIELD_VRU"
    LANE_DEPARTURE = "LANE_DEPARTURE"
    CAUTION = "CAUTION"
    UNKNOWN = "UNKNOWN"


class WarningCode(str, Enum):
    FOLLOW_DISTANCE = "FOLLOW_DISTANCE"
    STOP_RECOMMENDATION = "STOP_RECOMMENDATION"
    YIELD_VRU = "YIELD_VRU"
    LANE_DEPARTURE = "LANE_DEPARTURE"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"


@dataclass(frozen=True)
class PathPoint:
    x_px: int
    y_px: int
    x_m: float | None = None
    z_m: float | None = None


@dataclass(frozen=True)
class LaneInput:
    side: str
    points_px: tuple[PointPx, ...]
    confidence: float = 0.0


@dataclass(frozen=True)
class DetectionInput:
    kind: str
    label: str
    confidence: float
    box: dict[str, int]
    track_id: int | None = None
    distance_m: float | None = None
    ground_position_m: PointGround | None = None
    state: str | None = None


@dataclass(frozen=True)
class PlanningInput:
    frame_id: int
    timestamp_s: float
    image_width: int
    image_height: int
    lanes: list[LaneInput] = field(default_factory=list)
    polygon_px: list[PointPx] = field(default_factory=list)
    detections: list[DetectionInput] = field(default_factory=list)
    ego_speed_mps: float | None = None
    coordinate_frame: str = "image"
    schema_version: str = "perception.v0.1"


@dataclass(frozen=True)
class Warning:
    code: str
    message: str
    severity: str = "info"
    value: float | None = None


@dataclass
class PlannerProposal:
    behavior: Behavior
    target_path: list[PathPoint] = field(default_factory=list)
    target_speed_mps: float | None = None
    warnings: list[Warning] = field(default_factory=list)
    confidence: float = 0.0
    lead_object_id: int | None = None
    stop_reason: str | None = None
    target_path_px: list[PointPx] = field(default_factory=list)
    debug: dict[str, Any] = field(default_factory=dict)
    priority: int = 0


@dataclass
class PlanningResult:
    schema_version: str = "planning.v0.1"
    frame_id: int = 0
    timestamp_s: float = 0.0
    target_path: list[PathPoint] = field(default_factory=list)
    target_speed_mps: float | None = None
    behavior: Behavior = Behavior.UNKNOWN
    warnings: list[Warning] = field(default_factory=list)
    confidence: float = 0.0
    lead_object_id: int | None = None
    stop_reason: str | None = None
    target_path_px: list[PointPx] = field(default_factory=list)
    debug: dict[str, Any] = field(default_factory=dict)
