from __future__ import annotations

from typing import Any

from adas_planning.types import DetectionInput, LaneInput, PlanningInput, PointPx


VRU_KINDS = {"pedestrian", "cyclist", "bicycle", "motorcycle", "rider"}


def adapt_perception_document(payload: dict[str, Any]) -> list[PlanningInput]:
    if "frames" in payload:
        image_width = int(payload.get("video", {}).get("width", 0))
        image_height = int(payload.get("video", {}).get("height", 0))
        fps = float(payload.get("video", {}).get("fps", 30.0) or 30.0)
        frames = payload.get("frames") or []
        return [
            _adapt_frame(
                frame,
                default_width=image_width,
                default_height=image_height,
                fps=fps,
                schema_version=_perception_schema_version(payload),
            )
            for frame in frames
        ]
    image = payload.get("image") or {}
    return [
        _adapt_frame(
            payload.get("result") or payload,
            default_width=int(image.get("width", 0)),
            default_height=int(image.get("height", 0)),
            fps=30.0,
            frame_id=0,
            timestamp_s=0.0,
            schema_version=_perception_schema_version(payload),
        )
    ]


def _perception_schema_version(payload: dict[str, Any]) -> str:
    version = payload.get("schema_version")
    if version in (None, ""):
        return "perception.v0.1"
    return f"perception.v{version}" if not str(version).startswith("perception.") else str(version)


def _adapt_frame(
    frame: dict[str, Any],
    *,
    default_width: int,
    default_height: int,
    fps: float,
    schema_version: str,
    frame_id: int | None = None,
    timestamp_s: float | None = None,
) -> PlanningInput:
    frame_index = int(frame.get("frame_index", frame_id if frame_id is not None else 0))
    if timestamp_s is None:
        timestamp_ms = frame.get("timestamp_ms")
        if timestamp_ms is not None:
            timestamp_s = float(timestamp_ms) / 1000.0
        else:
            timestamp_s = frame_index / fps if fps > 0 else 0.0

    lanes_payload = frame.get("lanes") or {}
    lanes = [_adapt_lane(line) for line in lanes_payload.get("lines") or []]
    polygon = [_point(point) for point in lanes_payload.get("polygon") or []]
    detections = [_adapt_detection(item) for item in frame.get("detections") or []]

    image_width = int(frame.get("image_width", default_width))
    image_height = int(frame.get("image_height", default_height))
    if image_width <= 0 or image_height <= 0:
        image_width = default_width
        image_height = default_height

    return PlanningInput(
        frame_id=frame_index,
        timestamp_s=float(timestamp_s),
        image_width=image_width,
        image_height=image_height,
        lanes=lanes,
        polygon_px=polygon,
        detections=detections,
        ego_speed_mps=_optional_float(frame.get("ego_speed_mps")),
        coordinate_frame=str(frame.get("coordinate_frame", "image")),
        schema_version=schema_version,
    )


def _adapt_lane(line: dict[str, Any]) -> LaneInput:
    points = tuple(_point(point) for point in line.get("points") or [])
    if len(points) < 2:
        points = ((0, 0), (0, 0))
    return LaneInput(
        side=str(line.get("side", "unknown")),
        points_px=points,
        confidence=float(line.get("confidence", 0.0) or 0.0),
    )


def _adapt_detection(item: dict[str, Any]) -> DetectionInput:
    box = item.get("box") or {}
    ground = item.get("ground_position_m")
    ground_position = None
    if isinstance(ground, dict):
        ground_position = (float(ground.get("x", 0.0)), float(ground.get("z", 0.0)))
    elif isinstance(ground, (list, tuple)) and len(ground) >= 2:
        ground_position = (float(ground[0]), float(ground[1]))

    return DetectionInput(
        kind=str(item.get("kind", "unknown")),
        label=str(item.get("label", item.get("kind", "unknown"))),
        confidence=float(item.get("confidence", 0.0) or 0.0),
        box={
            "x1": int(box.get("x1", 0)),
            "y1": int(box.get("y1", 0)),
            "x2": int(box.get("x2", 0)),
            "y2": int(box.get("y2", 0)),
            "width": int(box.get("width", max(0, int(box.get("x2", 0)) - int(box.get("x1", 0))))),
            "height": int(box.get("height", max(0, int(box.get("y2", 0)) - int(box.get("y1", 0))))),
        },
        track_id=item.get("track_id"),
        distance_m=_optional_float(item.get("distance_m")),
        ground_position_m=ground_position,
        state=item.get("state"),
    )


def _point(raw: Any) -> PointPx:
    if isinstance(raw, dict):
        return int(raw.get("x", 0)), int(raw.get("y", 0))
    return int(raw[0]), int(raw[1])


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
