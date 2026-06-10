from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from adas_planning.io.schema import PLANNING_RESULT_SCHEMA_VERSION
from adas_planning.types import Behavior, PathPoint, PlanningResult, Warning


def planning_result_to_dict(result: PlanningResult) -> dict[str, Any]:
    return {
        "schema_version": result.schema_version or PLANNING_RESULT_SCHEMA_VERSION,
        "frame_id": result.frame_id,
        "timestamp_s": round(float(result.timestamp_s), 6),
        "coordinate_frame": "image",
        "units": {"distance": "m", "speed": "m/s"},
        "target_path": [_path_point_to_dict(point) for point in result.target_path],
        "target_speed_mps": round(float(result.target_speed_mps), 3) if result.target_speed_mps is not None else None,
        "behavior": result.behavior.value if isinstance(result.behavior, Behavior) else str(result.behavior),
        "warnings": [_warning_to_dict(warning) for warning in result.warnings],
        "confidence": round(float(result.confidence), 4),
        "lead_object_id": result.lead_object_id,
        "stop_reason": result.stop_reason,
        "target_path_px": [[int(x), int(y)] for x, y in result.target_path_px],
        "debug": result.debug,
    }


def write_planning_document(
    path: str | Path,
    *,
    frames: list[PlanningResult],
    source: str | None = None,
    config_path: str | None = None,
    config_hash: str | None = None,
    producer: str = "adas_planning",
) -> None:
    payload = {
        "schema_version": PLANNING_RESULT_SCHEMA_VERSION,
        "source": source,
        "config": config_path,
        "config_hash": config_hash,
        "producer": producer,
        "frames": [planning_result_to_dict(frame) for frame in frames],
    }
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def load_planning_document(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if "frames" in payload:
        return list(payload.get("frames") or [])
    return [payload]


def planning_result_from_dict(data: dict[str, Any]) -> PlanningResult:
    target_path = [
        PathPoint(
            x_px=int(point["x_px"]),
            y_px=int(point["y_px"]),
            x_m=point.get("x_m"),
            z_m=point.get("z_m"),
        )
        for point in data.get("target_path") or []
    ]
    warnings = [
        Warning(
            code=str(item.get("code", "")),
            message=str(item.get("message", "")),
            severity=str(item.get("severity", "info")),
            value=item.get("value"),
        )
        for item in data.get("warnings") or []
    ]
    behavior_raw = str(data.get("behavior", Behavior.UNKNOWN.value))
    try:
        behavior = Behavior(behavior_raw)
    except ValueError:
        behavior = Behavior.UNKNOWN
    return PlanningResult(
        schema_version=str(data.get("schema_version", PLANNING_RESULT_SCHEMA_VERSION)),
        frame_id=int(data.get("frame_id", 0)),
        timestamp_s=float(data.get("timestamp_s", 0.0)),
        target_path=target_path,
        target_speed_mps=data.get("target_speed_mps"),
        behavior=behavior,
        warnings=warnings,
        confidence=float(data.get("confidence", 0.0) or 0.0),
        lead_object_id=data.get("lead_object_id"),
        stop_reason=data.get("stop_reason"),
        target_path_px=[(int(x), int(y)) for x, y in data.get("target_path_px") or []],
        debug=dict(data.get("debug") or {}),
    )


def _path_point_to_dict(point: PathPoint) -> dict[str, Any]:
    payload = {"x_px": int(point.x_px), "y_px": int(point.y_px)}
    if point.x_m is not None:
        payload["x_m"] = round(float(point.x_m), 3)
    if point.z_m is not None:
        payload["z_m"] = round(float(point.z_m), 3)
    return payload


def _warning_to_dict(warning: Warning) -> dict[str, Any]:
    payload = {
        "code": warning.code,
        "message": warning.message,
        "severity": warning.severity,
    }
    if warning.value is not None:
        payload["value"] = round(float(warning.value), 3)
    return payload
