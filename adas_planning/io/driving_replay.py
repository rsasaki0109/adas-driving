from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from adas_planning.io.perception_adapter import adapt_perception_document
from adas_planning.io.planning_json import load_planning_document, planning_result_from_dict, planning_result_to_dict
from adas_planning.io.schema import (
    DRIVING_REPLAY_SCHEMA_VERSION,
    PLANNING_INPUT_SCHEMA_VERSION,
    PLANNING_RESULT_SCHEMA_VERSION,
)


def build_driving_replay_document(
    perception_payload: dict[str, Any],
    planning_frames: list[dict[str, Any]],
    *,
    source: str | None = None,
    config_path: str | None = None,
    config_hash: str | None = None,
    producer: str = "adas_planning",
) -> dict[str, Any]:
    planning_inputs = adapt_perception_document(perception_payload)
    input_by_frame = {item.frame_id: item for item in planning_inputs}
    frames: list[dict[str, Any]] = []

    for planning_frame in planning_frames:
        frame_id = int(planning_frame.get("frame_id", 0))
        planning_input = input_by_frame.get(frame_id)
        perception_frame = _planning_input_to_dict(planning_input) if planning_input else {}
        frames.append(
            {
                "frame_id": frame_id,
                "timestamp_s": float(planning_frame.get("timestamp_s", 0.0)),
                "perception": perception_frame,
                "planning": planning_frame,
            }
        )

    return {
        "schema_version": DRIVING_REPLAY_SCHEMA_VERSION,
        "source": source,
        "config": config_path,
        "config_hash": config_hash,
        "producer": producer,
        "perception_schema_version": str(perception_payload.get("schema_version", "0.1")),
        "planning_schema_version": PLANNING_RESULT_SCHEMA_VERSION,
        "frames": frames,
    }


def write_driving_replay_document(
    path: str | Path,
    *,
    perception_path: str | Path,
    planning_path: str | Path,
    producer: str = "adas_planning",
) -> dict[str, Any]:
    with Path(perception_path).open("r", encoding="utf-8") as handle:
        perception_payload = json.load(handle)
    planning_frames = load_planning_document(planning_path)
    planning_meta_path = Path(planning_path)
    with planning_meta_path.open("r", encoding="utf-8") as handle:
        planning_payload = json.load(handle)

    document = build_driving_replay_document(
        perception_payload,
        planning_frames,
        source=str(perception_path),
        config_path=planning_payload.get("config"),
        config_hash=planning_payload.get("config_hash"),
        producer=producer,
    )
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(document, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return document


def load_driving_replay_document(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    validate_driving_replay_document(payload)
    return payload


def validate_driving_replay_document(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != DRIVING_REPLAY_SCHEMA_VERSION:
        raise ValueError(f"unsupported driving replay schema: {payload.get('schema_version')}")
    frames = payload.get("frames")
    if not isinstance(frames, list):
        raise ValueError("driving replay document must include frames list")


def planning_results_from_driving_replay(payload: dict[str, Any]) -> list[Any]:
    from adas_planning.types import PlanningResult

    results: list[PlanningResult] = []
    for frame in payload.get("frames") or []:
        planning = frame.get("planning") or {}
        results.append(planning_result_from_dict(planning))
    return results


def _planning_input_to_dict(planning_input) -> dict[str, Any]:
    return {
        "schema_version": PLANNING_INPUT_SCHEMA_VERSION,
        "frame_id": planning_input.frame_id,
        "timestamp_s": planning_input.timestamp_s,
        "image_width": planning_input.image_width,
        "image_height": planning_input.image_height,
        "lanes": [
            {
                "side": lane.side,
                "points_px": [list(point) for point in lane.points_px],
                "confidence": lane.confidence,
            }
            for lane in planning_input.lanes
        ],
        "polygon_px": [list(point) for point in planning_input.polygon_px],
        "detections": [
            {
                "kind": detection.kind,
                "label": detection.label,
                "confidence": detection.confidence,
                "box": dict(detection.box),
                "track_id": detection.track_id,
                "distance_m": detection.distance_m,
                "ground_position_m": list(detection.ground_position_m)
                if detection.ground_position_m is not None
                else None,
                "state": detection.state,
            }
            for detection in planning_input.detections
        ],
        "ego_speed_mps": planning_input.ego_speed_mps,
        "coordinate_frame": planning_input.coordinate_frame,
    }


def planning_result_dicts_from_document(planning_frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [planning_result_to_dict(planning_result_from_dict(frame)) for frame in planning_frames]
