from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from adas_perception.types import Box, Detection, LaneResult, PerceptionResult, Point


SCHEMA_VERSION = "0.1"


def image_result_payload(
    *,
    source: str,
    output: str,
    result: PerceptionResult,
    image_shape: tuple[int, int, int],
    config_path: str,
) -> dict[str, Any]:
    height, width = image_shape[:2]
    return {
        "schema_version": SCHEMA_VERSION,
        "source": source,
        "output": output,
        "config": config_path,
        "image": {
            "width": width,
            "height": height,
        },
        "result": perception_result_to_dict(result),
    }


def video_result_payload(
    *,
    source: str,
    output: str,
    frames: list[dict[str, Any]],
    width: int,
    height: int,
    fps: float,
    processed_frames: int,
    config_path: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "source": source,
        "output": output,
        "config": config_path,
        "video": {
            "width": width,
            "height": height,
            "fps": fps,
            "processed_frames": processed_frames,
        },
        "frames": frames,
    }


def frame_result_to_dict(
    *,
    frame_index: int,
    timestamp_ms: float,
    result: PerceptionResult,
) -> dict[str, Any]:
    data = perception_result_to_dict(result)
    data["frame_index"] = frame_index
    data["timestamp_ms"] = round(float(timestamp_ms), 3)
    return data


def perception_result_to_dict(result: PerceptionResult) -> dict[str, Any]:
    return {
        "summary": result.count_by_kind(),
        "lanes": lane_result_to_dict(result.lanes),
        "detections": [detection_to_dict(detection) for detection in result.detections],
    }


def lane_result_to_dict(lanes: LaneResult) -> dict[str, Any]:
    return {
        "lines": [
            {
                "side": line.side,
                "points": [_point_to_list(point) for point in line.points],
                "confidence": round(float(line.confidence), 4),
            }
            for line in lanes.lines
        ],
        "polygon": [_point_to_list(point) for point in lanes.polygon],
        "raw_segments": [
            [_point_to_list(start), _point_to_list(end)] for start, end in lanes.raw_segments
        ],
    }


def detection_to_dict(detection: Detection) -> dict[str, Any]:
    payload = {
        "kind": detection.kind,
        "label": detection.label,
        "confidence": round(float(detection.confidence), 4),
        "box": box_to_dict(detection.box),
        "source": detection.source,
        "track_id": detection.track_id,
        "distance_m": round(float(detection.distance_m), 3) if detection.distance_m is not None else None,
    }
    if detection.ground_position_m is not None:
        x_m, z_m = detection.ground_position_m
        payload["ground_position_m"] = {"x": round(float(x_m), 3), "z": round(float(z_m), 3)}
    if detection.state is not None:
        payload["state"] = detection.state
    return payload


def box_to_dict(box: Box) -> dict[str, int]:
    return {
        "x1": box.x1,
        "y1": box.y1,
        "x2": box.x2,
        "y2": box.y2,
        "width": box.width,
        "height": box.height,
    }


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _point_to_list(point: Point) -> list[int]:
    return [int(point[0]), int(point[1])]
