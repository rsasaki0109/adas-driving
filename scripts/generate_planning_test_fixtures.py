#!/usr/bin/env python3
"""Build compact planning test fixtures under tests/fixtures/planning/."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _base_video(*, width: int = 640, height: int = 360, fps: float = 10.0) -> dict:
    return {"width": width, "height": height, "fps": fps}


def _lanes(width: int = 640, height: int = 360) -> dict:
    return {
        "lines": [
            {"side": "left", "points": [[180, height - 10], [220, int(height * 0.5)]], "confidence": 0.8},
            {"side": "right", "points": [[460, height - 10], [420, int(height * 0.5)]], "confidence": 0.78},
        ],
        "polygon": [[180, height - 10], [460, height - 10], [420, int(height * 0.5)], [220, int(height * 0.5)]],
    }


def _frame(frame_index: int, *, detections: list | None = None, lanes: dict | None = None, fps: float = 10.0) -> dict:
    return {
        "frame_index": frame_index,
        "timestamp_ms": frame_index * (1000.0 / fps),
        "lanes": lanes if lanes is not None else _lanes(),
        "detections": detections or [],
    }


def red_light_sequence() -> dict:
    frames = []
    for idx in range(8):
        detections = []
        if idx >= 2:
            detections.append(
                {
                    "kind": "traffic_light",
                    "label": "traffic light",
                    "confidence": 0.9,
                    "state": "red",
                    "box": {"x1": 300, "y1": 40, "x2": 330, "y2": 90, "width": 30, "height": 50},
                }
            )
        frames.append(_frame(idx, detections=detections))
    return {"schema_version": "0.1", "video": _base_video(), "frames": frames}


def lane_dropout_sequence() -> dict:
    frames = [_frame(idx) for idx in range(3)]
    for idx in range(3, 6):
        frames.append(_frame(idx, lanes={"lines": [], "polygon": []}))
    frames.extend([_frame(idx) for idx in range(6, 9)])
    return {"schema_version": "0.1", "video": _base_video(), "frames": frames}


def lead_close_sequence() -> dict:
    vehicle = {
        "kind": "vehicle",
        "label": "car",
        "confidence": 0.9,
        "track_id": 11,
        "distance_m": 7.5,
        "box": {"x1": 285, "y1": 210, "x2": 355, "y2": 270, "width": 70, "height": 60},
    }
    frames = [_frame(idx, detections=[vehicle]) for idx in range(6)]
    return {"schema_version": "0.1", "video": _base_video(), "frames": frames}


def vru_crossing_sequence() -> dict:
    pedestrian = {
        "kind": "pedestrian",
        "label": "person",
        "confidence": 0.82,
        "track_id": 5,
        "distance_m": 12.0,
        "box": {"x1": 305, "y1": 210, "x2": 335, "y2": 280, "width": 30, "height": 70},
    }
    frames = [_frame(idx, detections=[pedestrian]) for idx in range(5)]
    return {"schema_version": "0.1", "video": _base_video(), "frames": frames}


def id_switch_sequence() -> dict:
    frames = []
    for idx in range(4):
        vehicle = {
            "kind": "vehicle",
            "label": "car",
            "confidence": 0.88,
            "track_id": 1 if idx < 2 else 2,
            "distance_m": 10.0 - idx * 0.5,
            "box": {"x1": 285, "y1": 210, "x2": 355, "y2": 270, "width": 70, "height": 60},
        }
        frames.append(_frame(idx, detections=[vehicle]))
    return {"schema_version": "0.1", "video": _base_video(), "frames": frames}


def lead_closing_sequence() -> dict:
    frames = []
    distances = [15.0, 14.0, 12.5, 11.0, 9.5, 8.0]
    for idx, distance in enumerate(distances):
        vehicle = {
            "kind": "vehicle",
            "label": "car",
            "confidence": 0.9,
            "track_id": 3,
            "distance_m": distance,
            "box": {"x1": 285, "y1": 210, "x2": 355, "y2": 270, "width": 70, "height": 60},
        }
        frames.append(_frame(idx, detections=[vehicle]))
    return {"schema_version": "0.1", "video": _base_video(), "frames": frames}


FIXTURES = {
    "red_light_sequence.json": red_light_sequence,
    "lane_dropout.json": lane_dropout_sequence,
    "lead_close.json": lead_close_sequence,
    "vru_crossing.json": vru_crossing_sequence,
    "id_switch.json": id_switch_sequence,
    "lead_closing.json": lead_closing_sequence,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="tests/fixtures/planning")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, builder in FIXTURES.items():
        path = output_dir / filename
        with path.open("w", encoding="utf-8") as handle:
            json.dump(builder(), handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
