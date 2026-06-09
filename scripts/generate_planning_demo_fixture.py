#!/usr/bin/env python3
"""Generate synthetic perception JSON aligned with assets/demo_wbf7.mp4 (640x360 @ 10fps)."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _lane_lines(frame_index: int, *, width: int = 640, height: int = 360) -> dict:
    sway = int(8 * math.sin(frame_index * 0.18))
    left_x = 190 + sway
    right_x = 450 - sway
    y_bottom = height - 10
    y_top = int(height * 0.45)
    return {
        "lines": [
            {
                "side": "left",
                "points": [[left_x, y_bottom], [left_x + 30, y_top]],
                "confidence": 0.82,
            },
            {
                "side": "right",
                "points": [[right_x, y_bottom], [right_x - 30, y_top]],
                "confidence": 0.80,
            },
        ],
        "polygon": [
            [left_x, y_bottom],
            [right_x, y_bottom],
            [right_x - 30, y_top],
            [left_x + 30, y_top],
        ],
    }


def _vehicle(track_id: int, distance_m: float, *, width: int = 640) -> dict:
    center_x = width // 2
    box_w, box_h = 70, 55
    y2 = 250
    y1 = y2 - box_h
    x1 = center_x - box_w // 2
    x2 = center_x + box_w // 2
    return {
        "kind": "vehicle",
        "label": "car",
        "confidence": 0.88,
        "track_id": track_id,
        "distance_m": distance_m,
        "box": {
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "width": box_w,
            "height": box_h,
        },
    }


def _traffic_light(state: str, confidence: float = 0.85) -> dict:
    return {
        "kind": "traffic_light",
        "label": "traffic light",
        "confidence": confidence,
        "state": state,
        "box": {"x1": 305, "y1": 40, "x2": 335, "y2": 95, "width": 30, "height": 55},
    }


def _pedestrian(distance_m: float, *, width: int = 640) -> dict:
    center_x = width // 2 + 20
    box_w, box_h = 24, 60
    y2 = 280
    y1 = y2 - box_h
    return {
        "kind": "pedestrian",
        "label": "person",
        "confidence": 0.78,
        "track_id": 99,
        "distance_m": distance_m,
        "box": {
            "x1": center_x - box_w // 2,
            "y1": y1,
            "x2": center_x + box_w // 2,
            "y2": y2,
            "width": box_w,
            "height": box_h,
        },
    }


def build_demo_payload(*, frame_count: int = 40, width: int = 640, height: int = 360, fps: float = 10.0) -> dict:
    frames: list[dict] = []
    for frame_index in range(frame_count):
        detections: list[dict] = []
        if 12 <= frame_index <= 22:
            distance = 14.0 - (frame_index - 12) * 0.8
            detections.append(_vehicle(track_id=7, distance_m=max(5.5, distance), width=width))
        if 24 <= frame_index <= 27:
            detections.append(_pedestrian(18.0 - (frame_index - 24) * 2.0, width=width))
        if 28 <= frame_index <= 33:
            detections.append(_traffic_light("red"))
        elif frame_index >= 34:
            detections.append(_traffic_light("green"))

        frames.append(
            {
                "frame_index": frame_index,
                "timestamp_ms": frame_index * (1000.0 / fps),
                "lanes": _lane_lines(frame_index, width=width, height=height),
                "detections": detections,
            }
        )

    return {
        "schema_version": "0.1",
        "video": {"width": width, "height": height, "fps": fps, "frame_count": frame_count},
        "frames": frames,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="examples/fixtures/planning_demo_perception.json",
        help="Output perception JSON path.",
    )
    parser.add_argument("--frames", type=int, default=40)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_demo_payload(frame_count=args.frames)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"Wrote {output_path} ({args.frames} frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
