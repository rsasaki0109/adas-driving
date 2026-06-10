#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adas_planning.config import load_config
from adas_planning.io.planning_json import planning_result_from_dict
from adas_planning.viz.overlay import draw_planning_overlay


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render planning overlay video from saved JSON.")
    parser.add_argument("--video", required=True, help="Input video path.")
    parser.add_argument("--perception-json", required=True, help="Perception JSON path.")
    parser.add_argument("--planning-json", required=True, help="Planning JSON path.")
    parser.add_argument("--config", default="configs/planning/default.yaml", help="Planning overlay config.")
    parser.add_argument("--output", required=True, help="Output overlay video path.")
    parser.add_argument("--max-frames", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)

    with Path(args.perception_json).open("r", encoding="utf-8") as handle:
        perception_payload = json.load(handle)
    with Path(args.planning_json).open("r", encoding="utf-8") as handle:
        planning_payload = json.load(handle)

    perception_frames = perception_payload.get("frames") or []
    planning_frames = planning_payload.get("frames") or []
    if not perception_frames or not planning_frames:
        raise SystemExit("Both perception and planning JSON must contain frames[]")

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {args.video}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        str(args.output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open writer: {args.output}")

    total = min(len(perception_frames), len(planning_frames))
    if args.max_frames is not None:
        total = min(total, args.max_frames)

    for idx in tqdm(range(total), unit="frame"):
        ok, frame = capture.read()
        if not ok:
            break
        planning_result = planning_result_from_dict(planning_frames[idx])
        overlay = draw_planning_overlay(
            frame,
            planning_result,
            perception_frame=perception_frames[idx],
            config=config,
        )
        writer.write(overlay)

    capture.release()
    writer.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
