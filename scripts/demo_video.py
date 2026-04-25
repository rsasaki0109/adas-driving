#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adas_perception.config import apply_runtime_overrides, load_config
from adas_perception.pipeline import ADASPerceptionPipeline
from adas_perception.serialization import frame_result_to_dict, video_result_payload, write_json
from adas_perception.visualization import draw_perception


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ADAS perception demo on a video.")
    parser.add_argument("--input", required=True, help="Input video path.")
    parser.add_argument("--output", default=None, help="Output video path. Defaults to outputs/<name>_adas.mp4.")
    parser.add_argument("--config", default="configs/default.yaml", help="YAML config path.")
    parser.add_argument("--device", default=None, help="Object detector device: auto, cpu, cuda, cuda:0.")
    parser.add_argument("--json-output", default=None, help="Optional per-frame JSON result path.")
    parser.add_argument("--no-objects", action="store_true", help="Disable PyTorch object detection.")
    parser.add_argument("--display", action="store_true", help="Display frames while processing.")
    parser.add_argument("--max-frames", type=int, default=None, help="Process only the first N frames.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else Path("outputs") / f"{input_path.stem}_adas.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {input_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if args.max_frames:
        total_frames = min(total_frames, args.max_frames) if total_frames else args.max_frames

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {output_path}")

    config = apply_runtime_overrides(
        load_config(args.config),
        device=args.device,
        disable_objects=args.no_objects,
    )
    pipeline = ADASPerceptionPipeline(config)

    processed = 0
    json_frames = []
    progress = tqdm(total=total_frames or None, unit="frame")
    try:
        while True:
            if args.max_frames is not None and processed >= args.max_frames:
                break
            ok, frame = capture.read()
            if not ok:
                break
            timestamp_ms = capture.get(cv2.CAP_PROP_POS_MSEC)
            result = pipeline.run(frame)
            visualization = draw_perception(frame, result, config)
            writer.write(visualization)
            if args.json_output:
                json_frames.append(
                    frame_result_to_dict(
                        frame_index=processed,
                        timestamp_ms=timestamp_ms,
                        result=result,
                    )
                )
            processed += 1
            progress.update(1)

            if args.display:
                cv2.imshow("adas-perception", visualization)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        progress.close()
        capture.release()
        writer.release()
        if args.display:
            cv2.destroyAllWindows()

    if args.json_output:
        json_payload = video_result_payload(
            source=str(input_path),
            output=str(output_path),
            frames=json_frames,
            width=width,
            height=height,
            fps=float(fps),
            processed_frames=processed,
            config_path=str(args.config),
        )
        write_json(args.json_output, json_payload)

    print(f"Saved {output_path} | frames: {processed}")
    if args.json_output:
        print(f"Saved JSON {args.json_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
