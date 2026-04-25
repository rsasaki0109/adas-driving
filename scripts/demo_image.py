#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adas_perception.config import apply_runtime_overrides, load_config
from adas_perception.pipeline import ADASPerceptionPipeline
from adas_perception.serialization import image_result_payload, write_json
from adas_perception.visualization import draw_perception


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ADAS perception demo on one image.")
    parser.add_argument("--input", required=True, help="Input image path.")
    parser.add_argument("--output", default=None, help="Output image path. Defaults to outputs/<name>_adas.jpg.")
    parser.add_argument("--config", default="configs/default.yaml", help="YAML config path.")
    parser.add_argument("--device", default=None, help="Object detector device: auto, cpu, cuda, cuda:0.")
    parser.add_argument("--json-output", default=None, help="Optional JSON result path.")
    parser.add_argument("--no-objects", action="store_true", help="Disable PyTorch object detection.")
    parser.add_argument("--show", action="store_true", help="Show the result in an OpenCV window.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else Path("outputs") / f"{input_path.stem}_adas.jpg"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame = cv2.imread(str(input_path))
    if frame is None:
        raise FileNotFoundError(f"Could not read image: {input_path}")

    config = apply_runtime_overrides(
        load_config(args.config),
        device=args.device,
        disable_objects=args.no_objects,
    )
    pipeline = ADASPerceptionPipeline(config)
    result = pipeline.run(frame)
    visualization = draw_perception(frame, result, config)

    ok = cv2.imwrite(str(output_path), visualization)
    if not ok:
        raise RuntimeError(f"Could not write output image: {output_path}")

    if args.json_output:
        json_payload = image_result_payload(
            source=str(input_path),
            output=str(output_path),
            result=result,
            image_shape=frame.shape,
            config_path=str(args.config),
        )
        write_json(args.json_output, json_payload)

    counts = result.count_by_kind()
    print(
        "Saved",
        output_path,
        "| lanes:",
        counts.get("lane", 0),
        "vehicles:",
        counts.get("vehicle", 0),
        "pedestrians:",
        counts.get("pedestrian", 0),
        "signs:",
        counts.get("traffic_sign", 0),
        "lights:",
        counts.get("traffic_light", 0),
    )
    if args.json_output:
        print("Saved JSON", args.json_output)

    if args.show:
        cv2.imshow("adas-perception", visualization)
        cv2.waitKey(0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
