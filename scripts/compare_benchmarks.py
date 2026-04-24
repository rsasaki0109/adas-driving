#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from benchmark import _load_frames, _percentile, _resolve_input_type, _sync_if_cuda

from adas_perception.config import apply_runtime_overrides, load_config
from adas_perception.pipeline import ADASPerceptionPipeline
from adas_perception.visualization import draw_perception


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare multiple adas-perception configs on one input.")
    parser.add_argument("--input", required=True, help="Input image or video path.")
    parser.add_argument("--configs", nargs="+", required=True, help="Config YAML paths to compare.")
    parser.add_argument("--input-type", choices=["auto", "image", "video"], default="auto")
    parser.add_argument("--device", default=None, help="Override object detector device for all configs.")
    parser.add_argument("--no-objects", action="store_true", help="Disable TorchVision object detection for all configs.")
    parser.add_argument("--max-frames", type=int, default=120, help="Maximum video frames to benchmark.")
    parser.add_argument("--repeat", type=int, default=30, help="Number of image repeats for image input.")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations before timing.")
    parser.add_argument("--include-visualization", action="store_true", help="Include drawing time in latency.")
    parser.add_argument("--output", default=None, help="Optional JSON comparison report path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    input_type = _resolve_input_type(input_path, args.input_type)
    frames, source_info = _load_frames(input_path, input_type, args.max_frames, args.repeat)
    if not frames:
        raise RuntimeError(f"No frames loaded from {input_path}")

    results = []
    for config_path in args.configs:
        print(f"Running {config_path} ...")
        result = run_one_config(
            config_path=config_path,
            frames=frames,
            source_info=source_info,
            args=args,
        )
        results.append(result)
        _print_row(result)

    report = {
        "schema_version": "0.1",
        "source": source_info,
        "measured_stage": "pipeline+visualization" if args.include_visualization else "pipeline",
        "warmup": max(0, args.warmup),
        "frames": len(frames),
        "results": results,
        "best": _best_summary(results),
    }

    print_summary(report)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
            f.write("\n")
        print(f"Saved {output_path}")
    return 0


def run_one_config(
    *,
    config_path: str,
    frames: list[Any],
    source_info: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    config = apply_runtime_overrides(
        load_config(config_path),
        device=args.device,
        disable_objects=args.no_objects,
    )

    init_start = time.perf_counter()
    pipeline = ADASPerceptionPipeline(config)
    _sync_if_cuda()
    init_ms = (time.perf_counter() - init_start) * 1000.0

    warmup_frames = frames[: max(0, min(args.warmup, len(frames)))]
    for frame in warmup_frames:
        result = pipeline.run(frame)
        if args.include_visualization:
            draw_perception(frame, result, config)
    _sync_if_cuda()
    pipeline.reset()

    timings_ms: list[float] = []
    counts = Counter()
    lane_line_counts: list[int] = []
    frames_with_lanes = 0

    total_start = time.perf_counter()
    for frame in frames:
        _sync_if_cuda()
        frame_start = time.perf_counter()
        result = pipeline.run(frame)
        if args.include_visualization:
            draw_perception(frame, result, config)
        _sync_if_cuda()
        timings_ms.append((time.perf_counter() - frame_start) * 1000.0)

        for detection in result.detections:
            counts[detection.kind] += 1
        lane_lines = len(result.lanes.lines)
        lane_line_counts.append(lane_lines)
        if lane_lines:
            frames_with_lanes += 1

    total_ms = (time.perf_counter() - total_start) * 1000.0
    frame_count = len(frames)

    return {
        "config": config_path,
        "source": source_info,
        "device": args.device or str(config.get("objects", {}).get("device", "config")),
        "objects_enabled": bool(config.get("objects", {}).get("enabled", True)),
        "frames": frame_count,
        "model_init_ms": round(init_ms, 4),
        "total_ms": round(total_ms, 4),
        "fps": round(frame_count / max(total_ms / 1000.0, 1e-9), 4),
        "latency_ms": {
            "mean": round(statistics.mean(timings_ms), 4),
            "p50": round(_percentile(timings_ms, 50), 4),
            "p95": round(_percentile(timings_ms, 95), 4),
            "min": round(min(timings_ms), 4),
            "max": round(max(timings_ms), 4),
        },
        "detections": {
            "total": int(sum(counts.values())),
            "by_kind": dict(sorted(counts.items())),
            "per_frame_average": round(sum(counts.values()) / max(frame_count, 1), 4),
        },
        "lanes": {
            "frames_with_lanes": frames_with_lanes,
            "average_lines_per_frame": round(statistics.mean(lane_line_counts), 4)
            if lane_line_counts
            else 0.0,
            "max_lines_per_frame": max(lane_line_counts, default=0),
        },
    }


def print_summary(report: dict[str, Any]) -> None:
    best = report["best"]
    print("Comparison summary")
    print(f"- fastest_fps: {best.get('fastest_fps', {}).get('config')} ({best.get('fastest_fps', {}).get('fps')})")
    print(
        "- lowest_p95_latency: "
        f"{best.get('lowest_p95_latency', {}).get('config')} "
        f"({best.get('lowest_p95_latency', {}).get('p95_ms')} ms)"
    )
    print(
        "- most_detections: "
        f"{best.get('most_detections', {}).get('config')} "
        f"({best.get('most_detections', {}).get('detections')})"
    )


def _print_row(result: dict[str, Any]) -> None:
    latency = result["latency_ms"]
    detections = result["detections"]
    print(
        f"- {result['config']}: "
        f"fps={result['fps']:.3f}, "
        f"mean={latency['mean']:.3f}ms, "
        f"p95={latency['p95']:.3f}ms, "
        f"detections={detections['total']}, "
        f"lanes={result['lanes']['frames_with_lanes']}/{result['frames']}"
    )


def _best_summary(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    if not results:
        return {}
    fastest = max(results, key=lambda item: item["fps"])
    lowest_p95 = min(results, key=lambda item: item["latency_ms"]["p95"])
    most_detections = max(results, key=lambda item: item["detections"]["total"])
    return {
        "fastest_fps": {
            "config": fastest["config"],
            "fps": fastest["fps"],
        },
        "lowest_p95_latency": {
            "config": lowest_p95["config"],
            "p95_ms": lowest_p95["latency_ms"]["p95"],
        },
        "most_detections": {
            "config": most_detections["config"],
            "detections": most_detections["detections"]["total"],
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())
