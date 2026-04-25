#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adas_perception.config import apply_runtime_overrides, load_config
from adas_perception.pipeline import ADASPerceptionPipeline
from adas_perception.visualization import draw_perception


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark adas-perception latency and FPS.")
    parser.add_argument("--input", required=True, help="Input image or video path.")
    parser.add_argument("--input-type", choices=["auto", "image", "video"], default="auto")
    parser.add_argument("--config", default="configs/default.yaml", help="YAML config path.")
    parser.add_argument("--device", default=None, help="Object detector device: auto, cpu, cuda, cuda:0.")
    parser.add_argument("--no-objects", action="store_true", help="Disable PyTorch object detection.")
    parser.add_argument("--max-frames", type=int, default=120, help="Maximum video frames to benchmark.")
    parser.add_argument("--repeat", type=int, default=30, help="Number of image repeats for image input.")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations before timing.")
    parser.add_argument(
        "--include-visualization",
        action="store_true",
        help="Include OpenCV drawing time in per-frame latency.",
    )
    parser.add_argument("--output", default=None, help="Optional JSON benchmark report path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    input_type = _resolve_input_type(input_path, args.input_type)
    frames, source_info = _load_frames(input_path, input_type, args.max_frames, args.repeat)
    if not frames:
        raise RuntimeError(f"No frames loaded from {input_path}")

    config = apply_runtime_overrides(
        load_config(args.config),
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
    total_start = time.perf_counter()
    for frame in frames:
        _sync_if_cuda()
        frame_start = time.perf_counter()
        result = pipeline.run(frame)
        if args.include_visualization:
            draw_perception(frame, result, config)
        _sync_if_cuda()
        timings_ms.append((time.perf_counter() - frame_start) * 1000.0)
    total_ms = (time.perf_counter() - total_start) * 1000.0

    report = _build_report(
        args=args,
        source_info=source_info,
        frame_count=len(frames),
        init_ms=init_ms,
        total_ms=total_ms,
        timings_ms=timings_ms,
        objects_enabled=bool(config.get("objects", {}).get("enabled", True)),
    )
    print_report(report)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
            f.write("\n")
        print(f"Saved {output_path}")
    return 0


def print_report(report: dict[str, Any]) -> None:
    latency = report["latency_ms"]
    print("ADAS perception benchmark")
    print(f"- source: {report['source']['path']}")
    print(f"- input_type: {report['source']['type']}")
    print(f"- frames: {report['frames']}")
    print(f"- measured_stage: {report['measured_stage']}")
    print(f"- model_init_ms: {report['model_init_ms']:.3f}")
    print(f"- total_ms: {report['total_ms']:.3f}")
    print(f"- fps: {report['fps']:.3f}")
    print(
        "- latency_ms: "
        f"mean={latency['mean']:.3f}, "
        f"p50={latency['p50']:.3f}, "
        f"p95={latency['p95']:.3f}, "
        f"min={latency['min']:.3f}, "
        f"max={latency['max']:.3f}"
    )


def _resolve_input_type(input_path: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    return "image" if input_path.suffix.lower() in IMAGE_SUFFIXES else "video"


def _load_frames(
    input_path: Path,
    input_type: str,
    max_frames: int,
    repeat: int,
) -> tuple[list[Any], dict[str, Any]]:
    if input_type == "image":
        frame = cv2.imread(str(input_path))
        if frame is None:
            raise FileNotFoundError(f"Could not read image: {input_path}")
        repeats = max(1, repeat)
        return [frame] * repeats, {
            "path": str(input_path),
            "type": "image",
            "width": frame.shape[1],
            "height": frame.shape[0],
            "repeat": repeats,
        }

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {input_path}")

    frames = []
    limit = max(1, max_frames)
    try:
        while len(frames) < limit:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    finally:
        capture.release()

    return frames, {
        "path": str(input_path),
        "type": "video",
        "width": width,
        "height": height,
        "source_fps": fps,
        "max_frames": limit,
    }


def _build_report(
    *,
    args: argparse.Namespace,
    source_info: dict[str, Any],
    frame_count: int,
    init_ms: float,
    total_ms: float,
    timings_ms: list[float],
    objects_enabled: bool,
) -> dict[str, Any]:
    return {
        "schema_version": "0.1",
        "source": source_info,
        "config": args.config,
        "device": args.device or "config",
        "objects_enabled": objects_enabled,
        "measured_stage": "pipeline+visualization" if args.include_visualization else "pipeline",
        "warmup": max(0, args.warmup),
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
    }


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile / 100.0
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _sync_if_cuda() -> None:
    torch = sys.modules.get("torch")
    if torch is None:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()


if __name__ == "__main__":
    raise SystemExit(main())
