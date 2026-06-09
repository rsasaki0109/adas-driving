#!/usr/bin/env python3
"""Export a finetuned Ultralytics YOLO detector to ONNX for edge deployment."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_MODEL = ROOT / "outputs/models/adas_yolov8n_bdd100k.pt"
DEFAULT_OUTPUT_DIR = ROOT / "outputs/models"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL),
        help="Input .pt checkpoint path.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for the exported .onnx file.",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="Optional ONNX filename (default: <stem>_<imgsz>.onnx).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Square export input size. 640 is the Jetson-friendly starting point.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=12,
        help="ONNX opset version passed to Ultralytics export.",
    )
    parser.add_argument(
        "--simplify",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run ONNX simplifier when supported by Ultralytics.",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="Export FP16 ONNX (GPU/TensorRT oriented).",
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Export dynamic batch/height/width axes (usually off for Jetson).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Export device passed to Ultralytics (cpu is safest for CI/dev machines).",
    )
    parser.add_argument(
        "--write-manifest",
        action="store_true",
        help="Write a small JSON manifest next to the ONNX file.",
    )
    return parser.parse_args()


def resolve_output_path(model_path: Path, output_dir: Path, imgsz: int, output_name: str | None) -> Path:
    filename = output_name or f"{model_path.stem}_{imgsz}.onnx"
    return output_dir / filename


def export_yolo_onnx(
    *,
    model_path: Path,
    output_path: Path,
    imgsz: int,
    opset: int,
    simplify: bool,
    half: bool,
    dynamic: bool,
    device: str,
) -> Path:
    if not model_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "ultralytics is required for ONNX export. Install with: pip install -e '.[yolo]'"
        ) from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    exported = Path(
        YOLO(str(model_path)).export(
            format="onnx",
            imgsz=imgsz,
            opset=opset,
            simplify=simplify,
            half=half,
            dynamic=dynamic,
            device=device,
        )
    )
    if exported.resolve() != output_path.resolve():
        shutil.move(str(exported), str(output_path))
    return output_path


def write_manifest(path: Path, payload: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def main() -> int:
    args = parse_args()
    model_path = Path(args.model)
    output_dir = Path(args.output_dir)
    output_path = resolve_output_path(model_path, output_dir, args.imgsz, args.output_name)

    onnx_path = export_yolo_onnx(
        model_path=model_path,
        output_path=output_path,
        imgsz=args.imgsz,
        opset=args.opset,
        simplify=args.simplify,
        half=args.half,
        dynamic=args.dynamic,
        device=args.device,
    )

    summary = {
        "model": str(model_path),
        "onnx": str(onnx_path),
        "imgsz": args.imgsz,
        "opset": args.opset,
        "simplify": args.simplify,
        "half": args.half,
        "dynamic": args.dynamic,
        "device": args.device,
        "next_steps": [
            "Point configs/bdd100k_yolo_jetson_640_onnx.yaml at the exported ONNX path.",
            "On Jetson: trtexec --onnx=<onnx> --saveEngine=<engine> --fp16",
            "Benchmark with: python scripts/benchmark.py --config configs/bdd100k_yolo_jetson_640_onnx.yaml",
        ],
    }
    if args.write_manifest:
        write_manifest(onnx_path.with_suffix(".onnx.json"), summary)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
