#!/usr/bin/env python3
"""Compare lane detectors on a directory of images.

Two modes:

1. Without `--labels`: rolls each configured detector over a set of
   images and reports comparative stats (frames with >=1 lane, mean
   lane line count, mean polyline length, polygon coverage). Useful for
   quick A/B between the OpenCV detector and a learned segmentation
   model when no ground-truth labels are available.

2. With `--labels` pointing at a BDD100K-style lane label JSON
   (vector polylines, available from the official BDD100K download
   page as `bdd100k_lane_labels_trainval.zip`), additionally rasterize
   the GT polylines into a binary mask and compute pixel-level IoU /
   F1 against each detector's output mask. Reports per-detector
   averages.

Usage:

  # Comparative stats on 100 val images (no labels needed)
  python scripts/evaluate_lane.py \
    --images-root data/bdd100k/images/100k/val \
    --max-images 100 \
    --configs cv=configs/default.yaml seg=configs/lane_twinlitenet.yaml

  # With BDD100K lane labels for IoU/F1
  python scripts/evaluate_lane.py \
    --images-root data/bdd100k/images/100k/val \
    --labels data/bdd100k/labels/lane/lane_val.json \
    --configs cv=configs/default.yaml seg=configs/lane_twinlitenet.yaml \
    --output outputs/lane_eval_compare.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adas_perception.config import load_config
from adas_perception.detectors import create_lane_detector
from adas_perception.types import LaneResult


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare lane detectors with optional BDD100K lane labels.")
    parser.add_argument("--images-root", required=True, help="Directory containing images.")
    parser.add_argument(
        "--configs",
        nargs="+",
        required=True,
        metavar="NAME=CONFIG_PATH",
        help="One or more name=config pairs. Names are reported in the comparison.",
    )
    parser.add_argument("--labels", default=None, help="Optional BDD100K-style lane label JSON path.")
    parser.add_argument("--max-images", type=int, default=200)
    parser.add_argument("--mask-thickness", type=int, default=8, help="Pixel thickness when rasterizing polylines for IoU.")
    parser.add_argument("--output", default=None, help="Optional JSON output path for the comparison.")
    parser.add_argument("--progress-every", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    images_root = Path(args.images_root)
    if not images_root.is_dir():
        raise SystemExit(f"images-root not found: {images_root}")

    detectors: list[tuple[str, Any]] = []
    for entry in args.configs:
        if "=" not in entry:
            raise SystemExit(f"--configs expects name=path, got {entry}")
        name, path = entry.split("=", 1)
        cfg = load_config(path)
        lane_cfg = cfg.get("lane", {})
        lane_cfg.setdefault("enabled", True)
        detectors.append((name.strip(), create_lane_detector(lane_cfg)))

    label_polys = _load_lane_labels(args.labels) if args.labels else None
    image_paths = _select_images(images_root, label_polys, args.max_images)
    if not image_paths:
        raise SystemExit("no images to evaluate")

    accumulators = {name: _new_accumulator() for name, _ in detectors}

    for index, image_path in enumerate(image_paths, start=1):
        frame = cv2.imread(str(image_path))
        if frame is None:
            continue
        height, width = frame.shape[:2]
        gt_mask = None
        if label_polys is not None and image_path.name in label_polys:
            gt_mask = _rasterize_polylines(label_polys[image_path.name], width, height, args.mask_thickness)

        for name, detector in detectors:
            try:
                result = detector.detect(frame)
            except Exception:
                continue
            _update_accumulator(accumulators[name], result, gt_mask, width, height, args.mask_thickness)

        if args.progress_every and index % args.progress_every == 0:
            print(f"  processed {index}/{len(image_paths)}", flush=True)

    summary = {
        "images_root": str(images_root),
        "labels": str(args.labels) if args.labels else None,
        "processed_images": len(image_paths),
        "mask_thickness": args.mask_thickness,
        "results": {name: _finalize(acc) for name, acc in accumulators.items()},
    }
    _print_summary(summary)
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\nSaved {out_path}")
    return 0


def _new_accumulator() -> dict[str, Any]:
    return {
        "n": 0,
        "frames_with_lane": 0,
        "lines_total": 0,
        "polyline_length_total": 0,
        "polygon_pts_total": 0,
        "iou_sum": 0.0,
        "f1_sum": 0.0,
        "iou_n": 0,
    }


def _update_accumulator(
    acc: dict[str, Any],
    result: LaneResult,
    gt_mask: np.ndarray | None,
    width: int,
    height: int,
    thickness: int,
) -> None:
    acc["n"] += 1
    if result.lines:
        acc["frames_with_lane"] += 1
    acc["lines_total"] += len(result.lines)
    for line in result.lines:
        acc["polyline_length_total"] += len(line.polyline) if line.polyline else 2
    acc["polygon_pts_total"] += len(result.polygon)

    if gt_mask is None:
        return
    pred_mask = _result_to_mask(result, width, height, thickness)
    inter = int(np.logical_and(pred_mask, gt_mask).sum())
    pred_pos = int(pred_mask.sum())
    gt_pos = int(gt_mask.sum())
    union = pred_pos + gt_pos - inter
    if union == 0:
        return
    iou = inter / union
    precision = inter / max(pred_pos, 1)
    recall = inter / max(gt_pos, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)
    acc["iou_sum"] += iou
    acc["f1_sum"] += f1
    acc["iou_n"] += 1


def _finalize(acc: dict[str, Any]) -> dict[str, Any]:
    n = max(acc["n"], 1)
    out: dict[str, Any] = {
        "frames_evaluated": acc["n"],
        "detection_rate": acc["frames_with_lane"] / n,
        "mean_lines_per_frame": acc["lines_total"] / n,
        "mean_polyline_length": acc["polyline_length_total"] / max(acc["lines_total"], 1),
        "mean_polygon_points": acc["polygon_pts_total"] / n,
    }
    if acc["iou_n"] > 0:
        out["mean_iou"] = acc["iou_sum"] / acc["iou_n"]
        out["mean_f1"] = acc["f1_sum"] / acc["iou_n"]
        out["iou_frames"] = acc["iou_n"]
    return out


def _print_summary(summary: dict[str, Any]) -> None:
    print(f"Lane detector comparison ({summary['processed_images']} images)")
    if summary.get("labels"):
        print(f"  labels: {summary['labels']}")
    for name, stats in summary["results"].items():
        line = f"- {name:<10s} det_rate={stats['detection_rate']:.2%}"
        line += f"  lines/frame={stats['mean_lines_per_frame']:.2f}"
        line += f"  polyline_len={stats['mean_polyline_length']:.1f}"
        if "mean_iou" in stats:
            line += f"  IoU={stats['mean_iou']:.3f}"
            line += f"  F1={stats['mean_f1']:.3f}"
            line += f"  (n_iou={stats['iou_frames']})"
        print(line)


def _select_images(images_root: Path, label_polys: dict | None, max_images: int) -> list[Path]:
    if label_polys is not None:
        candidates = [images_root / name for name in label_polys.keys() if (images_root / name).is_file()]
    else:
        candidates = sorted(images_root.glob("*.jpg")) + sorted(images_root.glob("*.png"))
    return candidates[:max_images] if max_images else candidates


def _load_lane_labels(path: str) -> dict[str, list[list[tuple[int, int]]]]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    out: dict[str, list[list[tuple[int, int]]]] = {}
    for frame in data:
        name = frame.get("name") or frame.get("file_name") or ""
        lines: list[list[tuple[int, int]]] = []
        for label in frame.get("labels", []) or []:
            poly2d = label.get("poly2d") or []
            for poly in poly2d:
                vertices = poly.get("vertices") if isinstance(poly, dict) else poly
                if not vertices:
                    continue
                pts = [(int(round(v[0])), int(round(v[1]))) for v in vertices if len(v) >= 2]
                if len(pts) >= 2:
                    lines.append(pts)
        if name and lines:
            out[name] = lines
    return out


def _rasterize_polylines(
    polylines: list[list[tuple[int, int]]],
    width: int,
    height: int,
    thickness: int,
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    for pts in polylines:
        arr = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(mask, [arr], False, 255, max(1, thickness), cv2.LINE_AA)
    return (mask > 0).astype(np.uint8)


def _result_to_mask(result: LaneResult, width: int, height: int, thickness: int) -> np.ndarray:
    polylines: list[list[tuple[int, int]]] = []
    for line in result.lines:
        if line.polyline and len(line.polyline) >= 2:
            polylines.append(list(line.polyline))
        else:
            polylines.append(list(line.points))
    return _rasterize_polylines(polylines, width, height, thickness)


if __name__ == "__main__":
    raise SystemExit(main())
