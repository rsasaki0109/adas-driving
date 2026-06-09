#!/usr/bin/env python3
"""Sweep post-fusion NMS IoU grids on cached BDD100K prediction JSON."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from itertools import product
from pathlib import Path
import sys
from typing import Any

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adas_perception.postprocess import nms_by_kind
from adas_perception.types import Box, Detection
from scripts.evaluate_bdd100k import (
    DEFAULT_KINDS,
    _build_report,
    _empty_counter,
    _ground_truth_by_kind,
    _load_label_frames,
    _match_boxes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-root", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--kinds", nargs="+", default=DEFAULT_KINDS)
    parser.add_argument("--score-thresholds", nargs="+", type=float, default=[0.25])
    parser.add_argument(
        "--kind-score-thresholds",
        nargs="+",
        default=None,
        metavar="KIND=THRESHOLD",
        help="Fixed per-kind score thresholds, e.g. pedestrian=0.25 traffic_light=0.20",
    )
    parser.add_argument(
        "--default-nms-iou",
        nargs="+",
        type=float,
        default=[0.50],
        help="Default NMS IoU grid applied when a kind has no override.",
    )
    parser.add_argument(
        "--kind-nms-iou",
        nargs="+",
        default=None,
        metavar="KIND=IOU,...",
        help="Per-kind NMS IoU grid, e.g. traffic_light=0.35,0.40 traffic_sign=0.35,0.40",
    )
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--frame-offset", type=int, default=0)
    parser.add_argument("--iou-threshold", type=float, default=0.50)
    parser.add_argument("--top-k", type=int, default=10, help="Write top-K reports by macro F1.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_kinds = set(args.kinds)
    kind_thresholds = _parse_kind_values(args.kind_score_thresholds)
    kind_nms_grid = _parse_kind_value_grid(args.kind_nms_iou)
    frames, frame_selection = _load_label_frames(
        Path(args.labels),
        max_images=args.max_images,
        frame_stride=args.frame_stride,
        frame_offset=args.frame_offset,
    )
    predictions_by_name = _load_predictions(Path(args.predictions), selected_kinds)
    frame_records = _build_frame_records(
        frames,
        Path(args.images_root),
        selected_kinds,
        predictions_by_name,
    )

    results: list[dict[str, Any]] = []
    for default_threshold in args.score_thresholds:
        for default_nms_iou in args.default_nms_iou:
            for kind_nms in _iter_kind_value_combos(kind_nms_grid):
                nms_by_kind_map = dict(kind_nms)
                report = _evaluate_combo(
                    frame_records=frame_records,
                    labels_path=args.labels,
                    images_root=args.images_root,
                    predictions_path=args.predictions,
                    selected_kinds=selected_kinds,
                    default_threshold=default_threshold,
                    kind_thresholds=kind_thresholds,
                    default_nms_iou=default_nms_iou,
                    nms_by_kind_map=nms_by_kind_map,
                    iou_threshold=args.iou_threshold,
                    frame_selection=frame_selection,
                )
                results.append(report)

    results.sort(key=_macro_f1, reverse=True)
    summary = {
        "schema_version": "bdd100k_postprocess_sweep.v0.1",
        "predictions": args.predictions,
        "evaluated_combinations": len(results),
        "best": results[0] if results else None,
        "top": results[: max(args.top_k, 0)],
    }
    _write_json(output_dir / "summary.json", summary)
    for idx, report in enumerate(results[: max(args.top_k, 0)], start=1):
        tag = _report_tag(report)
        _write_json(output_dir / f"rank_{idx:02d}_{tag}.json", report)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def _evaluate_combo(
    *,
    frame_records: list[dict[str, Any]],
    labels_path: str,
    images_root: str,
    predictions_path: str,
    selected_kinds: set[str],
    default_threshold: float,
    kind_thresholds: dict[str, float],
    default_nms_iou: float,
    nms_by_kind_map: dict[str, float],
    iou_threshold: float,
    frame_selection: dict[str, int | None],
) -> dict[str, Any]:
    totals = {kind: _empty_counter() for kind in selected_kinds}
    for record in frame_records:
        filtered = _filter_by_score(
            record["predictions"],
            selected_kinds=selected_kinds,
            default_threshold=default_threshold,
            kind_thresholds=kind_thresholds,
        )
        nmsed = _apply_nms(filtered, default_nms_iou=default_nms_iou, nms_by_kind_map=nms_by_kind_map)
        pred_by_kind = _group_by_kind(nmsed, selected_kinds)
        gt_by_kind = record["gt_by_kind"]
        for kind in selected_kinds:
            matched = _match_boxes(gt_by_kind.get(kind, []), pred_by_kind.get(kind, []), iou_threshold)
            totals[kind]["tp"] += matched["tp"]
            totals[kind]["fp"] += matched["fp"]
            totals[kind]["fn"] += matched["fn"]

    report = _build_report(
        labels_path=labels_path,
        images_root=images_root,
        config_path=f"postprocess_sweep:{predictions_path}",
        kinds=selected_kinds,
        totals=totals,
        state_totals={"matched": 0, "correct": 0},
        lane_presence={"tp": 0, "fp": 0, "fn": 0},
        processed=len(frame_records),
        missing_images=[],
        total_inference_ms=0.0,
        iou_threshold=iou_threshold,
        frame_selection=frame_selection,
        group_by=[],
        grouped_totals={},
        size_bucket_areas=[],
        size_totals={},
    )
    report["postprocess"] = {
        "default_score_threshold": round(float(default_threshold), 4),
        "kind_score_thresholds": {k: round(v, 4) for k, v in sorted(kind_thresholds.items())},
        "default_nms_iou": round(float(default_nms_iou), 4),
        "nms_iou_by_kind": {k: round(v, 4) for k, v in sorted(nms_by_kind_map.items())},
    }
    return report


def _filter_by_score(
    predictions: list[Detection],
    *,
    selected_kinds: set[str],
    default_threshold: float,
    kind_thresholds: dict[str, float],
) -> list[Detection]:
    kept: list[Detection] = []
    for detection in predictions:
        if detection.kind not in selected_kinds:
            continue
        threshold = float(kind_thresholds.get(detection.kind, default_threshold))
        if detection.confidence >= threshold:
            kept.append(detection)
    return kept


def _apply_nms(
    detections: list[Detection],
    *,
    default_nms_iou: float,
    nms_by_kind_map: dict[str, float],
) -> list[Detection]:
    return nms_by_kind(detections, default_iou=default_nms_iou, iou_by_kind=nms_by_kind_map)


def _group_by_kind(detections: list[Detection], selected_kinds: set[str]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for detection in detections:
        if detection.kind not in selected_kinds:
            continue
        grouped[detection.kind].append(
            {
                "box": detection.box,
                "label": detection.label,
                "state": None,
                "confidence": detection.confidence,
            }
        )
    return grouped


def _build_frame_records(
    frames: list[dict[str, Any]],
    images_root: Path,
    selected_kinds: set[str],
    predictions_by_name: dict[str, list[Detection]],
) -> list[dict[str, Any]]:
    records = []
    for frame in frames:
        name = str(frame.get("name", ""))
        gt_by_kind, _ = _ground_truth_by_kind(frame, selected_kinds)
        image = cv2.imread(str(images_root / name))
        if image is None:
            continue
        records.append(
            {
                "name": name,
                "gt_by_kind": gt_by_kind,
                "predictions": predictions_by_name.get(name, []),
            }
        )
    return records


def _load_predictions(path: Path, selected_kinds: set[str]) -> dict[str, list[Detection]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    by_name: dict[str, list[Detection]] = {}
    for item in payload.get("predictions", []):
        detections: list[Detection] = []
        for detection in item.get("detections", []):
            kind = str(detection.get("kind", ""))
            if kind not in selected_kinds:
                continue
            box = detection.get("box", {})
            detections.append(
                Detection(
                    kind=kind,
                    label=str(detection.get("label", "")),
                    confidence=float(detection.get("confidence", 0.0)),
                    box=Box(
                        x1=int(box.get("x1", 0)),
                        y1=int(box.get("y1", 0)),
                        x2=int(box.get("x2", 0)),
                        y2=int(box.get("y2", 0)),
                    ),
                    source=str(detection.get("source", "cache")),
                )
            )
        by_name[str(item.get("name", ""))] = detections
    return by_name


def _parse_kind_values(raw_items: list[str] | None) -> dict[str, float]:
    if not raw_items:
        return {}
    values: dict[str, float] = {}
    for item in raw_items:
        if "=" not in item:
            raise ValueError(f"Invalid kind value item: {item}")
        kind, raw_value = item.split("=", 1)
        values[kind.strip()] = float(raw_value.strip())
    return values


def _parse_kind_value_grid(raw_items: list[str] | None) -> dict[str, list[float]]:
    if not raw_items:
        return {}
    grid: dict[str, list[float]] = {}
    for item in raw_items:
        if "=" not in item:
            raise ValueError(f"Invalid kind grid item: {item}")
        kind, raw_values = item.split("=", 1)
        values = [float(value) for value in raw_values.split(",") if value.strip()]
        if not kind or not values:
            raise ValueError(f"Invalid kind grid item: {item}")
        grid[kind.strip()] = values
    return grid


def _iter_kind_value_combos(grid: dict[str, list[float]]):
    if not grid:
        yield {}
        return
    kinds = sorted(grid)
    for values in product(*(grid[kind] for kind in kinds)):
        yield {kind: float(value) for kind, value in zip(kinds, values)}


def _macro_f1(report: dict[str, Any]) -> float:
    metrics = report.get("object_metrics", {})
    values = [float(item.get("f1", 0.0)) for item in metrics.values()]
    return sum(values) / max(len(values), 1)


def _report_tag(report: dict[str, Any]) -> str:
    post = report.get("postprocess", {})
    parts = [f"thr_{int(round(post.get('default_score_threshold', 0.0) * 1000)):03d}"]
    parts.append(f"nms_{int(round(post.get('default_nms_iou', 0.0) * 1000)):03d}")
    for kind, value in sorted((post.get("nms_iou_by_kind") or {}).items()):
        parts.append(f"{kind}_{int(round(value * 1000)):03d}")
    return "_".join(parts)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    raise SystemExit(main())
