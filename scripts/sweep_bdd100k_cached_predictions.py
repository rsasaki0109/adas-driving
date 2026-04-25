#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from itertools import product
import json
from pathlib import Path
import sys
from typing import Any

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adas_perception.types import Box
from scripts.compare_evaluations import main as compare_main
from scripts.evaluate_bdd100k import (
    DEFAULT_KINDS,
    DEFAULT_SIZE_BUCKET_AREAS,
    _build_report,
    _empty_counter,
    _empty_size_totals,
    _ground_truth_by_kind,
    _load_label_frames,
    _match_boxes,
    _update_size_totals,
    _validate_size_bucket_areas,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep BDD100K thresholds from saved prediction JSON.")
    parser.add_argument("--images-root", required=True, help="Directory containing BDD100K images.")
    parser.add_argument("--labels", required=True, help="BDD100K/Scalabel label JSON path.")
    parser.add_argument("--predictions", required=True, help="Prediction JSON saved by evaluate_bdd100k.py.")
    parser.add_argument("--output-dir", required=True, help="Directory for cached sweep reports.")
    parser.add_argument("--score-thresholds", nargs="+", type=float, default=[0.15])
    parser.add_argument(
        "--kind-score-thresholds",
        nargs="+",
        default=None,
        metavar="KIND=THRESHOLDS",
        help="Per-kind grid, e.g. pedestrian=0.20,0.25 traffic_sign=0.25,0.30.",
    )
    parser.add_argument(
        "--tiny-kind-score-thresholds",
        nargs="+",
        default=None,
        metavar="KIND=THRESHOLDS",
        help=(
            "Optional per-kind threshold grid applied only to bboxes falling in the "
            "tiny size bucket (defined by --size-bucket-areas). Non-tiny detections "
            "use the regular --kind-score-thresholds / --score-thresholds threshold. "
            "Example: pedestrian=0.20,0.25 traffic_light=0.15,0.20."
        ),
    )
    parser.add_argument(
        "--small-kind-score-thresholds",
        nargs="+",
        default=None,
        metavar="KIND=THRESHOLDS",
        help=(
            "Optional per-kind threshold grid applied only to bboxes in the small "
            "size bucket. Non-small detections use the regular kind threshold. "
            "Same format as --tiny-kind-score-thresholds."
        ),
    )
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--frame-offset", type=int, default=0)
    parser.add_argument("--iou-threshold", type=float, default=0.50)
    parser.add_argument("--kinds", nargs="+", default=DEFAULT_KINDS)
    parser.add_argument("--group-by-size", action="store_true")
    parser.add_argument("--size-bucket-areas", nargs=3, type=float, default=DEFAULT_SIZE_BUCKET_AREAS)
    parser.add_argument(
        "--runtime-report",
        default=None,
        help="Optional source evaluation report. Its average_inference_ms is copied into cached reports.",
    )
    parser.add_argument("--markdown", action="store_true", help="Write comparison.md/csv.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    report_dir = output_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    selected_kinds = set(args.kinds)
    size_bucket_areas = _validate_size_bucket_areas(args.size_bucket_areas)
    frames, frame_selection = _load_label_frames(
        Path(args.labels),
        max_images=args.max_images,
        frame_stride=args.frame_stride,
        frame_offset=args.frame_offset,
    )
    predictions_by_name = _load_predictions(Path(args.predictions), selected_kinds)
    tiny_kind_threshold_grid = _parse_kind_threshold_grid(args.tiny_kind_score_thresholds)
    small_kind_threshold_grid = _parse_kind_threshold_grid(args.small_kind_score_thresholds)
    needs_image_area = (
        args.group_by_size
        or bool(tiny_kind_threshold_grid)
        or bool(small_kind_threshold_grid)
    )
    frame_records = _build_frame_records(
        frames=frames,
        images_root=Path(args.images_root),
        selected_kinds=selected_kinds,
        predictions_by_name=predictions_by_name,
        needs_image_area=needs_image_area,
    )
    average_inference_ms = _average_inference_ms(args.runtime_report)

    generated_reports: list[str] = []
    generated_names: list[str] = []
    kind_threshold_grid = _parse_kind_threshold_grid(args.kind_score_thresholds)
    for threshold in args.score_thresholds:
        for kind_thresholds in _iter_kind_thresholds(kind_threshold_grid):
            for tiny_kind_thresholds in _iter_kind_thresholds(tiny_kind_threshold_grid):
                for small_kind_thresholds in _iter_kind_thresholds(small_kind_threshold_grid):
                    tag = (
                        f"score_{_threshold_tag(threshold)}"
                        f"{_kind_threshold_tag(kind_thresholds)}"
                        f"{_tiny_kind_threshold_tag(tiny_kind_thresholds)}"
                        f"{_small_kind_threshold_tag(small_kind_thresholds)}"
                    )
                    report = _evaluate_thresholds(
                        frame_records=frame_records,
                        labels_path=str(args.labels),
                        images_root=str(args.images_root),
                        predictions_path=str(args.predictions),
                        selected_kinds=selected_kinds,
                        default_threshold=float(threshold),
                        kind_thresholds=kind_thresholds,
                        tiny_kind_thresholds=tiny_kind_thresholds,
                        small_kind_thresholds=small_kind_thresholds,
                        iou_threshold=float(args.iou_threshold),
                        frame_selection=frame_selection,
                        group_by_size=bool(args.group_by_size),
                        size_bucket_areas=size_bucket_areas,
                        average_inference_ms=average_inference_ms,
                    )
                    report_path = report_dir / f"{tag}.json"
                    _write_json(report_path, report)
                    generated_reports.append(str(report_path))
                    generated_names.append(tag)
                    macro_f1 = _macro_f1(report)
                    print(f"{tag}: macro_f1={macro_f1:.4f}", flush=True)

    comparison_path = output_dir / "comparison.json"
    argv = [
        "compare_evaluations.py",
        "--reports",
        *generated_reports,
        "--names",
        *generated_names,
        "--output",
        str(comparison_path),
    ]
    if args.markdown:
        argv.extend(
            [
                "--markdown-output",
                str(output_dir / "comparison.md"),
                "--csv-output",
                str(output_dir / "comparison.csv"),
            ]
        )
    old_argv = sys.argv
    try:
        sys.argv = argv
        compare_main()
    finally:
        sys.argv = old_argv
    _write_json(
        output_dir / "manifest.json",
        {
            "schema_version": "0.1",
            "labels": args.labels,
            "images_root": args.images_root,
            "predictions": args.predictions,
            "reports": generated_reports,
            "comparison": str(comparison_path),
            "score_thresholds": args.score_thresholds,
            "kind_score_thresholds": args.kind_score_thresholds,
            "tiny_kind_score_thresholds": args.tiny_kind_score_thresholds,
            "small_kind_score_thresholds": args.small_kind_score_thresholds,
        },
    )
    return 0


def _build_frame_records(
    *,
    frames: list[dict[str, Any]],
    images_root: Path,
    selected_kinds: set[str],
    predictions_by_name: dict[str, list[dict[str, Any]]],
    needs_image_area: bool,
) -> list[dict[str, Any]]:
    records = []
    for frame in frames:
        name = str(frame.get("name", ""))
        gt_by_kind, _ = _ground_truth_by_kind(frame, selected_kinds)
        image_area = 1
        if needs_image_area:
            image = cv2.imread(str(images_root / name))
            if image is None:
                continue
            image_area = int(image.shape[0] * image.shape[1])
        records.append(
            {
                "name": name,
                "gt_by_kind": gt_by_kind,
                "predictions": predictions_by_name.get(name, []),
                "image_area": image_area,
            }
        )
    return records


def _evaluate_thresholds(
    *,
    frame_records: list[dict[str, Any]],
    labels_path: str,
    images_root: str,
    predictions_path: str,
    selected_kinds: set[str],
    default_threshold: float,
    kind_thresholds: dict[str, float],
    tiny_kind_thresholds: dict[str, float],
    small_kind_thresholds: dict[str, float],
    iou_threshold: float,
    frame_selection: dict[str, int | None],
    group_by_size: bool,
    size_bucket_areas: list[float],
    average_inference_ms: float,
) -> dict[str, Any]:
    totals = {kind: _empty_counter() for kind in selected_kinds}
    size_totals = _empty_size_totals(selected_kinds) if group_by_size else {}
    tiny_area_ratio = float(size_bucket_areas[0])
    small_area_ratio = float(size_bucket_areas[1])
    for record in frame_records:
        pred_by_kind = _filter_predictions(
            record["predictions"],
            selected_kinds=selected_kinds,
            default_threshold=default_threshold,
            kind_thresholds=kind_thresholds,
            tiny_kind_thresholds=tiny_kind_thresholds,
            small_kind_thresholds=small_kind_thresholds,
            image_area=int(record["image_area"]),
            tiny_area_ratio=tiny_area_ratio,
            small_area_ratio=small_area_ratio,
        )
        for kind in selected_kinds:
            matched = _match_boxes(
                record["gt_by_kind"].get(kind, []),
                pred_by_kind.get(kind, []),
                iou_threshold,
            )
            totals[kind]["tp"] += matched["tp"]
            totals[kind]["fp"] += matched["fp"]
            totals[kind]["fn"] += matched["fn"]
            if group_by_size:
                _update_size_totals(
                    kind_totals=size_totals[kind],
                    matched=matched,
                    ground_truth=record["gt_by_kind"].get(kind, []),
                    predictions=pred_by_kind.get(kind, []),
                    image_area=int(record["image_area"]),
                    thresholds=size_bucket_areas,
                )
    report = _build_report(
        labels_path=labels_path,
        images_root=images_root,
        config_path=f"cached:{predictions_path}",
        kinds=selected_kinds,
        totals=totals,
        state_totals={"matched": 0, "correct": 0},
        lane_presence=Counter({"tp": 0, "fp": 0, "fn": 0}),
        processed=len(frame_records),
        missing_images=[],
        total_inference_ms=average_inference_ms * len(frame_records),
        iou_threshold=iou_threshold,
        frame_selection=frame_selection,
        group_by=[],
        grouped_totals={},
        size_bucket_areas=size_bucket_areas,
        size_totals=size_totals,
    )
    report["cached_thresholds"] = {
        "default": round(float(default_threshold), 4),
        "by_kind": {kind: round(float(value), 4) for kind, value in sorted(kind_thresholds.items())},
        "tiny_by_kind": {kind: round(float(value), 4) for kind, value in sorted(tiny_kind_thresholds.items())},
        "small_by_kind": {kind: round(float(value), 4) for kind, value in sorted(small_kind_thresholds.items())},
    }
    return report


def _filter_predictions(
    predictions: list[dict[str, Any]],
    *,
    selected_kinds: set[str],
    default_threshold: float,
    kind_thresholds: dict[str, float],
    tiny_kind_thresholds: dict[str, float] | None = None,
    small_kind_thresholds: dict[str, float] | None = None,
    image_area: int = 1,
    tiny_area_ratio: float = 0.0,
    small_area_ratio: float = 0.0,
) -> dict[str, list[dict[str, Any]]]:
    by_kind: dict[str, list[dict[str, Any]]] = defaultdict(list)
    tiny_kind_thresholds = tiny_kind_thresholds or {}
    small_kind_thresholds = small_kind_thresholds or {}
    for prediction in predictions:
        kind = str(prediction.get("kind", ""))
        if kind not in selected_kinds:
            continue
        confidence = float(prediction.get("confidence", 0.0))
        threshold = float(kind_thresholds.get(kind, default_threshold))
        if (tiny_kind_thresholds or small_kind_thresholds) and image_area > 0:
            box = prediction["box"]
            area_ratio = max(box.area, 0) / float(image_area)
            if tiny_area_ratio > 0.0 and area_ratio < tiny_area_ratio:
                tiny_threshold = tiny_kind_thresholds.get(kind)
                if tiny_threshold is not None:
                    threshold = float(tiny_threshold)
            elif small_area_ratio > 0.0 and area_ratio < small_area_ratio:
                small_threshold = small_kind_thresholds.get(kind)
                if small_threshold is not None:
                    threshold = float(small_threshold)
        if confidence < threshold:
            continue
        by_kind[kind].append(
            {
                "box": prediction["box"],
                "label": str(prediction.get("label", "")),
                "state": None,
                "confidence": confidence,
            }
        )
    return by_kind


def _load_predictions(path: Path, selected_kinds: set[str]) -> dict[str, list[dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    by_name = {}
    for item in payload.get("predictions", []):
        detections = []
        for detection in item.get("detections", []):
            kind = str(detection.get("kind", ""))
            if kind not in selected_kinds:
                continue
            box = detection.get("box", {})
            detections.append(
                {
                    "kind": kind,
                    "label": str(detection.get("label", "")),
                    "confidence": float(detection.get("confidence", 0.0)),
                    "box": Box(
                        x1=int(box.get("x1", 0)),
                        y1=int(box.get("y1", 0)),
                        x2=int(box.get("x2", 0)),
                        y2=int(box.get("y2", 0)),
                    ),
                }
            )
        by_name[str(item.get("name", ""))] = detections
    return by_name


def _parse_kind_threshold_grid(raw_items: list[str] | None) -> dict[str, list[float]]:
    if not raw_items:
        return {}
    grid: dict[str, list[float]] = {}
    for item in raw_items:
        if "=" not in item:
            raise ValueError(f"Invalid --kind-score-thresholds item: {item}")
        kind, raw_values = item.split("=", 1)
        values = [float(value) for value in raw_values.split(",") if value.strip()]
        if not kind or not values:
            raise ValueError(f"Invalid --kind-score-thresholds item: {item}")
        grid[kind.strip()] = values
    return grid


def _iter_kind_thresholds(grid: dict[str, list[float]]):
    if not grid:
        yield {}
        return
    kinds = sorted(grid)
    for values in product(*(grid[kind] for kind in kinds)):
        yield {kind: float(value) for kind, value in zip(kinds, values)}


def _average_inference_ms(report_path: str | None) -> float:
    if not report_path:
        return 0.0
    with Path(report_path).open("r", encoding="utf-8") as f:
        report = json.load(f)
    return float(report.get("runtime", {}).get("average_inference_ms", 0.0))


def _macro_f1(report: dict[str, Any]) -> float:
    metrics = report.get("object_metrics", {})
    values = [float(item.get("f1", 0.0)) for item in metrics.values()]
    return sum(values) / max(len(values), 1)


def _threshold_tag(threshold: float) -> str:
    return f"{int(round(threshold * 1000)):03d}"


def _kind_threshold_tag(kind_thresholds: dict[str, float]) -> str:
    if not kind_thresholds:
        return ""
    return "_kind_" + "_".join(
        f"{kind}_{_threshold_tag(threshold)}" for kind, threshold in sorted(kind_thresholds.items())
    )


def _tiny_kind_threshold_tag(kind_thresholds: dict[str, float]) -> str:
    if not kind_thresholds:
        return ""
    return "_tiny_" + "_".join(
        f"{kind}_{_threshold_tag(threshold)}" for kind, threshold in sorted(kind_thresholds.items())
    )


def _small_kind_threshold_tag(kind_thresholds: dict[str, float]) -> str:
    if not kind_thresholds:
        return ""
    return "_small_" + "_".join(
        f"{kind}_{_threshold_tag(threshold)}" for kind, threshold in sorted(kind_thresholds.items())
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


if __name__ == "__main__":
    raise SystemExit(main())
