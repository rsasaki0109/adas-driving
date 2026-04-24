#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
import time
from typing import Any

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adas_perception.config import apply_runtime_overrides, load_config
from adas_perception.pipeline import ADASPerceptionPipeline
from adas_perception.types import Box, Detection


BDD_CATEGORY_TO_KIND = {
    "bike": "vehicle",
    "bicycle": "vehicle",
    "bus": "vehicle",
    "car": "vehicle",
    "motor": "vehicle",
    "motorcycle": "vehicle",
    "train": "vehicle",
    "truck": "vehicle",
    "person": "pedestrian",
    "pedestrian": "pedestrian",
    "rider": "pedestrian",
    "traffic light": "traffic_light",
    "traffic sign": "traffic_sign",
}

DEFAULT_KINDS = ["vehicle", "pedestrian", "traffic_sign", "traffic_light"]
DEFAULT_SIZE_BUCKET_AREAS = [0.0005, 0.0025, 0.01]
SIZE_BUCKET_NAMES = ["tiny", "small", "medium", "large"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate adas-perception on BDD100K-style labels.")
    parser.add_argument("--images-root", required=True, help="Directory containing BDD100K images.")
    parser.add_argument("--labels", required=True, help="BDD100K/Scalabel label JSON path.")
    parser.add_argument("--config", default="configs/default.yaml", help="YAML config path.")
    parser.add_argument("--device", default=None, help="Object detector device: auto, cpu, cuda, cuda:0.")
    parser.add_argument("--no-objects", action="store_true", help="Disable TorchVision object detection.")
    parser.add_argument("--max-images", type=int, default=None, help="Evaluate only the first N labeled images.")
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Evaluate every Nth frame after sorting by label order. Use with --frame-offset for split evaluation.",
    )
    parser.add_argument(
        "--frame-offset",
        type=int,
        default=0,
        help="Start offset used with --frame-stride. Example: offset 0/stride 2 and offset 1/stride 2 make two splits.",
    )
    parser.add_argument("--iou-threshold", type=float, default=0.50, help="IoU threshold for TP matching.")
    parser.add_argument("--min-confidence", type=float, default=0.0, help="Drop predictions below this score.")
    parser.add_argument("--kinds", nargs="+", default=DEFAULT_KINDS, help="Kinds to evaluate.")
    parser.add_argument("--output", default=None, help="Optional JSON report path.")
    parser.add_argument("--save-predictions", default=None, help="Optional per-image prediction JSON path.")
    parser.add_argument("--save-errors", default=None, help="Optional JSON file with sampled TP/FP/FN examples.")
    parser.add_argument(
        "--max-error-samples",
        type=int,
        default=50,
        help="Maximum TP/FP/FN examples per kind. Use 0 to save all examples.",
    )
    parser.add_argument(
        "--group-by",
        nargs="*",
        default=[],
        choices=["weather", "timeofday", "scene"],
        help="Also report metrics grouped by BDD100K frame attributes.",
    )
    parser.add_argument(
        "--group-by-size",
        action="store_true",
        help="Also report TP/FP/FN metrics by normalized bbox area bucket.",
    )
    parser.add_argument(
        "--size-bucket-areas",
        nargs=3,
        type=float,
        default=DEFAULT_SIZE_BUCKET_AREAS,
        metavar=("TINY_MAX", "SMALL_MAX", "MEDIUM_MAX"),
        help=(
            "Normalized bbox area thresholds for --group-by-size. "
            "Defaults to 0.0005 0.0025 0.01."
        ),
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=0,
        help="Print progress every N processed images. Disabled by default.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    images_root = Path(args.images_root)
    frames, frame_selection = _load_label_frames(
        Path(args.labels),
        max_images=args.max_images,
        frame_stride=args.frame_stride,
        frame_offset=args.frame_offset,
    )
    config = apply_runtime_overrides(
        load_config(args.config),
        device=args.device,
        disable_objects=args.no_objects,
    )
    pipeline = ADASPerceptionPipeline(config)

    selected_kinds = set(args.kinds)
    size_bucket_areas = _validate_size_bucket_areas(args.size_bucket_areas)
    totals = {kind: _empty_counter() for kind in selected_kinds}
    size_totals = _empty_size_totals(selected_kinds) if args.group_by_size else {}
    state_totals = {"matched": 0, "correct": 0}
    lane_presence = _empty_counter()
    predictions_dump = []
    error_dump = _empty_error_dump(selected_kinds, args.max_error_samples)
    group_by = list(dict.fromkeys(args.group_by))
    grouped_totals: dict[str, dict[str, dict[str, Any]]] = {attribute: {} for attribute in group_by}
    processed = 0
    missing_images = []
    total_inference_ms = 0.0

    for frame in frames:
        image_path = images_root / str(frame.get("name", ""))
        if not image_path.exists():
            missing_images.append(str(image_path))
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            missing_images.append(str(image_path))
            continue
        image_area = int(image.shape[0] * image.shape[1])

        gt_by_kind, gt_lanes_present = _ground_truth_by_kind(frame, selected_kinds)
        group_buckets = [
            _get_group_bucket(grouped_totals[attribute], _frame_attribute(frame, attribute), selected_kinds)
            for attribute in group_by
        ]

        start = time.perf_counter()
        result = pipeline.run(image)
        inference_ms = (time.perf_counter() - start) * 1000.0
        total_inference_ms += inference_ms
        for bucket in group_buckets:
            bucket["processed_images"] += 1
            bucket["total_inference_ms"] += inference_ms

        pred_by_kind = _predictions_by_kind(result.detections, selected_kinds, args.min_confidence)
        for kind in selected_kinds:
            matched = _match_boxes(
                gt_by_kind.get(kind, []),
                pred_by_kind.get(kind, []),
                args.iou_threshold,
            )
            totals[kind]["tp"] += matched["tp"]
            totals[kind]["fp"] += matched["fp"]
            totals[kind]["fn"] += matched["fn"]
            if args.group_by_size:
                _update_size_totals(
                    kind_totals=size_totals[kind],
                    matched=matched,
                    ground_truth=gt_by_kind.get(kind, []),
                    predictions=pred_by_kind.get(kind, []),
                    image_area=image_area,
                    thresholds=size_bucket_areas,
                )
            for bucket in group_buckets:
                bucket["object_totals"][kind]["tp"] += matched["tp"]
                bucket["object_totals"][kind]["fp"] += matched["fp"]
                bucket["object_totals"][kind]["fn"] += matched["fn"]
            if kind == "traffic_light":
                state_totals["matched"] += matched["state_matched"]
                state_totals["correct"] += matched["state_correct"]
                for bucket in group_buckets:
                    bucket["traffic_light_state"]["matched"] += matched["state_matched"]
                    bucket["traffic_light_state"]["correct"] += matched["state_correct"]
            if args.save_errors:
                _collect_error_samples(
                    store=error_dump,
                    kind=kind,
                    frame_name=str(frame.get("name")),
                    matched=matched,
                    ground_truth=gt_by_kind.get(kind, []),
                    predictions=pred_by_kind.get(kind, []),
                )

        pred_lanes_present = bool(result.lanes.lines)
        if gt_lanes_present is not None:
            _update_presence_counter(lane_presence, gt_lanes_present, pred_lanes_present)
            for bucket in group_buckets:
                _update_presence_counter(bucket["lane_presence"], gt_lanes_present, pred_lanes_present)

        processed += 1
        if args.progress_every and processed % args.progress_every == 0:
            print(f"Processed {processed}/{len(frames)} images", flush=True)
        if args.save_predictions:
            predictions_dump.append(
                {
                    "name": frame.get("name"),
                    "detections": [
                        {
                            "kind": detection.kind,
                            "label": detection.label,
                            "confidence": round(float(detection.confidence), 4),
                            "box": {
                                "x1": detection.box.x1,
                                "y1": detection.box.y1,
                                "x2": detection.box.x2,
                                "y2": detection.box.y2,
                            },
                        }
                        for detection in result.detections
                    ],
                    "lanes_detected": len(result.lanes.lines),
                }
            )

    report = _build_report(
        labels_path=str(args.labels),
        images_root=str(images_root),
        config_path=str(args.config),
        kinds=selected_kinds,
        totals=totals,
        state_totals=state_totals,
        lane_presence=lane_presence,
        processed=processed,
        missing_images=missing_images,
        total_inference_ms=total_inference_ms,
        iou_threshold=args.iou_threshold,
        frame_selection=frame_selection,
        group_by=group_by,
        grouped_totals=grouped_totals,
        size_bucket_areas=size_bucket_areas,
        size_totals=size_totals,
    )
    print_report(report)

    if args.output:
        _write_json(Path(args.output), report)
        print(f"Saved {args.output}")
    if args.save_predictions:
        _write_json(Path(args.save_predictions), {"schema_version": "0.1", "predictions": predictions_dump})
        print(f"Saved predictions {args.save_predictions}")
    if args.save_errors:
        _write_json(Path(args.save_errors), {"schema_version": "0.1", "errors": error_dump})
        print(f"Saved errors {args.save_errors}")
    return 0


def print_report(report: dict[str, Any]) -> None:
    print("BDD100K evaluation summary")
    print(f"- labels: {report['labels']}")
    print(f"- images_root: {report['images_root']}")
    print(f"- processed_images: {report['processed_images']}")
    print(f"- missing_images: {report['missing_images_count']}")
    selection = report.get("frame_selection", {})
    if selection:
        print(
            "- frame_selection: "
            f"offset={selection.get('frame_offset')}, "
            f"stride={selection.get('frame_stride')}, "
            f"selected={selection.get('selected_frames')}/"
            f"{selection.get('total_frames')}"
        )
    print(f"- average_inference_ms: {report['runtime']['average_inference_ms']:.3f}")
    print(f"- fps: {report['runtime']['fps']:.3f}")
    print("- object_metrics:")
    for kind, metrics in report["object_metrics"].items():
        print(
            f"  - {kind}: "
            f"precision={metrics['precision']:.3f}, "
            f"recall={metrics['recall']:.3f}, "
            f"f1={metrics['f1']:.3f}, "
            f"tp={metrics['tp']}, fp={metrics['fp']}, fn={metrics['fn']}"
        )
    size_metrics = report.get("object_size_metrics", {})
    if size_metrics:
        thresholds = size_metrics.get("area_ratio_thresholds", [])
        print(f"- object_size_metrics: area_ratio_thresholds={thresholds}")
        for kind, buckets in size_metrics.get("metrics", {}).items():
            print(f"  - {kind}:")
            for bucket_name, metrics in buckets.items():
                print(
                    f"    - {bucket_name}: "
                    f"precision={metrics['precision']:.3f}, "
                    f"recall={metrics['recall']:.3f}, "
                    f"f1={metrics['f1']:.3f}, "
                    f"tp={metrics['tp']}, fp={metrics['fp']}, fn={metrics['fn']}"
                )
    lane = report["lane_presence"]
    if lane["evaluated_frames"] > 0:
        print(
            "- lane_presence: "
            f"precision={lane['precision']:.3f}, recall={lane['recall']:.3f}, "
            f"f1={lane['f1']:.3f}, evaluated_frames={lane['evaluated_frames']}"
        )
    light_state = report["traffic_light_state"]
    if light_state["matched"] > 0:
        print(f"- traffic_light_state_accuracy: {light_state['accuracy']:.3f}")
    grouped = report.get("grouped_metrics", {})
    if grouped:
        print("- grouped_metrics:")
        for attribute, buckets in grouped.items():
            print(f"  - {attribute}:")
            for value, metrics in buckets.items():
                macro_f1 = _macro_metric(metrics["object_metrics"], "f1")
                kind_f1 = ", ".join(
                    f"{kind}={kind_metrics['f1']:.3f}"
                    for kind, kind_metrics in metrics["object_metrics"].items()
                )
                print(
                    f"    - {value}: images={metrics['processed_images']}, "
                    f"macro_f1={macro_f1:.3f}, {kind_f1}"
                )


def _load_label_frames(
    labels_path: Path,
    max_images: int | None,
    frame_stride: int,
    frame_offset: int,
) -> tuple[list[dict[str, Any]], dict[str, int | None]]:
    if frame_stride < 1:
        raise ValueError("--frame-stride must be >= 1.")
    if frame_offset < 0:
        raise ValueError("--frame-offset must be >= 0.")
    if frame_offset >= frame_stride:
        raise ValueError("--frame-offset must be smaller than --frame-stride.")

    with labels_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict) and "frames" in payload:
        frames = list(payload["frames"])
    elif isinstance(payload, list):
        frames = list(payload)
    else:
        raise ValueError("Unsupported BDD100K label JSON: expected a list or an object with 'frames'.")
    total_frames = len(frames)
    selected = frames[frame_offset::frame_stride]
    before_max = len(selected)
    if max_images is not None:
        selected = selected[:max_images]
    return selected, {
        "total_frames": total_frames,
        "frame_stride": frame_stride,
        "frame_offset": frame_offset,
        "selected_before_max_images": before_max,
        "max_images": max_images,
        "selected_frames": len(selected),
    }


def _ground_truth_by_kind(
    frame: dict[str, Any],
    selected_kinds: set[str],
) -> tuple[dict[str, list[dict[str, Any]]], bool | None]:
    by_kind: dict[str, list[dict[str, Any]]] = defaultdict(list)
    lane_seen = False
    lane_available = False
    for label in frame.get("labels", []):
        category = str(label.get("category", "")).lower()
        if category == "lane":
            lane_available = True
            if label.get("poly2d"):
                lane_seen = True
            continue
        kind = BDD_CATEGORY_TO_KIND.get(category)
        if kind not in selected_kinds or "box2d" not in label:
            continue
        box = _box_from_bdd(label["box2d"])
        if box.area <= 0:
            continue
        by_kind[kind].append(
            {
                "box": box,
                "label": category,
                "state": _traffic_light_state(label),
            }
        )
    return by_kind, lane_seen if lane_available else None


def _predictions_by_kind(
    detections: list[Detection],
    selected_kinds: set[str],
    min_confidence: float,
) -> dict[str, list[dict[str, Any]]]:
    by_kind: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for detection in detections:
        if detection.kind not in selected_kinds or detection.confidence < min_confidence:
            continue
        by_kind[detection.kind].append(
            {
                "box": detection.box,
                "label": detection.label,
                "state": _predicted_traffic_light_state(detection),
                "confidence": detection.confidence,
            }
        )
    return by_kind


def _match_boxes(
    ground_truth: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    iou_threshold: float,
) -> dict[str, Any]:
    candidates = []
    for gt_index, gt in enumerate(ground_truth):
        for pred_index, pred in enumerate(predictions):
            overlap = _iou(gt["box"], pred["box"])
            if overlap >= iou_threshold:
                candidates.append((overlap, gt_index, pred_index))

    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    matched_pairs: list[dict[str, int | float]] = []
    state_matched = 0
    state_correct = 0
    for overlap, gt_index, pred_index in sorted(candidates, reverse=True):
        if gt_index in matched_gt or pred_index in matched_pred:
            continue
        matched_gt.add(gt_index)
        matched_pred.add(pred_index)
        matched_pairs.append(
            {
                "gt_index": int(gt_index),
                "pred_index": int(pred_index),
                "iou": round(float(overlap), 4),
            }
        )
        gt_state = ground_truth[gt_index].get("state")
        pred_state = predictions[pred_index].get("state")
        if gt_state and pred_state:
            state_matched += 1
            if gt_state == pred_state:
                state_correct += 1

    tp = len(matched_gt)
    return {
        "tp": tp,
        "fp": len(predictions) - tp,
        "fn": len(ground_truth) - tp,
        "state_matched": state_matched,
        "state_correct": state_correct,
        "matched_gt_indices": sorted(matched_gt),
        "matched_pred_indices": sorted(matched_pred),
        "matched_pairs": matched_pairs,
    }


def _empty_error_dump(kinds: set[str], max_samples: int) -> dict[str, Any]:
    return {
        kind: {
            "max_samples": max_samples,
            "tp": [],
            "fp": [],
            "fn": [],
        }
        for kind in sorted(kinds)
    }


def _collect_error_samples(
    *,
    store: dict[str, Any],
    kind: str,
    frame_name: str,
    matched: dict[str, Any],
    ground_truth: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> None:
    kind_store = store[kind]
    max_samples = int(kind_store["max_samples"])
    matched_gt = set(matched.get("matched_gt_indices", []))
    matched_pred = set(matched.get("matched_pred_indices", []))

    for gt_index in matched_gt:
        if not _can_store_more(kind_store["tp"], max_samples):
            break
        gt = ground_truth[gt_index]
        pair = _matched_pair(gt_index, matched)
        pred = predictions[int(pair["pred_index"])]
        kind_store["tp"].append(
            {
                "frame": frame_name,
                "gt_label": gt.get("label"),
                "gt_box": _box_to_dict(gt["box"]),
                "pred_label": pred.get("label"),
                "confidence": round(float(pred.get("confidence", 0.0)), 4),
                "pred_box": _box_to_dict(pred["box"]),
                "iou": round(float(pair.get("iou", 0.0)), 4),
            }
        )

    for pred_index, pred in enumerate(predictions):
        if pred_index in matched_pred:
            continue
        if not _can_store_more(kind_store["fp"], max_samples):
            break
        kind_store["fp"].append(
            {
                "frame": frame_name,
                "pred_label": pred.get("label"),
                "confidence": round(float(pred.get("confidence", 0.0)), 4),
                "pred_box": _box_to_dict(pred["box"]),
            }
        )

    for gt_index, gt in enumerate(ground_truth):
        if gt_index in matched_gt:
            continue
        if not _can_store_more(kind_store["fn"], max_samples):
            break
        kind_store["fn"].append(
            {
                "frame": frame_name,
                "gt_label": gt.get("label"),
                "gt_box": _box_to_dict(gt["box"]),
            }
        )


def _can_store_more(items: list[Any], max_samples: int) -> bool:
    return max_samples <= 0 or len(items) < max_samples


def _matched_pair(gt_index: int, matched: dict[str, Any]) -> dict[str, Any]:
    for pair in matched.get("matched_pairs", []):
        if int(pair.get("gt_index", -1)) == gt_index:
            return pair
    raise ValueError(f"Matched prediction not found for gt index {gt_index}")


def _box_to_dict(box: Box) -> dict[str, int]:
    return {
        "x1": box.x1,
        "y1": box.y1,
        "x2": box.x2,
        "y2": box.y2,
    }


def _build_report(
    *,
    labels_path: str,
    images_root: str,
    config_path: str,
    kinds: set[str],
    totals: dict[str, Counter[str]],
    state_totals: dict[str, int],
    lane_presence: Counter[str],
    processed: int,
    missing_images: list[str],
    total_inference_ms: float,
    iou_threshold: float,
    frame_selection: dict[str, int | None],
    group_by: list[str],
    grouped_totals: dict[str, dict[str, dict[str, Any]]],
    size_bucket_areas: list[float],
    size_totals: dict[str, dict[str, Counter[str]]],
) -> dict[str, Any]:
    object_metrics = {
        kind: _metrics_from_counter(totals[kind])
        for kind in sorted(kinds)
    }
    lane_metrics = _metrics_from_counter(lane_presence)
    lane_metrics["evaluated_frames"] = int(sum(lane_presence.values()))
    report = {
        "schema_version": "0.1",
        "dataset": "bdd100k",
        "labels": labels_path,
        "images_root": images_root,
        "config": config_path,
        "iou_threshold": iou_threshold,
        "frame_selection": frame_selection,
        "group_by": group_by,
        "processed_images": processed,
        "missing_images_count": len(missing_images),
        "missing_images_sample": missing_images[:10],
        "runtime": _runtime_metrics(total_inference_ms, processed),
        "object_metrics": object_metrics,
        "lane_presence": lane_metrics,
        "traffic_light_state": {
            "matched": state_totals["matched"],
            "correct": state_totals["correct"],
            "accuracy": round(state_totals["correct"] / max(state_totals["matched"], 1), 4)
            if state_totals["matched"]
            else 0.0,
        },
        "grouped_metrics": _build_grouped_metrics(grouped_totals, kinds),
    }
    if size_totals:
        report["object_size_metrics"] = _build_size_metrics(size_totals, kinds, size_bucket_areas)
    return report


def _build_grouped_metrics(
    grouped_totals: dict[str, dict[str, dict[str, Any]]],
    kinds: set[str],
) -> dict[str, dict[str, Any]]:
    grouped_metrics = {}
    for attribute, buckets in grouped_totals.items():
        grouped_metrics[attribute] = {}
        for value, bucket in sorted(buckets.items()):
            lane_metrics = _metrics_from_counter(bucket["lane_presence"])
            lane_metrics["evaluated_frames"] = int(sum(bucket["lane_presence"].values()))
            state_totals = bucket["traffic_light_state"]
            grouped_metrics[attribute][value] = {
                "processed_images": int(bucket["processed_images"]),
                "runtime": _runtime_metrics(bucket["total_inference_ms"], int(bucket["processed_images"])),
                "object_metrics": {
                    kind: _metrics_from_counter(bucket["object_totals"][kind])
                    for kind in sorted(kinds)
                },
                "lane_presence": lane_metrics,
                "traffic_light_state": {
                    "matched": state_totals["matched"],
                    "correct": state_totals["correct"],
                    "accuracy": round(state_totals["correct"] / max(state_totals["matched"], 1), 4)
                    if state_totals["matched"]
                    else 0.0,
                },
            }
    return grouped_metrics


def _runtime_metrics(total_inference_ms: float, processed: int) -> dict[str, float]:
    return {
        "total_inference_ms": round(total_inference_ms, 4),
        "average_inference_ms": round(total_inference_ms / max(processed, 1), 4),
        "fps": round(processed / max(total_inference_ms / 1000.0, 1e-9), 4),
    }


def _validate_size_bucket_areas(thresholds: list[float]) -> list[float]:
    if len(thresholds) != 3:
        raise ValueError("--size-bucket-areas requires exactly three thresholds.")
    values = [float(value) for value in thresholds]
    if any(value <= 0.0 or value >= 1.0 for value in values):
        raise ValueError("--size-bucket-areas values must be between 0 and 1.")
    if values != sorted(values):
        raise ValueError("--size-bucket-areas values must be sorted ascending.")
    return values


def _empty_size_totals(kinds: set[str]) -> dict[str, dict[str, Counter[str]]]:
    return {
        kind: {bucket: _empty_counter() for bucket in SIZE_BUCKET_NAMES}
        for kind in kinds
    }


def _update_size_totals(
    *,
    kind_totals: dict[str, Counter[str]],
    matched: dict[str, Any],
    ground_truth: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    image_area: int,
    thresholds: list[float],
) -> None:
    matched_gt = set(int(index) for index in matched.get("matched_gt_indices", []))
    matched_pred = set(int(index) for index in matched.get("matched_pred_indices", []))

    for pair in matched.get("matched_pairs", []):
        gt_index = int(pair["gt_index"])
        bucket = _size_bucket(ground_truth[gt_index]["box"], image_area, thresholds)
        kind_totals[bucket]["tp"] += 1

    for gt_index, gt in enumerate(ground_truth):
        if gt_index in matched_gt:
            continue
        bucket = _size_bucket(gt["box"], image_area, thresholds)
        kind_totals[bucket]["fn"] += 1

    for pred_index, pred in enumerate(predictions):
        if pred_index in matched_pred:
            continue
        bucket = _size_bucket(pred["box"], image_area, thresholds)
        kind_totals[bucket]["fp"] += 1


def _size_bucket(box: Box, image_area: int, thresholds: list[float]) -> str:
    area_ratio = float(box.area) / float(max(image_area, 1))
    if area_ratio < thresholds[0]:
        return "tiny"
    if area_ratio < thresholds[1]:
        return "small"
    if area_ratio < thresholds[2]:
        return "medium"
    return "large"


def _build_size_metrics(
    size_totals: dict[str, dict[str, Counter[str]]],
    kinds: set[str],
    thresholds: list[float],
) -> dict[str, Any]:
    metrics = {
        kind: {
            bucket: _metrics_from_counter(size_totals[kind][bucket])
            for bucket in SIZE_BUCKET_NAMES
        }
        for kind in sorted(kinds)
    }
    macro_f1_by_bucket = {}
    for bucket in SIZE_BUCKET_NAMES:
        values = [float(metrics[kind][bucket]["f1"]) for kind in metrics]
        macro_f1_by_bucket[bucket] = round(sum(values) / max(len(values), 1), 4)
    return {
        "bucket_definition": (
            "bbox_area / image_area. TP and FN are bucketed by ground-truth box area; "
            "FP is bucketed by predicted box area."
        ),
        "area_ratio_thresholds": [round(float(value), 6) for value in thresholds],
        "buckets": SIZE_BUCKET_NAMES,
        "macro_f1_by_bucket": macro_f1_by_bucket,
        "metrics": metrics,
    }


def _get_group_bucket(
    buckets: dict[str, dict[str, Any]],
    value: str,
    kinds: set[str],
) -> dict[str, Any]:
    if value not in buckets:
        buckets[value] = {
            "processed_images": 0,
            "total_inference_ms": 0.0,
            "object_totals": {kind: _empty_counter() for kind in kinds},
            "lane_presence": _empty_counter(),
            "traffic_light_state": {"matched": 0, "correct": 0},
        }
    return buckets[value]


def _frame_attribute(frame: dict[str, Any], attribute: str) -> str:
    value = str(frame.get("attributes", {}).get(attribute, "")).strip().lower()
    return value or "unknown"


def _macro_metric(object_metrics: dict[str, dict[str, Any]], metric: str) -> float:
    values = [float(metrics.get(metric, 0.0)) for metrics in object_metrics.values()]
    return sum(values) / max(len(values), 1)


def _metrics_from_counter(counter: Counter[str]) -> dict[str, Any]:
    tp = int(counter["tp"])
    fp = int(counter["fp"])
    fn = int(counter["fn"])
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-9)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def _update_presence_counter(counter: Counter[str], gt_present: bool, pred_present: bool) -> None:
    if gt_present and pred_present:
        counter["tp"] += 1
    elif pred_present and not gt_present:
        counter["fp"] += 1
    elif gt_present and not pred_present:
        counter["fn"] += 1
    else:
        counter["tn"] += 1


def _empty_counter() -> Counter[str]:
    return Counter({"tp": 0, "fp": 0, "fn": 0})


def _box_from_bdd(box2d: dict[str, Any]) -> Box:
    return Box(
        x1=int(round(float(box2d["x1"]))),
        y1=int(round(float(box2d["y1"]))),
        x2=int(round(float(box2d["x2"]))),
        y2=int(round(float(box2d["y2"]))),
    )


def _iou(a: Box, b: Box) -> float:
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = a.area + b.area - inter
    if union <= 0:
        return 0.0
    return float(inter) / float(union)


def _traffic_light_state(label: dict[str, Any]) -> str | None:
    state = str(label.get("attributes", {}).get("trafficLightColor", "")).lower()
    return state if state in {"red", "yellow", "green"} else None


def _predicted_traffic_light_state(detection: Detection) -> str | None:
    label = detection.label.lower()
    for state in ["red", "yellow", "green"]:
        if label.startswith(state):
            return state
    return None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


if __name__ == "__main__":
    raise SystemExit(main())
