#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
from typing import Any

import cv2


DEFAULT_SIZE_BUCKET_AREAS = [0.0005, 0.0025, 0.01]
SIZE_BUCKET_NAMES = ["tiny", "small", "medium", "large"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize sampled BDD100K TP/FP/FN error JSON files.")
    parser.add_argument("--errors", nargs="+", required=True, help="Error JSON files from evaluate_bdd100k.py.")
    parser.add_argument("--names", nargs="*", default=None, help="Display names matching --errors.")
    parser.add_argument("--labels", default=None, help="Optional BDD100K/Scalabel label JSON for frame attributes.")
    parser.add_argument("--images-root", default=None, help="Optional image root used for exact bbox area ratios.")
    parser.add_argument("--image-width", type=int, default=1280, help="Fallback image width.")
    parser.add_argument("--image-height", type=int, default=720, help="Fallback image height.")
    parser.add_argument(
        "--size-bucket-areas",
        nargs=3,
        type=float,
        default=DEFAULT_SIZE_BUCKET_AREAS,
        metavar=("TINY_MAX", "SMALL_MAX", "MEDIUM_MAX"),
        help="Normalized bbox area thresholds. Defaults to 0.0005 0.0025 0.01.",
    )
    parser.add_argument(
        "--group-by",
        nargs="*",
        default=[],
        choices=["weather", "timeofday", "scene"],
        help="Also count TP/FP/FN samples by BDD100K frame attributes. Requires --labels.",
    )
    parser.add_argument("--output", default=None, help="Optional JSON report path.")
    parser.add_argument("--markdown-output", default=None, help="Optional Markdown report path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    error_paths = [Path(path) for path in args.errors]
    names = _resolve_names(error_paths, args.names)
    thresholds = _validate_thresholds(args.size_bucket_areas)
    images_root = Path(args.images_root) if args.images_root else None
    group_by = list(args.group_by)
    frame_attributes = _load_frame_attributes(Path(args.labels), group_by) if args.labels and group_by else {}
    if group_by and not args.labels:
        raise ValueError("--group-by requires --labels.")

    reports = []
    image_size_cache: dict[str, tuple[int, int]] = {}
    for name, path in zip(names, error_paths):
        reports.append(
            _analyze_file(
                name=name,
                path=path,
                images_root=images_root,
                fallback_size=(args.image_width, args.image_height),
                thresholds=thresholds,
                group_by=group_by,
                frame_attributes=frame_attributes,
                image_size_cache=image_size_cache,
            )
        )

    payload = {
        "schema_version": "0.1",
        "size_bucket_area_thresholds": thresholds,
        "reports": reports,
    }
    if args.output:
        _write_json(Path(args.output), payload)
        print(f"Saved {args.output}")
    if args.markdown_output:
        Path(args.markdown_output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.markdown_output).write_text(_markdown(payload), encoding="utf-8")
        print(f"Saved {args.markdown_output}")
    if not args.output and not args.markdown_output:
        print(_markdown(payload))
    return 0


def _resolve_names(paths: list[Path], raw_names: list[str] | None) -> list[str]:
    if raw_names:
        if len(raw_names) != len(paths):
            raise ValueError("--names must have the same length as --errors.")
        return [str(name) for name in raw_names]
    return [path.stem for path in paths]


def _load_frame_attributes(labels_path: Path, group_by: list[str]) -> dict[str, dict[str, str]]:
    with labels_path.open("r", encoding="utf-8") as f:
        frames = json.load(f)
    attributes_by_frame: dict[str, dict[str, str]] = {}
    for frame in frames:
        name = str(frame.get("name", ""))
        if not name:
            continue
        raw_attributes = frame.get("attributes", {})
        if not isinstance(raw_attributes, dict):
            raw_attributes = {}
        attributes_by_frame[name] = {
            attr: str(raw_attributes.get(attr) or "undefined")
            for attr in group_by
        }
    return attributes_by_frame


def _validate_thresholds(values: list[float]) -> list[float]:
    if len(values) != 3:
        raise ValueError("Expected exactly three --size-bucket-areas values.")
    thresholds = [float(value) for value in values]
    if any(value <= 0 for value in thresholds) or thresholds != sorted(thresholds):
        raise ValueError("--size-bucket-areas must be positive and sorted.")
    return thresholds


def _analyze_file(
    *,
    name: str,
    path: Path,
    images_root: Path | None,
    fallback_size: tuple[int, int],
    thresholds: list[float],
    group_by: list[str],
    frame_attributes: dict[str, dict[str, str]],
    image_size_cache: dict[str, tuple[int, int]],
) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    errors = payload.get("errors", {})
    return {
        "name": name,
        "errors": str(path),
        "kinds": {
            kind: _analyze_kind(
                kind_payload,
                images_root=images_root,
                fallback_size=fallback_size,
                thresholds=thresholds,
                group_by=group_by,
                frame_attributes=frame_attributes,
                image_size_cache=image_size_cache,
            )
            for kind, kind_payload in sorted(errors.items())
        },
    }


def _analyze_kind(
    kind_payload: dict[str, Any],
    *,
    images_root: Path | None,
    fallback_size: tuple[int, int],
    thresholds: list[float],
    group_by: list[str],
    frame_attributes: dict[str, dict[str, str]],
    image_size_cache: dict[str, tuple[int, int]],
) -> dict[str, Any]:
    buckets = {}
    for bucket in ["tp", "fp", "fn"]:
        samples = list(kind_payload.get(bucket, []))
        buckets[bucket] = _analyze_bucket(
            samples,
            bucket=bucket,
            images_root=images_root,
            fallback_size=fallback_size,
            thresholds=thresholds,
            group_by=group_by,
            frame_attributes=frame_attributes,
            image_size_cache=image_size_cache,
        )
    return {
        "max_samples": kind_payload.get("max_samples"),
        "buckets": buckets,
    }


def _analyze_bucket(
    samples: list[dict[str, Any]],
    *,
    bucket: str,
    images_root: Path | None,
    fallback_size: tuple[int, int],
    thresholds: list[float],
    group_by: list[str],
    frame_attributes: dict[str, dict[str, str]],
    image_size_cache: dict[str, tuple[int, int]],
) -> dict[str, Any]:
    label_counter = Counter()
    size_counter = Counter()
    frame_counter = Counter()
    attribute_counters = {attr: Counter() for attr in group_by}
    confidences = []
    ious = []
    area_ratios = []
    for sample in samples:
        frame = str(sample.get("frame", ""))
        if frame:
            frame_counter[frame] += 1
            sample_attributes = frame_attributes.get(frame, {})
            for attr in group_by:
                attribute_counters[attr][str(sample_attributes.get(attr) or "undefined")] += 1
        label = sample.get("gt_label") if bucket == "fn" else sample.get("pred_label") or sample.get("gt_label")
        if label:
            label_counter[str(label)] += 1
        confidence = sample.get("confidence")
        if confidence is not None:
            confidences.append(float(confidence))
        iou = sample.get("iou")
        if iou is not None:
            ious.append(float(iou))

        box = _sample_box(sample, bucket)
        if box:
            image_width, image_height = _image_size(frame, images_root, fallback_size, image_size_cache)
            area_ratio = _box_area_ratio(box, image_width=image_width, image_height=image_height)
            area_ratios.append(area_ratio)
            size_counter[_size_bucket(area_ratio, thresholds)] += 1

    return {
        "samples": len(samples),
        "labels": dict(label_counter.most_common()),
        "size_buckets": {name: int(size_counter.get(name, 0)) for name in SIZE_BUCKET_NAMES},
        "confidence": _numeric_summary(confidences),
        "iou": _numeric_summary(ious),
        "area_ratio": _numeric_summary(area_ratios),
        "attributes": {
            attr: dict(counter.most_common())
            for attr, counter in attribute_counters.items()
        },
        "top_frames": dict(frame_counter.most_common(10)),
    }


def _sample_box(sample: dict[str, Any], bucket: str) -> dict[str, Any] | None:
    if bucket == "fp":
        return sample.get("pred_box")
    if bucket == "fn":
        return sample.get("gt_box")
    return sample.get("gt_box") or sample.get("pred_box")


def _image_size(
    frame: str,
    images_root: Path | None,
    fallback_size: tuple[int, int],
    cache: dict[str, tuple[int, int]],
) -> tuple[int, int]:
    if not images_root or not frame:
        return fallback_size
    if frame in cache:
        return cache[frame]
    image = cv2.imread(str(images_root / frame))
    if image is None:
        cache[frame] = fallback_size
        return fallback_size
    height, width = image.shape[:2]
    cache[frame] = (int(width), int(height))
    return cache[frame]


def _box_area_ratio(box: dict[str, Any], *, image_width: int, image_height: int) -> float:
    width = max(0.0, float(box.get("x2", 0)) - float(box.get("x1", 0)))
    height = max(0.0, float(box.get("y2", 0)) - float(box.get("y1", 0)))
    return width * height / max(1.0, float(image_width * image_height))


def _size_bucket(area_ratio: float, thresholds: list[float]) -> str:
    if area_ratio <= thresholds[0]:
        return "tiny"
    if area_ratio <= thresholds[1]:
        return "small"
    if area_ratio <= thresholds[2]:
        return "medium"
    return "large"


def _numeric_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": round(float(statistics.fmean(values)), 6),
        "median": round(float(statistics.median(values)), 6),
        "min": round(float(min(values)), 6),
        "max": round(float(max(values)), 6),
    }


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# BDD100K Error Sample Analysis",
        "",
        f"- Size buckets: `{payload['size_bucket_area_thresholds']}`",
        "",
    ]
    for report in payload["reports"]:
        lines.extend([f"## {report['name']}", "", f"- Source: `{report['errors']}`", ""])
        for kind, kind_payload in report["kinds"].items():
            lines.extend([f"### {kind}", ""])
            lines.extend(
                [
                    "| bucket | samples | tiny | small | medium | large | conf mean | IoU mean | top labels |",
                    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
                ]
            )
            for bucket, bucket_payload in kind_payload["buckets"].items():
                sizes = bucket_payload["size_buckets"]
                conf_mean = _display_number(bucket_payload["confidence"]["mean"])
                iou_mean = _display_number(bucket_payload["iou"]["mean"])
                labels = _top_items(bucket_payload["labels"], 4)
                lines.append(
                    f"| {bucket} | {bucket_payload['samples']} | "
                    f"{sizes['tiny']} | {sizes['small']} | {sizes['medium']} | {sizes['large']} | "
                    f"{conf_mean} | {iou_mean} | {labels} |"
                )
            lines.append("")
            attrs = _attribute_names(kind_payload)
            if attrs:
                lines.append("| bucket | " + " | ".join(attrs) + " |")
                lines.append("| --- | " + " | ".join(["---"] * len(attrs)) + " |")
                for bucket, bucket_payload in kind_payload["buckets"].items():
                    values = [
                        _top_items(bucket_payload.get("attributes", {}).get(attr, {}), 4)
                        for attr in attrs
                    ]
                    lines.append("| " + bucket + " | " + " | ".join(values) + " |")
                lines.append("")
    return "\n".join(lines) + "\n"


def _attribute_names(kind_payload: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for bucket_payload in kind_payload["buckets"].values():
        for name in bucket_payload.get("attributes", {}):
            if name not in names:
                names.append(name)
    return names


def _top_items(items: dict[str, int], limit: int) -> str:
    if not items:
        return ""
    return ", ".join(f"{name}={count}" for name, count in list(items.items())[:limit])


def _display_number(value: float | int | None) -> str:
    if value is None:
        return ""
    return f"{float(value):.3f}"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


if __name__ == "__main__":
    raise SystemExit(main())
