#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check BDD100K/Scalabel label and image paths.")
    parser.add_argument("--images-root", required=True, help="Directory containing BDD100K images.")
    parser.add_argument("--labels", required=True, help="BDD100K/Scalabel label JSON path.")
    parser.add_argument("--max-samples", type=int, default=200, help="Number of image paths to check.")
    parser.add_argument("--output", default=None, help="Optional JSON check report path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = check_bdd100k(
        images_root=Path(args.images_root),
        labels_path=Path(args.labels),
        max_samples=args.max_samples,
    )
    print_report(report)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
            f.write("\n")
        print(f"Saved {output_path}")
    return 0 if report["ready_for_eval"] else 2


def check_bdd100k(images_root: Path, labels_path: Path, max_samples: int) -> dict[str, Any]:
    label_exists = labels_path.exists()
    images_root_exists = images_root.exists() and images_root.is_dir()
    frames: list[dict[str, Any]] = []
    load_error = None
    if label_exists:
        try:
            frames = _load_frames(labels_path)
        except Exception as exc:
            load_error = f"{type(exc).__name__}: {exc}"

    category_counts = Counter()
    traffic_light_colors = Counter()
    frames_with_box2d = 0
    frames_with_lane = 0
    labels_with_box2d = 0
    total_labels = 0
    missing_images = []
    checked_images = 0

    for index, frame in enumerate(frames):
        frame_labels = frame.get("labels", [])
        has_box2d = False
        has_lane = False
        for label in frame_labels:
            total_labels += 1
            category = str(label.get("category", "unknown"))
            category_counts[category] += 1
            if "box2d" in label:
                labels_with_box2d += 1
                has_box2d = True
            if category == "lane":
                has_lane = True
            color = str(label.get("attributes", {}).get("trafficLightColor", "")).lower()
            if color:
                traffic_light_colors[color] += 1
        if has_box2d:
            frames_with_box2d += 1
        if has_lane:
            frames_with_lane += 1

        if index < max_samples:
            checked_images += 1
            image_path = images_root / str(frame.get("name", ""))
            if not image_path.exists():
                missing_images.append(str(image_path))

    ready = bool(label_exists and images_root_exists and frames and load_error is None and not missing_images)
    return {
        "schema_version": "0.1",
        "images_root": str(images_root),
        "labels": str(labels_path),
        "ready_for_eval": ready,
        "exists": {
            "images_root": images_root_exists,
            "labels": label_exists,
        },
        "load_error": load_error,
        "frames": len(frames),
        "total_labels": total_labels,
        "labels_with_box2d": labels_with_box2d,
        "frames_with_box2d": frames_with_box2d,
        "frames_with_lane": frames_with_lane,
        "category_counts": dict(sorted(category_counts.items())),
        "traffic_light_colors": dict(sorted(traffic_light_colors.items())),
        "checked_images": checked_images,
        "missing_images_count": len(missing_images),
        "missing_images_sample": missing_images[:20],
    }


def print_report(report: dict[str, Any]) -> None:
    print("BDD100K dataset check")
    print(f"- images_root: {report['images_root']} ({report['exists']['images_root']})")
    print(f"- labels: {report['labels']} ({report['exists']['labels']})")
    if report["load_error"]:
        print(f"- load_error: {report['load_error']}")
    print(f"- frames: {report['frames']}")
    print(f"- total_labels: {report['total_labels']}")
    print(f"- labels_with_box2d: {report['labels_with_box2d']}")
    print(f"- frames_with_lane: {report['frames_with_lane']}")
    print(f"- checked_images: {report['checked_images']}")
    print(f"- missing_images_count: {report['missing_images_count']}")
    print(f"- ready_for_eval: {report['ready_for_eval']}")
    if report["category_counts"]:
        print("- top_categories:")
        for category, count in sorted(report["category_counts"].items(), key=lambda item: item[1], reverse=True)[:10]:
            print(f"  - {category}: {count}")


def _load_frames(labels_path: Path) -> list[dict[str, Any]]:
    with labels_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, list):
        return list(payload)
    if isinstance(payload, dict) and "frames" in payload:
        return list(payload["frames"])
    raise ValueError("Expected a list of frames or an object with a 'frames' field.")


if __name__ == "__main__":
    raise SystemExit(main())

