#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import shutil
from typing import Any

import cv2
import numpy as np
import yaml


DEFAULT_CLASSES = ["traffic sign", "traffic light"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export BDD100K-style labels to Ultralytics YOLO format.")
    parser.add_argument("--images-root", required=True, help="Directory containing BDD100K images.")
    parser.add_argument("--labels", required=True, help="BDD100K/Scalabel label JSON path.")
    parser.add_argument(
        "--val-images-root",
        default=None,
        help=(
            "Optional separate BDD100K validation image root. When set with --val-labels, "
            "--images-root/--labels are used as train and no ratio/alternate split is applied."
        ),
    )
    parser.add_argument(
        "--val-labels",
        default=None,
        help=(
            "Optional separate BDD100K/Scalabel validation label JSON. Must be used with "
            "--val-images-root."
        ),
    )
    parser.add_argument("--output-dir", default="data/bdd100k_yolo_sign_light", help="Output YOLO dataset root.")
    parser.add_argument(
        "--classes",
        nargs="+",
        default=DEFAULT_CLASSES,
        help="BDD categories to export, in YOLO class-id order.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Export first N frames after filtering. In separate train/val mode this limits the train source.",
    )
    parser.add_argument(
        "--max-train-images",
        type=int,
        default=None,
        help="Limit exported train source frames after filtering. Overrides --max-images for train.",
    )
    parser.add_argument(
        "--max-val-images",
        type=int,
        default=None,
        help="Limit exported validation source frames after filtering when --val-labels is used.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.20, help="Deterministic validation ratio.")
    parser.add_argument(
        "--split-mode",
        choices=["ratio", "alternate"],
        default="ratio",
        help="Use the existing ratio split or split by frame index modulo --frame-stride.",
    )
    parser.add_argument("--frame-stride", type=int, default=2, help="Frame stride used by --split-mode alternate.")
    parser.add_argument("--train-frame-offset", type=int, default=0, help="Train offset used by alternate split.")
    parser.add_argument("--val-frame-offset", type=int, default=1, help="Validation offset used by alternate split.")
    parser.add_argument(
        "--hard-frames-errors",
        nargs="*",
        default=[],
        help="Optional evaluate_bdd100k.py --save-errors JSON files. FP/FN frames are marked as hard.",
    )
    parser.add_argument(
        "--hard-buckets",
        nargs="+",
        default=["fp", "fn"],
        choices=["tp", "fp", "fn"],
        help="Error buckets that mark hard frames when --hard-frames-errors is used.",
    )
    parser.add_argument(
        "--small-object-area-threshold",
        type=float,
        default=None,
        help="Mark frames with at least one selected bbox area below this normalized area.",
    )
    parser.add_argument(
        "--extra-hard-copies",
        type=int,
        default=0,
        help="Extra training copies for frames marked by --hard-frames-errors.",
    )
    parser.add_argument(
        "--extra-small-object-copies",
        type=int,
        default=0,
        help="Extra training copies for frames containing selected small objects.",
    )
    parser.add_argument(
        "--extra-class-copies",
        nargs="*",
        default=[],
        metavar="CLASS=N",
        help=(
            "Extra training copies for frames containing selected classes, for example "
            "'pedestrian=1' 'traffic sign=1'. Class names use --classes names; underscores are accepted."
        ),
    )
    parser.add_argument(
        "--max-extra-copies-per-image",
        type=int,
        default=None,
        help="Optional cap for combined extra training copies per source frame.",
    )
    parser.add_argument(
        "--object-crop-classes",
        nargs="*",
        default=[],
        help=(
            "Create additional train crops around selected classes, for example "
            "'pedestrian' 'traffic sign'. Class names use --classes names; underscores are accepted."
        ),
    )
    parser.add_argument(
        "--object-crop-area-threshold",
        type=float,
        default=0.0025,
        help="Only crop selected objects at or below this normalized bbox area.",
    )
    parser.add_argument(
        "--object-crop-padding",
        type=float,
        default=4.0,
        help="Context padding around the target box, in target box widths/heights per side.",
    )
    parser.add_argument("--object-crop-min-size", type=int, default=320, help="Minimum crop side length in pixels.")
    parser.add_argument("--object-crop-max-size", type=int, default=640, help="Maximum crop side length in pixels.")
    parser.add_argument(
        "--max-object-crops-per-image",
        type=int,
        default=3,
        help="Maximum number of additional object crops per source train frame.",
    )
    parser.add_argument(
        "--object-crop-min-box-size",
        type=int,
        default=2,
        help="Drop clipped boxes smaller than this many pixels in object crops.",
    )
    parser.add_argument(
        "--copy-paste-classes",
        nargs="*",
        default=[],
        help=(
            "Create additional full-frame train images with selected small objects pasted into road context. "
            "Class names use --classes names; underscores are accepted."
        ),
    )
    parser.add_argument(
        "--copy-paste-area-threshold",
        type=float,
        default=0.0025,
        help="Use source objects at or below this normalized bbox area for copy-paste.",
    )
    parser.add_argument(
        "--copy-paste-source-min-area",
        type=float,
        default=0.0,
        help="Drop source objects below this normalized bbox area for copy-paste.",
    )
    parser.add_argument(
        "--copy-paste-source-min-box-size",
        type=int,
        default=1,
        help="Drop source objects whose original bbox width or height is smaller than this many pixels.",
    )
    parser.add_argument(
        "--copy-paste-source-max-aspect-ratio",
        type=float,
        default=0.0,
        help="Drop source objects with width/height or height/width above this ratio. Default 0 disables it.",
    )
    parser.add_argument(
        "--copy-paste-max-images",
        type=int,
        default=0,
        help="Maximum additional train images to generate with copy-paste. Default 0 disables it.",
    )
    parser.add_argument(
        "--copy-paste-objects-per-image",
        type=int,
        default=1,
        help="Maximum pasted objects per generated train image.",
    )
    parser.add_argument(
        "--copy-paste-context-padding",
        type=float,
        default=0.25,
        help="Context padding around the pasted source box, in source box widths/heights per side.",
    )
    parser.add_argument("--copy-paste-scale-min", type=float, default=0.8, help="Minimum paste scale.")
    parser.add_argument("--copy-paste-scale-max", type=float, default=1.2, help="Maximum paste scale.")
    parser.add_argument(
        "--copy-paste-blend",
        choices=["none", "feather"],
        default="none",
        help="Blend mode for generated copy-paste image patches.",
    )
    parser.add_argument(
        "--copy-paste-mask",
        choices=["none", "box", "grabcut"],
        default="none",
        help=(
            "Optional alpha mask for pasted patches. 'box' keeps only the source object bbox; "
            "'grabcut' estimates foreground inside that bbox."
        ),
    )
    parser.add_argument(
        "--copy-paste-feather-ratio",
        type=float,
        default=0.08,
        help="Feather width as a ratio of the smaller pasted patch side when --copy-paste-blend=feather.",
    )
    parser.add_argument(
        "--copy-paste-max-overlap",
        type=float,
        default=0.05,
        help="Maximum IoU allowed between a pasted object and existing labels.",
    )
    parser.add_argument(
        "--copy-paste-min-box-size",
        type=int,
        default=4,
        help="Drop pasted boxes smaller than this many pixels.",
    )
    parser.add_argument("--copy-paste-seed", type=int, default=0, help="Deterministic copy-paste seed.")
    parser.add_argument(
        "--include-hard-empty",
        action="store_true",
        help="Include hard frames even when they contain no selected ground-truth boxes.",
    )
    parser.add_argument("--image-width", type=int, default=1280, help="Fallback image width for bbox normalization.")
    parser.add_argument("--image-height", type=int, default=720, help="Fallback image height for bbox normalization.")
    parser.add_argument("--read-image-size", action="store_true", help="Read every image with OpenCV to get dimensions.")
    parser.add_argument("--copy-images", action="store_true", help="Copy images instead of using relative symlinks.")
    parser.add_argument("--clear-output", action="store_true", help="Remove the output directory before exporting.")
    parser.add_argument("--include-empty", action="store_true", help="Include frames with no selected labels.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    images_root = Path(args.images_root)
    labels_path = Path(args.labels)
    val_images_root = Path(args.val_images_root) if args.val_images_root else None
    val_labels_path = Path(args.val_labels) if args.val_labels else None
    output_dir = Path(args.output_dir)
    class_names = [str(name) for name in args.classes]
    class_to_id = {_normalize_class_key(name): index for index, name in enumerate(class_names)}
    extra_class_copies = _parse_class_copy_rules(args.extra_class_copies, class_to_id, class_names)
    object_crop_class_ids = _parse_class_id_list(
        args.object_crop_classes,
        class_to_id,
        class_names,
        option_name="--object-crop-classes",
    )
    copy_paste_class_ids = _parse_class_id_list(
        args.copy_paste_classes,
        class_to_id,
        class_names,
        option_name="--copy-paste-classes",
    )
    hard_frames = _load_hard_frame_names(args.hard_frames_errors, set(args.hard_buckets))

    if args.clear_output and output_dir.exists():
        shutil.rmtree(output_dir)

    frames = _load_frames(labels_path)
    explicit_val_split = val_images_root is not None or val_labels_path is not None
    if explicit_val_split and (val_images_root is None or val_labels_path is None):
        raise ValueError("--val-images-root and --val-labels must be used together.")

    train_limit = args.max_train_images if args.max_train_images is not None else args.max_images
    selected = _select_samples(
        frames=frames,
        images_root=images_root,
        class_to_id=class_to_id,
        fallback_size=(args.image_width, args.image_height),
        read_image_size=args.read_image_size,
        hard_frames=hard_frames,
        small_object_area_threshold=args.small_object_area_threshold,
        extra_class_copies=extra_class_copies,
        include_empty=args.include_empty,
        include_hard_empty=args.include_hard_empty,
        max_images=train_limit,
    )
    if explicit_val_split:
        val_frames = _load_frames(val_labels_path)
        val_selected = _select_samples(
            frames=val_frames,
            images_root=val_images_root,
            class_to_id=class_to_id,
            fallback_size=(args.image_width, args.image_height),
            read_image_size=args.read_image_size,
            hard_frames=set(),
            small_object_area_threshold=args.small_object_area_threshold,
            extra_class_copies={},
            include_empty=args.include_empty,
            include_hard_empty=False,
            max_images=args.max_val_images,
        )
        splits = {"train": selected, "val": val_selected}
        split_mode = "explicit"
    else:
        splits = _split_samples(
            selected,
            split_mode=args.split_mode,
            val_ratio=args.val_ratio,
            frame_stride=args.frame_stride,
            train_frame_offset=args.train_frame_offset,
            val_frame_offset=args.val_frame_offset,
        )
        split_mode = args.split_mode

    train_samples = splits["train"]
    object_crop_samples = _build_object_crop_samples(
        train_samples,
        class_ids=object_crop_class_ids,
        area_threshold=args.object_crop_area_threshold,
        padding=args.object_crop_padding,
        min_size=args.object_crop_min_size,
        max_size=args.object_crop_max_size,
        max_crops_per_image=args.max_object_crops_per_image,
        min_box_size=args.object_crop_min_box_size,
    )
    copy_paste_samples = _build_copy_paste_samples(
        train_samples,
        class_ids=copy_paste_class_ids,
        area_threshold=args.copy_paste_area_threshold,
        source_min_area=args.copy_paste_source_min_area,
        source_min_box_size=args.copy_paste_source_min_box_size,
        source_max_aspect_ratio=args.copy_paste_source_max_aspect_ratio,
        max_images=args.copy_paste_max_images,
        objects_per_image=args.copy_paste_objects_per_image,
        context_padding=args.copy_paste_context_padding,
        scale_min=args.copy_paste_scale_min,
        scale_max=args.copy_paste_scale_max,
        max_overlap=args.copy_paste_max_overlap,
        min_box_size=args.copy_paste_min_box_size,
        seed=args.copy_paste_seed,
    )
    splits["train"] = _expand_training_samples(
        train_samples,
        extra_hard_copies=max(0, args.extra_hard_copies),
        extra_small_object_copies=max(0, args.extra_small_object_copies),
        max_extra_copies_per_image=args.max_extra_copies_per_image,
    ) + object_crop_samples + copy_paste_samples

    stats = {
        "schema_version": "0.1",
        "source_images_root": str(images_root),
        "source_labels": str(labels_path),
        "source_val_images_root": str(val_images_root) if val_images_root is not None else None,
        "source_val_labels": str(val_labels_path) if val_labels_path is not None else None,
        "output_dir": str(output_dir),
        "classes": class_names,
        "split_mode": split_mode,
        "frame_stride": args.frame_stride,
        "train_frame_offset": args.train_frame_offset,
        "val_frame_offset": args.val_frame_offset,
        "max_images": args.max_images,
        "max_train_images": args.max_train_images,
        "max_val_images": args.max_val_images,
        "hard_frames_errors": args.hard_frames_errors,
        "hard_buckets": args.hard_buckets,
        "small_object_area_threshold": args.small_object_area_threshold,
        "extra_hard_copies": args.extra_hard_copies,
        "extra_small_object_copies": args.extra_small_object_copies,
        "extra_class_copies": {
            class_names[class_id]: copies for class_id, copies in sorted(extra_class_copies.items())
        },
        "max_extra_copies_per_image": args.max_extra_copies_per_image,
        "object_crop_classes": [class_names[class_id] for class_id in sorted(object_crop_class_ids)],
        "object_crop_area_threshold": args.object_crop_area_threshold,
        "object_crop_padding": args.object_crop_padding,
        "object_crop_min_size": args.object_crop_min_size,
        "object_crop_max_size": args.object_crop_max_size,
        "max_object_crops_per_image": args.max_object_crops_per_image,
        "object_crop_min_box_size": args.object_crop_min_box_size,
        "copy_paste_classes": [class_names[class_id] for class_id in sorted(copy_paste_class_ids)],
        "copy_paste_area_threshold": args.copy_paste_area_threshold,
        "copy_paste_source_min_area": args.copy_paste_source_min_area,
        "copy_paste_source_min_box_size": args.copy_paste_source_min_box_size,
        "copy_paste_source_max_aspect_ratio": args.copy_paste_source_max_aspect_ratio,
        "copy_paste_max_images": args.copy_paste_max_images,
        "copy_paste_objects_per_image": args.copy_paste_objects_per_image,
        "copy_paste_context_padding": args.copy_paste_context_padding,
        "copy_paste_scale_min": args.copy_paste_scale_min,
        "copy_paste_scale_max": args.copy_paste_scale_max,
        "copy_paste_blend": args.copy_paste_blend,
        "copy_paste_mask": args.copy_paste_mask,
        "copy_paste_feather_ratio": args.copy_paste_feather_ratio,
        "copy_paste_max_overlap": args.copy_paste_max_overlap,
        "copy_paste_min_box_size": args.copy_paste_min_box_size,
        "copy_paste_seed": args.copy_paste_seed,
        "splits": {},
    }

    for split, items in splits.items():
        image_dir = output_dir / "images" / split
        label_dir = output_dir / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        label_counts = {name: 0 for name in class_names}
        unique_source_frames = set()
        hard_count = 0
        small_object_count = 0
        class_boost_count = 0
        duplicate_count = 0
        crop_count = 0
        copy_paste_count = 0
        for item in items:
            frame = item["frame"]
            boxes = item["boxes"]
            image_name = str(frame.get("name", ""))
            exported_name = str(item.get("exported_name") or _export_image_name(image_name, int(item.get("copy_index", 0))))
            item_images_root = Path(item.get("images_root", images_root))
            source_image = item_images_root / image_name
            target_image = image_dir / exported_name
            target_label = label_dir / f"{Path(exported_name).stem}.txt"

            if item.get("sample_type") == "object_crop":
                _write_crop_image(source_image, target_image, item["crop_rect"])
            elif item.get("sample_type") == "copy_paste":
                _write_copy_paste_image(
                    item_images_root,
                    source_image,
                    target_image,
                    item["paste_objects"],
                    blend_mode=args.copy_paste_blend,
                    mask_mode=args.copy_paste_mask,
                    feather_ratio=args.copy_paste_feather_ratio,
                )
            else:
                _link_or_copy(source_image, target_image, copy=args.copy_images)
            target_label.parent.mkdir(parents=True, exist_ok=True)
            target_label.write_text(_yolo_label_text(boxes), encoding="utf-8")
            unique_source_frames.add(image_name)
            hard_count += int(bool(item.get("hard", False)))
            small_object_count += int(bool(item.get("small_object", False)))
            class_boost_count += int(int(item.get("class_copy_extra", 0)) > 0)
            duplicate_count += int(int(item.get("copy_index", 0)) > 0)
            crop_count += int(item.get("sample_type") == "object_crop")
            copy_paste_count += int(item.get("sample_type") == "copy_paste")
            for box in boxes:
                label_counts[class_names[int(box["class_id"])]] += 1

        stats["splits"][split] = {
            "images": len(items),
            "unique_source_frames": len(unique_source_frames),
            "duplicate_images": duplicate_count,
            "hard_images": hard_count,
            "small_object_images": small_object_count,
            "class_boost_images": class_boost_count,
            "labels": sum(label_counts.values()),
            "object_crop_images": crop_count,
            "copy_paste_images": copy_paste_count,
            "class_counts": label_counts,
        }

    _write_dataset_yaml(output_dir, class_names)
    _write_json(output_dir / "export_stats.json", stats)
    _print_report(stats)
    return 0


def _select_samples(
    *,
    frames: list[dict[str, Any]],
    images_root: Path,
    class_to_id: dict[str, int],
    fallback_size: tuple[int, int],
    read_image_size: bool,
    hard_frames: set[str],
    small_object_area_threshold: float | None,
    extra_class_copies: dict[int, int],
    include_empty: bool,
    include_hard_empty: bool,
    max_images: int | None,
) -> list[dict[str, Any]]:
    selected = []
    for frame_index, frame in enumerate(frames):
        boxes = _frame_boxes(
            frame,
            class_to_id,
            images_root,
            fallback_size=fallback_size,
            read_image_size=read_image_size,
        )
        image_name = str(frame.get("name", ""))
        hard = image_name in hard_frames
        small_object = _has_small_object(boxes, small_object_area_threshold)
        class_copy_extra = _class_copy_extra(boxes, extra_class_copies)
        if boxes or include_empty or (include_hard_empty and hard):
            selected.append(
                {
                    "frame": frame,
                    "boxes": boxes,
                    "frame_index": frame_index,
                    "images_root": images_root,
                    "hard": hard,
                    "small_object": small_object,
                    "class_copy_extra": class_copy_extra,
                }
            )
        if max_images is not None and len(selected) >= max_images:
            break
    return selected


def _load_hard_frame_names(error_paths: list[str], buckets: set[str]) -> set[str]:
    frame_names: set[str] = set()
    for raw_path in error_paths:
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(f"Error JSON not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        for kind_payload in payload.get("errors", {}).values():
            for bucket in buckets:
                for sample in kind_payload.get(bucket, []):
                    frame = str(sample.get("frame", "")).strip()
                    if frame:
                        frame_names.add(frame)
    return frame_names


def _parse_class_copy_rules(
    raw_rules: list[str],
    class_to_id: dict[str, int],
    class_names: list[str],
) -> dict[int, int]:
    rules: dict[int, int] = {}
    for raw_rule in raw_rules:
        if "=" not in raw_rule:
            raise ValueError(f"Expected CLASS=N for --extra-class-copies, got: {raw_rule}")
        raw_class, raw_copies = raw_rule.split("=", 1)
        class_key = _normalize_class_key(raw_class)
        if class_key.isdigit():
            class_id = int(class_key)
            if not 0 <= class_id < len(class_names):
                raise ValueError(f"Class id out of range for --extra-class-copies: {class_id}")
        else:
            if class_key not in class_to_id:
                valid = ", ".join(class_names)
                raise ValueError(f"Unknown class for --extra-class-copies: {raw_class}. Valid classes: {valid}")
            class_id = class_to_id[class_key]
        copies = int(raw_copies)
        if copies < 0:
            raise ValueError(f"Copy count must be >= 0 for --extra-class-copies: {raw_rule}")
        if copies > 0:
            rules[class_id] = max(rules.get(class_id, 0), copies)
    return rules


def _parse_class_id_list(
    raw_classes: list[str],
    class_to_id: dict[str, int],
    class_names: list[str],
    *,
    option_name: str,
) -> set[int]:
    class_ids: set[int] = set()
    for raw_class in raw_classes:
        class_key = _normalize_class_key(raw_class)
        if class_key.isdigit():
            class_id = int(class_key)
            if not 0 <= class_id < len(class_names):
                raise ValueError(f"Class id out of range for {option_name}: {class_id}")
        else:
            if class_key not in class_to_id:
                valid = ", ".join(class_names)
                raise ValueError(f"Unknown class for {option_name}: {raw_class}. Valid classes: {valid}")
            class_id = class_to_id[class_key]
        class_ids.add(class_id)
    return class_ids


def _normalize_class_key(value: str) -> str:
    return " ".join(value.strip().lower().replace("_", " ").split())


def _has_small_object(boxes: list[dict[str, float]], threshold: float | None) -> bool:
    if threshold is None:
        return False
    return any(float(box["width"]) * float(box["height"]) <= threshold for box in boxes)


def _class_copy_extra(boxes: list[dict[str, float]], class_copy_rules: dict[int, int]) -> int:
    if not class_copy_rules:
        return 0
    class_ids = {int(box["class_id"]) for box in boxes}
    return sum(copies for class_id, copies in class_copy_rules.items() if class_id in class_ids)


def _split_samples(
    samples: list[dict[str, Any]],
    *,
    split_mode: str,
    val_ratio: float,
    frame_stride: int,
    train_frame_offset: int,
    val_frame_offset: int,
) -> dict[str, list[dict[str, Any]]]:
    if split_mode == "ratio":
        split_index = int(round(len(samples) * (1.0 - val_ratio)))
        split_index = max(0, min(split_index, len(samples)))
        return {
            "train": samples[:split_index],
            "val": samples[split_index:],
        }

    if frame_stride < 1:
        raise ValueError("--frame-stride must be >= 1.")
    if not 0 <= train_frame_offset < frame_stride:
        raise ValueError("--train-frame-offset must be in [0, frame_stride).")
    if not 0 <= val_frame_offset < frame_stride:
        raise ValueError("--val-frame-offset must be in [0, frame_stride).")
    if train_frame_offset == val_frame_offset:
        raise ValueError("--train-frame-offset and --val-frame-offset must be different.")

    return {
        "train": [
            sample for sample in samples if int(sample["frame_index"]) % frame_stride == train_frame_offset
        ],
        "val": [
            sample for sample in samples if int(sample["frame_index"]) % frame_stride == val_frame_offset
        ],
    }


def _expand_training_samples(
    samples: list[dict[str, Any]],
    *,
    extra_hard_copies: int,
    extra_small_object_copies: int,
    max_extra_copies_per_image: int | None,
) -> list[dict[str, Any]]:
    if max_extra_copies_per_image is not None and max_extra_copies_per_image < 0:
        raise ValueError("--max-extra-copies-per-image must be >= 0.")
    expanded = []
    for sample in samples:
        extra_copies = int(sample.get("class_copy_extra", 0))
        if sample.get("hard", False):
            extra_copies += extra_hard_copies
        if sample.get("small_object", False):
            extra_copies += extra_small_object_copies
        if max_extra_copies_per_image is not None:
            extra_copies = min(extra_copies, max_extra_copies_per_image)
        copies = 1 + extra_copies
        for copy_index in range(copies):
            item = dict(sample)
            item["copy_index"] = copy_index
            expanded.append(item)
    return expanded


def _build_object_crop_samples(
    samples: list[dict[str, Any]],
    *,
    class_ids: set[int],
    area_threshold: float,
    padding: float,
    min_size: int,
    max_size: int,
    max_crops_per_image: int,
    min_box_size: int,
) -> list[dict[str, Any]]:
    if not class_ids or max_crops_per_image == 0:
        return []
    if area_threshold < 0.0:
        raise ValueError("--object-crop-area-threshold must be >= 0.")
    if padding < 0.0:
        raise ValueError("--object-crop-padding must be >= 0.")
    if min_size < 1 or max_size < 1 or min_size > max_size:
        raise ValueError("--object-crop-min-size and --object-crop-max-size must be positive and min <= max.")
    if max_crops_per_image < 0:
        raise ValueError("--max-object-crops-per-image must be >= 0.")
    if min_box_size < 1:
        raise ValueError("--object-crop-min-box-size must be >= 1.")

    crop_samples = []
    for sample in samples:
        boxes = list(sample["boxes"])
        candidates = [
            box
            for box in boxes
            if int(box["class_id"]) in class_ids
            and float(box["width"]) * float(box["height"]) <= area_threshold
        ]
        candidates.sort(key=lambda box: float(box["width"]) * float(box["height"]))
        for crop_index, target_box in enumerate(candidates[:max_crops_per_image]):
            crop_rect = _object_crop_rect(
                target_box,
                padding=padding,
                min_size=min_size,
                max_size=max_size,
            )
            if crop_rect is None:
                continue
            crop_boxes = _boxes_for_crop(boxes, crop_rect, min_box_size=min_box_size)
            if not crop_boxes:
                continue
            crop_sample = dict(sample)
            crop_sample["boxes"] = crop_boxes
            crop_sample["copy_index"] = 0
            crop_sample["sample_type"] = "object_crop"
            crop_sample["crop_rect"] = crop_rect
            crop_sample["exported_name"] = _object_crop_image_name(str(sample["frame"].get("name", "")), crop_index)
            crop_sample["small_object"] = True
            crop_sample["class_copy_extra"] = 0
            crop_samples.append(crop_sample)
    return crop_samples


def _build_copy_paste_samples(
    samples: list[dict[str, Any]],
    *,
    class_ids: set[int],
    area_threshold: float,
    source_min_area: float,
    source_min_box_size: int,
    source_max_aspect_ratio: float,
    max_images: int,
    objects_per_image: int,
    context_padding: float,
    scale_min: float,
    scale_max: float,
    max_overlap: float,
    min_box_size: int,
    seed: int,
) -> list[dict[str, Any]]:
    if not class_ids or max_images == 0:
        return []
    if area_threshold < 0.0:
        raise ValueError("--copy-paste-area-threshold must be >= 0.")
    if source_min_area < 0.0:
        raise ValueError("--copy-paste-source-min-area must be >= 0.")
    if source_min_area > area_threshold:
        raise ValueError("--copy-paste-source-min-area must be <= --copy-paste-area-threshold.")
    if source_min_box_size < 1:
        raise ValueError("--copy-paste-source-min-box-size must be >= 1.")
    if source_max_aspect_ratio < 0.0:
        raise ValueError("--copy-paste-source-max-aspect-ratio must be >= 0.")
    if max_images < 0:
        raise ValueError("--copy-paste-max-images must be >= 0.")
    if objects_per_image < 1:
        raise ValueError("--copy-paste-objects-per-image must be >= 1.")
    if context_padding < 0.0:
        raise ValueError("--copy-paste-context-padding must be >= 0.")
    if scale_min <= 0.0 or scale_max <= 0.0 or scale_min > scale_max:
        raise ValueError("--copy-paste-scale-min/max must be positive and min <= max.")
    if not 0.0 <= max_overlap <= 1.0:
        raise ValueError("--copy-paste-max-overlap must be in [0, 1].")
    if min_box_size < 1:
        raise ValueError("--copy-paste-min-box-size must be >= 1.")

    sources = _copy_paste_sources(
        samples,
        class_ids=class_ids,
        area_threshold=area_threshold,
        source_min_area=source_min_area,
        source_min_box_size=source_min_box_size,
        source_max_aspect_ratio=source_max_aspect_ratio,
    )
    if not sources:
        return []

    rng = random.Random(seed)
    target_indices = list(range(len(samples)))
    source_indices = list(range(len(sources)))
    rng.shuffle(target_indices)
    rng.shuffle(source_indices)

    copy_samples = []
    for paste_index in range(max_images):
        target = samples[target_indices[paste_index % len(target_indices)]]
        target_width, target_height = _sample_image_size(target)
        if target_width <= 0 or target_height <= 0:
            continue

        boxes = [dict(box) for box in target["boxes"]]
        paste_objects = []
        for object_index in range(objects_per_image):
            source = sources[source_indices[(paste_index * objects_per_image + object_index) % len(source_indices)]]
            paste_spec = _make_copy_paste_spec(
                source,
                target_width=target_width,
                target_height=target_height,
                existing_boxes=boxes,
                context_padding=context_padding,
                scale_min=scale_min,
                scale_max=scale_max,
                max_overlap=max_overlap,
                min_box_size=min_box_size,
                rng=rng,
            )
            if paste_spec is None:
                continue
            paste_objects.append(paste_spec)
            boxes.append(_abs_box_to_yolo(paste_spec["target_box"], target_width, target_height))

        if not paste_objects:
            continue
        copy_sample = dict(target)
        copy_sample["boxes"] = boxes
        copy_sample["copy_index"] = 0
        copy_sample["sample_type"] = "copy_paste"
        copy_sample["paste_objects"] = paste_objects
        copy_sample["exported_name"] = _copy_paste_image_name(str(target["frame"].get("name", "")), paste_index)
        copy_sample["small_object"] = True
        copy_sample["class_copy_extra"] = 0
        copy_samples.append(copy_sample)
    return copy_samples


def _copy_paste_sources(
    samples: list[dict[str, Any]],
    *,
    class_ids: set[int],
    area_threshold: float,
    source_min_area: float,
    source_min_box_size: int,
    source_max_aspect_ratio: float,
) -> list[dict[str, Any]]:
    sources = []
    for sample in samples:
        image_name = str(sample["frame"].get("name", ""))
        for box in sample["boxes"]:
            if int(box["class_id"]) not in class_ids:
                continue
            box_width = float(box["width"])
            box_height = float(box["height"])
            area = box_width * box_height
            if area > area_threshold or area < source_min_area:
                continue
            image_width = float(box.get("image_width", 0.0))
            image_height = float(box.get("image_height", 0.0))
            abs_width = box_width * image_width
            abs_height = box_height * image_height
            if abs_width < source_min_box_size or abs_height < source_min_box_size:
                continue
            if source_max_aspect_ratio > 0.0:
                aspect_ratio = max(abs_width / max(1.0, abs_height), abs_height / max(1.0, abs_width))
                if aspect_ratio > source_max_aspect_ratio:
                    continue
            if not all(key in box for key in ["x1", "y1", "x2", "y2"]):
                continue
            sources.append({"image_name": image_name, "box": dict(box)})
    sources.sort(key=lambda source: float(source["box"]["width"]) * float(source["box"]["height"]))
    return sources


def _sample_image_size(sample: dict[str, Any]) -> tuple[int, int]:
    boxes = sample.get("boxes", [])
    if boxes:
        return int(round(float(boxes[0].get("image_width", 0)))), int(round(float(boxes[0].get("image_height", 0))))
    size = sample.get("frame", {}).get("size", {})
    return int(size.get("width", 0) or 0), int(size.get("height", 0) or 0)


def _make_copy_paste_spec(
    source: dict[str, Any],
    *,
    target_width: int,
    target_height: int,
    existing_boxes: list[dict[str, float]],
    context_padding: float,
    scale_min: float,
    scale_max: float,
    max_overlap: float,
    min_box_size: int,
    rng: random.Random,
) -> dict[str, Any] | None:
    source_box = source["box"]
    source_width = int(round(float(source_box.get("image_width", 0))))
    source_height = int(round(float(source_box.get("image_height", 0))))
    if source_width <= 0 or source_height <= 0:
        return None

    source_rect = _padded_source_rect(source_box, context_padding, source_width, source_height)
    patch_width = source_rect["x2"] - source_rect["x1"]
    patch_height = source_rect["y2"] - source_rect["y1"]
    if patch_width <= 1 or patch_height <= 1:
        return None

    source_obj_offset_x1 = float(source_box["x1"]) - float(source_rect["x1"])
    source_obj_offset_y1 = float(source_box["y1"]) - float(source_rect["y1"])
    source_obj_offset_x2 = float(source_box["x2"]) - float(source_rect["x1"])
    source_obj_offset_y2 = float(source_box["y2"]) - float(source_rect["y1"])
    source_center_x_ratio = ((float(source_box["x1"]) + float(source_box["x2"])) * 0.5) / max(1.0, float(source_width))
    source_center_y_ratio = ((float(source_box["y1"]) + float(source_box["y2"])) * 0.5) / max(1.0, float(source_height))
    existing_abs_boxes = [_yolo_box_to_abs(box) for box in existing_boxes]

    for _attempt in range(24):
        scale = rng.uniform(scale_min, scale_max)
        target_patch_width = max(1, int(round(patch_width * scale)))
        target_patch_height = max(1, int(round(patch_height * scale)))
        if target_patch_width >= target_width or target_patch_height >= target_height:
            fit_scale = min(
                (target_width - 1) / max(1.0, float(target_patch_width)),
                (target_height - 1) / max(1.0, float(target_patch_height)),
            )
            target_patch_width = max(1, int(round(target_patch_width * fit_scale)))
            target_patch_height = max(1, int(round(target_patch_height * fit_scale)))
            scale *= fit_scale

        center_x = source_center_x_ratio * target_width + (rng.random() - 0.5) * 0.45 * target_width
        center_y = source_center_y_ratio * target_height + (rng.random() - 0.5) * 0.25 * target_height
        left = int(round(_clamp(center_x - target_patch_width * 0.5, 0.0, float(target_width - target_patch_width))))
        top = int(round(_clamp(center_y - target_patch_height * 0.5, 0.0, float(target_height - target_patch_height))))

        target_box = {
            "class_id": int(source_box["class_id"]),
            "x1": left + source_obj_offset_x1 * scale,
            "y1": top + source_obj_offset_y1 * scale,
            "x2": left + source_obj_offset_x2 * scale,
            "y2": top + source_obj_offset_y2 * scale,
        }
        target_box["x1"] = _clamp(float(target_box["x1"]), 0.0, float(target_width))
        target_box["y1"] = _clamp(float(target_box["y1"]), 0.0, float(target_height))
        target_box["x2"] = _clamp(float(target_box["x2"]), 0.0, float(target_width))
        target_box["y2"] = _clamp(float(target_box["y2"]), 0.0, float(target_height))
        if target_box["x2"] - target_box["x1"] < min_box_size or target_box["y2"] - target_box["y1"] < min_box_size:
            continue
        if existing_abs_boxes and max(_box_iou_abs(target_box, existing) for existing in existing_abs_boxes) > max_overlap:
            continue
        return {
            "source_image": source["image_name"],
            "source_rect": source_rect,
            "source_object_rect": {
                "x1": source_obj_offset_x1,
                "y1": source_obj_offset_y1,
                "x2": source_obj_offset_x2,
                "y2": source_obj_offset_y2,
            },
            "target_rect": {
                "x1": left,
                "y1": top,
                "x2": left + target_patch_width,
                "y2": top + target_patch_height,
            },
            "target_box": target_box,
        }
    return None


def _padded_source_rect(
    box: dict[str, float],
    padding: float,
    image_width: int,
    image_height: int,
) -> dict[str, int]:
    x1 = float(box["x1"])
    y1 = float(box["y1"])
    x2 = float(box["x2"])
    y2 = float(box["y2"])
    box_width = max(1.0, x2 - x1)
    box_height = max(1.0, y2 - y1)
    return {
        "x1": int(round(_clamp(x1 - box_width * padding, 0.0, float(image_width)))),
        "y1": int(round(_clamp(y1 - box_height * padding, 0.0, float(image_height)))),
        "x2": int(round(_clamp(x2 + box_width * padding, 0.0, float(image_width)))),
        "y2": int(round(_clamp(y2 + box_height * padding, 0.0, float(image_height)))),
    }


def _yolo_box_to_abs(box: dict[str, float]) -> dict[str, float]:
    if all(key in box for key in ["x1", "y1", "x2", "y2"]):
        return {
            "x1": float(box["x1"]),
            "y1": float(box["y1"]),
            "x2": float(box["x2"]),
            "y2": float(box["y2"]),
        }
    image_width = float(box.get("image_width", 0.0))
    image_height = float(box.get("image_height", 0.0))
    width = float(box["width"]) * image_width
    height = float(box["height"]) * image_height
    center_x = float(box["x_center"]) * image_width
    center_y = float(box["y_center"]) * image_height
    return {
        "x1": center_x - width * 0.5,
        "y1": center_y - height * 0.5,
        "x2": center_x + width * 0.5,
        "y2": center_y + height * 0.5,
    }


def _abs_box_to_yolo(box: dict[str, Any], image_width: int, image_height: int) -> dict[str, float]:
    x1 = _clamp(float(box["x1"]), 0.0, float(image_width))
    y1 = _clamp(float(box["y1"]), 0.0, float(image_height))
    x2 = _clamp(float(box["x2"]), 0.0, float(image_width))
    y2 = _clamp(float(box["y2"]), 0.0, float(image_height))
    box_width = max(0.0, x2 - x1)
    box_height = max(0.0, y2 - y1)
    return {
        "class_id": int(box["class_id"]),
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "image_width": float(image_width),
        "image_height": float(image_height),
        "x_center": (x1 + x2) * 0.5 / image_width,
        "y_center": (y1 + y2) * 0.5 / image_height,
        "width": box_width / image_width,
        "height": box_height / image_height,
    }


def _box_iou_abs(a: dict[str, Any], b: dict[str, Any]) -> float:
    x1 = max(float(a["x1"]), float(b["x1"]))
    y1 = max(float(a["y1"]), float(b["y1"]))
    x2 = min(float(a["x2"]), float(b["x2"]))
    y2 = min(float(a["y2"]), float(b["y2"]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if intersection <= 0.0:
        return 0.0
    area_a = max(0.0, float(a["x2"]) - float(a["x1"])) * max(0.0, float(a["y2"]) - float(a["y1"]))
    area_b = max(0.0, float(b["x2"]) - float(b["x1"])) * max(0.0, float(b["y2"]) - float(b["y1"]))
    return intersection / max(1e-9, area_a + area_b - intersection)


def _object_crop_rect(
    target_box: dict[str, float],
    *,
    padding: float,
    min_size: int,
    max_size: int,
) -> dict[str, int] | None:
    image_width = int(round(float(target_box.get("image_width", 0))))
    image_height = int(round(float(target_box.get("image_height", 0))))
    if image_width <= 0 or image_height <= 0:
        return None

    x1 = float(target_box["x1"])
    y1 = float(target_box["y1"])
    x2 = float(target_box["x2"])
    y2 = float(target_box["y2"])
    box_width = max(1.0, x2 - x1)
    box_height = max(1.0, y2 - y1)
    crop_side = int(round(max(box_width * (1.0 + 2.0 * padding), box_height * (1.0 + 2.0 * padding), min_size)))
    crop_side = max(1, min(crop_side, max_size))
    crop_width = min(crop_side, image_width)
    crop_height = min(crop_side, image_height)
    center_x = (x1 + x2) * 0.5
    center_y = (y1 + y2) * 0.5
    left = int(round(_clamp(center_x - crop_width * 0.5, 0.0, float(image_width - crop_width))))
    top = int(round(_clamp(center_y - crop_height * 0.5, 0.0, float(image_height - crop_height))))
    return {
        "x1": left,
        "y1": top,
        "x2": left + crop_width,
        "y2": top + crop_height,
    }


def _boxes_for_crop(
    boxes: list[dict[str, float]],
    crop_rect: dict[str, int],
    *,
    min_box_size: int,
) -> list[dict[str, float]]:
    crop_x1 = float(crop_rect["x1"])
    crop_y1 = float(crop_rect["y1"])
    crop_x2 = float(crop_rect["x2"])
    crop_y2 = float(crop_rect["y2"])
    crop_width = crop_x2 - crop_x1
    crop_height = crop_y2 - crop_y1
    if crop_width <= 1.0 or crop_height <= 1.0:
        return []

    crop_boxes = []
    for box in boxes:
        center_x = (float(box["x1"]) + float(box["x2"])) * 0.5
        center_y = (float(box["y1"]) + float(box["y2"])) * 0.5
        if not (crop_x1 <= center_x < crop_x2 and crop_y1 <= center_y < crop_y2):
            continue
        clipped_x1 = _clamp(float(box["x1"]), crop_x1, crop_x2)
        clipped_y1 = _clamp(float(box["y1"]), crop_y1, crop_y2)
        clipped_x2 = _clamp(float(box["x2"]), crop_x1, crop_x2)
        clipped_y2 = _clamp(float(box["y2"]), crop_y1, crop_y2)
        box_width = clipped_x2 - clipped_x1
        box_height = clipped_y2 - clipped_y1
        if box_width < min_box_size or box_height < min_box_size:
            continue
        rel_x1 = clipped_x1 - crop_x1
        rel_y1 = clipped_y1 - crop_y1
        rel_x2 = clipped_x2 - crop_x1
        rel_y2 = clipped_y2 - crop_y1
        crop_boxes.append(
            {
                "class_id": int(box["class_id"]),
                "x1": rel_x1,
                "y1": rel_y1,
                "x2": rel_x2,
                "y2": rel_y2,
                "image_width": crop_width,
                "image_height": crop_height,
                "x_center": (rel_x1 + rel_x2) * 0.5 / crop_width,
                "y_center": (rel_y1 + rel_y2) * 0.5 / crop_height,
                "width": box_width / crop_width,
                "height": box_height / crop_height,
            }
        )
    return crop_boxes


def _object_crop_image_name(image_name: str, crop_index: int) -> str:
    path = Path(image_name)
    suffix = path.suffix or ".jpg"
    return f"{path.stem}__crop{crop_index:02d}{suffix}"


def _copy_paste_image_name(image_name: str, paste_index: int) -> str:
    path = Path(image_name)
    suffix = path.suffix or ".jpg"
    return f"{path.stem}__paste{paste_index:04d}{suffix}"


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def _export_image_name(image_name: str, copy_index: int) -> str:
    if copy_index <= 0:
        return image_name
    path = Path(image_name)
    suffix = path.suffix or ".jpg"
    return f"{path.stem}__copy{copy_index:02d}{suffix}"


def _load_frames(labels_path: Path) -> list[dict[str, Any]]:
    with labels_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict) and "frames" in payload:
        return list(payload["frames"])
    if isinstance(payload, list):
        return list(payload)
    raise ValueError("Expected a list of frames or an object with a 'frames' field.")


def _frame_boxes(
    frame: dict[str, Any],
    class_to_id: dict[str, int],
    images_root: Path,
    fallback_size: tuple[int, int],
    read_image_size: bool,
) -> list[dict[str, float]]:
    image_name = str(frame.get("name", ""))
    width, height = _frame_size(frame, images_root / image_name, fallback_size, read_image_size)
    if width <= 0 or height <= 0:
        return []

    boxes = []
    for label in frame.get("labels", []):
        category = _normalize_class_key(str(label.get("category", "")))
        if category not in class_to_id or "box2d" not in label:
            continue
        box = _normalize_box(label["box2d"], width, height)
        if box is None:
            continue
        boxes.append({"class_id": class_to_id[category], **box})
    return boxes


def _frame_size(
    frame: dict[str, Any],
    image_path: Path,
    fallback_size: tuple[int, int],
    read_image_size: bool,
) -> tuple[int, int]:
    if read_image_size:
        return _image_size(image_path)
    size = frame.get("size", {})
    width = int(size.get("width", 0) or 0)
    height = int(size.get("height", 0) or 0)
    if width > 0 and height > 0:
        return width, height
    return fallback_size


def _image_size(path: Path) -> tuple[int, int]:
    image = cv2.imread(str(path))
    if image is None:
        return 0, 0
    height, width = image.shape[:2]
    return width, height


def _normalize_box(box2d: dict[str, Any], width: int, height: int) -> dict[str, float] | None:
    x1 = float(box2d["x1"])
    y1 = float(box2d["y1"])
    x2 = float(box2d["x2"])
    y2 = float(box2d["y2"])
    x1 = min(max(x1, 0.0), float(width))
    y1 = min(max(y1, 0.0), float(height))
    x2 = min(max(x2, 0.0), float(width))
    y2 = min(max(y2, 0.0), float(height))
    box_width = x2 - x1
    box_height = y2 - y1
    if box_width <= 1.0 or box_height <= 1.0:
        return None
    return {
        "x_center": (x1 + x2) / 2.0 / width,
        "y_center": (y1 + y2) / 2.0 / height,
        "width": box_width / width,
        "height": box_height / height,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "image_width": float(width),
        "image_height": float(height),
    }


def _link_or_copy(source: Path, target: Path, copy: bool) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        return
    if copy:
        shutil.copy2(source, target)
        return
    target.symlink_to(source.resolve())


def _write_crop_image(source: Path, target: Path, crop_rect: dict[str, int]) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    image = cv2.imread(str(source))
    if image is None:
        raise FileNotFoundError(f"Could not read source image for crop: {source}")
    crop = image[int(crop_rect["y1"]) : int(crop_rect["y2"]), int(crop_rect["x1"]) : int(crop_rect["x2"])]
    if crop.size == 0:
        raise ValueError(f"Empty crop for {source}: {crop_rect}")
    if not cv2.imwrite(str(target), crop):
        raise OSError(f"Could not write crop image: {target}")


def _write_copy_paste_image(
    images_root: Path,
    base_image_path: Path,
    target: Path,
    paste_objects: list[dict[str, Any]],
    *,
    blend_mode: str,
    mask_mode: str,
    feather_ratio: float,
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    if feather_ratio < 0.0:
        raise ValueError("--copy-paste-feather-ratio must be >= 0.")
    image = cv2.imread(str(base_image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read base image for copy-paste: {base_image_path}")

    for paste in paste_objects:
        source_image = cv2.imread(str(images_root / str(paste["source_image"])))
        if source_image is None:
            raise FileNotFoundError(f"Could not read source image for copy-paste: {paste['source_image']}")
        source_rect = paste["source_rect"]
        target_rect = paste["target_rect"]
        patch = source_image[
            int(source_rect["y1"]) : int(source_rect["y2"]),
            int(source_rect["x1"]) : int(source_rect["x2"]),
        ]
        if patch.size == 0:
            continue
        source_patch_height, source_patch_width = patch.shape[:2]
        target_width = int(target_rect["x2"]) - int(target_rect["x1"])
        target_height = int(target_rect["y2"]) - int(target_rect["y1"])
        if target_width <= 0 or target_height <= 0:
            continue
        patch = cv2.resize(patch, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        alpha_mask = _copy_paste_alpha_mask(
            patch=patch,
            object_rect=_scaled_rect(
                paste.get("source_object_rect", {}),
                x_scale=target_width / max(1.0, float(source_patch_width)),
                y_scale=target_height / max(1.0, float(source_patch_height)),
            ),
            mask_mode=mask_mode,
            feather_ratio=feather_ratio,
        )
        y1 = int(target_rect["y1"])
        y2 = int(target_rect["y2"])
        x1 = int(target_rect["x1"])
        x2 = int(target_rect["x2"])
        target_slice = image[y1:y2, x1:x2]
        if target_slice.shape[:2] != patch.shape[:2]:
            continue
        image[y1:y2, x1:x2] = _blend_copy_paste_patch(
            base=target_slice,
            patch=patch,
            blend_mode=blend_mode,
            alpha_mask=alpha_mask,
            feather_ratio=feather_ratio,
        )

    if not cv2.imwrite(str(target), image):
        raise OSError(f"Could not write copy-paste image: {target}")


def _blend_copy_paste_patch(
    *,
    base: np.ndarray,
    patch: np.ndarray,
    blend_mode: str,
    alpha_mask: np.ndarray | None,
    feather_ratio: float,
) -> np.ndarray:
    if blend_mode == "none" and alpha_mask is None:
        return patch
    if blend_mode not in {"none", "feather"}:
        raise ValueError(f"Unknown copy-paste blend mode: {blend_mode}")

    height, width = patch.shape[:2]
    if blend_mode == "feather":
        alpha = _edge_alpha(width=width, height=height, feather_ratio=feather_ratio)
    else:
        alpha = np.ones((height, width), dtype=np.float32)
    if alpha_mask is not None:
        alpha = alpha * alpha_mask.astype(np.float32)
    alpha = np.clip(alpha, 0.0, 1.0).astype(np.float32)
    if patch.ndim == 3:
        alpha = alpha[:, :, None]

    blended = patch.astype(np.float32) * alpha + base.astype(np.float32) * (1.0 - alpha)
    return np.clip(blended, 0, 255).astype(np.uint8)


def _copy_paste_alpha_mask(
    *,
    patch: np.ndarray,
    object_rect: dict[str, float],
    mask_mode: str,
    feather_ratio: float,
) -> np.ndarray | None:
    if mask_mode == "none":
        return None
    if mask_mode not in {"box", "grabcut"}:
        raise ValueError(f"Unknown copy-paste mask mode: {mask_mode}")

    height, width = patch.shape[:2]
    rect = _clamped_int_rect(object_rect, width=width, height=height)
    if rect is None:
        return None
    if mask_mode == "box":
        return _soft_box_mask(width=width, height=height, rect=rect, feather_ratio=feather_ratio)

    try:
        return _grabcut_mask(patch=patch, rect=rect, feather_ratio=feather_ratio)
    except cv2.error:
        return _soft_box_mask(width=width, height=height, rect=rect, feather_ratio=feather_ratio)


def _scaled_rect(rect: dict[str, Any], *, x_scale: float, y_scale: float) -> dict[str, float]:
    return {
        "x1": float(rect.get("x1", 0.0)) * x_scale,
        "y1": float(rect.get("y1", 0.0)) * y_scale,
        "x2": float(rect.get("x2", 0.0)) * x_scale,
        "y2": float(rect.get("y2", 0.0)) * y_scale,
    }


def _clamped_int_rect(rect: dict[str, Any], *, width: int, height: int) -> dict[str, int] | None:
    x1 = int(round(_clamp(float(rect.get("x1", 0.0)), 0.0, float(width))))
    y1 = int(round(_clamp(float(rect.get("y1", 0.0)), 0.0, float(height))))
    x2 = int(round(_clamp(float(rect.get("x2", 0.0)), 0.0, float(width))))
    y2 = int(round(_clamp(float(rect.get("y2", 0.0)), 0.0, float(height))))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


def _soft_box_mask(
    *,
    width: int,
    height: int,
    rect: dict[str, int],
    feather_ratio: float,
) -> np.ndarray:
    alpha = np.zeros((height, width), dtype=np.float32)
    alpha[rect["y1"] : rect["y2"], rect["x1"] : rect["x2"]] = 1.0
    feather_pixels = int(round(min(width, height) * feather_ratio))
    if feather_pixels > 0:
        alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=max(1.0, feather_pixels / 3.0))
    return np.clip(alpha, 0.0, 1.0).astype(np.float32)


def _grabcut_mask(*, patch: np.ndarray, rect: dict[str, int], feather_ratio: float) -> np.ndarray:
    height, width = patch.shape[:2]
    rect_width = rect["x2"] - rect["x1"]
    rect_height = rect["y2"] - rect["y1"]
    if rect_width < 4 or rect_height < 4:
        return _soft_box_mask(width=width, height=height, rect=rect, feather_ratio=feather_ratio)

    grabcut_mask = np.zeros((height, width), dtype=np.uint8)
    bgd_model = np.zeros((1, 65), dtype=np.float64)
    fgd_model = np.zeros((1, 65), dtype=np.float64)
    cv2.grabCut(
        patch,
        grabcut_mask,
        (rect["x1"], rect["y1"], rect_width, rect_height),
        bgd_model,
        fgd_model,
        2,
        cv2.GC_INIT_WITH_RECT,
    )
    alpha = np.where(
        (grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD),
        1.0,
        0.0,
    ).astype(np.float32)
    foreground_pixels = float(alpha.sum())
    rect_area = float(rect_width * rect_height)
    if foreground_pixels < rect_area * 0.05:
        return _soft_box_mask(width=width, height=height, rect=rect, feather_ratio=feather_ratio)
    feather_pixels = int(round(min(width, height) * feather_ratio))
    if feather_pixels > 0:
        alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=max(1.0, feather_pixels / 3.0))
    return np.clip(alpha, 0.0, 1.0).astype(np.float32)


def _edge_alpha(*, width: int, height: int, feather_ratio: float) -> np.ndarray:
    feather_pixels = int(round(min(width, height) * feather_ratio))
    if feather_pixels <= 0:
        return np.ones((height, width), dtype=np.float32)

    x_distance = np.minimum(np.arange(width), np.arange(width)[::-1]).astype(np.float32)
    y_distance = np.minimum(np.arange(height), np.arange(height)[::-1]).astype(np.float32)
    edge_distance = np.minimum(y_distance[:, None], x_distance[None, :])
    alpha = np.clip(edge_distance / float(feather_pixels), 0.0, 1.0)
    alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=max(1.0, feather_pixels / 3.0))
    return np.clip(alpha, 0.0, 1.0).astype(np.float32)


def _yolo_label_text(boxes: list[dict[str, float]]) -> str:
    lines = [
        (
            f"{int(box['class_id'])} "
            f"{box['x_center']:.6f} {box['y_center']:.6f} "
            f"{box['width']:.6f} {box['height']:.6f}"
        )
        for box in boxes
    ]
    return "\n".join(lines) + ("\n" if lines else "")


def _write_dataset_yaml(output_dir: Path, class_names: list[str]) -> None:
    payload = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {index: name for index, name in enumerate(class_names)},
    }
    with (output_dir / "dataset.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _print_report(stats: dict[str, Any]) -> None:
    print("YOLO export summary")
    print(f"- output_dir: {stats['output_dir']}")
    print(f"- classes: {', '.join(stats['classes'])}")
    for split, split_stats in stats["splits"].items():
        print(
            f"- {split}: images={split_stats['images']} "
            f"unique={split_stats['unique_source_frames']} "
            f"duplicates={split_stats['duplicate_images']} "
            f"crops={split_stats['object_crop_images']} "
            f"copy_paste={split_stats['copy_paste_images']} "
            f"hard={split_stats['hard_images']} "
            f"small={split_stats['small_object_images']} "
            f"class_boost={split_stats['class_boost_images']} "
            f"labels={split_stats['labels']}"
        )
        for name, count in split_stats["class_counts"].items():
            print(f"  - {name}: {count}")
    print(f"- dataset_yaml: {Path(stats['output_dir']) / 'dataset.yaml'}")


if __name__ == "__main__":
    raise SystemExit(main())
