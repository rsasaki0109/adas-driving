#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any

import cv2
import numpy as np


BOX_COLORS = {
    "gt": (40, 220, 40),
    "pred": (255, 180, 40),
    "fp": (40, 40, 255),
    "fn": (255, 90, 60),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render BDD100K TP/FP/FN samples saved by evaluate_bdd100k.py.")
    parser.add_argument("--images-root", required=True, help="Directory containing BDD100K images.")
    parser.add_argument("--errors", required=True, help="Error JSON produced by evaluate_bdd100k.py --save-errors.")
    parser.add_argument("--labels", default=None, help="Optional BDD100K/Scalabel label JSON for --where filters.")
    parser.add_argument("--output-dir", default="outputs/bdd100k_error_gallery", help="Directory for rendered samples.")
    parser.add_argument("--kinds", nargs="*", default=None, help="Kinds to render. Defaults to all kinds in the file.")
    parser.add_argument("--buckets", nargs="+", default=["fp", "fn", "tp"], choices=["tp", "fp", "fn"])
    parser.add_argument(
        "--where",
        nargs="*",
        default=[],
        metavar="ATTR=VALUE",
        help="Optional frame attribute filters such as timeofday=night or scene=highway. Requires --labels.",
    )
    parser.add_argument("--max-per-bucket", type=int, default=12, help="Maximum samples per kind/bucket.")
    parser.add_argument("--crop-padding", type=int, default=80, help="Pixels around the error box in crop images.")
    parser.add_argument("--contact-width", type=int, default=320, help="Cell width for contact sheets.")
    parser.add_argument("--contact-cols", type=int, default=4, help="Contact sheet columns.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    images_root = Path(args.images_root)
    errors_path = Path(args.errors)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with errors_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    errors = payload.get("errors", {})
    kinds = args.kinds or sorted(errors)
    filters = _parse_filters(args.where)
    if filters and not args.labels:
        raise ValueError("--where requires --labels.")
    frame_attributes = _load_frame_attributes(Path(args.labels), filters) if filters else {}

    rendered: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for kind in kinds:
        if kind not in errors:
            continue
        rendered[kind] = {}
        for bucket in args.buckets:
            samples = _filter_samples(
                list(errors[kind].get(bucket, [])),
                frame_attributes=frame_attributes,
                filters=filters,
            )[: args.max_per_bucket]
            rendered[kind][bucket] = _render_bucket(
                images_root=images_root,
                output_dir=output_dir,
                kind=kind,
                bucket=bucket,
                samples=samples,
                crop_padding=args.crop_padding,
                contact_width=args.contact_width,
                contact_cols=max(1, args.contact_cols),
            )

    _write_index(output_dir / "index.md", errors_path, rendered)
    print(f"Saved error gallery: {output_dir}")
    print(f"Saved index: {output_dir / 'index.md'}")
    return 0


def _parse_filters(raw_filters: list[str]) -> dict[str, str]:
    filters: dict[str, str] = {}
    for raw_filter in raw_filters:
        if "=" not in raw_filter:
            raise ValueError(f"--where must use ATTR=VALUE, got: {raw_filter}")
        attr, value = raw_filter.split("=", 1)
        attr = attr.strip()
        value = value.strip()
        if not attr or not value:
            raise ValueError(f"--where must use non-empty ATTR=VALUE, got: {raw_filter}")
        filters[attr] = value
    return filters


def _load_frame_attributes(labels_path: Path, filters: dict[str, str]) -> dict[str, dict[str, str]]:
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
            for attr in filters
        }
    return attributes_by_frame


def _filter_samples(
    samples: list[dict[str, Any]],
    *,
    frame_attributes: dict[str, dict[str, str]],
    filters: dict[str, str],
) -> list[dict[str, Any]]:
    if not filters:
        return samples
    filtered = []
    for sample in samples:
        attrs = frame_attributes.get(str(sample.get("frame", "")), {})
        if all(attrs.get(attr) == value for attr, value in filters.items()):
            filtered.append(sample)
    return filtered


def _render_bucket(
    *,
    images_root: Path,
    output_dir: Path,
    kind: str,
    bucket: str,
    samples: list[dict[str, Any]],
    crop_padding: int,
    contact_width: int,
    contact_cols: int,
) -> list[dict[str, Any]]:
    rendered = []
    crop_images = []
    bucket_dir = output_dir / _safe_name(kind) / bucket
    bucket_dir.mkdir(parents=True, exist_ok=True)

    for index, sample in enumerate(samples, start=1):
        frame = str(sample.get("frame", ""))
        image_path = images_root / frame
        image = cv2.imread(str(image_path))
        if image is None:
            rendered.append({"frame": frame, "missing": True})
            continue

        boxes = _sample_boxes(sample, bucket)
        full = image.copy()
        title = f"{kind} {bucket.upper()} {frame}"
        _put_label(full, title, (12, 28), (0, 0, 0))
        for box_kind, label, box in boxes:
            _draw_box(full, box, label, BOX_COLORS[box_kind])

        crop, crop_offset = _crop_around_boxes(image, [box for _, _, box in boxes], crop_padding)
        for box_kind, label, box in boxes:
            _draw_box(crop, _offset_box(box, crop_offset), label, BOX_COLORS[box_kind])

        stem = f"{index:03d}_{_safe_name(Path(frame).stem)}"
        full_path = bucket_dir / f"{stem}_full.jpg"
        crop_path = bucket_dir / f"{stem}_crop.jpg"
        cv2.imwrite(str(full_path), full)
        cv2.imwrite(str(crop_path), crop)
        crop_images.append((crop, title))
        rendered.append(
            {
                "frame": frame,
                "bucket": bucket,
                "full": str(full_path.relative_to(output_dir)),
                "crop": str(crop_path.relative_to(output_dir)),
                "gt_label": sample.get("gt_label"),
                "pred_label": sample.get("pred_label"),
                "confidence": sample.get("confidence"),
                "iou": sample.get("iou"),
            }
        )

    if crop_images:
        contact = _contact_sheet(crop_images, cell_width=contact_width, cols=contact_cols)
        contact_path = bucket_dir / "contact.jpg"
        cv2.imwrite(str(contact_path), contact)
        rendered.insert(0, {"contact": str(contact_path.relative_to(output_dir))})
    return rendered


def _sample_boxes(sample: dict[str, Any], bucket: str) -> list[tuple[str, str, dict[str, int]]]:
    boxes = []
    gt_box = sample.get("gt_box")
    pred_box = sample.get("pred_box")
    if bucket == "fp" and pred_box:
        label = _sample_label("fp", sample.get("pred_label"), sample.get("confidence"), None)
        boxes.append(("fp", label, _box(pred_box)))
        return boxes
    if bucket == "fn" and gt_box:
        label = _sample_label("fn", sample.get("gt_label"), None, None)
        boxes.append(("fn", label, _box(gt_box)))
        return boxes
    if gt_box:
        label = _sample_label("gt", sample.get("gt_label"), None, sample.get("iou"))
        boxes.append(("gt", label, _box(gt_box)))
    if pred_box:
        label = _sample_label("pred", sample.get("pred_label"), sample.get("confidence"), sample.get("iou"))
        boxes.append(("pred", label, _box(pred_box)))
    return boxes


def _sample_label(prefix: str, label: Any, confidence: Any, iou: Any) -> str:
    parts = [prefix]
    if label:
        parts.append(str(label))
    if confidence is not None:
        parts.append(f"{float(confidence):.2f}")
    if iou is not None:
        parts.append(f"IoU {float(iou):.2f}")
    return " ".join(parts)


def _box(raw: dict[str, Any]) -> dict[str, int]:
    return {
        "x1": int(raw["x1"]),
        "y1": int(raw["y1"]),
        "x2": int(raw["x2"]),
        "y2": int(raw["y2"]),
    }


def _draw_box(image: np.ndarray, box: dict[str, int], label: str, color: tuple[int, int, int]) -> None:
    h, w = image.shape[:2]
    x1 = max(0, min(w - 1, int(box["x1"])))
    y1 = max(0, min(h - 1, int(box["y1"])))
    x2 = max(0, min(w - 1, int(box["x2"])))
    y2 = max(0, min(h - 1, int(box["y2"])))
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    _put_label(image, label, (x1, max(18, y1 - 6)), color)


def _put_label(image: np.ndarray, text: str, origin: tuple[int, int], color: tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.52
    thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = origin
    x = max(0, min(image.shape[1] - text_width - 8, x))
    y = max(text_height + 6, min(image.shape[0] - baseline - 4, y))
    cv2.rectangle(
        image,
        (x, y - text_height - baseline - 6),
        (x + text_width + 8, y + baseline + 4),
        color,
        -1,
    )
    cv2.putText(image, text, (x + 4, y - 4), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def _crop_around_boxes(
    image: np.ndarray,
    boxes: list[dict[str, int]],
    padding: int,
) -> tuple[np.ndarray, tuple[int, int]]:
    if not boxes:
        return image.copy(), (0, 0)
    h, w = image.shape[:2]
    x1 = max(0, min(box["x1"] for box in boxes) - padding)
    y1 = max(0, min(box["y1"] for box in boxes) - padding)
    x2 = min(w, max(box["x2"] for box in boxes) + padding)
    y2 = min(h, max(box["y2"] for box in boxes) + padding)
    if x2 <= x1 or y2 <= y1:
        return image.copy(), (0, 0)
    return image[y1:y2, x1:x2].copy(), (x1, y1)


def _offset_box(box: dict[str, int], offset: tuple[int, int]) -> dict[str, int]:
    ox, oy = offset
    return {
        "x1": box["x1"] - ox,
        "y1": box["y1"] - oy,
        "x2": box["x2"] - ox,
        "y2": box["y2"] - oy,
    }


def _contact_sheet(
    images: list[tuple[np.ndarray, str]],
    *,
    cell_width: int,
    cols: int,
) -> np.ndarray:
    cells = []
    cell_height = int(cell_width * 0.75)
    for image, title in images:
        resized = _letterbox(image, cell_width, cell_height)
        _put_label(resized, title[:52], (8, 22), (0, 0, 0))
        cells.append(resized)

    rows = int(np.ceil(len(cells) / cols))
    sheet = np.full((rows * cell_height, cols * cell_width, 3), 245, dtype=np.uint8)
    for index, cell in enumerate(cells):
        row = index // cols
        col = index % cols
        y = row * cell_height
        x = col * cell_width
        sheet[y : y + cell_height, x : x + cell_width] = cell
    return sheet


def _letterbox(image: np.ndarray, width: int, height: int) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(width / max(w, 1), height / max(h, 1))
    resized_w = max(1, int(round(w * scale)))
    resized_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((height, width, 3), 235, dtype=np.uint8)
    x = (width - resized_w) // 2
    y = (height - resized_h) // 2
    canvas[y : y + resized_h, x : x + resized_w] = resized
    return canvas


def _write_index(path: Path, errors_path: Path, rendered: dict[str, dict[str, list[dict[str, Any]]]]) -> None:
    lines = [
        "# BDD100K Error Gallery",
        "",
        f"- Source: `{errors_path}`",
        "",
        "Legend: `gt` is ground truth, `pred` is matched prediction, `fp` is false positive, `fn` is false negative.",
        "",
    ]
    for kind, buckets in rendered.items():
        lines.extend([f"## {kind}", ""])
        for bucket, items in buckets.items():
            if not items:
                continue
            lines.extend([f"### {bucket.upper()}", ""])
            contact = next((item["contact"] for item in items if "contact" in item), None)
            if contact:
                lines.extend([f"![{kind} {bucket}]({contact})", ""])
            lines.extend(["| frame | crop | full | labels |", "| --- | --- | --- | --- |"])
            for item in items:
                if "contact" in item:
                    continue
                if item.get("missing"):
                    lines.append(f"| {item['frame']} | missing image |  |  |")
                    continue
                label = _index_label(item)
                lines.append(
                    f"| `{item['frame']}` | [crop]({item['crop']}) | [full]({item['full']}) | {label} |"
                )
            lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _index_label(item: dict[str, Any]) -> str:
    parts = []
    if item.get("gt_label"):
        parts.append(f"gt={item['gt_label']}")
    if item.get("pred_label"):
        parts.append(f"pred={item['pred_label']}")
    if item.get("confidence") is not None:
        parts.append(f"conf={float(item['confidence']):.2f}")
    if item.get("iou") is not None:
        parts.append(f"IoU={float(item['iou']):.2f}")
    return ", ".join(parts)


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "item"


if __name__ == "__main__":
    raise SystemExit(main())
