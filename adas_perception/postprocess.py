from __future__ import annotations

from typing import Any

from adas_perception.types import Box, Detection

DEFAULT_SIZE_BUCKET_AREAS = (0.0005, 0.0025, 0.01)


def build_post_process_config(raw: dict[str, Any] | None) -> dict[str, Any]:
    raw = raw or {}
    nms_raw = raw.get("nms", {}) or {}
    kind_iou_raw = nms_raw.get("iou_by_kind") or raw.get("nms_iou_by_kind") or {}
    min_area_raw = raw.get("min_area_ratio_by_kind") or {}
    max_per_kind_raw = raw.get("max_detections_by_kind") or {}
    return {
        "enabled": bool(raw.get("enabled", False)),
        "nms": {
            "enabled": bool(nms_raw.get("enabled", raw.get("nms_enabled", False))),
            "default_iou": float(nms_raw.get("default_iou", raw.get("nms_default_iou", 0.50))),
            "iou_by_kind": {str(kind): float(value) for kind, value in kind_iou_raw.items()},
        },
        "min_area_ratio_by_kind": {str(kind): float(value) for kind, value in min_area_raw.items()},
        "max_detections_by_kind": {str(kind): int(value) for kind, value in max_per_kind_raw.items()},
    }


def area_ratio(box: Box, image_shape: tuple[int, int]) -> float:
    height, width = image_shape
    image_area = max(int(height) * int(width), 1)
    return max(box.area, 0) / float(image_area)


def apply_postprocess(
    detections: list[Detection],
    config: dict[str, Any],
    image_shape: tuple[int, int],
) -> list[Detection]:
    if not config.get("enabled", False):
        return detections

    filtered = detections
    min_area = config.get("min_area_ratio_by_kind") or {}
    if min_area:
        filtered = [
            det
            for det in filtered
            if area_ratio(det.box, image_shape) >= float(min_area.get(det.kind, 0.0))
        ]

    nms_config = config.get("nms") or {}
    if nms_config.get("enabled", False):
        filtered = nms_by_kind(
            filtered,
            default_iou=float(nms_config.get("default_iou", 0.50)),
            iou_by_kind=nms_config.get("iou_by_kind") or {},
        )

    max_by_kind = config.get("max_detections_by_kind") or {}
    if max_by_kind:
        filtered = cap_detections_by_kind(filtered, max_by_kind)

    return filtered


def nms_by_kind(
    detections: list[Detection],
    *,
    default_iou: float,
    iou_by_kind: dict[str, float],
) -> list[Detection]:
    kept: list[Detection] = []
    for detection in sorted(detections, key=lambda item: item.confidence, reverse=True):
        iou_threshold = float(iou_by_kind.get(detection.kind, default_iou))
        if all(
            detection.kind != kept_detection.kind
            or iou(detection.box, kept_detection.box) < iou_threshold
            for kept_detection in kept
        ):
            kept.append(detection)
    return kept


def cap_detections_by_kind(detections: list[Detection], max_by_kind: dict[str, int]) -> list[Detection]:
    counts: dict[str, int] = {}
    kept: list[Detection] = []
    for detection in sorted(detections, key=lambda item: item.confidence, reverse=True):
        limit = max_by_kind.get(detection.kind)
        if limit is None:
            kept.append(detection)
            continue
        seen = counts.get(detection.kind, 0)
        if seen >= limit:
            continue
        counts[detection.kind] = seen + 1
        kept.append(detection)
    return kept


def iou(a: Box, b: Box) -> float:
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = a.area + b.area - inter
    if union <= 0:
        return 0.0
    return float(inter) / float(union)
