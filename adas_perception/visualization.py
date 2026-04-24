from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from adas_perception.types import Detection, PerceptionResult


COLORS = {
    "lane": (0, 220, 255),
    "lane_fill": (20, 180, 60),
    "vehicle": (30, 170, 255),
    "pedestrian": (255, 90, 60),
    "traffic_sign": (60, 60, 255),
    "traffic_light": (60, 255, 120),
    "default": (230, 230, 230),
}


def draw_perception(
    frame_bgr: np.ndarray,
    result: PerceptionResult,
    config: dict[str, Any] | None = None,
) -> np.ndarray:
    config = config or {}
    viz_config = config.get("visualization", {})
    vis = frame_bgr.copy()
    _draw_lanes(vis, result, viz_config)

    include_kinds_raw = viz_config.get("include_kinds")
    include_kinds = (
        {str(k).strip() for k in include_kinds_raw if str(k).strip()}
        if include_kinds_raw
        else None
    )
    exclude_kinds_raw = viz_config.get("exclude_kinds") or []
    exclude_kinds = {str(k).strip() for k in exclude_kinds_raw if str(k).strip()}
    min_confidence = float(viz_config.get("min_confidence", 0.0))
    avoid_label_overlap = bool(viz_config.get("avoid_label_overlap", False))

    drawn_label_rects: list[tuple[int, int, int, int]] = []
    for detection in result.detections:
        if include_kinds is not None and detection.kind not in include_kinds:
            continue
        if detection.kind in exclude_kinds:
            continue
        if detection.confidence < min_confidence:
            continue
        _draw_detection(
            vis,
            detection,
            viz_config,
            drawn_label_rects=drawn_label_rects if avoid_label_overlap else None,
        )
    if viz_config.get("show_summary", True):
        _draw_summary(vis, result)
    return vis


def _draw_lanes(vis: np.ndarray, result: PerceptionResult, viz_config: dict[str, Any]) -> None:
    alpha = float(viz_config.get("lane_alpha", 0.28))
    if result.lanes.polygon:
        overlay = vis.copy()
        cv2.fillPoly(overlay, [np.array(result.lanes.polygon, dtype=np.int32)], COLORS["lane_fill"])
        cv2.addWeighted(overlay, alpha, vis, 1.0 - alpha, 0, vis)

    for start, end in result.lanes.raw_segments:
        cv2.line(vis, start, end, (90, 90, 90), 1, cv2.LINE_AA)

    for lane_line in result.lanes.lines:
        if lane_line.polyline and len(lane_line.polyline) >= 2:
            pts = np.array(lane_line.polyline, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(vis, [pts], False, COLORS["lane"], 5, cv2.LINE_AA)
            anchor = lane_line.polyline[0]
        else:
            start, end = lane_line.points
            cv2.line(vis, start, end, COLORS["lane"], 5, cv2.LINE_AA)
            anchor = start
        _put_label(vis, lane_line.side, anchor, COLORS["lane"], font_scale=0.50)


def _draw_detection(
    vis: np.ndarray,
    detection: Detection,
    viz_config: dict[str, Any],
    drawn_label_rects: list[tuple[int, int, int, int]] | None = None,
) -> None:
    color = COLORS.get(detection.kind, COLORS["default"])
    thickness = int(viz_config.get("box_thickness", 2))
    box = detection.box
    cv2.rectangle(vis, (box.x1, box.y1), (box.x2, box.y2), color, thickness)

    style = str(viz_config.get("label_style", "full")).lower()
    if style == "none":
        return
    track = f"#{detection.track_id} " if detection.track_id is not None else ""
    distance = _format_distance(detection.distance_m, viz_config)
    if style == "kind":
        label = f"{track}{detection.kind}{distance}"
    elif style == "compact":
        label = f"{track}{detection.kind} {detection.confidence:.2f}{distance}"
    else:  # full
        label = f"{track}{detection.kind}:{detection.label} {detection.confidence:.2f}{distance}"
    _put_label(
        vis,
        label,
        (box.x1, max(0, box.y1 - 6)),
        color,
        float(viz_config.get("font_scale", 0.55)),
        drawn_label_rects=drawn_label_rects,
    )


def _draw_summary(vis: np.ndarray, result: PerceptionResult) -> None:
    counts = result.count_by_kind()
    parts = [
        f"lane={counts.get('lane', 0)}",
        f"vehicle={counts.get('vehicle', 0)}",
        f"pedestrian={counts.get('pedestrian', 0)}",
        f"sign={counts.get('traffic_sign', 0)}",
        f"light={counts.get('traffic_light', 0)}",
    ]
    text = " | ".join(parts)
    _put_label(vis, text, (12, 28), (0, 0, 0), font_scale=0.60, text_color=(255, 255, 255))


def _put_label(
    vis: np.ndarray,
    text: str,
    origin: tuple[int, int],
    bg_color: tuple[int, int, int],
    font_scale: float,
    text_color: tuple[int, int, int] = (255, 255, 255),
    drawn_label_rects: list[tuple[int, int, int, int]] | None = None,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = origin
    y = max(text_height + 6, y)
    x = max(0, min(vis.shape[1] - text_width - 8, x))
    rect_h = text_height + baseline + 10
    # Collision avoidance: try offsetting y downward until no overlap with
    # previously-drawn label rects. If we walked off the bottom, fall back to
    # the original position (rare; very dense scene).
    if drawn_label_rects is not None:
        step = rect_h
        y_try = y
        y_max = vis.shape[0] - 2
        for _ in range(8):  # up to 8 stacked labels then give up
            rect = (x, y_try - text_height - baseline - 6, x + text_width + 8, y_try + baseline + 4)
            if not any(_rects_overlap(rect, r) for r in drawn_label_rects):
                y = y_try
                break
            y_try += step
            if y_try > y_max:
                break
        drawn_label_rects.append(
            (x, y - text_height - baseline - 6, x + text_width + 8, y + baseline + 4)
        )
    cv2.rectangle(
        vis,
        (x, y - text_height - baseline - 6),
        (x + text_width + 8, y + baseline + 4),
        bg_color,
        -1,
    )
    cv2.putText(vis, text, (x + 4, y - 4), font, font_scale, text_color, thickness, cv2.LINE_AA)


def _rects_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _format_distance(distance_m: float | None, viz_config: dict[str, Any]) -> str:
    """Format the distance portion of a detection label.

    Options via viz_config["distance_format"]:
      "precise" (default): ~15.2m — current behavior, implies more precision
         than the monocular estimator actually has.
      "rounded": ~15m (or ~30m+ for far objects), quantized to a step that
         matches the estimator's real uncertainty.
      "bucket": close / mid / far — rough category only.
      "hide": no distance displayed.
    """
    if distance_m is None:
        return ""
    fmt = str(viz_config.get("distance_format", "precise")).lower()
    if fmt == "hide":
        return ""
    if fmt == "bucket":
        if distance_m < 10.0:
            return " close"
        if distance_m < 30.0:
            return " mid"
        return " far"
    if fmt == "rounded":
        if distance_m >= 30.0:
            return " ~30m+"
        if distance_m >= 10.0:
            return f" ~{int(round(distance_m / 5) * 5)}m"
        return f" ~{int(round(distance_m))}m"
    return f" ~{distance_m:.1f}m"
