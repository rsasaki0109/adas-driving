from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from adas_planning.types import PlanningResult


OVERLAY_COLORS = {
    "target_path": (255, 180, 40),
    "lead": (0, 140, 255),
    "panel_bg": (20, 20, 20),
    "text": (240, 240, 240),
    "warning": (80, 80, 255),
    "confidence_ok": (80, 200, 80),
    "confidence_low": (80, 160, 255),
}


def draw_planning_overlay(
    frame_bgr: np.ndarray,
    planning_result: PlanningResult,
    *,
    perception_frame: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> np.ndarray:
    config = config or {}
    overlay_cfg = config.get("overlay", {})
    vis = frame_bgr.copy()

    if perception_frame:
        _draw_lane_context(vis, perception_frame)

    if planning_result.target_path_px:
        points = np.array(planning_result.target_path_px, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [points], False, OVERLAY_COLORS["target_path"], 3, cv2.LINE_AA)
        for x, y in planning_result.target_path_px[:: max(1, len(planning_result.target_path_px) // 6)]:
            cv2.circle(vis, (int(x), int(y)), 4, OVERLAY_COLORS["target_path"], -1, cv2.LINE_AA)

    if planning_result.lead_object_id is not None and perception_frame:
        _highlight_lead(vis, perception_frame, planning_result.lead_object_id)

    _draw_hud(vis, planning_result, overlay_cfg)
    return vis


def _draw_lane_context(frame: np.ndarray, perception_frame: dict[str, Any]) -> None:
    lanes = perception_frame.get("lanes") or {}
    polygon = lanes.get("polygon") or []
    if len(polygon) >= 3:
        pts = np.array(polygon, dtype=np.int32)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], (30, 120, 30))
        cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
    for line in lanes.get("lines") or []:
        points = line.get("points") or []
        if len(points) >= 2:
            pts = np.array(points, dtype=np.int32)
            color = (0, 220, 255) if line.get("side") == "left" else (255, 120, 0)
            cv2.polylines(frame, [pts], False, color, 2, cv2.LINE_AA)


def _highlight_lead(frame: np.ndarray, perception_frame: dict[str, Any], lead_object_id: int) -> None:
    for detection in perception_frame.get("detections") or []:
        if detection.get("track_id") != lead_object_id:
            continue
        box = detection.get("box") or {}
        x1, y1, x2, y2 = int(box.get("x1", 0)), int(box.get("y1", 0)), int(box.get("x2", 0)), int(box.get("y2", 0))
        cv2.rectangle(frame, (x1, y1), (x2, y2), OVERLAY_COLORS["lead"], 3)
        label = f"LEAD id={lead_object_id}"
        if detection.get("distance_m") is not None:
            label += f" d={float(detection['distance_m']):.1f}m"
        cv2.putText(frame, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, OVERLAY_COLORS["lead"], 2, cv2.LINE_AA)
        break


def _draw_hud(frame: np.ndarray, result: PlanningResult, overlay_cfg: dict[str, Any]) -> None:
    height, width = frame.shape[:2]
    panel_h = 130
    panel = np.zeros((panel_h, width, 3), dtype=np.uint8)
    panel[:] = OVERLAY_COLORS["panel_bg"]

    behavior = result.behavior.value if hasattr(result.behavior, "value") else str(result.behavior)
    speed_text = "n/a" if result.target_speed_mps is None else f"{result.target_speed_mps:.1f} m/s"
    ego_source = result.debug.get("ego_speed_source")
    ego_speed = result.debug.get("ego_speed_mps")
    lines = [
        f"behavior: {behavior}",
        f"target_speed: {speed_text}",
        f"confidence: {result.confidence:.2f}",
    ]
    if ego_source and ego_source != "none":
        ego_line = f"ego_speed: {ego_speed:.1f} m/s ({ego_source})" if ego_speed is not None else f"ego_speed: n/a ({ego_source})"
        lines.insert(2, ego_line)
    if result.stop_reason:
        lines.append(f"stop_reason: {result.stop_reason}")
    if result.warnings:
        lines.append("warnings: " + ", ".join(w.code for w in result.warnings[:3]))

    y = 24
    for line in lines:
        cv2.putText(panel, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, OVERLAY_COLORS["text"], 1, cv2.LINE_AA)
        y += 22

    bar_w = int((width - 24) * max(0.0, min(1.0, result.confidence)))
    bar_color = OVERLAY_COLORS["confidence_ok"] if result.confidence >= 0.5 else OVERLAY_COLORS["confidence_low"]
    cv2.rectangle(panel, (12, panel_h - 18), (12 + bar_w, panel_h - 8), bar_color, -1)
    cv2.rectangle(panel, (12, panel_h - 18), (width - 12, panel_h - 8), (120, 120, 120), 1)

    frame[0:panel_h, 0:width] = cv2.addWeighted(panel, float(overlay_cfg.get("panel_alpha", 0.82)), frame[0:panel_h, 0:width], 0.18, 0)
