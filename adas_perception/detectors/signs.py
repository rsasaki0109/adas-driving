from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from adas_perception.types import Box, Detection


class ColorSignDetector:
    """Simple color/shape based traffic sign candidate detector."""

    def __init__(self, config: dict[str, Any]):
        self.config = config

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        height, width = frame_bgr.shape[:2]
        frame_area = float(width * height)
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

        min_area = frame_area * float(self.config.get("min_area_ratio", 0.00008))
        max_area = frame_area * float(self.config.get("max_area_ratio", 0.03))
        min_aspect = float(self.config.get("min_aspect_ratio", 0.45))
        max_aspect = float(self.config.get("max_aspect_ratio", 2.20))
        max_y = int(height * float(self.config.get("max_y_ratio", 0.88)))

        detections: list[Detection] = []
        for color_name, ranges in self.config.get("color_masks", {}).items():
            mask = self._mask_for_ranges(hsv, ranges)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = float(cv2.contourArea(contour))
                if area < min_area or area > max_area:
                    continue

                x, y, w, h = cv2.boundingRect(contour)
                if y > max_y or h <= 0:
                    continue
                aspect = float(w) / float(h)
                if aspect < min_aspect or aspect > max_aspect:
                    continue

                shape = self._shape_name(contour)
                fill_ratio = area / max(float(w * h), 1.0)
                if fill_ratio < 0.20:
                    continue
                confidence = min(0.90, 0.35 + fill_ratio * 0.45 + min(area / max_area, 1.0) * 0.10)
                label = self._classify_candidate(color_name, shape, aspect)
                detections.append(
                    Detection(
                        kind="traffic_sign",
                        label=label,
                        confidence=confidence,
                        box=Box(x, y, x + w, y + h).clamp(width, height),
                        source="color_shape",
                    )
                )

        return self._nms(detections, iou_threshold=0.35)

    def _classify_candidate(self, color_name: str, shape: str, aspect: float) -> str:
        if not self.config.get("classification", {}).get("enabled", True):
            return f"{color_name} {shape} sign candidate"

        if color_name == "red" and shape == "round":
            return "red circular regulatory sign candidate"
        if color_name == "red" and shape == "triangle":
            return "red triangular warning sign candidate"
        if color_name == "red" and shape == "rectangle" and 0.75 <= aspect <= 1.35:
            return "stop-like red sign candidate"
        if color_name == "blue" and shape == "round":
            return "blue circular instruction sign candidate"
        if color_name == "yellow" and shape in {"triangle", "rectangle"}:
            return "yellow warning sign candidate"
        return f"{color_name} {shape} sign candidate"

    @staticmethod
    def _mask_for_ranges(hsv: np.ndarray, ranges: list[dict[str, list[int]]]) -> np.ndarray:
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for item in ranges:
            lower = np.array(item["lower"], dtype=np.uint8)
            upper = np.array(item["upper"], dtype=np.uint8)
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))
        return mask

    @staticmethod
    def _shape_name(contour: np.ndarray) -> str:
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            return "candidate"
        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
        vertices = len(approx)
        if vertices == 3:
            return "triangle"
        if vertices == 4:
            return "rectangle"
        if vertices >= 7:
            return "round"
        return "candidate"

    @staticmethod
    def _nms(detections: list[Detection], iou_threshold: float) -> list[Detection]:
        if not detections:
            return []
        ordered = sorted(detections, key=lambda d: d.confidence, reverse=True)
        kept: list[Detection] = []
        for detection in ordered:
            if all(_iou(detection.box, kept_detection.box) < iou_threshold for kept_detection in kept):
                kept.append(detection)
        return kept


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
