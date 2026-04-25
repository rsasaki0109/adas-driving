from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

from adas_perception.types import LaneLine, LaneResult, Point


class LaneDetector:
    """Lightweight lane marker detector based on edges and Hough segments."""

    def __init__(self, config: dict[str, Any]):
        self.config = config

    def detect(self, frame_bgr: np.ndarray) -> LaneResult:
        height, width = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        kernel = int(self.config.get("gaussian_kernel", 5))
        if kernel % 2 == 0:
            kernel += 1
        blurred = cv2.GaussianBlur(gray, (kernel, kernel), 0)

        edges = cv2.Canny(
            blurred,
            int(self.config.get("canny_low", 50)),
            int(self.config.get("canny_high", 150)),
        )

        color_mask_config = self.config.get("color_mask", {})
        if color_mask_config.get("enabled", False):
            color_mask = _build_lane_color_mask(frame_bgr, color_mask_config)
            dilate_iter = int(color_mask_config.get("dilate_iterations", 2))
            if dilate_iter > 0:
                color_mask = cv2.dilate(color_mask, np.ones((3, 3), np.uint8), iterations=dilate_iter)
            edges = cv2.bitwise_and(edges, color_mask)

        roi_mask = np.zeros_like(edges)
        roi = self._roi_polygon(width, height)
        cv2.fillPoly(roi_mask, [roi], 255)
        masked_edges = cv2.bitwise_and(edges, roi_mask)

        theta = math.radians(float(self.config.get("hough_theta_degrees", 1)))
        lines = cv2.HoughLinesP(
            masked_edges,
            rho=float(self.config.get("hough_rho", 2)),
            theta=theta,
            threshold=int(self.config.get("hough_threshold", 35)),
            minLineLength=int(self.config.get("min_line_length", 35)),
            maxLineGap=int(self.config.get("max_line_gap", 80)),
        )
        if lines is None:
            return LaneResult()

        raw_segments: list[tuple[Point, Point]] = []
        left: list[tuple[float, float, float]] = []
        right: list[tuple[float, float, float]] = []
        # Per-side endpoint pools for polynomial fit (each x,y pair).
        left_points: list[tuple[float, float]] = []
        right_points: list[tuple[float, float]] = []
        min_abs_slope = float(self.config.get("min_abs_slope", 0.45))

        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            dx = float(x2 - x1)
            if abs(dx) < 1.0:
                continue
            slope = float(y2 - y1) / dx
            if abs(slope) < min_abs_slope:
                continue
            intercept = float(y1) - slope * float(x1)
            length = math.hypot(float(x2 - x1), float(y2 - y1))
            raw_segments.append(((int(x1), int(y1)), (int(x2), int(y2))))
            if slope < 0:
                left.append((slope, intercept, length))
                left_points.append((float(x1), float(y1)))
                left_points.append((float(x2), float(y2)))
            else:
                right.append((slope, intercept, length))
                right_points.append((float(x1), float(y1)))
                right_points.append((float(x2), float(y2)))

        poly_cfg = self.config.get("polynomial_fit", {})
        poly_enabled = bool(poly_cfg.get("enabled", False))

        lane_lines: list[LaneLine] = []
        left_line = self._fit_lane("left", left, width, height)
        right_line = self._fit_lane("right", right, width, height)
        if poly_enabled:
            min_pts = int(poly_cfg.get("min_points", 6))
            samples = int(poly_cfg.get("polyline_samples", 12))
            if len(left_points) >= min_pts:
                left_line = self._poly_fit_lane("left", left_points, width, height, samples) or left_line
            if len(right_points) >= min_pts:
                right_line = self._poly_fit_lane("right", right_points, width, height, samples) or right_line
        if left_line:
            lane_lines.append(left_line)
        if right_line:
            lane_lines.append(right_line)

        polygon: list[Point] = []
        if left_line and right_line:
            polygon = self._build_lane_polygon(left_line, right_line)

        return LaneResult(lines=lane_lines, raw_segments=raw_segments, polygon=polygon)

    def _build_lane_polygon(self, left_line: LaneLine, right_line: LaneLine) -> list[Point]:
        left_pts = list(left_line.polyline) if left_line.polyline else list(left_line.points)
        right_pts = list(right_line.polyline) if right_line.polyline else list(right_line.points)
        if not left_pts or not right_pts:
            return []
        # Order each side bottom -> top; merge as left bottom..top + right top..bottom.
        left_pts.sort(key=lambda p: -p[1])
        right_pts.sort(key=lambda p: -p[1])
        return left_pts + list(reversed(right_pts))

    def _roi_polygon(self, width: int, height: int) -> np.ndarray:
        roi_config = self.config.get("roi", {})
        keys = ["bottom_left", "top_left", "top_right", "bottom_right"]
        default = {
            "bottom_left": [0.08, 0.96],
            "top_left": [0.42, 0.60],
            "top_right": [0.58, 0.60],
            "bottom_right": [0.95, 0.96],
        }
        points = []
        for key in keys:
            x_ratio, y_ratio = roi_config.get(key, default[key])
            points.append((int(width * x_ratio), int(height * y_ratio)))
        return np.array(points, dtype=np.int32)

    def _fit_lane(
        self,
        side: str,
        candidates: list[tuple[float, float, float]],
        width: int,
        height: int,
    ) -> LaneLine | None:
        if not candidates:
            return None
        weights = np.array([max(length, 1.0) for _, _, length in candidates], dtype=np.float32)
        slopes = np.array([slope for slope, _, _ in candidates], dtype=np.float32)
        intercepts = np.array([intercept for _, intercept, _ in candidates], dtype=np.float32)
        slope = float(np.average(slopes, weights=weights))
        intercept = float(np.average(intercepts, weights=weights))
        if abs(slope) < 1e-3:
            return None

        roi = self.config.get("roi", {})
        y_bottom = int(height * float(roi.get("bottom_left", [0.0, 0.96])[1]))
        y_top = int(height * float(roi.get("top_left", [0.0, 0.60])[1]))

        # Outlier rejection: drop slope/intercept candidates more than k MADs
        # away from the weighted median, then re-average. Helps when a strong
        # but spurious edge (curb, shadow) dominates the simple weighted mean.
        if len(candidates) >= 3:
            slope_med = float(np.median(slopes))
            mad = float(np.median(np.abs(slopes - slope_med))) or 1e-3
            keep = np.abs(slopes - slope_med) <= 3.0 * mad
            if keep.any() and keep.sum() != len(slopes):
                slope = float(np.average(slopes[keep], weights=weights[keep]))
                intercept = float(np.average(intercepts[keep], weights=weights[keep]))
                if abs(slope) < 1e-3:
                    return None

        x_bottom = int((y_bottom - intercept) / slope)
        x_top = int((y_top - intercept) / slope)
        x_bottom = max(0, min(width - 1, x_bottom))
        x_top = max(0, min(width - 1, x_top))
        confidence = min(1.0, len(candidates) / 8.0)
        return LaneLine(
            side=side,
            points=((x_bottom, y_bottom), (x_top, y_top)),
            confidence=confidence,
        )



    def _poly_fit_lane(
        self,
        side: str,
        points: list[tuple[float, float]],
        width: int,
        height: int,
        samples: int,
    ) -> LaneLine | None:
        if len(points) < 6:
            return None
        ys = np.array([p[1] for p in points], dtype=np.float64)
        xs = np.array([p[0] for p in points], dtype=np.float64)
        # Fit x = a*y^2 + b*y + c (treats y as the independent axis since
        # lane markings are nearly vertical in the camera view).
        try:
            coeffs = np.polyfit(ys, xs, 2)
        except Exception:
            return None
        roi = self.config.get("roi", {})
        y_bottom = int(height * float(roi.get("bottom_left", [0.0, 0.96])[1]))
        y_top = int(height * float(roi.get("top_left", [0.0, 0.60])[1]))
        sample_count = max(2, samples)
        ys_sampled = np.linspace(y_bottom, y_top, sample_count)
        xs_sampled = np.polyval(coeffs, ys_sampled)
        polyline: list[Point] = []
        for x, y in zip(xs_sampled, ys_sampled):
            xi = int(round(float(x)))
            yi = int(round(float(y)))
            xi = max(0, min(width - 1, xi))
            yi = max(0, min(height - 1, yi))
            polyline.append((xi, yi))
        confidence = min(1.0, len(points) / 16.0)
        return LaneLine(
            side=side,
            points=(polyline[0], polyline[-1]),
            confidence=confidence,
            polyline=polyline,
        )


def _build_lane_color_mask(frame_bgr: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    """White + yellow lane marker mask in HSV.

    Defaults are tuned for typical daytime road footage. Both ranges are
    OR-merged. Tweak via cfg["white"] and cfg["yellow"], each with
    {"hsv_lower": [H,S,V], "hsv_upper": [H,S,V]}.
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    white_cfg = cfg.get("white", {})
    yellow_cfg = cfg.get("yellow", {})
    white_lower = np.array(white_cfg.get("hsv_lower", [0, 0, 200]), dtype=np.uint8)
    white_upper = np.array(white_cfg.get("hsv_upper", [180, 40, 255]), dtype=np.uint8)
    yellow_lower = np.array(yellow_cfg.get("hsv_lower", [15, 80, 120]), dtype=np.uint8)
    yellow_upper = np.array(yellow_cfg.get("hsv_upper", [35, 255, 255]), dtype=np.uint8)
    white_mask = cv2.inRange(hsv, white_lower, white_upper)
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
    return cv2.bitwise_or(white_mask, yellow_mask)
