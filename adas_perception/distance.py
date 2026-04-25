from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import numpy as np

from adas_perception.types import Box, Detection


class MonocularDistanceEstimator:
    """Rough distance estimates from bounding-box height and assumed object size.

    Calibration priority for the focal length used in the height-projection
    formula `distance = object_height_m * focal_y_px / bbox_height_px`:

      1. `intrinsics.fy` (most accurate; vertical focal length in pixels)
      2. `focal_length_px` (direct override; treated as fy)
      3. `horizontal_fov_degrees` (default fallback; computes f_x from
         image width and assumes square pixels so f_y == f_x)
    """

    def __init__(self, config: dict[str, Any]):
        self.enabled = bool(config.get("enabled", False))
        self.horizontal_fov_degrees = float(config.get("horizontal_fov_degrees", 70.0))
        self.focal_length_px = config.get("focal_length_px")
        intrinsics = config.get("intrinsics") or {}
        self.intrinsics_fx = intrinsics.get("fx")
        self.intrinsics_fy = intrinsics.get("fy")
        self.intrinsics_cx = intrinsics.get("cx")
        self.intrinsics_cy = intrinsics.get("cy")
        self.camera_height_m = config.get("camera_height_m")
        self.min_box_height_px = int(config.get("min_box_height_px", 12))
        self.max_distance_m = float(config.get("max_distance_m", 120.0))
        self.object_heights_m = {
            str(kind): float(height)
            for kind, height in config.get(
                "object_heights_m",
                {
                    "pedestrian": 1.70,
                    "vehicle": 1.50,
                },
            ).items()
        }

    def estimate(self, detections: list[Detection], frame_bgr: np.ndarray) -> list[Detection]:
        if not self.enabled:
            return detections

        height, width = frame_bgr.shape[:2]
        if width <= 0 or height <= 0:
            return detections

        focal_px = self._focal_length_px(width)
        cx = float(self.intrinsics_cx) if self.intrinsics_cx is not None else 0.5 * width
        cy = float(self.intrinsics_cy) if self.intrinsics_cy is not None else 0.5 * height
        fx = float(self.intrinsics_fx) if self.intrinsics_fx is not None else focal_px
        camera_height_m = (
            float(self.camera_height_m) if self.camera_height_m is not None else None
        )
        output: list[Detection] = []
        for detection in detections:
            object_height_m = self.object_heights_m.get(detection.kind)
            if object_height_m is None or detection.box.height < self.min_box_height_px:
                output.append(detection)
                continue

            distance_m = object_height_m * focal_px / max(float(detection.box.height), 1.0)
            if not math.isfinite(distance_m) or distance_m <= 0:
                output.append(detection)
                continue
            distance_m = min(distance_m, self.max_distance_m)

            ground_position = self._ground_position(
                detection.box, fx=fx, fy=focal_px, cx=cx, cy=cy, camera_height_m=camera_height_m
            )
            output.append(
                replace(detection, distance_m=distance_m, ground_position_m=ground_position)
            )
        return output

    def _ground_position(
        self,
        box: Box,
        *,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        camera_height_m: float | None,
    ) -> tuple[float, float] | None:
        """Project the bbox bottom-center to (X, Z) on the ground plane.

        Assumes the camera is aligned with the road, with optical axis
        roughly horizontal and the ground at depth Y = camera_height_m
        below the camera. Returns None when camera_height_m is not provided
        or when the bottom of the box is at/above the horizon.
        """
        if camera_height_m is None or camera_height_m <= 0:
            return None
        u = 0.5 * (float(box.x1) + float(box.x2))
        v = float(box.y2)  # bottom of bbox = ground contact
        delta_y = v - cy
        if delta_y <= 0:
            return None  # box bottom at or above horizon → no projection
        z_m = float(camera_height_m) * fy / delta_y
        if not math.isfinite(z_m) or z_m <= 0 or z_m > self.max_distance_m * 1.5:
            return None
        x_m = (u - cx) * z_m / fx
        return (float(x_m), float(z_m))

    def _focal_length_px(self, image_width: int) -> float:
        if self.intrinsics_fy is not None:
            return float(self.intrinsics_fy)
        if self.focal_length_px is not None:
            return float(self.focal_length_px)
        fov = max(1.0, min(179.0, self.horizontal_fov_degrees))
        return image_width / (2.0 * math.tan(math.radians(fov) / 2.0))

