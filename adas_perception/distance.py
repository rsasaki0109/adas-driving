from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import numpy as np

from adas_perception.types import Detection


class MonocularDistanceEstimator:
    """Rough distance estimates from bounding-box height and assumed object size."""

    def __init__(self, config: dict[str, Any]):
        self.enabled = bool(config.get("enabled", False))
        self.horizontal_fov_degrees = float(config.get("horizontal_fov_degrees", 70.0))
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
            output.append(replace(detection, distance_m=distance_m))
        return output

    def _focal_length_px(self, image_width: int) -> float:
        fov = max(1.0, min(179.0, self.horizontal_fov_degrees))
        return image_width / (2.0 * math.tan(math.radians(fov) / 2.0))

