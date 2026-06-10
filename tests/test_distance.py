from __future__ import annotations

import numpy as np

from adas_perception.distance import MonocularDistanceEstimator
from adas_perception.types import Box, Detection


def test_distance_uses_intrinsics_fy_when_provided():
    estimator = MonocularDistanceEstimator(
        {
            "enabled": True,
            "intrinsics": {"fy": 900.0, "fx": 900.0, "cx": 640.0, "cy": 360.0},
            "camera_height_m": 1.35,
            "object_heights_m": {"vehicle": 1.5},
        }
    )
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    detection = Detection(
        kind="vehicle",
        label="car",
        confidence=0.9,
        box=Box(x1=600, y1=500, x2=700, y2=600),
        source="test",
    )
    results = estimator.estimate([detection], frame)
    assert results[0].distance_m is not None
    assert results[0].distance_m > 0
    assert results[0].ground_position_m is not None
    x_m, z_m = results[0].ground_position_m
    assert z_m > 0


def test_ground_fusion_pulls_distance_toward_ground_plane_depth():
    base_config = {
        "enabled": True,
        "intrinsics": {"fy": 900.0, "fx": 900.0, "cx": 640.0, "cy": 360.0},
        "camera_height_m": 1.35,
        "object_heights_m": {"vehicle": 1.5},
    }
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    detection = Detection(
        kind="vehicle",
        label="car",
        confidence=0.9,
        box=Box(x1=600, y1=500, x2=700, y2=600),
        source="test",
    )
    height_only = 1.5 * 900.0 / 100.0  # 13.5 m from bbox height alone
    ground_z = 1.35 * 900.0 / (600.0 - 360.0)  # 5.0625 m from ground plane

    fused = MonocularDistanceEstimator(base_config).estimate([detection], frame)
    assert fused[0].distance_m is not None
    assert ground_z < fused[0].distance_m < height_only
    expected = 0.7 * ground_z + 0.3 * height_only
    assert abs(fused[0].distance_m - expected) < 1e-6

    disabled = MonocularDistanceEstimator(
        {**base_config, "ground_fusion": {"enabled": False}}
    ).estimate([detection], frame)
    assert abs(disabled[0].distance_m - height_only) < 1e-6

    no_height = MonocularDistanceEstimator(
        {key: value for key, value in base_config.items() if key != "camera_height_m"}
    ).estimate([detection], frame)
    assert abs(no_height[0].distance_m - height_only) < 1e-6


def test_distance_skips_unknown_kind():
    estimator = MonocularDistanceEstimator({"enabled": True})
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detection = Detection(
        kind="traffic_sign",
        label="stop sign",
        confidence=0.8,
        box=Box(x1=10, y1=10, x2=50, y2=90),
        source="test",
    )
    results = estimator.estimate([detection], frame)
    assert results[0].distance_m is None
