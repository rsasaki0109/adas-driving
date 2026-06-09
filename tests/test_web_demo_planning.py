from __future__ import annotations

import numpy as np

from scripts.web_demo import DEFAULT_PLANNING_PRESET, DEFAULT_PRESET, render_frame_bgr


def test_render_frame_bgr_with_planning_overlay():
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    frame[100:140, 120:200] = (40, 40, 40)

    vis_bgr, stats = render_frame_bgr(
        frame,
        preset=DEFAULT_PRESET,
        device="cpu",
        enable_planning=True,
        planning_preset=DEFAULT_PLANNING_PRESET,
    )

    assert vis_bgr.shape == frame.shape
    assert stats["planning"]["behavior"]
    assert "confidence" in stats["planning"]


def test_render_frame_bgr_without_planning():
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    vis_bgr, stats = render_frame_bgr(
        frame,
        preset=DEFAULT_PRESET,
        device="cpu",
        enable_planning=False,
    )
    assert vis_bgr.shape == frame.shape
    assert "planning" not in stats
