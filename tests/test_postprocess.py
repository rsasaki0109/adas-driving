from __future__ import annotations

from adas_perception.postprocess import apply_postprocess, build_post_process_config, nms_by_kind
from adas_perception.types import Box, Detection


def _det(kind: str, confidence: float, x1: int, y1: int, x2: int, y2: int) -> Detection:
    return Detection(
        kind=kind,
        label=kind,
        confidence=confidence,
        box=Box(x1=x1, y1=y1, x2=x2, y2=y2),
        source="test",
    )


def test_nms_by_kind_keeps_highest_confidence_overlap():
    detections = [
        _det("traffic_light", 0.9, 10, 10, 30, 30),
        _det("traffic_light", 0.7, 12, 12, 32, 32),
        _det("vehicle", 0.8, 100, 100, 200, 200),
    ]
    kept = nms_by_kind(detections, default_iou=0.50, iou_by_kind={"traffic_light": 0.35})
    assert len(kept) == 2
    assert kept[0].kind == "traffic_light"
    assert kept[0].confidence == 0.9


def test_apply_postprocess_filters_min_area_and_nms():
    config = build_post_process_config(
        {
            "enabled": True,
            "nms": {"enabled": True, "default_iou": 0.50, "iou_by_kind": {"traffic_sign": 0.35}},
            "min_area_ratio_by_kind": {"traffic_sign": 0.01},
        }
    )
    detections = [
        _det("traffic_sign", 0.9, 0, 0, 5, 5),
        _det("traffic_sign", 0.8, 1, 1, 6, 6),
        _det("traffic_sign", 0.95, 100, 100, 200, 200),
    ]
    kept = apply_postprocess(detections, config, (360, 640))
    assert len(kept) == 1
    assert kept[0].confidence == 0.95
