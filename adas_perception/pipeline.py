from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np

from adas_perception.detectors import (
    ColorSignDetector,
    ColorTrafficLightDetector,
    create_lane_detector,
    create_object_detector,
)
from adas_perception.distance import MonocularDistanceEstimator
from adas_perception.lane_smoothing import LaneSmoother
from adas_perception.postprocess import apply_postprocess, build_post_process_config
from adas_perception.tracking import SimpleTracker
from adas_perception.traffic_light_state import TrafficLightStateClassifier
from adas_perception.types import Box, Detection, LaneResult, PerceptionResult


class ADASPerceptionPipeline:
    """Minimal camera perception pipeline for demo images and videos."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        lane_config = config.get("lane", {}) or {}
        self.lane_detector = (
            create_lane_detector(lane_config) if lane_config.get("enabled", True) else None
        )
        self.object_detectors = _create_object_detectors(config.get("objects", {}))
        self.fusion_config = _build_fusion_config(config.get("objects", {}).get("fusion", {}))
        self.post_fusion_score_thresholds = {
            str(kind): float(value)
            for kind, value in (config.get("objects", {}).get("post_fusion_score_thresholds_by_kind", {}) or {}).items()
        }
        self.post_process_config = build_post_process_config(config.get("objects", {}).get("post_process", {}))
        self._wbf_fn = _load_wbf() if self.fusion_config["mode"] == "wbf" else None
        self.sign_detector = (
            ColorSignDetector(config.get("signs", {})) if config.get("signs", {}).get("enabled", True) else None
        )
        self.traffic_light_detector = (
            ColorTrafficLightDetector(config.get("traffic_lights", {}))
            if config.get("traffic_lights", {}).get("enabled", True)
            else None
        )
        self.lane_smoother = LaneSmoother(config.get("lane_smoothing", {}))
        self.tracker = SimpleTracker(config.get("tracking", {}))
        self.distance_estimator = MonocularDistanceEstimator(config.get("distance_estimation", {}))
        self.traffic_light_state_classifier = TrafficLightStateClassifier(
            config.get("traffic_light_state", {})
        )

    def reset(self) -> None:
        self.lane_smoother.reset()
        self.tracker.reset()

    def run(self, frame_bgr: np.ndarray) -> PerceptionResult:
        lanes = self.lane_detector.detect(frame_bgr) if self.lane_detector else LaneResult()
        lanes = self.lane_smoother.smooth(lanes)
        per_detector_detections: list[list[Detection]] = []
        for object_detector in self.object_detectors:
            per_detector_detections.append(object_detector.detect(frame_bgr))
        heuristic_detections: list[Detection] = []
        if self.sign_detector:
            heuristic_detections.extend(self.sign_detector.detect(frame_bgr))
        if self.traffic_light_detector:
            heuristic_detections.extend(self.traffic_light_detector.detect(frame_bgr))

        if self.fusion_config["mode"] == "wbf" and self._wbf_fn is not None and per_detector_detections:
            fused = _fuse_detections_wbf(
                per_detector_detections,
                weights=self.fusion_config["weights"],
                iou_thr=self.fusion_config["iou_thr"],
                kind_iou_thr=self.fusion_config["kind_iou_thr"],
                image_shape=frame_bgr.shape[:2],
                wbf_fn=self._wbf_fn,
            )
            detections = fused + heuristic_detections
        else:
            detections = [det for dets in per_detector_detections for det in dets]
            detections.extend(heuristic_detections)
        if self.post_fusion_score_thresholds:
            detections = [
                det for det in detections
                if det.confidence >= self.post_fusion_score_thresholds.get(det.kind, 0.0)
            ]
        detections = _suppress_signs_over_traffic_lights(detections)
        if self.post_process_config.get("enabled", False):
            detections = apply_postprocess(detections, self.post_process_config, frame_bgr.shape[:2])
        else:
            nms_iou = float(self.fusion_config.get("post_nms_iou", 0.50))
            post_nms_enabled = bool(
                self.fusion_config.get(
                    "post_nms",
                    self.fusion_config["mode"] != "wbf",
                )
            )
            if post_nms_enabled and nms_iou > 0.0:
                detections = _nms_by_kind(detections, iou_threshold=nms_iou)
        detections = self.tracker.update(detections)
        detections = self.distance_estimator.estimate(detections, frame_bgr)
        detections = self.traffic_light_state_classifier.classify(frame_bgr, detections)
        return PerceptionResult(lanes=lanes, detections=detections)


def _create_object_detectors(config: dict[str, Any]) -> list[Any]:
    if not config.get("enabled", True):
        return []

    detector_configs = config.get("detectors")
    if not detector_configs:
        return [create_object_detector(config)]

    detectors = []
    for detector_config in detector_configs:
        detector_config = dict(detector_config)
        if not detector_config.get("enabled", True):
            continue
        parent_device = config.get("device")
        if parent_device and (parent_device != "auto" or "device" not in detector_config):
            detector_config["device"] = parent_device
        detectors.append(create_object_detector(detector_config))
    return detectors


def _suppress_signs_over_traffic_lights(detections: list[Detection]) -> list[Detection]:
    traffic_lights = [detection for detection in detections if detection.kind == "traffic_light"]
    if not traffic_lights:
        return detections

    output: list[Detection] = []
    for detection in detections:
        if detection.kind == "traffic_sign" and any(_iou(detection.box, light.box) > 0.25 for light in traffic_lights):
            continue
        output.append(detection)
    return output


def _build_fusion_config(raw: dict[str, Any]) -> dict[str, Any]:
    mode = str(raw.get("mode", "nms")).lower()
    weights_raw = raw.get("weights")
    kind_iou_raw = raw.get("kind_iou_thr") or {}
    return {
        "mode": mode,
        "iou_thr": float(raw.get("iou_thr", 0.55)),
        "kind_iou_thr": {str(k): float(v) for k, v in kind_iou_raw.items()},
        "weights": [float(w) for w in weights_raw] if weights_raw else None,
        "post_nms": bool(raw.get("post_nms", False)),
        "post_nms_iou": float(raw.get("post_nms_iou", 0.50)),
    }


def _load_wbf():
    candidate_dirs = [
        Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages",
    ]
    for base in sys.path:
        if base:
            candidate_dirs.append(Path(base))
    for base in candidate_dirs:
        target = base / "ensemble_boxes" / "ensemble_boxes_wbf.py"
        if target.is_file():
            spec = importlib.util.spec_from_file_location("ensemble_boxes_wbf", target)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module.weighted_boxes_fusion
    raise RuntimeError(
        "ensemble_boxes_wbf.py not found; install with `pip install ensemble-boxes` to use fusion.mode=wbf."
    )


def _fuse_detections_wbf(
    per_source_detections: list[list[Detection]],
    *,
    weights: list[float] | None,
    iou_thr: float,
    kind_iou_thr: dict[str, float],
    image_shape: tuple[int, int],
    wbf_fn,
) -> list[Detection]:
    height, width = int(image_shape[0]), int(image_shape[1])
    height = max(height, 1)
    width = max(width, 1)
    n_sources = len(per_source_detections)
    if weights is None:
        weights = [1.0] * n_sources
    by_kind: dict[str, list[tuple[int, Detection]]] = {}
    for src_idx, dets in enumerate(per_source_detections):
        for det in dets:
            by_kind.setdefault(det.kind, []).append((src_idx, det))

    fused: list[Detection] = []
    for kind, entries in by_kind.items():
        boxes_per_source: list[list[list[float]]] = [[] for _ in range(n_sources)]
        scores_per_source: list[list[float]] = [[] for _ in range(n_sources)]
        labels_per_source: list[list[int]] = [[] for _ in range(n_sources)]
        source_labels: dict[int, str] = {}
        for src_idx, det in entries:
            box = det.box
            if box.x2 <= box.x1 or box.y2 <= box.y1:
                continue
            norm = [
                max(0.0, min(1.0, box.x1 / width)),
                max(0.0, min(1.0, box.y1 / height)),
                max(0.0, min(1.0, box.x2 / width)),
                max(0.0, min(1.0, box.y2 / height)),
            ]
            boxes_per_source[src_idx].append(norm)
            scores_per_source[src_idx].append(float(det.confidence))
            labels_per_source[src_idx].append(0)
            source_labels.setdefault(src_idx, det.label)
        if not any(boxes_per_source):
            continue
        effective_iou = float(kind_iou_thr.get(kind, iou_thr))
        fused_boxes, fused_scores, _ = wbf_fn(
            boxes_per_source,
            scores_per_source,
            labels_per_source,
            weights=weights,
            iou_thr=effective_iou,
            skip_box_thr=0.0,
        )
        label_guess = ""
        for _, label in sorted(source_labels.items()):
            if label:
                label_guess = label
                break
        for box, score in zip(fused_boxes, fused_scores):
            x1 = float(box[0]) * width
            y1 = float(box[1]) * height
            x2 = float(box[2]) * width
            y2 = float(box[3]) * height
            if x2 <= x1 or y2 <= y1:
                continue
            fused.append(
                Detection(
                    kind=kind,
                    label=label_guess,
                    confidence=float(score),
                    box=Box(
                        x1=int(round(x1)),
                        y1=int(round(y1)),
                        x2=int(round(x2)),
                        y2=int(round(y2)),
                    ).clamp(width, height),
                    source="wbf",
                )
            )
    return fused


def _nms_by_kind(detections: list[Detection], iou_threshold: float) -> list[Detection]:
    kept: list[Detection] = []
    for detection in sorted(detections, key=lambda item: item.confidence, reverse=True):
        if all(
            detection.kind != kept_detection.kind or _iou(detection.box, kept_detection.box) < iou_threshold
            for kept_detection in kept
        ):
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
