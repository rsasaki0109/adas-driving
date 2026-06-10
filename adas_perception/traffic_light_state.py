"""HSV-based traffic-light state classifier.

Given a traffic_light bbox detection (from YOLO / WBF / heuristic) and the
source frame, decide the lit-lamp color (red / yellow / green / off).

Why not a CNN classifier?
- Traffic-light bulbs are intentionally saturated, narrow-hue light sources;
  HSV thresholding inside a known bbox is reliable in normal daylight,
  dawn/dusk, and even rain. We get state info for free without shipping a
  model file.
- ONNX classifier paths can be plugged in later behind the same interface
  if needed. The visualization / serialization sides only care about
  Detection.state being one of the labels below (or None).

Returned state values:
  "red"     - red lamp is dominant
  "yellow"  - yellow lamp is dominant
  "green"   - green lamp is dominant
  "off"     - no lamp lit (housing detected but nothing dominant)
  None      - skipped (disabled, kind not traffic_light, bbox too small).
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from adas_perception.types import Detection

# Class order shared by the HSV path, the trained ONNX classifier, and
# scripts/train_traffic_light_classifier.py. Do not reorder.
STATE_NAMES = ("red", "yellow", "green", "off")

# Input size of the trained ONNX crop classifier (width, height).
ONNX_CROP_SIZE = (32, 64)


# Default HSV ranges in OpenCV convention (H 0..179, S 0..255, V 0..255).
DEFAULT_RANGES: dict[str, list[tuple[tuple[int, int, int], tuple[int, int, int]]]] = {
    "red": [
        ((0, 110, 110), (10, 255, 255)),
        ((170, 110, 110), (179, 255, 255)),
    ],
    "yellow": [
        ((15, 110, 130), (35, 255, 255)),
    ],
    "green": [
        ((40, 80, 110), (90, 255, 255)),
    ],
}


class TrafficLightStateClassifier:
    """Classify the lit-lamp state of traffic_light detections.

    method "hsv" (default) keeps the dependency-free thresholding below.
    method "onnx" runs the trained crop classifier from
    scripts/train_traffic_light_classifier.py via onnxruntime; if the model
    file or onnxruntime is unavailable it falls back to HSV so configs can
    enable it unconditionally.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        config = config or {}
        self.enabled = bool(config.get("enabled", True))
        self.method = str(config.get("method", "hsv")).lower()
        self.onnx_model = config.get("onnx_model", "outputs/models/traffic_light_state.onnx")
        self._onnx_session: Any | None = None
        self._onnx_failed = False
        self.min_box_pixels = int(config.get("min_box_pixels", 64))
        self.min_lit_ratio = float(config.get("min_lit_ratio", 0.02))
        self.tie_margin = float(config.get("tie_margin", 0.15))
        ranges_cfg = config.get("hsv_ranges") or {}
        self.ranges = {
            state: [
                (
                    tuple(item.get("lower", DEFAULT_RANGES.get(state, [((0, 0, 0), (0, 0, 0))])[0][0])),
                    tuple(item.get("upper", DEFAULT_RANGES.get(state, [((0, 0, 0), (0, 0, 0))])[0][1])),
                )
                for item in ranges_cfg.get(state, [])
            ]
            or DEFAULT_RANGES[state]
            for state in ("red", "yellow", "green")
        }

    def classify(self, frame_bgr: np.ndarray, detections: list[Detection]) -> list[Detection]:
        if not self.enabled or not detections:
            return detections
        height, width = frame_bgr.shape[:2]
        if height <= 0 or width <= 0:
            return detections
        if self.method == "onnx" and self._ensure_onnx_session():
            return self._classify_onnx(frame_bgr, detections)
        out: list[Detection] = []
        for det in detections:
            if det.kind != "traffic_light" or det.box.area < self.min_box_pixels:
                out.append(det)
                continue
            new_state = self._state_for_box(frame_bgr, det)
            if new_state is None:
                out.append(det)
                continue
            from dataclasses import replace
            out.append(replace(det, state=new_state))
        return out

    def _ensure_onnx_session(self) -> bool:
        if self._onnx_session is not None:
            return True
        if self._onnx_failed:
            return False
        try:
            import os

            import onnxruntime

            if not os.path.exists(str(self.onnx_model)):
                raise FileNotFoundError(self.onnx_model)
            self._onnx_session = onnxruntime.InferenceSession(
                str(self.onnx_model), providers=["CPUExecutionProvider"]
            )
            return True
        except Exception as error:  # noqa: BLE001 - any failure → HSV fallback
            print(f"[traffic_light_state] onnx unavailable ({error}); falling back to hsv")
            self._onnx_failed = True
            return False

    def _classify_onnx(self, frame_bgr: np.ndarray, detections: list[Detection]) -> list[Detection]:
        from dataclasses import replace

        h_img, w_img = frame_bgr.shape[:2]
        crops: list[np.ndarray] = []
        crop_indexes: list[int] = []
        for index, det in enumerate(detections):
            if det.kind != "traffic_light" or det.box.area < self.min_box_pixels:
                continue
            x1 = max(0, min(w_img - 1, det.box.x1))
            x2 = max(0, min(w_img, det.box.x2))
            y1 = max(0, min(h_img - 1, det.box.y1))
            y2 = max(0, min(h_img, det.box.y2))
            if x2 <= x1 or y2 <= y1:
                continue
            crop = cv2.resize(
                frame_bgr[y1:y2, x1:x2], ONNX_CROP_SIZE, interpolation=cv2.INTER_LINEAR
            )
            crops.append(crop)
            crop_indexes.append(index)
        if not crops:
            return detections
        batch = np.stack(crops).astype(np.float32) / 255.0
        batch = np.transpose(batch, (0, 3, 1, 2))  # NHWC (BGR) → NCHW, matches training
        logits = self._onnx_session.run(None, {"crops": batch})[0]
        states = [STATE_NAMES[int(i)] for i in np.argmax(logits, axis=1)]
        out = list(detections)
        for index, state in zip(crop_indexes, states):
            out[index] = replace(detections[index], state=state)
        return out

    def _state_for_box(self, frame_bgr: np.ndarray, detection: Detection) -> str | None:
        x1, y1, x2, y2 = detection.box.x1, detection.box.y1, detection.box.x2, detection.box.y2
        h_img, w_img = frame_bgr.shape[:2]
        x1 = max(0, min(w_img - 1, x1))
        x2 = max(0, min(w_img, x2))
        y1 = max(0, min(h_img - 1, y1))
        y2 = max(0, min(h_img, y2))
        if x2 <= x1 or y2 <= y1:
            return None
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        total = hsv.shape[0] * hsv.shape[1]
        scores: dict[str, float] = {}
        for state, ranges in self.ranges.items():
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for lower, upper in ranges:
                mask = cv2.bitwise_or(
                    mask,
                    cv2.inRange(hsv, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8)),
                )
            scores[state] = float(np.count_nonzero(mask)) / float(total)

        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        top_state, top_ratio = ranked[0]
        if top_ratio < self.min_lit_ratio:
            return "off"
        # Reject ambiguous states (e.g. red & yellow nearly tied).
        if len(ranked) >= 2 and ranked[1][1] > 0 and (top_ratio - ranked[1][1]) < self.tie_margin * top_ratio:
            return "off"
        return top_state
