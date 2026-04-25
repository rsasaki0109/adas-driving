"""ONNX-based lane segmentation detector.

Pluggable backend for `LaneDetector` configured via `lane.backend: segmentation`.
Reads a binary (or per-channel) lane mask from a pre-trained segmentation model
exported to ONNX, then post-processes the mask into left/right LaneLine
polylines compatible with the rest of the pipeline.

By design this module does NOT ship a model file: the user supplies a
pretrained ONNX (e.g. TuSimple/CULane-trained LaneNet, UltraFastLane, or any
binary-mask lane segmentation that takes an RGB image and outputs a mask).

Required config:
  lane.backend: "segmentation"
  lane.segmentation.model_path: path to .onnx file
  lane.segmentation.input_size: [H, W] expected by the model (e.g. [288, 800])

Optional config (defaults shown):
  lane.segmentation.output_name: ""      # ONNX output name to read. Empty
                                          # picks the first output. Useful for
                                          # multi-head models like TwinLiteNet
                                          # that emit drivable_area + lane_line.
  lane.segmentation.lane_channel: -1     # -1 means binary mask; >= 0 picks a
                                          # specific output channel for multi-class
                                          # segmentation (e.g. 1 for the lane
                                          # class in [bg, lane] logits).
  lane.segmentation.lane_threshold: 0.5  # probability threshold to binarize.
  lane.segmentation.min_blob_pixels: 200 # drop tiny connected components.
  lane.segmentation.polyline_samples: 20 # samples per fitted polyline.
  lane.segmentation.providers: ["CUDAExecutionProvider", "CPUExecutionProvider"]
                                          # onnxruntime providers in priority order.
  lane.segmentation.input_layout: "NCHW"  # or "NHWC".
  lane.segmentation.normalize:
    mean: [0.485, 0.456, 0.406]   # ImageNet by default
    std:  [0.229, 0.224, 0.225]
    scale: 255.0                  # divide pixel values by this before mean/std

Note: blob assignment to left/right is heuristic (left = the blob with
smaller mean x in the lower 60% of the image; right = the one with larger
mean x). For more than two blobs, the two with the largest area win.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from adas_perception.types import LaneLine, LaneResult, Point


class LaneSegmentationDetector:
    def __init__(self, config: dict[str, Any]):
        seg_cfg = config.get("segmentation", {}) or {}
        self.model_path = seg_cfg.get("model_path")
        if not self.model_path:
            raise ValueError("lane.segmentation.model_path is required when backend=segmentation")
        if not Path(self.model_path).is_file():
            raise FileNotFoundError(f"lane.segmentation.model_path not found: {self.model_path}")

        input_size = seg_cfg.get("input_size", [288, 800])
        if len(input_size) != 2:
            raise ValueError("lane.segmentation.input_size must be [H, W]")
        self.input_h, self.input_w = int(input_size[0]), int(input_size[1])

        self.output_name = str(seg_cfg.get("output_name", "") or "")
        self.lane_channel = int(seg_cfg.get("lane_channel", -1))
        self.lane_threshold = float(seg_cfg.get("lane_threshold", 0.5))
        self.min_blob_pixels = int(seg_cfg.get("min_blob_pixels", 200))
        self.polyline_samples = int(seg_cfg.get("polyline_samples", 20))
        self.input_layout = str(seg_cfg.get("input_layout", "NCHW")).upper()

        normalize = seg_cfg.get("normalize", {}) or {}
        self.norm_mean = np.array(normalize.get("mean", [0.485, 0.456, 0.406]), dtype=np.float32)
        self.norm_std = np.array(normalize.get("std", [0.229, 0.224, 0.225]), dtype=np.float32)
        self.norm_scale = float(normalize.get("scale", 255.0))

        providers = seg_cfg.get(
            "providers", ["CUDAExecutionProvider", "CPUExecutionProvider"]
        )

        # Lazy-import onnxruntime so the rest of the pipeline still works
        # when the segmentation backend is not used.
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "onnxruntime is required for lane.backend=segmentation. "
                "Install it via `pip install onnxruntime-gpu` (CUDA) or `onnxruntime` (CPU)."
            ) from exc

        self._session = ort.InferenceSession(str(self.model_path), providers=providers)
        self._input_name = self._session.get_inputs()[0].name
        output_names = [out.name for out in self._session.get_outputs()]
        if self.output_name:
            if self.output_name not in output_names:
                raise ValueError(
                    f"lane.segmentation.output_name '{self.output_name}' not in model outputs {output_names}"
                )
            self._output_index = output_names.index(self.output_name)
        else:
            self._output_index = 0

    # Mirror LaneDetector.detect() signature so the pipeline can swap them.
    def detect(self, frame_bgr: np.ndarray) -> LaneResult:
        if frame_bgr is None or frame_bgr.size == 0:
            return LaneResult()
        height, width = frame_bgr.shape[:2]

        net_input = self._preprocess(frame_bgr)
        try:
            outputs = self._session.run(None, {self._input_name: net_input})
        except Exception:
            return LaneResult()

        mask = self._extract_lane_mask(outputs[self._output_index])  # (input_h, input_w) float in [0,1]
        if mask is None or mask.size == 0:
            return LaneResult()
        mask_resized = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
        binary = (mask_resized >= self.lane_threshold).astype(np.uint8) * 255

        return self._mask_to_lanes(binary, width, height)

    # --- internal helpers -------------------------------------------------

    def _preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR)
        arr = resized.astype(np.float32) / max(self.norm_scale, 1e-6)
        arr = (arr - self.norm_mean) / np.where(self.norm_std == 0, 1.0, self.norm_std)
        if self.input_layout == "NCHW":
            arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW
        arr = np.expand_dims(arr, axis=0).astype(np.float32)
        return arr

    def _extract_lane_mask(self, output: np.ndarray) -> np.ndarray | None:
        if output is None:
            return None
        # Handle common shapes:
        # (N, 1, H, W) sigmoid map
        # (N, C, H, W) class logits → softmax across C, take `lane_channel`
        # (N, H, W) raw mask
        a = np.asarray(output)
        if a.ndim == 4 and a.shape[1] == 1:
            mask = a[0, 0].astype(np.float32)
        elif a.ndim == 4:
            channel = self.lane_channel if 0 <= self.lane_channel < a.shape[1] else 1
            logits = a[0].astype(np.float32)  # (C, H, W)
            mask = _softmax_along_axis(logits, axis=0)[channel]
        elif a.ndim == 3:
            mask = a[0].astype(np.float32)
        else:
            return None
        if mask.max() > 1.5:
            mask = _sigmoid(mask)
        return np.clip(mask, 0.0, 1.0)

    def _mask_to_lanes(self, binary_mask: np.ndarray, width: int, height: int) -> LaneResult:
        n_components, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
        candidates: list[tuple[int, int]] = []  # (area, label)
        for label_id in range(1, n_components):
            area = int(stats[label_id, cv2.CC_STAT_AREA])
            if area < self.min_blob_pixels:
                continue
            candidates.append((area, label_id))
        if not candidates:
            return LaneResult()
        candidates.sort(reverse=True)
        keep = [label_id for _, label_id in candidates[:6]]  # top 6 by area, then split

        polylines: list[tuple[float, list[Point]]] = []  # (mean_x_in_lower_half, points)
        for label_id in keep:
            ys, xs = np.where(labels == label_id)
            if len(ys) == 0:
                continue
            poly_pts = self._fit_polyline(xs, ys, width, height)
            if not poly_pts:
                continue
            lower_mask = ys >= int(height * 0.6)
            mean_x = float(np.mean(xs[lower_mask])) if lower_mask.any() else float(np.mean(xs))
            polylines.append((mean_x, poly_pts))

        if not polylines:
            return LaneResult()

        polylines.sort(key=lambda kv: kv[0])
        # Pick best left = leftmost, best right = rightmost.
        left_points = polylines[0][1]
        right_points = polylines[-1][1] if len(polylines) > 1 else []

        lines: list[LaneLine] = []
        if left_points:
            lines.append(
                LaneLine(
                    side="left",
                    points=(left_points[0], left_points[-1]),
                    confidence=1.0,
                    polyline=left_points,
                )
            )
        if right_points and right_points is not left_points:
            lines.append(
                LaneLine(
                    side="right",
                    points=(right_points[0], right_points[-1]),
                    confidence=1.0,
                    polyline=right_points,
                )
            )

        polygon: list[Point] = []
        if left_points and right_points:
            sorted_left = sorted(left_points, key=lambda p: -p[1])
            sorted_right = sorted(right_points, key=lambda p: -p[1])
            polygon = sorted_left + list(reversed(sorted_right))

        return LaneResult(lines=lines, raw_segments=[], polygon=polygon)

    def _fit_polyline(self, xs: np.ndarray, ys: np.ndarray, width: int, height: int) -> list[Point]:
        if len(ys) < 6:
            return []
        try:
            coeffs = np.polyfit(ys.astype(np.float64), xs.astype(np.float64), 2)
        except Exception:
            return []
        ymin = max(int(np.percentile(ys, 5)), 0)
        ymax = min(int(np.percentile(ys, 95)), height - 1)
        if ymax <= ymin:
            return []
        sample_count = max(2, self.polyline_samples)
        sampled_ys = np.linspace(ymin, ymax, sample_count)
        sampled_xs = np.polyval(coeffs, sampled_ys)
        polyline: list[Point] = []
        for x, y in zip(sampled_xs, sampled_ys):
            xi = int(round(float(x)))
            yi = int(round(float(y)))
            xi = max(0, min(width - 1, xi))
            yi = max(0, min(height - 1, yi))
            polyline.append((xi, yi))
        # Order bottom -> top so downstream polygon construction is consistent.
        polyline.sort(key=lambda p: -p[1])
        return polyline


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _softmax_along_axis(x: np.ndarray, axis: int) -> np.ndarray:
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x_shifted)
    return e / np.sum(e, axis=axis, keepdims=True)
