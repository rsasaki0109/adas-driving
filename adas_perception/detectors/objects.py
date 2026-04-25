from __future__ import annotations

from math import ceil
from typing import Any

import cv2
import numpy as np

from adas_perception.types import Box, Detection


def create_object_detector(config: dict[str, Any]):
    backend = str(config.get("backend", "torchvision")).lower()
    if backend == "torchvision":
        return TorchVisionObjectDetector(config)
    if backend in {"ultralytics", "yolo"}:
        return UltralyticsObjectDetector(config)
    raise ValueError("Unsupported object detector backend: " f"{backend}. Supported: torchvision, ultralytics")


COCO_FALLBACK_CATEGORIES = [
    "__background__",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "N/A",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "N/A",
    "backpack",
    "umbrella",
    "N/A",
    "N/A",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "N/A",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "N/A",
    "dining table",
    "N/A",
    "N/A",
    "toilet",
    "N/A",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "N/A",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

SIZE_BUCKET_NAMES = ("tiny", "small", "medium", "large")
DEFAULT_SIZE_BUCKET_AREAS = (0.0005, 0.0025, 0.01)


class TorchVisionObjectDetector:
    """PyTorch/TorchVision COCO detector filtered to ADAS-like classes."""

    SUPPORTED_MODELS = {
        "ssdlite320_mobilenet_v3_large",
        "fasterrcnn_mobilenet_v3_large_320_fpn",
        "fasterrcnn_mobilenet_v3_large_fpn",
        "fasterrcnn_resnet50_fpn",
        "fasterrcnn_resnet50_fpn_v2",
        "retinanet_resnet50_fpn",
        "retinanet_resnet50_fpn_v2",
    }

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.score_threshold = float(config.get("score_threshold", 0.45))
        self.score_thresholds_by_kind = _build_kind_score_thresholds(config.get("score_thresholds_by_kind", {}))
        self.min_score_threshold = min([self.score_threshold, *self.score_thresholds_by_kind.values()])
        self.max_detections = int(config.get("max_detections", 80))
        self.label_to_kind = self._build_label_map(config.get("class_groups", {}))

        import torch

        self.torch = torch
        self.device = self._resolve_device(str(config.get("device", "auto")))
        self.model, self.categories = self._load_model(
            model_name=str(config.get("model", "ssdlite320_mobilenet_v3_large")),
            weights_name=str(config.get("weights", "DEFAULT")),
        )
        self.model.to(self.device)
        self.model.eval()

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        height, width = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        tensor = tensor.to(self.device)

        with self.torch.inference_mode():
            output = self.model([tensor])[0]

        boxes = output.get("boxes", [])
        labels = output.get("labels", [])
        scores = output.get("scores", [])

        detections: list[Detection] = []
        for box_tensor, label_tensor, score_tensor in zip(boxes, labels, scores):
            score = float(score_tensor.detach().cpu().item())
            if score < self.min_score_threshold:
                continue
            label_idx = int(label_tensor.detach().cpu().item())
            label = self._label_name(label_idx)
            kind = self.label_to_kind.get(label.lower())
            if kind is None:
                continue
            if score < _score_threshold_for_kind(kind, self.score_threshold, self.score_thresholds_by_kind):
                continue

            x1, y1, x2, y2 = [int(round(v)) for v in box_tensor.detach().cpu().tolist()]
            box = Box(x1=x1, y1=y1, x2=x2, y2=y2).clamp(width, height)
            if box.area <= 0:
                continue
            detections.append(
                Detection(
                    kind=kind,
                    label=label,
                    confidence=score,
                    box=box,
                    source="torchvision",
                )
            )
            if len(detections) >= self.max_detections:
                break
        return detections

    def _resolve_device(self, configured: str) -> str:
        if configured == "auto":
            return "cuda" if self.torch.cuda.is_available() else "cpu"
        return configured

    def _load_model(self, model_name: str, weights_name: str):
        if model_name not in self.SUPPORTED_MODELS:
            choices = ", ".join(sorted(self.SUPPORTED_MODELS))
            raise ValueError(f"Unsupported object model: {model_name}. Supported: {choices}")

        from torchvision.models import get_model, get_model_weights

        weights = None
        categories = COCO_FALLBACK_CATEGORIES
        normalized_weights = weights_name.strip()
        if normalized_weights.upper() == "DEFAULT":
            weights_enum = get_model_weights(model_name)
            weights = weights_enum.DEFAULT
            categories = list(weights.meta.get("categories", COCO_FALLBACK_CATEGORIES))
        elif normalized_weights and normalized_weights.lower() not in {"none", "false"}:
            weights_enum = get_model_weights(model_name)
            try:
                weights = weights_enum[normalized_weights.upper()]
            except KeyError as exc:
                choices = ", ".join(member.name for member in weights_enum)
                raise ValueError(
                    f"Unsupported weights setting for {model_name}: {weights_name}. "
                    f"Supported: DEFAULT, none, {choices}"
                ) from exc
            categories = list(weights.meta.get("categories", COCO_FALLBACK_CATEGORIES))

        model_kwargs = {"weights": weights}
        if weights is None:
            model_kwargs["weights_backbone"] = None
        model = get_model(model_name, **model_kwargs)
        return model, categories

    def _label_name(self, label_idx: int) -> str:
        if 0 <= label_idx < len(self.categories):
            return str(self.categories[label_idx])
        if 0 <= label_idx < len(COCO_FALLBACK_CATEGORIES):
            return COCO_FALLBACK_CATEGORIES[label_idx]
        return f"class_{label_idx}"

    @staticmethod
    def _build_label_map(class_groups: dict[str, list[str]]) -> dict[str, str]:
        label_to_kind: dict[str, str] = {}
        for kind, labels in class_groups.items():
            for label in labels:
                label_to_kind[str(label).lower()] = str(kind)
        return label_to_kind


class UltralyticsObjectDetector:
    """Ultralytics YOLO detector filtered to ADAS-like classes."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.score_threshold = float(config.get("score_threshold", 0.35))
        self.score_thresholds_by_kind = _build_kind_score_thresholds(config.get("score_thresholds_by_kind", {}))
        self.score_thresholds_by_size = _build_size_score_thresholds(config.get("score_thresholds_by_size", {}))
        self.size_bucket_areas = _build_size_bucket_areas(config.get("size_bucket_areas", DEFAULT_SIZE_BUCKET_AREAS))
        self.min_score_threshold = min(
            [
                self.score_threshold,
                *self.score_thresholds_by_kind.values(),
                *_flatten_size_thresholds(self.score_thresholds_by_size),
            ]
        )
        self.iou_threshold = float(config.get("iou_threshold", 0.50))
        self.max_detections = int(config.get("max_detections", 80))
        self.image_size = int(config.get("image_size", 640))
        self.augment = bool(config.get("augment", False))
        self.preprocess = str(config.get("preprocess", "")).lower() or None
        self.label_to_kind = TorchVisionObjectDetector._build_label_map(config.get("class_groups", {}))
        self.tile_config = _build_tile_config(config.get("tile_inference", {}), self.image_size, self.max_detections)

        import torch
        from ultralytics import YOLO

        self.torch = torch
        self.device = self._resolve_device(str(config.get("device", "auto")))
        self.model_name = str(config.get("model", "yolov8n.pt"))
        self.model = YOLO(self.model_name)

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        if self.tile_config["enabled"]:
            return self._detect_with_tiles(frame_bgr)
        return self._predict_frame(
            frame_bgr,
            image_size=self.image_size,
            max_detections=self.max_detections,
            x_offset=0,
            y_offset=0,
            full_width=frame_bgr.shape[1],
            full_height=frame_bgr.shape[0],
            source="ultralytics",
        )

    def _detect_with_tiles(self, frame_bgr: np.ndarray) -> list[Detection]:
        height, width = frame_bgr.shape[:2]
        detections: list[Detection] = []
        if self.tile_config["include_full_frame"]:
            detections.extend(
                self._predict_frame(
                    frame_bgr,
                    image_size=self.image_size,
                    max_detections=self.max_detections,
                    x_offset=0,
                    y_offset=0,
                    full_width=width,
                    full_height=height,
                    source="ultralytics",
                )
            )

        full_area = width * height
        for x1, y1, x2, y2 in _tile_windows(
            width=width,
            height=height,
            rows=int(self.tile_config["rows"]),
            cols=int(self.tile_config["cols"]),
            overlap_ratio=float(self.tile_config["overlap_ratio"]),
        ):
            crop = frame_bgr[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            tile_detections = self._predict_frame(
                crop,
                image_size=int(self.tile_config["image_size"]),
                max_detections=int(self.tile_config["max_detections"]),
                x_offset=x1,
                y_offset=y1,
                full_width=width,
                full_height=height,
                source="ultralytics_tile",
            )
            for detection in tile_detections:
                if detection.kind not in self.tile_config["kinds"]:
                    continue
                tile_threshold = _score_threshold_for_kind(
                    detection.kind,
                    float(self.tile_config["score_threshold"]),
                    self.tile_config["score_thresholds_by_kind"],
                )
                if detection.confidence < tile_threshold:
                    continue
                area_ratio = detection.box.area / max(full_area, 1)
                if area_ratio > float(self.tile_config["max_box_area_ratio"]):
                    continue
                detections.append(detection)

        return sorted(detections, key=lambda item: item.confidence, reverse=True)[: self.tile_config["combined_max_detections"]]

    def _predict_frame(
        self,
        frame_bgr: np.ndarray,
        *,
        image_size: int,
        max_detections: int,
        x_offset: int,
        y_offset: int,
        full_width: int,
        full_height: int,
        source: str,
    ) -> list[Detection]:
        predict_frame = frame_bgr
        if self.preprocess == "clahe":
            predict_frame = _apply_clahe(frame_bgr)
        results = self.model.predict(
            source=predict_frame,
            imgsz=image_size,
            conf=self.min_score_threshold,
            iou=self.iou_threshold,
            max_det=max_detections,
            device=self.device,
            augment=self.augment,
            verbose=False,
        )
        if not results:
            return []

        result = results[0]
        names = result.names or getattr(self.model, "names", {})
        detections: list[Detection] = []
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return detections

        for box in boxes:
            score = float(box.conf.detach().cpu().item())
            if score < self.min_score_threshold:
                continue
            class_id = int(box.cls.detach().cpu().item())
            label = str(names.get(class_id, f"class_{class_id}"))
            kind = self.label_to_kind.get(label.lower())
            if kind is None:
                continue
            x1, y1, x2, y2 = [int(round(v)) for v in box.xyxy[0].detach().cpu().tolist()]
            det_box = Box(
                x1=x1 + x_offset,
                y1=y1 + y_offset,
                x2=x2 + x_offset,
                y2=y2 + y_offset,
            ).clamp(full_width, full_height)
            if det_box.area <= 0:
                continue
            threshold = _score_threshold_for_detection(
                kind=kind,
                box=det_box,
                image_area=full_width * full_height,
                default_threshold=self.score_threshold,
                thresholds_by_kind=self.score_thresholds_by_kind,
                thresholds_by_size=self.score_thresholds_by_size,
                size_bucket_areas=self.size_bucket_areas,
            )
            if score < threshold:
                continue
            detections.append(
                Detection(
                    kind=kind,
                    label=label,
                    confidence=score,
                    box=det_box,
                    source=source,
                )
            )
            if len(detections) >= self.max_detections:
                break
        return detections

    def _resolve_device(self, configured: str) -> str:
        if configured == "auto":
            return "cuda" if self.torch.cuda.is_available() else "cpu"
        return configured


def _build_kind_score_thresholds(raw_thresholds: Any) -> dict[str, float]:
    if not isinstance(raw_thresholds, dict):
        return {}
    return {str(kind): float(threshold) for kind, threshold in raw_thresholds.items()}


def _score_threshold_for_kind(kind: str, default_threshold: float, thresholds_by_kind: dict[str, float]) -> float:
    return thresholds_by_kind.get(kind, default_threshold)


def _build_size_score_thresholds(raw_thresholds: Any) -> dict[str, dict[str, float]]:
    if not isinstance(raw_thresholds, dict):
        return {}
    thresholds: dict[str, dict[str, float]] = {}
    for bucket, raw_value in raw_thresholds.items():
        bucket_name = str(bucket)
        if bucket_name not in SIZE_BUCKET_NAMES:
            continue
        if isinstance(raw_value, dict):
            thresholds[bucket_name] = {
                str(kind): float(threshold)
                for kind, threshold in raw_value.items()
            }
        else:
            thresholds[bucket_name] = {"default": float(raw_value)}
    return thresholds


def _build_size_bucket_areas(raw_thresholds: Any) -> tuple[float, float, float]:
    if raw_thresholds is None:
        return DEFAULT_SIZE_BUCKET_AREAS
    if not isinstance(raw_thresholds, (list, tuple)) or len(raw_thresholds) != 3:
        raise ValueError("size_bucket_areas must contain exactly three thresholds.")
    values = tuple(float(value) for value in raw_thresholds)
    if any(value <= 0.0 or value >= 1.0 for value in values):
        raise ValueError("size_bucket_areas values must be between 0 and 1.")
    if tuple(sorted(values)) != values:
        raise ValueError("size_bucket_areas values must be sorted ascending.")
    return values


def _flatten_size_thresholds(thresholds_by_size: dict[str, dict[str, float]]) -> list[float]:
    return [
        float(threshold)
        for bucket_thresholds in thresholds_by_size.values()
        for threshold in bucket_thresholds.values()
    ]


def _score_threshold_for_detection(
    *,
    kind: str,
    box: Box,
    image_area: int,
    default_threshold: float,
    thresholds_by_kind: dict[str, float],
    thresholds_by_size: dict[str, dict[str, float]],
    size_bucket_areas: tuple[float, float, float],
) -> float:
    threshold = _score_threshold_for_kind(kind, default_threshold, thresholds_by_kind)
    bucket = _size_bucket(box, image_area, size_bucket_areas)
    bucket_thresholds = thresholds_by_size.get(bucket, {})
    return float(bucket_thresholds.get(kind, bucket_thresholds.get("default", threshold)))


def _size_bucket(box: Box, image_area: int, thresholds: tuple[float, float, float]) -> str:
    area_ratio = float(box.area) / float(max(image_area, 1))
    if area_ratio < thresholds[0]:
        return "tiny"
    if area_ratio < thresholds[1]:
        return "small"
    if area_ratio < thresholds[2]:
        return "medium"
    return "large"


_CLAHE = None


def _apply_clahe(frame_bgr: np.ndarray) -> np.ndarray:
    global _CLAHE
    import cv2
    if _CLAHE is None:
        _CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = _CLAHE.apply(l)
    enhanced = cv2.merge([l, a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


def _build_tile_config(raw_config: Any, image_size: int, max_detections: int) -> dict[str, Any]:
    if not isinstance(raw_config, dict):
        raw_config = {}
    kinds = raw_config.get("kinds", ["pedestrian", "vehicle", "traffic_sign", "traffic_light"])
    return {
        "enabled": bool(raw_config.get("enabled", False)),
        "include_full_frame": bool(raw_config.get("include_full_frame", True)),
        "rows": max(1, int(raw_config.get("rows", 2))),
        "cols": max(1, int(raw_config.get("cols", 2))),
        "overlap_ratio": max(0.0, min(0.8, float(raw_config.get("overlap_ratio", 0.20)))),
        "image_size": int(raw_config.get("image_size", image_size)),
        "max_detections": int(raw_config.get("max_detections", max_detections)),
        "combined_max_detections": int(raw_config.get("combined_max_detections", max_detections * 3)),
        "max_box_area_ratio": float(raw_config.get("max_box_area_ratio", 0.010)),
        "kinds": {str(kind) for kind in kinds},
        "score_threshold": float(raw_config.get("score_threshold", 0.35)),
        "score_thresholds_by_kind": _build_kind_score_thresholds(raw_config.get("score_thresholds_by_kind", {})),
    }


def _tile_windows(
    *,
    width: int,
    height: int,
    rows: int,
    cols: int,
    overlap_ratio: float,
) -> list[tuple[int, int, int, int]]:
    x_windows = _axis_windows(width, cols, overlap_ratio)
    y_windows = _axis_windows(height, rows, overlap_ratio)
    return [(x1, y1, x2, y2) for y1, y2 in y_windows for x1, x2 in x_windows]


def _axis_windows(length: int, count: int, overlap_ratio: float) -> list[tuple[int, int]]:
    if count <= 1 or length <= 1:
        return [(0, length)]
    tile_size = min(length, int(ceil(length / (count - (count - 1) * overlap_ratio))))
    stride = max(1, int(round(tile_size * (1.0 - overlap_ratio))))
    starts = [min(index * stride, max(0, length - tile_size)) for index in range(count)]
    starts[-1] = max(0, length - tile_size)
    unique_starts = sorted(set(starts))
    return [(start, min(length, start + tile_size)) for start in unique_starts]
