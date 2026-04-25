from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from adas_perception.types import Box, Detection


@dataclass
class _Track:
    track_id: int
    kind: str
    label: str
    box: Box
    prev_box: Box | None = None
    missed: int = 0

    def predicted_box(self) -> Box:
        if self.prev_box is None:
            return self.box
        dx1 = self.box.x1 - self.prev_box.x1
        dy1 = self.box.y1 - self.prev_box.y1
        dx2 = self.box.x2 - self.prev_box.x2
        dy2 = self.box.y2 - self.prev_box.y2
        return Box(
            x1=int(self.box.x1 + dx1),
            y1=int(self.box.y1 + dy1),
            x2=int(self.box.x2 + dx2),
            y2=int(self.box.y2 + dy2),
        )


class SimpleTracker:
    """Greedy IoU tracker with optional linear motion prediction and
    centroid-distance fallback.

    Config keys:
      enabled (bool, default False)
      iou_threshold (float, default 0.30)
      max_missed (int, default 8)
      kinds (list[str], default [vehicle, pedestrian])
      motion_prediction (bool, default True): propagate the last observed
        box by one step using linear velocity before IoU-matching.
      centroid_distance_fraction (float, default 0.0): fallback threshold on
        centroid distance, expressed as fraction of the mean box diagonal.
        0 disables the fallback (matches the previous IoU-only behavior).
    """

    def __init__(self, config: dict[str, Any]):
        self.enabled = bool(config.get("enabled", False))
        self.iou_threshold = float(config.get("iou_threshold", 0.30))
        self.max_missed = int(config.get("max_missed", 8))
        self.kinds = set(config.get("kinds", ["vehicle", "pedestrian"]))
        self.motion_prediction = bool(config.get("motion_prediction", True))
        self.centroid_distance_fraction = float(config.get("centroid_distance_fraction", 0.0))
        self._next_id = 1
        self._tracks: list[_Track] = []

    def reset(self) -> None:
        self._next_id = 1
        self._tracks.clear()

    def _match_box(self, track: _Track) -> Box:
        return track.predicted_box() if self.motion_prediction else track.box

    def update(self, detections: list[Detection]) -> list[Detection]:
        if not self.enabled:
            return detections

        tracked_indexes = [idx for idx, detection in enumerate(detections) if detection.kind in self.kinds]
        assigned_detection_indexes: set[int] = set()
        assigned_track_indexes: set[int] = set()
        updated = list(detections)
        existing_track_count = len(self._tracks)
        match_boxes = [self._match_box(track) for track in self._tracks]

        candidates: list[tuple[float, int, int]] = []
        for track_index, track in enumerate(self._tracks):
            match_box = match_boxes[track_index]
            for detection_index in tracked_indexes:
                detection = detections[detection_index]
                if detection.kind != track.kind:
                    continue
                score = _iou(match_box, detection.box)
                if score >= self.iou_threshold:
                    candidates.append((score, track_index, detection_index))

        for _, track_index, detection_index in sorted(candidates, reverse=True):
            if track_index in assigned_track_indexes or detection_index in assigned_detection_indexes:
                continue
            track = self._tracks[track_index]
            detection = detections[detection_index]
            track.prev_box = track.box
            track.box = detection.box
            track.label = detection.label
            track.missed = 0
            assigned_track_indexes.add(track_index)
            assigned_detection_indexes.add(detection_index)
            updated[detection_index] = replace(detection, track_id=track.track_id)

        if self.centroid_distance_fraction > 0.0:
            fallback_candidates: list[tuple[float, int, int]] = []
            for track_index, track in enumerate(self._tracks):
                if track_index in assigned_track_indexes:
                    continue
                match_box = match_boxes[track_index]
                for detection_index in tracked_indexes:
                    if detection_index in assigned_detection_indexes:
                        continue
                    detection = detections[detection_index]
                    if detection.kind != track.kind:
                        continue
                    mean_diag = 0.5 * (_diagonal(match_box) + _diagonal(detection.box))
                    if mean_diag <= 0:
                        continue
                    norm = _centroid_distance(match_box, detection.box) / mean_diag
                    if norm <= self.centroid_distance_fraction:
                        fallback_candidates.append((-norm, track_index, detection_index))

            for _, track_index, detection_index in sorted(fallback_candidates, reverse=True):
                if track_index in assigned_track_indexes or detection_index in assigned_detection_indexes:
                    continue
                track = self._tracks[track_index]
                detection = detections[detection_index]
                track.prev_box = track.box
                track.box = detection.box
                track.label = detection.label
                track.missed = 0
                assigned_track_indexes.add(track_index)
                assigned_detection_indexes.add(detection_index)
                updated[detection_index] = replace(detection, track_id=track.track_id)

        for detection_index in tracked_indexes:
            if detection_index in assigned_detection_indexes:
                continue
            detection = detections[detection_index]
            track = _Track(
                track_id=self._next_id,
                kind=detection.kind,
                label=detection.label,
                box=detection.box,
            )
            self._next_id += 1
            self._tracks.append(track)
            updated[detection_index] = replace(detection, track_id=track.track_id)

        for track_index, track in enumerate(self._tracks[:existing_track_count]):
            if track_index not in assigned_track_indexes:
                track.missed += 1
        self._tracks = [track for track in self._tracks if track.missed <= self.max_missed]
        return updated


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


def _centroid_distance(a: Box, b: Box) -> float:
    ax = 0.5 * (a.x1 + a.x2)
    ay = 0.5 * (a.y1 + a.y2)
    bx = 0.5 * (b.x1 + b.x2)
    by = 0.5 * (b.y1 + b.y2)
    dx = ax - bx
    dy = ay - by
    return float((dx * dx + dy * dy) ** 0.5)


def _diagonal(box: Box) -> float:
    w = max(0, box.x2 - box.x1)
    h = max(0, box.y2 - box.y1)
    return float((w * w + h * h) ** 0.5)
