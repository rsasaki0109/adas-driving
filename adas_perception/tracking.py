from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from adas_perception.types import Box, Detection


def _box_to_cxcywh(box: Box) -> tuple[float, float, float, float]:
    w = float(box.x2 - box.x1)
    h = float(box.y2 - box.y1)
    return (float(box.x1) + 0.5 * w, float(box.y1) + 0.5 * h, w, h)


def _cxcywh_to_box(state: np.ndarray) -> Box:
    cx, cy, w, h = (float(v) for v in state[:4])
    w = max(1.0, w)
    h = max(1.0, h)
    return Box(
        x1=int(round(cx - 0.5 * w)),
        y1=int(round(cy - 0.5 * h)),
        x2=int(round(cx + 0.5 * w)),
        y2=int(round(cy + 0.5 * h)),
    )


class _BoxKalmanFilter:
    """Constant-velocity Kalman filter over (cx, cy, w, h).

    Noise standard deviations scale with box height (DeepSORT convention),
    so the filter behaves consistently for near/large and far/small objects.
    Unlike the one-step linear extrapolation, the velocity estimate is
    smoothed over the track's lifetime and keeps advancing while the track
    coasts through missed frames.
    """

    _STD_POSITION = 1.0 / 20.0
    _STD_VELOCITY = 1.0 / 160.0

    def __init__(self, box: Box):
        cx, cy, w, h = _box_to_cxcywh(box)
        self.x = np.array([cx, cy, w, h, 0.0, 0.0, 0.0, 0.0], dtype=float)
        self.F = np.eye(8, dtype=float)
        self.F[:4, 4:] = np.eye(4, dtype=float)
        self.H = np.eye(4, 8, dtype=float)
        ref = max(1.0, h)
        std = [2.0 * self._STD_POSITION * ref] * 4 + [10.0 * self._STD_VELOCITY * ref] * 4
        self.P = np.diag(np.square(np.array(std, dtype=float)))

    def predict(self) -> Box:
        ref = max(1.0, float(self.x[3]))
        std = [self._STD_POSITION * ref] * 4 + [self._STD_VELOCITY * ref] * 4
        process_noise = np.diag(np.square(np.array(std, dtype=float)))
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + process_noise
        return _cxcywh_to_box(self.x)

    def update(self, box: Box) -> None:
        z = np.array(_box_to_cxcywh(box), dtype=float)
        ref = max(1.0, float(z[3]))
        measurement_noise = np.diag(
            np.square(np.full(4, self._STD_POSITION * ref, dtype=float))
        )
        innovation_cov = self.H @ self.P @ self.H.T + measurement_noise
        gain = self.P @ self.H.T @ np.linalg.inv(innovation_cov)
        self.x = self.x + gain @ (z - self.H @ self.x)
        self.P = (np.eye(8, dtype=float) - gain @ self.H) @ self.P


@dataclass
class _Track:
    track_id: int
    kind: str
    label: str
    box: Box
    prev_box: Box | None = None
    missed: int = 0
    kf: _BoxKalmanFilter | None = field(default=None, repr=False)

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
    """Greedy IoU tracker with optional linear motion prediction,
    centroid-distance fallback, and ByteTrack-style two-stage matching.

    Config keys:
      enabled (bool, default False)
      iou_threshold (float, default 0.30)
      max_missed (int, default 8)
      kinds (list[str], default [vehicle, pedestrian])
      motion_prediction (bool, default True): propagate the last observed
        box by one step before IoU-matching.
      motion_model (str, default "kalman"): "kalman" predicts with a
        constant-velocity Kalman filter (smoothed velocity, keeps coasting
        through missed frames); "linear" keeps the previous one-step
        extrapolation from the last two observed boxes. Only used when
        motion_prediction is true.
      centroid_distance_fraction (float, default 0.0): fallback threshold on
        centroid distance, expressed as fraction of the mean box diagonal.
        0 disables the fallback (matches the previous IoU-only behavior).
      two_stage (bool, default False): ByteTrack-style two-stage matching.
        When enabled, detections are split into HIGH (>= high_score_threshold)
        and LOW (>= low_score_threshold AND < high_score_threshold). Stage 1
        matches all tracks against HIGH detections; stage 2 matches still-
        unmatched tracks against LOW detections to recover from temporary
        score dips. Only HIGH detections seed new tracks (LOW is too noisy).
      high_score_threshold (float, default 0.50)
      low_score_threshold (float, default 0.10)
    """

    def __init__(self, config: dict[str, Any]):
        self.enabled = bool(config.get("enabled", False))
        self.iou_threshold = float(config.get("iou_threshold", 0.30))
        self.max_missed = int(config.get("max_missed", 8))
        self.kinds = set(config.get("kinds", ["vehicle", "pedestrian"]))
        self.motion_prediction = bool(config.get("motion_prediction", True))
        self.motion_model = str(config.get("motion_model", "kalman")).lower()
        self.centroid_distance_fraction = float(config.get("centroid_distance_fraction", 0.0))
        self.two_stage = bool(config.get("two_stage", False))
        self.high_score_threshold = float(config.get("high_score_threshold", 0.50))
        self.low_score_threshold = float(config.get("low_score_threshold", 0.10))
        self._next_id = 1
        self._tracks: list[_Track] = []

    def reset(self) -> None:
        self._next_id = 1
        self._tracks.clear()

    def _match_box(self, track: _Track) -> Box:
        if not self.motion_prediction:
            return track.box
        if self.motion_model == "kalman":
            if track.kf is None:
                track.kf = _BoxKalmanFilter(track.box)
            # Advances the filter by one step; called once per track per
            # update() because match boxes are computed once up front.
            return track.kf.predict()
        return track.predicted_box()

    def update(self, detections: list[Detection]) -> list[Detection]:
        if not self.enabled:
            return detections

        tracked_indexes = [idx for idx, detection in enumerate(detections) if detection.kind in self.kinds]
        if self.two_stage:
            high_indexes = [
                idx for idx in tracked_indexes if detections[idx].confidence >= self.high_score_threshold
            ]
            low_indexes = [
                idx for idx in tracked_indexes
                if self.low_score_threshold <= detections[idx].confidence < self.high_score_threshold
            ]
        else:
            high_indexes = list(tracked_indexes)
            low_indexes = []

        assigned_detection_indexes: set[int] = set()
        assigned_track_indexes: set[int] = set()
        updated = list(detections)
        existing_track_count = len(self._tracks)
        match_boxes = [self._match_box(track) for track in self._tracks]

        # Stage 1: IoU match against HIGH-confidence detections.
        self._iou_associate(
            detections,
            match_boxes,
            candidate_detection_indexes=high_indexes,
            assigned_track_indexes=assigned_track_indexes,
            assigned_detection_indexes=assigned_detection_indexes,
            updated=updated,
        )

        # Stage 2: IoU match the still-unmatched tracks against LOW-conf
        # detections to recover scoring dips. (No-op if two_stage disabled.)
        if low_indexes:
            self._iou_associate(
                detections,
                match_boxes,
                candidate_detection_indexes=low_indexes,
                assigned_track_indexes=assigned_track_indexes,
                assigned_detection_indexes=assigned_detection_indexes,
                updated=updated,
            )

        # Centroid-distance fallback (independent of two_stage). Considers
        # all detections (both HIGH and LOW), since the fallback already
        # has its own normalized distance gate.
        if self.centroid_distance_fraction > 0.0:
            self._centroid_associate(
                detections,
                match_boxes,
                candidate_detection_indexes=tracked_indexes,
                assigned_track_indexes=assigned_track_indexes,
                assigned_detection_indexes=assigned_detection_indexes,
                updated=updated,
            )

        # Birth new tracks only from HIGH detections (LOW is too noisy to
        # seed). When two_stage is off this still equals tracked_indexes.
        for detection_index in high_indexes:
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

    def _iou_associate(
        self,
        detections: list[Detection],
        match_boxes: list[Box],
        *,
        candidate_detection_indexes: list[int],
        assigned_track_indexes: set[int],
        assigned_detection_indexes: set[int],
        updated: list[Detection],
    ) -> None:
        candidates: list[tuple[float, int, int]] = []
        for track_index, track in enumerate(self._tracks):
            if track_index in assigned_track_indexes:
                continue
            match_box = match_boxes[track_index]
            for detection_index in candidate_detection_indexes:
                if detection_index in assigned_detection_indexes:
                    continue
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
            if track.kf is not None:
                track.kf.update(detection.box)
            assigned_track_indexes.add(track_index)
            assigned_detection_indexes.add(detection_index)
            updated[detection_index] = replace(detection, track_id=track.track_id)

    def _centroid_associate(
        self,
        detections: list[Detection],
        match_boxes: list[Box],
        *,
        candidate_detection_indexes: list[int],
        assigned_track_indexes: set[int],
        assigned_detection_indexes: set[int],
        updated: list[Detection],
    ) -> None:
        fallback_candidates: list[tuple[float, int, int]] = []
        for track_index, track in enumerate(self._tracks):
            if track_index in assigned_track_indexes:
                continue
            match_box = match_boxes[track_index]
            for detection_index in candidate_detection_indexes:
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
            if track.kf is not None:
                track.kf.update(detection.box)
            assigned_track_indexes.add(track_index)
            assigned_detection_indexes.add(detection_index)
            updated[detection_index] = replace(detection, track_id=track.track_id)


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
