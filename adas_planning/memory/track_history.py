from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TrackSample:
    frame_id: int
    timestamp_s: float
    distance_m: float | None
    ground_z_m: float | None


@dataclass
class TrackHistory:
    config: dict[str, Any]
    _history: dict[int, list[TrackSample]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        follow_cfg = self.config.get("lead_follow", {})
        self.max_history = int(follow_cfg.get("track_history_frames", 12))

    def update(
        self,
        *,
        frame_id: int,
        timestamp_s: float,
        track_id: int | None,
        distance_m: float | None,
        ground_z_m: float | None,
    ) -> None:
        if track_id is None or distance_m is None:
            return
        samples = self._history.setdefault(track_id, [])
        samples.append(
            TrackSample(
                frame_id=frame_id,
                timestamp_s=timestamp_s,
                distance_m=distance_m,
                ground_z_m=ground_z_m,
            )
        )
        if len(samples) > self.max_history:
            del samples[: len(samples) - self.max_history]

    def relative_velocity_mps(self, track_id: int | None) -> float | None:
        if track_id is None:
            return None
        samples = self._history.get(track_id) or []
        if len(samples) < 2:
            return None
        first = samples[-2]
        last = samples[-1]
        dt = last.timestamp_s - first.timestamp_s
        if dt <= 1e-3 or first.distance_m is None or last.distance_m is None:
            return None
        return (last.distance_m - first.distance_m) / dt

    def reset_track(self, track_id: int) -> None:
        self._history.pop(track_id, None)
