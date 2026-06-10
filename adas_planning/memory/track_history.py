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
        """Least-squares slope of distance over the whole history window.

        Using every sample instead of only the last two smooths out
        single-frame distance noise, which otherwise dominates TTC.
        """
        if track_id is None:
            return None
        samples = [
            sample
            for sample in (self._history.get(track_id) or [])
            if sample.distance_m is not None
        ]
        if len(samples) < 2:
            return None
        t0 = samples[0].timestamp_s
        times = [sample.timestamp_s - t0 for sample in samples]
        distances = [float(sample.distance_m) for sample in samples]
        count = len(samples)
        mean_t = sum(times) / count
        mean_d = sum(distances) / count
        variance = sum((t - mean_t) ** 2 for t in times)
        if variance <= 1e-6:
            return None
        covariance = sum(
            (t - mean_t) * (d - mean_d) for t, d in zip(times, distances)
        )
        return covariance / variance

    def reset_track(self, track_id: int) -> None:
        self._history.pop(track_id, None)
