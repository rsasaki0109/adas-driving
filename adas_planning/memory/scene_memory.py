from __future__ import annotations

from typing import Any

from adas_planning.types import Behavior, PathPoint, PlannerProposal


class SceneMemory:
    """Temporal hold/decay for lane path and traffic-light state."""

    def __init__(self, config: dict[str, Any]) -> None:
        memory_cfg = config.get("memory", {})
        self.lane_hold_frames = int(memory_cfg.get("lane_hold_frames", 5))
        self.lane_decay_per_frame = float(memory_cfg.get("lane_decay_per_frame", 0.12))
        self._held_path: list[PathPoint] = []
        self._held_path_px: list[tuple[int, int]] = []
        self._held_confidence = 0.0
        self._lane_miss_frames = 0

    def update_lane(
        self,
        proposal: PlannerProposal | None,
        *,
        lane_valid: bool,
    ) -> PlannerProposal | None:
        if lane_valid and proposal and proposal.target_path:
            self._held_path = list(proposal.target_path)
            self._held_path_px = list(proposal.target_path_px)
            self._held_confidence = float(proposal.confidence)
            self._lane_miss_frames = 0
            return proposal

        self._lane_miss_frames += 1
        if self._lane_miss_frames > self.lane_hold_frames or not self._held_path:
            return proposal

        decayed = max(0.05, self._held_confidence - self.lane_decay_per_frame * self._lane_miss_frames)
        held = PlannerProposal(
            behavior=proposal.behavior if proposal else Behavior.CAUTION,
            target_path=list(self._held_path),
            target_path_px=list(self._held_path_px),
            confidence=decayed,
            debug={"lane_hold": True, "miss_frames": self._lane_miss_frames},
        )
        if proposal is None:
            return held
        proposal.target_path = list(self._held_path)
        proposal.target_path_px = list(self._held_path_px)
        proposal.confidence = min(proposal.confidence, decayed)
        proposal.debug = {**proposal.debug, "lane_hold": True, "miss_frames": self._lane_miss_frames}
        return proposal
