from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from adas_planning.types import Behavior, PlannerProposal, PlanningResult, Warning


BEHAVIOR_PRIORITY = {
    Behavior.STOP_FOR_RED: 100,
    Behavior.YIELD_VRU: 90,
    Behavior.FOLLOW_LEAD: 30,
    Behavior.LANE_DEPARTURE: 15,
    Behavior.KEEP_LANE: 10,
    Behavior.GO_CAUTION: 5,
    Behavior.CAUTION: 3,
    Behavior.UNKNOWN: 0,
}


@dataclass
class LaneDepartureState:
    active: bool = False


def arbitrate_proposals(
    proposals: list[PlannerProposal | None],
    *,
    lane_proposal: PlannerProposal | None,
) -> PlanningResult:
    valid = [proposal for proposal in proposals if proposal is not None]
    if not valid and lane_proposal is None:
        return PlanningResult(behavior=Behavior.UNKNOWN, confidence=0.0)

    chosen = max(valid, key=lambda proposal: (proposal.priority, BEHAVIOR_PRIORITY.get(proposal.behavior, 0))) if valid else lane_proposal
    if chosen is None:
        return PlanningResult(behavior=Behavior.UNKNOWN, confidence=0.0)

    target_path = list(lane_proposal.target_path) if lane_proposal and lane_proposal.target_path else list(chosen.target_path)
    target_path_px = list(lane_proposal.target_path_px) if lane_proposal and lane_proposal.target_path_px else list(chosen.target_path_px)

    speed_caps = [proposal.target_speed_mps for proposal in valid if proposal.target_speed_mps is not None]
    target_speed = min(speed_caps) if speed_caps else chosen.target_speed_mps

    warnings: list[Warning] = []
    for proposal in valid:
        warnings.extend(proposal.warnings)

    confidence_values = [proposal.confidence for proposal in valid if proposal.confidence > 0]
    confidence = min(confidence_values) if confidence_values else chosen.confidence

    debug: dict[str, Any] = {"selected_behavior": chosen.behavior.value}
    for proposal in valid:
        if proposal.debug:
            debug.setdefault("modules", []).append(
                {"behavior": proposal.behavior.value, **proposal.debug}
            )

    return PlanningResult(
        behavior=chosen.behavior,
        target_path=target_path,
        target_path_px=target_path_px,
        target_speed_mps=target_speed,
        warnings=warnings,
        confidence=confidence,
        lead_object_id=chosen.lead_object_id,
        stop_reason=chosen.stop_reason,
        debug=debug,
    )
