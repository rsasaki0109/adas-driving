from __future__ import annotations

from typing import Any

from adas_planning.ego.pseudo_speed import apply_confidence_factor, resolve_ego_speed
from adas_planning.types import Behavior, DetectionInput, PlannerProposal, PlanningInput, Warning

VRU_KINDS = {"pedestrian", "cyclist", "bicycle", "motorcycle", "rider"}


def _in_lane_corridor(detection: DetectionInput, planning_input: PlanningInput, corridor_ratio: float) -> bool:
    width = max(1, planning_input.image_width)
    center_x = (detection.box["x1"] + detection.box["x2"]) / 2.0
    margin = width * (1.0 - corridor_ratio) / 2.0
    return margin <= center_x <= width - margin


def compute_vru_yield(planning_input: PlanningInput, config: dict[str, Any]) -> PlannerProposal | None:
    vru_cfg = config.get("vru_yield", {})
    corridor_ratio = float(vru_cfg.get("lane_corridor_width_ratio", 0.70))
    warning_distance_m = float(vru_cfg.get("warning_distance_m", 25.0))
    speed_cap_ratio = float(vru_cfg.get("speed_cap_ratio", 0.5))
    min_confidence = float(vru_cfg.get("min_confidence", 0.30))

    candidates = [
        det
        for det in planning_input.detections
        if det.kind in VRU_KINDS
        and det.confidence >= min_confidence
        and _in_lane_corridor(det, planning_input, corridor_ratio)
        and det.distance_m is not None
        and det.distance_m <= warning_distance_m
    ]
    if not candidates:
        return None

    closest = min(candidates, key=lambda det: det.distance_m or 1e9)
    distance_m = float(closest.distance_m or 0.0)
    ego_estimate = resolve_ego_speed(planning_input, config)
    ego_speed = ego_estimate.speed_mps
    target_speed = ego_speed * speed_cap_ratio if ego_speed is not None else None
    confidence = apply_confidence_factor(min(1.0, closest.confidence), ego_estimate)
    return PlannerProposal(
        behavior=Behavior.YIELD_VRU,
        target_speed_mps=target_speed,
        warnings=[
            Warning(
                code="YIELD_VRU",
                message=f"Yield VRU within {distance_m:.1f} m",
                severity="warning",
                value=distance_m,
            )
        ],
        confidence=confidence,
        priority=90,
        debug={
            "distance_m": distance_m,
            "kind": closest.kind,
            "ego_speed_mps": ego_speed,
            "ego_speed_source": ego_estimate.source,
            "ego_speed_confidence_factor": ego_estimate.confidence_factor,
        },
    )
