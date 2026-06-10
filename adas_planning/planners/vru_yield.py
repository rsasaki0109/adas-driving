from __future__ import annotations

from typing import Any

from adas_planning.ego.pseudo_speed import apply_confidence_factor, resolve_ego_speed
from adas_planning.memory.track_history import TrackHistory
from adas_planning.types import Behavior, DetectionInput, PlannerProposal, PlanningInput, Warning

VRU_KINDS = {"pedestrian", "cyclist", "bicycle", "motorcycle", "rider"}


def _in_lane_corridor(detection: DetectionInput, planning_input: PlanningInput, corridor_ratio: float) -> bool:
    width = max(1, planning_input.image_width)
    center_x = (detection.box["x1"] + detection.box["x2"]) / 2.0
    margin = width * (1.0 - corridor_ratio) / 2.0
    return margin <= center_x <= width - margin


def compute_vru_yield(
    planning_input: PlanningInput,
    config: dict[str, Any],
    track_history: TrackHistory | None = None,
) -> PlannerProposal | None:
    vru_cfg = config.get("vru_yield", {})
    corridor_ratio = float(vru_cfg.get("lane_corridor_width_ratio", 0.70))
    warning_distance_m = float(vru_cfg.get("warning_distance_m", 25.0))
    speed_cap_ratio = float(vru_cfg.get("speed_cap_ratio", 0.5))
    ttc_warning_s = float(vru_cfg.get("ttc_warning_s", 3.0))
    ttc_speed_cap_ratio = float(vru_cfg.get("ttc_speed_cap_ratio", 0.25))
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

    relative_velocity = None
    ttc_s = None
    if track_history is not None:
        track_history.update(
            frame_id=planning_input.frame_id,
            timestamp_s=planning_input.timestamp_s,
            track_id=closest.track_id,
            distance_m=closest.distance_m,
            ground_z_m=closest.ground_position_m[1] if closest.ground_position_m else None,
        )
        relative_velocity = track_history.relative_velocity_mps(closest.track_id)
        if relative_velocity is not None and relative_velocity < -0.05:
            ttc_s = distance_m / abs(relative_velocity)
    ttc_critical = ttc_s is not None and ttc_s <= ttc_warning_s

    warnings = [
        Warning(
            code="YIELD_VRU",
            message=f"Yield VRU within {distance_m:.1f} m",
            severity="warning",
            value=distance_m,
        )
    ]
    if ttc_critical:
        warnings.append(
            Warning(
                code="YIELD_VRU",
                message=f"VRU TTC {ttc_s:.1f} s",
                severity="warning",
                value=ttc_s,
            )
        )

    ego_estimate = resolve_ego_speed(planning_input, config)
    ego_speed = ego_estimate.speed_mps
    target_speed = None
    if ego_speed is not None:
        cap_ratio = ttc_speed_cap_ratio if ttc_critical else speed_cap_ratio
        target_speed = ego_speed * cap_ratio
    confidence = apply_confidence_factor(min(1.0, closest.confidence), ego_estimate)
    return PlannerProposal(
        behavior=Behavior.YIELD_VRU,
        target_speed_mps=target_speed,
        warnings=warnings,
        confidence=confidence,
        priority=90,
        debug={
            "distance_m": distance_m,
            "kind": closest.kind,
            "ttc_s": ttc_s,
            "relative_velocity_mps": relative_velocity,
            "ego_speed_mps": ego_speed,
            "ego_speed_source": ego_estimate.source,
            "ego_speed_confidence_factor": ego_estimate.confidence_factor,
        },
    )
