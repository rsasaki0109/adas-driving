from __future__ import annotations

from typing import Any

from adas_planning.ego.pseudo_speed import apply_confidence_factor, resolve_ego_speed
from adas_planning.memory.track_history import TrackHistory
from adas_planning.types import Behavior, DetectionInput, PlannerProposal, PlanningInput, Warning


def _box_center_x(detection: DetectionInput) -> float:
    return (detection.box["x1"] + detection.box["x2"]) / 2.0


def _in_lane_corridor(detection: DetectionInput, planning_input: PlanningInput, corridor_ratio: float) -> bool:
    width = max(1, planning_input.image_width)
    center_x = _box_center_x(detection)
    margin = width * (1.0 - corridor_ratio) / 2.0
    return margin <= center_x <= width - margin


def _lead_candidates(planning_input: PlanningInput, config: dict[str, Any]) -> list[DetectionInput]:
    follow_cfg = config.get("lead_follow", {})
    corridor_ratio = float(follow_cfg.get("lane_corridor_width_ratio", 0.55))
    min_confidence = float(follow_cfg.get("min_vehicle_confidence", 0.25))
    candidates = [
        det
        for det in planning_input.detections
        if det.kind == "vehicle"
        and det.confidence >= min_confidence
        and _in_lane_corridor(det, planning_input, corridor_ratio)
        and det.distance_m is not None
    ]
    candidates.sort(key=lambda det: det.distance_m if det.distance_m is not None else 1e9)
    return candidates


def compute_lead_follow(
    planning_input: PlanningInput,
    config: dict[str, Any],
    track_history: TrackHistory,
) -> PlannerProposal | None:
    follow_cfg = config.get("lead_follow", {})
    warning_distance_m = float(follow_cfg.get("warning_distance_m", 12.0))
    critical_distance_m = float(follow_cfg.get("critical_distance_m", 6.0))
    ttc_warning_s = float(follow_cfg.get("ttc_warning_s", 2.5))

    candidates = _lead_candidates(planning_input, config)
    if not candidates:
        return None

    lead = candidates[0]
    distance_m = float(lead.distance_m or 0.0)
    track_history.update(
        frame_id=planning_input.frame_id,
        timestamp_s=planning_input.timestamp_s,
        track_id=lead.track_id,
        distance_m=lead.distance_m,
        ground_z_m=lead.ground_position_m[1] if lead.ground_position_m else None,
    )
    relative_velocity = track_history.relative_velocity_mps(lead.track_id)
    ttc_s = None
    if relative_velocity is not None and relative_velocity < -0.05:
        ttc_s = distance_m / abs(relative_velocity)

    warnings: list[Warning] = []
    if distance_m <= warning_distance_m:
        warnings.append(
            Warning(
                code="FOLLOW_DISTANCE",
                message=f"Lead vehicle within {distance_m:.1f} m",
                severity="warning" if distance_m <= critical_distance_m else "info",
                value=distance_m,
            )
        )
    if ttc_s is not None and ttc_s <= ttc_warning_s:
        warnings.append(
            Warning(
                code="FOLLOW_DISTANCE",
                message=f"TTC {ttc_s:.1f} s",
                severity="warning",
                value=ttc_s,
            )
        )

    ego_estimate = resolve_ego_speed(
        planning_input,
        config,
        relative_velocity_mps=relative_velocity,
    )
    ego_speed = ego_estimate.speed_mps

    target_speed = None
    if ego_speed is not None:
        if distance_m <= critical_distance_m:
            target_speed = max(0.0, ego_speed * 0.35)
        elif distance_m <= warning_distance_m:
            target_speed = max(0.0, ego_speed * 0.65)

    confidence = apply_confidence_factor(min(1.0, lead.confidence), ego_estimate)

    return PlannerProposal(
        behavior=Behavior.FOLLOW_LEAD,
        target_speed_mps=target_speed,
        warnings=warnings,
        confidence=confidence,
        lead_object_id=lead.track_id,
        priority=30 if warnings else 20,
        debug={
            "distance_m": distance_m,
            "ttc_s": ttc_s,
            "relative_velocity_mps": relative_velocity,
            "ego_speed_mps": ego_speed,
            "ego_speed_source": ego_estimate.source,
            "ego_speed_confidence_factor": ego_estimate.confidence_factor,
        },
    )
