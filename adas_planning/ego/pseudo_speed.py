from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from adas_planning.types import PlanningInput


@dataclass(frozen=True)
class EgoSpeedEstimate:
    speed_mps: float | None
    source: str
    confidence_factor: float = 1.0

    @property
    def is_available(self) -> bool:
        return self.speed_mps is not None


def resolve_ego_speed(
    planning_input: PlanningInput,
    config: dict[str, Any],
    *,
    relative_velocity_mps: float | None = None,
) -> EgoSpeedEstimate:
    """Resolve ego speed from measurement, config default, or optional closing-rate hint."""
    if planning_input.ego_speed_mps is not None:
        speed = float(planning_input.ego_speed_mps)
        return EgoSpeedEstimate(speed_mps=speed, source="measurement", confidence_factor=1.0)

    pseudo_cfg = config.get("pseudo_ego_speed", {})
    config_factor = float(pseudo_cfg.get("config_confidence_factor", 0.65))
    max_mps = float(pseudo_cfg.get("max_mps", 40.0))

    default_mps = _config_default_mps(config)
    if default_mps is not None:
        return EgoSpeedEstimate(
            speed_mps=min(max_mps, float(default_mps)),
            source="config_default",
            confidence_factor=config_factor,
        )

    closing_cfg = pseudo_cfg.get("closing_rate") or {}
    if closing_cfg.get("enabled", True) and relative_velocity_mps is not None:
        min_range_rate = float(closing_cfg.get("min_range_rate_mps", 0.5))
        closing_factor = float(closing_cfg.get("confidence_factor", 0.45))
        if relative_velocity_mps < -min_range_rate:
            pseudo_speed = min(max_mps, abs(relative_velocity_mps))
            return EgoSpeedEstimate(
                speed_mps=pseudo_speed,
                source="closing_rate",
                confidence_factor=closing_factor,
            )

    return EgoSpeedEstimate(speed_mps=None, source="none", confidence_factor=0.0)


def apply_confidence_factor(base_confidence: float, estimate: EgoSpeedEstimate) -> float:
    return min(1.0, max(0.0, base_confidence * estimate.confidence_factor))


def merge_ego_speed_estimates(*estimates: EgoSpeedEstimate) -> EgoSpeedEstimate:
    priority = {"measurement": 3, "config_default": 2, "closing_rate": 1, "none": 0}
    best = EgoSpeedEstimate(speed_mps=None, source="none", confidence_factor=0.0)
    for estimate in estimates:
        if priority.get(estimate.source, 0) > priority.get(best.source, 0):
            best = estimate
    return best


def estimate_from_debug(debug: dict[str, Any]) -> EgoSpeedEstimate | None:
    source = debug.get("ego_speed_source")
    if not source:
        return None
    return EgoSpeedEstimate(
        speed_mps=debug.get("ego_speed_mps"),
        source=str(source),
        confidence_factor=float(debug.get("ego_speed_confidence_factor", 1.0)),
    )


def _config_default_mps(config: dict[str, Any]) -> float | None:
    pseudo_cfg = config.get("pseudo_ego_speed", {})
    if pseudo_cfg.get("default_mps") is not None:
        return float(pseudo_cfg["default_mps"])
    for section in ("lead_follow", "vru_yield"):
        value = config.get(section, {}).get("default_ego_speed_mps")
        if value is not None:
            return float(value)
    return None
