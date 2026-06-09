from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from adas_planning.types import Behavior, PlanningResult

METRICS_SCHEMA_VERSION = "planning_metrics.v0.1"


def compute_offline_metrics(results: list[PlanningResult]) -> dict[str, Any]:
    if not results:
        return {"frame_count": 0}

    valid_path = sum(1 for result in results if result.target_path_px)
    behaviors = [result.behavior.value if isinstance(result.behavior, Behavior) else str(result.behavior) for result in results]
    behavior_switches = sum(1 for idx in range(1, len(behaviors)) if behaviors[idx] != behaviors[idx - 1])
    warning_frames = sum(1 for result in results if result.warnings)
    target_speed_frames = sum(1 for result in results if result.target_speed_mps is not None)
    ego_sources = [str(result.debug.get("ego_speed_source", "none")) for result in results]
    durations = [results[idx].timestamp_s - results[idx - 1].timestamp_s for idx in range(1, len(results))]
    duration_s = max(results[-1].timestamp_s - results[0].timestamp_s, sum(durations))

    return {
        "frame_count": len(results),
        "target_path_valid_rate": valid_path / len(results),
        "behavior_output_rate": sum(1 for behavior in behaviors if behavior != Behavior.UNKNOWN.value) / len(results),
        "behavior_switch_count": behavior_switches,
        "behavior_switch_count_per_min": behavior_switches / max(duration_s / 60.0, 1e-6),
        "warning_frame_rate": warning_frames / len(results),
        "target_speed_output_rate": target_speed_frames / len(results),
        "mean_confidence": sum(result.confidence for result in results) / len(results),
        "behavior_counts": _count_values(behaviors),
        "ego_speed_source_counts": _count_values(ego_sources),
    }


def write_metrics_artifact(
    path: str | Path,
    metrics: dict[str, Any],
    *,
    source: str | None = None,
    config_path: str | None = None,
    config_hash: str | None = None,
    producer: str = "adas_planning",
) -> None:
    payload = {
        "schema_version": METRICS_SCHEMA_VERSION,
        "source": source,
        "config": config_path,
        "config_hash": config_hash,
        "producer": producer,
        "metrics": metrics,
    }
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _count_values(values: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts
