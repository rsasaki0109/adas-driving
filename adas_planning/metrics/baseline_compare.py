from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from adas_planning.config import config_hash, load_config
from adas_planning.io.perception_adapter import adapt_perception_document
from adas_planning.metrics.offline import (
    COMPARE_SCHEMA_VERSION,
    METRICS_SCHEMA_VERSION,
    compute_offline_metrics,
    validate_metrics_artifact,
    write_metrics_artifact,
)
from adas_planning.pipeline import PlanningPipeline

DEFAULT_BASELINE_CONFIGS: dict[str, str] = {
    "default": "configs/planning/default.yaml",
    "conservative": "configs/planning/conservative.yaml",
    "aggressive_demo": "configs/planning/aggressive_demo.yaml",
}

DELTA_METRICS = (
    "target_path_valid_rate",
    "behavior_output_rate",
    "behavior_switch_count",
    "behavior_switch_count_per_min",
    "warning_frame_rate",
    "target_speed_output_rate",
    "mean_confidence",
)


def compare_planning_configs(
    perception_payload: dict[str, Any],
    config_entries: dict[str, str],
    *,
    source: str | None = None,
) -> dict[str, Any]:
    planning_inputs = adapt_perception_document(perception_payload)
    configs_payload: dict[str, Any] = {}
    for name, config_path in config_entries.items():
        config = load_config(config_path)
        pipeline = PlanningPipeline(config)
        results = [pipeline.plan(planning_input) for planning_input in planning_inputs]
        metrics = compute_offline_metrics(results)
        configs_payload[name] = {
            "config_path": config_path,
            "config_hash": config_hash(config),
            "metrics": metrics,
            "metrics_artifact": {
                "schema_version": METRICS_SCHEMA_VERSION,
                "source": source,
                "config": config_path,
                "config_hash": config_hash(config),
                "producer": "adas_planning",
                "metrics": metrics,
            },
            "sample_behaviors": [result.behavior.value for result in results[:5]],
        }

    return {
        "schema_version": COMPARE_SCHEMA_VERSION,
        "source": source,
        "baseline": "default" if "default" in configs_payload else next(iter(configs_payload)),
        "configs": configs_payload,
        "deltas_vs_baseline": _compute_deltas(configs_payload),
    }


def write_baseline_compare_artifact(
    path: str | Path,
    comparison: dict[str, Any],
    *,
    producer: str = "adas_planning",
) -> None:
    payload = dict(comparison)
    payload["producer"] = producer
    validate_baseline_compare_artifact(payload)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def load_baseline_compare_artifact(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    validate_baseline_compare_artifact(payload)
    return payload


def validate_baseline_compare_artifact(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != COMPARE_SCHEMA_VERSION:
        raise ValueError(f"unsupported compare schema: {payload.get('schema_version')}")
    configs = payload.get("configs")
    if not isinstance(configs, dict) or not configs:
        raise ValueError("baseline compare artifact must include non-empty configs")
    for name, entry in configs.items():
        if not isinstance(entry, dict):
            raise ValueError(f"invalid config entry: {name}")
        metrics_artifact = entry.get("metrics_artifact")
        if metrics_artifact is not None:
            validate_metrics_artifact(metrics_artifact)
        elif "metrics" not in entry:
            raise ValueError(f"config entry missing metrics: {name}")


def write_per_config_metrics_artifacts(
    comparison: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    for name, entry in comparison.get("configs", {}).items():
        artifact = entry.get("metrics_artifact")
        if not artifact:
            continue
        path = root / f"{name}_metrics.json"
        write_metrics_artifact(
            path,
            artifact["metrics"],
            source=artifact.get("source"),
            config_path=artifact.get("config"),
            config_hash=artifact.get("config_hash"),
            producer=str(artifact.get("producer", "adas_planning")),
        )
        written[name] = str(path)
    return written


def _compute_deltas(configs_payload: dict[str, Any]) -> dict[str, dict[str, float]]:
    baseline_name = "default" if "default" in configs_payload else next(iter(configs_payload))
    baseline_metrics = configs_payload[baseline_name]["metrics"]
    deltas: dict[str, dict[str, float]] = {}
    for name, entry in configs_payload.items():
        if name == baseline_name:
            continue
        metrics = entry["metrics"]
        row: dict[str, float] = {}
        for metric in DELTA_METRICS:
            left = metrics.get(metric)
            right = baseline_metrics.get(metric)
            if isinstance(left, (int, float)) and isinstance(right, (int, float)):
                row[metric] = float(left) - float(right)
        deltas[name] = row
    return deltas
