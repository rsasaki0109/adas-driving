from __future__ import annotations

import json
from pathlib import Path

import pytest

from adas_planning.metrics.baseline_compare import (
    compare_planning_configs,
    load_baseline_compare_artifact,
    validate_baseline_compare_artifact,
    write_baseline_compare_artifact,
)
from adas_planning.metrics.offline import (
    METRICS_SCHEMA_VERSION,
    compute_offline_metrics,
    load_metrics_artifact,
    validate_metrics_artifact,
    write_metrics_artifact,
)
from adas_planning.types import Behavior, PlanningResult


FIXTURE = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "planning" / "red_light_sequence.json"


def test_write_and_load_metrics_artifact(tmp_path: Path):
    metrics = compute_offline_metrics(
        [
            PlanningResult(frame_id=0, timestamp_s=0.0, behavior=Behavior.STOP_FOR_RED, confidence=0.8),
            PlanningResult(frame_id=1, timestamp_s=0.1, behavior=Behavior.STOP_FOR_RED, confidence=0.7),
        ]
    )
    path = tmp_path / "metrics.json"
    write_metrics_artifact(
        path,
        metrics,
        source="fixture",
        config_path="configs/planning/default.yaml",
        config_hash="abc123",
    )
    loaded = load_metrics_artifact(path)
    assert loaded["schema_version"] == METRICS_SCHEMA_VERSION
    assert loaded["metrics"]["frame_count"] == 2


def test_baseline_compare_artifact_round_trip(tmp_path: Path):
    with FIXTURE.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    comparison = compare_planning_configs(
        payload,
        {
            "default": "configs/planning/default.yaml",
            "conservative": "configs/planning/conservative.yaml",
        },
        source=str(FIXTURE),
    )
    validate_baseline_compare_artifact(comparison)
    path = tmp_path / "compare.json"
    write_baseline_compare_artifact(path, comparison)
    loaded = load_baseline_compare_artifact(path)
    assert "default" in loaded["configs"]
    assert "conservative" in loaded["deltas_vs_baseline"]
