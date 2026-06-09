from __future__ import annotations

import json
from pathlib import Path

import pytest

from adas_planning.io.driving_replay import (
    build_driving_replay_document,
    load_driving_replay_document,
    validate_driving_replay_document,
    write_driving_replay_document,
)
from adas_planning.metrics.benchmark_export import (
    build_benchmark_export_artifact,
    export_baseline_compare_csv,
    export_baseline_compare_markdown,
)
from adas_planning.metrics.baseline_compare import compare_planning_configs


FIXTURE = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "planning" / "red_light_sequence.json"


def test_benchmark_export_csv_and_markdown():
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
    csv_text = export_baseline_compare_csv(comparison)
    markdown = export_baseline_compare_markdown(comparison)
    artifact = build_benchmark_export_artifact(comparison)
    assert "config_name" in csv_text
    assert "Planning baseline benchmark" in markdown
    assert artifact["schema_version"] == "planning_benchmark_export.v0.1"
    assert len(artifact["rows"]) == 2


def test_driving_replay_round_trip(tmp_path: Path):
    perception_path = FIXTURE
    planning_path = tmp_path / "planning.json"
    replay_path = tmp_path / "replay.json"
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "scripts/replay_planning_json.py",
            "--input",
            str(perception_path),
            "--config",
            "configs/planning/default.yaml",
            "--output",
            str(planning_path),
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[2],
    )
    document = write_driving_replay_document(
        replay_path,
        perception_path=perception_path,
        planning_path=planning_path,
    )
    validate_driving_replay_document(document)
    loaded = load_driving_replay_document(replay_path)
    assert loaded["frames"]
    assert "planning" in loaded["frames"][0]
