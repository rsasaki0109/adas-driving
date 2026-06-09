from __future__ import annotations

from pathlib import Path

from adas_planning.metrics.scenario_eval import evaluate_scenarios_dir


SCENARIOS_DIR = Path(__file__).resolve().parents[2] / "scenarios"


def test_scenarios_dir_all_pass():
    summary = evaluate_scenarios_dir(SCENARIOS_DIR)
    assert summary["scenario_count"] >= 7
    assert summary["all_passed"], summary
