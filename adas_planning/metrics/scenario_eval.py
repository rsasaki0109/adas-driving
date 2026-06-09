from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from adas_planning.config import config_hash, load_config
from adas_planning.io.perception_adapter import adapt_perception_document
from adas_planning.metrics.offline import compute_offline_metrics
from adas_planning.pipeline import PlanningPipeline
from adas_planning.types import Behavior, PlanningResult


@dataclass
class ScenarioCheckResult:
    name: str
    passed: bool
    detail: str
    expected: dict[str, Any] = field(default_factory=dict)
    actual: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScenarioResult:
    name: str
    passed: bool
    config_path: str
    perception_json: str
    checks: list[ScenarioCheckResult] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)


def load_scenario(path: Path | str) -> dict[str, Any]:
    scenario_path = Path(path)
    with scenario_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Scenario must be a mapping: {scenario_path}")
    return payload


def _resolve_path(base_dir: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    candidates = [base_dir / path, base_dir.parent / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return base_dir / path


def _behavior_value(result: PlanningResult) -> str:
    return result.behavior.value if isinstance(result.behavior, Behavior) else str(result.behavior)


def _warning_codes(result: PlanningResult) -> set[str]:
    return {warning.code for warning in result.warnings}


def _evaluate_check(
    check: dict[str, Any],
    *,
    results: list[PlanningResult],
    metrics: dict[str, Any],
    config_metrics: dict[str, dict[str, Any]] | None = None,
) -> ScenarioCheckResult:
    name = str(check.get("name") or check.get("type") or "check")
    check_type = str(check.get("type", "behavior_rate"))

    if check_type == "behavior_rate":
        frame_indices = [int(idx) for idx in check.get("frame_indices", [])]
        expected = str(check.get("behavior", Behavior.UNKNOWN.value))
        min_rate = float(check.get("min_rate", 1.0))
        selected = [results[idx] for idx in frame_indices if 0 <= idx < len(results)]
        matches = sum(1 for result in selected if _behavior_value(result) == expected)
        rate = matches / max(len(selected), 1)
        passed = rate >= min_rate
        return ScenarioCheckResult(
            name=name,
            passed=passed,
            detail=f"behavior={expected} rate={rate:.3f} (min {min_rate})",
            expected={"behavior": expected, "min_rate": min_rate, "frame_indices": frame_indices},
            actual={"rate": rate, "matches": matches, "selected_frames": len(selected)},
        )

    if check_type == "warning_rate":
        frame_indices = [int(idx) for idx in check.get("frame_indices", [])]
        code = str(check.get("code", ""))
        min_rate = float(check.get("min_rate", 1.0))
        selected = [results[idx] for idx in frame_indices if 0 <= idx < len(results)]
        matches = sum(1 for result in selected if code in _warning_codes(result))
        rate = matches / max(len(selected), 1)
        passed = rate >= min_rate
        return ScenarioCheckResult(
            name=name,
            passed=passed,
            detail=f"warning={code} rate={rate:.3f} (min {min_rate})",
            expected={"code": code, "min_rate": min_rate, "frame_indices": frame_indices},
            actual={"rate": rate, "matches": matches, "selected_frames": len(selected)},
        )

    if check_type == "max_behavior_rate":
        frame_indices = [int(idx) for idx in check.get("frame_indices", [])]
        behavior = str(check.get("behavior", Behavior.UNKNOWN.value))
        max_rate = float(check.get("max_rate", 0.0))
        selected = [results[idx] for idx in frame_indices if 0 <= idx < len(results)]
        matches = sum(1 for result in selected if _behavior_value(result) == behavior)
        rate = matches / max(len(selected), 1)
        passed = rate <= max_rate
        return ScenarioCheckResult(
            name=name,
            passed=passed,
            detail=f"behavior={behavior} rate={rate:.3f} (max {max_rate})",
            expected={"behavior": behavior, "max_rate": max_rate, "frame_indices": frame_indices},
            actual={"rate": rate, "matches": matches, "selected_frames": len(selected)},
        )

    if check_type == "metric_gte":
        metric = str(check.get("metric", ""))
        minimum = float(check.get("min", 0.0))
        actual = float(metrics.get(metric, 0.0))
        passed = actual >= minimum
        return ScenarioCheckResult(
            name=name,
            passed=passed,
            detail=f"{metric}={actual:.3f} (min {minimum})",
            expected={"metric": metric, "min": minimum},
            actual={"value": actual},
        )

    if check_type == "target_speed_rate":
        frame_indices = [int(idx) for idx in check.get("frame_indices", [])]
        min_rate = float(check.get("min_rate", 1.0))
        selected = [results[idx] for idx in frame_indices if 0 <= idx < len(results)]
        matches = sum(1 for result in selected if result.target_speed_mps is not None)
        rate = matches / max(len(selected), 1)
        passed = rate >= min_rate
        return ScenarioCheckResult(
            name=name,
            passed=passed,
            detail=f"target_speed set rate={rate:.3f} (min {min_rate})",
            expected={"min_rate": min_rate, "frame_indices": frame_indices},
            actual={"rate": rate, "matches": matches, "selected_frames": len(selected)},
        )

    if check_type == "debug_field_rate":
        frame_indices = [int(idx) for idx in check.get("frame_indices", [])]
        field = str(check.get("field", ""))
        expected_value = check.get("value")
        min_rate = float(check.get("min_rate", 1.0))
        selected = [results[idx] for idx in frame_indices if 0 <= idx < len(results)]
        if expected_value is None:
            matches = sum(1 for result in selected if result.debug.get(field) not in (None, "", "none"))
        else:
            matches = sum(1 for result in selected if result.debug.get(field) == expected_value)
        rate = matches / max(len(selected), 1)
        passed = rate >= min_rate
        return ScenarioCheckResult(
            name=name,
            passed=passed,
            detail=f"debug.{field} rate={rate:.3f} (min {min_rate})",
            expected={"field": field, "value": expected_value, "min_rate": min_rate, "frame_indices": frame_indices},
            actual={"rate": rate, "matches": matches, "selected_frames": len(selected)},
        )

    if check_type == "config_metric_gte":
        if not config_metrics:
            return ScenarioCheckResult(name=name, passed=False, detail="config_metrics missing")
        config_name = str(check.get("config", ""))
        than_config = str(check.get("than_config", ""))
        metric = str(check.get("metric", ""))
        left = float(config_metrics.get(config_name, {}).get(metric, 0.0))
        right = float(config_metrics.get(than_config, {}).get(metric, 0.0))
        passed = left >= right
        return ScenarioCheckResult(
            name=name,
            passed=passed,
            detail=f"{config_name}.{metric}={left:.3f} >= {than_config}.{metric}={right:.3f}",
            expected={"config": config_name, "than_config": than_config, "metric": metric},
            actual={"left": left, "right": right},
        )

    if check_type == "result_valid_all_frames":
        passed = all(0.0 <= result.confidence <= 1.0 for result in results)
        return ScenarioCheckResult(
            name=name,
            passed=passed,
            detail="all frame confidences within [0, 1]" if passed else "invalid confidence detected",
            expected={"valid_confidence": True},
            actual={"frame_count": len(results)},
        )

    return ScenarioCheckResult(name=name, passed=False, detail=f"unknown check type: {check_type}")


def evaluate_scenario(scenario: dict[str, Any], *, base_dir: Path | None = None) -> ScenarioResult:
    base = base_dir or Path(".")
    name = str(scenario.get("name", "unnamed"))
    perception_path = _resolve_path(base, str(scenario["perception_json"]))
    config_path = _resolve_path(base, str(scenario.get("config", "configs/planning/default.yaml")))

    with perception_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    config = load_config(str(config_path))
    pipeline = PlanningPipeline(config)
    planning_inputs = adapt_perception_document(payload)
    results = [pipeline.plan(planning_input) for planning_input in planning_inputs]
    metrics = compute_offline_metrics(results)

    config_metrics: dict[str, dict[str, Any]] | None = None
    if scenario.get("configs"):
        config_metrics = {}
        for config_name, config_value in scenario["configs"].items():
            compare_path = _resolve_path(base, str(config_value))
            compare_config = load_config(str(compare_path))
            compare_pipeline = PlanningPipeline(compare_config)
            compare_results = [compare_pipeline.plan(planning_input) for planning_input in planning_inputs]
            config_metrics[config_name] = compute_offline_metrics(compare_results)

    checks: list[ScenarioCheckResult] = []
    for check in scenario.get("checks", []):
        checks.append(
            _evaluate_check(
                check,
                results=results,
                metrics=metrics,
                config_metrics=config_metrics,
            )
        )

    passed = all(check.passed for check in checks) if checks else True
    return ScenarioResult(
        name=name,
        passed=passed,
        config_path=str(config_path),
        perception_json=str(perception_path),
        checks=checks,
        metrics=metrics,
    )


def evaluate_scenarios_dir(scenarios_dir: Path | str) -> dict[str, Any]:
    root = Path(scenarios_dir)
    scenario_paths = sorted(root.glob("*.yaml"))
    results = [evaluate_scenario(load_scenario(path), base_dir=root) for path in scenario_paths]
    passed_count = sum(1 for result in results if result.passed)
    return {
        "scenarios_dir": str(root),
        "scenario_count": len(results),
        "passed_count": passed_count,
        "failed_count": len(results) - passed_count,
        "all_passed": passed_count == len(results),
        "scenarios": [
            {
                "name": result.name,
                "passed": result.passed,
                "config_path": result.config_path,
                "perception_json": result.perception_json,
                "metrics": result.metrics,
                "checks": [
                    {
                        "name": check.name,
                        "passed": check.passed,
                        "detail": check.detail,
                        "expected": check.expected,
                        "actual": check.actual,
                    }
                    for check in result.checks
                ],
            }
            for result in results
        ],
    }


def scenario_result_to_dict(result: ScenarioResult) -> dict[str, Any]:
    return {
        "name": result.name,
        "passed": result.passed,
        "config_path": result.config_path,
        "perception_json": result.perception_json,
        "metrics": result.metrics,
        "checks": [
            {
                "name": check.name,
                "passed": check.passed,
                "detail": check.detail,
                "expected": check.expected,
                "actual": check.actual,
            }
            for check in result.checks
        ],
    }
