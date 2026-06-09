#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adas_planning.config import config_hash, load_config
from adas_planning.io.perception_adapter import adapt_perception_document
from adas_planning.metrics.offline import compute_offline_metrics
from adas_planning.metrics.scenario_eval import evaluate_scenario, evaluate_scenarios_dir, load_scenario
from adas_planning.pipeline import PlanningPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare planning configs on saved perception JSON.")
    parser.add_argument("--input", default=None, help="Perception JSON path.")
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["default=configs/planning/default.yaml", "conservative=configs/planning/conservative.yaml"],
        metavar="NAME=PATH",
        help="One or more name=config pairs.",
    )
    parser.add_argument("--output", required=True, help="Comparison JSON output path.")
    parser.add_argument(
        "--scenarios-dir",
        default=None,
        help="Optional scenarios/ directory for YAML-based correctness checks.",
    )
    parser.add_argument(
        "--scenario",
        default=None,
        help="Optional single scenario YAML path (overrides --scenarios-dir).",
    )
    return parser.parse_args()


def _parse_config_entries(entries: list[str]) -> list[tuple[str, str]]:
    parsed: list[tuple[str, str]] = []
    for entry in entries:
        if "=" not in entry:
            raise SystemExit(f"Invalid config entry: {entry}")
        name, config_path = entry.split("=", 1)
        parsed.append((name, config_path))
    return parsed


def main() -> int:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.scenario or args.scenarios_dir:
        if args.scenario:
            scenario_result = evaluate_scenario(load_scenario(args.scenario), base_dir=Path(args.scenario).parent)
            payload = {
                "mode": "scenario",
                "scenario": args.scenario,
                "passed": scenario_result.passed,
                "result": {
                    "name": scenario_result.name,
                    "passed": scenario_result.passed,
                    "metrics": scenario_result.metrics,
                    "checks": [
                        {
                            "name": check.name,
                            "passed": check.passed,
                            "detail": check.detail,
                        }
                        for check in scenario_result.checks
                    ],
                },
            }
        else:
            payload = {"mode": "scenarios_dir", **evaluate_scenarios_dir(args.scenarios_dir)}
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0 if payload.get("all_passed", payload.get("passed", False)) else 1

    if not args.input:
        raise SystemExit("--input is required unless --scenarios-dir or --scenario is set")

    with Path(args.input).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    planning_inputs = adapt_perception_document(payload)

    comparison: dict[str, object] = {"input": str(args.input), "configs": {}}
    for name, config_path in _parse_config_entries(args.configs):
        config = load_config(config_path)
        pipeline = PlanningPipeline(config)
        results = [pipeline.plan(planning_input) for planning_input in planning_inputs]
        comparison["configs"][name] = {
            "config_path": config_path,
            "config_hash": config_hash(config),
            "metrics": compute_offline_metrics(results),
            "sample_behaviors": [result.behavior.value for result in results[:5]],
        }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(comparison, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(comparison, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
