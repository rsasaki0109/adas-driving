#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adas_planning.metrics.baseline_compare import compare_planning_configs, write_baseline_compare_artifact
from adas_planning.metrics.scenario_eval import evaluate_scenario, evaluate_scenarios_dir, load_scenario


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

    config_entries = _parse_config_entries(args.configs)
    comparison_payload = compare_planning_configs(
        payload,
        {name: path for name, path in config_entries},
        source=str(args.input),
    )
    write_baseline_compare_artifact(args.output, comparison_payload)
    print(json.dumps(comparison_payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
