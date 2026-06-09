#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adas_planning.metrics.baseline_compare import (
    DEFAULT_BASELINE_CONFIGS,
    compare_planning_configs,
    write_baseline_compare_artifact,
    write_per_config_metrics_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare planning baseline configs with versioned artifacts.")
    parser.add_argument(
        "--input",
        default="examples/fixtures/planning_demo_perception.json",
        help="Perception JSON path.",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=[f"{name}={path}" for name, path in DEFAULT_BASELINE_CONFIGS.items()],
        metavar="NAME=PATH",
        help="One or more name=config pairs.",
    )
    parser.add_argument("--output", required=True, help="Baseline compare JSON output path.")
    parser.add_argument(
        "--metrics-dir",
        default=None,
        help="Optional directory for per-config planning_metrics.v0.1 artifacts.",
    )
    return parser.parse_args()


def _parse_config_entries(entries: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for entry in entries:
        if "=" not in entry:
            raise SystemExit(f"Invalid config entry: {entry}")
        name, config_path = entry.split("=", 1)
        parsed[name.strip()] = config_path.strip()
    return parsed


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    with input_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    comparison = compare_planning_configs(
        payload,
        _parse_config_entries(args.configs),
        source=str(input_path),
    )
    write_baseline_compare_artifact(args.output, comparison)
    if args.metrics_dir:
        write_per_config_metrics_artifacts(comparison, args.metrics_dir)

    print(json.dumps(comparison, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
