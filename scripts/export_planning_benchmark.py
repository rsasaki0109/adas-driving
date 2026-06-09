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
)
from adas_planning.metrics.benchmark_export import load_and_export_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export planning baseline benchmark tables from compare artifacts.")
    parser.add_argument(
        "--compare-json",
        default=None,
        help="Existing planning_baseline_compare.v0.1 JSON path.",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Perception JSON path used when --compare-json is omitted.",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=[f"{name}={path}" for name, path in DEFAULT_BASELINE_CONFIGS.items()],
        metavar="NAME=PATH",
    )
    parser.add_argument("--compare-output", default="outputs/planning_baseline_compare.json")
    parser.add_argument("--csv", default="outputs/planning_benchmark.csv")
    parser.add_argument("--markdown", default="outputs/planning_benchmark.md")
    parser.add_argument("--json", default="outputs/planning_benchmark_export.json")
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
    compare_path = Path(args.compare_json) if args.compare_json else Path(args.compare_output)
    if args.compare_json is None:
        if not args.input:
            raise SystemExit("--input is required when --compare-json is omitted")
        with Path(args.input).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        comparison = compare_planning_configs(
            payload,
            _parse_config_entries(args.configs),
            source=str(args.input),
        )
        write_baseline_compare_artifact(compare_path, comparison)
    artifact = load_and_export_benchmark(
        compare_path,
        csv_path=args.csv,
        markdown_path=args.markdown,
        json_path=args.json,
    )
    print(json.dumps({"compare_json": str(compare_path), "artifact": artifact}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
