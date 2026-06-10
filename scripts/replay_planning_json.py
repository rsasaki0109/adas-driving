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
from adas_planning.io.planning_json import write_planning_document
from adas_planning.metrics.offline import compute_offline_metrics, write_metrics_artifact
from adas_planning.pipeline import PlanningPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay planning from saved perception JSON.")
    parser.add_argument("--input", required=True, help="Perception JSON path (image or video format).")
    parser.add_argument("--config", default="configs/planning/default.yaml", help="Planning YAML config.")
    parser.add_argument("--output", required=True, help="Planning JSON output path.")
    parser.add_argument("--metrics-output", default=None, help="Optional offline metrics JSON path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    with input_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    config = load_config(args.config)
    pipeline = PlanningPipeline(config)
    planning_inputs = adapt_perception_document(payload)
    results = [pipeline.plan(planning_input) for planning_input in planning_inputs]

    write_planning_document(
        args.output,
        frames=results,
        source=str(input_path),
        config_path=str(args.config),
        config_hash=config_hash(config),
    )

    metrics = compute_offline_metrics(results)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    if args.metrics_output:
        write_metrics_artifact(
            args.metrics_output,
            metrics,
            source=str(input_path),
            config_path=str(args.config),
            config_hash=config_hash(config),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
