#!/usr/bin/env python3
"""End-to-end planning demo: perception JSON -> planning JSON -> overlay video."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default="assets/demo_wbf7.mp4", help="Input video path.")
    parser.add_argument(
        "--perception-json",
        default="examples/fixtures/planning_demo_perception.json",
        help="Perception JSON path.",
    )
    parser.add_argument("--config", default="configs/planning/default.yaml", help="Planning config.")
    parser.add_argument("--output-dir", default="outputs/planning_demo", help="Output directory.")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit processed frames.")
    parser.add_argument(
        "--generate-fixture",
        action="store_true",
        help="Regenerate synthetic perception JSON before replay.",
    )
    parser.add_argument(
        "--compare-configs",
        action="store_true",
        help="Also compare default vs conservative configs.",
    )
    return parser.parse_args()


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    perception_json = Path(args.perception_json)
    if args.generate_fixture or not perception_json.exists():
        _run(
            [
                sys.executable,
                "scripts/generate_planning_demo_fixture.py",
                "--output",
                str(perception_json),
            ]
        )

    planning_json = output_dir / "planning_frames.json"
    metrics_json = output_dir / "planning_metrics.json"
    overlay_mp4 = output_dir / "planning_overlay.mp4"
    compare_json = output_dir / "planning_config_compare.json"

    replay_cmd = [
        sys.executable,
        "scripts/replay_planning_json.py",
        "--input",
        str(perception_json),
        "--config",
        args.config,
        "--output",
        str(planning_json),
        "--metrics-output",
        str(metrics_json),
    ]
    _run(replay_cmd)

    overlay_cmd = [
        sys.executable,
        "scripts/demo_planning_video.py",
        "--video",
        args.video,
        "--perception-json",
        str(perception_json),
        "--planning-json",
        str(planning_json),
        "--config",
        args.config,
        "--output",
        str(overlay_mp4),
    ]
    if args.max_frames is not None:
        overlay_cmd.extend(["--max-frames", str(args.max_frames)])
    _run(overlay_cmd)

    if args.compare_configs:
        _run(
            [
                sys.executable,
                "scripts/compare_planning_baselines.py",
                "--input",
                str(perception_json),
                "--output",
                str(compare_json),
                "--metrics-dir",
                str(output_dir / "baseline_metrics"),
            ]
        )

    summary = {
        "video": str(args.video),
        "perception_json": str(perception_json),
        "planning_json": str(planning_json),
        "metrics_json": str(metrics_json),
        "overlay_mp4": str(overlay_mp4),
    }
    if args.compare_configs:
        summary["config_compare_json"] = str(compare_json)
    with (output_dir / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
