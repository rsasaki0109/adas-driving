#!/usr/bin/env python3
"""End-to-end planning demo: perception JSON -> planning JSON -> overlay video."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adas_planning.io.driving_replay import write_driving_replay_document

DEFAULT_PERCEPTION_CONFIG = "configs/bdd100k_yolo_kind_tuned_post_nms.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default="assets/demo_wbf7.mp4", help="Input video path.")
    parser.add_argument(
        "--perception-json",
        default="examples/fixtures/planning_demo_perception.json",
        help="Perception JSON path (ignored when --run-perception).",
    )
    parser.add_argument(
        "--perception-config",
        default=DEFAULT_PERCEPTION_CONFIG,
        help=(
            "Perception YAML when --run-perception is set "
            f"(default: {DEFAULT_PERCEPTION_CONFIG})."
        ),
    )
    parser.add_argument("--config", default="configs/planning/default.yaml", help="Planning config.")
    parser.add_argument("--output-dir", default="outputs/planning_demo", help="Output directory.")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit processed frames.")
    parser.add_argument(
        "--run-perception",
        action="store_true",
        help="Run perception on --video and write JSON under --output-dir.",
    )
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
    parser.add_argument(
        "--export-benchmark",
        action="store_true",
        help="Export CSV/Markdown benchmark tables when --compare-configs is set.",
    )
    parser.add_argument(
        "--no-driving-replay",
        action="store_true",
        help="Skip driving_replay.v0.1 export.",
    )
    return parser.parse_args()


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.run_perception:
        perception_json = output_dir / "perception_frames.json"
        perception_video = output_dir / "perception_overlay.mp4"
        perception_cmd = [
            sys.executable,
            "scripts/demo_video.py",
            "--input",
            args.video,
            "--output",
            str(perception_video),
            "--config",
            args.perception_config,
            "--json-output",
            str(perception_json),
        ]
        if args.max_frames is not None:
            perception_cmd.extend(["--max-frames", str(args.max_frames)])
        _run(perception_cmd)
    else:
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
    driving_replay_json = output_dir / "driving_replay.json"
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

    if not args.no_driving_replay:
        write_driving_replay_document(
            driving_replay_json,
            perception_path=perception_json,
            planning_path=planning_json,
            producer="run_planning_demo",
        )

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
        if args.export_benchmark:
            _run(
                [
                    sys.executable,
                    "scripts/export_planning_benchmark.py",
                    "--compare-json",
                    str(compare_json),
                    "--csv",
                    str(output_dir / "planning_benchmark.csv"),
                    "--markdown",
                    str(output_dir / "planning_benchmark.md"),
                    "--json",
                    str(output_dir / "planning_benchmark_export.json"),
                ]
            )

    summary = {
        "video": str(args.video),
        "perception_json": str(perception_json),
        "planning_json": str(planning_json),
        "metrics_json": str(metrics_json),
        "overlay_mp4": str(overlay_mp4),
    }
    if not args.no_driving_replay:
        summary["driving_replay_json"] = str(driving_replay_json)
    if args.run_perception:
        summary["perception_config"] = args.perception_config
    if args.compare_configs:
        summary["config_compare_json"] = str(compare_json)
        if args.export_benchmark:
            summary["benchmark_csv"] = str(output_dir / "planning_benchmark.csv")
            summary["benchmark_markdown"] = str(output_dir / "planning_benchmark.md")
    with (output_dir / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
