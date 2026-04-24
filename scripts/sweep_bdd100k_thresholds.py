#!/usr/bin/env python3
from __future__ import annotations

import argparse
from itertools import product
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adas_perception.config import deep_update, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep object models and score thresholds on BDD100K labels.")
    parser.add_argument("--images-root", required=True, help="Directory containing BDD100K images.")
    parser.add_argument("--labels", required=True, help="BDD100K/Scalabel label JSON path.")
    parser.add_argument("--base-config", default="configs/bdd100k_eval.yaml", help="Base YAML config.")
    parser.add_argument(
        "--score-thresholds",
        nargs="+",
        type=float,
        default=[0.20, 0.30, 0.40, 0.50],
        help="Object detector score thresholds to evaluate.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Optional TorchVision object model names to evaluate. Defaults to the base config model.",
    )
    parser.add_argument(
        "--kind-score-thresholds",
        nargs="+",
        default=None,
        metavar="KIND=THRESHOLDS",
        help=(
            "Optional per-kind score threshold grid, e.g. "
            "vehicle=0.20,0.25 pedestrian=0.10,0.15 traffic_sign=0.10 traffic_light=0.10,0.15."
        ),
    )
    parser.add_argument("--max-images", type=int, default=500, help="Evaluate first N labeled images.")
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Evaluate every Nth label frame. Use with --frame-offset for split evaluation.",
    )
    parser.add_argument(
        "--frame-offset",
        type=int,
        default=0,
        help="Start offset used with --frame-stride.",
    )
    parser.add_argument("--iou-threshold", type=float, default=0.50, help="IoU threshold for TP matching.")
    parser.add_argument("--device", default=None, help="Object detector device override.")
    parser.add_argument("--output-dir", default="outputs/bdd100k_sweep", help="Directory for configs/reports.")
    parser.add_argument("--save-predictions", action="store_true", help="Save per-threshold predictions.")
    parser.add_argument("--markdown", action="store_true", help="Also write Markdown and CSV comparison summaries.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running evaluation.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    config_dir = output_dir / "configs"
    report_dir = output_dir / "reports"
    pred_dir = output_dir / "predictions"
    config_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    if args.save_predictions:
        pred_dir.mkdir(parents=True, exist_ok=True)

    models = args.models or [None]
    kind_threshold_grid = _parse_kind_threshold_grid(args.kind_score_thresholds)
    generated_reports = []
    generated_names = []
    for model in models:
        model_tag = _model_tag(model) if model else "base"
        for threshold in args.score_thresholds:
            for kind_thresholds in _iter_kind_thresholds(kind_threshold_grid):
                threshold_tag = _threshold_tag(threshold)
                kind_tag = _kind_threshold_tag(kind_thresholds)
                tag = f"{model_tag}_score_{threshold_tag}{kind_tag}"
                config_path = config_dir / f"{tag}.yaml"
                report_path = report_dir / f"{tag}.json"
                prediction_path = pred_dir / f"{tag}_predictions.json"

                config = _sweep_config(args.base_config, threshold, model, kind_thresholds)
                _write_yaml(config_path, config)

                command = [
                    sys.executable,
                    str(Path(__file__).resolve().parent / "evaluate_bdd100k.py"),
                    "--images-root",
                    args.images_root,
                    "--labels",
                    args.labels,
                    "--config",
                    str(config_path),
                    "--max-images",
                    str(args.max_images),
                    "--frame-stride",
                    str(args.frame_stride),
                    "--frame-offset",
                    str(args.frame_offset),
                    "--iou-threshold",
                    str(args.iou_threshold),
                    "--output",
                    str(report_path),
                ]
                if args.device:
                    command.extend(["--device", args.device])
                if args.save_predictions:
                    command.extend(["--save-predictions", str(prediction_path)])

                print(" ".join(command), flush=True)
                if not args.dry_run:
                    subprocess.run(command, check=True)
                generated_reports.append(str(report_path))
                generated_names.append(tag)

    compare_path = output_dir / "comparison.json"
    compare_command = [
        sys.executable,
        str(Path(__file__).resolve().parent / "compare_evaluations.py"),
        "--reports",
        *generated_reports,
        "--names",
        *generated_names,
        "--output",
        str(compare_path),
    ]
    if args.markdown:
        compare_command.extend(
            [
                "--markdown-output",
                str(output_dir / "comparison.md"),
                "--csv-output",
                str(output_dir / "comparison.csv"),
            ]
        )
    print(" ".join(compare_command), flush=True)
    if not args.dry_run:
        subprocess.run(compare_command, check=True)
        _write_manifest(output_dir / "manifest.json", args, generated_reports, str(compare_path))
    return 0


def _sweep_config(
    base_config_path: str,
    threshold: float,
    model: str | None,
    kind_thresholds: dict[str, float],
) -> dict[str, Any]:
    base = load_config(base_config_path)
    override = {
        "objects": {
            "score_threshold": float(threshold),
        }
    }
    if kind_thresholds:
        override["objects"]["score_thresholds_by_kind"] = kind_thresholds
    if model:
        override["objects"]["model"] = model
    return deep_update(base, override)


def _threshold_tag(threshold: float) -> str:
    return f"{int(round(threshold * 1000)):03d}"


def _model_tag(model: str | None) -> str:
    if not model:
        return "base"
    return model.replace("-", "_").replace("/", "_")


def _parse_kind_threshold_grid(raw_items: list[str] | None) -> dict[str, list[float]]:
    if not raw_items:
        return {}

    grid: dict[str, list[float]] = {}
    for item in raw_items:
        if "=" not in item:
            raise ValueError(f"Invalid --kind-score-thresholds item: {item}. Expected KIND=0.10,0.20")
        kind, raw_values = item.split("=", 1)
        kind = kind.strip()
        values = [float(value) for value in raw_values.split(",") if value.strip()]
        if not kind or not values:
            raise ValueError(f"Invalid --kind-score-thresholds item: {item}. Expected KIND=0.10,0.20")
        grid[kind] = values
    return grid


def _iter_kind_thresholds(grid: dict[str, list[float]]):
    if not grid:
        yield {}
        return

    kinds = sorted(grid)
    for values in product(*(grid[kind] for kind in kinds)):
        yield {kind: float(value) for kind, value in zip(kinds, values)}


def _kind_threshold_tag(kind_thresholds: dict[str, float]) -> str:
    if not kind_thresholds:
        return ""
    parts = [f"{kind}_{_threshold_tag(threshold)}" for kind, threshold in sorted(kind_thresholds.items())]
    return "_kind_" + "_".join(parts)


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def _write_manifest(
    path: Path,
    args: argparse.Namespace,
    reports: list[str],
    comparison: str,
) -> None:
    payload = {
        "schema_version": "0.1",
        "images_root": args.images_root,
        "labels": args.labels,
        "base_config": args.base_config,
        "models": args.models,
        "score_thresholds": args.score_thresholds,
        "kind_score_thresholds": args.kind_score_thresholds,
        "max_images": args.max_images,
        "frame_stride": args.frame_stride,
        "frame_offset": args.frame_offset,
        "iou_threshold": args.iou_threshold,
        "reports": reports,
        "comparison": comparison,
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


if __name__ == "__main__":
    raise SystemExit(main())
