from __future__ import annotations

import csv
import json
from io import StringIO
from pathlib import Path
from typing import Any

from adas_planning.metrics.baseline_compare import load_baseline_compare_artifact
from adas_planning.metrics.offline import METRICS_SCHEMA_VERSION

BENCHMARK_EXPORT_SCHEMA_VERSION = "planning_benchmark_export.v0.1"

EXPORT_COLUMNS = (
    "config_name",
    "config_path",
    "config_hash",
    "frame_count",
    "target_path_valid_rate",
    "behavior_output_rate",
    "behavior_switch_count",
    "behavior_switch_count_per_min",
    "warning_frame_rate",
    "target_speed_output_rate",
    "mean_confidence",
)


def export_baseline_compare_rows(comparison: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, entry in comparison.get("configs", {}).items():
        metrics = entry.get("metrics") or {}
        rows.append(
            {
                "config_name": name,
                "config_path": entry.get("config_path"),
                "config_hash": entry.get("config_hash"),
                "frame_count": metrics.get("frame_count"),
                "target_path_valid_rate": metrics.get("target_path_valid_rate"),
                "behavior_output_rate": metrics.get("behavior_output_rate"),
                "behavior_switch_count": metrics.get("behavior_switch_count"),
                "behavior_switch_count_per_min": metrics.get("behavior_switch_count_per_min"),
                "warning_frame_rate": metrics.get("warning_frame_rate"),
                "target_speed_output_rate": metrics.get("target_speed_output_rate"),
                "mean_confidence": metrics.get("mean_confidence"),
            }
        )
    return rows


def export_baseline_compare_csv(comparison: dict[str, Any]) -> str:
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=EXPORT_COLUMNS)
    writer.writeheader()
    for row in export_baseline_compare_rows(comparison):
        writer.writerow(row)
    return buffer.getvalue()


def export_baseline_compare_markdown(comparison: dict[str, Any]) -> str:
    baseline = str(comparison.get("baseline", "default"))
    lines = [
        "# Planning baseline benchmark",
        "",
        f"- schema: `{comparison.get('schema_version')}`",
        f"- source: `{comparison.get('source')}`",
        f"- baseline: `{baseline}`",
        "",
        "| config | path valid | behavior out | warnings | target speed | mean conf | switches/min |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in export_baseline_compare_rows(comparison):
        lines.append(
            "| {name} | {path:.3f} | {behavior:.3f} | {warn:.3f} | {speed:.3f} | {conf:.3f} | {switch:.3f} |".format(
                name=row["config_name"],
                path=float(row.get("target_path_valid_rate") or 0.0),
                behavior=float(row.get("behavior_output_rate") or 0.0),
                warn=float(row.get("warning_frame_rate") or 0.0),
                speed=float(row.get("target_speed_output_rate") or 0.0),
                conf=float(row.get("mean_confidence") or 0.0),
                switch=float(row.get("behavior_switch_count_per_min") or 0.0),
            )
        )

    deltas = comparison.get("deltas_vs_baseline") or {}
    if deltas:
        lines.extend(["", f"## Deltas vs `{baseline}`", ""])
        for name, delta in deltas.items():
            parts = [f"{metric}={value:+.3f}" for metric, value in sorted(delta.items())]
            lines.append(f"- **{name}**: {', '.join(parts)}")
    lines.append("")
    return "\n".join(lines)


def build_benchmark_export_artifact(
    comparison: dict[str, Any],
    *,
    producer: str = "adas_planning",
) -> dict[str, Any]:
    return {
        "schema_version": BENCHMARK_EXPORT_SCHEMA_VERSION,
        "source_compare_schema": comparison.get("schema_version"),
        "source": comparison.get("source"),
        "baseline": comparison.get("baseline"),
        "producer": producer,
        "metrics_schema_version": METRICS_SCHEMA_VERSION,
        "rows": export_baseline_compare_rows(comparison),
        "deltas_vs_baseline": comparison.get("deltas_vs_baseline") or {},
    }


def write_benchmark_export_artifact(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_benchmark_csv(path: str | Path, comparison: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(export_baseline_compare_csv(comparison), encoding="utf-8")


def write_benchmark_markdown(path: str | Path, comparison: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(export_baseline_compare_markdown(comparison), encoding="utf-8")


def load_and_export_benchmark(
    compare_path: str | Path,
    *,
    csv_path: str | Path | None = None,
    markdown_path: str | Path | None = None,
    json_path: str | Path | None = None,
) -> dict[str, Any]:
    comparison = load_baseline_compare_artifact(compare_path)
    artifact = build_benchmark_export_artifact(comparison)
    if csv_path:
        write_benchmark_csv(csv_path, comparison)
    if markdown_path:
        write_benchmark_markdown(markdown_path, comparison)
    if json_path:
        write_benchmark_export_artifact(json_path, artifact)
    return artifact
