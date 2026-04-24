#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare BDD100K evaluation JSON reports.")
    parser.add_argument("--reports", nargs="+", required=True, help="Evaluation JSON reports to compare.")
    parser.add_argument("--names", nargs="+", default=None, help="Optional display names matching --reports.")
    parser.add_argument("--baseline", default=None, help="Report path or name to use as baseline. Defaults to first.")
    parser.add_argument("--output", default=None, help="Optional JSON comparison report path.")
    parser.add_argument("--markdown-output", default=None, help="Optional Markdown summary path.")
    parser.add_argument("--csv-output", default=None, help="Optional CSV summary path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.names is not None and len(args.names) != len(args.reports):
        raise ValueError("--names length must match --reports length.")

    reports = []
    for index, report_path in enumerate(args.reports):
        path = Path(report_path)
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        name = args.names[index] if args.names else path.stem
        reports.append(_summarize_report(name=name, path=str(path), payload=payload))

    baseline = _select_baseline(reports, args.baseline)
    comparison = {
        "schema_version": "0.1",
        "baseline": baseline["name"],
        "reports": [_with_deltas(report, baseline) for report in reports],
        "best": _best_summary(reports),
    }

    print_report(comparison)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(comparison, f, ensure_ascii=False, indent=2)
            f.write("\n")
        print(f"Saved {output_path}")
    if args.markdown_output:
        _write_text(Path(args.markdown_output), _markdown_summary(comparison))
        print(f"Saved {args.markdown_output}")
    if args.csv_output:
        _write_text(Path(args.csv_output), _csv_summary(comparison))
        print(f"Saved {args.csv_output}")
    return 0


def print_report(comparison: dict[str, Any]) -> None:
    print("Evaluation comparison")
    print(f"- baseline: {comparison['baseline']}")
    for report in comparison["reports"]:
        print(
            f"- {report['name']}: "
            f"macro_f1={report['macro_f1']:.3f} "
            f"(delta={report['delta']['macro_f1']:+.3f}), "
            f"fps={report['fps']:.3f}, "
            f"processed={report['processed_images']}"
        )
        for kind, metrics in report["object_metrics"].items():
            delta = report["delta"]["object_metrics"].get(kind, {})
            print(
                f"  - {kind}: "
                f"f1={metrics['f1']:.3f} ({delta.get('f1', 0.0):+.3f}), "
                f"p={metrics['precision']:.3f}, "
                f"r={metrics['recall']:.3f}"
            )
        lane = report.get("lane_presence")
        if lane and lane.get("evaluated_frames", 0) > 0:
            lane_delta = report["delta"].get("lane_presence", {})
            print(f"  - lane_presence: f1={lane['f1']:.3f} ({lane_delta.get('f1', 0.0):+.3f})")
        state = report.get("traffic_light_state")
        if state and state.get("matched", 0) > 0:
            state_delta = report["delta"].get("traffic_light_state", {})
            print(
                "  - traffic_light_state: "
                f"accuracy={state['accuracy']:.3f} ({state_delta.get('accuracy', 0.0):+.3f})"
            )

    print("Best")
    for metric, item in comparison["best"].items():
        if metric == "object_f1_by_kind":
            print("- object_f1_by_kind:")
            for kind, best in item.items():
                print(f"  - {kind}: {best['name']} ({best['value']})")
            continue
        print(f"- {metric}: {item['name']} ({item['value']})")


def _summarize_report(name: str, path: str, payload: dict[str, Any]) -> dict[str, Any]:
    object_metrics = payload.get("object_metrics", {})
    f1_values = [float(metrics.get("f1", 0.0)) for metrics in object_metrics.values()]
    precision_values = [float(metrics.get("precision", 0.0)) for metrics in object_metrics.values()]
    recall_values = [float(metrics.get("recall", 0.0)) for metrics in object_metrics.values()]
    return {
        "name": name,
        "path": path,
        "config": payload.get("config"),
        "processed_images": int(payload.get("processed_images", 0)),
        "missing_images_count": int(payload.get("missing_images_count", 0)),
        "fps": float(payload.get("runtime", {}).get("fps", 0.0)),
        "average_inference_ms": float(payload.get("runtime", {}).get("average_inference_ms", 0.0)),
        "macro_f1": round(mean(f1_values), 4) if f1_values else 0.0,
        "macro_precision": round(mean(precision_values), 4) if precision_values else 0.0,
        "macro_recall": round(mean(recall_values), 4) if recall_values else 0.0,
        "object_metrics": object_metrics,
        "lane_presence": payload.get("lane_presence", {}),
        "traffic_light_state": payload.get("traffic_light_state", {}),
        "grouped_metrics": payload.get("grouped_metrics", {}),
        "object_size_metrics": payload.get("object_size_metrics", {}),
    }


def _select_baseline(reports: list[dict[str, Any]], requested: str | None) -> dict[str, Any]:
    if not reports:
        raise ValueError("At least one report is required.")
    if requested is None:
        return reports[0]
    for report in reports:
        if requested in {report["name"], report["path"]}:
            return report
    raise ValueError(f"Baseline not found: {requested}")


def _with_deltas(report: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(report)
    object_deltas: dict[str, dict[str, float]] = {}
    all_kinds = set(report.get("object_metrics", {})) | set(baseline.get("object_metrics", {}))
    for kind in sorted(all_kinds):
        current = report.get("object_metrics", {}).get(kind, {})
        base = baseline.get("object_metrics", {}).get(kind, {})
        object_deltas[kind] = {
            "precision": round(float(current.get("precision", 0.0)) - float(base.get("precision", 0.0)), 4),
            "recall": round(float(current.get("recall", 0.0)) - float(base.get("recall", 0.0)), 4),
            "f1": round(float(current.get("f1", 0.0)) - float(base.get("f1", 0.0)), 4),
        }

    enriched["delta"] = {
        "macro_f1": round(report["macro_f1"] - baseline["macro_f1"], 4),
        "macro_precision": round(report["macro_precision"] - baseline["macro_precision"], 4),
        "macro_recall": round(report["macro_recall"] - baseline["macro_recall"], 4),
        "fps": round(report["fps"] - baseline["fps"], 4),
        "average_inference_ms": round(report["average_inference_ms"] - baseline["average_inference_ms"], 4),
        "object_metrics": object_deltas,
        "lane_presence": {
            "f1": round(
                float(report.get("lane_presence", {}).get("f1", 0.0))
                - float(baseline.get("lane_presence", {}).get("f1", 0.0)),
                4,
            )
        },
        "traffic_light_state": {
            "accuracy": round(
                float(report.get("traffic_light_state", {}).get("accuracy", 0.0))
                - float(baseline.get("traffic_light_state", {}).get("accuracy", 0.0)),
                4,
            )
        },
        "grouped_metrics": _grouped_deltas(report.get("grouped_metrics", {}), baseline.get("grouped_metrics", {})),
        "object_size_metrics": _size_deltas(
            report.get("object_size_metrics", {}),
            baseline.get("object_size_metrics", {}),
        ),
    }
    return enriched


def _best_summary(reports: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    if not reports:
        return {}
    best_macro_f1 = max(reports, key=lambda item: item["macro_f1"])
    best_fps = max(reports, key=lambda item: item["fps"])
    best_lane = max(reports, key=lambda item: float(item.get("lane_presence", {}).get("f1", 0.0)))
    best_by_kind = {}
    all_kinds = sorted({kind for report in reports for kind in report.get("object_metrics", {})})
    for kind in all_kinds:
        best = max(reports, key=lambda item: float(item.get("object_metrics", {}).get(kind, {}).get("f1", 0.0)))
        best_by_kind[kind] = {
            "name": best["name"],
            "value": best.get("object_metrics", {}).get(kind, {}).get("f1", 0.0),
        }
    return {
        "macro_f1": {
            "name": best_macro_f1["name"],
            "value": best_macro_f1["macro_f1"],
        },
        "fps": {
            "name": best_fps["name"],
            "value": best_fps["fps"],
        },
        "lane_presence_f1": {
            "name": best_lane["name"],
            "value": best_lane.get("lane_presence", {}).get("f1", 0.0),
        },
        "object_f1_by_kind": best_by_kind,
    }


def _markdown_summary(comparison: dict[str, Any]) -> str:
    lines = [
        "# BDD100K Evaluation Comparison",
        "",
        f"- Baseline: `{comparison['baseline']}`",
        "",
        "## Summary",
        "",
        "| name | processed | fps | avg ms | macro F1 | macro precision | macro recall | delta F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for report in comparison["reports"]:
        lines.append(
            "| "
            f"{report['name']} | "
            f"{report['processed_images']} | "
            f"{report['fps']:.3f} | "
            f"{report['average_inference_ms']:.3f} | "
            f"{report['macro_f1']:.4f} | "
            f"{report['macro_precision']:.4f} | "
            f"{report['macro_recall']:.4f} | "
            f"{report['delta']['macro_f1']:+.4f} |"
        )

    lines.extend(["", "## Object F1", ""])
    kinds = sorted({kind for report in comparison["reports"] for kind in report.get("object_metrics", {})})
    header = "| name | " + " | ".join(kinds) + " |"
    separator = "| --- | " + " | ".join("---:" for _ in kinds) + " |"
    lines.extend([header, separator])
    for report in comparison["reports"]:
        values = [
            f"{float(report.get('object_metrics', {}).get(kind, {}).get('f1', 0.0)):.4f}"
            for kind in kinds
        ]
        lines.append("| " + report["name"] + " | " + " | ".join(values) + " |")

    grouped_attributes = sorted(
        {
            attribute
            for report in comparison["reports"]
            for attribute in report.get("grouped_metrics", {})
        }
    )
    if grouped_attributes:
        lines.extend(["", "## Grouped Macro F1", ""])
        for attribute in grouped_attributes:
            values = sorted(
                {
                    value
                    for report in comparison["reports"]
                    for value in report.get("grouped_metrics", {}).get(attribute, {})
                }
            )
            if not values:
                continue
            lines.extend(
                [
                    f"### {attribute}",
                    "",
                    "| group | " + " | ".join(report["name"] for report in comparison["reports"]) + " |",
                    "| --- | " + " | ".join("---:" for _ in comparison["reports"]) + " |",
                ]
            )
            for value in values:
                scores = [
                    f"{_group_macro_f1(report.get('grouped_metrics', {}).get(attribute, {}).get(value, {})):.4f}"
                    for report in comparison["reports"]
                ]
                lines.append("| " + value + " | " + " | ".join(scores) + " |")
            lines.append("")

    size_buckets = sorted(
        {
            bucket
            for report in comparison["reports"]
            for bucket in report.get("object_size_metrics", {}).get("macro_f1_by_bucket", {})
        }
    )
    if size_buckets:
        lines.extend(
            [
                "",
                "## Object Size Macro F1",
                "",
                "| bucket | " + " | ".join(report["name"] for report in comparison["reports"]) + " |",
                "| --- | " + " | ".join("---:" for _ in comparison["reports"]) + " |",
            ]
        )
        for bucket in size_buckets:
            values = [
                f"{float(report.get('object_size_metrics', {}).get('macro_f1_by_bucket', {}).get(bucket, 0.0)):.4f}"
                for report in comparison["reports"]
            ]
            lines.append("| " + bucket + " | " + " | ".join(values) + " |")

    best = comparison.get("best", {})
    if best:
        lines.extend(["", "## Best", ""])
        for metric, item in best.items():
            if metric == "object_f1_by_kind":
                continue
            lines.append(f"- {metric}: `{item['name']}` ({item['value']})")
        by_kind = best.get("object_f1_by_kind", {})
        for kind, item in by_kind.items():
            lines.append(f"- {kind} F1: `{item['name']}` ({item['value']})")
    lines.append("")
    return "\n".join(lines)


def _csv_summary(comparison: dict[str, Any]) -> str:
    kinds = sorted({kind for report in comparison["reports"] for kind in report.get("object_metrics", {})})
    columns = [
        "name",
        "processed_images",
        "fps",
        "average_inference_ms",
        "macro_f1",
        "macro_precision",
        "macro_recall",
        "delta_macro_f1",
    ]
    for kind in kinds:
        columns.extend([f"{kind}_precision", f"{kind}_recall", f"{kind}_f1"])
    size_buckets = sorted(
        {
            bucket
            for report in comparison["reports"]
            for bucket in report.get("object_size_metrics", {}).get("macro_f1_by_bucket", {})
        }
    )
    for bucket in size_buckets:
        columns.append(f"size_{bucket}_macro_f1")
    lines = [",".join(columns)]
    for report in comparison["reports"]:
        row = [
            _csv_cell(report["name"]),
            str(report["processed_images"]),
            f"{report['fps']:.6f}",
            f"{report['average_inference_ms']:.6f}",
            f"{report['macro_f1']:.6f}",
            f"{report['macro_precision']:.6f}",
            f"{report['macro_recall']:.6f}",
            f"{report['delta']['macro_f1']:.6f}",
        ]
        for kind in kinds:
            metrics = report.get("object_metrics", {}).get(kind, {})
            row.extend(
                [
                    f"{float(metrics.get('precision', 0.0)):.6f}",
                    f"{float(metrics.get('recall', 0.0)):.6f}",
                    f"{float(metrics.get('f1', 0.0)):.6f}",
                ]
            )
        for bucket in size_buckets:
            row.append(
                f"{float(report.get('object_size_metrics', {}).get('macro_f1_by_bucket', {}).get(bucket, 0.0)):.6f}"
            )
        lines.append(",".join(row))
    return "\n".join(lines) + "\n"


def _grouped_deltas(
    current: dict[str, dict[str, Any]],
    baseline: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    deltas = {}
    for attribute in sorted(set(current) | set(baseline)):
        deltas[attribute] = {}
        values = set(current.get(attribute, {})) | set(baseline.get(attribute, {}))
        for value in sorted(values):
            current_group = current.get(attribute, {}).get(value, {})
            baseline_group = baseline.get(attribute, {}).get(value, {})
            current_objects = current_group.get("object_metrics", {})
            baseline_objects = baseline_group.get("object_metrics", {})
            object_deltas = {}
            for kind in sorted(set(current_objects) | set(baseline_objects)):
                object_deltas[kind] = {
                    "f1": round(
                        float(current_objects.get(kind, {}).get("f1", 0.0))
                        - float(baseline_objects.get(kind, {}).get("f1", 0.0)),
                        4,
                    )
                }
            deltas[attribute][value] = {
                "macro_f1": round(_group_macro_f1(current_group) - _group_macro_f1(baseline_group), 4),
                "object_metrics": object_deltas,
            }
    return deltas


def _size_deltas(current: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    current_buckets = current.get("macro_f1_by_bucket", {})
    baseline_buckets = baseline.get("macro_f1_by_bucket", {})
    return {
        "macro_f1_by_bucket": {
            bucket: round(float(current_buckets.get(bucket, 0.0)) - float(baseline_buckets.get(bucket, 0.0)), 4)
            for bucket in sorted(set(current_buckets) | set(baseline_buckets))
        }
    }


def _group_macro_f1(group: dict[str, Any]) -> float:
    object_metrics = group.get("object_metrics", {})
    values = [float(metrics.get("f1", 0.0)) for metrics in object_metrics.values()]
    return round(mean(values), 4) if values else 0.0


def _csv_cell(value: object) -> str:
    text = str(value)
    if any(char in text for char in [",", '"', "\n"]):
        return '"' + text.replace('"', '""') + '"'
    return text


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
