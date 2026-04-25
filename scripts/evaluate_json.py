#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize adas-perception JSON output.")
    parser.add_argument("--input", required=True, help="JSON file from demo_image.py or demo_video.py.")
    parser.add_argument("--output", default=None, help="Optional path to save the summary JSON.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    with input_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    report = evaluate_payload(payload)
    print_report(report)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
            f.write("\n")
        print(f"Saved {output_path}")
    return 0


def evaluate_payload(payload: dict[str, Any]) -> dict[str, Any]:
    frames = _extract_frames(payload)
    detections = [detection for frame in frames for detection in frame.get("detections", [])]
    lanes_per_frame = [len(frame.get("lanes", {}).get("lines", [])) for frame in frames]
    raw_segments_per_frame = [len(frame.get("lanes", {}).get("raw_segments", [])) for frame in frames]

    counts_by_kind = Counter(str(detection.get("kind", "unknown")) for detection in detections)
    labels_by_kind: dict[str, Counter[str]] = defaultdict(Counter)
    confidences_by_kind: dict[str, list[float]] = defaultdict(list)
    distances_by_kind: dict[str, list[float]] = defaultdict(list)
    tracks_by_kind: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))

    for frame_index, frame in enumerate(frames):
        for detection in frame.get("detections", []):
            kind = str(detection.get("kind", "unknown"))
            label = str(detection.get("label", "unknown"))
            labels_by_kind[kind][label] += 1
            if "confidence" in detection:
                confidences_by_kind[kind].append(float(detection["confidence"]))
            distance_m = detection.get("distance_m")
            if distance_m is not None:
                distances_by_kind[kind].append(float(distance_m))
            track_id = detection.get("track_id")
            if track_id is not None:
                tracks_by_kind[kind][int(track_id)].append(frame_index)

    return {
        "schema_version": "0.1",
        "source": payload.get("source"),
        "input_type": "video" if "frames" in payload else "image",
        "frames": len(frames),
        "detections": {
            "total": len(detections),
            "by_kind": dict(sorted(counts_by_kind.items())),
            "per_frame_average": round(len(detections) / max(len(frames), 1), 4),
            "average_confidence_by_kind": _average_confidences(confidences_by_kind),
            "top_labels_by_kind": _top_labels(labels_by_kind),
        },
        "lanes": {
            "frames_with_lanes": sum(1 for count in lanes_per_frame if count > 0),
            "average_lines_per_frame": round(mean(lanes_per_frame), 4) if lanes_per_frame else 0.0,
            "max_lines_per_frame": max(lanes_per_frame, default=0),
            "average_raw_segments_per_frame": round(mean(raw_segments_per_frame), 4)
            if raw_segments_per_frame
            else 0.0,
        },
        "distances": _distance_summary(distances_by_kind),
        "tracks": _track_summary(tracks_by_kind),
    }


def print_report(report: dict[str, Any]) -> None:
    print("ADAS perception JSON summary")
    print(f"- source: {report.get('source')}")
    print(f"- input_type: {report['input_type']}")
    print(f"- frames: {report['frames']}")
    print(f"- detections_total: {report['detections']['total']}")

    by_kind = report["detections"]["by_kind"]
    if by_kind:
        print("- detections_by_kind:")
        for kind, count in by_kind.items():
            avg_conf = report["detections"]["average_confidence_by_kind"].get(kind)
            avg_text = f", avg_conf={avg_conf:.3f}" if avg_conf is not None else ""
            print(f"  - {kind}: {count}{avg_text}")
    else:
        print("- detections_by_kind: none")

    lanes = report["lanes"]
    print(
        "- lanes: "
        f"frames_with_lanes={lanes['frames_with_lanes']}, "
        f"avg_lines={lanes['average_lines_per_frame']:.3f}, "
        f"max_lines={lanes['max_lines_per_frame']}"
    )

    distances = report["distances"]
    if distances:
        print("- distances_m:")
        for kind, summary in distances.items():
            print(
                f"  - {kind}: avg={summary['average']:.3f}, "
                f"nearest={summary['nearest']:.3f}, "
                f"farthest={summary['farthest']:.3f}, "
                f"count={summary['count']}"
            )
    else:
        print("- distances_m: none")

    tracks = report["tracks"]
    if tracks:
        print("- tracks:")
        for kind, summary in tracks.items():
            print(
                f"  - {kind}: unique={summary['unique_tracks']}, "
                f"longest_seen_frames={summary['longest_seen_frames']}, "
                f"avg_seen_frames={summary['average_seen_frames']:.3f}"
            )
    else:
        print("- tracks: none")


def _extract_frames(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if "frames" in payload:
        return list(payload["frames"])
    if "result" in payload:
        return [dict(payload["result"], frame_index=0, timestamp_ms=0.0)]
    raise ValueError("Unsupported JSON format: expected 'result' or 'frames'.")


def _average_confidences(confidences_by_kind: dict[str, list[float]]) -> dict[str, float]:
    return {
        kind: round(mean(values), 4)
        for kind, values in sorted(confidences_by_kind.items())
        if values
    }


def _top_labels(labels_by_kind: dict[str, Counter[str]]) -> dict[str, list[dict[str, Any]]]:
    return {
        kind: [{"label": label, "count": count} for label, count in labels.most_common(5)]
        for kind, labels in sorted(labels_by_kind.items())
    }


def _distance_summary(distances_by_kind: dict[str, list[float]]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for kind, values in sorted(distances_by_kind.items()):
        if not values:
            continue
        summary[kind] = {
            "count": len(values),
            "average": round(mean(values), 4),
            "nearest": round(min(values), 4),
            "farthest": round(max(values), 4),
        }
    return summary


def _track_summary(tracks_by_kind: dict[str, dict[int, list[int]]]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for kind, tracks in sorted(tracks_by_kind.items()):
        lengths = [len(frame_indexes) for frame_indexes in tracks.values()]
        if not lengths:
            continue
        summary[kind] = {
            "unique_tracks": len(tracks),
            "longest_seen_frames": max(lengths),
            "average_seen_frames": round(mean(lengths), 4),
        }
    return summary


if __name__ == "__main__":
    raise SystemExit(main())
