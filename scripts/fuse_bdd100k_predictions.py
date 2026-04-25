#!/usr/bin/env python3
"""Fuse two BDD100K prediction JSONs via Weighted Box Fusion.

Consumes the per-image prediction JSON format saved by
scripts/evaluate_bdd100k.py --save-predictions and emits a single JSON in the
same shape. The fused JSON can then be fed into
scripts/sweep_bdd100k_cached_predictions.py unchanged.

WBF from ensemble_boxes is imported via importlib so it still works on envs
where the package's __init__.py pulls in a broken numba.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any


def _load_wbf():
    candidate_dirs = [
        Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages",
    ]
    for base in sys.path:
        if base:
            candidate_dirs.append(Path(base))
    for base in candidate_dirs:
        target = base / "ensemble_boxes" / "ensemble_boxes_wbf.py"
        if target.is_file():
            spec = importlib.util.spec_from_file_location("ensemble_boxes_wbf", target)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module.weighted_boxes_fusion
    raise RuntimeError("Could not locate ensemble_boxes_wbf.py; pip install ensemble-boxes.")


weighted_boxes_fusion = _load_wbf()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fuse BDD100K prediction JSONs via WBF.")
    parser.add_argument("--predictions-a", required=True, help="First prediction JSON (e.g. no-TTA).")
    parser.add_argument("--predictions-b", required=True, help="Second prediction JSON (e.g. TTA).")
    parser.add_argument(
        "--extra-predictions",
        nargs="*",
        default=None,
        help="Optional additional prediction JSONs for N-way fusion.",
    )
    parser.add_argument("--output", required=True, help="Output fused prediction JSON path.")
    parser.add_argument("--weight-a", type=float, default=1.0, help="Weight for predictions-a (default 1.0).")
    parser.add_argument("--weight-b", type=float, default=1.0, help="Weight for predictions-b (default 1.0).")
    parser.add_argument(
        "--extra-weights",
        nargs="*",
        type=float,
        default=None,
        help="Weights for --extra-predictions (default 1.0 each).",
    )
    parser.add_argument("--iou-thr", type=float, default=0.55, help="IoU threshold for WBF (default 0.55).")
    parser.add_argument(
        "--kind-iou-thr",
        nargs="+",
        default=None,
        metavar="KIND=THR",
        help="Per-kind IoU threshold override. Example: pedestrian=0.5 traffic_light=0.4.",
    )
    parser.add_argument("--skip-box-thr", type=float, default=0.0, help="Drop boxes below this score before fusion.")
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        metavar=("HEIGHT", "WIDTH"),
        default=[720, 1280],
        help="Image (H, W) used to normalize coordinates for WBF (BDD100K default 720x1280).",
    )
    parser.add_argument(
        "--kinds",
        nargs="+",
        default=["pedestrian", "vehicle", "traffic_sign", "traffic_light"],
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    height, width = int(args.image_size[0]), int(args.image_size[1])
    allowed_kinds = set(args.kinds)

    all_predictions = [
        _load_predictions(Path(args.predictions_a)),
        _load_predictions(Path(args.predictions_b)),
    ]
    weights = [float(args.weight_a), float(args.weight_b)]

    extras = list(args.extra_predictions or [])
    extra_weights_cli = list(args.extra_weights or [])
    for idx, extra in enumerate(extras):
        all_predictions.append(_load_predictions(Path(extra)))
        weights.append(float(extra_weights_cli[idx]) if idx < len(extra_weights_cli) else 1.0)

    names: set[str] = set()
    for preds in all_predictions:
        names.update(preds.keys())
    names_sorted = sorted(names)

    kind_iou_map = _parse_kind_iou(args.kind_iou_thr)

    fused_items = []
    for name in names_sorted:
        detections_per_source = [preds.get(name, []) for preds in all_predictions]
        fused_detections = _fuse_frame(
            detections_per_source,
            weights=weights,
            allowed_kinds=allowed_kinds,
            height=height,
            width=width,
            iou_thr=args.iou_thr,
            kind_iou_thr=kind_iou_map,
            skip_box_thr=args.skip_box_thr,
        )
        fused_items.append({"name": name, "detections": fused_detections})

    payload = {"schema_version": "0.1", "predictions": fused_items}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output).open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
        f.write("\n")
    print(f"Wrote {args.output} with {len(fused_items)} frames.")
    return 0


def _load_predictions(path: Path) -> dict[str, list[dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    by_name: dict[str, list[dict[str, Any]]] = {}
    for item in payload.get("predictions", []):
        name = str(item.get("name", ""))
        by_name[name] = item.get("detections", [])
    return by_name


def _parse_kind_iou(raw_items: list[str] | None) -> dict[str, float]:
    if not raw_items:
        return {}
    out: dict[str, float] = {}
    for item in raw_items:
        if "=" not in item:
            raise ValueError(f"Invalid --kind-iou-thr item: {item}")
        key, val = item.split("=", 1)
        out[key.strip()] = float(val.strip())
    return out


def _fuse_frame(
    detections_per_source: list[list[dict[str, Any]]],
    *,
    weights: list[float],
    allowed_kinds: set[str],
    height: int,
    width: int,
    iou_thr: float,
    kind_iou_thr: dict[str, float] | None,
    skip_box_thr: float,
) -> list[dict[str, Any]]:
    n_sources = len(detections_per_source)
    # Group per-kind and run WBF independently so box coordinates only fuse
    # within the same kind.
    fused: list[dict[str, Any]] = []
    by_kind: dict[str, tuple[list, list, list, list]] = defaultdict(lambda: ([], [], [], []))

    def _push(target_idx: int, detections: list[dict[str, Any]]):
        for det in detections:
            kind = str(det.get("kind", ""))
            if kind not in allowed_kinds:
                continue
            box = det.get("box", {})
            x1 = float(box.get("x1", 0))
            y1 = float(box.get("y1", 0))
            x2 = float(box.get("x2", 0))
            y2 = float(box.get("y2", 0))
            if x2 <= x1 or y2 <= y1:
                continue
            score = float(det.get("confidence", 0.0))
            label = str(det.get("label", ""))
            norm_box = [
                max(0.0, min(1.0, x1 / max(width, 1))),
                max(0.0, min(1.0, y1 / max(height, 1))),
                max(0.0, min(1.0, x2 / max(width, 1))),
                max(0.0, min(1.0, y2 / max(height, 1))),
            ]
            boxes_list, scores_list, labels_list, meta_list = by_kind[kind]
            boxes_list.append(norm_box)
            scores_list.append(score)
            labels_list.append(0)
            meta_list.append({"source": target_idx, "label": label})

    for idx, dets in enumerate(detections_per_source):
        _push(idx, dets)

    for kind, (boxes, scores, labels, meta_list) in by_kind.items():
        if not boxes:
            continue
        boxes_per_source: list[list[list[float]]] = [[] for _ in range(n_sources)]
        scores_per_source: list[list[float]] = [[] for _ in range(n_sources)]
        labels_per_source: list[list[int]] = [[] for _ in range(n_sources)]
        for box, score, label, meta in zip(boxes, scores, labels, meta_list):
            src = int(meta["source"])
            boxes_per_source[src].append(box)
            scores_per_source[src].append(score)
            labels_per_source[src].append(label)
        effective_iou = iou_thr
        if kind_iou_thr and kind in kind_iou_thr:
            effective_iou = float(kind_iou_thr[kind])
        fused_boxes, fused_scores, _fused_labels = weighted_boxes_fusion(
            boxes_per_source,
            scores_per_source,
            labels_per_source,
            weights=weights,
            iou_thr=effective_iou,
            skip_box_thr=skip_box_thr,
        )
        label_guess = _dominant_label(meta_list)
        for box, score in zip(fused_boxes, fused_scores):
            x1 = float(box[0]) * width
            y1 = float(box[1]) * height
            x2 = float(box[2]) * width
            y2 = float(box[3]) * height
            if x2 <= x1 or y2 <= y1:
                continue
            fused.append(
                {
                    "kind": kind,
                    "label": label_guess,
                    "confidence": float(score),
                    "box": {
                        "x1": int(round(x1)),
                        "y1": int(round(y1)),
                        "x2": int(round(x2)),
                        "y2": int(round(y2)),
                    },
                }
            )
    return fused


def _dominant_label(meta_list: list[dict[str, Any]]) -> str:
    counts: dict[str, int] = defaultdict(int)
    for meta in meta_list:
        counts[str(meta.get("label", ""))] += 1
    if not counts:
        return ""
    return max(counts.items(), key=lambda kv: kv[1])[0]


if __name__ == "__main__":
    raise SystemExit(main())
