#!/usr/bin/env bash
# Production post-NMS sweep on the BDD100K odd 5,000 report split.
#
# If outputs/models/adas_yolov8n_bdd100k.pt is missing, bootstraps a proxy
# weight with 1 epoch on the even-index val mirror split (yolov8n.pt -> even 1ep).
#
# Usage:
#   bash scripts/run_postprocess_sweep_production.sh
#   AUTO_TRAIN=0 bash scripts/run_postprocess_sweep_production.sh   # require existing weight
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PY="${PY:-.venv/bin/python}"
AUTO_TRAIN="${AUTO_TRAIN:-1}"
WEIGHT="outputs/models/adas_yolov8n_bdd100k.pt"
PROXY_WEIGHT="outputs/models/adas_yolov8n_bdd100k_even_1024_1ep.pt"
EXPORT_DIR="data/bdd100k_yolo_even_odd_val_mirror"
RUN_NAME="adas_yolov8n_bdd100k_even_1024_1ep"
PREDICTIONS="outputs/bdd100k_yolo_current_best_cache_low_odd_5000_predictions.json"
RAW_REPORT="outputs/bdd100k_yolo_current_best_cache_low_odd_5000_report.json"
SWEEP_DIR="outputs/postprocess_sweep_production"
COMPARE_MD="outputs/postprocess_sweep_production/compare.md"

step() { printf "\n========== %s ==========\n" "$1"; }

step "Ensure Python venv"
if [[ ! -x "$PY" ]]; then
  python3 -m venv .venv
  "$PY" -m pip install -U pip
  "$PY" -m pip install -r requirements.txt -r requirements-yolo.txt -r requirements-bdd100k.txt -e ".[dev]"
fi

step "Ensure BDD100K val mirror"
VAL_COUNT=0
if [[ -d data/bdd100k/images/100k/val ]]; then
  VAL_COUNT="$(find data/bdd100k/images/100k/val -maxdepth 1 -type f -name '*.jpg' | wc -l)"
fi
if [[ ! -f data/bdd100k/labels/det_20/det_val.json || "$VAL_COUNT" -lt 1000 ]]; then
  "$PY" scripts/prepare_bdd100k.py --download-val --data-root data/bdd100k
fi

mkdir -p outputs/models

step "Ensure finetuned weight"
if [[ ! -f "$WEIGHT" ]]; then
  if [[ "$AUTO_TRAIN" != "1" ]]; then
    echo "Missing $WEIGHT and AUTO_TRAIN=0." >&2
    exit 2
  fi
  echo "Bootstrapping proxy weight via even-split 1 epoch fine-tune..."
  "$PY" scripts/export_bdd100k_yolo.py \
    --images-root data/bdd100k/images/100k/val \
    --labels data/bdd100k/labels/det_20/det_val.json \
    --output-dir "$EXPORT_DIR" \
    --split-mode alternate \
    --frame-stride 2 \
    --train-frame-offset 0 \
    --val-frame-offset 1 \
    --classes car truck bus bicycle motorcycle train pedestrian rider "traffic sign" "traffic light" \
    --clear-output
  if [[ ! -f yolov8n.pt ]]; then
    "$PY" -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
  fi
  "$PY" - <<PYEOF
from ultralytics import YOLO
YOLO("yolov8n.pt").train(
    data="${EXPORT_DIR}/dataset.yaml",
    epochs=1,
    imgsz=1024,
    batch=8,
    device=0,
    workers=4,
    project="outputs/yolo_train",
    name="${RUN_NAME}",
    exist_ok=True,
    plots=False,
)
PYEOF
  TRAIN_BEST="runs/detect/outputs/yolo_train/${RUN_NAME}/weights/best.pt"
  if [[ ! -f "$TRAIN_BEST" ]]; then
    TRAIN_BEST="outputs/yolo_train/${RUN_NAME}/weights/best.pt"
  fi
  if [[ ! -f "$TRAIN_BEST" ]]; then
    echo "Training did not produce weights under expected paths." >&2
    exit 1
  fi
  cp "$TRAIN_BEST" "$PROXY_WEIGHT"
  cp "$PROXY_WEIGHT" "$WEIGHT"
  echo "Bootstrapped $WEIGHT from $TRAIN_BEST"
else
  echo "Using existing weight: $WEIGHT"
fi

step "Generate cache-low predictions on odd 5,000 report split"
if [[ ! -f "$PREDICTIONS" ]]; then
  "$PY" scripts/evaluate_bdd100k.py \
    --images-root data/bdd100k/images/100k/val \
    --labels data/bdd100k/labels/det_20/det_val.json \
    --config configs/bdd100k_yolo_finetuned_all_tuned_split_img1024_kind_tuned_cache_low.yaml \
    --frame-stride 2 \
    --frame-offset 1 \
    --save-predictions "$PREDICTIONS" \
    --output "$RAW_REPORT" \
    --group-by-size \
    --progress-every 250
else
  echo "Reusing existing predictions: $PREDICTIONS"
fi

step "Run post-NMS sweep (production grid)"
"$PY" scripts/sweep_bdd100k_postprocess.py \
  --images-root data/bdd100k/images/100k/val \
  --labels data/bdd100k/labels/det_20/det_val.json \
  --predictions "$PREDICTIONS" \
  --output-dir "$SWEEP_DIR" \
  --frame-stride 2 \
  --frame-offset 1 \
  --score-thresholds 0.20 0.25 \
  --kind-score-thresholds pedestrian=0.20 vehicle=0.25 traffic_sign=0.25 traffic_light=0.20 \
  --default-nms-iou 0.45 0.50 0.55 \
  --kind-nms-iou traffic_light=0.35,0.40 traffic_sign=0.35,0.40 pedestrian=0.45,0.50 \
  --top-k 10

step "Write comparison summary"
"$PY" - <<'PYEOF'
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(".").resolve()
sys.path.insert(0, str(ROOT))

from scripts.evaluate_bdd100k import DEFAULT_KINDS, _load_label_frames
from scripts.sweep_bdd100k_postprocess import (
    _build_frame_records,
    _evaluate_combo,
    _load_predictions,
)


def macro_f1(report: dict) -> float:
    metrics = report.get("object_metrics") or {}
    values = [float(item.get("f1", 0.0)) for item in metrics.values()]
    return sum(values) / max(len(values), 1)


selected_kinds = set(DEFAULT_KINDS)
frames, frame_selection = _load_label_frames(
    Path("data/bdd100k/labels/det_20/det_val.json"),
    max_images=None,
    frame_stride=2,
    frame_offset=1,
)
preds = _load_predictions(
    Path("outputs/bdd100k_yolo_current_best_cache_low_odd_5000_predictions.json"),
    selected_kinds,
)
records = _build_frame_records(
    frames,
    Path("data/bdd100k/images/100k/val"),
    selected_kinds,
    preds,
)
threshold_only = _evaluate_combo(
    frame_records=records,
    labels_path="data/bdd100k/labels/det_20/det_val.json",
    images_root="data/bdd100k/images/100k/val",
    predictions_path="outputs/bdd100k_yolo_current_best_cache_low_odd_5000_predictions.json",
    selected_kinds=selected_kinds,
    default_threshold=0.20,
    kind_thresholds={
        "pedestrian": 0.20,
        "vehicle": 0.25,
        "traffic_sign": 0.25,
        "traffic_light": 0.20,
    },
    default_nms_iou=1.0,
    nms_by_kind_map={},
    iou_threshold=0.5,
    frame_selection=frame_selection,
)

sweep_summary = json.loads(Path("outputs/postprocess_sweep_production/summary.json").read_text())
best = sweep_summary.get("best") or {}
threshold_macro = macro_f1(threshold_only)
best_macro = macro_f1(best)
delta = best_macro - threshold_macro
post = best.get("postprocess") or {}
weight_note = "proxy even 1ep bootstrap" if Path("outputs/models/adas_yolov8n_bdd100k_even_1024_1ep.pt").exists() else "existing weight"

lines = [
    "# Post-NMS production sweep",
    "",
    "## Setup",
    "- weight: `outputs/models/adas_yolov8n_bdd100k.pt` "
    f"({weight_note})",
    "- predictions: `outputs/bdd100k_yolo_current_best_cache_low_odd_5000_predictions.json`",
    "- report split: odd index, stride=2, offset=1 (5,000 frames)",
    "",
    "## Macro F1",
    f"- kind thresholds only (no post-NMS): **{threshold_macro:.4f}**",
    f"- best post-NMS combo: **{best_macro:.4f}** ({delta:+.4f})",
    "",
    "## Best post-NMS settings",
    f"- default score threshold: {post.get('default_score_threshold')}",
    f"- kind score thresholds: `{post.get('kind_score_thresholds')}`",
    f"- default NMS IoU: {post.get('default_nms_iou')}",
    f"- kind NMS IoU: `{post.get('nms_iou_by_kind')}`",
    "",
    "## Per-kind F1 (best post-NMS)",
]
for kind, metrics in sorted((best.get("object_metrics") or {}).items()):
    lines.append(
        f"- {kind}: f1={metrics.get('f1'):.4f} "
        f"(p={metrics.get('precision'):.4f}, r={metrics.get('recall'):.4f})"
    )
lines.extend(
    [
        "",
        "## Note",
        "- PLAN baseline macro F1 **0.6355** uses the original finetuned weight; "
        "this run bootstrapped a proxy weight when the canonical checkpoint was missing.",
        "- Re-run with the real `adas_yolov8n_bdd100k.pt` to compare against PLAN numbers.",
        "",
    ]
)
Path("outputs/postprocess_sweep_production/compare.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
PYEOF

echo
echo "Done."
echo "  weight: $WEIGHT"
echo "  predictions: $PREDICTIONS"
echo "  raw report: $RAW_REPORT"
echo "  sweep summary: $SWEEP_DIR/summary.json"
echo "  compare: $COMPARE_MD"
