#!/usr/bin/env bash
# Generate a small BDD100K prediction cache and run post-NMS sweep (no official train required).
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PY="${PY:-.venv/bin/python}"
MAX_IMAGES="${MAX_IMAGES:-200}"
FRAME_STRIDE="${FRAME_STRIDE:-25}"
FRAME_OFFSET="${FRAME_OFFSET:-1}"
PREDICTIONS="${PREDICTIONS:-outputs/bdd100k_yolo_bootstrap_cache_low_odd_predictions.json}"
SWEEP_DIR="${SWEEP_DIR:-outputs/postprocess_sweep_bootstrap}"

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

step "Generate cache-low predictions (bootstrap yolov8n.pt)"
mkdir -p outputs
if [[ ! -f "$PREDICTIONS" ]]; then
  "$PY" scripts/evaluate_bdd100k.py \
    --images-root data/bdd100k/images/100k/val \
    --labels data/bdd100k/labels/det_20/det_val.json \
    --config configs/bdd100k_yolo_bootstrap_cache_low.yaml \
    --max-images "$MAX_IMAGES" \
    --frame-stride "$FRAME_STRIDE" \
    --frame-offset "$FRAME_OFFSET" \
    --save-predictions "$PREDICTIONS" \
    --output outputs/bdd100k_yolo_bootstrap_cache_low_odd_report.json
else
  echo "Reusing existing predictions: $PREDICTIONS"
fi

step "Run post-NMS sweep"
"$PY" scripts/sweep_bdd100k_postprocess.py \
  --images-root data/bdd100k/images/100k/val \
  --labels data/bdd100k/labels/det_20/det_val.json \
  --predictions "$PREDICTIONS" \
  --output-dir "$SWEEP_DIR" \
  --max-images "$MAX_IMAGES" \
  --frame-stride "$FRAME_STRIDE" \
  --frame-offset "$FRAME_OFFSET" \
  --score-thresholds 0.20 0.25 \
  --kind-score-thresholds pedestrian=0.25 vehicle=0.25 traffic_sign=0.25 traffic_light=0.20 \
  --default-nms-iou 0.45 0.50 \
  --kind-nms-iou traffic_light=0.35,0.40 traffic_sign=0.35,0.40 \
  --top-k 5

echo
echo "Done."
echo "  predictions: $PREDICTIONS"
echo "  sweep summary: $SWEEP_DIR/summary.json"
