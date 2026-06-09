#!/usr/bin/env bash
# Prepare prerequisites for scripts/run_bdd100k_official_train.sh:
#   1. ensure Python venv + deps
#   2. download BDD100K val mirror if missing (Hugging Face)
#   3. verify official train split is present
#   4. bootstrap YOLO base weights if outputs/models/adas_yolov8n_bdd100k.pt is missing
#   5. optionally launch the full official-train pipeline
set -euo pipefail

EPOCHS="${1:-10}"
RUN_TRAIN="${RUN_TRAIN:-0}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PY="${PY:-.venv/bin/python}"
TRAIN_IMG_DIR="data/bdd100k/images/100k/train"
TRAIN_LABELS="data/bdd100k/labels/det_20/det_train.json"
VAL_IMG_DIR="data/bdd100k/images/100k/val"
VAL_LABELS="data/bdd100k/labels/det_20/det_val.json"
BASE_WEIGHT="outputs/models/adas_yolov8n_bdd100k.pt"

step() { printf "\n========== %s ==========\n" "$1"; }
fail() { printf "ERROR: %s\n" "$1" >&2; exit 1; }

step "Ensure Python venv"
if [[ ! -x "$PY" ]]; then
  python3 -m venv .venv
  "$PY" -m pip install -U pip
  "$PY" -m pip install -r requirements.txt -r requirements-yolo.txt -r requirements-bdd100k.txt -e ".[dev]"
fi

step "Ensure BDD100K val mirror"
VAL_COUNT=0
if [[ -d "$VAL_IMG_DIR" ]]; then
  VAL_COUNT="$(find "$VAL_IMG_DIR" -maxdepth 1 -type f -name '*.jpg' | wc -l)"
fi
if [[ ! -f "$VAL_LABELS" || "$VAL_COUNT" -lt 1000 ]]; then
  echo "Downloading validation mirror to data/bdd100k (this can take a while)..."
  "$PY" scripts/prepare_bdd100k.py --download-val --data-root data/bdd100k
else
  echo "Validation mirror already present ($VAL_COUNT images)."
fi

step "Check official train split"
TRAIN_COUNT=0
if [[ -d "$TRAIN_IMG_DIR" ]]; then
  TRAIN_COUNT="$(find "$TRAIN_IMG_DIR" -maxdepth 1 -type f -name '*.jpg' | wc -l)"
fi
if [[ ! -f "$TRAIN_LABELS" || "$TRAIN_COUNT" -lt 10000 ]]; then
  cat >&2 <<EOF
Official BDD100K train split is not ready yet.

Required paths:
  - $TRAIN_IMG_DIR   (~70,000 jpg)
  - $TRAIN_LABELS

Download from https://bdd-data.berkeley.edu/ (registration required):
  - 100K Images archive
  - Detection 2020 Labels archive

Then extract into data/bdd100k, or place archives under ~/Downloads and run:
  $PY scripts/prepare_bdd100k.py --download-dir ~/Downloads --data-root data/bdd100k --force

Current status:
  train images: $TRAIN_COUNT
  train labels: $([ -f "$TRAIN_LABELS" ] && echo present || echo missing)
EOF
  exit 2
fi

step "Bootstrap base YOLO weights"
mkdir -p outputs/models
if [[ ! -f "$BASE_WEIGHT" ]]; then
  echo "Missing $BASE_WEIGHT; Ultralytics yolov8n.pt will be used as the train starting point."
  "$PY" -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
else
  echo "Using existing base weight: $BASE_WEIGHT"
fi

if [[ "$RUN_TRAIN" == "1" ]]; then
  step "Launch official train pipeline (${EPOCHS} epochs)"
  bash scripts/run_bdd100k_official_train.sh "$EPOCHS"
else
  echo
  echo "Bootstrap complete. To start training:"
  echo "  RUN_TRAIN=1 bash scripts/bootstrap_bdd100k_official_train.sh ${EPOCHS}"
  echo "  bash scripts/run_bdd100k_official_train.sh ${EPOCHS}"
fi
