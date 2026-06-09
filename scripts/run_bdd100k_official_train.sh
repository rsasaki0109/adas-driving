#!/usr/bin/env bash
# Run the official BDD100K train split pipeline end-to-end:
#   1. validate train + val data is present
#   2. export YOLO-format dataset combining official train + val
#   3. train YOLOv8n at 1024px from outputs/models/adas_yolov8n_bdd100k.pt
#   4. evaluate the new model untuned on the odd-index 5,000 report split
#   5. cache low-threshold predictions on even 1,000 tune split
#   6. sweep per-kind thresholds
#   7. evaluate tuned config on odd 5,000
#   8. compare against current best
#
# Usage:
#   bash scripts/run_bdd100k_official_train.sh [EPOCHS]
#
# Default EPOCHS=10. Expected runtime on GPU: ~3-4 hours total
# (most of it is training).
set -euo pipefail

EPOCHS="${1:-10}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PY="${PY:-.venv/bin/python}"

TRAIN_IMG_DIR="data/bdd100k/images/100k/train"
TRAIN_LABELS="data/bdd100k/labels/det_20/det_train.json"
VAL_IMG_DIR="data/bdd100k/images/100k/val"
VAL_LABELS="data/bdd100k/labels/det_20/det_val.json"
BASE_WEIGHT="outputs/models/adas_yolov8n_bdd100k.pt"
FALLBACK_WEIGHT="yolov8n.pt"

EXPORT_DIR="data/bdd100k_yolo_adas_objects_official_train_val"
RUN_NAME="adas_yolov8n_bdd100k_official_train_${EPOCHS}ep_1024"
WEIGHT_OUT="outputs/models/${RUN_NAME}.pt"
CONFIG_BASE="configs/bdd100k_yolo_finetuned_all_official_train_${EPOCHS}ep_1024.yaml"
CONFIG_CACHE="configs/bdd100k_yolo_finetuned_all_official_train_${EPOCHS}ep_1024_cache_low.yaml"
CONFIG_TUNED="configs/bdd100k_yolo_finetuned_all_official_train_${EPOCHS}ep_1024_tuned.yaml"
REPORT_UNTUNED="outputs/bdd100k_yolo_finetuned_all_official_train_${EPOCHS}ep_1024_report_odd_5000_size.json"
PRED_CACHE="outputs/bdd100k_yolo_finetuned_all_official_train_${EPOCHS}ep_1024_cache_low_even_1000_predictions.json"
REPORT_CACHE="outputs/bdd100k_yolo_finetuned_all_official_train_${EPOCHS}ep_1024_cache_low_even_1000_report.json"
SWEEP_DIR="outputs/bdd100k_official_train_${EPOCHS}ep_1024_cached_threshold_sweep_even_1000"
REPORT_TUNED="outputs/bdd100k_yolo_finetuned_all_official_train_${EPOCHS}ep_1024_tuned_report_odd_5000_size.json"
COMPARE_BASE="outputs/bdd100k_yolo_official_train_${EPOCHS}ep_1024_compare_report_odd_5000"

CURRENT_BEST_REPORT="outputs/bdd100k_yolo_finetuned_all_tuned_split_img1024_kind_tuned_tta_tuned_tiny_report_odd_5000_size.json"
PREVIOUS_BEST_REPORT="outputs/bdd100k_yolo_finetuned_all_tuned_split_img1024_kind_tuned_report_odd_5000_size.json"

step() { printf "\n========== %s ==========\n" "$1"; }
fail() { printf "ERROR: %s\n" "$1" >&2; exit 1; }

step "Step 1: validate train + val data"
[[ -d "$TRAIN_IMG_DIR" ]] || fail "Missing $TRAIN_IMG_DIR (expected official BDD100K train images)."
[[ -f "$TRAIN_LABELS" ]] || fail "Missing $TRAIN_LABELS (expected official BDD100K train detection labels)."
[[ -d "$VAL_IMG_DIR" ]]   || fail "Missing $VAL_IMG_DIR (expected official BDD100K val images)."
[[ -f "$VAL_LABELS" ]]    || fail "Missing $VAL_LABELS (expected official BDD100K val detection labels)."
TRAIN_IMG_COUNT="$(find "$TRAIN_IMG_DIR" -maxdepth 1 -type f -name '*.jpg' | wc -l)"
VAL_IMG_COUNT="$(find "$VAL_IMG_DIR" -maxdepth 1 -type f -name '*.jpg' | wc -l)"
echo "train images: $TRAIN_IMG_COUNT"
echo "val   images: $VAL_IMG_COUNT"
"$PY" scripts/check_bdd100k.py --images-root "$TRAIN_IMG_DIR" --labels "$TRAIN_LABELS" --max-samples 200
"$PY" scripts/check_bdd100k.py --images-root "$VAL_IMG_DIR"   --labels "$VAL_LABELS"   --max-samples 200

step "Step 2: export YOLO-format dataset (official train + val)"
"$PY" scripts/export_bdd100k_yolo.py \
  --images-root "$TRAIN_IMG_DIR" \
  --labels "$TRAIN_LABELS" \
  --val-images-root "$VAL_IMG_DIR" \
  --val-labels "$VAL_LABELS" \
  --output-dir "$EXPORT_DIR" \
  --classes car truck bus bicycle motorcycle train pedestrian rider "traffic sign" "traffic light" \
  --clear-output

step "Step 3: train YOLOv8n 1024px ${EPOCHS}ep from ${BASE_WEIGHT:-$FALLBACK_WEIGHT}"
if [[ ! -f "$BASE_WEIGHT" ]]; then
  echo "WARNING: Missing $BASE_WEIGHT; bootstrapping Ultralytics $FALLBACK_WEIGHT for official-train fine-tune."
  "$PY" -c "from ultralytics import YOLO; YOLO('${FALLBACK_WEIGHT}')"
  BASE_WEIGHT="$FALLBACK_WEIGHT"
fi
"$PY" - <<PYEOF
from ultralytics import YOLO
YOLO("${BASE_WEIGHT}").train(
    data="${EXPORT_DIR}/dataset.yaml",
    epochs=${EPOCHS},
    imgsz=1024,
    batch=8,
    device=0,
    workers=4,
    project="outputs/yolo_train",
    name="${RUN_NAME}",
    exist_ok=True,
    plots=False,
    save_period=1,
)
PYEOF
TRAIN_BEST="runs/detect/outputs/yolo_train/${RUN_NAME}/weights/best.pt"
[[ -f "$TRAIN_BEST" ]] || fail "Training did not produce $TRAIN_BEST"
mkdir -p outputs/models
cp "$TRAIN_BEST" "$WEIGHT_OUT"
echo "Copied trained weights to $WEIGHT_OUT"

step "Step 4: write configs that point at the new weights"
cat > "$CONFIG_BASE" <<YEOF
# Official BDD100K train + val export, ${EPOCHS}ep at 1024px from the previous
# adas_yolov8n_bdd100k.pt. Use this as an untuned baseline before threshold
# tuning. Report on the odd-index 5,000 frames.
objects:
  enabled: true
  backend: ultralytics
  model: ${WEIGHT_OUT}
  device: auto
  image_size: 1024
  score_threshold: 0.15
  score_thresholds_by_kind:
    pedestrian: 0.20
    vehicle: 0.25
    traffic_sign: 0.25
    traffic_light: 0.20
  iou_threshold: 0.50
  max_detections: 240
  class_groups:
    pedestrian:
      - pedestrian
      - rider
    vehicle:
      - bicycle
      - bus
      - car
      - motorcycle
      - train
      - truck
    traffic_sign:
      - traffic sign
    traffic_light:
      - traffic light

signs:
  enabled: false

traffic_lights:
  enabled: false

lane_smoothing:
  enabled: false

tracking:
  enabled: false

distance_estimation:
  enabled: false

visualization:
  show_summary: true
YEOF

cat > "$CONFIG_CACHE" <<YEOF
# Low-threshold cache config for official train ${EPOCHS}ep threshold sweeps.
objects:
  enabled: true
  backend: ultralytics
  model: ${WEIGHT_OUT}
  device: auto
  image_size: 1024
  score_threshold: 0.05
  score_thresholds_by_kind:
    pedestrian: 0.05
    vehicle: 0.05
    traffic_sign: 0.05
    traffic_light: 0.05
  iou_threshold: 0.50
  max_detections: 500
  class_groups:
    pedestrian:
      - pedestrian
      - rider
    vehicle:
      - bicycle
      - bus
      - car
      - motorcycle
      - train
      - truck
    traffic_sign:
      - traffic sign
    traffic_light:
      - traffic light

signs:
  enabled: false

traffic_lights:
  enabled: false

lane_smoothing:
  enabled: false

tracking:
  enabled: false

distance_estimation:
  enabled: false

visualization:
  show_summary: true
YEOF

step "Step 5: evaluate untuned config on odd 5,000 report split"
"$PY" scripts/evaluate_bdd100k.py \
  --images-root "$VAL_IMG_DIR" \
  --labels "$VAL_LABELS" \
  --config "$CONFIG_BASE" \
  --device cuda \
  --frame-stride 2 \
  --frame-offset 1 \
  --group-by-size \
  --progress-every 1000 \
  --output "$REPORT_UNTUNED"

step "Step 6: cache low-threshold predictions on even 1,000 tune split"
"$PY" scripts/evaluate_bdd100k.py \
  --images-root "$VAL_IMG_DIR" \
  --labels "$VAL_LABELS" \
  --config "$CONFIG_CACHE" \
  --device cuda \
  --frame-stride 2 \
  --frame-offset 0 \
  --max-images 1000 \
  --progress-every 200 \
  --save-predictions "$PRED_CACHE" \
  --output "$REPORT_CACHE"

step "Step 7: sweep per-kind thresholds (60 combos)"
"$PY" scripts/sweep_bdd100k_cached_predictions.py \
  --images-root "$VAL_IMG_DIR" \
  --labels "$VAL_LABELS" \
  --predictions "$PRED_CACHE" \
  --output-dir "$SWEEP_DIR" \
  --score-thresholds 0.05 \
  --kind-score-thresholds \
    pedestrian=0.10,0.15,0.20,0.25,0.30 \
    vehicle=0.20,0.25,0.30 \
    traffic_sign=0.10,0.15,0.20,0.25,0.30 \
    traffic_light=0.10,0.15,0.20,0.25 \
  --frame-stride 2 \
  --frame-offset 0 \
  --max-images 1000 \
  --group-by-size \
  --markdown

step "Step 8: pick best threshold from sweep, write tuned config"
BEST_NAME="$("$PY" - <<PYEOF
import json
with open("${SWEEP_DIR}/comparison.json") as f:
    data = json.load(f)
print(data["best"]["macro_f1"]["name"])
PYEOF
)"
echo "tune-split best: $BEST_NAME"
"$PY" - <<PYEOF
import json, re, pathlib
name = "${BEST_NAME}"
m = re.match(
    r"score_(\d{3})_kind_pedestrian_(\d{3})_traffic_light_(\d{3})_traffic_sign_(\d{3})_vehicle_(\d{3})",
    name,
)
if not m:
    raise SystemExit(f"unexpected best name: {name}")
default_t, ped, tl, ts, veh = (int(g) / 1000.0 for g in m.groups())
config = f"""# Official BDD100K train + val export, ${EPOCHS}ep at 1024px from the previous
# adas_yolov8n_bdd100k.pt, with per-kind score thresholds tuned on the
# even-index 1,000 tune split via scripts/sweep_bdd100k_cached_predictions.py.
# Report on the odd-index 5,000 frames.
objects:
  enabled: true
  backend: ultralytics
  model: ${WEIGHT_OUT}
  device: auto
  image_size: 1024
  score_threshold: {default_t:.2f}
  score_thresholds_by_kind:
    pedestrian: {ped:.2f}
    vehicle: {veh:.2f}
    traffic_sign: {ts:.2f}
    traffic_light: {tl:.2f}
  iou_threshold: 0.50
  max_detections: 240
  class_groups:
    pedestrian:
      - pedestrian
      - rider
    vehicle:
      - bicycle
      - bus
      - car
      - motorcycle
      - train
      - truck
    traffic_sign:
      - traffic sign
    traffic_light:
      - traffic light

signs:
  enabled: false

traffic_lights:
  enabled: false

lane_smoothing:
  enabled: false

tracking:
  enabled: false

distance_estimation:
  enabled: false

visualization:
  show_summary: true
"""
pathlib.Path("${CONFIG_TUNED}").write_text(config)
print("wrote ${CONFIG_TUNED}")
PYEOF

step "Step 9: evaluate tuned config on odd 5,000 report split"
"$PY" scripts/evaluate_bdd100k.py \
  --images-root "$VAL_IMG_DIR" \
  --labels "$VAL_LABELS" \
  --config "$CONFIG_TUNED" \
  --device cuda \
  --frame-stride 2 \
  --frame-offset 1 \
  --group-by-size \
  --progress-every 1000 \
  --output "$REPORT_TUNED"

step "Step 10: compare against current best and previous best"
"$PY" scripts/compare_evaluations.py \
  --reports \
    "$CURRENT_BEST_REPORT" \
    "$PREVIOUS_BEST_REPORT" \
    "$REPORT_UNTUNED" \
    "$REPORT_TUNED" \
  --names \
    current_best_tta_tuned_tiny \
    previous_best_no_tta \
    official_train_${EPOCHS}ep_untuned \
    official_train_${EPOCHS}ep_tuned \
  --output "${COMPARE_BASE}.json" \
  --markdown-output "${COMPARE_BASE}.md" \
  --csv-output "${COMPARE_BASE}.csv"

echo
echo "Done. Key artifacts:"
echo "  - $WEIGHT_OUT"
echo "  - $CONFIG_BASE"
echo "  - $CONFIG_TUNED"
echo "  - $REPORT_UNTUNED"
echo "  - $REPORT_TUNED"
echo "  - ${COMPARE_BASE}.md"
