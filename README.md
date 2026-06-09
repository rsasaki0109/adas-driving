# adas-perception

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Tests](https://img.shields.io/badge/tests-pytest-brightgreen)

単眼 dashcam から **車線・物体・標識・信号** を検出し、JSON / 動画 overlay で見せる軽量 Python ADAS デモ。  
保存済み perception JSON から **rule-based planning overlay** も再実行できます。

**研究・デモ・教育向け** — 車両制御・安全性能保証用途ではありません。

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip && pip install -r requirements.txt -r requirements-yolo.txt
python scripts/demo_video.py --input path/to/drive.mp4 --output outputs/drive_adas.mp4
```

[![Demo](assets/demo_wbf7_poster.png)](assets/demo_wbf7.mp4)

---

## できること（概要）

| 領域 | 内容 |
|---|---|
| Perception | YOLO 物体検出、車線 (OpenCV / ONNX)、tracker、単眼距離、信号 state |
| Planning | lane target、lead follow、信号 stop/go、VRU yield、lane departure warning |
| 評価 | BDD100K macro F1、WBF 融合、scenario YAML、pytest |

BDD100K odd 5,000 での macro F1 目安: **0.6355** (高速) → **0.6763** (7-way WBF online)。  
詳細な実験記録・採用判断は [PLAN.md](PLAN.md)、方向性は [ROADMAP.md](ROADMAP.md)。

![WBF ladder](assets/wbf_ladder.png)

---

## セットアップ

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt -r requirements-yolo.txt
pip install -e ".[dev]"          # pytest 用
pip install -r requirements-bdd100k.txt   # BDD100K 評価・export 用（任意）
```

CUDA 利用時は環境に合う PyTorch を先に入れてから上記を実行してください。

---

## クイックスタート

### 動画デモ（デフォルト）

```bash
python scripts/demo_video.py \
  --input path/to/drive.mp4 \
  --output outputs/drive_adas.mp4 \
  --json-output outputs/drive.json
```

### 精度優先 / 高速デモ

| 用途 | config | macro F1 目安 |
|---|---|---|
| 精度優先 (online WBF) | `configs/bdd100k_yolo_wbf7_perkind_iou_online.yaml` | 0.6763 |
| 単一 config + TTA | `configs/bdd100k_yolo_finetuned_all_tuned_split_img1024_kind_tuned_tta_tuned_tiny.yaml` | 0.6389 |
| 高速 demo | `configs/bdd100k_yolo_finetuned_all_tuned_split_img1024_kind_tuned.yaml` | 0.6355 |

重み `outputs/models/adas_yolov8n_bdd100k.pt` はローカル配置が必要です（git 非追跡）。

### Web デモ

```bash
pip install gradio
python scripts/web_demo.py
```

### Planning overlay

**クイックデモ** (同梱 fixture、GPU 重み不要):

```bash
python scripts/run_planning_demo.py \
  --video assets/demo_wbf7.mp4 \
  --output-dir outputs/planning_demo \
  --compare-configs \
  --export-benchmark
```

出力: `planning_frames.json`, `planning_overlay.mp4`, `driving_replay.json` (perception + planning 統合)

**動画から perception → planning まで一括** (torchvision baseline、追加重み不要):

```bash
python scripts/run_planning_demo.py \
  --run-perception \
  --video assets/demo_wbf7.mp4 \
  --max-frames 120 \
  --output-dir outputs/planning_demo_live
```

**finetuned 重みあり** (`outputs/models/adas_yolov8n_bdd100k.pt` 配置済み):

```bash
python scripts/run_planning_demo.py \
  --run-perception \
  --perception-config configs/bdd100k_yolo_wbf7_demo.yaml \
  --video assets/demo_wbf7.mp4 \
  --output-dir outputs/planning_demo_wbf7
```

設定: `configs/planning/default.yaml` / `conservative.yaml` / `aggressive_demo.yaml`  
scenario 評価: `python scripts/eval_planning_scenarios.py --scenarios-dir scenarios --output outputs/scenarios.json`

---

## BDD100K 評価

**val mirror の取得** (Hugging Face、約 1GB):

```bash
python scripts/prepare_bdd100k.py --download-val --data-root data/bdd100k
```

**評価例** (odd 5,000 report split):

```bash
python scripts/evaluate_bdd100k.py \
  --images-root data/bdd100k/images/100k/val \
  --labels data/bdd100k/labels/det_20/det_val.json \
  --config configs/bdd100k_yolo_finetuned_all_tuned_split_img1024_kind_tuned_tta_tuned_tiny.yaml \
  --device cuda \
  --frame-stride 2 --frame-offset 1 \
  --group-by-size \
  --output outputs/bdd100k_eval.json
```

### 公式 train で再学習（任意・当面スキップ可）

1. [BDD100K 公式](https://bdd-data.berkeley.edu/) から train 画像 + labels を取得
2. 以下に配置:

```text
data/bdd100k/images/100k/train/
data/bdd100k/labels/det_20/det_train.json
data/bdd100k/images/100k/val/          # prepare --download-val で可
data/bdd100k/labels/det_20/det_val.json
```

3. 実行:

```bash
bash scripts/bootstrap_bdd100k_official_train.sh
RUN_TRAIN=1 bash scripts/bootstrap_bdd100k_official_train.sh 10
```

`adas_yolov8n_bdd100k.pt` が無い場合は `yolov8n.pt` から fine-tune します。

---

## 主要スクリプト

| スクリプト | 用途 |
|---|---|
| `scripts/demo_image.py` / `demo_video.py` | 画像・動画デモ |
| `scripts/replay_planning_json.py` | perception JSON → planning JSON |
| `scripts/demo_planning_video.py` | planning overlay 動画 |
| `scripts/evaluate_bdd100k.py` | BDD100K 評価 |
| `scripts/evaluate_lane.py` | 車線 detector 比較 |
| `scripts/benchmark.py` | FPS / レイテンシ計測 |
| `scripts/prepare_bdd100k.py` | データ配置・検証 |

---

## ディレクトリ（最小）

```text
adas_perception/     # 認識パイプライン
adas_planning/       # rule-based planning
configs/             # YAML 設定
scripts/             # CLI
scenarios/           # planning scenario YAML
tests/               # pytest
PLAN.md              # 実験記録・次タスク（詳細はこちら）
```

---

## テスト

```bash
pytest -q -k "not slow"
python scripts/eval_planning_scenarios.py --scenarios-dir scenarios --output outputs/scenarios.json
```

---

## 制約

- 単眼距離・planning 出力は **粗い推定 / recommendation** であり、実車 ADAS 性能を意味しません
- `data/`・`outputs/`・`*.pt`・`*.onnx` は gitignore（大容量）
- 実験の長い changelog は README ではなく [PLAN.md](PLAN.md) に集約

## ライセンス・データ

BDD100K 利用時は [公式ライセンス](https://bdd-data.berkeley.edu/) に従ってください。デモ動画素材: [Pexels CC0](https://www.pexels.com/video/dash-cam-view-of-the-road-5921059/).
