# PLAN

Last updated: 2026-04-25 (v0.x milestone wrap-up: WBF ceiling 0.6763 + Phase 7 demo improvements + headless web demo. Jetson edge deploy moved to future work)

`adas-perception` は、単眼カメラ画像/動画から車線、車両、歩行者、標識、信号候補を検出して可視化する、デモ重視のPython製ADAS認識OSSである。これは運転判断や車両制御のためのスタックではなく、公開データ上で再現可能に改善していくカメラ認識プロジェクトとして進める。

この文書は、現状の実装、評価済み実験、採用判断、次にやる作業を一か所にまとめるための作業計画である。READMEは利用者向け、ROADMAPは大枠の方向性、このPLANは開発中の意思決定と実行順を残す場所として使う。

## Claude Handoff

このセクションは、Claude へそのまま引き継ぐための短い状況整理である。細部は後ろの各Phaseにあるが、まずここを見れば今の到達点と次の一手が分かるようにする。

### 現在地

- 最終 accuracy ceiling: 7 source (no-TTA + TTA@1024 + tl-only TTA + TTA@960/1280/1536/1792)
  を WBF で融合、**per-kind iou_thr (ped=0.50, tl=0.40, ts=0.40, v=0.50)** + per-kind threshold で
  macro F1 = 0.6753 (offline) / **0.6763 (online pipeline)** に到達。全 4 クラスが
  previous best 0.6355 を大幅に超え、pedestrian +0.053、traffic_light +0.049、
  traffic_sign +0.035 (online)、vehicle +0.027。
  imgsz=2048 や class-balanced/ped-only/yolo11n retrain を 8th source に加えても改善せず、
  retrain diversity も scale diversity も 7-way で飽和している。
  online pipeline 版は single-config で実行可能 (~3.4 FPS on GPU)。

### Online WBF の conditional evaluation (odd 5000 で --group-by weather timeofday scene)

残存 weakness (n >= 100):

```text
scene/highway        n=1269  macro=0.6421  worst=pedestrian:0.546
weather/rainy        n= 373  macro=0.6492  worst=traffic_light:0.598
timeofday/night      n=1962  macro=0.6520  worst=pedestrian:0.593
scene/residential    n= 633  macro=0.6628  worst=pedestrian:0.586
weather/clear        n=2672  macro=0.6680  worst=pedestrian:0.627
```

最強条件 (参照):

```text
timeofday/daytime    n=2627  macro=0.6879  worst=traffic_sign:0.639
weather/partly cloudy n=381  macro=0.6865  worst=pedestrian:0.637
weather/snowy        n= 365  macro=0.6787  worst=traffic_light:0.617
```

### Residual error analysis on 7-way online WBF (save-errors on odd 5000)

WBF 7-way の TP/FP/FN を scene × size_bucket で breakdown すると、条件別 macro F1 の
見た目とは異なる本質的 weakness が見える:

```text
pedestrian FN total = 2934
  city street small   n=1197  (41%)  <- 最大の塊
  city street tiny    n=1033  (35%)  <- 次いで
  city street medium  n= 314  (11%)
  其他                n= 390  (13%)
  -> 全 FN の 76% が city street の small/tiny bucket

traffic_light FN total = 5471
  clear tiny          n=2235  (41%)
  tiny (全天候 合計)   n=4233  (77%)
  -> 全 FN の 77% が tiny bucket (天候問わず)
```

重要な観察:

- **highway/pedestrian F1=0.546 は misleading**: highway/pedestrian FN は全体の 4.5%
  だけ (highway 自体に歩行者が少ない)。macro F1 差は分母が小さいことで増幅されているだけ
- **rainy/traffic_light F1=0.598 も同様**: rainy/TL FN は全体の 9% のみ
- **本質的 weakness は 2 つ**:
  1. city street の small/tiny pedestrian (全 ped FN の 76%)
  2. 全天候 tiny traffic_light (全 TL FN の 77%)
- condition-specific tuning より **tiny/small bucket 向けの training data quality/quantity** が
  本当に必要な改善方向
- WBF ensemble は条件横断で pull up するが、物理的に 10-20px の小物体は recall しきれない

Gallery 生成済み (highway/pedestrian、night/pedestrian、rainy/traffic_light) は定性的
参考用として保持、future work の direction として記録。

比較表: `outputs/bdd100k_yolo_wbf7_online_errors_compare.md`
(baseline no-TTA vs WBF の TP/FP/FN 変化: pedestrian で FP -691、FN -176、TP +176。
fusion は FP を大きく削減しつつ TP も増やしている)。

```text
7-way fusion inputs (all score_threshold=0.05 low-threshold cache):
  A: outputs/bdd100k_yolo_current_best_cache_low_odd_5000_predictions.json
     (no-TTA 1024px current best)
  B: outputs/bdd100k_yolo_current_best_tta_cache_low_odd_5000_predictions.json
     (TTA 1024px current best)
  C: outputs/bdd100k_yolo_tl_only_tta_cache_low_odd_5000_predictions.json
     (tl-only retrain weights + TTA 1024)
  D: outputs/bdd100k_yolo_img1280_tta_cache_low_odd_5000_predictions.json
     (current best weights at imgsz=1280 + TTA)
  E: outputs/bdd100k_yolo_img960_tta_cache_low_odd_5000_predictions.json
     (current best weights at imgsz=960 + TTA)
  F: outputs/bdd100k_yolo_img1536_tta_cache_low_odd_5000_predictions.json
     (current best weights at imgsz=1536 + TTA)
  G: outputs/bdd100k_yolo_img1792_tta_cache_low_odd_5000_predictions.json
     (current best weights at imgsz=1792 + TTA)
fused: outputs/bdd100k_yolo_wbf7_perkind_iou_cache_low_odd_5000_predictions.json
  iou_thr=0.55 (default), kind_iou_thr=pedestrian=0.50 traffic_light=0.40
  traffic_sign=0.40 vehicle=0.50, weights=[1.0]*7, 720x1280 normalize
tuned thresholds: pedestrian=0.25, traffic_light=0.25, traffic_sign=0.30, vehicle=0.30
report: outputs/bdd100k_yolo_wbf7_perkind_iou_report_odd_5000_size.json
macro F1 = 0.6753  (+0.0006 vs iou=0.45 0.6747、+0.0398 vs previous best 0.6355)
```

クラス別 F1 (vs previous best 0.6355 delta):

```text
pedestrian    0.647  +0.053
traffic_light 0.632  +0.049
traffic_sign  0.642  +0.029
vehicle       0.780  +0.028
```

WBF ladder (cumulative, all on odd 5000 report split):

```text
previous best no-TTA          0.6355  (1 pass, 29.15 FPS)
TTA tuned tiny (single cfg)   0.6389  (1 pass, 24.71 FPS)
2-way WBF  (no-TTA + TTA)     0.6447  (2 pass, ~10.5 FPS)
3-way WBF  (+ tl-only TTA)    0.6489  (3 pass, ~7.8 FPS)
4-way WBF  (+ TTA@1280)       0.6602  (4 pass, ~4.7 FPS)
5-way WBF  (+ TTA@960)        0.6627  (5 pass, ~4.0 FPS)
6-way WBF  (+ TTA@1536)       0.6686  (6 pass, ~3.3 FPS)
7-way WBF  (+ TTA@1792) iou=0.55   0.6724  (7 pass, ~2.8 FPS)
7-way WBF  iou_thr=0.45 tuning     0.6747  (7 pass, ~2.8 FPS)
7-way WBF  per-kind iou (offline)  0.6753  (7 pass, offline batch only)
7-way WBF  per-kind iou (online)   0.6763  (single config, ~3.4 FPS)  <- 最終 ceiling
8-way WBF  (+ TTA@2048)            0.6725  plateau、採用しない
8-way WBF  (+ class-balanced TTA)  0.6717  tune overfit、採用しない
8-way WBF  (+ ped-only TTA)        0.6709  redundant、採用しない
8-way WBF  (+ yolo11n TTA)         0.6733  model family の fuse にも寄与せず、採用しない
```

観察:
- extreme scale (1280, 1536, 1792) は 1024 との overlap が少なく大きく効く
- 960 は 1024 と overlap するため incremental gain が小さい (+0.0025)
- 2048 は plateau、scale diversity からの gain がほぼ尽きた
- iou_thr=0.45 (default 0.55) で fusion clustering を緩めると traffic_light/traffic_sign
  で明確に改善 (+0.007, +0.004)、pedestrian/vehicle は微減だが macro は +0.0023
- **per-kind iou_thr** (ped=0.50, tl=0.40, ts=0.40, v=0.50) でさらに +0.0006 改善
  (小/密 objects の tl/ts は loose clustering、大/疎 objects の vehicle は tight clustering が合う)
- 8th source として追加した retrain (class-balanced, ped-only) は全て失敗
  (scale diversity で saturated な ensemble には redundant)

制約:

- 7 回の forward pass が必要 (~2.8 FPS 見込み)、offline batch 限定。
- pipeline 側の multi-config 融合サポートは未実装。

```text
fusion inputs (all at score_threshold=0.05 low-threshold cache):
  A: outputs/bdd100k_yolo_current_best_cache_low_odd_5000_predictions.json
     (no-TTA 1024px current best)
  B: outputs/bdd100k_yolo_current_best_tta_cache_low_odd_5000_predictions.json
     (TTA 1024px current best)
  C: outputs/bdd100k_yolo_tl_only_tta_cache_low_odd_5000_predictions.json
     (tl-only retrain weights + TTA)
fused: outputs/bdd100k_yolo_wbf3_cache_low_odd_5000_predictions.json
  iou_thr=0.55, weights=[1.0, 1.0, 1.0], image_size=720x1280
tuned thresholds: pedestrian=0.30, traffic_light=0.25, traffic_sign=0.25, vehicle=0.30
report: outputs/bdd100k_yolo_wbf3_report_odd_5000_size.json
macro F1 = 0.6489  (+0.0042 vs 2-way WBF 0.6447, +0.0134 vs no-TTA 0.6355)
```

クラス別 F1 (vs previous best 0.6355 delta):

```text
pedestrian    0.617  +0.023
traffic_light 0.601  +0.018   <- 初めて 0.60 超え
traffic_sign  0.614  +0.001
vehicle       0.764  +0.012
```

前の 2-way WBF (no-TTA + TTA) も参照設定として残す。こちらは 2 回の forward pass で済む:

```text
fused: outputs/bdd100k_yolo_wbf_cache_low_odd_5000_predictions.json
tuned thresholds: pedestrian=0.25, traffic_light=0.25, traffic_sign=0.30, vehicle=0.30
macro F1 = 0.6447
```

制約:

- これは単一 config ではなく、no-TTA + TTA 1024 + tl-only retrain + TTA の 3 回の
  forward pass + WBF 融合のパイプライン (~128 ms / frame ~= 7.8 FPS)。
- pipeline 側の multi-config 融合サポートは未実装。cached predictions を生成して
  `scripts/fuse_bdd100k_predictions.py` でオフライン融合する手順で運用する。
- 単一 config で動く現時点の best は TTA tuned + tiny (macro F1 = 0.6389)。
- tl-only retrain 単体は macro F1 = 0.6325 で baseline より弱いが、WBF の
  complementary source として +0.0042 を寄与する (ensemble diversity の効果)。

### 使い分け

精度最優先 (accuracy ceiling, online single config, 7 forward passes in-process):

```text
configs/bdd100k_yolo_wbf7_perkind_iou_online.yaml
macro F1 = 0.6763, ~3.4 FPS (GPU)
```

精度最優先 (accuracy ceiling, offline batch, 7 forward passes):

```text
inputs (score_threshold=0.05 low-threshold cache):
  configs/bdd100k_yolo_finetuned_all_tuned_split_img1024_kind_tuned_cache_low.yaml
  configs/bdd100k_yolo_finetuned_all_tuned_split_img1024_kind_tuned_tta_cache_low.yaml
  configs/bdd100k_yolo_finetuned_all_even_copy_paste_v2_tl_1000_1024_1ep_tta_cache_low.yaml
  configs/bdd100k_yolo_finetuned_all_tuned_split_img1280_tta_cache_low.yaml
  configs/bdd100k_yolo_finetuned_all_tuned_split_img960_tta_cache_low.yaml
  configs/bdd100k_yolo_finetuned_all_tuned_split_img1536_tta_cache_low.yaml
  configs/bdd100k_yolo_finetuned_all_tuned_split_img1792_tta_cache_low.yaml
fusion: scripts/fuse_bdd100k_predictions.py --iou-thr 0.55 \
  --kind-iou-thr pedestrian=0.50 traffic_light=0.40 traffic_sign=0.40 vehicle=0.50 \
  --image-size 720 1280
thresholds: pedestrian=0.25 / vehicle=0.30 / traffic_sign=0.30 / traffic_light=0.25
macro F1 = 0.6753, ~2.8 FPS (offline batch workflow; online config above is +0.0010 better)
```

精度優先 (accuracy ceiling, offline batch, 4 forward passes、marginal 精度差):

```text
inputs:
  ... (上の A〜D)
fusion: scripts/fuse_bdd100k_predictions.py --iou-thr 0.55 --image-size 720 1280
thresholds: pedestrian=0.25 / vehicle=0.30 / traffic_sign=0.30 / traffic_light=0.25
macro F1 = 0.6602, ~4.7 FPS
```

精度優先 (accuracy ceiling, offline batch, 3 forward passes):

```text
inputs:
  configs/bdd100k_yolo_finetuned_all_tuned_split_img1024_kind_tuned_cache_low.yaml
  configs/bdd100k_yolo_finetuned_all_tuned_split_img1024_kind_tuned_tta_cache_low.yaml
  configs/bdd100k_yolo_finetuned_all_even_copy_paste_v2_tl_1000_1024_1ep_tta_cache_low.yaml
fusion: scripts/fuse_bdd100k_predictions.py --iou-thr 0.55 --image-size 720 1280
thresholds: pedestrian=0.30 / vehicle=0.30 / traffic_sign=0.25 / traffic_light=0.25
macro F1 = 0.6489, ~7.8 FPS
```

精度優先 (accuracy ceiling, offline batch, 2 forward passes):

```text
inputs: no-TTA current best + TTA current best
fusion: scripts/fuse_bdd100k_predictions.py --iou-thr 0.55 --image-size 720 1280
thresholds: pedestrian=0.25 / vehicle=0.30 / traffic_sign=0.30 / traffic_light=0.25
macro F1 = 0.6447, ~10.5 FPS
```

精度優先 (single-config, online):

```text
configs/bdd100k_yolo_finetuned_all_tuned_split_img1024_kind_tuned_tta_tuned_tiny.yaml
macro F1 = 0.6389, FPS = 24.71
```

途中の TTA tuned (tiny override なし、参照値):

```text
configs/bdd100k_yolo_finetuned_all_tuned_split_img1024_kind_tuned_tta_tuned.yaml
macro F1 = 0.6385, FPS = 25.69
```

高速 demo 用 (TTAなし、previous best):

```text
configs/bdd100k_yolo_finetuned_all_tuned_split_img1024_kind_tuned.yaml
macro F1 = 0.6355, FPS = 29.15
```

- official BDD100K train split はローカルにまだ無い。

```text
present:
data/bdd100k/images/100k/val
data/bdd100k/labels/det_20/det_val.json

missing:
data/bdd100k/images/100k/train
data/bdd100k/labels/det_20/det_train.json
```

- そのため、公式train再学習ではなく、validation mirror上での error analysis と copy-paste v2 準備まで進めた。

### 直近で入った変更

1. 条件別評価を追加して current best の弱点を切った。

```text
outputs/bdd100k_yolo_finetuned_all_tuned_split_img1024_kind_tuned_report_odd_5000_grouped_size.json
```

主要な弱点:

```text
scene/highway       macro F1=0.6026  worst=pedestrian
timeofday/night     macro F1=0.6170  worst=pedestrian
weather/rainy       macro F1=0.6147  worst=traffic_light
timeofday/dawn/dusk macro F1=0.6244  worst=traffic_sign
```

2. エラー集計スクリプトとギャラリースクリプトに BDD100K frame属性フィルタを追加した。

```text
scripts/analyze_bdd100k_errors.py
- --labels
- --group-by weather timeofday scene

scripts/visualize_bdd100k_errors.py
- --labels
- --where ATTR=VALUE
```

生成済み成果物:

```text
outputs/bdd100k_yolo_img1024_kind_tuned_report_odd_5000_errors_all_psl_grouped_analysis.json
outputs/bdd100k_yolo_img1024_kind_tuned_report_odd_5000_errors_all_psl_grouped_analysis.md
outputs/bdd100k_yolo_img1024_kind_tuned_error_gallery_night_pedestrian/index.md
outputs/bdd100k_yolo_img1024_kind_tuned_error_gallery_highway_pedestrian/index.md
outputs/bdd100k_yolo_img1024_kind_tuned_error_gallery_rainy_traffic_light/index.md
```

3. copy-paste v2 の初期実装を exporter に追加した。

```text
scripts/export_bdd100k_yolo.py
- --copy-paste-source-min-area
- --copy-paste-source-min-box-size
- --copy-paste-source-max-aspect-ratio
- --copy-paste-mask none|box|grabcut
```

4. copy-paste v2 の smoke export は通した。

```text
outputs/smoke_bdd100k_copy_paste_v2_yolo
copy_paste_images=8
checked_labels=683
bad_labels=0
mask=grabcut
blend=feather
```

### すでに結論が出ていること

- feather copy-paste は current best を超えなかった。ここは深追いしない。

```text
untuned macro F1 = 0.6234
tuned   macro F1 = 0.6242
baseline macro F1 = 0.6355
```

- object crop 単体 oversampling は悪化した。
- class-balanced repeat も current best を超えなかった。
- tile inference と tiny-only size-aware threshold も current best を超えなかった。
- copy-paste v2 (grabcut mask + feather blend, 1000 paste images) も current best を超えなかった。

```text
untuned macro F1 = 0.6263
tuned   macro F1 = 0.6310
baseline macro F1 = 0.6355
```

  traffic_light F1 だけは current best を少し上回った (0.5887 vs 0.5830, +0.006)。
  他クラスは全て下がり、macro では届かなかった。
  grabcut mask + source filter でも bbox 矩形 copy-paste の範囲を出ていない。

- copy-paste v2 traffic_light only (1000 paste images) も同様に超えなかった。

```text
untuned macro F1 = 0.6285
tuned   macro F1 = 0.6325
baseline macro F1 = 0.6355
```

  `traffic_light` F1 は全実験中最大 (0.5915, current best +0.009) になったが、
  `traffic_sign` と `pedestrian` が同程度下がり、macro では届かない。
  traffic_sign を copy-paste 対象から外しても traffic_sign F1 は下がっている (-0.011)。
  これは追加学習自体による分布変化であり、copy-paste source class の選択では抑えられない。

### Claude が次にやるべきこと (v0.x milestone wrap-up 後)

**v0.x milestone は到達済み**。inference-side accuracy 探索 (WBF 7-way ceiling 0.6763)、
Phase 7 demo improvements (visualization filter / label collision / distance format /
demo-oriented WBF config)、Phase 9 tracker upgrade (motion prediction + centroid fallback)、
headless web demo (gradio) はすべて完了している。下記は今 session の総まとめと、
次に作業を再開するときの優先順。

#### 完了 (v0.x milestone)

- ~~README / ROADMAP / demo scripts を新 current best (online WBF) に追従~~
- ~~size-bucket threshold (tiny は採用、small は overfit)~~
- ~~WBF 2/3/4/5/6/7-way ladder (3-way から 7-way まで全 step 採用)~~
- ~~8-way 試行 5 連敗 (2048 / class-balanced / ped-only / yolo11n / CLAHE)~~
- ~~WBF iou_thr grid + per-kind iou_thr (ped=0.50/v=0.50/tl=0.40/ts=0.40)~~
- ~~WBF pipeline online 化 (`ADASPerceptionPipeline` に `objects.fusion.mode=wbf`)~~
- ~~conditional evaluation + residual error analysis (city street small/tiny ped が本質的弱点)~~
- ~~Phase 7: visualization 拡張 (`include_kinds` / `exclude_kinds` / `min_confidence` /
  `label_style` / `avoid_label_overlap` / `distance_format`)~~
- ~~Phase 9: tracker に `motion_prediction` + `centroid_distance_fraction` 追加~~
- ~~headless web demo (`scripts/web_demo.py`、gradio ベース)~~

#### 次に作業を再開する時の優先順 (高い順)

1. **official BDD100K train split の配置 + 公式 train で再学習** (最優先、user action 待ち)
   - 追加学習自体が macro -0.005 になる現状の問題は train data が val mirror で短い epoch だけ
     回しているせい。官公 train で long epoch を回す価値が高い
   - `scripts/run_bdd100k_official_train.sh` で配置後ワンコマンド実行
2. **将来的な Jetson 対応** (実機入手後に着手):
   - ONNX export → TensorRT engine 化 (FP16 / INT8)
   - Jetson Orin Nano / Nano 級向け軽量 config (640px YOLO + no-TTA)
   - benchmark.py を Jetson で回して FPS 計測
   - `scripts/web_demo.py` は ROS/RViz 不要なので、Jetson に挿して LAN 内ブラウザから確認できる
     可視化として既に動く構造になっている (実機確認は未)
3. inference-side の更なる積み上げ (期待値低、saturated):
   - YOLOv8s への置換 (model size diversity、長 epoch 必要)
   - 条件別 (scene/weather/timeofday) post-processing tuning (過去 overfit 実績あり、慎重に)
   - 別 dataset からの pretrain (open-images、nuscenes 等)
4. Phase 8: lane recognition 強化 (TuSimple/CULane で segmentation モデル)。完全な未着手項目
5. Phase 7 残り: distance estimation の実 calibration (`fx`, `fy` 入力の追加サポート)

過去の不採用:

copy-paste v2 mask 1000 と tl-only 1000 の両方が採用基準を満たさなかった。

Ablation で分かったこと (odd 5000 report split):

```text
current_best (no retrain)                 macro F1 = 0.6355
even_1024_1ep (plain retrain, no copy-paste) macro F1 = 0.6309   delta=-0.005
v2 mixed tuned (copy-paste 1000)          macro F1 = 0.6310   delta=-0.005
v2 tl-only tuned (copy-paste 1000)        macro F1 = 0.6325   delta=-0.003
```

主な regression は「1 epoch 追加学習そのもの」。
copy-paste を plain retrain と比べると:

```text
v2 mixed   vs plain retrain   macro +0.0001  TL +0.004  TS -0.006
v2 tl-only vs plain retrain   macro +0.0016  TL +0.007  TS -0.002
```

つまり tl-only copy-paste は plain retrain に対して純利得だが、
その plain retrain 自体が current best を -0.005 下げるため、結局届かない。
validation mirror 上の短い追加学習で current best を超える道は、ここからは細い。

優先順:

1. official BDD100K train split の配置待ち (最優先、期待値が段違い)
2. 追加学習を経ないアプローチ:
   current best model に対して inference-side で改善する方向
   (例: class/kind-specific post-processing, small-object-specific NMS, better TTA)
3. traffic_light 特化 config (tl-only tuned) を README の demo とは別の
   experiment として正式に残す

copy-paste v2 mask 1000 の再現コマンド (結果は採用不可):

```bash
.venv/bin/python scripts/export_bdd100k_yolo.py \
  --images-root data/bdd100k/images/100k/val \
  --labels data/bdd100k/labels/det_20/det_val.json \
  --output-dir data/bdd100k_yolo_adas_objects_even_copy_paste_v2_mask_1000 \
  --classes car truck bus bicycle motorcycle train pedestrian rider "traffic sign" "traffic light" \
  --split-mode alternate \
  --frame-stride 2 \
  --train-frame-offset 0 \
  --val-frame-offset 1 \
  --copy-paste-classes pedestrian "traffic sign" "traffic light" \
  --copy-paste-area-threshold 0.0025 \
  --copy-paste-source-min-area 0.00002 \
  --copy-paste-source-min-box-size 8 \
  --copy-paste-source-max-aspect-ratio 8.0 \
  --copy-paste-max-images 1000 \
  --copy-paste-objects-per-image 1 \
  --copy-paste-context-padding 0.20 \
  --copy-paste-scale-min 0.9 \
  --copy-paste-scale-max 1.1 \
  --copy-paste-max-overlap 0.05 \
  --copy-paste-mask grabcut \
  --copy-paste-blend feather \
  --copy-paste-feather-ratio 0.08 \
  --copy-paste-seed 17 \
  --clear-output
```

学習は `yolo` CLI が無い環境だったので python API で回した:

```bash
.venv/bin/python -c "
from ultralytics import YOLO
YOLO('outputs/models/adas_yolov8n_bdd100k.pt').train(
    data='data/bdd100k_yolo_adas_objects_even_copy_paste_v2_mask_1000/dataset.yaml',
    epochs=1, imgsz=1024, batch=8, device=0, workers=4,
    project='outputs/yolo_train',
    name='adas_yolov8n_bdd100k_even_copy_paste_v2_mask_1000_1024_1ep',
    exist_ok=True, plots=False, save_period=1,
)
"
```

ultralytics は `runs/detect/` を prefix するので、実際の best.pt は:

```text
runs/detect/outputs/yolo_train/adas_yolov8n_bdd100k_even_copy_paste_v2_mask_1000_1024_1ep/weights/best.pt
-> outputs/models/adas_yolov8n_bdd100k_even_copy_paste_v2_mask_1000_1024_1ep.pt (copy)
```

### 先に見ておくとよいファイル

- [PLAN.md](adas-perception/PLAN.md)
- [README.md](adas-perception/README.md)
- [scripts/export_bdd100k_yolo.py](adas-perception/scripts/export_bdd100k_yolo.py)
- [scripts/analyze_bdd100k_errors.py](adas-perception/scripts/analyze_bdd100k_errors.py)
- [scripts/visualize_bdd100k_errors.py](adas-perception/scripts/visualize_bdd100k_errors.py)
- [outputs/bdd100k_yolo_finetuned_all_tuned_split_img1024_kind_tuned_report_odd_5000_grouped_size.json](adas-perception/outputs/bdd100k_yolo_finetuned_all_tuned_split_img1024_kind_tuned_report_odd_5000_grouped_size.json)
- [outputs/bdd100k_yolo_img1024_kind_tuned_report_odd_5000_errors_all_psl_grouped_analysis.md](adas-perception/outputs/bdd100k_yolo_img1024_kind_tuned_report_odd_5000_errors_all_psl_grouped_analysis.md)

### 引き継ぎ時の注意

- このrepoは未コミットの作業ツリー前提なので、`git diff` ではなくファイル実体を読むこと。
- 公式train splitが入ったら、validation mirror augmentationより official train 再学習を優先すること。
- 現時点では `night pedestrian` と `rainy traffic light` を改善仮説の中心に置くこと。

## 目的

v0の目的は「ADASっぽい認識が1コマンドで見える」ことだった。現在はそこから一段進めて、公開データで定量評価しながら、車両、歩行者、標識、信号候補の認識性能を改善する段階に入っている。

短期目標は、BDD100Kの公開データで現行ベストを明確に超えること。現行ベストは validation mirror の odd 5,000 frames report split における `macro F1=0.6355` である。これは本リポジトリ内の評価定義であり、BDD100K公式ランキングや商用ADAS性能を意味しない。

中期目標は、公式BDD100K train/val splitを使って、評価条件をよりまともにし、YOLOv8n以外の軽量モデルも含めて速度と精度のParetoを作ること。ここでの「勝つ」は、公開データと再現コマンドに基づく比較で勝つという意味に限定する。

## 明示的な非目標

- safety-critical や production-ready であるかのような表現はしない。
- 特定企業の固有実装を模倣しない。
- 自動運転スタック、経路計画、車両制御は作らない。
- v0/v0.xでは巨大なフレームワーク化をしない。
- 実験結果が弱い設定を、見栄えのためにデフォルト扱いしない。

## 現状の実装範囲

実装済み:

- OpenCVベースの車線候補検出
- TorchVision COCOモデルによる車両/歩行者検出
- Ultralytics YOLOバックエンド
- BDD100K fine-tuned YOLOモデルによる車両/歩行者/標識/信号候補の単一モデル検出
- 色/形状ベースの標識候補検出
- 色/形状ベースの信号候補検出
- 画像デモ
- 動画デモ
- JSON出力
- JSON集計
- BDD100K評価
- BDD100K評価比較
- TP/FP/FNエラーサンプル保存
- サイズ別メトリクス
- threshold sweep
- 保存済み予測JSONからのcached threshold sweep
- BDD100K YOLO形式export
- official train/val split用export引数
- validation mirror even/odd split export
- hard-frame/small-object oversampling export
- object crop export
- copy-paste export
- feather-blended copy-paste export

主要ファイル:

- `adas_perception/pipeline.py`
- `adas_perception/detectors/objects.py`
- `adas_perception/detectors/lane.py`
- `adas_perception/visualization.py`
- `scripts/demo_image.py`
- `scripts/demo_video.py`
- `scripts/evaluate_bdd100k.py`
- `scripts/export_bdd100k_yolo.py`
- `scripts/sweep_bdd100k_cached_predictions.py`
- `scripts/compare_evaluations.py`

## 現在の推奨設定

精度優先のBDD100K実験設定:

```text
configs/bdd100k_yolo_finetuned_all_tuned_split_img1024_kind_tuned_tta_tuned_tiny.yaml
```

高速 demo 用設定 (TTA なし):

```text
configs/bdd100k_yolo_finetuned_all_tuned_split_img1024_kind_tuned.yaml
```

モデル:

```text
outputs/models/adas_yolov8n_bdd100k.pt
```

report split:

```text
BDD100K validation mirror odd-index 5,000 frames
frame_stride=2
frame_offset=1
```

現行ベスト (精度優先、TTA tuned + tiny override):

```text
macro F1 = 0.6389
FPS      = 24.71
```

TTA tuned (tiny override なし、参照値):

```text
macro F1 = 0.6385
FPS      = 25.69
```

前の best (高速 demo 基準):

```text
macro F1 = 0.6355
FPS      = 29.15
```

クラス別F1:

```text
pedestrian    0.5940
traffic_light 0.5830
traffic_sign  0.6127
vehicle       0.7523
```

サイズ別macro F1:

```text
tiny   0.4505
small  0.6842
medium 0.7810
large  0.7167
```

この設定を当面の基準線にする。新しい実験はまずこの設定をreport splitで超えるかを見る。

## 現在のデータ状況

ローカルで確認済み:

```text
data/bdd100k/images/100k/val
data/bdd100k/labels/det_20/det_val.json
```

未確認または未配置 (最大のデータ側ブロッカー):

```text
data/bdd100k/images/100k/train
data/bdd100k/labels/det_20/det_train.json
```

exporterはすでに official train/val を分けて受け取れる。

2026-04-24時点のローカル確認:

```text
train images dir   missing
train labels file  missing
val images dir     present
val labels file    present
val frames         10,000
val labels         186,033
```

train データを配置するには [BDD100K公式サイト](https://bdd-data.berkeley.edu/) (要登録) から
`bdd100k_images_100k.zip` と `bdd100k_det_20_labels_trainval.zip` を取得し、
以下のパスに展開する:

```text
data/bdd100k/images/100k/train/   # 約7GB / 70,000枚
data/bdd100k/labels/det_20/det_train.json
```

配置後、`scripts/run_bdd100k_official_train.sh` を実行すると以下を一括で実行する:

```text
1. check_bdd100k.py で train + val のパスとラベル整合性確認
2. export_bdd100k_yolo.py で official train + val を YOLO 形式へ export
3. adas_yolov8n_bdd100k.pt から 1024px / N epoch 追加学習 (default 10, 引数で変更可)
4. odd 5000 report split で untuned 評価
5. even 1000 tune split で cache low predictions 保存
6. cached threshold sweep (60 combos)
7. tuned config を生成し odd 5000 で評価
8. 現行ベスト (TTA tuned + tiny) と previous best (no-TTA) との比較表生成
```

GPU 想定で 10 epoch 全工程は約 3〜4 時間。
script は train データが無いと step 1 で fail fast する。

## 評価ルール

短期の採用判断は以下で行う。

```text
primary metric:   object macro F1
secondary metric: class-wise F1
secondary metric: tiny/small/medium/large macro F1
secondary metric: FPS
report split:     odd-index 5,000 frames
tune split:       even-index 1,000 frames
```

採用基準:

- report splitのmacro F1が現行ベスト `0.6355` を上回る
- できれば `+0.005` 以上の改善がある
- 1クラスだけ改善して他3クラスを大きく落とす設定は採用しない
- FPSが大きく落ちる場合は、精度優先設定と高速設定を分ける
- tune splitだけで良い結果が出ても、report splitで再現しない場合は採用しない

## これまでの主要実験

### COCO YOLO baseline

BDD100K validation mirror先頭500枚:

```text
macro F1 = 0.2944
FPS      = 31.36
```

traffic signがほぼ拾えない。BDD100Kの標識/信号にはCOCO事前学習だけでは足りない。

### Sign/light fine-tuned ensemble

BDD100K validation mirror全10,000枚:

```text
COCO YOLO only              macro F1 = 0.2860, FPS = 48.38
YOLO + sign/light finetune  macro F1 = 0.5404, FPS = 25.81
```

traffic sign/lightは大幅に改善したが、構成が2モデルになり速度も落ちる。v0の方向としては、最終的に単一モデルへ寄せたい。

### Single YOLOv8n BDD100K fine-tune

BDD100K validation mirror全10,000枚:

```text
macro F1 = 0.5689
FPS      = 31.35
```

threshold tuning後:

```text
macro F1 = 0.5700
FPS      = 38.05
```

640px段階では単一YOLOが扱いやすいが、小物体が弱い。

### Split tuning

偶数index 1,000枚でthresholdを選び、奇数index 5,000枚で報告。

```text
untuned      macro F1 = 0.5701
existing     macro F1 = 0.5715
split-tuned  macro F1 = 0.5720
```

以後、tune splitとreport splitを分ける方針にした。

### Resolution sweep

同じ重みと近いthresholdで入力解像度だけ上げた。

```text
640px  split-tuned    macro F1 = 0.572
960px                 macro F1 = 0.631
1024px                macro F1 = 0.633
1024px kind-tuned     macro F1 = 0.6355
```

1024pxが現時点の最良。1280pxはspot評価で伸びなかったため、フル評価は保留。

判断:

- 小物体対策として、まず入力解像度の効果が大きい
- 1024pxは速度と精度のバランスが良い
- 以後の訓練系実験は1024px基準で見る

### Even split 1024px 1epoch retraining

既存fine-tuned weightから、BDD100K validation mirror偶数index 5,000枚で1epoch追加学習。

```text
macro F1 = 0.6309
FPS      = 25.83
```

traffic lightだけ微増したが、総合では1024px kind-tunedを下回った。

判断:

- val mirrorでの短い追加学習は、単純には現行ベストを超えない
- 公式train splitでの再学習が必要

### Class-aware oversampling

pedestrian、traffic sign、traffic lightを含むframeを最大2回まで重複exportして1epoch追加学習。

```text
untuned macro F1 = 0.6332
tuned   macro F1 = 0.6324
```

traffic light F1は `0.5830 -> 0.5888` に改善したが、macro F1では届かない。

エラー分析:

```text
pedestrian TP 3832 -> 3880
pedestrian FN 3110 -> 3062
pedestrian FP 2128 -> 2390

traffic light FP 5283 -> 3728
traffic light TP 7517 -> 6975

traffic sign FP 5633 -> 4840
traffic sign TP 10061 -> 9621
```

判断:

- frame repeatは校正を変える効果が強い
- FPを下げる代わりにtiny/smallのTPも落ちる
- class imbalance対策は単純な重複よりloss/サンプル設計を変えるべき

### Object crop training

小さなpedestrian/sign/light cropを12,012枚追加して1epoch追加学習。

```text
untuned macro F1 = 0.5938
tuned   macro F1 = 0.5943
```

判断:

- crop-onlyは全体画像の文脈を壊す
- 大量のcrop追加はモデルを不自然な分布へ寄せる
- 採用しない

### Full-frame copy-paste

小さなpedestrian/sign/light bboxを別のfull-frame画像に貼り付け、2,500枚追加。

```text
untuned macro F1 = 0.6223
tuned   macro F1 = 0.6298
```

object cropよりは大きく改善。ただし現行ベストには届かない。

判断:

- full-frame文脈に戻す方向はcrop-onlyより良い
- ただしbbox矩形貼り付けは分布ギャップが残る
- threshold tuningでかなり回復するが、伸びしろは限定的

### Light copy-paste

追加copy-pasteを約1,000枚に抑えた。

```text
untuned macro F1 = 0.6295
tuned   macro F1 = 0.6280
```

traffic light F1だけは現行ベストを少し上回った。

```text
traffic_light F1 = 0.5881
```

判断:

- copy-paste枚数を増やせば良いわけではない
- 少なめのほうが分布破壊は小さい
- ただし総合採用には足りない

### Feather copy-paste

bbox矩形patchの境界をfeather blendして2,500枚追加。

```text
untuned macro F1 = 0.6234
tuned   macro F1 = 0.6242
```

判断:

- 単純なedge blendingだけでは分布ギャップは埋まらない
- bbox矩形で背景ごと貼ること自体が問題
- 次はsegmentation maskに近い貼り付けや品質フィルタが必要

### Copy-paste v2 (grabcut mask + source filter, 1000 paste images)

source側に最低面積/最低bbox辺/最大aspect ratio制約を入れ、GrabCut foreground maskで
ほぼ物体輪郭に近い形で貼り付けた1,000枚追加。train=6,000 images, labels=114,024。

```text
config (tuned): configs/bdd100k_yolo_finetuned_all_even_copy_paste_v2_mask_1000_1024_1ep_tuned.yaml
report: outputs/bdd100k_yolo_finetuned_all_even_copy_paste_v2_mask_1000_1024_1ep_tuned_report_odd_5000_size.json
untuned macro F1 = 0.6263
tuned   macro F1 = 0.6310
baseline macro F1 = 0.6355
```

クラス別 F1 (tuned, vs current best delta):

```text
pedestrian    0.590  -0.004
traffic_light 0.589  +0.006
traffic_sign  0.598  -0.015
vehicle       0.747  -0.005
```

tune split (even 1000) best combo:

```text
pedestrian=0.25 traffic_light=0.25 traffic_sign=0.25 vehicle=0.25
tune split macro F1 = 0.6419
```

判断:

- grabcut mask + source quality filterでも current best には届かない
- 改善クラスは traffic_light のみで、traffic_sign が -0.015 と目立って下がる
- bbox矩形からmask寄りに変えても、validation mirrorでの短い追加学習という枠から出ていない
- このレシピはこれ以上深追いしない

### Copy-paste v2 traffic_light only (1000 paste images)

`--copy-paste-classes "traffic light"` に限定して pedestrian / traffic_sign の混在を
外した 1,000 枚追加。v2 mixed で唯一改善した traffic_light だけに絞れば、regression を
抑えて macro を超える可能性を検証した。

```text
config (tuned): configs/bdd100k_yolo_finetuned_all_even_copy_paste_v2_tl_1000_1024_1ep_tuned.yaml
report: outputs/bdd100k_yolo_finetuned_all_even_copy_paste_v2_tl_1000_1024_1ep_tuned_report_odd_5000_size.json
untuned macro F1 = 0.6285
tuned   macro F1 = 0.6325
baseline macro F1 = 0.6355
```

クラス別 F1 (tuned, vs current best delta):

```text
pedestrian    0.588  -0.006
traffic_light 0.5915 +0.009   <- 全実験中最大
traffic_sign  0.602  -0.011
vehicle       0.748  -0.004
```

tune split (even 1000) best combo:

```text
pedestrian=0.30 traffic_light=0.25 traffic_sign=0.25 vehicle=0.25
tune split macro F1 = 0.6401
```

判断:

- traffic_light F1 は全実験中 best、mixed v2 tuned より macro も +0.0015
- しかし traffic_sign が -0.011、pedestrian が -0.006 と下がる
- copy-paste source から traffic_sign を外しても追加学習自体で traffic_sign が下がる
- つまり validation mirror 上の短い追加学習は、どのクラスを copy-paste 対象にしても
  そのクラス以外が下がる構造がある
- traffic_light 特化 config としてなら採用可能だが、macro ベースの current best は更新しない
- このレシピはこれ以上深追いしない

### Tile inference

1024px kind-tunedに左右2tileのtiny bbox補助検出を追加してspot評価。

```text
tileあり spot macro F1 = 0.641
tileなし spot macro F1 = 0.649
```

tiny/smallは微増したが、FP増加でmacro F1は落ちた。

判断:

- 現状のtile inferenceは採用しない
- tileを使うなら検出候補の統合、NMS、サイズ制約、領域制約をもっと厳しくする

### Size-aware threshold

tiny bboxだけthresholdを下げるspot評価。

```text
size-aware spot macro F1 = 0.641
baseline spot macro F1   = 0.649
```

判断:

- tiny recallは一部上がるがFP増加が勝つ
- 単純なしきい値低下は採用しない

### TTA (Ultralytics augment=True) + threshold re-tune

previous best (`adas_yolov8n_bdd100k.pt`) の重みをそのまま使い、inference 時に
`ultralytics.YOLO.predict(augment=True)` で水平反転 + multi-scale TTA を有効化。
TTA は recall を上げ precision を下げる傾向があるので、even 1000 tune split で
per-kind threshold を再調整し直す必要がある。再調整なしでは macro F1 が -0.013 下がる。

実装: `adas_perception/detectors/objects.py::UltralyticsObjectDetector` に
`augment` config フラグを追加し、`model.predict(..., augment=self.augment)` へ流した。

```text
config (tuned): configs/bdd100k_yolo_finetuned_all_tuned_split_img1024_kind_tuned_tta_tuned.yaml
report: outputs/bdd100k_yolo_finetuned_all_tuned_split_img1024_kind_tuned_tta_tuned_report_odd_5000_size.json
untuned (TTA only)   macro F1 = 0.6229  FPS 18.05
tuned   (TTA + thr)  macro F1 = 0.6385  FPS 25.69
previous best        macro F1 = 0.6355  FPS 29.15
```

クラス別 F1 (tuned, vs previous best delta):

```text
pedestrian    0.601  +0.007
traffic_light 0.585  +0.002
traffic_sign  0.613  +0.000
vehicle       0.755  +0.003
```

tune split (even 1000) best combo:

```text
pedestrian=0.35 traffic_light=0.30 traffic_sign=0.35 vehicle=0.35
tune split macro F1 = 0.6468
```

判断:

- 全クラスで regression なしで macro +0.003、初めて previous best を純粋に超えた
- 追加学習も追加データも不要で、inference-side の threshold 再調整のみで達成
- FPS は ~12% 低下するが、精度優先 config としては十分許容範囲
- 新しい current best としてこの config を採用する
- 高速 demo 用には no-TTA current best を維持する

follow-up で試したが効かなかったもの:

- tl-only copy-paste weights + TTA + threshold re-tune
  - tune split best macro F1 = 0.6363 (TTA current_best tune 0.6468 より -0.011)
  - report odd 5000 macro F1 = 0.6319 (TTA tuned 0.6385 より -0.007)
  - 追加学習による base model の -0.005 劣化は TTA でも回復しない
- fine threshold grid search (pedestrian / vehicle / traffic_sign 0.30-0.425、
  traffic_light 0.25-0.35 を 0.025 刻みで振った 900 combos)
  - tune split best macro F1 = 0.6478 (coarse 0.6468 より +0.001)
  - 主な差は traffic_light 0.30 → 0.275
  - report odd 5000 macro F1 = 0.6384 (TTA tuned v1 0.6385 と実質同値)
  - tune split の +0.001 は report に transfer せず、overfit。v1 を保持する

follow-up で小さく効いたもの:

- TTA tuned + tiny size-bucket threshold override (192 combos)
  - sweep で tiny に対してのみ kind threshold を下げる組み合わせを探索
  - best: tiny に traffic_light=0.25 (kind 0.30)、traffic_sign=0.30 (kind 0.35)
    を設定、pedestrian / vehicle の tiny は通常 kind threshold のまま
  - tune split macro F1 = 0.6486 (TTA tuned 0.6468 +0.0018)
  - report odd 5000 macro F1 = 0.6389 (TTA tuned 0.6385 +0.0004)
  - 4 クラス全てで previous best に対して正 delta (TTA tuned だけでは
    traffic_sign が ±0 だったのが +0.002 に改善)
  - 新 current best として採用

follow-up で効かなかったもの (継続):

- small size-bucket threshold override (54 combos, tiny は best 固定)
  - tune split best macro F1 = 0.6487 (tiny-only 0.6486 より +0.0001)
  - 主な差は vehicle small を 0.35 → 0.30 に下げる
  - report odd 5000 macro F1 = 0.6387 (tiny-only 0.6389 より -0.0002)
  - vehicle F1 が -0.0008 で微減し、overfit 傾向
  - 採用しない。tiny-only の single-config best を維持する

大きく効いたもの:

- WBF (Weighted Box Fusion) for non-TTA + TTA predictions
  - no-TTA と TTA の低 threshold predictions を `ensemble_boxes.weighted_boxes_fusion`
    で融合 (iou_thr=0.55, weights=[1.0, 1.0], BDD100K 720x1280 normalize)
  - 融合後に per-kind threshold sweep (256 combos)
  - best thresholds: pedestrian=0.25, traffic_light=0.25, traffic_sign=0.30, vehicle=0.30
  - tune split macro F1 = 0.6539 (TTA tuned + tiny tune 0.6486 より +0.0053)
  - report odd 5000 macro F1 = 0.6447 (+0.0058 vs 0.6389 tiny、+0.0092 vs 0.6355 no-TTA)
  - クラス別 delta vs previous tiny best:
    pedestrian +0.012、traffic_light +0.010、traffic_sign -0.004、vehicle +0.006
  - WBF weights [1, 1.5] / [1, 2] と iou_thr 0.50 / 0.60 の grid も試したが
    [1, 1] / 0.55 が最適。fine threshold grid はまた overfit (tune +0.001, report 0)
  - 制約: 2 回 forward pass + offline fusion が必要 (~10.5 FPS)
    単一 config pipeline ではないため、accuracy ceiling として別枠扱い

- 7-way WBF (extreme scale): + TTA@1792 (最終 ceiling)
  - imgsz=1792 TTA multi-scale は 1490-2100 range
  - 7-way fuse: iou_thr=0.55, weights=[1.0]*7
  - threshold sweep の best は pedestrian=0.25, traffic_light=0.20, traffic_sign=0.30, vehicle=0.25
    だが tune overfit 気味 (report では traffic_light -0.006)。安全のため 6-way threshold
    (pedestrian=0.25, traffic_light=0.25, traffic_sign=0.30, vehicle=0.30) を採用
  - tune split macro F1 = 0.6809 (6-way tune 0.6767 +0.0042)
  - report odd 5000 macro F1 = 0.6724 (6-way 0.6686 +0.0038、previous best 0.6355 +0.0369)
  - クラス別 delta vs 6-way:
    pedestrian +0.004、traffic_light +0.002、traffic_sign +0.006、vehicle +0.004
  - 新 accuracy ceiling として採用

- 8-way WBF (2048 TTA 追加): plateau、採用しない
  - imgsz=2048 TTA を 8th source に追加
  - 8-way fuse: iou_thr=0.55, weights=[1.0]*8
  - tune split macro F1 = 0.6798 (7-way tune 0.6809 -0.0011、noise 化)
  - report odd 5000 macro F1 = 0.6725 (7-way 0.6724 +0.0001)
  - クラス別 delta vs 7-way:
    pedestrian -0.001、traffic_light -0.002、traffic_sign +0.003、vehicle +0.001
  - plateau。extreme scale diversity は 1792 で尽きている

- 8-way WBF (class-balanced retrain TTA 追加): tune overfit、採用しない
  - 2048 の代わりに class-balanced retrain weights + TTA を 8th source に
    (class-balanced 単体は macro F1 = 0.6332、frame-level class repeat augmentation)
  - 8-way fuse: iou_thr=0.55, weights=[1.0]*8
  - tune split macro F1 = 0.6833 (7-way tune 0.6809 +0.0024、正方向に見える)
  - report odd 5000 macro F1 = 0.6717 (7-way stable threshold)、0.6708 (own-threshold tuned)
  - **tune +0.0024 → report -0.0007** の tune overfit パターン (v2-mask と同じ)
  - クラス別 delta vs 7-way: 全て -0.0003 〜 -0.0010 と微減

- 7-way WBF iou_thr sweep: iou=0.45 が最適
  - default iou=0.55 で 7-way tune 0.6806 / report 0.6724
  - iou_thr grid (0.35-0.70 を 0.025 刻み) on tune:
    0.35=0.6819, 0.40=0.6851, 0.425=0.6853, 0.45=0.6854, 0.50=0.6843,
    0.55=0.6806, 0.60=0.6771, 0.65=0.6709, 0.70=0.6622
  - peak は iou=0.45 で tune 0.6854 (default 0.55 より +0.0048)
  - report odd 5000 macro F1 = 0.6747 (default iou=0.55 の 0.6724 より +0.0023)
  - iou=0.50 も同水準 (report 0.6746) だが iou=0.45 が tune/report 両方でわずかに上
  - クラス別 delta vs iou=0.55:
    pedestrian -0.001、traffic_light +0.007、traffic_sign +0.004、vehicle -0.001
  - traffic_light/traffic_sign は fusion clustering が緩めの方が良い (小物体で
    近接するが別物体の bbox をより区別できる可能性)
  - **新 accuracy ceiling として採用**

- 8-way WBF (yolo11n retrain TTA 追加): 採用しない
  - 別 model family (yolo11n, architectural diversity) を 8th source に
  - yolo11n.pt を COCO 事前学習から `data/bdd100k_yolo_adas_objects_even` で
    1024px / 1 epoch fine-tune → TTA cache → 8-way fuse
  - yolo11n 個別 mAP50: pedestrian 0.45, traffic_sign 0.40, traffic_light 0.31 (yolov8n より弱め)
  - tune split macro F1 = 0.6863 (7-way per-kind 0.6868 -0.0005)
  - report odd 5000 macro F1 = 0.6733 (7-way per-kind 0.6753 -0.0020)
  - クラス別 delta vs 7-way: traffic_sign -0.005 が主因、他は ±0.002 以内
  - 採用しない。model family diversity も 7-way saturated ensemble には寄与しない
  - **8th source 試行はこれで 4 回目の失敗** (2048 / class-balanced / ped-only / yolo11n)
  - 7-way per-kind iou (0.6753) が真の最終 ceiling と確定

- 8-way WBF (CLAHE preprocessing TTA 追加): 採用しない
  - Autoware/OpenPilot 調査からの night/rain visibility 改善狙い
  - BGR → LAB → CLAHE on L → BGR で brightness enhancement してから TTA inference
  - tune split macro F1 = 0.6847 (7-way per-kind 0.6868 -0.0021)
  - report odd 5000 macro F1 = 0.6742 (7-way per-kind 0.6753 -0.0011)
  - クラス別 delta vs 7-way: traffic_light -0.002, traffic_sign -0.003, ped/v ±0
  - CLAHE 予測は既存 TTA sources と overlap が大きく、night/rain 特有の signal を十分加えられず
  - **8th source 試行 5 回連続失敗**: scale/weight/architecture/preprocessing いずれも 7-way 以上にならない
  - `adas_perception/detectors/objects.py` に `preprocess: clahe` オプション追加 (将来的に reusable)

- 8-way WBF (pedestrian-only retrain TTA 追加): 採用しない
  - tl-only の成功パターンを複製: export → train → cache → fuse
  - `scripts/export_bdd100k_yolo.py --copy-paste-classes pedestrian` で新 retrain
  - weights: outputs/models/adas_yolov8n_bdd100k_even_copy_paste_v2_ped_1000_1024_1ep.pt
  - tune split macro F1 = 0.6797 (7-way tune 0.6809 -0.0012)
  - report odd 5000 macro F1 = 0.6709 (7-way 0.6724 -0.0015)
  - クラス別 delta vs 7-way: traffic_light -0.001、traffic_sign -0.004、
    pedestrian -0.001、vehicle -0.001
  - 採用しない

- 教訓 (確定):
  - **retrain diversity は ensemble 初期段階 (3rd source) では効く** が、
    scale diversity で saturated な 8-way ensemble には効かない (redundant)
  - tl-only (3rd source) は +0.0042 効いたが、class-balanced や ped-only を
    8th source として追加しても効果なし
  - scale diversity は 1792 で plateau、retrain diversity は 8th 位置で plateau
  - **7-way (0.6724) が真の最終 accuracy ceiling**

- 7-way を最終 ceiling として確定

- 6-way WBF (extreme scale diversity): + TTA@1536
  - imgsz=1536 TTA multi-scale は 1275-1800 range で、1280 TTA (1060-1500) の
    上限を超える新領域をカバー
  - 6-way fuse: iou_thr=0.55, weights=[1.0]*6
  - threshold sweep best: pedestrian=0.25, traffic_light=0.25, traffic_sign=0.30, vehicle=0.30
    (5-way と同じ)
  - tune split macro F1 = 0.6767 (5-way tune 0.6731 +0.0036)
  - report odd 5000 macro F1 = 0.6686 (5-way 0.6627 +0.0059、previous best 0.6355 +0.0331)
  - クラス別 delta vs 5-way:
    pedestrian +0.008、traffic_light +0.004、traffic_sign +0.008、vehicle +0.004
  - 960 が与えた +0.0025 より大きい gain
  - 教訓: scale diversity では extreme (1024 からの距離が大きい) ほど効く
  - 新 accuracy ceiling として採用

- 5-way WBF (scale diversity 拡張): + TTA@960
  - 5th source として current best weights を imgsz=960 で走らせた TTA を追加
    (960 TTA multi-scale は 800-1120 range で、1024 TTA と overlap するが更に小さい scale 帯もカバー)
  - 5-way fuse: iou_thr=0.55, weights=[1.0, 1.0, 1.0, 1.0, 1.0]
  - threshold sweep で best: pedestrian=0.25, traffic_light=0.25, traffic_sign=0.30, vehicle=0.30
    (4-way と同じ threshold)
  - tune split macro F1 = 0.6731 (4-way tune 0.6710 +0.0021)
  - report odd 5000 macro F1 = 0.6627 (4-way 0.6602 +0.0025、previous best 0.6355 +0.0272)
  - クラス別 delta vs 4-way:
    pedestrian +0.003、traffic_light +0.006、traffic_sign ±0、vehicle +0.001
  - 全クラス非負の小さな gain。diminishing returns が始まっている
  - 新 accuracy ceiling として採用だが、4-way との差は marginal

- 4-way WBF (scale diversity): no-TTA + TTA@1024 + tl-only TTA + TTA@1280
  - 4th source は **current best weights を imgsz=1280 で走らせた TTA** (weight diversity
    ではなく scale diversity を狙った)
  - TTA@1024 の multi-scale は 850-1200 range、TTA@1280 は 1060-1500 range で
    新しい scale 域をカバー
  - 4-way fuse: iou_thr=0.55, weights=[1.0, 1.0, 1.0, 1.0]
  - threshold sweep (108 combos) で best: pedestrian=0.25, traffic_light=0.25,
    traffic_sign=0.30, vehicle=0.30
  - tune split macro F1 = 0.6710 (3-way tune 0.6587 +0.0123)
  - report odd 5000 macro F1 = 0.6602 (3-way 0.6489 +0.0113、previous best 0.6355 +0.0247)
  - クラス別 delta vs previous best:
    pedestrian +0.038、traffic_light +0.030、traffic_sign +0.013、vehicle +0.019
  - 新 accuracy ceiling として採用
  - 教訓: 同じ weights でも scale を変えると genuinely complementary な予測が得られる

- 3-way WBF: no-TTA + TTA + tl-only retrain + TTA
  - 3rd source として tl-only retrain weights の TTA predictions を追加
    (tl-only 単体は macro F1 = 0.6325 で baseline より弱いが、ensemble diversity に寄与)
  - 3-way fuse: iou_thr=0.55, weights=[1.0, 1.0, 1.0], 720x1280 normalize
  - threshold sweep (256 combos) 再実行
  - best thresholds: pedestrian=0.30, traffic_light=0.25, traffic_sign=0.25, vehicle=0.30
  - tune split macro F1 = 0.6587 (2-way tune 0.6539 より +0.0048)
  - report odd 5000 macro F1 = 0.6489 (+0.0042 vs 2-way、+0.0134 vs previous best 0.6355)
  - クラス別 delta vs previous best 0.6355:
    pedestrian +0.023、traffic_light +0.018、traffic_sign +0.001、vehicle +0.012
  - 初めて全 4 クラスが previous best を超え、traffic_light は初 0.60 超え
  - 制約: 3 回 forward pass (~7.8 FPS)。新 accuracy ceiling として採用
  - weight grid (tl-only weight 0.5/0.75/1.0/1.25/1.5) × iou (0.50/0.55/0.60) も試したが
    [1,1,1] / iou 0.55 が最適

follow-up で効かなかったもの:

- 4-way WBF: no-TTA + TTA + tl-only TTA + v2-mask TTA
  - 4th source として v2-mask-1000 retrain weights の TTA を追加
    (v2-mask 単体は 0.6310、ped/tl/ts の 3 クラスに copy-paste)
  - tune split macro F1 = 0.6567 (3-way tune 0.6587 より -0.0020)
  - v2-mask weight を 0.3/0.5/0.75 に下げる grid も試したが最高 0.6572 で 3-way に届かない
  - report odd 5000 macro F1 = 0.6431 (3-way 0.6489 より -0.0058)
  - 全クラスの F1 が 3-way より下がり、v2-mask は diversity ではなくノイズを加える
  - 採用しない。3-way を ceiling として維持する
  - 教訓: 弱い retrain weights でも fusion source としては機能するが、
    全てが機能するわけではない。tl-only (単一クラス集中) は効き、v2-mask
    (3 クラス分散) は効かなかった。source 選定は careful に。

## 現在の結論

現時点で一番効いた改善は、訓練データの小手先augmentationではなく入力解像度1024px化とkind別threshold調整である。

現在の勝ち筋:

1. 公式BDD100K train splitに移る
2. 1024px基準で学習をやり直す
3. YOLOv8nだけでなく、少し大きい軽量モデルも比較する
4. 小物体augmentationはbbox矩形ではなくmask寄りにする
5. scene/weather/timeofday別にthresholdやerror profileを見る
6. visualization/demo品質を維持しながら、評価で強い設定をREADMEの推奨にする

## 次の作業計画

### Phase 0: 再現性の整備

目的:

評価結果と実験条件をあとで再現できるようにする。

作業:

- `PLAN.md`をこのまま更新し続ける
- 重要な評価JSONと比較Markdownの一覧をREADMEから辿れるようにする
- training command、dataset export command、threshold sweep commandを各実験の近くに残す
- `outputs/`に増えた実験名の命名規則を揃える
- 使わないsmoke出力や一時出力は削除する

完了条件:

- 新しい実験を始める前に、どのconfig、どのweight、どのreport splitを使うか迷わない
- README、ROADMAP、PLANの役割が分かれている

### Phase 1: BDD100K official train/val の導入

目的:

validation mirror実験から脱出し、よりまともな公開データ評価へ移る。

必要なローカル配置:

```text
data/bdd100k/images/100k/train
data/bdd100k/images/100k/val
data/bdd100k/labels/det_20/det_train.json
data/bdd100k/labels/det_20/det_val.json
```

チェック:

```bash
python scripts/check_bdd100k.py \
  --images-root data/bdd100k/images/100k/val \
  --labels data/bdd100k/labels/det_20/det_val.json
```

official train/val export:

```bash
python scripts/export_bdd100k_yolo.py \
  --images-root data/bdd100k/images/100k/train \
  --labels data/bdd100k/labels/det_20/det_train.json \
  --val-images-root data/bdd100k/images/100k/val \
  --val-labels data/bdd100k/labels/det_20/det_val.json \
  --output-dir data/bdd100k_yolo_adas_objects_train_val \
  --classes car truck bus bicycle motorcycle train pedestrian rider "traffic sign" "traffic light" \
  --clear-output
```

完了条件:

- `data/bdd100k_yolo_adas_objects_train_val/dataset.yaml` ができる
- train/valの画像数とラベル数がexport statsに記録される
- val splitにtrain由来のcopy/crop画像が混ざらない

### Phase 2: 公式train splitでYOLOv8nを1024px再学習

目的:

現在のvalidation mirror追加学習ではなく、公式train splitから1024pxモデルを作る。

初回training:

```bash
yolo detect train \
  model=outputs/models/adas_yolov8n_bdd100k.pt \
  data=data/bdd100k_yolo_adas_objects_train_val/dataset.yaml \
  epochs=10 \
  imgsz=1024 \
  batch=8 \
  device=0 \
  workers=4 \
  project=outputs/yolo_train \
  name=adas_yolov8n_bdd100k_official_train_1024_10ep \
  exist_ok=True \
  plots=False \
  save_period=1
```

評価:

```bash
python scripts/evaluate_bdd100k.py \
  --images-root data/bdd100k/images/100k/val \
  --labels data/bdd100k/labels/det_20/det_val.json \
  --config configs/<new_config>.yaml \
  --device cuda \
  --group-by-size \
  --progress-every 1000 \
  --output outputs/<new_report>.json
```

threshold tuning:

1. 低threshold configを作る
2. `--save-predictions` で予測JSONを保存
3. `scripts/sweep_bdd100k_cached_predictions.py` でkind別threshold sweep
4. val全体またはval内のtune/report分割で再評価

採用判断:

- validation mirrorではなくofficial valでのmacro F1を基準にする
- ただし既存のodd 5,000 report splitとの比較も補助的に残す

### Phase 3: モデルサイズ比較

目的:

YOLOv8nだけで限界を見ず、速度/精度のParetoを作る。

候補:

```text
yolov8n
yolov8s
yolo11n
yolo11s
```

優先:

1. `yolov8s` または `yolo11n` を1024pxで比較
2. 同じofficial train/val exportを使う
3. threshold sweepはcachedで行う
4. FPSを同じGPU/同じ画像数で測る

採用基準:

- 精度優先設定として `+0.01` 以上のmacro F1改善があるなら、FPS低下を許容
- デモ標準設定は軽量モデルを維持
- 精度優先と高速デモを分ける

### Phase 4: 小物体augmentationを作り直す

目的:

pedestrian、traffic sign、traffic lightのtiny/smallを伸ばす。

これまでの失敗:

- object cropは文脈を壊した
- hard/small重複はFP/FNのバランスを崩した
- bbox矩形copy-pasteは現行ベストを超えない
- feather blendだけでは足りない

次の方針:

- 貼り付け元bboxの品質フィルタを強くする
- 極端に小さい/ぼけた/遮蔽が強いsourceを除外する
- bbox全体ではなく対象物輪郭に近いmaskを作る
- 可能ならBDD100K以外の公開segmentation系データでsource maskを作る
- 貼り付け先は道路文脈に限定する
- paste後のoverlap制約を厳しくする
- copy-paste枚数を1,000以下から探索する

具体案:

```text
copy-paste v2
- source bbox area range: 0.0001 to 0.0025
- source min box size: 8px or 12px
- source aspect ratio clamp
- target scene balance
- paste scale jitter smaller
- max overlap lower
- optional color matching
- optional edge alpha from GrabCut or simple foreground mask
```

最初にやるべき実装:

- `--copy-paste-source-min-area`
- `--copy-paste-area-threshold` をsource max areaとして使う
- `--copy-paste-source-min-box-size`
- `--copy-paste-source-max-aspect-ratio`
- `--copy-paste-source-min-confidence` はannotationだけでは使えないので保留
- `--copy-paste-mask box|grabcut`
- `--copy-paste-max-images 500/1000/1500` の比較

2026-04-23実装:

```text
scripts/export_bdd100k_yolo.py
- --copy-paste-source-min-area
- --copy-paste-source-min-box-size
- --copy-paste-source-max-aspect-ratio
- --copy-paste-mask none|box|grabcut
```

互換性:

- defaultは従来どおり `--copy-paste-mask none`
- source min areaはdefault `0.0`
- source min box sizeはdefault `1`
- source max aspect ratioはdefault `0.0` で無効

smoke検証:

```text
outputs/smoke_bdd100k_copy_paste_v2_yolo
copy_paste_images=8
checked_labels=683
bad_labels=0
mask=grabcut
blend=feather
```

次の実験候補:

```bash
python scripts/export_bdd100k_yolo.py \
  --images-root data/bdd100k/images/100k/val \
  --labels data/bdd100k/labels/det_20/det_val.json \
  --output-dir data/bdd100k_yolo_adas_objects_even_copy_paste_v2_mask_1000 \
  --classes car truck bus bicycle motorcycle train pedestrian rider "traffic sign" "traffic light" \
  --split-mode alternate \
  --frame-stride 2 \
  --train-frame-offset 0 \
  --val-frame-offset 1 \
  --copy-paste-classes pedestrian "traffic sign" "traffic light" \
  --copy-paste-area-threshold 0.0025 \
  --copy-paste-source-min-area 0.00002 \
  --copy-paste-source-min-box-size 8 \
  --copy-paste-source-max-aspect-ratio 8.0 \
  --copy-paste-max-images 1000 \
  --copy-paste-objects-per-image 1 \
  --copy-paste-context-padding 0.20 \
  --copy-paste-scale-min 0.9 \
  --copy-paste-scale-max 1.1 \
  --copy-paste-max-overlap 0.05 \
  --copy-paste-mask grabcut \
  --copy-paste-blend feather \
  --copy-paste-feather-ratio 0.08 \
  --copy-paste-seed 17 \
  --clear-output
```

採用基準:

- current best `0.6355` を超える
- tiny/small macro F1が上がり、medium/largeを大きく落とさない
- traffic_lightだけ改善する場合は、traffic_light特化設定として分離する

### Phase 5: 条件別評価を増やす

目的:

macro F1の単一値だけでなく、どの環境で弱いかを見る。

BDD100K attributes:

```text
weather
timeofday
scene
```

すでに `evaluate_bdd100k.py --group-by weather timeofday scene` が使える。

やること:

- current bestを条件別評価する
- class-balancedを条件別評価する
- copy-paste/light/featherを条件別評価する
- night/rainy/city streetなどで弱点を見る
- scene別threshold tuningの価値を判断する

候補コマンド:

```bash
python scripts/evaluate_bdd100k.py \
  --images-root data/bdd100k/images/100k/val \
  --labels data/bdd100k/labels/det_20/det_val.json \
  --config configs/bdd100k_yolo_finetuned_all_tuned_split_img1024_kind_tuned.yaml \
  --device cuda \
  --frame-stride 2 \
  --frame-offset 1 \
  --group-by weather timeofday scene \
  --group-by-size \
  --progress-every 1000 \
  --output outputs/bdd100k_yolo_img1024_kind_tuned_report_odd_5000_grouped_size.json
```

採用判断:

- 条件別に特に弱いところがあれば、augmentationやthresholdをそこに寄せる
- 条件別thresholdは複雑化するため、全体macro F1を明確に上げる場合だけ採用する

2026-04-23実行結果:

```text
report:
outputs/bdd100k_yolo_finetuned_all_tuned_split_img1024_kind_tuned_report_odd_5000_grouped_size.json

overall macro F1 = 0.6355
```

100枚以上あるgroupで弱いもの:

```text
scene/highway       images=1269  macro F1=0.6026  delta=-0.0329  worst=pedestrian -0.1003
timeofday/night     images=1962  macro F1=0.6170  delta=-0.0185  worst=pedestrian -0.0502
weather/rainy       images=373   macro F1=0.6147  delta=-0.0208  worst=traffic_light -0.0354
timeofday/dawn/dusk images=391   macro F1=0.6244  delta=-0.0111  worst=traffic_sign -0.0302
```

サンプル数が少ないため参考扱いにするgroup:

```text
weather/foggy       images=5
scene/gas stations  images=4
scene/parking lot   images=24
scene/tunnel        images=15
```

次のerror analysisでは、`highway` と `night` のpedestrian FN/FP、`rainy` のtraffic light FN/FPを優先して見る。

### Phase 6: Error analysisを次の実験入力にする

目的:

TP/FP/FNを見て、augmentationや後処理の仮説を作る。

やること:

- current bestの全TP/FP/FNサンプルを保存
- tiny/smallのFNを抽出
- false positiveが多いtraffic sign/lightの画像を可視化
- source object品質フィルタを設計する
- hard frameを再学習に使う場合は、重複ではなくloss/augmentationに反映する

候補コマンド:

```bash
python scripts/evaluate_bdd100k.py \
  --images-root data/bdd100k/images/100k/val \
  --labels data/bdd100k/labels/det_20/det_val.json \
  --config configs/bdd100k_yolo_finetuned_all_tuned_split_img1024_kind_tuned.yaml \
  --device cuda \
  --frame-stride 2 \
  --frame-offset 1 \
  --group-by-size \
  --save-errors outputs/bdd100k_yolo_img1024_kind_tuned_report_odd_5000_errors_all.json \
  --max-error-samples 0 \
  --output outputs/bdd100k_yolo_img1024_kind_tuned_report_odd_5000_errors_report.json
```

可視化:

```bash
python scripts/visualize_bdd100k_errors.py \
  --images-root data/bdd100k/images/100k/val \
  --errors outputs/bdd100k_yolo_img1024_kind_tuned_report_odd_5000_errors_all.json \
  --output-dir outputs/bdd100k_yolo_img1024_kind_tuned_error_samples
```

2026-04-23実行結果:

属性別エラー集計を追加し、以下を生成した。

```text
outputs/bdd100k_yolo_img1024_kind_tuned_report_odd_5000_errors_all_psl_grouped_analysis.json
outputs/bdd100k_yolo_img1024_kind_tuned_report_odd_5000_errors_all_psl_grouped_analysis.md
```

重点条件のエラー数:

```text
pedestrian FN scene=highway       147/3110  (4.7%)
pedestrian FP scene=highway        54/2128  (2.5%)
pedestrian FN timeofday=night     685/3110  (22.0%)
pedestrian FP timeofday=night     414/2128  (19.5%)
traffic_light FN weather=rainy    502/5471  (9.2%)
traffic_light FP weather=rainy    519/5283  (9.8%)
traffic_sign FN timeofday=dawn/dusk 546/7089 (7.7%)
traffic_sign FP timeofday=dawn/dusk 446/5633 (7.9%)
```

重点ギャラリー:

```text
outputs/bdd100k_yolo_img1024_kind_tuned_error_gallery_night_pedestrian/index.md
outputs/bdd100k_yolo_img1024_kind_tuned_error_gallery_highway_pedestrian/index.md
outputs/bdd100k_yolo_img1024_kind_tuned_error_gallery_rainy_traffic_light/index.md
```

観察:

- night pedestrian FNは暗所、遠距離、小bbox、部分遮蔽、車体や柱との近接が多い。
- highway pedestrian FNは横断歩道・路肩・中央分離帯付近の小さい人物やriderが多く、画像内の歩行者絶対数は多くないがF1低下が大きい。
- rainy traffic light FNは小さい灯器、雨滴/霧/白飛び、強いヘッドライトや反射に埋もれた灯器が目立つ。
- 単純にthresholdを下げるとFPが増えるため、次は条件別thresholdより先に、source品質フィルタとhard negative/positive設計を見直す。

### Phase 7: Demo体験を整える

目的:

評価で強いだけでなく、OSSとして「ADASっぽい認識」がすぐ見える状態を保つ。

やること:

- demo image/videoのREADMEコマンドを最新推奨configに揃える
- dense sceneでラベルが重なりすぎる問題を軽減する
- class filterを可視化側に追加する
- confidence表示を簡潔にする
- distance推定が粗いことを可視化上でも誤解しにくくする
- JSON出力と画像出力のファイル名を分かりやすくする

採用基準:

- 依存関係を入れたあと、画像または動画に1コマンドで可視化結果が出る
- default configは重すぎない
- BDD100K fine-tuned configは追加weightが必要な実験設定として分ける

### Phase 8: Lane recognitionを強化する (完了 — 非ML改善 + ONNX backend + TwinLiteNet 動作確認)

目的:

現状のHough/edgeベース車線検出はデモ向けで、BDD100K object評価とは別に弱い。

やること:

- lane専用public datasetを調べる
- TuSimple/CULaneなど、ライセンスと入手性を確認する
- 軽量lane segmentationモデルの候補を選ぶ
- OpenCV lane detectorはfallbackとして残す
- lane評価スクリプトを別途作る

注意:

- 車線認識は物体検出と評価軸が違う
- 物体検出改善と混ぜず、別トラックで進める

### Phase 9: Tracking and distance (完了 — motion + centroid + two-stage + intrinsics + ground projection)

Tracker upgrade の A/B 検証 (test_video.mp4 50 frame、2-way WBF + tracking 有効):

```text
                                  old IoU only   new motion+centroid   delta
pedestrian unique IDs             12             11                    -1 (-8.3%)
vehicle unique IDs                 3              3                    +0
```

slow-movement 主体の短い動画での差分は限定的だが、pedestrian で 1 track の ID 断絶を抑制。
Highway や高速ego-motion のある本番動画ではより大きな効果が期待できる。

目的:

ADASっぽさを上げるが、v0の評価主軸からは外す。

tracking:

- 現状は簡易IoU tracker
- ByteTrack系やKalman filterに置き換え候補
- まずは動画デモのID安定性改善

distance:

- 現状はbbox高さと仮定物体高さによる粗い目安
- カメラキャリブレーション入力を追加する
- 推定値の表示は控えめにする

採用基準:

- デモの見た目が改善する
- safety-criticalな距離推定であるように見せない

## 直近でやる順番

1. `PLAN.md`を作る
2. official BDD100K train dataの有無を再確認する
3. official train/val exportを実行する
4. YOLOv8n 1024pxをofficial trainで10epoch学習する
5. official valで評価する
6. cached threshold sweepを実行する
7. current validation mirror bestとの補助比較を残す
8. YOLOv8sまたはYOLO11nを同じ条件で比較する
9. small-object error samplesを見てcopy-paste v2のsource filterを決める
10. copy-paste v2を実装する

## 直近のコマンド集

現行ベストの再評価:

```bash
python scripts/evaluate_bdd100k.py \
  --images-root data/bdd100k/images/100k/val \
  --labels data/bdd100k/labels/det_20/det_val.json \
  --config configs/bdd100k_yolo_finetuned_all_tuned_split_img1024_kind_tuned.yaml \
  --device cuda \
  --frame-stride 2 \
  --frame-offset 1 \
  --group-by-size \
  --progress-every 1000 \
  --output outputs/bdd100k_yolo_finetuned_all_tuned_split_img1024_kind_tuned_report_odd_5000_size.json
```

feather copy-paste結果との比較:

```bash
python scripts/compare_evaluations.py \
  --reports \
    outputs/bdd100k_yolo_finetuned_all_tuned_split_img1024_kind_tuned_report_odd_5000_size.json \
    outputs/bdd100k_yolo_finetuned_all_even_copy_paste_1024_1ep_tuned_report_odd_5000_size.json \
    outputs/bdd100k_yolo_finetuned_all_even_copy_paste_light_1024_1ep_report_odd_5000_size.json \
    outputs/bdd100k_yolo_finetuned_all_even_copy_paste_feather_1024_1ep_tuned_report_odd_5000_size.json \
  --names \
    current_best \
    copy_paste_2500_tuned \
    copy_paste_light_1000 \
    copy_paste_feather_2500_tuned \
  --output outputs/bdd100k_yolo_even_copy_paste_feather_1024_1ep_tuned_compare_report_odd_5000.json \
  --markdown-output outputs/bdd100k_yolo_even_copy_paste_feather_1024_1ep_tuned_compare_report_odd_5000.md \
  --csv-output outputs/bdd100k_yolo_even_copy_paste_feather_1024_1ep_tuned_compare_report_odd_5000.csv
```

demo image:

```bash
python scripts/demo_image.py \
  --input path/to/road.jpg \
  --config configs/bdd100k_yolo_finetuned_all_tuned_split_img1024_kind_tuned.yaml \
  --output outputs/demo_road_annotated.jpg \
  --json-output outputs/demo_road.json
```

demo video:

```bash
python scripts/demo_video.py \
  --input path/to/road.mp4 \
  --config configs/bdd100k_yolo_finetuned_all_tuned_split_img1024_kind_tuned.yaml \
  --output outputs/demo_road_annotated.mp4 \
  --json-output outputs/demo_road.json
```

## 採用しない設定

現時点でデフォルトや推奨設定にしないもの:

```text
hard_small
hard_small_tuned
even_1024_1ep
even_class_balanced_1024_1ep
even_class_balanced_1024_1ep_tuned
even_object_crops_1024_1ep
even_object_crops_1024_1ep_tuned
even_copy_paste_1024_1ep
even_copy_paste_1024_1ep_tuned
even_copy_paste_light_1024_1ep
even_copy_paste_light_1024_1ep_tuned
even_copy_paste_feather_1024_1ep
even_copy_paste_feather_1024_1ep_tuned
even_copy_paste_v2_mask_1000_1024_1ep
even_copy_paste_v2_mask_1000_1024_1ep_tuned
even_copy_paste_v2_tl_1000_1024_1ep
even_copy_paste_v2_tl_1000_1024_1ep_tuned
img1024_kind_tuned_tiles_tiny
img1024_size_tuned_tiny_recall
```

理由:

- 現行ベスト `macro F1=0.6355` を超えていない
- 一部クラスだけ伸びて総合が落ちる
- FP増加が大きい
- 速度低下に見合う改善がない
- validation mirrorの追加学習であり、official train/valではない

## 残すべき成果物

重要:

```text
outputs/models/adas_yolov8n_bdd100k.pt
configs/bdd100k_yolo_finetuned_all_tuned_split_img1024_kind_tuned.yaml
outputs/bdd100k_yolo_finetuned_all_tuned_split_img1024_kind_tuned_report_odd_5000_size.json
outputs/bdd100k_yolo_finetuned_all_tuned_split_img1024_kind_tuned_report_odd_5000_grouped_size.json
outputs/bdd100k_yolo_even_copy_paste_feather_1024_1ep_tuned_compare_report_odd_5000.md
outputs/bdd100k_copy_paste_feather_1024_cached_threshold_sweep_even_1000/comparison.md
```

実験比較として残す:

```text
configs/bdd100k_yolo_finetuned_all_even_copy_paste_feather_1024_1ep.yaml
configs/bdd100k_yolo_finetuned_all_even_copy_paste_feather_1024_1ep_tuned.yaml
configs/bdd100k_yolo_finetuned_all_even_copy_paste_feather_1024_1ep_cache_low.yaml
outputs/models/adas_yolov8n_bdd100k_even_copy_paste_feather_1024_1ep.pt
```

削除してよいもの:

- smoke export
- 途中検証だけの一時output
- 同名で上書き済みの古いreport

ただし、削除はユーザー作業との衝突を避けるため、明示的に自分が作った一時ファイルだけにする。

## リスク

### Public datasetの入手とライセンス

BDD100Kは公開データだが、利用条件と入手手順がある。リポジトリにデータ本体を含めない。READMEには配置方法と期待パスだけを書く。

### validation mirrorへの過適合

現在の多くの実験はBDD100K valをeven/oddに分けたもの。これは反復には便利だが、正式な学習/評価分離ではない。official train split導入後は、過去結果を参考値として扱う。

### 小物体augmentationの分布破壊

crop-onlyやbbox copy-pasteは小物体を増やせるが、見え方の分布を壊す。改善にはsource品質、貼り付け位置、mask、枚数、学習epochのバランスが必要。

### 指標の単純化

macro F1は分かりやすいが、ADAS用途の全てを表さない。distance、tracking、lane、traffic light stateは別評価が必要。

### 速度測定のばらつき

FPSはGPU状態、max_detections、threshold、画像サイズで変わる。同一条件で比較する。

## 長期の方向性

v0.x milestone は到達済み。v0.x+ / v1.0 で以下の順に積む。

1. 1コマンドで動くデモを維持する (online demo / web demo)
2. BDD100K public data で再現可能な評価を維持する (cache + sweep workflow)
3. object detection の accuracy ceiling 更新は **官公 BDD100K train split** がキー
4. lane segmentation を追加する (TuSimple / CULane segmentation モデル)
5. traffic light state 分類を学習済みに置き換える
6. tracking を ByteTrack / DeepSORT 系に寄せる
   (現在は IoU + linear motion + centroid fallback)
7. distance 推定に camera calibration を入れる
8. 複数モデルの速度 / 精度表を README に載せる
9. **エッジデプロイ (Jetson Nano / Orin Nano クラス)**:
   - Autoware の RViz heavy GUI / OpenPilot の vendor HW lock との差別化軸
   - ONNX export → TensorRT engine (FP16 / INT8) → 軽量 config preset
   - `scripts/web_demo.py` (gradio) を実機 LAN ブラウザから利用
   - ROS なし、pip install 数分、Jetson Nano 級でも動く位置付け
   - **要実機**: 開発機 (GPU) では動作確認のみ、FPS 実測は実機で
10. 日本仕様 fine-tune (Japan Traffic Sign dataset、narrow streets)

## 次の一手 (v0.x milestone wrap-up 後)

inference-side accuracy 探索は飽和済み (WBF 7-way online = 0.6763 が最終 ceiling)。
追加学習も validation mirror の短い epoch では macro -0.005 で頭打ち。

**次に意味のある作業はすべて user action / 実機入手待ち** で停止している:

- BDD100K official train split の配置 (accuracy 続伸の唯一の道、最優先)
- Jetson 実機入手 (エッジデプロイ実証、差別化軸)

それまでに保留の参考メモ:

```text
current_best         0.6355  (no retrain)
plain retrain 1ep    0.6309  (-0.005 from current best)
v2 mixed tuned       0.6310  (-0.005)
v2 tl-only tuned     0.6325  (-0.003)
```

主な regression は追加学習 (1 epoch) そのものに由来する。
copy-paste (特に tl-only) は plain retrain に対して純利得 (+0.0016) だが、
追加学習の -0.005 を補うほどではない。

次善策 (期待値順):

1. official BDD100K train split の配置待ち (最優先、段違いに期待値が高い)
2. 追加学習を介さず current best model に inference-side で手を加える方向:
   - class-wise post-processing
   - small object 向け NMS tuning
   - TTA (horizontal flip, multi-scale) on current best weights
   これなら追加学習の lossy 性を回避できる
3. traffic_light 特化 config (tl-only tuned) を experiment として別枠で残す
4. plain retrain の epoch 数 / lr scheduling を見直し、-0.005 を圧縮できるか

現時点では、validation mirror 上の copy-paste / bbox系 augmentation はこれ以上深追いしない。
traffic_light 特化用途の config として tl-only tuned を残す。
