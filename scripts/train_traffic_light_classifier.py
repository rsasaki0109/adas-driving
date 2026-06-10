"""Train a tiny traffic-light state classifier from BDD100K val-mirror crops.

BDD100K det_20 labels carry a `trafficLightColor` attribute
(red / yellow / green / unknown), so a learned state classifier can be
trained without the official train split: even-index frames of the local
val mirror are the train split, odd-index frames are the report split
(the same odd-5,000 convention used by evaluate_bdd100k.py).

The HSV baseline (`TrafficLightStateClassifier`) is evaluated on the same
odd-split crops for a like-for-like comparison.

Usage:
    python scripts/train_traffic_light_classifier.py \
        --images-root data/bdd100k/images/100k/val \
        --labels data/bdd100k/labels/det_20/det_val.json \
        --device cuda

Outputs:
    outputs/models/traffic_light_state.pt
    outputs/models/traffic_light_state.onnx
    outputs/tl_state_eval.json (CNN vs HSV metrics on the odd split)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

import sys

sys.path.insert(0, str(REPO_ROOT))

from adas_perception.traffic_light_state import (  # noqa: E402
    STATE_NAMES,
    TrafficLightStateClassifier,
)
from adas_perception.types import Box, Detection  # noqa: E402

STATE_TO_INDEX = {state: index for index, state in enumerate(STATE_NAMES)}
CROP_WIDTH = 32
CROP_HEIGHT = 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-root", default="data/bdd100k/images/100k/val")
    parser.add_argument("--labels", default="data/bdd100k/labels/det_20/det_val.json")
    parser.add_argument("--cache-dir", default="outputs/tl_state_cache")
    parser.add_argument("--model-output", default="outputs/models/traffic_light_state.pt")
    parser.add_argument("--onnx-output", default="outputs/models/traffic_light_state.onnx")
    parser.add_argument("--metrics-output", default="outputs/tl_state_eval.json")
    parser.add_argument("--min-box-pixels", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--rebuild-cache", action="store_true", help="Re-extract crops even if cache exists"
    )
    return parser.parse_args()


def _color_to_state(color: str | None) -> str:
    if color in ("red", "yellow", "green"):
        return color
    return "off"  # unknown / missing → not a lit, actionable lamp


def build_crop_cache(args: argparse.Namespace) -> dict[str, Path]:
    """Extract resized traffic-light crops + labels + HSV predictions per split."""
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = {split: cache_dir / f"crops_{split}.npz" for split in ("even", "odd")}
    if not args.rebuild_cache and all(path.exists() for path in paths.values()):
        print(f"[cache] reusing {paths['even']} / {paths['odd']}")
        return paths

    frames = json.loads(Path(args.labels).read_text())
    frames.sort(key=lambda frame: frame["name"])
    images_root = Path(args.images_root)
    hsv_baseline = TrafficLightStateClassifier({"min_box_pixels": args.min_box_pixels})

    buffers: dict[str, dict[str, list]] = {
        split: {"crops": [], "labels": [], "hsv": []} for split in ("even", "odd")
    }
    skipped_missing = 0
    for frame_index, frame in enumerate(frames):
        split = "even" if frame_index % 2 == 0 else "odd"
        tl_labels = [
            label
            for label in (frame.get("labels") or [])
            if label.get("category") == "traffic light" and label.get("box2d")
        ]
        if not tl_labels:
            continue
        image_path = images_root / frame["name"]
        image = cv2.imread(str(image_path))
        if image is None:
            skipped_missing += 1
            continue
        for label in tl_labels:
            box2d = label["box2d"]
            x1, y1 = int(box2d["x1"]), int(box2d["y1"])
            x2, y2 = int(box2d["x2"]), int(box2d["y2"])
            if (x2 - x1) * (y2 - y1) < args.min_box_pixels:
                continue
            crop = image[max(0, y1) : max(0, y2), max(0, x1) : max(0, x2)]
            if crop.size == 0:
                continue
            state = _color_to_state((label.get("attributes") or {}).get("trafficLightColor"))
            detection = Detection(
                kind="traffic_light",
                label="traffic light",
                confidence=1.0,
                box=Box(x1=x1, y1=y1, x2=x2, y2=y2),
                source="gt",
            )
            hsv_state = hsv_baseline._state_for_box(image, detection) or "off"
            resized = cv2.resize(crop, (CROP_WIDTH, CROP_HEIGHT), interpolation=cv2.INTER_LINEAR)
            buffers[split]["crops"].append(resized)
            buffers[split]["labels"].append(STATE_TO_INDEX[state])
            buffers[split]["hsv"].append(STATE_TO_INDEX[hsv_state])
        if frame_index % 1000 == 0:
            done = {s: len(b["labels"]) for s, b in buffers.items()}
            print(f"[cache] frame {frame_index}/{len(frames)} crops={done}")

    if skipped_missing:
        print(f"[cache] WARNING: {skipped_missing} frames had missing image files")
    for split, buffer in buffers.items():
        np.savez_compressed(
            paths[split],
            crops=np.stack(buffer["crops"]).astype(np.uint8),
            labels=np.array(buffer["labels"], dtype=np.int64),
            hsv=np.array(buffer["hsv"], dtype=np.int64),
        )
        print(f"[cache] wrote {paths[split]} ({len(buffer['labels'])} crops)")
    return paths


def metrics_from_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    per_class = {}
    f1_values = []
    for state, index in STATE_TO_INDEX.items():
        tp = int(np.sum((y_pred == index) & (y_true == index)))
        fp = int(np.sum((y_pred == index) & (y_true != index)))
        fn = int(np.sum((y_pred != index) & (y_true == index)))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[state] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": int(np.sum(y_true == index)),
        }
        f1_values.append(f1)
    return {
        "accuracy": round(float(np.mean(y_true == y_pred)), 4),
        "macro_f1": round(float(np.mean(f1_values)), 4),
        "per_class": per_class,
    }


def main() -> None:
    args = parse_args()
    import torch
    from torch import nn

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    paths = build_crop_cache(args)
    train_data = np.load(paths["even"])
    eval_data = np.load(paths["odd"])
    train_crops = train_data["crops"]
    train_labels = train_data["labels"]
    eval_crops = eval_data["crops"]
    eval_labels = eval_data["labels"]
    print(f"[data] train={len(train_labels)} eval={len(eval_labels)}")
    print(f"[data] train class counts: {np.bincount(train_labels, minlength=4).tolist()} ({STATE_NAMES})")

    class TinyTrafficLightNet(nn.Module):
        def __init__(self, num_classes: int = len(STATE_NAMES)):
            super().__init__()
            def block(cin: int, cout: int) -> nn.Sequential:
                return nn.Sequential(
                    nn.Conv2d(cin, cout, 3, padding=1),
                    nn.BatchNorm2d(cout),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                )
            self.features = nn.Sequential(block(3, 16), block(16, 32), block(32, 64))
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, num_classes)
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.head(self.features(x))

    model = TinyTrafficLightNet().to(device)
    counts = np.bincount(train_labels, minlength=len(STATE_NAMES)).astype(np.float64)
    class_weights = counts.sum() / np.maximum(counts, 1.0)
    class_weights = class_weights / class_weights.mean()
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    def to_tensor(batch_crops: np.ndarray) -> "torch.Tensor":
        # uint8 BGR HWC → float CHW in [0, 1]; BGR order is kept at inference too.
        array = batch_crops.astype(np.float32) / 255.0
        return torch.from_numpy(array).permute(0, 3, 1, 2).to(device)

    def predict(crops: np.ndarray, batch_size: int = 1024) -> np.ndarray:
        model.eval()
        outputs = []
        with torch.no_grad():
            for start in range(0, len(crops), batch_size):
                logits = model(to_tensor(crops[start : start + batch_size]))
                outputs.append(logits.argmax(dim=1).cpu().numpy())
        return np.concatenate(outputs) if outputs else np.empty(0, dtype=np.int64)

    indices = np.arange(len(train_labels))
    for epoch in range(args.epochs):
        model.train()
        np.random.shuffle(indices)
        losses = []
        for start in range(0, len(indices), args.batch_size):
            batch_idx = indices[start : start + args.batch_size]
            crops = train_crops[batch_idx].copy()
            flip = np.random.rand(len(crops)) < 0.5
            crops[flip] = crops[flip, :, ::-1]
            gain = np.random.uniform(0.7, 1.3, size=(len(crops), 1, 1, 1)).astype(np.float32)
            inputs = torch.from_numpy(
                np.clip(crops.astype(np.float32) * gain, 0, 255) / 255.0
            ).permute(0, 3, 1, 2).to(device)
            targets = torch.from_numpy(train_labels[batch_idx]).to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        train_acc = float(np.mean(predict(train_crops[:5000]) == train_labels[:5000]))
        print(f"[train] epoch {epoch + 1}/{args.epochs} loss={np.mean(losses):.4f} train_acc~={train_acc:.4f}")

    cnn_metrics = metrics_from_predictions(eval_labels, predict(eval_crops))
    hsv_metrics = metrics_from_predictions(eval_labels, eval_data["hsv"])
    report = {
        "train_crops": int(len(train_labels)),
        "eval_crops": int(len(eval_labels)),
        "state_names": list(STATE_NAMES),
        "cnn": cnn_metrics,
        "hsv_baseline": hsv_metrics,
    }
    print(json.dumps(report, indent=2))

    model_path = Path(args.model_output)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    dummy = torch.zeros(1, 3, CROP_HEIGHT, CROP_WIDTH, device=device)
    torch.onnx.export(
        model,
        (dummy,),
        args.onnx_output,
        input_names=["crops"],
        output_names=["logits"],
        dynamic_axes={"crops": {0: "batch"}, "logits": {0: "batch"}},
        dynamo=False,
    )
    Path(args.metrics_output).write_text(json.dumps(report, indent=2))
    print(f"[done] model={model_path} onnx={args.onnx_output} metrics={args.metrics_output}")


if __name__ == "__main__":
    main()
