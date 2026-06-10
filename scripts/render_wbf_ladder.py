#!/usr/bin/env python3
"""Render the WBF macro-F1 ladder chart for README."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "assets" / "wbf_ladder.png"

LADDER = [
    ("no-TTA baseline", 0.6355, 29.15),
    ("TTA tuned tiny", 0.6389, 24.71),
    ("2-way WBF", 0.6447, 10.5),
    ("3-way WBF", 0.6489, 7.8),
    ("4-way WBF", 0.6602, 4.7),
    ("5-way WBF", 0.6627, 4.0),
    ("6-way WBF", 0.6686, 3.3),
    ("7-way WBF", 0.6724, 2.8),
    ("7-way + iou tune", 0.6747, 2.8),
    ("7-way per-kind iou (online)", 0.6763, 3.4),
]


def main() -> int:
    labels = [item[0] for item in LADDER]
    f1 = [item[1] for item in LADDER]
    fps = [item[2] for item in LADDER]

    fig, ax1 = plt.subplots(figsize=(11, 5.5))
    x = range(len(labels))
    bars = ax1.bar(x, f1, color="#3b82f6", alpha=0.85, label="macro F1")
    ax1.plot(x, f1, color="#1d4ed8", marker="o", linewidth=2)
    ax1.set_ylabel("macro F1 (BDD100K odd 5000)")
    ax1.set_ylim(0.62, 0.685)
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(labels, rotation=35, ha="right")
    ax1.axhline(0.6355, color="#94a3b8", linestyle="--", linewidth=1, label="previous best")
    ax1.grid(axis="y", alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(x, fps, color="#f97316", marker="s", linewidth=1.8, label="FPS (approx)")
    ax2.set_ylabel("FPS (RTX 4070, approx)")
    ax2.set_ylim(0, 32)

    ax1.set_title("Inference-side WBF ladder: +0.0408 macro F1 without retraining")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=160)
    print(f"saved {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
