#!/usr/bin/env python3
"""Headless-friendly web demo for adas-perception.

Runs the camera perception pipeline with an uploaded image or webcam frame
and shows the annotated output in a browser, so Jetson / remote boxes can
demo without RViz-style heavy GUIs. The server binds to 0.0.0.0:7860 by
default so it is reachable from another machine on the LAN.

Requires `gradio` (`.venv/bin/pip install gradio`). Config presets come from
this repo, so running this from the project root is expected.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gradio as gr

from adas_perception.config import apply_runtime_overrides, load_config
from adas_perception.pipeline import ADASPerceptionPipeline
from adas_perception.visualization import draw_perception


PRESETS: dict[str, str] = {
    "Fast — 1024px (no TTA, demo baseline)": "configs/bdd100k_yolo_finetuned_all_tuned_split_img1024_kind_tuned.yaml",
    "Single-config accurate — TTA + tiny override": "configs/bdd100k_yolo_finetuned_all_tuned_split_img1024_kind_tuned_tta_tuned_tiny.yaml",
    "WBF 7-way (slow, accuracy ceiling)": "configs/bdd100k_yolo_wbf7_demo.yaml",
    "Heuristic only (CPU-friendly)": "configs/default.yaml",
}


_pipeline_cache: dict[str, tuple[ADASPerceptionPipeline, dict]] = {}


def _get_pipeline(preset: str, device: str | None) -> tuple[ADASPerceptionPipeline, dict]:
    path = PRESETS[preset]
    cache_key = f"{path}|{device or ''}"
    if cache_key not in _pipeline_cache:
        config = apply_runtime_overrides(load_config(path), device=device, disable_objects=False)
        _pipeline_cache[cache_key] = (ADASPerceptionPipeline(config), config)
    return _pipeline_cache[cache_key]


def _stats(counts: dict[str, int], elapsed: float) -> dict:
    return {
        "inference_ms": round(elapsed * 1000.0, 1),
        "fps_estimate": round(1.0 / elapsed, 2) if elapsed > 0 else None,
        "lanes": int(counts.get("lane", 0)),
        "vehicles": int(counts.get("vehicle", 0)),
        "pedestrians": int(counts.get("pedestrian", 0)),
        "traffic_signs": int(counts.get("traffic_sign", 0)),
        "traffic_lights": int(counts.get("traffic_light", 0)),
    }


def infer_image(image_rgb: np.ndarray | None, preset: str, device: str) -> tuple[np.ndarray | None, dict]:
    if image_rgb is None:
        return None, {}
    pipeline, config = _get_pipeline(preset, device or None)
    frame_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    pipeline.reset()
    t0 = time.perf_counter()
    result = pipeline.run(frame_bgr)
    elapsed = time.perf_counter() - t0
    vis_bgr = draw_perception(frame_bgr, result, config)
    vis_rgb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)
    return vis_rgb, _stats(result.count_by_kind(), elapsed)


def infer_video(video_path: str | None, preset: str, device: str, max_frames: int):
    if not video_path:
        return None, {}
    pipeline, config = _get_pipeline(preset, device or None)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, {"error": f"could not open {video_path}"}
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 24.0
    out_path = Path("outputs") / "web_demo_out.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    pipeline.reset()
    total = 0
    elapsed_total = 0.0
    counts_total: dict[str, int] = {}
    try:
        while True:
            if max_frames and total >= max_frames:
                break
            ok, frame = cap.read()
            if not ok:
                break
            t0 = time.perf_counter()
            result = pipeline.run(frame)
            elapsed_total += time.perf_counter() - t0
            vis = draw_perception(frame, result, config)
            writer.write(vis)
            for k, v in result.count_by_kind().items():
                counts_total[k] = counts_total.get(k, 0) + int(v)
            total += 1
    finally:
        cap.release()
        writer.release()
    avg_ms = 1000.0 * elapsed_total / max(total, 1)
    stats = {
        "frames": total,
        "avg_inference_ms": round(avg_ms, 1),
        "fps_estimate": round(1000.0 / avg_ms, 2) if avg_ms > 0 else None,
        "per_frame_detections_total": counts_total,
    }
    return str(out_path), stats


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="adas-perception web demo") as demo:
        gr.Markdown(
            "# adas-perception web demo\n"
            "Headless-friendly camera perception (lane / vehicle / pedestrian / sign / light). "
            "Unlike Autoware RViz this is a browser-only UI, no X11 / ROS required — drop the "
            "pipeline on a Jetson and point a browser at `http://<jetson>:7860`.\n\n"
            "**Presets** trade off accuracy vs FPS. See PLAN.md for the full WBF ladder and BDD100K numbers."
        )
        with gr.Tabs():
            with gr.Tab("Image"):
                with gr.Row():
                    with gr.Column():
                        inp_img = gr.Image(label="Input (upload or webcam)", sources=["upload", "webcam"], type="numpy")
                        preset_img = gr.Radio(list(PRESETS.keys()), value=list(PRESETS.keys())[0], label="Config preset")
                        device_img = gr.Radio(["auto", "cuda", "cpu"], value="auto", label="Device")
                        btn_img = gr.Button("Detect", variant="primary")
                    with gr.Column():
                        out_img = gr.Image(label="Annotated output")
                        stats_img = gr.JSON(label="Stats")
                btn_img.click(infer_image, [inp_img, preset_img, device_img], [out_img, stats_img])
            with gr.Tab("Video"):
                with gr.Row():
                    with gr.Column():
                        inp_vid = gr.Video(label="Input video (mp4/mov)")
                        preset_vid = gr.Radio(list(PRESETS.keys()), value=list(PRESETS.keys())[0], label="Config preset")
                        device_vid = gr.Radio(["auto", "cuda", "cpu"], value="auto", label="Device")
                        max_frames_vid = gr.Slider(
                            label="Max frames (0 = full video)",
                            minimum=0,
                            maximum=600,
                            step=10,
                            value=120,
                        )
                        btn_vid = gr.Button("Process video", variant="primary")
                    with gr.Column():
                        out_vid = gr.Video(label="Annotated video")
                        stats_vid = gr.JSON(label="Stats")
                btn_vid.click(infer_video, [inp_vid, preset_vid, device_vid, max_frames_vid], [out_vid, stats_vid])
        gr.Markdown(
            "### Notes\n"
            "- Presets load pipelines lazily; first detection with a given preset pays the model-load cost.\n"
            "- WBF 7-way is accuracy-priority (macro F1 ≈ 0.676 on BDD100K val) and runs ~3 FPS on RTX 4070; expect well below 1 FPS on Jetson Nano.\n"
            "- Distance estimates (~Nm) are rough monocular values. See `visualization.distance_format` in the README for display options."
        )
    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Browser-based demo for adas-perception (headless friendly).")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address (default 0.0.0.0).")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Generate a gradio share link (tunnel).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ui = build_ui()
    ui.launch(server_name=args.host, server_port=args.port, share=args.share)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
