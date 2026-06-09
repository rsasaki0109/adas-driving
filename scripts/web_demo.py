#!/usr/bin/env python3
"""Headless-friendly web demo for adas-perception (+ optional planning overlay).

Runs the camera perception pipeline with an uploaded image or webcam frame
and shows the annotated output in a browser, so Jetson / remote boxes can
demo without RViz-style heavy GUIs. The server binds to 0.0.0.0:7860 by
default so it is reachable from another machine on the LAN.

Optional planning overlay runs rule-based planners on each frame and draws
target path / behavior / warnings on top of perception visualization.

Requires `gradio` (`.venv/bin/pip install gradio`). Config presets come from
this repo, so running this from the project root is expected.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adas_perception.config import apply_runtime_overrides, load_config
from adas_perception.pipeline import ADASPerceptionPipeline
from adas_perception.serialization import frame_result_to_dict
from adas_perception.visualization import draw_perception
from adas_planning.config import load_config as load_planning_config
from adas_planning.io.perception_adapter import adapt_perception_frame
from adas_planning.pipeline import PlanningPipeline
from adas_planning.viz.overlay import draw_planning_overlay


PRESETS: dict[str, str] = {
    "Fast + post-NMS (sweep best)": "configs/bdd100k_yolo_kind_tuned_post_nms.yaml",
    "Fast — 1024px (no TTA, demo baseline)": "configs/bdd100k_yolo_finetuned_all_tuned_split_img1024_kind_tuned.yaml",
    "Single-config accurate — TTA + tiny override": "configs/bdd100k_yolo_finetuned_all_tuned_split_img1024_kind_tuned_tta_tuned_tiny.yaml",
    "WBF 7-way (slow, accuracy ceiling)": "configs/bdd100k_yolo_wbf7_demo.yaml",
    "Heuristic only (CPU-friendly)": "configs/default.yaml",
}

PLANNING_PRESETS: dict[str, str] = {
    "Default": "configs/planning/default.yaml",
    "Conservative": "configs/planning/conservative.yaml",
}

DEFAULT_PRESET = "Heuristic only (CPU-friendly)"
DEFAULT_PLANNING_PRESET = "Default"

_pipeline_cache: dict[str, tuple[ADASPerceptionPipeline, dict]] = {}
_planning_cache: dict[str, dict[str, Any]] = {}


def _get_pipeline(preset: str, device: str | None) -> tuple[ADASPerceptionPipeline, dict]:
    path = PRESETS[preset]
    cache_key = f"{path}|{device or ''}"
    if cache_key not in _pipeline_cache:
        config = apply_runtime_overrides(load_config(path), device=device, disable_objects=False)
        _pipeline_cache[cache_key] = (ADASPerceptionPipeline(config), config)
    return _pipeline_cache[cache_key]


def _get_planning_config(preset: str) -> dict[str, Any]:
    path = PLANNING_PRESETS[preset]
    if path not in _planning_cache:
        _planning_cache[path] = load_planning_config(path)
    return _planning_cache[path]


def _new_planning_pipeline(preset: str) -> tuple[PlanningPipeline, dict[str, Any]]:
    config = _get_planning_config(preset)
    return PlanningPipeline(config), config


def _stats(counts: dict[str, int], elapsed: float, planning_result: Any | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "inference_ms": round(elapsed * 1000.0, 1),
        "fps_estimate": round(1.0 / elapsed, 2) if elapsed > 0 else None,
        "lanes": int(counts.get("lane", 0)),
        "vehicles": int(counts.get("vehicle", 0)),
        "pedestrians": int(counts.get("pedestrian", 0)),
        "traffic_signs": int(counts.get("traffic_sign", 0)),
        "traffic_lights": int(counts.get("traffic_light", 0)),
    }
    if planning_result is not None:
        behavior = planning_result.behavior.value if hasattr(planning_result.behavior, "value") else str(planning_result.behavior)
        payload["planning"] = {
            "behavior": behavior,
            "target_speed_mps": planning_result.target_speed_mps,
            "confidence": round(float(planning_result.confidence), 3),
            "warnings": [warning.code for warning in planning_result.warnings[:5]],
        }
    return payload


def render_frame_bgr(
    frame_bgr: np.ndarray,
    *,
    preset: str,
    device: str | None,
    enable_planning: bool = False,
    planning_preset: str = DEFAULT_PLANNING_PRESET,
    frame_index: int = 0,
    timestamp_ms: float = 0.0,
    perception_pipeline: ADASPerceptionPipeline | None = None,
    planning_pipeline: PlanningPipeline | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    if perception_pipeline is None:
        perception_pipeline, config = _get_pipeline(preset, device)
    else:
        _, config = _get_pipeline(preset, device)

    height, width = frame_bgr.shape[:2]
    t0 = time.perf_counter()
    result = perception_pipeline.run(frame_bgr)
    vis_bgr = draw_perception(frame_bgr, result, config)

    planning_result = None
    if enable_planning:
        if planning_pipeline is None:
            planning_pipeline, planning_config = _new_planning_pipeline(planning_preset)
        else:
            planning_config = _get_planning_config(planning_preset)
        frame_dict = frame_result_to_dict(
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
            result=result,
        )
        planning_input = adapt_perception_frame(
            frame_dict,
            image_width=width,
            image_height=height,
        )
        planning_result = planning_pipeline.plan(planning_input)
        overlay_config = dict(planning_config)
        overlay_section = dict(overlay_config.get("overlay") or {})
        overlay_section["draw_lane_context"] = False
        overlay_config["overlay"] = overlay_section
        vis_bgr = draw_planning_overlay(
            vis_bgr,
            planning_result,
            perception_frame=frame_dict,
            config=overlay_config,
        )

    elapsed = time.perf_counter() - t0
    return vis_bgr, _stats(result.count_by_kind(), elapsed, planning_result)


def infer_image(
    image_rgb: np.ndarray | None,
    preset: str,
    device: str,
    enable_planning: bool,
    planning_preset: str,
) -> tuple[np.ndarray | None, dict]:
    if image_rgb is None:
        return None, {}
    pipeline, _ = _get_pipeline(preset, device or None)
    pipeline.reset()
    planning_pipeline = _new_planning_pipeline(planning_preset)[0] if enable_planning else None
    frame_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    vis_bgr, stats = render_frame_bgr(
        frame_bgr,
        preset=preset,
        device=device or None,
        enable_planning=enable_planning,
        planning_preset=planning_preset,
        perception_pipeline=pipeline,
        planning_pipeline=planning_pipeline,
    )
    vis_rgb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)
    return vis_rgb, stats


def infer_video(
    video_path: str | None,
    preset: str,
    device: str,
    max_frames: int,
    enable_planning: bool,
    planning_preset: str,
):
    if not video_path:
        return None, {}
    pipeline, _ = _get_pipeline(preset, device or None)
    planning_pipeline = _new_planning_pipeline(planning_preset)[0] if enable_planning else None
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
    last_planning_stats: dict[str, Any] | None = None
    try:
        while True:
            if max_frames and total >= max_frames:
                break
            ok, frame = cap.read()
            if not ok:
                break
            timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            t0 = time.perf_counter()
            vis_bgr, stats = render_frame_bgr(
                frame,
                preset=preset,
                device=device or None,
                enable_planning=enable_planning,
                planning_preset=planning_preset,
                frame_index=total,
                timestamp_ms=timestamp_ms,
                perception_pipeline=pipeline,
                planning_pipeline=planning_pipeline,
            )
            elapsed_total += time.perf_counter() - t0
            writer.write(vis_bgr)
            for k, v in stats.items():
                if k == "planning":
                    last_planning_stats = v
                    continue
            for key in ("lanes", "vehicles", "pedestrians", "traffic_signs", "traffic_lights"):
                value = stats.get(key)
                if isinstance(value, int):
                    counts_total[key] = counts_total.get(key, 0) + value
            total += 1
    finally:
        cap.release()
        writer.release()
    avg_ms = 1000.0 * elapsed_total / max(total, 1)
    payload: dict[str, Any] = {
        "frames": total,
        "avg_inference_ms": round(avg_ms, 1),
        "fps_estimate": round(1000.0 / avg_ms, 2) if avg_ms > 0 else None,
        "per_frame_detections_total": counts_total,
    }
    if last_planning_stats is not None:
        payload["planning_last_frame"] = last_planning_stats
    return str(out_path), payload


def build_ui():
    import gradio as gr

    with gr.Blocks(title="adas-perception web demo") as demo:
        gr.Markdown(
            "# adas-perception web demo\n"
            "Headless-friendly camera perception (lane / vehicle / pedestrian / sign / light) with "
            "optional rule-based planning overlay (target path, behavior, warnings). "
            "Unlike Autoware RViz this is a browser-only UI, no X11 / ROS required — drop the "
            "pipeline on a Jetson and point a browser at `http://<jetson>:7860`.\n\n"
            "**Presets** trade off accuracy vs FPS. See PLAN.md for the full WBF ladder and BDD100K numbers."
        )
        with gr.Tabs():
            with gr.Tab("Image"):
                with gr.Row():
                    with gr.Column():
                        inp_img = gr.Image(label="Input (upload or webcam)", sources=["upload", "webcam"], type="numpy")
                        preset_img = gr.Radio(list(PRESETS.keys()), value=DEFAULT_PRESET, label="Config preset")
                        device_img = gr.Radio(["auto", "cuda", "cpu"], value="auto", label="Device")
                        enable_planning_img = gr.Checkbox(label="Enable planning overlay", value=False)
                        planning_preset_img = gr.Radio(
                            list(PLANNING_PRESETS.keys()),
                            value=DEFAULT_PLANNING_PRESET,
                            label="Planning config",
                            visible=False,
                        )
                        btn_img = gr.Button("Detect", variant="primary")
                    with gr.Column():
                        out_img = gr.Image(label="Annotated output")
                        stats_img = gr.JSON(label="Stats")
                enable_planning_img.change(
                    lambda enabled: gr.update(visible=enabled),
                    [enable_planning_img],
                    [planning_preset_img],
                )
                btn_img.click(
                    infer_image,
                    [inp_img, preset_img, device_img, enable_planning_img, planning_preset_img],
                    [out_img, stats_img],
                )
            with gr.Tab("Video"):
                with gr.Row():
                    with gr.Column():
                        inp_vid = gr.Video(label="Input video (mp4/mov)")
                        preset_vid = gr.Radio(list(PRESETS.keys()), value=DEFAULT_PRESET, label="Config preset")
                        device_vid = gr.Radio(["auto", "cuda", "cpu"], value="auto", label="Device")
                        enable_planning_vid = gr.Checkbox(label="Enable planning overlay", value=False)
                        planning_preset_vid = gr.Radio(
                            list(PLANNING_PRESETS.keys()),
                            value=DEFAULT_PLANNING_PRESET,
                            label="Planning config",
                            visible=False,
                        )
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
                enable_planning_vid.change(
                    lambda enabled: gr.update(visible=enabled),
                    [enable_planning_vid],
                    [planning_preset_vid],
                )
                btn_vid.click(
                    infer_video,
                    [inp_vid, preset_vid, device_vid, max_frames_vid, enable_planning_vid, planning_preset_vid],
                    [out_vid, stats_vid],
                )
        gr.Markdown(
            "### Notes\n"
            "- Presets load pipelines lazily; first detection with a given preset pays the model-load cost.\n"
            "- **Fast + post-NMS** applies the production sweep thresholds/NMS IoU (`configs/bdd100k_yolo_kind_tuned_post_nms.yaml`).\n"
            "- WBF 7-way is accuracy-priority (macro F1 ≈ 0.676 on BDD100K val) and runs ~3 FPS on GPU; expect well below 1 FPS on Jetson Nano.\n"
            "- Planning overlay is research/demo only — not for vehicle control.\n"
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
