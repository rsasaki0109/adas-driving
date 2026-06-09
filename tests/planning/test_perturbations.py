from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from adas_planning.io.perception_adapter import adapt_perception_document
from adas_planning.pipeline import PlanningPipeline
from adas_planning.types import Behavior


FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "planning"


def _load_fixture(name: str) -> dict:
    with (FIXTURES_DIR / name).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _run_pipeline(payload: dict, config: dict | None = None) -> list:
    pipeline = PlanningPipeline(config or {})
    return [pipeline.plan(planning_input) for planning_input in adapt_perception_document(payload)]


def test_lane_dropout_keeps_valid_results():
    results = _run_pipeline(_load_fixture("lane_dropout.json"))
    assert len(results) == 9
    assert all(0.0 <= result.confidence <= 1.0 for result in results)
    assert sum(1 for result in results if result.target_path_px) >= 5


def test_traffic_light_flicker_hold_under_unknown():
    payload = _load_fixture("red_light_sequence.json")
    frames = payload["frames"]
    frames[6]["detections"] = []
    frames[7]["detections"] = []
    pipeline = PlanningPipeline({"traffic_light": {"red_enter_frames": 2, "unknown_hold_frames": 2}})
    results = [pipeline.plan(planning_input) for planning_input in adapt_perception_document(payload)]
    assert results[5].behavior == Behavior.STOP_FOR_RED
    assert results[6].behavior == Behavior.STOP_FOR_RED


def test_lead_id_switch_does_not_crash():
    results = _run_pipeline(_load_fixture("id_switch.json"))
    assert len(results) == 4
    assert all(result.behavior in {Behavior.FOLLOW_LEAD, Behavior.KEEP_LANE, Behavior.CAUTION} for result in results)


def test_distance_noise_still_warns_on_close_lead():
    payload = _load_fixture("lead_close.json")
    noisy = copy.deepcopy(payload)
    for frame in noisy["frames"]:
        for detection in frame["detections"]:
            detection["distance_m"] = float(detection["distance_m"]) + 1.5
    results = _run_pipeline(noisy)
    assert any(result.behavior == Behavior.FOLLOW_LEAD for result in results)
    assert any(any(warning.code == "FOLLOW_DISTANCE" for warning in result.warnings) for result in results)


def test_empty_detections_and_lanes_fail_soft():
    payload = {
        "video": {"width": 640, "height": 360, "fps": 10.0},
        "frames": [
            {"frame_index": 0, "timestamp_ms": 0, "lanes": {"lines": []}, "detections": []},
            {"frame_index": 1, "timestamp_ms": 100, "lanes": {"lines": []}, "detections": []},
        ],
    }
    results = _run_pipeline(payload)
    assert len(results) == 2
    assert all(0.0 <= result.confidence <= 1.0 for result in results)
