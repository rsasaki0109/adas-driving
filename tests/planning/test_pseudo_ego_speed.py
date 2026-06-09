from __future__ import annotations

import json
from pathlib import Path

from adas_planning.ego.pseudo_speed import resolve_ego_speed
from adas_planning.io.perception_adapter import adapt_perception_document
from adas_planning.memory.track_history import TrackHistory
from adas_planning.metrics.offline import METRICS_SCHEMA_VERSION, compute_offline_metrics, write_metrics_artifact
from adas_planning.pipeline import PlanningPipeline
from adas_planning.planners.lead_follow import compute_lead_follow
from adas_planning.types import PlanningInput


def test_measurement_has_highest_priority():
    estimate = resolve_ego_speed(
        PlanningInput(
            frame_id=0,
            timestamp_s=0.0,
            image_width=640,
            image_height=360,
            ego_speed_mps=12.0,
        ),
        {"pseudo_ego_speed": {"default_mps": 8.0}},
        relative_velocity_mps=-5.0,
    )
    assert estimate.source == "measurement"
    assert estimate.speed_mps == 12.0
    assert estimate.confidence_factor == 1.0


def test_config_default_used_when_no_measurement():
    estimate = resolve_ego_speed(
        PlanningInput(frame_id=0, timestamp_s=0.0, image_width=640, image_height=360),
        {"pseudo_ego_speed": {"default_mps": 8.0, "config_confidence_factor": 0.65}},
    )
    assert estimate.source == "config_default"
    assert estimate.speed_mps == 8.0
    assert estimate.confidence_factor == 0.65


def test_closing_rate_used_when_no_measurement_or_config():
    estimate = resolve_ego_speed(
        PlanningInput(frame_id=0, timestamp_s=0.0, image_width=640, image_height=360),
        {
            "pseudo_ego_speed": {
                "default_mps": None,
                "closing_rate": {"enabled": True, "min_range_rate_mps": 0.5, "confidence_factor": 0.45},
            }
        },
        relative_velocity_mps=-2.0,
    )
    assert estimate.source == "closing_rate"
    assert estimate.speed_mps == 2.0


def test_lead_follow_emits_target_speed_with_config_default():
    payload = {
        "video": {"width": 640, "height": 360, "fps": 10.0},
        "frames": [
            {
                "frame_index": 0,
                "timestamp_ms": 0,
                "lanes": {"lines": []},
                "detections": [
                    {
                        "kind": "vehicle",
                        "label": "car",
                        "confidence": 0.9,
                        "track_id": 1,
                        "distance_m": 7.5,
                        "box": {"x1": 285, "y1": 210, "x2": 355, "y2": 270, "width": 70, "height": 60},
                    }
                ],
            }
        ],
    }
    config = {"lead_follow": {"warning_distance_m": 12.0, "critical_distance_m": 6.0}, "pseudo_ego_speed": {"default_mps": 10.0}}
    planning_input = adapt_perception_document(payload)[0]
    proposal = compute_lead_follow(planning_input, config, TrackHistory(config))
    assert proposal is not None
    assert proposal.target_speed_mps is not None
    assert proposal.debug["ego_speed_source"] == "config_default"
    assert proposal.confidence < 0.9


def test_closing_rate_fixture_produces_target_speed(tmp_path: Path):
    fixture_path = Path(__file__).resolve().parents[1] / "fixtures" / "planning" / "lead_closing.json"
    with fixture_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    pipeline = PlanningPipeline({"pseudo_ego_speed": {"default_mps": None, "closing_rate": {"enabled": True}}})
    results = [pipeline.plan(planning_input) for planning_input in adapt_perception_document(payload)]
    assert any(result.target_speed_mps is not None for result in results[2:])
    assert any(result.debug.get("ego_speed_source") == "closing_rate" for result in results[2:])


def test_metrics_artifact_has_schema_version(tmp_path: Path):
    metrics = compute_offline_metrics([])
    output = tmp_path / "metrics.json"
    write_metrics_artifact(output, metrics, source="test", config_path="cfg.yaml", config_hash="abc")
    with output.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    assert payload["schema_version"] == METRICS_SCHEMA_VERSION
    assert payload["config_hash"] == "abc"
