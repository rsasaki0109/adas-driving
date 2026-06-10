from __future__ import annotations

from adas_planning.memory.track_history import TrackHistory
from adas_planning.planners.vru_yield import compute_vru_yield
from adas_planning.types import DetectionInput, PlanningInput


def _pedestrian_input(frame_id: int, distance_m: float) -> PlanningInput:
    return PlanningInput(
        frame_id=frame_id,
        timestamp_s=float(frame_id) / 10.0,
        image_width=1280,
        image_height=720,
        detections=[
            DetectionInput(
                kind="pedestrian",
                label="person",
                confidence=0.9,
                box={"x1": 600, "y1": 400, "x2": 660, "y2": 600, "width": 60, "height": 200},
                track_id=7,
                distance_m=distance_m,
            )
        ],
    )


def test_vru_yield_emits_ttc_warning_for_closing_pedestrian():
    config = {
        "vru_yield": {"warning_distance_m": 25.0, "ttc_warning_s": 3.0},
        "pseudo_ego_speed": {"default_mps": 10.0},
    }
    history = TrackHistory(config)
    proposal = None
    # 2 m per 0.1 s => closing at 20 m/s, TTC at 12 m is 0.6 s.
    for frame_id, distance in enumerate([20.0, 18.0, 16.0, 14.0, 12.0]):
        proposal = compute_vru_yield(_pedestrian_input(frame_id, distance), config, history)
    assert proposal is not None
    assert proposal.debug["relative_velocity_mps"] is not None
    assert proposal.debug["relative_velocity_mps"] < -15.0
    assert proposal.debug["ttc_s"] is not None
    assert proposal.debug["ttc_s"] < 3.0
    ttc_warnings = [w for w in proposal.warnings if "TTC" in w.message]
    assert ttc_warnings
    # TTC-critical caps to ttc_speed_cap_ratio (0.25) of ego speed instead of 0.5.
    assert proposal.target_speed_mps == 10.0 * 0.25


def test_vru_yield_stationary_pedestrian_has_no_ttc():
    config = {"vru_yield": {"warning_distance_m": 25.0}}
    history = TrackHistory(config)
    proposal = None
    for frame_id in range(4):
        proposal = compute_vru_yield(_pedestrian_input(frame_id, 15.0), config, history)
    assert proposal is not None
    assert proposal.debug["ttc_s"] is None
    assert len([w for w in proposal.warnings if w.code == "YIELD_VRU"]) == 1


def test_vru_yield_works_without_track_history():
    proposal = compute_vru_yield(_pedestrian_input(0, 15.0), {"vru_yield": {}}, None)
    assert proposal is not None
    assert proposal.debug["ttc_s"] is None
