from __future__ import annotations

from typing import Any

from adas_planning.behavior.arbiter import arbitrate_proposals
from adas_planning.ego.pseudo_speed import estimate_from_debug, merge_ego_speed_estimates, resolve_ego_speed
from adas_planning.memory.scene_memory import SceneMemory
from adas_planning.memory.track_history import TrackHistory
from adas_planning.planners.lane_departure import LaneDeparturePlanner
from adas_planning.planners.lane_target import compute_lane_target
from adas_planning.planners.lead_follow import compute_lead_follow
from adas_planning.planners.traffic_light import TrafficLightFSM, compute_traffic_light
from adas_planning.planners.vru_yield import compute_vru_yield
from adas_planning.types import Behavior, PlanningInput, PlanningResult


class PlanningPipeline:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.memory = SceneMemory(config)
        self.track_history = TrackHistory(config)
        tl_cfg = config.get("traffic_light", {})
        self.traffic_light_fsm = TrafficLightFSM(
            red_enter_frames=int(tl_cfg.get("red_enter_frames", 2)),
            green_exit_frames=int(tl_cfg.get("green_exit_frames", 3)),
            unknown_hold_frames=int(tl_cfg.get("unknown_hold_frames", 2)),
            min_confidence=float(tl_cfg.get("min_confidence", 0.25)),
        )
        self.lane_departure = LaneDeparturePlanner()

    def plan(self, planning_input: PlanningInput) -> PlanningResult:
        lane_proposal = compute_lane_target(planning_input, self.config)
        lane_valid = lane_proposal.behavior == Behavior.KEEP_LANE and bool(lane_proposal.target_path)
        lane_proposal = self.memory.update_lane(lane_proposal, lane_valid=lane_valid)

        proposals = [
            compute_traffic_light(planning_input, self.config, self.traffic_light_fsm),
            compute_vru_yield(planning_input, self.config, self.track_history),
            compute_lead_follow(planning_input, self.config, self.track_history),
            self.lane_departure.compute(planning_input, self.config),
        ]

        result = arbitrate_proposals(proposals, lane_proposal=lane_proposal)
        result.frame_id = planning_input.frame_id
        result.timestamp_s = planning_input.timestamp_s
        module_estimates = [
            estimate
            for module in result.debug.get("modules", [])
            if (estimate := estimate_from_debug(module)) is not None
        ]
        ego_estimate = merge_ego_speed_estimates(resolve_ego_speed(planning_input, self.config), *module_estimates)
        result.debug["ego_speed_mps"] = ego_estimate.speed_mps
        result.debug["ego_speed_source"] = ego_estimate.source
        result.debug["ego_speed_confidence_factor"] = ego_estimate.confidence_factor
        if not lane_valid and result.behavior == Behavior.KEEP_LANE:
            result.behavior = Behavior.CAUTION
            result.confidence = min(result.confidence, 0.25)
        return result
