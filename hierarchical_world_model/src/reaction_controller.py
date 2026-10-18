"""Causal inputs and diagnostics for NPC ownership and maneuver execution."""

from dataclasses import dataclass
from enum import IntEnum
import torch
from .config import WorldModelConfig
from .influence_graph import DynamicInteractionRelations


class ManeuverPhase(IntEnum):
    TRACK = 0
    RESPOND = 1
    EXECUTE = 2
    REBASE = 3


@dataclass(frozen=True)
class ReactionControllerContext:
    history: torch.Tensor
    history_valid: torch.Tensor
    current: torch.Tensor
    current_valid: torch.Tensor
    base_actions: torch.Tensor
    cfg: WorldModelConfig
    vehicle_dimensions: torch.Tensor
    map_polylines: torch.Tensor
    map_polyline_valid: torch.Tensor
    interaction_relations: DynamicInteractionRelations
    previous_background_actions: torch.Tensor | None = None
    policy_standard_normal: torch.Tensor | None = None
    policy_initial_standard_normal: torch.Tensor | None = None
    planned_current_xy: torch.Tensor | None = None
    planned_terminal_xy: torch.Tensor | None = None
    planned_path_xy: torch.Tensor | None = None
    response_index: int = 0


@dataclass(frozen=True)
class ReactionControllerOutput:
    actions: torch.Tensor
    alpha: torch.Tensor
    delta_ax: torch.Tensor
    active: torch.Tensor
    rule_action_ax: torch.Tensor | None = None
    autonomous_lane_active: torch.Tensor | None = None
    mobil_incentive_mps2: torch.Tensor | None = None
    realized_leader_index: torch.Tensor | None = None
    takeover_started: torch.Tensor | None = None
    takeover_owned: torch.Tensor | None = None
    takeover_motion_changed: torch.Tensor | None = None
    takeover_reason: torch.Tensor | None = None
    takeover_score: torch.Tensor | None = None
    takeover_conflict_time_s: torch.Tensor | None = None
    takeover_influencer: torch.Tensor | None = None
    takeover_interaction_score: torch.Tensor | None = None
    takeover_following_score_s: torch.Tensor | None = None
    takeover_release_score_per_s: torch.Tensor | None = None
    takeover_interaction_time_s: torch.Tensor | None = None
