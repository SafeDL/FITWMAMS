"""Causal influence response policies applied after factual decoding.

The factual dynamics model remains a frozen producer of a one-frame background action.
Controllers in this module are applied *after* that action has been produced
and before the physics backend advances.  In particular, they never receive
the ego action passed to the current ``advance_response`` call.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import torch
from torch import nn

from .reference import response_relevance
from .influence_graph import DynamicInteractionRelations


class ManeuverPhase(IntEnum):
    TRACK = 0
    RESPOND = 1
    EXECUTE = 2
    REBASE = 3


@dataclass(frozen=True)
class ReactionControllerContext:
    """Causally available inputs at one background-response boundary."""

    history: torch.Tensor
    history_valid: torch.Tensor
    current: torch.Tensor
    current_valid: torch.Tensor
    committed_ego_controls: torch.Tensor
    base_actions: torch.Tensor
    reference_actions: torch.Tensor
    intervention_trigger: torch.Tensor
    intervention_memory: torch.Tensor
    lateral_intervention_memory: torch.Tensor
    agent_style_state: torch.Tensor
    response_field_gain: torch.Tensor | None
    response_sensitivity_bounds: torch.Tensor | None
    adapter_gain: torch.Tensor | None
    reaction_enabled: torch.Tensor | None
    cfg: Any
    # Executed background actions from the preceding tick.  This
    # is causal actuator state, never a preview of the action about to run.
    previous_background_actions: torch.Tensor | None = None
    reaction_phase: torch.Tensor | None = None
    reaction_age_frames: torch.Tensor | None = None
    reaction_max_frames: int = 1
    reaction_safety_ttc_s: float = 2.0
    # A release threshold, not an action-window duration.  While the
    # realized same-rear relation is closing inside this horizon, a triggered
    # controller retains authority irrespective of how long the ADS command
    # itself lasted.
    reaction_release_ttc_s: float = 4.0
    reaction_recovery_remaining: torch.Tensor | None = None
    reaction_recovery_frames: int = 1
    influence_authority: torch.Tensor | None = None
    influence_role: torch.Tensor | None = None
    influence_parent: torch.Tensor | None = None
    influence_direct: torch.Tensor | None = None
    influence_secondary: torch.Tensor | None = None
    influence_predicted_ttc_s: torch.Tensor | None = None
    influence_predicted_min_gap_m: torch.Tensor | None = None
    policy_standard_normal: torch.Tensor | None = None
    policy_initial_standard_normal: torch.Tensor | None = None
    nominal_ego_velocity: torch.Tensor | None = None
    previous_base_background_actions: torch.Tensor | None = None
    # Model-generated Diffusion plan only; never a simulated or logged future.
    planned_origin_xy: torch.Tensor | None = None
    planned_current_xy: torch.Tensor | None = None
    planned_terminal_xy: torch.Tensor | None = None
    planned_path_xy: torch.Tensor | None = None
    reference_retired: torch.Tensor | None = None
    map_polylines: torch.Tensor | None = None
    map_polyline_valid: torch.Tensor | None = None
    vehicle_dimensions: torch.Tensor | None = None
    interaction_relations: DynamicInteractionRelations | None = None
    # HiQR attention-encoded realised history, before any soft-preview conditioning.
    observation_tokens: torch.Tensor | None = None
    response_index: int = 0


@dataclass(frozen=True)
class ReactionControllerOutput:
    actions: torch.Tensor
    alpha: torch.Tensor
    delta_ax: torch.Tensor
    active: torch.Tensor
    rule_action_ax: torch.Tensor | None = None
    desired_action_ax: torch.Tensor | None = None
    causal_delta_ax: torch.Tensor | None = None
    policy_active: torch.Tensor | None = None
    spatial_path_active: torch.Tensor | None = None
    autonomous_lane_active: torch.Tensor | None = None
    autonomous_lane_initial_active: torch.Tensor | None = None
    data_driver_active: torch.Tensor | None = None
    execution_active: torch.Tensor | None = None
    release_active: torch.Tensor | None = None
    mobil_incentive_mps2: torch.Tensor | None = None
    mobil_selected_side: torch.Tensor | None = None
    response_mode: torch.Tensor | None = None
    response_gate: torch.Tensor | None = None
    response_gate_features: torch.Tensor | None = None
    realized_leader_index: torch.Tensor | None = None


def apply_handcrafted_response(
    actions: torch.Tensor,
    context: ReactionControllerContext,
) -> torch.Tensor:
    """Apply the legacy decoder adapter to an already decoded action tensor.

    This is the former adapter algebra verbatim, deliberately kept outside the
    decoder so controller modes are mutually exclusive at the execution hook.
    """
    cfg = context.cfg
    if not bool(cfg.intervention_adapter_enabled):
        return actions
    batch = actions.shape[0]
    relevance = response_relevance(context.current, context.current_valid)
    brake_limit = (
        cfg.intervention_adapter_max_gain
        if cfg.intervention_brake_adapter_max_gain is None
        else cfg.intervention_brake_adapter_max_gain
    )
    accelerate_limit = (
        cfg.intervention_adapter_max_gain
        if cfg.intervention_accelerate_adapter_max_gain is None
        else cfg.intervention_accelerate_adapter_max_gain
    )
    adapter_limit = torch.where(
        context.intervention_trigger[:, None] < 0.0,
        relevance.new_full((batch, 1), brake_limit),
        relevance.new_full((batch, 1), accelerate_limit),
    )
    # The scalar parameter lives on the frozen decoder for checkpoint
    # compatibility; callers provide its sigmoid through the context when
    # needed.  Normal released configurations have no response field.
    base_gain = context.adapter_gain
    if base_gain is None:
        raise ValueError("handcrafted reaction controller requires adapter_gain")
    if not isinstance(base_gain, torch.Tensor):
        base_gain = actions.new_tensor(float(base_gain))
    adapter_gain = adapter_limit[..., None] * base_gain
    if context.response_field_gain is not None:
        adapter_gain = adapter_gain * context.response_field_gain[:, None]
    sensitivity = context.response_sensitivity_bounds
    if (
        bool(cfg.natural_response_kernel_enabled)
        and sensitivity is not None
        and torch.isfinite(sensitivity).all()
    ):
        kind = (context.intervention_trigger >= 0.0).long()
        bounds = sensitivity.to(adapter_gain)[kind]
        quantile = torch.sigmoid(context.agent_style_state[:, 1:, 0])
        selected = bounds[:, :1] + quantile * (bounds[:, 1:] - bounds[:, :1])
        effective = (
            context.intervention_trigger.abs() + cfg.intervention_trigger_threshold_mps2
        )
        desired = selected * effective[:, None]
        horizon, decay = 20, float(cfg.intervention_memory_decay)
        kernel_average = (1.0 - decay**horizon) / (horizon * (1.0 - decay))
        expected = (
            context.intervention_trigger.abs()[:, None] * relevance * kernel_average
        )
        calibrated = desired / expected.clamp_min(1.0e-4)
        brake_limit = (
            cfg.natural_response_kernel_max_gain
            if cfg.natural_response_kernel_brake_max_gain is None
            else cfg.natural_response_kernel_brake_max_gain
        )
        accelerate_limit = (
            cfg.natural_response_kernel_max_gain
            if cfg.natural_response_kernel_accelerate_max_gain is None
            else cfg.natural_response_kernel_accelerate_max_gain
        )
        kernel_limit = torch.where(
            context.intervention_trigger[:, None] < 0.0,
            calibrated.new_full(calibrated.shape, brake_limit),
            calibrated.new_full(calibrated.shape, accelerate_limit),
        )
        adapter_gain = calibrated.clamp(min=0.0).minimum(kernel_limit)[:, None, :]
    correction = torch.zeros_like(actions)
    correction[..., 0] = (
        context.intervention_trigger[:, None, None] * relevance[:, None] * adapter_gain
    )
    if bool(cfg.lateral_response_adapter_enabled):
        side = torch.sign(context.current[:, 1:, 1] - context.current[:, :1, 1])
        side = torch.where(side == 0.0, torch.ones_like(side), side)
        strength = (
            context.lateral_intervention_memory.abs()
            / cfg.lateral_response_trigger_scale_rps
        ).clamp(max=1.0)
        correction[..., 1] = (
            -torch.sign(context.lateral_intervention_memory)[:, None, None]
            * strength[:, None, None]
            * relevance[:, None]
            * side[:, None]
            * cfg.lateral_response_adapter_max_yaw_rate_rps
        )
    result = actions + correction
    result = torch.stack(
        (
            result[..., 0].clamp(cfg.min_acceleration_mps2, cfg.max_acceleration_mps2),
            result[..., 1].clamp(-cfg.max_yaw_rate_rps, cfg.max_yaw_rate_rps),
        ),
        dim=-1,
    )
    return result * context.current_valid[:, None, 1:, None].float()


class ReactionController(nn.Module):
    """Base class for policies applied after frozen factual inference."""

    mode = "none"

    def snapshot_runtime(self) -> dict[str, torch.Tensor]:
        """Mutable episode state needed for exact closed-loop replay."""
        return {}

    def restore_runtime(self, state: dict[str, torch.Tensor]) -> None:
        if state:
            raise ValueError(f"{type(self).__name__} has no mutable episode state")

    def forward(
        self, context: ReactionControllerContext, *, deterministic: bool = False
    ) -> ReactionControllerOutput:
        raise NotImplementedError
