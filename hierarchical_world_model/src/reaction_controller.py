"""Post-HiQR causal reaction controllers.

The HiQR decoder remains a frozen producer of a one-frame background action.
Controllers in this module are applied *after* that action has been produced
and before the physics backend advances.  In particular, they never receive
the ego action passed to the current ``advance_response`` call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import nn

from .reference import response_relevance
from .rule_models import RuleModelBundle


ControllerMode = Literal[
    "none", "handcrafted", "rl_residual", "rl_residual_idm",
    "idm_only", "calibrated_residual",
]
# One second of realized relative history, committed ego controls, nominal
# action/reference/event scalars, fixed-slot role, and six *causal* authority
# scalars (gap, closing speed, TTC, phase age).
REACTION_FEATURE_DIM = 25 * 6 + 5 + 3 + 1 + 6 + 6


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
    # Executed HighwayEnv background actions from the preceding tick.  This
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


@dataclass(frozen=True)
class ReactionControllerOutput:
    actions: torch.Tensor
    alpha: torch.Tensor
    delta_ax: torch.Tensor
    active: torch.Tensor
    log_prob: torch.Tensor | None = None
    entropy: torch.Tensor | None = None
    value: torch.Tensor | None = None
    raw_action: torch.Tensor | None = None
    rule_action_ax: torch.Tensor | None = None
    policy_features: torch.Tensor | None = None
    desired_action_ax: torch.Tensor | None = None


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
        effective = context.intervention_trigger.abs() + cfg.intervention_trigger_threshold_mps2
        desired = selected * effective[:, None]
        horizon, decay = 20, float(cfg.intervention_memory_decay)
        kernel_average = (1.0 - decay**horizon) / (horizon * (1.0 - decay))
        expected = context.intervention_trigger.abs()[:, None] * relevance * kernel_average
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
            context.lateral_intervention_memory.abs() / cfg.lateral_response_trigger_scale_rps
        ).clamp(max=1.0)
        correction[..., 1] = (
            -torch.sign(context.lateral_intervention_memory)[:, None, None]
            * strength[:, None, None]
            * relevance[:, None]
            * side[:, None]
            * cfg.lateral_response_adapter_max_yaw_rate_rps
        )
    result = actions + correction
    result[..., 0] = result[..., 0].clamp(cfg.min_acceleration_mps2, cfg.max_acceleration_mps2)
    result[..., 1] = result[..., 1].clamp(-cfg.max_yaw_rate_rps, cfg.max_yaw_rate_rps)
    return result * context.current_valid[:, None, 1:, None].float()


class ReactionController(nn.Module):
    """Base class for controller modes applied after frozen HiQR inference."""

    mode: ControllerMode = "none"

    def forward(
        self, context: ReactionControllerContext, *, deterministic: bool = False
    ) -> ReactionControllerOutput:
        raise NotImplementedError


def _policy_sample(
    distribution: torch.distributions.Normal,
    context: ReactionControllerContext,
    deterministic: bool,
) -> torch.Tensor:
    if deterministic:
        return distribution.mean
    if context.policy_standard_normal is None:
        return distribution.rsample()
    return distribution.mean + distribution.stddev * context.policy_standard_normal.to(distribution.mean)


class NoReactionController(ReactionController):
    mode = "none"

    def forward(self, context: ReactionControllerContext, *, deterministic: bool = False) -> ReactionControllerOutput:
        del deterministic
        batch = context.base_actions.shape[0]
        zeros = context.base_actions.new_zeros((batch, 6))
        return ReactionControllerOutput(
            actions=context.base_actions,
            alpha=zeros,
            delta_ax=zeros,
            active=torch.zeros_like(zeros, dtype=torch.bool),
        )


class HandcraftedReactionController(ReactionController):
    mode = "handcrafted"

    def __init__(self, adapter_logit: torch.Tensor | float) -> None:
        super().__init__()
        self.register_buffer("adapter_logit", torch.as_tensor(adapter_logit).detach().clone())

    def forward(self, context: ReactionControllerContext, *, deterministic: bool = False) -> ReactionControllerOutput:
        del deterministic
        # Copy the immutable context so the frozen legacy scalar remains
        # explicit in the execution contract.
        values = {name: getattr(context, name) for name in context.__dataclass_fields__}
        values["adapter_gain"] = torch.sigmoid(self.adapter_logit).to(context.base_actions)
        actions = apply_handcrafted_response(context.base_actions, ReactionControllerContext(**values))
        delta = actions[..., 0].mean(1) - context.base_actions[..., 0].mean(1)
        zeros = torch.zeros_like(delta)
        return ReactionControllerOutput(
            actions=actions,
            alpha=zeros,
            delta_ax=delta,
            active=context.current_valid[:, 1:],
        )


def controller_features(context: ReactionControllerContext) -> torch.Tensor:
    """Return one fixed-size causal feature vector per background slot."""
    history = context.history[:, -25:]
    valid = context.history_valid[:, -25:]
    if history.shape[1] < 25:
        padding = history.new_zeros((len(history), 25 - history.shape[1], 7, 6))
        history = torch.cat((padding, history), dim=1)
        valid = torch.cat((valid.new_zeros((len(valid), 25 - valid.shape[1], 7)), valid), dim=1)
    background = history[:, :, 1:]
    if context.influence_parent is None:
        parent_history = history[:, :, :1].expand_as(background)
    else:
        parent_index = context.influence_parent.clamp_min(0)
        gather_index = parent_index[:, None, :, None].expand(-1, history.shape[1], -1, 6)
        parent_history = torch.gather(history, 2, gather_index)
    relative = torch.stack(
        (
            (background[..., 0] - parent_history[..., 0]) / 100.0,
            (background[..., 1] - parent_history[..., 1]) / 10.0,
            (background[..., 2] - parent_history[..., 2]) / 30.0,
            (background[..., 3] - parent_history[..., 3]) / 10.0,
            background[..., 4] / 8.0,
            parent_history[..., 4] / 8.0,
        ),
        dim=-1,
    ) * valid[:, :, 1:, None].float()
    temporal = relative.permute(0, 2, 1, 3).reshape(len(history), 6, -1)
    controls = context.committed_ego_controls[:, -5:, 0]
    if controls.shape[1] < 5:
        controls = torch.cat((controls.new_zeros((len(controls), 5 - controls.shape[1])), controls), dim=1)
    controls = controls[:, None].expand(-1, 6, -1) / 8.0
    relevance = response_relevance(context.current, context.current_valid)[..., None]
    scalars = torch.cat(
        (
            context.base_actions[:, 0, :, :1] / 8.0,
            context.reference_actions[:, 0, :, :1] / 8.0,
            context.intervention_memory[:, None, None].expand(-1, 6, -1) / 4.0,
        ),
        dim=-1,
    )
    if context.previous_background_actions is None:
        previous_ax = context.base_actions[:, 0, :, :1]
    else:
        previous_ax = context.previous_background_actions[:, :, :1]
    # These quantities are obtained from the state that HighwayEnv has
    # already realized.  They describe why authority remains active after an
    # ADS action ends; neither contains the current/future ego command.
    background_now = context.current[:, 1:]
    if context.influence_parent is None:
        parent_now = context.current[:, :1].expand_as(background_now)
    else:
        parent_now = torch.gather(
            context.current,
            1,
            context.influence_parent.clamp_min(0)[..., None].expand(-1, -1, 6),
        )
    gap = parent_now[..., 0] - background_now[..., 0] - 4.8
    closing = background_now[..., 2] - parent_now[..., 2]
    if context.influence_predicted_ttc_s is None:
        rear_ttc = torch.where(
            (gap > 0.0) & (closing > 1.0e-4),
            gap / closing.clamp_min(1.0e-4),
            torch.full_like(gap, 10.0),
        ).clamp(0.0, 10.0)
    else:
        rear_ttc = context.influence_predicted_ttc_s.clamp(0.0, 10.0)
    if context.reaction_phase is None:
        phase = torch.zeros_like(gap)
    else:
        phase = context.reaction_phase.to(gap).clamp(0, 2) / 2.0
        if phase.ndim == 1:
            phase = phase[:, None].expand_as(gap)
    if context.reaction_age_frames is None:
        age = torch.zeros_like(gap)
    else:
        age = context.reaction_age_frames.to(gap).clamp_min(0) / float(max(context.reaction_max_frames, 1))
        if age.ndim == 1:
            age = age[:, None].expand_as(gap)
    if context.reaction_recovery_remaining is None:
        recovery_remaining = torch.zeros_like(gap)
    else:
        recovery_remaining = context.reaction_recovery_remaining.to(gap).clamp_min(0) / float(max(context.reaction_recovery_frames, 1))
        if recovery_remaining.ndim == 1:
            recovery_remaining = recovery_remaining[:, None].expand_as(gap)
    authority = torch.stack(
        (gap.clamp(-10.0, 60.0) / 60.0, closing.clamp(-15.0, 25.0) / 25.0, rear_ttc / 10.0, phase, age, recovery_remaining),
        dim=-1,
    )
    if context.influence_role is None:
        roles = torch.eye(6, device=history.device, dtype=history.dtype)[None].expand(len(history), -1, -1)
    else:
        roles = torch.nn.functional.one_hot(context.influence_role.clamp(0, 5), num_classes=6).to(history.dtype)
    return torch.cat((
        temporal,
        controls,
        scalars * torch.cat((relevance, relevance, torch.ones_like(relevance)), dim=-1),
        previous_ax / 8.0,
        authority,
        roles,
    ), dim=-1)


def unresolved_following_brake_guard(
    final_ax: torch.Tensor,
    context: ReactionControllerContext,
    *,
    target_slot_index: int,
) -> torch.Tensor:
    """Forbid a HiQR acceleration rebound while a realized risk is unresolved.

    This is a controller-independent kinematic safety guard shared by the
    frozen transfer baselines.  It is deliberately not an IDM reference.
    It uses only the last committed ego acceleration and the state already
    realized by HighwayEnv.  In particular it cannot see a pending ego action.
    """
    if context.reaction_phase is None:
        return final_ax
    ego = context.current[:, 0]
    rear = context.current[:, target_slot_index + 1]
    gap = ego[:, 0] - rear[:, 0]
    closing = rear[:, 2] - ego[:, 2]
    same_lane = (ego[:, 1] - rear[:, 1]).abs() < 1.8
    ttc = torch.where(
        same_lane & (gap > 0.0) & (closing > 1.0e-4),
        gap / closing.clamp_min(1.0e-4),
        torch.full_like(gap, float("inf")),
    )
    # Phase 2 is a zero-authority monitoring latch after a causally observed
    # event.  It must still guard a risk that reappears before the next state
    # machine update can promote it back to phase 1.
    unresolved = context.reaction_phase.ne(0) & (ttc < float(context.reaction_release_ttc_s))
    if not unresolved.any():
        return final_ax

    # Constant-relative-deceleration bound for a 2 m residual clearance.  It
    # is only a no-rebound guard: PPO may choose a stronger legal brake.
    available_gap = (gap - 2.0).clamp_min(0.5)
    ego_ax = context.committed_ego_controls[:, -1, 0]
    required_ax = ego_ax - closing.square() / (2.0 * available_gap)
    required_ax = required_ax.clamp(context.cfg.min_acceleration_mps2, context.cfg.max_acceleration_mps2)
    base_ax = context.base_actions[:, 0, target_slot_index, 0]
    # An unresolved closing relation must never hand back a positive HiQR
    # acceleration.  Zero is a conservative upper bound; the learned policy
    # may select a stronger legal brake.
    safe_cap = torch.minimum(torch.minimum(base_ax, required_ax), torch.zeros_like(base_ax))
    guarded = final_ax.clone()
    guarded[:, target_slot_index] = torch.where(
        unresolved,
        torch.minimum(final_ax[:, target_slot_index], safe_cap),
        final_ax[:, target_slot_index],
    )
    return guarded


def unresolved_dynamic_brake_guard(
    final_ax: torch.Tensor,
    context: ReactionControllerContext,
    *,
    base_ax: torch.Tensor,
) -> torch.Tensor:
    """Apply the no-rebound safety cap to every dynamic graph follower.

    The legacy guard addressed one fixed same-rear slot.  With a dynamic
    influence graph, recovery authority can decay independently for several
    direct/secondary followers; handing any of them back to HiQR while its
    realised parent is still closing could reintroduce a positive-acceleration
    rebound.  This vectorized guard uses only the current realized states and
    previously submitted actions, and therefore remains causal.
    """
    if context.influence_parent is None or context.reaction_phase is None:
        return final_ax
    current = context.current
    child = current[:, 1:]
    parent_index = context.influence_parent.clamp_min(0)
    parent = torch.gather(current, 1, parent_index[..., None].expand(-1, -1, current.shape[-1]))
    gap = parent[..., 0] - child[..., 0] - 4.8
    closing = child[..., 2] - parent[..., 2]
    same_lane = (parent[..., 1] - child[..., 1]).abs() < 1.8
    ttc = torch.where(
        same_lane & (gap > 0.0) & (closing > 1.0e-4),
        gap / closing.clamp_min(1.0e-4), torch.full_like(gap, float("inf")),
    )
    phase = context.reaction_phase
    unresolved = phase.ne(0) & (ttc < float(context.reaction_release_ttc_s))
    if not unresolved.any():
        return final_ax
    parent_ax = context.committed_ego_controls[:, -1, 0][:, None].expand_as(gap)
    if context.previous_background_actions is not None:
        previous = context.previous_background_actions[..., 0]
        non_ego = (parent_index - 1).clamp_min(0)
        previous_parent = torch.gather(previous, 1, non_ego)
        parent_ax = torch.where(parent_index > 0, previous_parent, parent_ax)
    available_gap = (gap - 2.0).clamp_min(0.5)
    required_ax = parent_ax - closing.square() / (2.0 * available_gap)
    required_ax = required_ax.clamp(context.cfg.min_acceleration_mps2, context.cfg.max_acceleration_mps2)
    safe_cap = torch.minimum(torch.minimum(base_ax, required_ax), torch.zeros_like(base_ax))
    brake_role = (context.influence_role == 1) | (context.influence_role == 4)
    mask = unresolved & brake_role
    return torch.where(mask, torch.minimum(final_ax, safe_cap), final_ax)


def jerk_limited_reaction_action(
    desired_ax: torch.Tensor,
    context: ReactionControllerContext,
    *,
    target_slot_index: int,
) -> torch.Tensor:
    """Execute a causal reaction target through a stateful jerk-limited layer."""
    if context.previous_background_actions is None or context.reaction_phase is None:
        return desired_ax
    prior = context.previous_background_actions[:, target_slot_index, 0]
    phase = context.reaction_phase
    armed = phase.ne(0)
    ego, rear = context.current[:, 0], context.current[:, target_slot_index + 1]
    gap = ego[:, 0] - rear[:, 0]
    closing = rear[:, 2] - ego[:, 2]
    same_lane = (ego[:, 1] - rear[:, 1]).abs() < 1.8
    ttc = torch.where(
        same_lane & (gap > 0.) & (closing > 1.e-4),
        gap / closing.clamp_min(1.e-4),
        torch.full_like(gap, float("inf")),
    )
    unresolved = armed & (ttc < float(context.reaction_release_ttc_s))
    # Nominal braking jerk is 20 m/s^3 and rises continuously to 60 only at
    # realized TTC <= 1 s.  Release is capped at 8 m/s^3.
    urgency = ((4.0 - ttc) / 3.0).clamp(0., 1.)
    brake_step = (20.0 + 40.0 * urgency) * .04
    release_step = desired_ax.new_full((len(desired_ax),), 8.0 * .04)
    target = desired_ax[:, target_slot_index]
    executed_target = prior + (target - prior).clamp(-brake_step, release_step)
    # An already-observed risk may never receive positive acceleration.
    executed_target = torch.where(unresolved, torch.minimum(executed_target, torch.zeros_like(executed_target)), executed_target)
    result = desired_ax.clone()
    result[:, target_slot_index] = torch.where(armed, executed_target, target)
    return result.clamp(context.cfg.min_acceleration_mps2, context.cfg.max_acceleration_mps2)


def dynamic_safety_and_jerk_guard(
    desired_ax: torch.Tensor,
    context: ReactionControllerContext,
) -> torch.Tensor:
    """Apply per-agent follower safety caps and a stateful jerk envelope."""
    if context.influence_authority is None or context.influence_role is None:
        return desired_ax
    from .influence_graph import ROLE_SAME_LANE_FOLLOWER, ROLE_SECONDARY_FOLLOWER

    active = context.influence_authority > 0.0
    following = (context.influence_role == ROLE_SAME_LANE_FOLLOWER) | (
        context.influence_role == ROLE_SECONDARY_FOLLOWER
    )
    ttc = (
        torch.full_like(desired_ax, float("inf"))
        if context.influence_predicted_ttc_s is None
        else context.influence_predicted_ttc_s
    )
    unresolved = active & following & (ttc < float(context.reaction_safety_ttc_s))
    guarded = torch.where(unresolved, torch.minimum(desired_ax, torch.zeros_like(desired_ax)), desired_ax)
    if context.previous_background_actions is None:
        return guarded.clamp(context.cfg.min_acceleration_mps2, context.cfg.max_acceleration_mps2)
    prior = context.previous_background_actions[..., 0]
    urgency = ((float(context.reaction_safety_ttc_s) - ttc) / max(float(context.reaction_safety_ttc_s) - .25, 1.0e-3)).clamp(0.0, 1.0)
    brake_step = (20.0 + 40.0 * urgency) * float(context.cfg.dt_s)
    release_step = torch.full_like(brake_step, 8.0 * float(context.cfg.dt_s))
    executed = prior + (guarded - prior).clamp(-brake_step, release_step)
    executed = torch.where(
        unresolved,
        torch.minimum(torch.minimum(executed, prior), torch.zeros_like(executed)),
        executed,
    )
    return torch.where(active, executed, desired_ax).clamp(
        context.cfg.min_acceleration_mps2,
        context.cfg.max_acceleration_mps2,
    )


def dynamic_bounded_action_distribution(
    context: ReactionControllerContext,
    *,
    base: torch.Tensor,
    raw: torch.Tensor,
    authority: torch.Tensor,
    active: torch.Tensor,
    nominal_lower: torch.Tensor,
    nominal_upper: torch.Tensor,
    decreasing: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample an action whose support already satisfies safety and jerk.

    Returns final action, gate, effective target bounds, executable action
    bounds and the safety-imposed minimum gate. Because the
    bounds are inside the sigmoid transform, the executed action retains a
    continuous density for the frozen transfer-policy action mapping.
    """
    alpha = torch.sigmoid(raw[..., 0]) * authority * active.float()
    lower, upper = nominal_lower.clone(), nominal_upper.clone()
    execute_lower = torch.full_like(base, context.cfg.min_acceleration_mps2)
    execute_upper = torch.full_like(base, context.cfg.max_acceleration_mps2)
    minimum_gate = torch.zeros_like(base)
    if context.previous_background_actions is not None:
        prior = context.previous_background_actions[..., 0]
        ttc = (
            torch.full_like(base, float("inf"))
            if context.influence_predicted_ttc_s is None
            else context.influence_predicted_ttc_s
        )
        urgency = ((float(context.reaction_release_ttc_s) - ttc) /
            max(float(context.reaction_release_ttc_s) - 1.0, 1.e-3)).clamp(0., 1.)
        brake_step = (20.0 + 40.0 * urgency) * float(context.cfg.dt_s)
        execute_lower = (prior - brake_step).clamp_min(context.cfg.min_acceleration_mps2)
        execute_upper = (prior + 8.0 * float(context.cfg.dt_s)).clamp_max(context.cfg.max_acceleration_mps2)
        if context.influence_role is not None:
            from .influence_graph import ROLE_SAME_LANE_FOLLOWER, ROLE_SECONDARY_FOLLOWER
            following = (context.influence_role == ROLE_SAME_LANE_FOLLOWER) | (
                context.influence_role == ROLE_SECONDARY_FOLLOWER
            )
            unresolved = active & following & (ttc < float(context.reaction_release_ttc_s))
            execute_upper = torch.where(
                unresolved,
                torch.minimum(torch.minimum(execute_upper, prior), torch.zeros_like(execute_upper)),
                execute_upper,
            )
            # Ensure the gate is large enough that the legal minimum target
            # can reach the unresolved-risk upper cap.  This prevents a small
            # sampled gate from reintroducing positive HiQR acceleration.
            required = ((base - execute_upper) / (base - nominal_lower).clamp_min(1.e-4)).clamp(0., 1.)
            minimum_gate = torch.where(unresolved, required, minimum_gate)
            alpha = torch.where(unresolved, torch.maximum(alpha, required), alpha)
        safe_alpha = alpha.clamp_min(1.e-4)
        inverse_lower = (execute_lower - (1.0 - safe_alpha) * base) / safe_alpha
        inverse_upper = (execute_upper - (1.0 - safe_alpha) * base) / safe_alpha
        lower = torch.maximum(lower, inverse_lower)
        upper = torch.minimum(upper, inverse_upper)
    # Empty intersections can arise only from an already-invalid preceding
    # actuator state. Collapse continuously to the nearest legal endpoint;
    # the invalid-state reward still records the episode.
    feasible = lower <= upper
    midpoint = ((lower + upper) * .5).clamp(
        context.cfg.min_acceleration_mps2, context.cfg.max_acceleration_mps2
    )
    lower = torch.where(feasible, lower, midpoint)
    upper = torch.where(feasible, upper, midpoint)
    span = (upper - lower).clamp_min(1.e-5)
    q = torch.sigmoid(raw[..., 1])
    target = torch.where(decreasing, upper - span * q, lower + span * q)
    final = (1.0 - alpha) * base + alpha * target
    final = torch.where(active, final, base)
    return final, alpha, lower, upper, execute_lower, execute_upper, minimum_gate


class RLResidualReactionController(ReactionController):
    """Shared per-vehicle actor/critic for longitudinal residual PPO."""

    mode = "rl_residual"

    def __init__(
        self,
        hidden_dim: int = 128,
        relevance_threshold: float = 0.10,
        target_slot_index: int = 1,
        brake_trigger_threshold_mps2: float = 0.5,
    ) -> None:
        super().__init__()
        self.relevance_threshold = float(relevance_threshold)
        self.target_slot_index = int(target_slot_index)
        self.brake_trigger_threshold_mps2 = float(brake_trigger_threshold_mps2)
        if not 0 <= self.target_slot_index < 6:
            raise ValueError("target_slot_index must address one of six background slots")
        self.actor = nn.Sequential(nn.Linear(REACTION_FEATURE_DIM, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim), nn.SiLU())
        self.actor_mean = nn.Linear(hidden_dim, 2)
        self.actor_log_std = nn.Parameter(torch.full((2,), -0.7))
        self.critic = nn.Sequential(nn.Linear(REACTION_FEATURE_DIM, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))

    def distribution_and_value(self, features: torch.Tensor) -> tuple[torch.distributions.Normal, torch.Tensor]:
        hidden = self.actor(features)
        mean = self.actor_mean(hidden)
        std = self.actor_log_std.exp().expand_as(mean)
        return torch.distributions.Normal(mean, std), self.critic(features).squeeze(-1)

    def evaluate_raw_action(self, features: torch.Tensor, raw_action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution, value = self.distribution_and_value(features)
        return distribution.log_prob(raw_action).sum(-1), distribution.entropy().sum(-1), value

    def forward(self, context: ReactionControllerContext, *, deterministic: bool = False) -> ReactionControllerOutput:
        features = controller_features(context)
        distribution, value = self.distribution_and_value(features)
        raw = _policy_sample(distribution, context, deterministic)
        relevance = response_relevance(context.current, context.current_valid)
        if context.influence_authority is not None:
            authority = context.influence_authority
            active = context.current_valid[:, 1:] & (authority > 0.0)
        else:
            role = torch.zeros_like(relevance, dtype=torch.bool)
            role[:, self.target_slot_index] = True
            enabled = torch.ones_like(relevance, dtype=torch.bool) if context.reaction_enabled is None else context.reaction_enabled[:, None].bool()
            active = context.current_valid[:, 1:] & role & enabled & (relevance > self.relevance_threshold)
            if context.reaction_phase is not None and context.reaction_recovery_remaining is not None:
                recovery_scale = torch.where(
                    context.reaction_phase[:, None].eq(2),
                    (context.reaction_recovery_remaining[:, None].to(relevance) / float(max(context.reaction_recovery_frames, 1))).clamp(0.0, 1.0),
                    torch.ones_like(relevance),
                )
            else:
                recovery_scale = torch.ones_like(relevance)
            authority = active.float() * recovery_scale
        base = context.base_actions[:, 0, :, 0]
        if context.influence_role is None:
            alpha = torch.sigmoid(raw[..., 0]) * active.float() * authority
            delta = (context.cfg.min_acceleration_mps2 - base) * torch.sigmoid(raw[..., 1])
            final_ax = base + alpha * delta
            desired_ax = unresolved_following_brake_guard(
                final_ax, context, target_slot_index=self.target_slot_index
            )
            final_ax = jerk_limited_reaction_action(
                desired_ax, context, target_slot_index=self.target_slot_index
            )
        else:
            from .influence_graph import ROLE_SAME_LANE_FOLLOWER, ROLE_SECONDARY_FOLLOWER
            brake_only = (context.influence_role == ROLE_SAME_LANE_FOLLOWER) | (context.influence_role == ROLE_SECONDARY_FOLLOWER)
            nominal_lower = torch.full_like(base, context.cfg.min_acceleration_mps2)
            nominal_upper = torch.where(
                brake_only, base.clamp_max(context.cfg.max_acceleration_mps2),
                torch.full_like(base, context.cfg.max_acceleration_mps2),
            )
            final_ax, alpha, _, _, _, _, _ = dynamic_bounded_action_distribution(
                context, base=base, raw=raw, authority=authority,
                active=active, nominal_lower=nominal_lower,
                nominal_upper=nominal_upper, decreasing=brake_only,
            )
            desired_ax = final_ax
            safe_alpha = alpha.clamp_min(1.e-5)
            target = (final_ax - (1. - alpha) * base) / safe_alpha
            delta = torch.where(active, target - base, torch.zeros_like(base))
        actions = context.base_actions.clone()
        actions[:, 0, :, 0] = final_ax
        log_prob = distribution.log_prob(raw).sum(-1) * active.float()
        entropy = distribution.entropy().sum(-1) * active.float()
        return ReactionControllerOutput(actions, alpha, delta, active, log_prob, entropy, value, raw, desired_action_ax=desired_ax)


class IDMResidualReactionController(ReactionController):
    """Legacy A2-transfer residual controller with its frozen IDM mapping."""

    mode = "rl_residual_idm"

    def __init__(
        self, rule_model: RuleModelBundle,
        hidden_dim: int = 128, relevance_threshold: float = .10, target_slot_index: int = 1,
    ) -> None:
        super().__init__()
        self.rule_model = rule_model
        self.relevance_threshold, self.target_slot_index = float(relevance_threshold), int(target_slot_index)
        if not 0 <= self.target_slot_index < 6:
            raise ValueError("target_slot_index must address one of six background slots")
        feature_dim = REACTION_FEATURE_DIM + 1
        self.actor = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim), nn.SiLU())
        self.actor_mean = nn.Linear(hidden_dim, 2)
        self.actor_log_std = nn.Parameter(torch.full((2,), -.7))
        self.critic = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))

    def _features(self, context: ReactionControllerContext, rule_actions: torch.Tensor) -> torch.Tensor:
        return torch.cat((controller_features(context), rule_actions[:, :, None] / 8.), dim=-1)

    def distribution_and_value(self, features: torch.Tensor) -> tuple[torch.distributions.Normal, torch.Tensor]:
        hidden = self.actor(features)
        return torch.distributions.Normal(self.actor_mean(hidden), self.actor_log_std.exp().expand_as(self.actor_mean(hidden))), self.critic(features).squeeze(-1)

    def evaluate_raw_action(self, features: torch.Tensor, raw_action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution, value = self.distribution_and_value(features)
        return distribution.log_prob(raw_action).sum(-1), distribution.entropy().sum(-1), value

    def forward(self, context: ReactionControllerContext, *, deterministic: bool = False) -> ReactionControllerOutput:
        rule_actions = context.base_actions.new_zeros((len(context.current), 6))
        if context.influence_role is None:
            rule_target, _ = self.rule_model.idm_reference(
                context.history, context.current, context.current_valid, target_slot_index=self.target_slot_index,
                min_acceleration=context.cfg.min_acceleration_mps2, max_acceleration=context.cfg.max_acceleration_mps2,
            )
            rule_actions[:, self.target_slot_index] = rule_target
        else:
            from .influence_graph import ROLE_SAME_LANE_FOLLOWER
            for slot in range(6):
                candidate, _ = self.rule_model.idm_reference(
                    context.history, context.current, context.current_valid, target_slot_index=slot,
                    min_acceleration=context.cfg.min_acceleration_mps2, max_acceleration=context.cfg.max_acceleration_mps2,
                )
                rule_actions[:, slot] = torch.where(
                    context.influence_role[:, slot] == ROLE_SAME_LANE_FOLLOWER,
                    candidate,
                    context.base_actions[:, 0, slot, 0],
                )
        features = self._features(context, rule_actions)
        distribution, value = self.distribution_and_value(features)
        raw = _policy_sample(distribution, context, deterministic)
        relevance = response_relevance(context.current, context.current_valid)
        if context.influence_authority is not None:
            authority = context.influence_authority
            active = context.current_valid[:, 1:] & (authority > 0.0)
        else:
            role = torch.zeros_like(relevance, dtype=torch.bool); role[:, self.target_slot_index] = True
            enabled = torch.ones_like(relevance, dtype=torch.bool) if context.reaction_enabled is None else context.reaction_enabled[:, None].bool()
            active = context.current_valid[:, 1:] & role & enabled & (relevance > self.relevance_threshold)
            if context.reaction_phase is not None and context.reaction_recovery_remaining is not None:
                recovery_scale = torch.where(context.reaction_phase[:, None].eq(2),
                    (context.reaction_recovery_remaining[:, None].to(relevance) / float(max(context.reaction_recovery_frames, 1))).clamp(0., 1.), torch.ones_like(relevance))
            else:
                recovery_scale = torch.ones_like(relevance)
            authority = active.float() * recovery_scale
        base = context.base_actions[:, 0, :, 0]
        if context.influence_role is None:
            alpha = torch.sigmoid(raw[..., 0]) * active.float() * authority
            delta = (context.cfg.min_acceleration_mps2 - rule_actions) * torch.sigmoid(raw[..., 1])
            final_ax = ((1. - alpha) * base + alpha * (rule_actions + delta)).clamp(
                context.cfg.min_acceleration_mps2, context.cfg.max_acceleration_mps2
            )
            nominal_lower = nominal_upper = decreasing = None
        else:
            from .influence_graph import ROLE_SAME_LANE_FOLLOWER, ROLE_SECONDARY_FOLLOWER
            brake_only = (context.influence_role == ROLE_SAME_LANE_FOLLOWER) | (context.influence_role == ROLE_SECONDARY_FOLLOWER)
            nominal_lower = torch.full_like(base, context.cfg.min_acceleration_mps2)
            nominal_upper = torch.where(
                brake_only,
                torch.minimum(base, rule_actions).clamp_min(context.cfg.min_acceleration_mps2),
                torch.full_like(base, context.cfg.max_acceleration_mps2),
            )
            decreasing = brake_only
            (
                final_ax, alpha, nominal_lower, nominal_upper,
                execute_lower, execute_upper, minimum_gate,
            ) = dynamic_bounded_action_distribution(
                context, base=base, raw=raw, authority=authority,
                active=active, nominal_lower=nominal_lower,
                nominal_upper=nominal_upper, decreasing=decreasing,
            )
            safe_alpha = alpha.clamp_min(1.e-5)
            target_action = (final_ax - (1. - alpha) * base) / safe_alpha
            delta = torch.where(active, target_action - rule_actions, torch.zeros_like(base))
        # This mapping is retained only for the frozen A2-transfer baseline.
        # The calibrated policy below does not use IDM as an action bound.
        ego, rear = context.current[:, 0], context.current[:, self.target_slot_index + 1]
        gap = ego[:, 0] - rear[:, 0]
        closing = rear[:, 2] - ego[:, 2]
        same_lane = (ego[:, 1] - rear[:, 1]).abs() < 1.8
        # Four seconds is an anticipatory following-risk region, not a
        # post-impact trigger.  This matches the highD prior sampling split
        # between ordinary following and compressed/TTC-critical contexts.
        emergency = same_lane & (gap > 0.) & (closing > 1.e-4) & (gap / closing.clamp_min(1.e-4) < 4.0)
        target = self.target_slot_index
        if context.influence_authority is None:
            final_ax[:, target] = torch.where(emergency & active[:, target], torch.minimum(final_ax[:, target], rule_actions[:, target]), final_ax[:, target])
        # Dynamic same-lane bounds above already include the IDM reference;
        # cut-in and secondary roles remain pure residual by construction.
        actions = context.base_actions.clone(); actions[:, 0, :, 0] = final_ax
        if context.influence_authority is None:
            desired_ax = unresolved_following_brake_guard(final_ax, context, target_slot_index=self.target_slot_index)
            final_ax = jerk_limited_reaction_action(desired_ax, context, target_slot_index=self.target_slot_index)
        else:
            desired_ax = unresolved_dynamic_brake_guard(final_ax, context, base_ax=base)
            if context.previous_background_actions is not None:
                previous_ax = context.previous_background_actions[..., 0]
                max_step = float(getattr(context.cfg, "jerk_limit_mps3", 60.0)) * float(context.cfg.dt_s)
                limited = torch.maximum(torch.minimum(desired_ax, previous_ax + max_step), previous_ax - max_step)
                # No-authority vehicles are an exact HiQR passthrough, even
                # when the historical HiQR action itself has a large jerk.
                final_ax = torch.where(active, limited, base)
            final_ax = torch.where(active, final_ax.clamp(
                context.cfg.min_acceleration_mps2, context.cfg.max_acceleration_mps2), base)
        actions[:, 0, :, 0] = final_ax
        log_prob = distribution.log_prob(raw).sum(-1) * active.float()
        entropy = distribution.entropy().sum(-1) * active.float()
        return ReactionControllerOutput(
            actions, alpha, delta, active, log_prob, entropy, value, raw,
            rule_action_ax=rule_actions, policy_features=features,
            desired_action_ax=desired_ax,
        )


def _dynamic_idm_reference(
    rule_model: RuleModelBundle, context: ReactionControllerContext,
) -> torch.Tensor:
    """Compute a causal IDM feature for each currently scoped follower."""
    from .influence_graph import ROLE_SAME_LANE_FOLLOWER

    reference = context.base_actions[:, 0, :, 0].clone()
    if context.influence_role is None:
        return reference
    for slot in range(6):
        candidate, _ = rule_model.idm_reference(
            context.history, context.current, context.current_valid,
            target_slot_index=slot,
            min_acceleration=context.cfg.min_acceleration_mps2,
            max_acceleration=context.cfg.max_acceleration_mps2,
        )
        reference[:, slot] = torch.where(
            context.influence_role[:, slot] == ROLE_SAME_LANE_FOLLOWER,
            candidate,
            reference[:, slot],
        )
    return reference


class IDMOnlyReactionController(ReactionController):
    """A rule-only local response under the same autonomous scope as PPO."""

    mode = "idm_only"

    def __init__(self, rule_model: RuleModelBundle) -> None:
        super().__init__()
        self.rule_model = rule_model

    def forward(self, context: ReactionControllerContext, *, deterministic: bool = False) -> ReactionControllerOutput:
        del deterministic
        from .influence_graph import ROLE_SAME_LANE_FOLLOWER

        base = context.base_actions[:, 0, :, 0]
        rule = _dynamic_idm_reference(self.rule_model, context)
        authority = (
            torch.zeros_like(base)
            if context.influence_authority is None
            else context.influence_authority
        )
        following = (
            torch.zeros_like(base, dtype=torch.bool)
            if context.influence_role is None
            else context.influence_role == ROLE_SAME_LANE_FOLLOWER
        )
        active = context.current_valid[:, 1:] & following & authority.gt(0.0)
        desired = torch.where(active, rule, base)
        final = dynamic_safety_and_jerk_guard(desired, context)
        actions = context.base_actions.clone()
        actions[:, 0, :, 0] = torch.where(active, final, base)
        delta = actions[:, 0, :, 0] - base
        return ReactionControllerOutput(
            actions=actions,
            alpha=authority * active.float(),
            delta_ax=delta,
            active=active,
            rule_action_ax=rule,
            desired_action_ax=desired,
        )


class CalibratedResidualReactionController(ReactionController):
    """Small signed residual policy trained against observed final behaviour.

    IDM is an input feature only.  It never constrains this policy's nominal
    interval, so a safe HiQR action can remain unchanged and a learned
    correction can either brake or recover speed.
    """

    mode = "calibrated_residual"

    def __init__(self, rule_model: RuleModelBundle, hidden_dim: int = 128) -> None:
        super().__init__()
        self.rule_model = rule_model
        feature_dim = REACTION_FEATURE_DIM + 1
        self.actor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
        )
        # [authority preference, signed bounded residual]
        self.actor_mean = nn.Linear(hidden_dim, 2)
        nn.init.zeros_(self.actor_mean.weight)
        nn.init.zeros_(self.actor_mean.bias)
        self.actor_log_std = nn.Parameter(torch.full((2,), -0.7))
        self.critic = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1),
        )

    def _features(self, context: ReactionControllerContext, rule: torch.Tensor) -> torch.Tensor:
        return torch.cat((controller_features(context), rule[:, :, None] / 8.0), dim=-1)

    def distribution_and_value(self, features: torch.Tensor) -> tuple[torch.distributions.Normal, torch.Tensor]:
        hidden = self.actor(features)
        mean = self.actor_mean(hidden)
        return torch.distributions.Normal(mean, self.actor_log_std.exp().expand_as(mean)), self.critic(features).squeeze(-1)

    def evaluate_raw_action(self, features: torch.Tensor, raw_action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution, value = self.distribution_and_value(features)
        return distribution.log_prob(raw_action).sum(-1), distribution.entropy().sum(-1), value

    @staticmethod
    def mapped_action(
        base: torch.Tensor, authority: torch.Tensor, active: torch.Tensor,
        raw: torch.Tensor, minimum: float, maximum: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Map actor output to the final desired acceleration interval."""
        gate = torch.sigmoid(raw[..., 0]) * authority * active.float()
        signed = torch.tanh(raw[..., 1])
        span = torch.where(signed < 0.0, base - minimum, maximum - base)
        return base + gate * signed * span, gate

    def forward(self, context: ReactionControllerContext, *, deterministic: bool = False) -> ReactionControllerOutput:
        rule = _dynamic_idm_reference(self.rule_model, context)
        features = self._features(context, rule)
        distribution, value = self.distribution_and_value(features)
        raw = _policy_sample(distribution, context, deterministic)
        base = context.base_actions[:, 0, :, 0]
        authority = (
            torch.zeros_like(base)
            if context.influence_authority is None
            else context.influence_authority
        )
        from .influence_graph import ROLE_SAME_LANE_FOLLOWER, ROLE_SECONDARY_FOLLOWER
        same_lane = (
            torch.zeros_like(authority, dtype=torch.bool)
            if context.influence_role is None
            else (context.influence_role == ROLE_SAME_LANE_FOLLOWER)
            | (context.influence_role == ROLE_SECONDARY_FOLLOWER)
        )
        active = context.current_valid[:, 1:] & authority.gt(0.0) & same_lane
        desired, gate = self.mapped_action(
            base, authority, active, raw,
            context.cfg.min_acceleration_mps2,
            context.cfg.max_acceleration_mps2,
        )
        final = dynamic_safety_and_jerk_guard(desired, context)
        final = torch.where(active, final, base)
        actions = context.base_actions.clone()
        actions[:, 0, :, 0] = final
        return ReactionControllerOutput(
            actions=actions,
            alpha=gate,
            delta_ax=final - base,
            active=active,
            log_prob=distribution.log_prob(raw).sum(-1) * active.float(),
            entropy=distribution.entropy().sum(-1) * active.float(),
            value=value,
            raw_action=raw,
            rule_action_ax=rule,
            policy_features=features,
            desired_action_ax=desired,
        )


def make_reaction_controller(mode: ControllerMode, *, adapter_logit: torch.Tensor | float | None = None, **kwargs: Any) -> ReactionController:
    if mode == "none":
        return NoReactionController()
    if mode == "handcrafted":
        if adapter_logit is None:
            raise ValueError("handcrafted controller requires the frozen decoder intervention logit")
        return HandcraftedReactionController(adapter_logit)
    if mode == "rl_residual":
        return RLResidualReactionController(**kwargs)
    if mode == "rl_residual_idm":
        if "rule_model" not in kwargs:
            raise ValueError(f"{mode} requires a calibrated rule_model")
        return IDMResidualReactionController(**kwargs)
    if mode == "idm_only":
        return IDMOnlyReactionController(**kwargs)
    if mode == "calibrated_residual":
        return CalibratedResidualReactionController(**kwargs)
    raise ValueError(f"unknown reaction controller mode {mode!r}")
