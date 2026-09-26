"""Causal single-rollout executor and factual metric adapters."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch

from diffusion.src.data import ANCHOR_INDEX
from traffic_components.src.core.highd_metrics import (
    factual_metrics as _shared_factual_metrics,
)
from traffic_components.src.core.highd_metrics import (
    temporal_factual_metrics as _shared_temporal_factual_metrics,
)
from traffic_components.src.core.evaluation_scope import scoped_canonical_trajectory

from .data import ego_controls
from .reaction_controller import (
    ReactionController,
    ReactionControllerContext,
    make_reaction_controller,
)
from .influence_graph import CausalInfluenceGraph, InfluenceGraphState
from .randomness import WorldExogenousState


@dataclass(frozen=True)
class Rollout:
    """Numpy rollout tensors and controls returned by the offline evaluator."""

    states: np.ndarray
    background_actions: np.ndarray
    ego_actions: np.ndarray
    reference_actions: np.ndarray
    base_background_actions: np.ndarray | None = None
    controller_diagnostics: dict[str, np.ndarray] | None = None
    snapshots: dict[int, "RolloutState"] | None = None


@dataclass(frozen=True)
class RolloutState:
    """Complete mutable state at one formal rollout decision boundary."""

    response_index: int
    states: torch.Tensor
    valid: torch.Tensor
    history: torch.Tensor
    history_valid: torch.Tensor
    filter_state: Any
    slow_scene: Any
    slow_scene_noise: Any
    agent_noise_state: Any
    agent_style_state: Any
    previous_current: Any
    committed_ego_controls: torch.Tensor
    intervention_memory: Any
    lateral_intervention_memory: Any
    influence_state: InfluenceGraphState | None
    previous_background_actions: Any
    previous_calibration_correction: Any
    previous_causal_gate: Any


EgoActionPolicy = Callable[[dict[str, torch.Tensor | int]], torch.Tensor | np.ndarray]


def _logged_ego_actions(
    states: np.ndarray, valid: np.ndarray | None = None
) -> np.ndarray:
    actions = ego_controls(
        states[:, ANCHOR_INDEX:173, 0],
        states[:, ANCHOR_INDEX + 1 : 174, 0],
        0.04,
    )
    if valid is not None:
        present = valid[:, ANCHOR_INDEX:173, 0] & valid[:, ANCHOR_INDEX + 1 : 174, 0]
        actions = actions.copy()
        actions[~present] = 0.0
    return actions


def _intervene(
    actions: torch.Tensor,
    kind: str | None,
    dose: float,
) -> torch.Tensor:
    result = actions.clone()
    if kind is None:
        return result
    start, stop = 25, 50
    if kind == "brake":
        result[:, start:stop, 0] = (result[:, start:stop, 0] - dose).clamp_min(-8.0)
    elif kind == "accelerate":
        result[:, start:stop, 0] = (result[:, start:stop, 0] + dose).clamp_max(4.0)
    elif kind == "left":
        result[:, start:stop, 1] = (result[:, start:stop, 1] + dose).clamp_max(0.6)
    else:
        raise ValueError(f"unknown intervention {kind!r}")
    return result


@torch.no_grad()
def rollout(
    model,
    logged_states: np.ndarray,
    logged_valid: np.ndarray,
    soft_plans: np.ndarray,
    map_polylines: np.ndarray,
    map_polyline_valid: np.ndarray,
    *,
    device: torch.device,
    history_frames: int,
    motion_seed: int | None,
    intervention: str | None = None,
    dose: float = 0.0,
    ads_policy: EgoActionPolicy | None = None,
    controller: ReactionController | str | None = None,
    controller_deterministic: bool = False,
    excluded_slots: tuple[str, ...] | None = None,
    influence_graph_config: dict[str, float | int] | None = None,
    exogenous_state: WorldExogenousState | None = None,
    teacher_forced_logged_context: bool = False,
    resume_state: RolloutState | None = None,
    snapshot_at_steps: tuple[int, ...] = (),
    record_policy_trace: bool = True,
) -> Rollout:
    """Run one causal offline response rollout for a matched batch."""
    if excluded_slots is None:
        logged_states, logged_valid = scoped_canonical_trajectory(
            logged_states, logged_valid
        )
    else:
        logged_states, logged_valid = scoped_canonical_trajectory(
            logged_states, logged_valid, excluded_slots=excluded_slots
        )
    states = torch.from_numpy(logged_states[:, ANCHOR_INDEX].copy()).to(device)
    valid = torch.from_numpy(logged_valid[:, ANCHOR_INDEX].copy()).to(device)
    history = torch.from_numpy(
        logged_states[:, ANCHOR_INDEX - history_frames + 1 : ANCHOR_INDEX + 1].copy()
    ).to(device)
    history_valid = torch.from_numpy(
        logged_valid[:, ANCHOR_INDEX - history_frames + 1 : ANCHOR_INDEX + 1].copy()
    ).to(device)
    reference = torch.from_numpy(np.asarray(soft_plans, np.float32)).to(device)
    maps = torch.from_numpy(np.asarray(map_polylines, np.float32)).to(device)
    map_valid = torch.from_numpy(np.asarray(map_polyline_valid, bool)).to(device)
    initial_reference = states[:, 1:, :2].clone()
    logged_ego = torch.from_numpy(_logged_ego_actions(logged_states, logged_valid)).to(
        device
    )
    scheduled_ego = _intervene(logged_ego, intervention, dose)
    if exogenous_state is not None:
        exogenous_state.validate(
            response_steps=149,
            scene_refresh_responses=model.cfg.scene_refresh_responses,
            scene_dim=model.cfg.scene_latent_dim,
            agent_dim=model.cfg.agent_latent_dim,
        )
        if exogenous_state.batch_size != len(states):
            raise ValueError("exogenous state batch does not match rollout batch")
    if isinstance(controller, str):
        controller = make_reaction_controller(
            controller, adapter_logit=model.decoder.intervention_logit
        ).to(device)
    influence_graph = (
        None
        if controller is None
        else CausalInfluenceGraph(**(influence_graph_config or {}))
    )
    influence_state: InfluenceGraphState | None = None
    historical_start = max(0, ANCHOR_INDEX - history_frames + 1)
    historical_ego_values = ego_controls(
        logged_states[:, historical_start:ANCHOR_INDEX, 0],
        logged_states[:, historical_start + 1 : ANCHOR_INDEX + 1, 0],
        0.04,
    )
    historical_present = (
        logged_valid[:, historical_start:ANCHOR_INDEX, 0]
        & logged_valid[:, historical_start + 1 : ANCHOR_INDEX + 1, 0]
    )
    historical_ego_values[~historical_present] = 0.0
    historical_ego = torch.from_numpy(historical_ego_values).to(device)
    generator = None
    if motion_seed is not None:
        generator = torch.Generator(device=device).manual_seed(int(motion_seed))
    generated: list[torch.Tensor] = []
    background_actions: list[torch.Tensor] = []
    reference_actions: list[torch.Tensor] = []
    base_background_actions: list[torch.Tensor] = []
    controller_alpha: list[torch.Tensor] = []
    controller_delta: list[torch.Tensor] = []
    controller_active: list[torch.Tensor] = []
    controller_rule_action: list[torch.Tensor] = []
    controller_calibration: list[torch.Tensor] = []
    controller_causal_gate: list[torch.Tensor] = []
    controller_causal_delta: list[torch.Tensor] = []
    controller_influence_authority: list[torch.Tensor] = []
    controller_influence_role: list[torch.Tensor] = []
    controller_influence_direct: list[torch.Tensor] = []
    controller_influence_secondary: list[torch.Tensor] = []
    # PPO samples are collected from this exact formal rollout.  They are
    # diagnostics rather than part of the released factual metric. Keeping
    # them in this sole executor prevents a second rollout implementation
    # from becoming the de-facto training environment.
    controller_policy_features: list[torch.Tensor] = []
    controller_raw_action: list[torch.Tensor] = []
    controller_log_prob: list[torch.Tensor] = []
    controller_value: list[torch.Tensor] = []
    controller_prior_nominal: list[torch.Tensor] = []
    controller_prior_correction: list[torch.Tensor] = []
    controller_extra: dict[str, list[torch.Tensor]] = {
        name: []
        for name in (
            "policy_active",
            "spatial_path_active",
            "autonomous_lane_active",
            "execution_active",
            "release_active",
            "requested_total_correction_ax",
            "projected_total_correction_ax",
            "executed_total_correction_ax",
            "calibration_relative_to_prior_ax",
            "correction_constraint_infeasible",
        )
    }
    filter_state = None
    slow_scene = None
    slow_scene_noise = None
    agent_noise_state = None
    agent_style_state = None
    previous_current = None
    committed_ego_controls = historical_ego
    executed_ego: list[torch.Tensor] = []
    intervention_memory = None
    lateral_intervention_memory = None
    # A post-HiQR controller is stateful: a jerk bound applies to its own
    # previously issued correction, not to a fresh zero command every frame.
    # Keep this state in the formal evaluator so controller-on factual metrics
    # describe the executable policy.
    previous_background_actions = None
    previous_calibration_correction = None
    previous_causal_gate = None
    # The controller is armed by a change in the action actually submitted on
    # this tick relative to the already committed preceding command.  It must
    # never be defined as ``submitted - logged_future``: that made every
    # natural logged-ego factual rollout controller-off and therefore hid the
    # factual effect of a controller-on policy.
    controller_enabled = torch.zeros(len(states), dtype=torch.bool, device=device)
    execute = model.cfg.execute_frames
    first_step = 0
    if resume_state is not None:
        if resume_state.states.shape[0] != len(states):
            raise ValueError("rollout snapshot batch does not match input batch")
        first_step = int(resume_state.response_index)
        for name in (
            "states",
            "valid",
            "history",
            "history_valid",
            "filter_state",
            "slow_scene",
            "slow_scene_noise",
            "agent_noise_state",
            "agent_style_state",
            "previous_current",
            "committed_ego_controls",
            "intervention_memory",
            "lateral_intervention_memory",
            "influence_state",
            "previous_background_actions",
            "previous_calibration_correction",
            "previous_causal_gate",
        ):
            locals_value = copy.deepcopy(getattr(resume_state, name))
            if name == "states":
                states = locals_value
            elif name == "valid":
                valid = locals_value
            elif name == "history":
                history = locals_value
            elif name == "history_valid":
                history_valid = locals_value
            elif name == "filter_state":
                filter_state = locals_value
            elif name == "slow_scene":
                slow_scene = locals_value
            elif name == "slow_scene_noise":
                slow_scene_noise = locals_value
            elif name == "agent_noise_state":
                agent_noise_state = locals_value
            elif name == "agent_style_state":
                agent_style_state = locals_value
            elif name == "previous_current":
                previous_current = locals_value
            elif name == "committed_ego_controls":
                committed_ego_controls = locals_value
            elif name == "intervention_memory":
                intervention_memory = locals_value
            elif name == "lateral_intervention_memory":
                lateral_intervention_memory = locals_value
            elif name == "influence_state":
                influence_state = locals_value
            elif name == "previous_background_actions":
                previous_background_actions = locals_value
            elif name == "previous_calibration_correction":
                previous_calibration_correction = locals_value
            elif name == "previous_causal_gate":
                previous_causal_gate = locals_value
    snapshots: dict[int, RolloutState] = {}
    for start in range(first_step, 149, execute):
        count = min(execute, 149 - start)
        preview = reference[:, start : start + model.cfg.preview_frames]
        if preview.shape[1] < model.cfg.preview_frames:
            preview = torch.cat(
                (
                    preview,
                    preview[:, -1:].expand(
                        -1, model.cfg.preview_frames - preview.shape[1], -1, -1
                    ),
                ),
                dim=1,
            )
        base = initial_reference if start == 0 else reference[:, start - 1]
        ego_block = scheduled_ego[:, start : start + execute]
        if ads_policy is not None:
            proposed = torch.as_tensor(
                ads_policy(
                    {
                        "agent_states": states.detach().clone(),
                        "agent_valid": valid.detach().clone(),
                        "reference_index": start,
                    }
                ),
                dtype=states.dtype,
                device=device,
            )
            if proposed.shape == (len(states), 2):
                ego_block = proposed[:, None].expand(-1, execute, -1)
            elif proposed.shape == (len(states), execute, 2):
                ego_block = proposed
            else:
                raise ValueError(
                    "ads_policy must return [batch,2] or "
                    "[batch,execute_frames,2] controls"
                )
        if count < execute:
            ego_block = torch.cat(
                (
                    ego_block,
                    ego_block[:, -1:].expand(-1, execute - count, -1),
                ),
                dim=1,
            )
        scene_noise = torch.zeros(
            (len(states), model.cfg.scene_latent_dim),
            device=device,
            dtype=states.dtype,
        )
        agent_noise = torch.zeros(
            (len(states), 7, model.cfg.agent_latent_dim),
            device=device,
            dtype=states.dtype,
        )
        if exogenous_state is not None:
            scene_index = min(
                start // int(exogenous_state.scene_refresh_responses),
                exogenous_state.scene_innovation_count - 1,
            )
            scene_noise = torch.from_numpy(
                exogenous_state.scene_innovations[:, scene_index]
            ).to(device=device, dtype=states.dtype)
            agent_noise = torch.from_numpy(
                exogenous_state.agent_response_innovations[:, start]
            ).to(device=device, dtype=states.dtype)
        elif generator is not None:
            scene_noise.normal_(generator=generator)
            agent_noise.normal_(generator=generator)
        response = model(
            history,
            history_valid,
            states,
            valid,
            preview,
            base,
            maps,
            map_valid,
            filter_state=filter_state,
            previous_current=previous_current,
            slow_scene=slow_scene,
            slow_scene_noise=slow_scene_noise,
            agent_noise_state=agent_noise_state,
            agent_style_state=agent_style_state,
            committed_ego_controls=committed_ego_controls,
            intervention_memory=intervention_memory,
            lateral_intervention_memory=lateral_intervention_memory,
            response_index=start // execute,
            scene_standard_normal=scene_noise,
            agent_standard_normal=agent_noise,
            deterministic=motion_seed is None and exogenous_state is None,
            # A controller-free ablation means the learned HiQR action alone.
            # The checkpoint's legacy handcrafted adapter is not a second
            # response policy silently substituted when online MA-IDM is off.
            apply_intervention_adapter=False,
            apply_explicit_ego_response=True,
        )
        filter_state = response.filter_state
        slow_scene = response.slow_scene
        slow_scene_noise = response.slow_scene_noise
        agent_noise_state = response.agent_noise_state
        agent_style_state = response.agent_style_state
        intervention_memory = response.intervention_memory
        lateral_intervention_memory = response.lateral_intervention_memory
        previous_current = states
        base_actions = response.actions
        if controller is not None:
            assert influence_graph is not None
            influence_state = influence_graph.update(
                states,
                valid,
                history,
                influence_state,
                previous_background_actions,
            )
            threshold = float(
                getattr(
                    controller,
                    "reaction_trigger_threshold_mps2",
                    model.cfg.intervention_trigger_threshold_mps2,
                )
            )
            controller_enabled = (
                (
                    committed_ego_controls[:, -1, 0] - committed_ego_controls[:, -2, 0]
                    < -threshold
                )
                if committed_ego_controls.shape[1] >= 2
                else torch.zeros(len(states), dtype=torch.bool, device=device)
            )
            context = ReactionControllerContext(
                history=history,
                history_valid=history_valid,
                current=states,
                current_valid=valid,
                committed_ego_controls=committed_ego_controls,
                base_actions=base_actions,
                reference_actions=response.reference_actions,
                intervention_trigger=response.intervention_trigger,
                intervention_memory=response.intervention_memory,
                lateral_intervention_memory=response.lateral_intervention_memory,
                agent_style_state=response.agent_style_state,
                response_field_gain=response.response_field_gain,
                response_sensitivity_bounds=model.response_sensitivity_bounds,
                adapter_gain=torch.sigmoid(model.decoder.intervention_logit),
                cfg=model.cfg,
                reaction_enabled=controller_enabled,
                reaction_phase=influence_state.phase,
                reaction_age_frames=influence_state.age_frames,
                reaction_max_frames=75,
                reaction_recovery_remaining=influence_state.recovery_remaining,
                reaction_recovery_frames=influence_graph.recovery_frames,
                reaction_release_ttc_s=influence_graph.release_ttc_s,
                previous_background_actions=previous_background_actions,
                influence_authority=influence_state.authority,
                influence_role=influence_state.role,
                influence_parent=influence_state.parent,
                influence_direct=influence_state.direct,
                influence_secondary=influence_state.secondary,
                influence_predicted_ttc_s=influence_state.predicted_ttc_s,
                influence_predicted_min_gap_m=influence_state.predicted_min_gap_m,
                previous_calibration_correction=previous_calibration_correction,
                previous_causal_gate=previous_causal_gate,
                policy_standard_normal=(
                    None
                    if exogenous_state is None
                    else torch.from_numpy(
                        exogenous_state.policy_response_innovations[:, start]
                    ).to(device=device, dtype=states.dtype)
                ),
                policy_calibration_standard_normal=(
                    None
                    if exogenous_state is None
                    else torch.from_numpy(
                        exogenous_state.policy_calibration_innovations[:, start]
                    ).to(device=device, dtype=states.dtype)
                ),
                planned_origin_xy=initial_reference,
                planned_current_xy=reference[:, start],
                planned_terminal_xy=reference[:, -1],
                planned_path_xy=reference,
                map_polylines=maps,
                map_polyline_valid=map_valid,
                response_index=start,
            )
            output = controller(context, deterministic=controller_deterministic)
            response_actions = output.actions
            controller_alpha.append(output.alpha.detach())
            controller_delta.append(output.delta_ax.detach())
            controller_active.append(output.active.detach())
            controller_rule_action.append(
                torch.zeros_like(output.delta_ax)
                if output.rule_action_ax is None
                else output.rule_action_ax.detach()
            )
            controller_calibration.append(
                torch.zeros_like(output.delta_ax)
                if output.calibration_correction_ax is None
                else output.calibration_correction_ax.detach()
            )
            controller_causal_gate.append(
                torch.zeros_like(output.delta_ax)
                if output.causal_gate is None
                else output.causal_gate.detach()
            )
            controller_causal_delta.append(
                torch.zeros_like(output.delta_ax)
                if output.causal_delta_ax is None
                else output.causal_delta_ax.detach()
            )
            controller_influence_authority.append(influence_state.authority.detach())
            controller_influence_role.append(influence_state.role.detach())
            controller_influence_direct.append(influence_state.direct.detach())
            controller_influence_secondary.append(influence_state.secondary.detach())
            # PPO training consumes these tensors. Formal evaluation does not,
            # and a complete split otherwise retains tens of gigabytes of
            # per-frame policy features before metrics are reduced.
            if record_policy_trace:
                if output.policy_features is not None:
                    controller_policy_features.append(output.policy_features.detach())
                if output.raw_action is not None:
                    controller_raw_action.append(output.raw_action.detach())
                if output.log_prob is not None:
                    controller_log_prob.append(output.log_prob.detach())
                if output.value is not None:
                    controller_value.append(output.value.detach())
            if output.response_prior_nominal_action_ax is not None:
                controller_prior_nominal.append(
                    output.response_prior_nominal_action_ax.detach()
                )
            if output.response_prior_correction_ax is not None:
                controller_prior_correction.append(
                    output.response_prior_correction_ax.detach()
                )
            for name, values in controller_extra.items():
                value = getattr(output, name)
                if value is not None:
                    values.append(value.detach())
            # ``rollout`` evaluates the one-frame execution contract, whereas
            # controllers may still expose an explicit horizon dimension.  The
            # carried actuator state must therefore be the action actually
            # committed on this tick, with shape ``[batch, slots, ...]``.
            previous_background_actions = response_actions[:, 0].detach()
            previous_calibration_correction = (
                None
                if output.calibration_correction_ax is None
                else (
                    output.calibration_correction_ax[:, 0]
                    if output.calibration_correction_ax.ndim > 2
                    else output.calibration_correction_ax
                ).detach()
            )
            previous_causal_gate = (
                None if output.causal_gate is None else output.causal_gate.detach()
            )
            if previous_causal_gate is not None and previous_causal_gate.ndim > 2:
                previous_causal_gate = previous_causal_gate[:, 0]
        else:
            response_actions = base_actions
        new_frames: list[torch.Tensor] = []
        for frame in range(count):
            controls = torch.cat(
                (ego_block[:, frame, None], response_actions[:, frame]), dim=1
            )
            states = model.dynamics.step(states, controls, valid, model.cfg.dt_s)
            if teacher_forced_logged_context:
                states = torch.from_numpy(
                    logged_states[:, ANCHOR_INDEX + start + frame + 1].copy()
                ).to(device)
                valid = torch.from_numpy(
                    logged_valid[:, ANCHOR_INDEX + start + frame + 1].copy()
                ).to(device)
            new_frames.append(states)
        executed_ego.append(ego_block[:, :count])
        committed_ego_controls = torch.cat(
            (committed_ego_controls, ego_block[:, :count]), dim=1
        )[:, -model.cfg.intervention_trigger_history_frames - 1 :]
        block = torch.stack(new_frames, dim=1)
        generated.append(block)
        background_actions.append(response_actions[:, :count])
        base_background_actions.append(base_actions[:, :count])
        reference_actions.append(response.reference_actions[:, :count])
        block_valid = valid[:, None].expand(-1, count, -1)
        history = torch.cat((history, block), dim=1)[:, -history_frames:]
        history_valid = torch.cat((history_valid, block_valid), dim=1)[
            :, -history_frames:
        ]
        boundary = start + count
        if boundary in snapshot_at_steps:
            snapshots[boundary] = copy.deepcopy(
                RolloutState(
                    response_index=boundary,
                    states=states,
                    valid=valid,
                    history=history,
                    history_valid=history_valid,
                    filter_state=filter_state,
                    slow_scene=slow_scene,
                    slow_scene_noise=slow_scene_noise,
                    agent_noise_state=agent_noise_state,
                    agent_style_state=agent_style_state,
                    previous_current=previous_current,
                    committed_ego_controls=committed_ego_controls,
                    intervention_memory=intervention_memory,
                    lateral_intervention_memory=lateral_intervention_memory,
                    influence_state=influence_state,
                    previous_background_actions=previous_background_actions,
                    previous_calibration_correction=previous_calibration_correction,
                    previous_causal_gate=previous_causal_gate,
                )
            )
    return Rollout(
        torch.cat(generated, dim=1).cpu().numpy(),
        torch.cat(background_actions, dim=1).cpu().numpy(),
        torch.cat(executed_ego, dim=1).cpu().numpy(),
        torch.cat(reference_actions, dim=1).cpu().numpy(),
        torch.cat(base_background_actions, dim=1).cpu().numpy(),
        (
            None
            if not controller_alpha
            else {
                "alpha": torch.stack(controller_alpha, dim=1).cpu().numpy(),
                "delta_ax": torch.stack(controller_delta, dim=1).cpu().numpy(),
                "active": torch.stack(controller_active, dim=1).cpu().numpy(),
                "rule_action_ax": torch.stack(controller_rule_action, dim=1)
                .cpu()
                .numpy(),
                "calibration_correction_ax": torch.stack(controller_calibration, dim=1)
                .cpu()
                .numpy(),
                "causal_gate": torch.stack(controller_causal_gate, dim=1).cpu().numpy(),
                "causal_delta_ax": torch.stack(controller_causal_delta, dim=1)
                .cpu()
                .numpy(),
                "influence_authority": torch.stack(
                    controller_influence_authority, dim=1
                )
                .cpu()
                .numpy(),
                "influence_role": torch.stack(controller_influence_role, dim=1)
                .cpu()
                .numpy(),
                "influence_direct": torch.stack(controller_influence_direct, dim=1)
                .cpu()
                .numpy(),
                "influence_secondary": torch.stack(
                    controller_influence_secondary, dim=1
                )
                .cpu()
                .numpy(),
                **(
                    {
                        "policy_features": torch.stack(
                            controller_policy_features, dim=1
                        )
                        .cpu()
                        .numpy()
                    }
                    if len(controller_policy_features) == len(controller_alpha)
                    else {}
                ),
                **(
                    {
                        "raw_action": torch.stack(controller_raw_action, dim=1)
                        .cpu()
                        .numpy()
                    }
                    if len(controller_raw_action) == len(controller_alpha)
                    else {}
                ),
                **(
                    {"log_prob": torch.stack(controller_log_prob, dim=1).cpu().numpy()}
                    if len(controller_log_prob) == len(controller_alpha)
                    else {}
                ),
                **(
                    {"value": torch.stack(controller_value, dim=1).cpu().numpy()}
                    if len(controller_value) == len(controller_alpha)
                    else {}
                ),
                **(
                    {
                        "response_prior_nominal_action_ax": torch.stack(
                            controller_prior_nominal, dim=1
                        )
                        .cpu()
                        .numpy()
                    }
                    if len(controller_prior_nominal) == len(controller_alpha)
                    else {}
                ),
                **(
                    {
                        "response_prior_correction_ax": torch.stack(
                            controller_prior_correction, dim=1
                        )
                        .cpu()
                        .numpy()
                    }
                    if len(controller_prior_correction) == len(controller_alpha)
                    else {}
                ),
                **{
                    name: torch.stack(values, dim=1).cpu().numpy()
                    for name, values in controller_extra.items()
                    if len(values) == len(controller_alpha)
                },
            }
        ),
        snapshots or None,
    )


def _factual_metrics(
    generated: np.ndarray,
    target: np.ndarray,
    active: np.ndarray,
) -> dict[str, float]:
    return _shared_factual_metrics(generated, target, active)


def _temporal_factual_metrics(
    generated: np.ndarray,
    target: np.ndarray,
    active: np.ndarray,
) -> dict[str, list[float]]:
    """Return per-horizon errors for drift rather than only end-point summaries."""
    return _shared_temporal_factual_metrics(generated, target, active)
