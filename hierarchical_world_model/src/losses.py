"""Factual, distributional and paired-intervention objectives."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as functional

from .model import DiffusionGuidedHiQR, ResponseDistribution
from .reference import response_relevance


def masked_mean(value: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    mask = valid.to(dtype=value.dtype)
    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(value)
    return (value * mask).sum() / mask.sum().clamp_min(1.0)


def quantile_match(
    predicted: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Match robust marginal shape without globally smoothing actions."""
    mask = valid.bool()
    if not mask.any():
        return predicted.new_zeros(())
    levels = predicted.new_tensor((0.1, 0.25, 0.5, 0.75, 0.9))
    pred = predicted[mask]
    truth = target[mask]
    return (torch.quantile(pred, levels) - torch.quantile(truth, levels)).abs().mean()


def factual_losses(
    model: DiffusionGuidedHiQR,
    output: ResponseDistribution,
    current: torch.Tensor,
    current_valid: torch.Tensor,
    target_actions: torch.Tensor,
    target_states: torch.Tensor,
    previous_actions: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Supervise factual response and boundary dynamics."""
    valid = current_valid[:, 1:]
    frame_valid = valid[:, None].expand(-1, output.actions.shape[1], -1)
    action_error = (output.mean - target_actions).abs()
    mean_states = model._integrate(current, output.mean, current_valid)
    position_error = torch.linalg.vector_norm(
        mean_states[..., :2] - target_states[..., :2], dim=-1
    )
    velocity_error = torch.linalg.vector_norm(
        mean_states[..., 2:4] - target_states[..., 2:4], dim=-1
    )
    variance = output.std.square().clamp_min(1.0e-8)
    nll = 0.5 * (
        (target_actions - output.mean).square() / variance
        + torch.log(variance)
        + math.log(2.0 * math.pi)
    )
    predicted_jerk = (
        torch.diff(
            output.mean,
            dim=1,
            prepend=previous_actions[:, None],
        )
        / model.cfg.dt_s
    )
    target_jerk = (
        torch.diff(
            target_actions,
            dim=1,
            prepend=previous_actions[:, None],
        )
        / model.cfg.dt_s
    )
    boundary = (predicted_jerk[:, 0] - target_jerk[:, 0]).abs()
    jerk = quantile_match(predicted_jerk[..., 0], target_jerk[..., 0], frame_valid)
    lateral_jerk = quantile_match(
        predicted_jerk[..., 1], target_jerk[..., 1], frame_valid
    )
    angular_acceleration = (
        torch.diff(
            output.mean[..., 1],
            dim=1,
            prepend=previous_actions[:, None, :, 1],
        )
        / model.cfg.dt_s
    )
    target_angular_acceleration = (
        torch.diff(
            target_actions[..., 1],
            dim=1,
            prepend=previous_actions[:, None, :, 1],
        )
        / model.cfg.dt_s
    )
    angular = quantile_match(
        output.mean[..., 1], target_actions[..., 1], frame_valid
    ) + quantile_match(angular_acceleration, target_angular_acceleration, frame_valid)
    terms = {
        "action": masked_mean(action_error, frame_valid),
        "position": masked_mean(position_error, frame_valid),
        "velocity": masked_mean(velocity_error, frame_valid),
        "nll": masked_mean(nll, frame_valid),
        "boundary": masked_mean(boundary, valid),
        "jerk": jerk + lateral_jerk,
        "angular": angular,
    }
    cfg = model.cfg
    terms["factual"] = (
        cfg.action_weight * terms["action"]
        + cfg.position_weight * terms["position"]
        + cfg.velocity_weight * terms["velocity"]
        + cfg.nll_weight * terms["nll"]
        + cfg.boundary_weight * terms["boundary"]
        + cfg.jerk_weight * terms["jerk"]
        + cfg.angular_weight * terms["angular"]
    )
    return terms


def paired_intervention_losses(
    model: DiffusionGuidedHiQR,
    batch: dict[str, torch.Tensor],
    *,
    maximum_sequences: int = 8,
    deterministic_forward: bool = False,
) -> dict[str, torch.Tensor]:
    """Match 0.8 s responses under a causal, common-history intervention.

    Each branch sees identical history at its first response boundary.  The
    changed ego control is then integrated, so a background action can change
    only at the following 25 Hz boundary.  This is the same causal contract
    used by the offline ``ClosedLoopWorld`` and the held-out intervention
    evaluator; directly
    editing ``current`` would leak an unexecuted ego action into the policy.
    """

    count = min(int(maximum_sequences), len(batch["current"]))
    selected = {
        name: value[:count]
        for name, value in batch.items()
        if isinstance(value, torch.Tensor)
    }
    responses_count = int(
        round(0.8 / (model.cfg.execute_frames * model.cfg.dt_s))
    )
    # These are drawn once, then addressed by response boundary and reused in
    # every nominal/treatment branch.  This is genuine common-random-number
    # training: a branch difference can no longer be reduced by resampling a
    # driver latent instead of learning an ego-caused response.
    common_scene_noise = torch.randn(
        responses_count,
        count,
        model.cfg.scene_latent_dim,
        device=selected["current"].device,
        dtype=selected["current"].dtype,
    )
    common_agent_noise = torch.randn(
        responses_count,
        count,
        7,
        model.cfg.agent_latent_dim,
        device=selected["current"].device,
        dtype=selected["current"].dtype,
    )

    def rollout(acceleration_delta: float) -> torch.Tensor:
        history = selected["history"].clone()
        current = selected["current"].clone()
        valid = selected["current_valid"]
        history_valid = selected["history_valid"]
        filter_state = None
        slow_scene = None
        slow_scene_noise = None
        agent_noise_state = None
        agent_style_state = None
        previous_current = None
        committed_ego_controls = selected.get("committed_ego_controls")
        intervention_memory = None
        lateral_intervention_memory = None
        actions = []
        for response in range(responses_count):
            start = response * model.cfg.execute_frames
            preview = selected["soft_reference"][:, start:]
            if preview.shape[1] < model.cfg.preview_frames:
                preview = torch.cat(
                    (
                        preview,
                        preview[:, -1:].expand(
                            -1,
                            model.cfg.preview_frames - preview.shape[1],
                            -1,
                            -1,
                        ),
                    ),
                    dim=1,
                )
            base = (
                selected["reference_base"]
                if response == 0
                else selected["soft_reference"][:, start - 1]
            )
            output = model(
                history,
                history_valid,
                current,
                valid,
                preview,
                base,
                selected["map_polylines"],
                selected["map_polyline_valid"],
                filter_state=filter_state,
                previous_current=previous_current,
                slow_scene=slow_scene,
                slow_scene_noise=slow_scene_noise,
                agent_noise_state=agent_noise_state,
                agent_style_state=agent_style_state,
                committed_ego_controls=committed_ego_controls,
                intervention_memory=intervention_memory,
                lateral_intervention_memory=lateral_intervention_memory,
                response_index=response,
                scene_standard_normal=common_scene_noise[response],
                agent_standard_normal=common_agent_noise[response],
                deterministic=deterministic_forward,
            )
            actions.append(output.mean)
            ego_control = selected["closed_ego_actions"][:, response].clone()
            if acceleration_delta:
                ego_control[:, 0] = (ego_control[:, 0] + acceleration_delta).clamp(
                    model.cfg.min_acceleration_mps2,
                    model.cfg.max_acceleration_mps2,
                )
            previous_current = current
            frames = []
            for frame in range(model.cfg.execute_frames):
                controls = torch.cat(
                    (ego_control[:, None], output.mean[:, frame]), dim=1
                )
                current = model.dynamics.step(
                    current, controls, valid, model.cfg.dt_s
                )
                frames.append(current)
            if committed_ego_controls is not None:
                committed_ego_controls = torch.cat(
                    (committed_ego_controls, ego_control[:, None]), dim=1
                )[:, -model.cfg.intervention_trigger_history_frames - 1 :]
            block = torch.stack(frames, dim=1)
            block_valid = valid[:, None].expand(-1, len(frames), -1)
            history = torch.cat((history, block), dim=1)[
                :, -model.cfg.history_frames :
            ]
            history_valid = torch.cat((history_valid, block_valid), dim=1)[
                :, -model.cfg.history_frames :
            ]
            filter_state = output.filter_state
            slow_scene = output.slow_scene
            slow_scene_noise = output.slow_scene_noise
            agent_noise_state = output.agent_noise_state
            agent_style_state = output.agent_style_state
            intervention_memory = output.intervention_memory
            lateral_intervention_memory = output.lateral_intervention_memory
        return torch.cat(actions, dim=1)

    baseline = rollout(0.0)
    responses: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for name, deltas in {
        "brake": (-1.5, -3.0),
        "accelerate": (1.0, 2.0),
    }.items():
        responses[name] = (rollout(deltas[0]), rollout(deltas[1]))
    relevance = response_relevance(selected["current"], selected["current_valid"])
    low_relevance = 1.0 - relevance
    locality_weight = low_relevance[:, None, :, None]
    locality_terms = []
    for mild, _ in responses.values():
        delta = mild - baseline
        locality_terms.append(
            (delta.square() * locality_weight).sum()
            / locality_weight.expand_as(delta).sum().clamp_min(1.0)
        )
    locality = torch.stack(locality_terms).mean()
    current = selected["current"]
    following_ego = (
        (current[:, 1:, 0] < current[:, :1, 0])
        & ((current[:, 1:, 1] - current[:, :1, 1]).abs() < 1.8)
        & selected["current_valid"][:, 1:]
    )
    margin = 0.03 * relevance[:, None]
    monotonicity_terms = []
    strength_terms = []
    for name, (mild, strong) in responses.items():
        mild_delta = mild[..., 0] - baseline[..., 0]
        dose_delta = strong[..., 0] - mild[..., 0]
        sign = -1.0 if name == "brake" else 1.0
        monotonicity_terms.append(
            masked_mean(
                functional.relu(margin - sign * mild_delta),
                following_ego[:, None],
            )
            + masked_mean(
                functional.relu(margin - sign * dose_delta),
                following_ego[:, None],
            )
        )
        kind_index = 0 if name == "brake" else 1
        if (
            model.cfg.dose_calibrated_intervention_loss
            and "natural_response_sensitivity_bounds" in selected
        ):
            sensitivity = selected["natural_response_sensitivity_bounds"][:, kind_index]
            mild_dose, strong_dose = {
                "brake": (1.5, 3.0),
                "accelerate": (1.0, 2.0),
            }[name]
            mild_bounds = sensitivity * mild_dose
            strong_bounds = sensitivity * strong_dose
        elif "natural_response_bounds" in selected:
            # `[batch, brake/accelerate, slot, P10/P90]`, fitted from train
            # recordings only.  Sparse condition cells already fall back to
            # the split-level interval in ``NaturalResponseCalibrator``.
            bounds = selected["natural_response_bounds"][:, kind_index]
            mild_bounds = bounds
            strong_bounds = bounds
        else:
            bounds = model.matched_response_bounds[kind_index].view(1, 1, 2)
            mild_bounds = bounds
            strong_bounds = bounds
        mild_effect = (sign * mild_delta).mean(dim=1)
        strong_effect = (
            sign * (strong[..., 0] - baseline[..., 0])
        ).mean(dim=1)
        strength_terms.append(
            masked_mean(
                functional.relu(mild_bounds[..., 0] - mild_effect)
                + functional.relu(mild_effect - mild_bounds[..., 1]),
                following_ego,
            )
            + masked_mean(
                functional.relu(strong_bounds[..., 0] - strong_effect)
                + functional.relu(strong_effect - strong_bounds[..., 1]),
                following_ego,
            )
        )
    monotonicity = torch.stack(monotonicity_terms).mean()
    strength = torch.stack(strength_terms).mean()
    return {
        "locality": locality,
        "monotonicity": monotonicity,
        "response_strength": strength,
        "intervention": (
            model.cfg.locality_weight * locality
            + model.cfg.monotonicity_weight * monotonicity
            + model.cfg.response_strength_weight * strength
        ),
    }


def closed_loop_factual_loss(
    model: DiffusionGuidedHiQR,
    batch: dict[str, torch.Tensor],
    *,
    maximum_sequences: int = 8,
    deterministic_forward: bool = True,
) -> torch.Tensor:
    """Expose the 25 Hz policy to one second of logged closed-loop state drift."""
    count = min(int(maximum_sequences), len(batch["current"]))
    selected = {
        name: value[:count]
        for name, value in batch.items()
        if isinstance(value, torch.Tensor)
    }
    current = selected["current"]
    valid = selected["current_valid"]
    history = selected["history"]
    history_valid = selected["history_valid"]
    filter_state = slow_scene = slow_scene_noise = agent_noise_state = None
    agent_style_state = None
    previous_current = None
    committed_ego_controls = selected.get("committed_ego_controls")
    intervention_memory = None
    lateral_intervention_memory = None
    predicted_states = []
    for step in range(selected["closed_ego_actions"].shape[1]):
        preview = selected["soft_reference"][:, step:]
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
        base = (
            selected["reference_base"]
            if step == 0
            else selected["soft_reference"][:, step - 1]
        )
        output = model(
            history,
            history_valid,
            current,
            valid,
            preview,
            base,
            selected["map_polylines"],
            selected["map_polyline_valid"],
            filter_state=filter_state,
            previous_current=previous_current,
            slow_scene=slow_scene,
            slow_scene_noise=slow_scene_noise,
            agent_noise_state=agent_noise_state,
            agent_style_state=agent_style_state,
            committed_ego_controls=committed_ego_controls,
            intervention_memory=intervention_memory,
            lateral_intervention_memory=lateral_intervention_memory,
            response_index=step,
            deterministic=deterministic_forward,
        )
        controls = torch.cat(
            (selected["closed_ego_actions"][:, step, None], output.mean[:, 0]), dim=1
        )
        previous_current = current
        current = model.dynamics.step(current, controls, valid, model.cfg.dt_s)
        if committed_ego_controls is not None:
            committed_ego_controls = torch.cat(
                (
                    committed_ego_controls,
                    selected["closed_ego_actions"][:, step, None],
                ),
                dim=1,
            )[:, -model.cfg.intervention_trigger_history_frames - 1 :]
        predicted_states.append(current[:, 1:])
        history = torch.cat((history, current[:, None]), dim=1)[
            :, -model.cfg.history_frames :
        ]
        history_valid = torch.cat((history_valid, valid[:, None]), dim=1)[
            :, -model.cfg.history_frames :
        ]
        filter_state = output.filter_state
        slow_scene = output.slow_scene
        slow_scene_noise = output.slow_scene_noise
        agent_noise_state = output.agent_noise_state
        agent_style_state = output.agent_style_state
        intervention_memory = output.intervention_memory
        lateral_intervention_memory = output.lateral_intervention_memory
    predicted = torch.stack(predicted_states, dim=1)
    target = selected["closed_target_states"]
    frame_valid = valid[:, None, 1:].expand(-1, predicted.shape[1], -1)
    position = torch.linalg.vector_norm(predicted[..., :2] - target[..., :2], dim=-1)
    velocity = torch.linalg.vector_norm(predicted[..., 2:4] - target[..., 2:4], dim=-1)
    return masked_mean(position, frame_valid) + 0.25 * masked_mean(velocity, frame_valid)


def training_losses(
    model: DiffusionGuidedHiQR,
    batch: dict[str, torch.Tensor],
    scene_noise: torch.Tensor,
    agent_noise: torch.Tensor,
    *,
    include_intervention: bool = True,
    intervention_sequences: int = 8,
    deterministic_forward: bool = False,
) -> dict[str, torch.Tensor]:
    output = model(
        batch["history"],
        batch["history_valid"],
        batch["current"],
        batch["current_valid"],
        batch["soft_reference"],
        batch["reference_base"],
        batch["map_polylines"],
        batch["map_polyline_valid"],
        committed_ego_controls=batch.get("committed_ego_controls"),
        scene_standard_normal=scene_noise,
        agent_standard_normal=agent_noise,
        deterministic=deterministic_forward,
    )
    terms = factual_losses(
        model,
        output,
        batch["current"],
        batch["current_valid"],
        batch["target_actions"],
        batch["target_states"],
        batch["previous_actions"],
    )
    if deterministic_forward:
        # A two-sample diversity objective has no stochastic semantics in the
        # base stage.  Do not silently turn it into a duplicate factual term.
        terms["energy"] = terms["factual"].new_zeros(())
    else:
        second = model(
            batch["history"], batch["history_valid"], batch["current"],
            batch["current_valid"], batch["soft_reference"], batch["reference_base"],
            batch["map_polylines"], batch["map_polyline_valid"],
            scene_standard_normal=-scene_noise, agent_standard_normal=-agent_noise,
            deterministic=False,
        )
        frame_valid = batch["current_valid"][:, None, 1:].expand(
            -1, output.actions.shape[1], -1
        )
        target_position = batch["target_states"][..., :2]
        first_distance = torch.linalg.vector_norm(output.states[..., :2] - target_position, dim=-1)
        second_distance = torch.linalg.vector_norm(second.states[..., :2] - target_position, dim=-1)
        pair_distance = torch.linalg.vector_norm(output.states[..., :2] - second.states[..., :2], dim=-1)
        terms["energy"] = masked_mean(0.5 * (first_distance + second_distance - pair_distance), frame_valid)
    if include_intervention:
        terms.update(
            paired_intervention_losses(
                model, batch, maximum_sequences=intervention_sequences,
                deterministic_forward=deterministic_forward,
            )
        )
    else:
        zero = terms["factual"].new_zeros(())
        terms.update(
            {
                "locality": zero,
                "monotonicity": zero,
                "response_strength": zero,
                "intervention": zero,
            }
        )
    terms["closed_loop_factual"] = (
        closed_loop_factual_loss(
            model, batch, maximum_sequences=intervention_sequences,
            deterministic_forward=deterministic_forward,
        )
        if include_intervention
        else terms["factual"].new_zeros(())
    )
    terms["loss"] = (
        terms["factual"]
        + model.cfg.energy_weight * terms["energy"]
        + terms["intervention"]
        + model.cfg.closed_loop_factual_weight * terms["closed_loop_factual"]
    )
    return terms
