"""Differentiable paired rollouts and response-distance losses for Idea A."""

from __future__ import annotations

import numpy as np
import torch

from traffic_components.src.core.dynamics import KinematicTrafficDynamics

from .shared_bc import SharedBCModel
from npc_behavior_benchmark.evaluation.rollout import _torch_features


def student_paired_response(
    model: SharedBCModel,
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
    history: torch.Tensor,
    history_valid: torch.Tensor,
    logged_future: torch.Tensor,
    intervention_stimulus: torch.Tensor,
    stimulus_index: torch.Tensor,
    response_index: torch.Tensor,
    lengths: torch.Tensor,
    widths: torch.Tensor,
    map_polylines: torch.Tensor,
    map_valid: torch.Tensor,
    *,
    futures: int = 2,
    style_resampling: str = "persistent",
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, samples = len(history), int(futures)
    expanded = batch * samples
    repeat = (
        lambda value: value[:, None]
        .expand(-1, samples, *value.shape[1:])
        .reshape(expanded, *value.shape[1:])
    )
    base_history = repeat(history)
    intervention_history = base_history.clone()
    valid_history = repeat(history_valid)
    logged = repeat(logged_future)
    stimulus_future = repeat(intervention_stimulus)
    stimulus = repeat(stimulus_index)
    response = repeat(response_index)
    length = repeat(lengths)
    width = repeat(widths)
    lines = repeat(map_polylines)
    line_valid = repeat(map_valid)
    lane_valid = line_valid.any(-1)
    lane_centers = (lines[..., 1] * line_valid).sum(-1) / line_valid.sum(-1).clamp_min(
        1
    )
    lane_width = (lines[..., 4] * line_valid).sum(-1) / line_valid.sum(-1).clamp_min(1)
    active = valid_history[:, -1]
    base_state = base_history[:, -1].clone()
    intervention_state = base_state.clone()
    initial_response_y = base_state[
        torch.arange(expanded, device=history.device), response, 1
    ]
    epsilon = torch.randn((expanded, 15, 6, 2), device=history.device)
    if style_resampling not in {"persistent", "per_decision"}:
        raise ValueError(f"unknown style_resampling {style_resampling!r}")
    style_steps = 15 if style_resampling == "per_decision" else 1
    style = (
        torch.randn((expanded, style_steps, 7, model.style_dim), device=history.device)
        if model.style_dim
        else None
    )
    base_features, reactive_features = [], []
    dynamics = KinematicTrafficDynamics()
    index = torch.arange(expanded, device=history.device)
    for decision in range(15):
        base_input = (
            _torch_features(
                base_history,
                valid_history,
                length,
                width,
                lane_centers,
                lane_width,
                lane_valid,
            )
            - feature_mean
        ) / feature_std
        intervention_input = (
            _torch_features(
                intervention_history,
                valid_history,
                length,
                width,
                lane_centers,
                lane_width,
                lane_valid,
            )
            - feature_mean
        ) / feature_std
        style_index = decision if style_steps > 1 else 0
        current_style = style[:, style_index] if style is not None else None
        base_mean, base_log_std = model(base_input, valid_history, current_style)
        reactive_mean, reactive_log_std = model(
            intervention_input, valid_history, current_style
        )
        base_action = base_mean + base_log_std.exp() * epsilon[:, decision]
        reactive_action = reactive_mean + reactive_log_std.exp() * epsilon[:, decision]
        limited = []
        for action, state in (
            (base_action, base_state),
            (reactive_action, intervention_state),
        ):
            speed = torch.linalg.vector_norm(state[:, 1:, 2:4], dim=-1).clamp_min(
                1.0e-3
            )
            yaw_limit = torch.minimum(torch.full_like(speed, 0.6), 4.0 / speed)
            action = (
                torch.stack(
                    (
                        action[..., 0].clamp(-8.0, 4.0),
                        torch.maximum(
                            torch.minimum(action[..., 1], yaw_limit), -yaw_limit
                        ),
                    ),
                    -1,
                )
                * active[:, 1:, None]
            )
            limited.append(action)
        base_action, reactive_action = limited
        base_control = torch.zeros((expanded, 7, 2), device=history.device)
        reactive_control = torch.zeros_like(base_control)
        base_control[:, 1:] = base_action
        reactive_control[:, 1:] = reactive_action
        for substep in range(5):
            frame = decision * 5 + substep
            next_base = dynamics.step(base_state, base_control, active, 0.04)
            next_intervention = dynamics.step(
                intervention_state, reactive_control, active, 0.04
            )
            external_base = logged[:, frame]
            external_intervention = external_base.clone()
            external_intervention[index, stimulus] = stimulus_future[:, frame]
            external_mask = torch.nn.functional.one_hot(stimulus, 7).bool()
            external_mask[:, 0] = True
            base_state = torch.where(external_mask[..., None], external_base, next_base)
            intervention_state = torch.where(
                external_mask[..., None], external_intervention, next_intervention
            )
            base_history = torch.cat((base_history[:, 1:], base_state[:, None]), 1)
            intervention_history = torch.cat(
                (intervention_history[:, 1:], intervention_state[:, None]), 1
            )
            valid_history = torch.cat((valid_history[:, 1:], active[:, None]), 1)
        base_response = base_state[index, response]
        reactive_response = intervention_state[index, response]
        base_gap = (
            base_state[index, stimulus, 0]
            - base_response[:, 0]
            - 0.5 * (length[index, stimulus] + length[index, response])
        )
        reactive_gap = (
            intervention_state[index, stimulus, 0]
            - reactive_response[:, 0]
            - 0.5 * (length[index, stimulus] + length[index, response])
        )
        response_slot = response - 1
        base_features.append(
            torch.stack(
                (
                    base_response[:, 2],
                    base_gap,
                    base_action[index, response_slot, 0],
                    base_response[:, 1] - initial_response_y,
                ),
                -1,
            )
        )
        reactive_features.append(
            torch.stack(
                (
                    reactive_response[:, 2],
                    reactive_gap,
                    reactive_action[index, response_slot, 0],
                    reactive_response[:, 1] - initial_response_y,
                ),
                -1,
            )
        )
    base_value = torch.stack(base_features, 1).reshape(batch, samples, 15, 4)
    reactive_value = torch.stack(reactive_features, 1).reshape(batch, samples, 15, 4)
    return reactive_value - base_value, reactive_value


def energy_distance_loss(
    student: torch.Tensor, teacher: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    batch = len(student)
    student_v = (student / scale).flatten(2) / np.sqrt(
        student.shape[-2] * student.shape[-1]
    )
    teacher_v = (teacher / scale).flatten(2) / np.sqrt(
        teacher.shape[-2] * teacher.shape[-1]
    )
    cross = torch.cdist(student_v, teacher_v).mean((1, 2))

    def off_diagonal(values):
        count = values.shape[1]
        if count < 2:
            return values.new_zeros(batch)
        distance = torch.cdist(values, values)
        return distance.sum((1, 2)) / (count * (count - 1))

    return (2.0 * cross - off_diagonal(student_v) - off_diagonal(teacher_v)).mean()
