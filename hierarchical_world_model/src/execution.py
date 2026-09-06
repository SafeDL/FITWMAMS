"""AMS-ready rollout and human-EVT risk interface for complete worlds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

import numpy as np
import torch

from process_highD.src.safety_envelope_risk import (
    SafetyEnvelopeRiskOptions,
    pairwise_safety_envelope_intrusion,
    smoothmax_axis,
    trajectory_safety_envelope_risk_trace,
)
from tools.evt import GPDTailModel

from .composition import HierarchicalWorldSampler
from .randomness import WorldExogenousState
from world_model.src.core.evaluation_scope import scoped_agent_valid

ADSActionPolicy = Callable[[dict[str, torch.Tensor | int]], torch.Tensor | np.ndarray]


@dataclass(frozen=True)
class WorldRollout:
    """HighwayEnv world trajectory, controls and EVT risk traces."""

    states: np.ndarray
    initial_valid: np.ndarray
    ego_actions: np.ndarray
    background_actions: np.ndarray
    event_risk: np.ndarray
    evt_score: np.ndarray | None
    numerical_valid: np.ndarray


def hold_current_ego_action(observation: dict[str, torch.Tensor | int]) -> torch.Tensor:
    """A deterministic no-new-intervention ADS baseline for interface audits."""
    state = observation["agent_states"]
    assert isinstance(state, torch.Tensor)
    ego = state[:, 0]
    speed = torch.linalg.vector_norm(ego[:, 2:4], dim=-1).clamp_min(0.5)
    heading = torch.atan2(ego[:, 3], ego[:, 2].clamp_min(1.0e-4))
    acceleration = ego[:, 4] * torch.cos(heading) + ego[:, 5] * torch.sin(heading)
    yaw_rate = (-ego[:, 4] * torch.sin(heading) + ego[:, 5] * torch.cos(heading)) / speed
    return torch.stack((acceleration, yaw_rate), dim=-1).clamp(
        min=torch.tensor([-8.0, -0.6], device=state.device),
        max=torch.tensor([4.0, 0.6], device=state.device),
    )


def trajectory_event_risk(
    states: np.ndarray,
    valid: np.ndarray,
    *,
    options: SafetyEnvelopeRiskOptions | None = None,
    dt_s: float = 0.04,
    excluded_slots: Iterable[str] = (),
) -> np.ndarray:
    """Evaluate safety-envelope severity over an explicit vehicle scope."""
    values = np.asarray(states, np.float32)
    present = np.asarray(scoped_agent_valid(valid, excluded_slots=excluded_slots), bool)
    if values.ndim != 4 or values.shape[2:] != (7, 6):
        raise ValueError("states must have shape [batch,frames,7,6]")
    if present.shape != (values.shape[0], 7):
        raise ValueError("valid must have shape [batch,7]")
    options = options or SafetyEnvelopeRiskOptions()
    n, frames = values.shape[:2]
    instantaneous = np.zeros((n, frames), np.float32)
    ego = values[:, :, 0]
    pair_scores: list[np.ndarray] = []
    for slot in range(6):
        other = values[:, :, slot + 1]
        pair, _ = pairwise_safety_envelope_intrusion(
            ego_x=ego[..., 0].reshape(-1), ego_y=ego[..., 1].reshape(-1),
            ego_vx=ego[..., 2].reshape(-1), ego_vy=ego[..., 3].reshape(-1),
            ego_ax=ego[..., 4].reshape(-1), ego_ay=ego[..., 5].reshape(-1),
            other_x=other[..., 0].reshape(-1), other_y=other[..., 1].reshape(-1),
            other_vx=other[..., 2].reshape(-1), other_vy=other[..., 3].reshape(-1),
            other_ax=other[..., 4].reshape(-1), other_ay=other[..., 5].reshape(-1),
            ego_length=4.8, ego_width=1.8, other_length=4.8, other_width=1.8,
            valid=np.broadcast_to(present[:, None, slot + 1], (n, frames)).reshape(-1),
            options=options,
        )
        pair_values = pair.reshape(n, frames)
        pair_values[~present[:, slot + 1], :] = -np.inf
        pair_scores.append(pair_values)
    if pair_scores:
        instantaneous = smoothmax_axis(
            np.stack(pair_scores, axis=-1), options.pair_smooth_beta, axis=-1
        )
    return np.asarray(
        [
            trajectory_safety_envelope_risk_trace(
                trace,
                options=options,
                dt_seconds=dt_s,
            ).max(initial=0.0)
            for trace in instantaneous
        ],
        np.float32,
    )


def rollout_world(
    sampler: HierarchicalWorldSampler,
    exogenous_state: WorldExogenousState,
    ads_policy: ADSActionPolicy | Any = hold_current_ego_action,
    *,
    steps: int | None = None,
    evt_model: GPDTailModel | None = None,
    risk_options: SafetyEnvelopeRiskOptions | None = None,
    reaction_controller: Any = None,
    reference_rebase_weights: tuple[float, float] | None = (1.0, 0.0),
    excluded_risk_slots: Iterable[str] = (),
) -> WorldRollout:
    """Replay a complete world; no ADS future action reaches a current response."""
    sample = sampler.compose_exogenous(exogenous_state)
    idm_config = getattr(ads_policy, "highway_env_idm_config", None)
    world = sampler.create_world(
        sample,
        idm_config=idm_config,
        controller=reaction_controller,
        reference_rebase_weights=reference_rebase_weights,
    )
    horizon = min(sample.soft_plan.shape[1], exogenous_state.response_steps)
    if steps is not None:
        horizon = min(horizon, int(steps))
    state_frames = [world.observe()["agent_states"].cpu().numpy()]
    ego_actions: list[np.ndarray] = []
    background_actions: list[np.ndarray] = []
    for _ in range(horizon):
        if idm_config is None:
            action = torch.as_tensor(
                ads_policy(world.observe()),
                device=world.device,
                dtype=torch.float32,
            )
        else:
            action = world.idm_actions()
        if action.ndim == 2:
            action = action[:, None]
        expected = (sample.initial_states.shape[0], 1, 2)
        if tuple(action.shape) != expected:
            raise ValueError(f"ADS policy must return shape {expected} or [batch,2]")
        transition = world.advance_response(action)
        state_frames.append(transition["agent_state_frames"][:, 0].cpu().numpy())
        ego_actions.append(transition["ego_actions"][:, 0].cpu().numpy())
        background_actions.append(transition["background_actions"][:, 0].cpu().numpy())
    states = np.stack(state_frames, axis=1).astype(np.float32)
    numerical_valid = np.isfinite(states).all(axis=(1, 2, 3))
    event_risk = trajectory_event_risk(
        states,
        sample.initial_valid,
        options=risk_options,
        excluded_slots=excluded_risk_slots,
    )
    evt_score = None if evt_model is None else np.asarray(evt_model.score(event_risk), np.float64)
    return WorldRollout(
        states=states,
        initial_valid=sample.initial_valid.copy(),
        ego_actions=np.stack(ego_actions, axis=1).astype(np.float32),
        background_actions=np.stack(background_actions, axis=1).astype(np.float32),
        event_risk=event_risk,
        evt_score=evt_score,
        numerical_valid=numerical_valid,
    )
