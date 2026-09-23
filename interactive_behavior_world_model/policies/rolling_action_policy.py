"""Idea B: joint action-space diffusion with a causal rolling interface."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from diffusion.src.model import (
    FactorizedSpatiotemporalBlock,
    cosine_beta_schedule,
    sinusoidal_embedding,
)
from world_model.src.core.dynamics import KinematicTrafficDynamics
from interactive_behavior_world_model.data.behavior_condition import (
    BEHAVIOR_CONDITION_MODES,
    append_behavior_condition,
)
from interactive_behavior_world_model.data.dataset import policy_features
from .interface import NPCPolicy, PolicyAction, PolicyObservation, RandomKey


@dataclass(frozen=True)
class ActionDiffusionConfig:
    feature_dim: int = 12
    action_horizon: int = 15
    agents: int = 6
    hidden_dim: int = 96
    num_layers: int = 3
    num_heads: int = 4
    dropout: float = 0.1
    diffusion_steps: int = 100
    prediction_type: str = "v_prediction"
    training_objective: str = "diffusion"


def runtime_action_method_id(trained_method: str, warm_start: bool) -> str:
    """Preserve objective/data ablation suffixes while naming B0/B1 runtime."""
    method = str(trained_method)
    for cold, warm in (
        ("rolling_action_flow_b0", "rolling_action_flow_b1"),
        ("rolling_action_b0", "rolling_action_b1"),
    ):
        if method.startswith(cold):
            return method.replace(cold, warm, 1) if warm_start else method
    return method


def _coefficient(
    values: torch.Tensor, timestep: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    return values.gather(0, timestep).reshape(
        (len(timestep),) + (1,) * (target.ndim - 1)
    )


class HistoryConditionEncoder(nn.Module):
    def __init__(self, config: ActionDiffusionConfig) -> None:
        super().__init__()
        self.temporal = nn.GRU(config.feature_dim, config.hidden_dim, batch_first=True)
        layer = nn.TransformerEncoderLayer(
            config.hidden_dim,
            config.num_heads,
            config.hidden_dim * 3,
            config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.interaction = nn.TransformerEncoder(layer, 2)
        self.global_projection = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
        )

    def forward(self, features: torch.Tensor, valid: torch.Tensor):
        batch, time, agents, dim = features.shape
        temporal = features.transpose(1, 2).reshape(batch * agents, time, dim)
        _, hidden = self.temporal(temporal)
        tokens = hidden[-1].reshape(batch, agents, -1)
        padding = ~valid[:, -1]
        tokens = self.interaction(tokens, src_key_padding_mask=padding)
        weights = valid[:, -1, :, None].to(tokens.dtype)
        pooled = (tokens * weights).sum(1) / weights.sum(1).clamp_min(1.0)
        return tokens, padding, self.global_projection(pooled)


class ActionDenoiser(nn.Module):
    def __init__(self, config: ActionDiffusionConfig) -> None:
        super().__init__()
        self.config = config
        self.condition_encoder = HistoryConditionEncoder(config)
        self.timestep_encoder = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        self.input_projection = nn.Linear(2, config.hidden_dim)
        self.time_embedding = nn.Parameter(
            torch.zeros(1, config.action_horizon, 1, config.hidden_dim)
        )
        self.agent_embedding = nn.Parameter(
            torch.zeros(1, 1, config.agents, config.hidden_dim)
        )
        self.blocks = nn.ModuleList(
            FactorizedSpatiotemporalBlock(config) for _ in range(config.num_layers)
        )
        self.output = nn.Sequential(
            nn.LayerNorm(config.hidden_dim), nn.Linear(config.hidden_dim, 2)
        )

    def forward(
        self,
        noisy: torch.Tensor,
        timestep: torch.Tensor,
        features: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        condition, padding, pooled = self.condition_encoder(features, valid)
        time_condition = self.timestep_encoder(
            sinusoidal_embedding(timestep, self.config.hidden_dim)
        )
        condition = condition + time_condition[:, None]
        pooled = pooled + time_condition
        token = (
            self.input_projection(noisy)
            + self.time_embedding
            + self.agent_embedding
            + time_condition[:, None, None]
        )
        background_padding = ~valid[:, -1, 1:]
        for block in self.blocks:
            token = block(token, condition, padding, pooled, background_padding)
        return self.output(token)


class RollingActionDiffusion(nn.Module):
    def __init__(
        self,
        config: ActionDiffusionConfig,
        action_mean: np.ndarray,
        action_std: np.ndarray,
    ) -> None:
        super().__init__()
        if config.training_objective not in {"diffusion", "flow_matching"}:
            raise ValueError(
                f"unsupported training objective: {config.training_objective}"
            )
        self.config = config
        self.denoiser = ActionDenoiser(config)
        beta = cosine_beta_schedule(config.diffusion_steps)
        alpha_bar = torch.cumprod(1.0 - beta, 0)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("sqrt_alpha_bar", torch.sqrt(alpha_bar))
        self.register_buffer("sqrt_one_minus_alpha_bar", torch.sqrt(1.0 - alpha_bar))
        mean = torch.as_tensor(action_mean)
        std = torch.as_tensor(action_std)
        if mean.shape == (2,):
            mean, std = mean.reshape(1, 1, 1, 2), std.reshape(1, 1, 1, 2)
        elif mean.shape == (config.action_horizon, 2):
            mean, std = mean.reshape(1, config.action_horizon, 1, 2), std.reshape(
                1, config.action_horizon, 1, 2
            )
        else:
            raise ValueError("target normalization must be [2] or [action_horizon,2]")
        self.register_buffer("action_mean", mean)
        self.register_buffer("action_std", std)

    def q_sample(self, clean, timestep, noise):
        return (
            _coefficient(self.sqrt_alpha_bar, timestep, clean) * clean
            + _coefficient(self.sqrt_one_minus_alpha_bar, timestep, clean) * noise
        )

    def predict_clean(self, noisy, timestep, output):
        alpha = _coefficient(self.alpha_bar, timestep, noisy)
        return torch.sqrt(alpha) * noisy - torch.sqrt(1.0 - alpha) * output

    def predict_noise(self, noisy, timestep, output):
        alpha = _coefficient(self.alpha_bar, timestep, noisy)
        return torch.sqrt(1.0 - alpha) * noisy + torch.sqrt(alpha) * output

    def loss(
        self,
        clean,
        features,
        history_valid,
        target_valid,
        current_state,
        future_states,
        trajectory_weight: float = 0.02,
        lateral_trajectory_multiplier: float = 1.0,
    ):
        mask = target_valid[..., None].to(clean.dtype)
        normalized = (clean - self.action_mean) / self.action_std
        batch = len(clean)
        noise = torch.randn_like(normalized) * mask
        if self.config.training_objective == "flow_matching":
            flow_time = torch.rand((batch,), device=clean.device, dtype=clean.dtype)
            interpolation = flow_time.reshape((batch,) + (1,) * (normalized.ndim - 1))
            noisy = ((1.0 - interpolation) * noise + interpolation * normalized) * mask
            network_time = flow_time * float(self.config.diffusion_steps - 1)
            output = self.denoiser(noisy, network_time, features, history_valid) * mask
            objective = (normalized - noise) * mask
            predicted = (noisy + (1.0 - interpolation) * output) * mask
        else:
            timestep = torch.randint(
                self.config.diffusion_steps, (batch,), device=clean.device
            )
            noisy = self.q_sample(normalized, timestep, noise) * mask
            output = self.denoiser(noisy, timestep, features, history_valid) * mask
            alpha = _coefficient(self.alpha_bar, timestep, normalized)
            objective = (
                torch.sqrt(alpha) * noise - torch.sqrt(1.0 - alpha) * normalized
            ) * mask
            predicted = self.predict_clean(noisy, timestep, output) * mask
        denominator = mask.sum().clamp_min(1.0)
        denoising = ((output - objective).square() * mask).sum() / denominator
        x0_l1 = ((predicted - normalized).abs() * mask).sum() / denominator
        pair_mask = mask[:, 1:] * mask[:, :-1]
        smooth = (
            (
                (predicted[:, 1:] - predicted[:, :-1])
                - (normalized[:, 1:] - normalized[:, :-1])
            ).abs()
            * pair_mask
        ).sum() / pair_mask.sum().clamp_min(1.0)
        trajectory = clean.new_zeros(())
        if trajectory_weight > 0:
            controls = predicted * self.action_std + self.action_mean
            state = current_state
            dynamics = KinematicTrafficDynamics()
            error_sum = clean.new_zeros(())
            error_count = clean.new_zeros(())
            for decision in range(self.config.action_horizon):
                for _ in range(5):
                    state = dynamics.step(
                        state, controls[:, decision], target_valid[:, decision], 0.04
                    )
                state_error = torch.cat(
                    (
                        (state[..., :2] - future_states[:, decision, ..., :2]) / 10.0,
                        (state[..., 2:4] - future_states[:, decision, ..., 2:4]) / 5.0,
                    ),
                    -1,
                )
                if lateral_trajectory_multiplier != 1.0:
                    state_error = state_error * state_error.new_tensor(
                        [
                            1.0,
                            lateral_trajectory_multiplier,
                            1.0,
                            lateral_trajectory_multiplier,
                        ]
                    )
                valid = target_valid[:, decision, :, None].to(clean.dtype)
                error_sum = (
                    error_sum
                    + (
                        F.smooth_l1_loss(
                            state_error, torch.zeros_like(state_error), reduction="none"
                        )
                        * valid
                    ).sum()
                )
                error_count = error_count + valid.sum() * 4
            trajectory = error_sum / error_count.clamp_min(1.0)
        return {
            "loss": denoising
            + 0.1 * x0_l1
            + 0.02 * smooth
            + trajectory_weight * trajectory,
            "denoising": denoising,
            "x0_l1": x0_l1,
            "smooth": smooth,
            "trajectory": trajectory,
        }

    @torch.no_grad()
    def _sample_schedule(self, trajectory, schedule, features, history_valid, mask):
        batch = len(trajectory)
        for index in reversed(range(len(schedule))):
            step = int(schedule[index])
            previous = int(schedule[index - 1]) if index else -1
            timestep = torch.full(
                (batch,), step, dtype=torch.long, device=trajectory.device
            )
            output = self.denoiser(trajectory, timestep, features, history_valid) * mask
            clean = (
                self.predict_clean(trajectory, timestep, output).clamp(-6.0, 6.0) * mask
            )
            if previous < 0:
                trajectory = clean
            else:
                noise = self.predict_noise(trajectory, timestep, output) * mask
                alpha = self.alpha_bar[previous]
                trajectory = (
                    torch.sqrt(alpha) * clean + torch.sqrt(1.0 - alpha) * noise
                ) * mask
        return trajectory

    @torch.no_grad()
    def _sample_flow(
        self,
        trajectory,
        features,
        history_valid,
        mask,
        inference_steps: int,
        start_time: float = 0.0,
    ):
        if inference_steps < 1:
            raise ValueError("inference_steps must be positive")
        batch = len(trajectory)
        step_size = (1.0 - float(start_time)) / float(inference_steps)
        for index in range(inference_steps):
            flow_time = float(start_time) + index * step_size
            network_time = torch.full(
                (batch,),
                flow_time * float(self.config.diffusion_steps - 1),
                dtype=trajectory.dtype,
                device=trajectory.device,
            )
            velocity = (
                self.denoiser(trajectory, network_time, features, history_valid) * mask
            )
            trajectory = (trajectory + step_size * velocity).clamp(-6.0, 6.0) * mask
        return trajectory

    @torch.no_grad()
    def sample(
        self,
        features,
        history_valid,
        target_valid,
        initial_noise,
        inference_steps: int = 8,
    ):
        mask = target_valid[..., None].to(features.dtype)
        if self.config.training_objective == "flow_matching":
            normalized = self._sample_flow(
                initial_noise * mask, features, history_valid, mask, inference_steps
            )
        else:
            schedule = sorted(
                {
                    int(round(x))
                    for x in np.linspace(
                        0, self.config.diffusion_steps - 1, inference_steps
                    )
                }
            )
            normalized = self._sample_schedule(
                initial_noise * mask, schedule, features, history_valid, mask
            )
        return (normalized * self.action_std + self.action_mean) * mask

    @torch.no_grad()
    def sample_warm(
        self,
        features,
        history_valid,
        target_valid,
        shifted_clean,
        noise,
        inference_steps: int = 8,
        start_step: int = 49,
    ):
        mask = target_valid[..., None].to(features.dtype)
        normalized = (shifted_clean - self.action_mean) / self.action_std
        if self.config.training_objective == "flow_matching":
            start_time = float(start_step) / float(
                max(self.config.diffusion_steps - 1, 1)
            )
            noisy = ((1.0 - start_time) * noise + start_time * normalized) * mask
            result = self._sample_flow(
                noisy, features, history_valid, mask, inference_steps, start_time
            )
            return (result * self.action_std + self.action_mean) * mask
        timestep = torch.full(
            (len(features),), int(start_step), dtype=torch.long, device=features.device
        )
        noisy = self.q_sample(normalized, timestep, noise) * mask
        schedule = sorted(
            {int(round(x)) for x in np.linspace(0, int(start_step), inference_steps)}
        )
        result = self._sample_schedule(noisy, schedule, features, history_valid, mask)
        return (result * self.action_std + self.action_mean) * mask


@dataclass
class _RollingSnapshot:
    history: np.ndarray
    history_valid: np.ndarray
    cached_actions: np.ndarray | None
    rng_state: dict[str, Any]
    decision_index: int


class RollingActionPolicy(NPCPolicy):
    """Stateful NPCPolicy adapter for cold B0 or warm-cache B1 inference."""

    def __init__(
        self,
        model: RollingActionDiffusion,
        feature_mean: np.ndarray,
        feature_std: np.ndarray,
        *,
        inference_steps: int = 4,
        warm_start: bool = True,
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device).eval()
        self.device = torch.device(device)
        self.feature_mean = np.asarray(feature_mean, np.float32)
        self.feature_std = np.asarray(feature_std, np.float32)
        self.inference_steps = int(inference_steps)
        self.warm_start = bool(warm_start)

    def reset(
        self,
        history: np.ndarray,
        history_valid: np.ndarray,
        agent_ids: np.ndarray,
        map_context: Mapping[str, np.ndarray],
        random_key: RandomKey,
        behavior_condition: Mapping[str, Any] | None = None,
    ) -> None:
        if self.model.config.feature_dim == 12:
            if behavior_condition is not None:
                raise ValueError(
                    "this unconditional 12-feature model cannot accept a T4a behavior condition"
                )
            self.behavior_condition = None
        elif self.model.config.feature_dim == 12 + len(BEHAVIOR_CONDITION_MODES):
            # Validate and retain an immutable copy at the policy boundary.
            self.behavior_condition = (
                None if behavior_condition is None else dict(behavior_condition)
            )
            append_behavior_condition(
                np.zeros((1, len(agent_ids), 12), np.float32), self.behavior_condition
            )
        else:
            raise ValueError(
                f"unsupported policy feature_dim={self.model.config.feature_dim}"
            )
        self.history = np.asarray(history, np.float32).copy()
        self.history_valid = np.asarray(history_valid, bool).copy()
        self.agent_ids = np.asarray(agent_ids, np.int64).copy()
        self.map_context = {
            key: np.asarray(value).copy() for key, value in map_context.items()
        }
        self.rng = np.random.default_rng(random_key.seed())
        self.cached_actions = None
        self.decision_index = 0

    @torch.no_grad()
    def act(self, observation: PolicyObservation) -> PolicyAction:
        observation.validate()
        if not np.array_equal(observation.agent_ids, self.agent_ids):
            raise ValueError("agent identity/order changed without reset")
        feature = policy_features(
            observation.history_states,
            observation.history_valid,
            observation.lengths_m,
            observation.widths_m,
            observation.map_polylines,
            observation.map_polyline_valid,
        )
        if self.model.config.feature_dim == 12 + len(BEHAVIOR_CONDITION_MODES):
            feature = append_behavior_condition(feature, self.behavior_condition)
        feature_t = torch.from_numpy(
            ((feature - self.feature_mean) / self.feature_std)[None].astype(np.float32)
        ).to(self.device)
        valid_t = torch.from_numpy(observation.history_valid[None]).to(self.device)
        target_valid = valid_t[:, -1:, 1:].expand(
            -1, self.model.config.action_horizon, -1
        )
        noise = torch.from_numpy(
            self.rng.standard_normal(
                (1, self.model.config.action_horizon, self.model.config.agents, 2)
            ).astype(np.float32)
        ).to(self.device)
        if self.warm_start and self.cached_actions is not None:
            cached = torch.from_numpy(self.cached_actions).to(self.device)
            shifted = torch.cat((cached[:, 1:], cached[:, -1:]), 1)
            block = self.model.sample_warm(
                feature_t, valid_t, target_valid, shifted, noise, self.inference_steps
            )
        else:
            block = self.model.sample(
                feature_t, valid_t, target_valid, noise, self.inference_steps
            )
        self.cached_actions = block.cpu().numpy()
        self.history = np.asarray(observation.history_states).copy()
        self.history_valid = np.asarray(observation.history_valid).copy()
        self.decision_index += 1
        return PolicyAction(
            block[0, 0].cpu().numpy(),
            diagnostics={
                "nn_evaluations": self.inference_steps,
                "replan": True,
                "warm_start": self.warm_start,
            },
        )

    def snapshot(self) -> _RollingSnapshot:
        return _RollingSnapshot(
            self.history.copy(),
            self.history_valid.copy(),
            None if self.cached_actions is None else self.cached_actions.copy(),
            copy.deepcopy(self.rng.bit_generator.state),
            self.decision_index,
        )

    def restore(self, snapshot: _RollingSnapshot) -> None:
        self.history = snapshot.history.copy()
        self.history_valid = snapshot.history_valid.copy()
        self.cached_actions = (
            None if snapshot.cached_actions is None else snapshot.cached_actions.copy()
        )
        self.rng.bit_generator.state = copy.deepcopy(snapshot.rng_state)
        self.decision_index = int(snapshot.decision_index)
