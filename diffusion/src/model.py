"""Conditional diffusion over the joint six-background trajectory."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class DiffusionModelConfig:
    condition_dim: int = 118
    horizon_steps: int = 149
    target_dim: int = 12
    hidden_dim: int = 128
    num_layers: int = 4
    num_heads: int = 4
    dropout: float = 0.1
    diffusion_steps: int = 100
    x0_weight: float = 0.1
    smooth_weight: float = 0.0
    long_horizon_weight: float = 0.0
    trajectory_position_weight: float = 0.0
    trajectory_velocity_weight: float = 0.0
    target_scale_x: float = 1.0
    target_scale_y: float = 1.0
    dt_s: float = 0.04
    x0_clip_abs: float = 10.0
    prediction_type: str = "epsilon"
    condition_dropout_prob: float = 0.0
    min_snr_gamma: float = 0.0
    constraint_agents: int = 6
    constraint_dim: int = 12
    denoiser: str = "factorized_spatiotemporal"

    def __post_init__(self) -> None:
        if self.horizon_steps != 149 or self.target_dim != 12:
            raise ValueError("highD background diffusion requires [149, 6 * 2] targets")
        if self.hidden_dim % self.num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if self.diffusion_steps < 2:
            raise ValueError("diffusion_steps must be at least two")
        if self.prediction_type not in {"epsilon", "v_prediction"}:
            raise ValueError(f"unsupported prediction_type {self.prediction_type!r}")
        if not 0.0 <= self.condition_dropout_prob < 1.0:
            raise ValueError("condition_dropout_prob must be in [0, 1)")
        if self.min_snr_gamma < 0.0:
            raise ValueError("min_snr_gamma must be non-negative")
        expected_condition_dim = 40 + 6 + self.constraint_agents * self.constraint_dim
        if self.constraint_agents != 6 or self.condition_dim != expected_condition_dim:
            raise ValueError(
                "condition_dim must contain 40-D C0, mask and three state knots"
            )
        if self.constraint_dim != 12:
            raise ValueError("each vehicle requires 12 state-knot values")
        if self.denoiser != "factorized_spatiotemporal":
            raise ValueError("only the factorized spatiotemporal denoiser is supported")

    def to_dict(self) -> dict[str, int | float | str]:
        return asdict(self)


def sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    frequencies = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, device=timesteps.device, dtype=torch.float32)
        / max(half - 1, 1)
    )
    angles = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
    embedding = torch.cat((torch.sin(angles), torch.cos(angles)), dim=1)
    return F.pad(embedding, (0, dim - embedding.shape[1]))


class StructuredConditionEncoder(nn.Module):
    """Preserve agent/time structure in the flattened condition.

    The public data contract stays flat for storage and interoperability, but
    invalid background slots are excluded from cross attention rather than
    represented by a learned-looking zero placeholder.
    """

    NUM_BACKGROUND = 6
    C0_DIM = 40

    def __init__(self, config: DiffusionModelConfig) -> None:
        super().__init__()
        self.config = config
        hidden = config.hidden_dim
        self.constraint_agents = config.constraint_agents
        self.constraint_dim = config.constraint_dim
        self.c0_projection = nn.Linear(self.C0_DIM, hidden)
        self.constraint_projection = nn.Linear(config.constraint_dim, hidden)
        self.agent_embedding = nn.Parameter(torch.zeros(1, self.NUM_BACKGROUND, hidden))
        self.agent_norm = nn.LayerNorm(hidden)
        self.global_projection = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )

    def forward(
        self, condition: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if condition.ndim != 2 or condition.shape[1] != self.config.condition_dim:
            raise ValueError(
                "structured condition has the wrong shape: " f"{tuple(condition.shape)}"
            )
        batch = condition.shape[0]
        c0 = condition[:, :40]
        background_valid = condition[:, 40:46] > 0.5
        constraint = condition[:, 46:].reshape(batch, 6, self.constraint_dim)
        c0_token = self.agent_norm(self.c0_projection(c0))[:, None]
        plan_tokens = self.agent_norm(
            self.constraint_projection(constraint) + self.agent_embedding
        )
        tokens = torch.cat((c0_token, plan_tokens), dim=1)
        c0_padding = torch.zeros(batch, 1, dtype=torch.bool, device=condition.device)
        padding = torch.cat((c0_padding, ~background_valid), dim=1)
        pooled = c0_token[:, 0]
        pooled += (plan_tokens * background_valid[..., None]).sum(dim=1)
        pooled /= (1 + background_valid.sum(dim=1, keepdim=True)).clamp_min(1)
        return tokens, padding, self.global_projection(pooled)


class FactorizedSpatiotemporalBlock(nn.Module):
    """Alternate per-agent temporal and per-frame interaction attention."""

    def __init__(self, config: DiffusionModelConfig) -> None:
        super().__init__()
        self.temporal = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=4 * config.hidden_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.spatial = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=2 * config.hidden_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.cross_norm = nn.LayerNorm(config.hidden_dim)
        self.cross_attention = nn.MultiheadAttention(
            config.hidden_dim,
            config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.film = nn.Sequential(
            nn.SiLU(),
            nn.Linear(config.hidden_dim, 2 * config.hidden_dim),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        condition_tokens: torch.Tensor,
        condition_padding: torch.Tensor,
        global_condition: torch.Tensor,
        background_padding: torch.Tensor,
    ) -> torch.Tensor:
        batch, time, agents, hidden = tokens.shape
        temporal = tokens.permute(0, 2, 1, 3).reshape(batch * agents, time, hidden)
        temporal = self.temporal(temporal)
        tokens = temporal.reshape(batch, agents, time, hidden).permute(0, 2, 1, 3)
        spatial = tokens.reshape(batch * time, agents, hidden)
        spatial_padding = background_padding[:, None].expand(-1, time, -1)
        spatial_padding = spatial_padding.reshape(batch * time, agents).clone()
        all_padding = spatial_padding.all(dim=1)
        spatial_padding[all_padding, 0] = False
        spatial = self.spatial(spatial, src_key_padding_mask=spatial_padding)
        tokens = spatial.reshape(batch, time, agents, hidden)
        query = self.cross_norm(tokens).reshape(batch, time * agents, hidden)
        attended, _ = self.cross_attention(
            query,
            condition_tokens,
            condition_tokens,
            key_padding_mask=condition_padding,
            need_weights=False,
        )
        tokens = tokens + attended.reshape(batch, time, agents, hidden)
        scale, shift = self.film(global_condition).chunk(2, dim=-1)
        return tokens * (1.0 + scale[:, None, None]) + shift[:, None, None]


class FactorizedSpatiotemporalDenoiser(nn.Module):
    """Denoise each vehicle path while explicitly modeling traffic interaction."""

    NUM_BACKGROUND = 6

    def __init__(self, config: DiffusionModelConfig) -> None:
        super().__init__()
        self.config = config
        self.condition_encoder = StructuredConditionEncoder(config)
        self.timestep_encoder = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        self.input_projection = nn.Linear(2, config.hidden_dim)
        self.time_embedding = nn.Parameter(
            torch.zeros(1, config.horizon_steps, 1, config.hidden_dim)
        )
        self.agent_embedding = nn.Parameter(
            torch.zeros(1, 1, self.NUM_BACKGROUND, config.hidden_dim)
        )
        self.blocks = nn.ModuleList(
            FactorizedSpatiotemporalBlock(config) for _ in range(config.num_layers)
        )
        self.output_projection = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, 2),
        )

    def forward(
        self,
        noisy_trajectory: torch.Tensor,
        timesteps: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        batch = noisy_trajectory.shape[0]
        expected = (batch, self.config.horizon_steps, self.config.target_dim)
        if tuple(noisy_trajectory.shape) != expected:
            raise ValueError(
                f"expected noisy trajectory {expected}, got {tuple(noisy_trajectory.shape)}"
            )
        condition_tokens, padding, global_condition = self.condition_encoder(condition)
        time_condition = self.timestep_encoder(
            sinusoidal_embedding(timesteps, self.config.hidden_dim)
        )
        condition_tokens = condition_tokens + time_condition[:, None]
        global_condition = global_condition + time_condition
        background_padding = ~(condition[:, 40:46] > 0.5)
        noisy = noisy_trajectory.reshape(
            batch, self.config.horizon_steps, self.NUM_BACKGROUND, 2
        )
        tokens = (
            self.input_projection(noisy)
            + self.time_embedding
            + self.agent_embedding
            + time_condition[:, None, None]
        )
        for block in self.blocks:
            tokens = block(
                tokens,
                condition_tokens,
                padding,
                global_condition,
                background_padding,
            )
        return self.output_projection(tokens).reshape(expected)


def cosine_beta_schedule(steps: int, offset: float = 0.008) -> torch.Tensor:
    points = np.linspace(0, steps, steps + 1, dtype=np.float64)
    alpha = np.cos(((points / steps) + offset) / (1.0 + offset) * np.pi / 2) ** 2
    alpha = alpha / alpha[0]
    beta = 1.0 - alpha[1:] / alpha[:-1]
    return torch.tensor(np.clip(beta, 1.0e-5, 0.999), dtype=torch.float32)


def _coefficient(
    values: torch.Tensor, timesteps: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    shape = (len(timesteps),) + (1,) * (target.ndim - 1)
    return values.gather(0, timesteps).reshape(shape)


class BackgroundTrajectoryDiffusion(nn.Module):
    """Masked DDPM objective with deterministic DDIM sampling.

    Supplying the same condition and ``initial_noise`` always produces the
    same trajectory.  That explicit latent interface is the future path-level
    subset-simulation boundary.
    """

    def __init__(self, config: DiffusionModelConfig) -> None:
        super().__init__()
        self.config = config
        self.denoiser = FactorizedSpatiotemporalDenoiser(config)
        beta = cosine_beta_schedule(config.diffusion_steps)
        alpha_bar = torch.cumprod(1.0 - beta, dim=0)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("sqrt_alpha_bar", torch.sqrt(alpha_bar))
        self.register_buffer("sqrt_one_minus_alpha_bar", torch.sqrt(1.0 - alpha_bar))

    def q_sample(
        self,
        clean: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        return (
            _coefficient(self.sqrt_alpha_bar, timesteps, clean) * clean
            + _coefficient(self.sqrt_one_minus_alpha_bar, timesteps, clean) * noise
        )

    def predict_clean(
        self,
        noisy: torch.Tensor,
        timesteps: torch.Tensor,
        model_output: torch.Tensor,
    ) -> torch.Tensor:
        alpha = _coefficient(self.alpha_bar, timesteps, noisy)
        if self.config.prediction_type == "v_prediction":
            return torch.sqrt(alpha) * noisy - torch.sqrt(1.0 - alpha) * model_output
        return (noisy - torch.sqrt(1.0 - alpha) * model_output) / torch.sqrt(alpha)

    def predict_noise(
        self,
        noisy: torch.Tensor,
        timesteps: torch.Tensor,
        model_output: torch.Tensor,
    ) -> torch.Tensor:
        if self.config.prediction_type == "epsilon":
            return model_output
        alpha = _coefficient(self.alpha_bar, timesteps, noisy)
        return torch.sqrt(1.0 - alpha) * noisy + torch.sqrt(alpha) * model_output

    def training_target(
        self,
        clean: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        if self.config.prediction_type == "epsilon":
            return noise
        alpha = _coefficient(self.alpha_bar, timesteps, clean)
        return torch.sqrt(alpha) * noise - torch.sqrt(1.0 - alpha) * clean

    def _denoise(
        self,
        noisy: torch.Tensor,
        timesteps: torch.Tensor,
        condition: torch.Tensor,
        *,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        scale = float(guidance_scale)
        if scale == 1.0:
            return self.denoiser(noisy, timesteps, condition)
        unconditional = self.denoiser(noisy, timesteps, torch.zeros_like(condition))
        if scale == 0.0:
            return unconditional
        conditional = self.denoiser(noisy, timesteps, condition)
        return unconditional + scale * (conditional - unconditional)

    def loss(
        self,
        clean: torch.Tensor,
        condition: torch.Tensor,
        target_mask: torch.Tensor,
        *,
        timesteps: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch = clean.shape[0]
        if timesteps is None:
            timesteps = torch.randint(
                self.config.diffusion_steps, (batch,), device=clean.device
            )
        mask = target_mask.to(dtype=clean.dtype)
        if noise is None:
            noise = torch.randn_like(clean)
        noise = noise * mask
        noisy = self.q_sample(clean, timesteps, noise) * mask
        denoiser_condition = condition
        if self.training and self.config.condition_dropout_prob > 0.0:
            dropped = (
                torch.rand(batch, device=clean.device)
                < self.config.condition_dropout_prob
            )
            denoiser_condition = condition.masked_fill(dropped[:, None], 0.0)
        model_output = self.denoiser(noisy, timesteps, denoiser_condition) * mask
        objective_target = self.training_target(clean, timesteps, noise) * mask
        per_sample_denominator = mask.flatten(1).sum(dim=1).clamp_min(1.0)
        per_sample_mse = ((model_output - objective_target).square() * mask).flatten(
            1
        ).sum(dim=1) / per_sample_denominator
        if self.config.min_snr_gamma > 0.0:
            alpha = self.alpha_bar.gather(0, timesteps)
            snr = alpha / (1.0 - alpha).clamp_min(1.0e-8)
            capped = torch.minimum(
                snr, snr.new_full(snr.shape, self.config.min_snr_gamma)
            )
            weight = (
                capped / (snr + 1.0)
                if self.config.prediction_type == "v_prediction"
                else capped / snr.clamp_min(1.0e-8)
            )
            denoising_mse = (per_sample_mse * weight).mean()
        else:
            denoising_mse = per_sample_mse.mean()
        denominator = mask.sum().clamp_min(1.0)
        predicted_clean = self.predict_clean(noisy, timesteps, model_output) * mask
        x0_l1 = (torch.abs(predicted_clean - clean) * mask).sum() / denominator
        if clean.shape[1] > 1:
            pair_mask = mask[:, 1:] * mask[:, :-1]
            pair_denominator = pair_mask.sum().clamp_min(1.0)
            smooth = (
                torch.abs(
                    (predicted_clean[:, 1:] - predicted_clean[:, :-1])
                    - (clean[:, 1:] - clean[:, :-1])
                )
                * pair_mask
            ).sum() / pair_denominator
        else:
            smooth = clean.new_zeros(())
        long_horizon = clean.new_zeros(())
        trajectory_position = clean.new_zeros(())
        trajectory_velocity = clean.new_zeros(())
        position_error = (predicted_clean - clean).reshape(
            batch, self.config.horizon_steps, 6, 2
        )
        scale = clean.new_tensor(
            (self.config.target_scale_x, self.config.target_scale_y)
        )
        position_error = position_error * scale
        agent_mask = mask[:, 0].reshape(batch, 6, 2)
        if self.config.long_horizon_weight > 0.0:
            knot_error = position_error[:, (49, 99, 148)].abs()
            knot_mask = agent_mask[:, None].expand_as(knot_error)
            long_horizon = (knot_error * knot_mask).sum() / knot_mask.sum().clamp_min(
                1.0
            )
        if (
            self.config.trajectory_position_weight > 0.0
            or self.config.trajectory_velocity_weight > 0.0
        ):
            series_mask = agent_mask[:, None].expand_as(position_error)
            series_denominator = series_mask.sum().clamp_min(1.0)
            trajectory_position = (
                F.smooth_l1_loss(
                    position_error,
                    torch.zeros_like(position_error),
                    reduction="none",
                )
                * series_mask
            ).sum() / series_denominator
            velocity_error_series = (
                torch.diff(
                    position_error,
                    dim=1,
                    prepend=torch.zeros_like(position_error[:, :1]),
                )
                / self.config.dt_s
            )
            trajectory_velocity = (
                F.smooth_l1_loss(
                    velocity_error_series,
                    torch.zeros_like(velocity_error_series),
                    reduction="none",
                )
                * series_mask
            ).sum() / series_denominator
        total = (
            denoising_mse
            + self.config.x0_weight * x0_l1
            + self.config.smooth_weight * smooth
            + self.config.long_horizon_weight * long_horizon
            + self.config.trajectory_position_weight * trajectory_position
            + self.config.trajectory_velocity_weight * trajectory_velocity
        )
        return {
            "loss": total,
            "denoising_mse": denoising_mse,
            "x0_l1": x0_l1,
            "smooth": smooth,
            "long_horizon": long_horizon,
            "trajectory_position": trajectory_position,
            "trajectory_velocity": trajectory_velocity,
        }

    @torch.no_grad()
    def sample_ddim(
        self,
        condition: torch.Tensor,
        target_mask: torch.Tensor,
        *,
        inference_steps: int = 20,
        initial_noise: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        x0_clip_abs: float | None = None,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        batch = condition.shape[0]
        shape = (batch, self.config.horizon_steps, self.config.target_dim)
        if initial_noise is None:
            initial_noise = torch.randn(
                shape,
                device=condition.device,
                dtype=condition.dtype,
                generator=generator,
            )
        if tuple(initial_noise.shape) != shape:
            raise ValueError(
                f"expected initial_noise {shape}, got {tuple(initial_noise.shape)}"
            )
        mask = target_mask.to(device=condition.device, dtype=condition.dtype)
        trajectory = initial_noise * mask
        requested = min(max(int(inference_steps), 2), self.config.diffusion_steps)
        schedule = sorted(
            {
                int(round(value))
                for value in np.linspace(0, self.config.diffusion_steps - 1, requested)
            }
        )
        for index in reversed(range(len(schedule))):
            step = schedule[index]
            previous = schedule[index - 1] if index else -1
            timesteps = torch.full(
                (batch,), step, device=condition.device, dtype=torch.long
            )
            model_output = (
                self._denoise(
                    trajectory,
                    timesteps,
                    condition,
                    guidance_scale=guidance_scale,
                )
                * mask
            )
            predicted_noise = (
                self.predict_noise(trajectory, timesteps, model_output) * mask
            )
            clean = self.predict_clean(trajectory, timesteps, model_output)
            clip = (
                self.config.x0_clip_abs if x0_clip_abs is None else float(x0_clip_abs)
            )
            clean = clean.clamp(-clip, clip)
            clean = clean * mask
            if previous < 0:
                trajectory = clean
            else:
                previous_alpha = self.alpha_bar[previous]
                trajectory = (
                    torch.sqrt(previous_alpha) * clean
                    + torch.sqrt(1.0 - previous_alpha) * predicted_noise
                ) * mask
        return trajectory
