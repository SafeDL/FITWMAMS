"""Anchor scenario-condition action diffusion prior."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion.src.features import CUTIN_SCENARIO_CONDITION_KEYS


CUTIN_FINAL_LATERAL_OFFSET_IDX = CUTIN_SCENARIO_CONDITION_KEYS.index(
    "final_lateral_offset"
)
CUTIN_TIME_TO_CROSS_IDX = CUTIN_SCENARIO_CONDITION_KEYS.index("time_to_cross")


@dataclass
class ActionDiffusionConfig:
    scenario_condition_dim: int
    horizon_steps: int
    action_dim: int
    dt: float = 0.04
    hidden_dim: int = 128
    num_layers: int = 4
    num_heads: int = 4
    dropout: float = 0.1
    diffusion_steps: int = 100
    x0_clip_abs: float = 0.0
    x0_weight: float = 0.0
    smooth_weight: float = 0.0
    action_representation: str = "acceleration"


def sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    if half == 0:
        return timesteps.float().unsqueeze(-1)
    freqs = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, device=timesteps.device).float()
        / max(half - 1, 1)
    )
    args = timesteps.float().unsqueeze(-1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class ScenarioConditionEncoder(nn.Module):
    def __init__(self, cfg: ActionDiffusionConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.net = nn.Sequential(
            nn.Linear(cfg.scenario_condition_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )

    def forward(self, scenario_conditions: torch.Tensor) -> torch.Tensor:
        if scenario_conditions.ndim != 2:
            raise ValueError(
                "scenario_conditions must have shape [batch, condition_dim], "
                f"got {tuple(scenario_conditions.shape)}"
            )
        if int(scenario_conditions.shape[1]) != int(self.cfg.scenario_condition_dim):
            raise ValueError(
                "Expected scenario_condition_dim="
                f"{self.cfg.scenario_condition_dim}, got {scenario_conditions.shape[1]}"
            )
        return self.net(scenario_conditions)


class FiLMTransformerBlock(nn.Module):
    def __init__(self, cfg: ActionDiffusionConfig) -> None:
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim,
            nhead=max(1, cfg.num_heads),
            dim_feedforward=cfg.hidden_dim * 4,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.film = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim * 2),
        )

    def forward(self, tokens: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        tokens = self.layer(tokens)
        gamma, beta = self.film(cond).chunk(2, dim=-1)
        return tokens * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)


class ActionDenoiser(nn.Module):
    """Predict diffusion noise for an action sequence."""

    def __init__(self, cfg: ActionDiffusionConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.cond_encoder = ScenarioConditionEncoder(cfg)
        self.action_proj = nn.Linear(cfg.action_dim, cfg.hidden_dim)
        self.action_pos = nn.Parameter(torch.zeros(1, cfg.horizon_steps, cfg.hidden_dim))
        self.timestep_mlp = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )
        self.layers = nn.ModuleList(
            [FiLMTransformerBlock(cfg) for _ in range(max(1, cfg.num_layers))]
        )
        self.out = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim),
            nn.Linear(cfg.hidden_dim, cfg.action_dim),
        )

    def forward(
        self,
        noisy_actions: torch.Tensor,
        timesteps: torch.Tensor,
        scenario_conditions: torch.Tensor,
    ) -> torch.Tensor:
        if noisy_actions.shape[1] != self.cfg.horizon_steps:
            raise ValueError(
                f"Expected horizon={self.cfg.horizon_steps}, got {noisy_actions.shape[1]}"
            )
        cond = self.cond_encoder(scenario_conditions)
        t_emb = self.timestep_mlp(sinusoidal_embedding(timesteps, self.cfg.hidden_dim))
        cond = cond + t_emb
        tokens = self.action_proj(noisy_actions) + self.action_pos + cond.unsqueeze(1)
        for layer in self.layers:
            tokens = layer(tokens, cond)
        return self.out(tokens)


def cosine_beta_schedule(
    timesteps: int,
    s: float = 0.008,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Cosine beta schedule for DDPM noise training."""
    steps = int(timesteps) + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.tensor(np.clip(betas, 1e-5, 0.999), dtype=dtype)


def extract_coeff(coeff: torch.Tensor, timesteps: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    b = timesteps.shape[0]
    out = coeff.gather(0, timesteps)
    return out.reshape(b, *((1,) * (len(shape) - 1)))


def _alpha_at(
    alphas_cumprod: torch.Tensor,
    timesteps: torch.Tensor,
    shape: torch.Size,
) -> torch.Tensor:
    safe_t = torch.clamp(timesteps, min=0)
    value = extract_coeff(alphas_cumprod, safe_t, shape)
    one = torch.ones_like(value)
    mask = (timesteps < 0).view(-1, *((1,) * (len(shape) - 1)))
    return torch.where(mask, one, value)


class GaussianActionDiffusion(nn.Module):
    def __init__(self, denoiser: ActionDenoiser, diffusion_steps: int) -> None:
        super().__init__()
        self.denoiser = denoiser
        betas = cosine_beta_schedule(diffusion_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])
        self.num_steps = int(diffusion_steps)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1.0))
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_log_variance_clipped", torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer("posterior_mean_coef1", betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

    def q_sample(self, x_start: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return (
            extract_coeff(self.sqrt_alphas_cumprod, timesteps, x_start.shape) * x_start
            + extract_coeff(self.sqrt_one_minus_alphas_cumprod, timesteps, x_start.shape) * noise
        )

    def predict_start_from_noise(self, x_t: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return (
            extract_coeff(self.sqrt_recip_alphas_cumprod, timesteps, x_t.shape) * x_t
            - extract_coeff(self.sqrt_recipm1_alphas_cumprod, timesteps, x_t.shape) * noise
        )

    def maybe_clip_x0(self, x0: torch.Tensor) -> torch.Tensor:
        clip = float(getattr(self.denoiser.cfg, "x0_clip_abs", 0.0))
        if clip <= 0.0:
            return x0
        return torch.clamp(x0, -clip, clip)

    def ddim_step(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        prev_timesteps: torch.Tensor,
        eps: torch.Tensor,
    ) -> torch.Tensor:
        x0 = self.maybe_clip_x0(self.predict_start_from_noise(x_t, timesteps, eps))
        alpha_prev = _alpha_at(self.alphas_cumprod, prev_timesteps, x_t.shape)
        return (
            torch.sqrt(torch.clamp(alpha_prev, min=0.0)) * x0
            + torch.sqrt(torch.clamp(1.0 - alpha_prev, min=0.0)) * eps
        )

    def p_losses(
        self,
        actions: torch.Tensor,
        scenario_conditions: torch.Tensor,
        trajectory_context: dict[str, torch.Tensor] | None = None,
        trajectory_loss_cfg: dict[str, Any] | None = None,
    ) -> dict[str, torch.Tensor]:
        b = actions.shape[0]
        t = torch.randint(0, self.num_steps, (b,), device=actions.device, dtype=torch.long)
        noise = torch.randn_like(actions)
        noisy = self.q_sample(actions, t, noise)
        pred = self.denoiser(noisy, t, scenario_conditions)
        noise_mse = F.mse_loss(pred, noise)
        x0 = self.predict_start_from_noise(noisy, t, pred)
        x0_l1 = F.l1_loss(x0, actions)
        if x0.shape[1] > 1:
            smooth = torch.mean(torch.abs(x0[:, 1:] - x0[:, :-1]))
        else:
            smooth = torch.zeros((), device=actions.device, dtype=actions.dtype)
        loss = (
            noise_mse
            + self.denoiser.cfg.x0_weight * x0_l1
            + self.denoiser.cfg.smooth_weight * smooth
        )
        out = {
            "loss": loss,
            "noise_mse": noise_mse.detach(),
            "x0_l1": x0_l1.detach(),
            "smooth": smooth.detach(),
        }
        if trajectory_context is not None and trajectory_loss_cfg is not None:
            cutin_losses = cutin_trajectory_constraint_losses(
                x0,
                trajectory_context,
                trajectory_loss_cfg,
                dt=float(self.denoiser.cfg.dt),
            )
            weighted = torch.zeros((), device=actions.device, dtype=actions.dtype)
            for key, value in cutin_losses.items():
                if not key.endswith("_loss"):
                    continue
                weight_key = f"{key[:-5]}_weight"
                weight = float(trajectory_loss_cfg.get(weight_key, 0.0))
                if weight > 0.0:
                    weighted = weighted + weight * value
            out["loss"] = out["loss"] + weighted
            out["cutin_constraint_loss"] = weighted.detach()
            for key, value in cutin_losses.items():
                out[key] = value.detach()
        return out

    @torch.no_grad()
    def p_sample(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        scenario_conditions: torch.Tensor,
    ) -> torch.Tensor:
        eps = self.denoiser(x_t, timesteps, scenario_conditions)
        x0 = self.maybe_clip_x0(self.predict_start_from_noise(x_t, timesteps, eps))
        mean = (
            extract_coeff(self.posterior_mean_coef1, timesteps, x_t.shape) * x0
            + extract_coeff(self.posterior_mean_coef2, timesteps, x_t.shape) * x_t
        )
        log_var = extract_coeff(self.posterior_log_variance_clipped, timesteps, x_t.shape)
        noise = torch.randn_like(x_t)
        mask = (timesteps != 0).float().reshape(x_t.shape[0], *((1,) * (x_t.ndim - 1)))
        return mean + mask * torch.exp(0.5 * log_var) * noise

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        scenario_conditions: torch.Tensor,
    ) -> torch.Tensor:
        cfg = self.denoiser.cfg
        x = torch.randn(
            batch_size,
            cfg.horizon_steps,
            cfg.action_dim,
            device=scenario_conditions.device,
        )
        for i in reversed(range(self.num_steps)):
            t = torch.full((batch_size,), i, device=scenario_conditions.device, dtype=torch.long)
            x = self.p_sample(x, t, scenario_conditions)
        return x

    @torch.no_grad()
    def sample_ddim(
        self,
        batch_size: int,
        scenario_conditions: torch.Tensor,
        *,
        inference_steps: int | None = None,
    ) -> torch.Tensor:
        cfg = self.denoiser.cfg
        steps = self.num_steps if inference_steps is None else int(inference_steps)
        steps = min(max(steps, 1), self.num_steps)
        raw_steps = np.linspace(0, self.num_steps - 1, steps)
        timesteps = sorted({int(round(step)) for step in raw_steps})
        x = torch.randn(
            batch_size,
            cfg.horizon_steps,
            cfg.action_dim,
            device=scenario_conditions.device,
        )
        for idx in reversed(range(len(timesteps))):
            step = int(timesteps[idx])
            prev_step = int(timesteps[idx - 1]) if idx > 0 else -1
            t = torch.full((batch_size,), step, device=scenario_conditions.device, dtype=torch.long)
            prev_t = torch.full_like(t, prev_step)
            eps = self.denoiser(x, t, scenario_conditions)
            x = self.ddim_step(x, t, prev_t, eps)
        return x

    def sample_ddim_with_guidance(
        self,
        batch_size: int,
        scenario_conditions: torch.Tensor,
        *,
        inference_steps: int | None = None,
        guidance_scale: float = 0.1,
        guidance_context: dict[str, torch.Tensor] | None = None,
        guidance_config: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """DDIM sampling with gradient-based validity guidance.

        At each step the predicted x0 is scored by cut-in semantic constraints
        and the gradient nudges the denoised estimate toward valid lane changes.

        ``guidance_scale=0`` recovers standard DDIM sampling.
        """
        cfg = self.denoiser.cfg
        steps = self.num_steps if inference_steps is None else int(inference_steps)
        steps = min(max(steps, 1), self.num_steps)
        raw_steps = np.linspace(0, self.num_steps - 1, steps)
        timesteps = sorted({int(round(step)) for step in raw_steps})

        device = scenario_conditions.device
        x = torch.randn(batch_size, cfg.horizon_steps, cfg.action_dim, device=device)

        for idx in reversed(range(len(timesteps))):
            step = int(timesteps[idx])
            prev_step = int(timesteps[idx - 1]) if idx > 0 else -1
            t = torch.full((batch_size,), step, device=device, dtype=torch.long)
            prev_t = torch.full_like(t, prev_step)

            with torch.no_grad():
                eps = self.denoiser(x, t, scenario_conditions)

            if guidance_scale > 0.0:
                with torch.enable_grad():
                    x_grad = x.detach().requires_grad_(True)
                    eps_grad = self.denoiser(x_grad, t, scenario_conditions)
                    x0_pred = self.maybe_clip_x0(
                        self.predict_start_from_noise(x_grad, t, eps_grad)
                    )
                    score = cutin_semantic_guidance_score(
                        x0_pred,
                        guidance_context=guidance_context,
                        guidance_config=guidance_config,
                        dt=float(cfg.dt),
                    )
                    (grad,) = torch.autograd.grad(
                        outputs=score.sum(),
                        inputs=x_grad,
                        create_graph=False,
                        retain_graph=False,
                    )
                if grad is not None:
                    sqrt_om_alpha = extract_coeff(
                        self.sqrt_one_minus_alphas_cumprod, t, x.shape
                    )
                    eps = eps - guidance_scale * sqrt_om_alpha * grad.detach()

            with torch.no_grad():
                x = self.ddim_step(x, t, prev_t, eps)
        return x


def _decode_actions_from_context(
    actions: torch.Tensor,
    context: dict[str, torch.Tensor],
) -> torch.Tensor:
    mean = context.get("action_mean")
    std = context.get("action_std")
    if mean is None or std is None:
        return actions
    return actions * std.to(actions.device, actions.dtype) + mean.to(actions.device, actions.dtype)


def _integrate_cutin_actions_torch(
    initial_states: torch.Tensor,
    actions: torch.Tensor,
    *,
    dt: float = 0.04,
    ax_min: float = -8.0,
    ax_max: float = 4.0,
    ay_abs_max: float = 4.0,
    speed_min: float = 0.0,
    speed_max: float = 50.0,
) -> torch.Tensor:
    ctx = initial_states.to(actions.device, actions.dtype)
    seq = actions
    initial = ctx[:, 1]
    x = initial[:, 0]
    y = initial[:, 1]
    vx = initial[:, 2]
    vy = initial[:, 3]
    states: list[torch.Tensor] = []
    dt_f = float(dt)
    for step in range(seq.shape[1]):
        ax = torch.clamp(seq[:, step, 0], float(ax_min), float(ax_max))
        ay = torch.clamp(seq[:, step, 1], -float(ay_abs_max), float(ay_abs_max))
        x = x + vx * dt_f + 0.5 * ax * dt_f * dt_f
        y = y + vy * dt_f + 0.5 * ay * dt_f * dt_f
        vx = torch.clamp(vx + ax * dt_f, float(speed_min), float(speed_max))
        vy = vy + ay * dt_f
        states.append(torch.stack([x, y, vx, vy, ax, ay], dim=-1))
    return torch.stack(states, dim=1)


def _batch_gather_time(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    batch = values.shape[0]
    safe = torch.clamp(indices.to(values.device), 0, values.shape[1] - 1)
    return values[torch.arange(batch, device=values.device), safe]


def cutin_trajectory_constraint_losses(
    x0: torch.Tensor,
    context: dict[str, torch.Tensor],
    cfg: dict[str, Any],
    *,
    dt: float = 0.04,
) -> dict[str, torch.Tensor]:
    """Trajectory and semantic losses used only for cut-in diffusion training."""
    actions = _decode_actions_from_context(x0, context)
    initial_states = context["initial_states"].to(actions.device, actions.dtype)
    raw_conditions = context.get("scenario_conditions")

    pred = _integrate_cutin_actions_torch(
        initial_states,
        actions,
        dt=dt,
        ax_min=float(cfg.get("ax_min", -8.0)),
        ax_max=float(cfg.get("ax_max", 4.0)),
        ay_abs_max=float(cfg.get("ay_abs_max", 4.0)),
        speed_min=float(cfg.get("speed_min", 0.0)),
        speed_max=float(cfg.get("speed_max", 50.0)),
    )
    ego_y = initial_states[:, 0, 1].unsqueeze(1)
    rel_y = pred[:, :, 1] - ego_y
    final_rel_y = rel_y[:, -1]
    out: dict[str, torch.Tensor] = {}

    if raw_conditions is not None:
        cond = raw_conditions.to(actions.device, actions.dtype)
        target_final_y = cond[:, CUTIN_FINAL_LATERAL_OFFSET_IDX]
    else:
        target_final_y = torch.zeros_like(final_rel_y)

    lane_threshold = float(
        cfg.get("cutin_lateral_offset", cfg.get("lateral_overlap_threshold", 1.0))
    )
    final_lane_steps = min(
        pred.shape[1],
        max(
            1,
            int(
                round(
                    float(cfg.get("post_lane_window_seconds", 0.5))
                    / max(float(dt), 1.0e-6)
                )
            ),
        ),
    )

    out["end_y_loss"] = F.smooth_l1_loss(final_rel_y, target_final_y)
    out["post_lane_loss"] = torch.mean(
        torch.relu(torch.abs(rel_y[:, -final_lane_steps:]) - lane_threshold)
    )
    if pred.shape[1] > 1:
        lateral_jerk = torch.diff(actions[:, :, 1], dim=1) / max(float(dt), 1.0e-6)
        jerk_limit = float(cfg.get("lateral_jerk_abs_max", 8.0))
        out["lateral_jerk_loss"] = torch.mean(torch.relu(torch.abs(lateral_jerk) - jerk_limit))
    else:
        out["lateral_jerk_loss"] = torch.zeros((), device=actions.device, dtype=actions.dtype)

    return out


def cutin_semantic_guidance_score(
    x0: torch.Tensor,
    *,
    guidance_context: dict[str, torch.Tensor] | None = None,
    guidance_config: dict[str, Any] | None = None,
    dt: float = 0.04,
) -> torch.Tensor:
    """Higher-is-better differentiable score for semantic cut-in DDIM guidance.

    With ``guidance_context`` this integrates decoded target actions from the
    actual initial state. Without context it falls back to action-only scoring.
    """
    cfg = guidance_config or {}
    if x0.dim() != 3 or x0.shape[-1] < 2:
        raise ValueError(f"Expected x0 shape [B, H, >=2], got {tuple(x0.shape)}")
    if guidance_context is None or "initial_states" not in guidance_context:
        ay = x0[..., 1]
        vy = torch.cumsum(ay * float(dt), dim=-1)
        lateral_disp = torch.cumsum(vy * float(dt), dim=-1)
        final_disp_abs = torch.abs(lateral_disp[:, -1])
        lateral_activity = -torch.relu(float(cfg.get("min_lateral_displacement", 1.5)) - final_disp_abs)
        lateral_magnitude = -torch.relu(final_disp_abs - float(cfg.get("max_lateral_displacement", 5.625)))
        lateral_settle = -torch.mean(torch.abs(ay[:, -5:]), dim=-1)
        return lateral_activity + lateral_magnitude + 2.0 * lateral_settle

    score = torch.zeros(x0.shape[0], device=x0.device, dtype=x0.dtype)
    actions = _decode_actions_from_context(x0, guidance_context)
    initial_states = guidance_context["initial_states"].to(actions.device, actions.dtype)
    cond = guidance_context.get("scenario_conditions")
    pred = _integrate_cutin_actions_torch(
        initial_states,
        actions,
        dt=dt,
        ax_min=float(cfg.get("ax_min", -8.0)),
        ax_max=float(cfg.get("ax_max", 4.0)),
        ay_abs_max=float(cfg.get("ay_abs_max", 4.0)),
        speed_min=float(cfg.get("speed_min", 0.0)),
        speed_max=float(cfg.get("speed_max", 50.0)),
    )
    rel_y = pred[:, :, 1] - initial_states[:, 0, 1].unsqueeze(1)
    if cond is not None:
        cond_t = cond.to(actions.device, actions.dtype)
        target_final_y = cond_t[:, CUTIN_FINAL_LATERAL_OFFSET_IDX]
        cross_idx = torch.clamp(
            torch.round(
                cond_t[:, CUTIN_TIME_TO_CROSS_IDX] / max(float(dt), 1.0e-6)
            ).long() - 1,
            0,
            pred.shape[1] - 1,
        )
    else:
        target_final_y = torch.zeros_like(rel_y[:, -1])
        cross_idx = torch.full((x0.shape[0],), pred.shape[1] // 4, device=x0.device, dtype=torch.long)
    cross_rel_y = _batch_gather_time(rel_y, cross_idx)
    lane_threshold = float(cfg.get("cutin_lateral_offset", cfg.get("lateral_overlap_threshold", 1.0)))
    final_term = -torch.abs(rel_y[:, -1] - target_final_y)
    cross_term = -torch.relu(torch.abs(cross_rel_y) - float(cfg.get("lateral_overlap_threshold", 1.0)))
    final_lane_steps = min(
        pred.shape[1],
        max(
            1,
            int(
                round(
                    float(cfg.get("guidance_final_lane_window_seconds", 0.5))
                    / max(float(dt), 1.0e-6)
                )
            ),
        ),
    )
    post_term = -torch.mean(
        torch.relu(torch.abs(rel_y[:, -final_lane_steps:]) - lane_threshold),
        dim=1,
    )
    front_term = score
    ego_length = guidance_context.get("ego_length")
    adv_length = guidance_context.get("adv_length")
    if ego_length is not None and adv_length is not None:
        ego_x = initial_states[:, 0, 0].unsqueeze(1) + initial_states[:, 0, 2].unsqueeze(1) * (
            torch.arange(1, pred.shape[1] + 1, device=x0.device, dtype=x0.dtype).unsqueeze(0)
            * float(dt)
        )
        gap = pred[:, :, 0] - ego_x - 0.5 * (
            ego_length.to(x0.device, x0.dtype).unsqueeze(1)
            + adv_length.to(x0.device, x0.dtype).unsqueeze(1)
        )
        cross_gap = _batch_gather_time(gap, cross_idx)
        front_term = -torch.relu(float(cfg.get("min_cutin_front_gap", 0.0)) - cross_gap)
    if actions.shape[1] > 1:
        jerk = torch.diff(actions[:, :, 1], dim=1) / max(float(dt), 1.0e-6)
        jerk_term = -torch.mean(torch.relu(torch.abs(jerk) - float(cfg.get("lateral_jerk_abs_max", 8.0))), dim=1)
    else:
        jerk_term = score
    return (
        float(cfg.get("guidance_end_y_weight", 1.0)) * final_term
        + float(cfg.get("guidance_cross_y_weight", 1.0)) * cross_term
        + float(cfg.get("guidance_post_lane_weight", 1.0)) * post_term
        + float(cfg.get("guidance_front_at_cross_weight", 1.0)) * front_term
        + float(cfg.get("guidance_lateral_jerk_weight", 0.2)) * jerk_term
    )


def build_model_from_schema(schema: dict, config: dict) -> GaussianActionDiffusion:
    if str(schema.get("conditioning_mode", "")) != "anchor_scenario":
        raise RuntimeError(
            "Diffusion schema is not an anchor-scenario-condition dataset. "
            "Rebuild the dataset and retrain checkpoints with the current pipeline."
        )
    model_cfg = config.get("model", {})
    diffusion_cfg = config.get("diffusion", {})
    cfg = ActionDiffusionConfig(
        scenario_condition_dim=len(schema["condition_keys"]),
        horizon_steps=int(schema["horizon_steps"]),
        action_dim=len(schema["action_keys"]),
        dt=float(schema.get("dt", 0.04)),
        hidden_dim=int(model_cfg.get("hidden_dim", 128)),
        num_layers=int(model_cfg.get("num_layers", 4)),
        num_heads=int(model_cfg.get("num_heads", 4)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        diffusion_steps=int(diffusion_cfg.get("steps", 100)),
        x0_clip_abs=float(diffusion_cfg.get("x0_clip_abs", 0.0)),
        x0_weight=float(config.get("loss", {}).get("x0_weight", 0.0)),
        smooth_weight=float(config.get("loss", {}).get("smooth_weight", 0.0)),
        action_representation=str(schema["action_representation"]),
    )
    return GaussianActionDiffusion(ActionDenoiser(cfg), diffusion_steps=cfg.diffusion_steps)
