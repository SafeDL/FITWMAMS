"""Conditional normalizing flow for FIRM-WM's executed joint controls."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class _AffineCoupling(nn.Module):
    def __init__(
        self,
        dimensions: int,
        context_dim: int,
        mask: torch.Tensor,
        *,
        log_scale_limit: float,
    ) -> None:
        super().__init__()
        self.register_buffer("mask", mask.float())
        self.network = nn.Sequential(
            nn.Linear(dimensions + context_dim, context_dim),
            nn.SiLU(),
            nn.Linear(context_dim, context_dim),
            nn.SiLU(),
            nn.Linear(context_dim, dimensions * 2),
        )
        self.log_scale_limit = float(log_scale_limit)
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def _affine_parameters(
        self, fixed: torch.Tensor, context: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.network(torch.cat((fixed, context), dim=-1))
        shift, log_scale = output.chunk(2, -1)
        zero = self.network(torch.cat((torch.zeros_like(fixed), context), dim=-1))
        shift = shift - zero[:, : shift.shape[-1]]
        free = 1.0 - self.mask
        return shift * free, torch.tanh(log_scale) * self.log_scale_limit * free

    def forward(
        self, value: torch.Tensor, context: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        fixed = value * self.mask
        shift, log_scale = self._affine_parameters(fixed, context)
        out = fixed + (1.0 - self.mask) * (value * torch.exp(log_scale) + shift)
        return out, log_scale

    def inverse(
        self, value: torch.Tensor, context: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        fixed = value * self.mask
        shift, log_scale = self._affine_parameters(fixed, context)
        out = fixed + (1.0 - self.mask) * (value - shift) * torch.exp(-log_scale)
        return out, log_scale


class JointActionFlow(nn.Module):
    """Exact-density flow over the five executed 25 Hz controls of six cars."""

    def __init__(
        self,
        context_dim: int,
        *,
        execute_frames: int,
        agents: int = 6,
        layers: int = 4,
        max_jerk: tuple[float, float] = (5.0, 0.5),
    ) -> None:
        super().__init__()
        self.execute_frames = int(execute_frames)
        self.agents = int(agents)
        self.dimensions = self.execute_frames * self.agents * 2
        self.base_scale_min = 0.002
        self.base_scale_max = 0.060
        base = torch.arange(self.dimensions) % 2
        self.layers = nn.ModuleList(
            [
                _AffineCoupling(
                    self.dimensions,
                    context_dim,
                    (base if index % 2 == 0 else 1 - base),
                    log_scale_limit=0.080,
                )
                for index in range(int(layers))
            ]
        )
        # The conditional centre gives zero-noise rollouts a direct,
        # trainable closed-loop control prediction.  Coupling layers model a
        # density-preserving residual around it instead of learning both the
        # deterministic trajectory and stochastic variation from the same
        # zero-initialized shifts.
        self.base_location = nn.Sequential(
            nn.Linear(context_dim, context_dim),
            nn.SiLU(),
            nn.Linear(context_dim, self.dimensions),
        )
        nn.init.zeros_(self.base_location[-1].weight)
        nn.init.zeros_(self.base_location[-1].bias)
        self.base_scale = nn.Sequential(
            nn.Linear(context_dim, context_dim),
            nn.SiLU(),
            nn.Linear(context_dim, self.dimensions),
        )
        nn.init.zeros_(self.base_scale[-1].weight)
        relative = (0.040 - self.base_scale_min) / (self.base_scale_max - self.base_scale_min)
        nn.init.constant_(self.base_scale[-1].bias, math.log(relative / (1.0 - relative)))
        scale = torch.tensor(max_jerk, dtype=torch.float32).repeat(
            self.execute_frames * self.agents
        )
        self.register_buffer("jerk_scale", scale)

    def _flat(self, value: torch.Tensor) -> torch.Tensor:
        return value.reshape(value.shape[0], self.dimensions)

    def _unflat(self, value: torch.Tensor) -> torch.Tensor:
        return value.reshape(value.shape[0], self.execute_frames, self.agents, 2)

    def _base_noise_scale(self, context: torch.Tensor) -> torch.Tensor:
        return self.base_scale_min + (self.base_scale_max - self.base_scale_min) * torch.sigmoid(
            self.base_scale(context)
        )

    def sample(
        self,
        context: torch.Tensor,
        noise: torch.Tensor,
        *,
        center: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Map a standard-normal joint noise draw to bounded jerk values."""
        value = self._flat(noise)
        location = self.base_location(context) if center is None else self._flat(center)
        for layer in self.layers:
            value, _ = layer(value, context)
        value = location + self._base_noise_scale(context) * value
        normalized = torch.tanh(value)
        return self._unflat(normalized * self.jerk_scale)

    def nll(
        self,
        jerks: torch.Tensor,
        valid: torch.Tensor,
        context: torch.Tensor,
        *,
        center: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Negative conditional log likelihood per valid executed control."""
        score_context = context.detach()
        flat = self._flat(jerks) / self.jerk_scale
        flat = flat.clamp(-0.999, 0.999)
        value = torch.atanh(flat)
        output_log_det = torch.log1p(-flat.square()).clamp_min(-20.0) + self.jerk_scale.log()
        base_scale = self._base_noise_scale(score_context)
        location = self.base_location(score_context) if center is None else self._flat(center)
        location = location.detach()
        flow_log_det = torch.zeros_like(value)
        value = (value - location) / base_scale
        for layer in reversed(self.layers):
            value, update = layer.inverse(value, score_context)
            flow_log_det = flow_log_det + update
        base = -0.5 * (value.square() + math.log(2.0 * math.pi))
        log_probability = (
            base
            - flow_log_det
            - output_log_det
            - base_scale.log()
        )
        mask = self._flat(valid[..., None].expand_as(jerks)).bool()
        return -(log_probability * mask.float()).sum() / mask.float().sum().clamp_min(1.0)
