"""Persistent-mode plus fresh-response control decoder."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class IntentResponseDecoderConfig:
    hidden_dim: int = 128
    control_limit_accel_mps2: float = 8.0
    control_limit_yaw_rate_rps: float = 0.8
    reference_control_scale: float = 1.0


class IntentResponseDecoder(nn.Module):
    """Decode ``u_mode + gate * u_response`` in `[acceleration, yaw_rate]`."""

    def __init__(self, cfg: IntentResponseDecoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        h = int(cfg.hidden_dim)
        self.mode = nn.Sequential(nn.Linear(h * 2 + 1, h), nn.SiLU(), nn.Linear(h, 2))
        self.response = nn.Sequential(nn.Linear(h * 2, h), nn.SiLU(), nn.Linear(h, 2))
        self.gate = nn.Sequential(nn.Linear(h * 2, h), nn.SiLU(), nn.Linear(h, 1))
        # Start from reference controls until data supports a correction.
        for head in (self.mode, self.response):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    def forward(
        self,
        agent_context: torch.Tensor,
        scene_context: torch.Tensor,
        state_context: torch.Tensor,
        elapsed_steps: torch.Tensor,
        agent_valid: torch.Tensor,
        reference_controls: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        b, n, h = agent_context.shape
        scene = scene_context[:, None, :].expand(b, n, h)
        latent = state_context[:, None, :].expand(b, n, h)
        elapsed = elapsed_steps.float().view(b, 1, 1).expand(b, n, 1) / 30.0
        mode = self.mode(torch.cat((agent_context, latent, elapsed), dim=-1))
        response = self.response(torch.cat((agent_context, scene), dim=-1))
        gate = torch.sigmoid(self.gate(torch.cat((agent_context, scene), dim=-1)))
        residual = mode + gate * response
        residual = torch.stack((
            torch.tanh(residual[..., 0]) * float(self.cfg.control_limit_accel_mps2),
            torch.tanh(residual[..., 1]) * float(self.cfg.control_limit_yaw_rate_rps),
        ), dim=-1)
        reference = torch.zeros_like(residual) if reference_controls is None else reference_controls
        controls = self._bounded_controls(torch.stack((
            float(self.cfg.reference_control_scale) * reference[..., 0] + residual[..., 0],
            float(self.cfg.reference_control_scale) * reference[..., 1] + residual[..., 1],
        ), dim=-1))
        controls = controls * agent_valid[..., None].float()
        return {
            "controls": controls,
            "mode_controls": mode, "response_controls": response, "response_gate": gate,
        }

    def _bounded_controls(self, controls: torch.Tensor) -> torch.Tensor:
        return torch.stack((
            controls[..., 0].clamp(-8.0, 4.0),
            controls[..., 1].clamp(
                -float(self.cfg.control_limit_yaw_rate_rps),
                float(self.cfg.control_limit_yaw_rate_rps),
            ),
        ), dim=-1)
