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
    use_intent_response: bool = True
    control_plan_steps: int = 1
    control_plan_as_jerk: bool = False
    simulation_dt_s: float = 0.04
    control_jerk_limit_accel_mps3: float = 8.0
    control_jerk_limit_yaw_accel_rps2: float = 0.5


class IntentResponseDecoder(nn.Module):
    """Decode ``u_mode + gate * u_response`` in `[acceleration, yaw_rate]`."""

    def __init__(self, cfg: IntentResponseDecoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        h = int(cfg.hidden_dim)
        self.mode = nn.Sequential(nn.Linear(h * 2 + 1, h), nn.SiLU(), nn.Linear(h, 2))
        self.response = nn.Sequential(nn.Linear(h * 2, h), nn.SiLU(), nn.Linear(h, 2))
        self.gate = nn.Sequential(nn.Linear(h * 2, h), nn.SiLU(), nn.Linear(h, 1))
        self.control_plan_steps = max(1, int(cfg.control_plan_steps))
        # A within-response control curve is optional.  Keeping the legacy
        # single-control path parameter-identical makes old checkpoints strict
        # and auditable; a curve checkpoint adds only this zero-initialized
        # residual head.  The model integrates every curve element at 25 Hz.
        # When enabled, the head emits bounded control derivatives and its
        # cumulative integral.  This has the same low-dimensional output size
        # as direct point controls but preserves a continuous physical curve.
        self.plan = None
        if self.control_plan_steps > 1:
            self.plan = nn.Sequential(
                nn.Linear(h * 3 + 1, h), nn.SiLU(), nn.Linear(h, self.control_plan_steps * 2),
            )
        # A zero residual is a stable, physically meaningful initialization:
        # continue the observed acceleration/yaw-rate until evidence for a
        # mode or response correction is learned.
        for head in (self.mode, self.response):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
        if self.plan is not None:
            nn.init.zeros_(self.plan[-1].weight)
            nn.init.zeros_(self.plan[-1].bias)

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
        if self.cfg.use_intent_response:
            response = self.response(torch.cat((agent_context, scene), dim=-1))
            gate = torch.sigmoid(self.gate(torch.cat((agent_context, scene), dim=-1)))
        else:
            # B2 duration-aware ablation: preserve the same mode decoder and
            # physics while removing every instantaneous response path.
            response = torch.zeros_like(mode)
            gate = torch.zeros((*mode.shape[:-1], 1), dtype=mode.dtype, device=mode.device)
        residual = mode + gate * response
        residual = torch.stack((
            torch.tanh(residual[..., 0]) * float(self.cfg.control_limit_accel_mps2),
            torch.tanh(residual[..., 1]) * float(self.cfg.control_limit_yaw_rate_rps),
        ), dim=-1)
        reference = torch.zeros_like(residual) if reference_controls is None else reference_controls
        controls = torch.stack((
            (float(self.cfg.reference_control_scale) * reference[..., 0] + residual[..., 0]).clamp(-8.0, 4.0),
            (float(self.cfg.reference_control_scale) * reference[..., 1] + residual[..., 1]).clamp(-float(self.cfg.control_limit_yaw_rate_rps), float(self.cfg.control_limit_yaw_rate_rps)),
        ), dim=-1)
        controls = controls * agent_valid[..., None].float()
        if self.plan is None:
            control_plan = controls.unsqueeze(-2)
        else:
            plan_input = torch.cat((agent_context, scene, latent, elapsed), dim=-1)
            raw_plan = self.plan(plan_input).view(b, n, self.control_plan_steps, 2)
            if self.cfg.control_plan_as_jerk:
                plan_delta = torch.stack((
                    torch.tanh(raw_plan[..., 0]) * float(self.cfg.control_jerk_limit_accel_mps3),
                    torch.tanh(raw_plan[..., 1]) * float(self.cfg.control_jerk_limit_yaw_accel_rps2),
                ), dim=-1)
                control_plan = controls.unsqueeze(-2) + torch.cumsum(
                    plan_delta * float(self.cfg.simulation_dt_s), dim=-2,
                )
            else:
                plan_delta = torch.stack((
                    torch.tanh(raw_plan[..., 0]) * float(self.cfg.control_limit_accel_mps2),
                    torch.tanh(raw_plan[..., 1]) * float(self.cfg.control_limit_yaw_rate_rps),
                ), dim=-1)
                control_plan = controls.unsqueeze(-2) + plan_delta
            control_plan = torch.stack((
                control_plan[..., 0].clamp(-8.0, 4.0),
                control_plan[..., 1].clamp(-float(self.cfg.control_limit_yaw_rate_rps), float(self.cfg.control_limit_yaw_rate_rps)),
            ), dim=-1)
            control_plan = control_plan * agent_valid[..., None, None].float()
            # Retain a representative per-response value for legacy callers;
            # SemiMarkovRelationalWorldModel consumes ``control_plan`` to
            # integrate the full curve rather than merely using this entry.
            controls = control_plan.mean(dim=-2)
        return {
            "controls": controls, "control_plan": control_plan,
            "mode_controls": mode, "response_controls": response, "response_gate": gate,
        }
