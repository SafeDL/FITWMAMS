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
    plan_horizon_frames: int = 1
    execute_frames: int = 5


class IntentResponseDecoder(nn.Module):
    """Decode persistent intent plus local response controls.

    ``plan_horizon_frames=1`` preserves the BARS-M1 interface exactly.  The
    M2 path keeps those trained heads and adds zero-initialized temporal terms,
    so its initial one-second plan is the M1 control repeated over the horizon.
    """

    def __init__(self, cfg: IntentResponseDecoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        h = int(cfg.hidden_dim)
        self.mode = nn.Sequential(nn.Linear(h * 2 + 1, h), nn.SiLU(), nn.Linear(h, 2))
        self.response = nn.Sequential(nn.Linear(h * 2, h), nn.SiLU(), nn.Linear(h, 2))
        self.gate = nn.Sequential(nn.Linear(h * 2, h), nn.SiLU(), nn.Linear(h, 1))
        self.plan_horizon_frames = int(cfg.plan_horizon_frames)
        self.execute_frames = min(int(cfg.execute_frames), self.plan_horizon_frames)
        if self.plan_horizon_frames < 1:
            raise ValueError("plan_horizon_frames must be positive")
        if self.execute_frames < 1:
            raise ValueError("execute_frames must be positive")
        if self.plan_horizon_frames > 1:
            self.plan_time = nn.Embedding(self.plan_horizon_frames, h)
            # A scene-level intent is shared across agents at each planned
            # instant; the persistent per-agent mode below preserves each
            # vehicle's established role within that scene-level intent.
            self.intent_time = nn.Sequential(nn.Linear(h * 3, h), nn.SiLU(), nn.Linear(h, 2))
            # This is deliberately one light joint interaction after time
            # expansion, rather than a separate per-vehicle latent variable.
            self.cross_vehicle = nn.Sequential(nn.Linear(h, h), nn.SiLU(), nn.Linear(h, h))
            self.local_time = nn.Sequential(nn.Linear(h * 3, h), nn.SiLU(), nn.Linear(h, 2))
        # Start from reference controls until data supports a correction.
        for head in (self.mode, self.response):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
        if self.plan_horizon_frames > 1:
            # With transferred M1 mode/response/gate weights these two heads
            # produce exactly zero, so the plan begins as the M1 response
            # repeated for one second.  This makes the first M2 stage a
            # controlled temporal extension rather than a decoder restart.
            for head in (self.intent_time, self.local_time):
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
        *,
        suppress_residual: bool = False,
    ) -> dict[str, torch.Tensor]:
        b, n, h = agent_context.shape
        scene = scene_context[:, None, :].expand(b, n, h)
        latent = state_context[:, None, :].expand(b, n, h)
        elapsed = elapsed_steps.float().view(b, 1, 1).expand(b, n, 1) / 30.0
        mode = self.mode(torch.cat((agent_context, latent, elapsed), dim=-1))
        response = self.response(torch.cat((agent_context, scene), dim=-1))
        gate = torch.sigmoid(self.gate(torch.cat((agent_context, scene), dim=-1)))
        if self.plan_horizon_frames == 1:
            raw = torch.zeros_like(mode) if suppress_residual else mode + gate * response
            residual = self._squash_residual(raw)
            reference = torch.zeros_like(residual) if reference_controls is None else reference_controls
            controls = self._bounded_controls(float(self.cfg.reference_control_scale) * reference + residual)
            controls = controls * agent_valid[..., None].float()
            return {
                "controls": controls,
                "mode_controls": mode, "response_controls": response, "response_gate": gate,
                "control_plan": controls[:, None], "forecast_control_plan": controls[:, None], "applied_controls": controls[:, None],
                "intent_plan": residual[:, None], "local_residual_plan": torch.zeros_like(residual[:, None]),
            }

        time_index = torch.arange(self.plan_horizon_frames, device=agent_context.device)
        time = self.plan_time(time_index).view(1, self.plan_horizon_frames, 1, h).expand(b, -1, n, -1)
        scene_time = scene_context[:, None, None, :].expand(b, self.plan_horizon_frames, n, h)
        latent_time = state_context[:, None, None, :].expand(b, self.plan_horizon_frames, n, h)
        agent_time = agent_context[:, None, :, :].expand(b, self.plan_horizon_frames, n, h)
        joint = self.cross_vehicle(agent_context.mean(dim=1))[:, None, None, :].expand(b, self.plan_horizon_frames, n, h)
        intent_offset = self.intent_time(torch.cat((scene_time, latent_time, time), dim=-1))
        local_offset = self.local_time(torch.cat((agent_time, joint, time), dim=-1))
        # Replanning executes only the first 0.2 s.  Temporal offsets are
        # forecasts only, leaving the transferred M1 execution prefix intact.
        forecast = (time_index >= self.execute_frames).view(1, self.plan_horizon_frames, 1, 1)
        temporal_mix = forecast.to(dtype=intent_offset.dtype)
        forecast_intent_raw = mode[:, None] + intent_offset
        forecast_local_raw = gate[:, None] * response[:, None] + local_offset
        intent_raw = mode[:, None] + intent_offset * temporal_mix
        local_raw = gate[:, None] * response[:, None] + local_offset * temporal_mix
        if suppress_residual:
            # B0 plus the anchor residual remains the exact executed START
            # prefix.  The unexecuted remainder is still a learned forecast,
            # which lets Stage A train overlap plans without perturbing B0.
            intent_raw = intent_raw.clone()
            local_raw = local_raw.clone()
            forecast_intent_raw = forecast_intent_raw.clone()
            forecast_local_raw = forecast_local_raw.clone()
            intent_raw[:, : self.execute_frames] = 0.0
            local_raw[:, : self.execute_frames] = 0.0
            forecast_intent_raw[:, : self.execute_frames] = 0.0
            forecast_local_raw[:, : self.execute_frames] = 0.0
        total_residual = self._squash_residual(intent_raw + local_raw)
        forecast_total_residual = self._squash_residual(forecast_intent_raw + forecast_local_raw)
        intent_plan = self._squash_residual(intent_raw)
        # Define the local contribution in control space so the two audit
        # tensors sum exactly to the residual actually applied by dynamics.
        local_plan = total_residual - intent_plan
        if reference_controls is None:
            reference = torch.zeros_like(total_residual)
        elif reference_controls.ndim == 3:
            reference = reference_controls[:, None].expand(-1, self.plan_horizon_frames, -1, -1)
        elif reference_controls.ndim == 4 and reference_controls.shape[1] == self.plan_horizon_frames:
            reference = reference_controls
        elif reference_controls.ndim == 4 and reference_controls.shape[1] == self.execute_frames:
            # START's audited residual is known at physics resolution for the
            # next response interval only.  Preserve those executable frames
            # exactly and use its terminal value merely as a non-B0 plan
            # placeholder beyond the observed prefix.
            reference = reference_controls[:, -1:].expand(-1, self.plan_horizon_frames, -1, -1).clone()
            reference[:, : self.execute_frames] = reference_controls
        else:
            raise ValueError("reference_controls must be [B, agents, 2] or align with the plan horizon")
        control_plan = self._bounded_controls(float(self.cfg.reference_control_scale) * reference + total_residual)
        forecast_control_plan = self._bounded_controls(
            float(self.cfg.reference_control_scale) * reference + forecast_total_residual
        )
        control_plan = control_plan * agent_valid[:, None, :, None].float()
        forecast_control_plan = forecast_control_plan * agent_valid[:, None, :, None].float()
        applied = control_plan[:, : self.execute_frames]
        return {
            "controls": applied,
            "control_plan": control_plan,
            "forecast_control_plan": forecast_control_plan,
            "applied_controls": applied,
            "intent_plan": intent_plan * agent_valid[:, None, :, None].float(),
            "local_residual_plan": local_plan * agent_valid[:, None, :, None].float(),
            "mode_controls": mode,
            "response_controls": response,
            "response_gate": gate,
        }

    def _squash_residual(self, residual: torch.Tensor) -> torch.Tensor:
        return torch.stack((
            torch.tanh(residual[..., 0]) * float(self.cfg.control_limit_accel_mps2),
            torch.tanh(residual[..., 1]) * float(self.cfg.control_limit_yaw_rate_rps),
        ), dim=-1)

    def _bounded_controls(self, controls: torch.Tensor) -> torch.Tensor:
        return torch.stack((
            controls[..., 0].clamp(-8.0, 4.0),
            controls[..., 1].clamp(
                -float(self.cfg.control_limit_yaw_rate_rps),
                float(self.cfg.control_limit_yaw_rate_rps),
            ),
        ), dim=-1)
