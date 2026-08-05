"""Adaptive one-pass joint plan continuation decoder for HiQR-WM."""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import HiQRWorldModelConfig


class AdaptiveJointPlanContinuationDecoder(nn.Module):
    """Jointly decode all agent-time actions with adaptive carry revisions.

    A single agent-time Transformer produces direct START actions and, for a
    ROLL response, the carry correction gate, correction residual and newly
    generated tail.  It intentionally has no fresh-plan/refiner cascade.
    """

    def __init__(self, cfg: HiQRWorldModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        h, z = int(cfg.hidden_dim), int(cfg.agent_residual_dim)
        self.agent = nn.Sequential(nn.Linear(h + z, h), nn.SiLU(), nn.Linear(h, h))
        self.context = nn.Sequential(
            nn.Linear(h + int(cfg.scene_latent_dim), h), nn.SiLU(), nn.Linear(h, h)
        )
        self.time = nn.Parameter(torch.zeros(int(cfg.plan_frames), h))
        layer = nn.TransformerEncoderLayer(
            d_model=h,
            nhead=int(cfg.num_heads),
            dim_feedforward=h * 2,
            dropout=float(cfg.dropout),
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(
            layer, num_layers=max(1, int(cfg.decoder_layers))
        )
        self.action = nn.Linear(h, 2)
        self.gate = nn.Linear(h, 2)
        self.delta = nn.Linear(h, 2)
        nn.init.normal_(self.time, std=0.02)

    def _clamp(self, actions: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            (
                actions[..., 0].clamp(
                    float(self.cfg.min_acceleration), float(self.cfg.max_acceleration)
                ),
                actions[..., 1].clamp(
                    -float(self.cfg.max_yaw_rate), float(self.cfg.max_yaw_rate)
                ),
            ),
            dim=-1,
        )

    def _direct_actions(self, tokens: torch.Tensor) -> torch.Tensor:
        """Map unconstrained logits onto the configured physical action range."""
        acceleration_midpoint = 0.5 * (
            float(self.cfg.min_acceleration) + float(self.cfg.max_acceleration)
        )
        acceleration_radius = 0.5 * (
            float(self.cfg.max_acceleration) - float(self.cfg.min_acceleration)
        )
        scale = tokens.new_tensor((acceleration_radius, float(self.cfg.max_yaw_rate)))
        offset = tokens.new_tensor((acceleration_midpoint, 0.0))
        return self._clamp(torch.tanh(self.action(tokens)) * scale + offset)

    def _decode_tokens(
        self,
        agents: torch.Tensor,
        hidden: torch.Tensor,
        scene_latent: torch.Tensor,
        agent_residual: torch.Tensor,
        background_valid: torch.Tensor,
    ) -> torch.Tensor:
        batch, horizon, backgrounds = (
            agents.shape[0],
            int(self.cfg.plan_frames),
            agents.shape[1] - 1,
        )
        if backgrounds != 6:
            raise ValueError("HiQR-WM requires ego plus six background slots")
        base = self.agent(torch.cat((agents[:, 1:], agent_residual[:, 1:]), dim=-1))
        context = self.context(torch.cat((hidden, scene_latent), dim=-1))
        token = base[:, None] + self.time[None, :, None] + context[:, None, None]
        token = token.reshape(batch, horizon * backgrounds, -1)
        valid = (
            background_valid[:, None]
            .expand(-1, horizon, -1)
            .reshape(batch, horizon * backgrounds)
        )
        padding = ~valid
        empty = padding.all(dim=1)
        if empty.any():
            padding = padding.clone()
            padding[empty, 0] = False
        return self.blocks(token, src_key_padding_mask=padding).reshape(
            batch, horizon, backgrounds, -1
        )

    def forward(
        self,
        agents: torch.Tensor,
        hidden: torch.Tensor,
        scene_latent: torch.Tensor,
        agent_residual: torch.Tensor,
        current_valid: torch.Tensor,
        previous_buffer: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        background_valid = current_valid[:, 1:].bool()
        tokens = self._decode_tokens(
            agents, hidden, scene_latent, agent_residual, background_valid
        )
        direct = self._direct_actions(tokens)
        valid = background_valid[:, None].expand(-1, int(self.cfg.plan_frames), -1)
        carried = torch.zeros_like(valid)
        appended = torch.ones_like(valid)
        gate = torch.zeros_like(direct)
        correction = torch.zeros_like(direct)
        if previous_buffer is None:
            actions = direct
        else:
            if previous_buffer.shape != direct.shape:
                raise ValueError(
                    "previous_buffer must have shape [batch, plan_frames, 6, 2]"
                )
            execute = int(self.cfg.execute_frames)
            remain = int(self.cfg.plan_frames) - execute
            carry = previous_buffer[:, execute:]
            gate[:, :remain] = torch.sigmoid(self.gate(tokens[:, :remain]))
            scale = tokens.new_tensor((2.0, 0.12))
            correction[:, :remain] = torch.tanh(self.delta(tokens[:, :remain])) * scale
            revised = self._clamp(carry + gate[:, :remain] * correction[:, :remain])
            actions = torch.cat((revised, direct[:, remain:]), dim=1)
            carried[:, :remain] = valid[:, :remain]
            appended[:, :remain] = False
        actions = actions * valid[..., None].float()
        return {
            "background_future_actions": actions,
            "continuation_gate": gate,
            "background_future_action_masks": {
                "carried": carried,
                "appended": appended & valid,
                "refinable": valid,
                "valid": valid,
            },
        }
