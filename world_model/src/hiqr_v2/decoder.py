"""Low-frequency physical-control decoder for HiQR-v2."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as functional

from world_model.src.core.dynamics import KinematicTrafficDynamics

from .config import HiQRV2Config


class JerkResidualDecoder(nn.Module):
    """Integrate coordinated jerk knots from each observed current control."""

    def __init__(self, cfg: HiQRV2Config) -> None:
        super().__init__()
        self.cfg = cfg
        hidden, knots = int(cfg.hidden_dim), int(cfg.jerk_knots)
        self.agent = nn.Sequential(
            nn.Linear(2 * hidden + int(cfg.agent_residual_dim), hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.context = nn.Sequential(
            nn.Linear(hidden + int(cfg.scene_latent_dim), hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.knot = nn.Parameter(torch.empty(knots, hidden))
        layer = nn.TransformerEncoderLayer(
            hidden,
            int(cfg.num_heads),
            hidden * 2,
            float(cfg.dropout),
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.coordination = nn.TransformerEncoder(
            layer, num_layers=max(1, int(cfg.decoder_layers))
        )
        self.jerk = nn.Linear(hidden, 2)
        nn.init.normal_(self.knot, std=0.02)
        nn.init.zeros_(self.jerk.weight)
        nn.init.zeros_(self.jerk.bias)

    def forward(
        self,
        agents: torch.Tensor,
        global_hidden: torch.Tensor,
        agent_hidden: torch.Tensor,
        scene_latent: torch.Tensor,
        agent_residual: torch.Tensor,
        current: torch.Tensor,
        current_valid: torch.Tensor,
    ) -> torch.Tensor:
        batch, backgrounds = current.shape[0], current.shape[1] - 1
        knots, frames = int(self.cfg.jerk_knots), int(self.cfg.plan_frames)
        if backgrounds != 6:
            raise ValueError("HiQR-v2 requires ego plus six background slots")
        base = self.agent(
            torch.cat((agents[:, 1:], agent_hidden[:, 1:], agent_residual[:, 1:]), -1)
        )
        context = self.context(torch.cat((global_hidden, scene_latent), -1))
        token = base[:, None] + context[:, None, None] + self.knot[None, :, None]
        valid = current_valid[:, None, 1:].expand(-1, knots, -1)
        padding = ~valid.reshape(batch, knots * backgrounds)
        padding = torch.where(
            padding.all(dim=1, keepdim=True), torch.zeros_like(padding), padding
        )
        raw = self.coordination(
            token.reshape(batch, knots * backgrounds, -1),
            src_key_padding_mask=padding,
        ).reshape(batch, knots, backgrounds, -1)
        limits = raw.new_tensor((self.cfg.max_longitudinal_jerk, self.cfg.max_yaw_jerk))
        jerk_knots = torch.tanh(self.jerk(raw)) * limits
        jerk = (
            functional.interpolate(
                jerk_knots.permute(0, 2, 3, 1).reshape(batch * backgrounds, 2, knots),
                size=frames,
                mode="linear",
                align_corners=True,
            )
            .reshape(batch, backgrounds, 2, frames)
            .permute(0, 3, 1, 2)
        )
        initial = KinematicTrafficDynamics.controls_from_highd_actions(
            current[:, 1:, 4:6], current[:, 1:]
        )
        plan = initial[:, None] + jerk.cumsum(1) * float(self.cfg.simulation_dt_s)
        plan = torch.stack(
            (
                plan[..., 0].clamp(
                    float(self.cfg.min_acceleration), float(self.cfg.max_acceleration)
                ),
                plan[..., 1].clamp(
                    -float(self.cfg.max_yaw_rate), float(self.cfg.max_yaw_rate)
                ),
            ),
            -1,
        )
        return plan * current_valid[:, None, 1:, None].float()
