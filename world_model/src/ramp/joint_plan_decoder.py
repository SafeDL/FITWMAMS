"""Joint candidate decoder with bounded jerk-control residuals."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .plan_attention import PlanRelationAttention


class JointPlanDecoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        candidates: int,
        controls: int,
        frames: int,
        max_jerk: tuple[float, float],
    ) -> None:
        super().__init__()
        self.candidates, self.controls, self.frames, self.max_jerk = (
            candidates,
            controls,
            frames,
            max_jerk,
        )
        self.nominal = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, frames * 2),
        )
        self.candidate_embed = nn.Embedding(candidates - 1, hidden_dim)
        self.residual = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, controls * 2),
        )
        self.probability = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, candidates),
        )
        self.attention = PlanRelationAttention(hidden_dim)
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(
        self,
        agents: torch.Tensor,
        scene: torch.Tensor,
        memory: torch.Tensor,
        states: torch.Tensor,
        valid: torch.Tensor,
        previous_plan: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, all_agents, hidden = agents.shape
        background = agents[:, 1:]
        context = torch.cat(
            (
                background,
                scene[:, None].expand(-1, 6, -1),
                memory[:, None].expand(-1, 6, -1),
            ),
            dim=-1,
        )
        nominal_raw = (
            self.nominal(context).reshape(b, 6, self.frames, 2).permute(0, 2, 1, 3)
        )
        nominal = torch.stack(
            (
                nominal_raw[..., 0].clamp(-8.0, 4.0),
                nominal_raw[..., 1].clamp(-0.6, 0.6),
            ),
            dim=-1,
        )
        plans = nominal[:, None].expand(-1, self.candidates, -1, -1, -1).clone()
        if self.candidates > 1:
            embedding = self.candidate_embed.weight[None, :, None, :].expand(
                b, -1, 6, -1
            )
            base = torch.cat(
                (
                    background[:, None].expand(-1, self.candidates - 1, -1, -1),
                    scene[:, None, None].expand(-1, self.candidates - 1, 6, -1),
                    embedding,
                ),
                dim=-1,
            )
            jerk = (
                self.residual(base)
                .reshape(b, self.candidates - 1, 6, self.controls, 2)
                .permute(0, 1, 3, 2, 4)
            )
            jerk = self.attention(states, jerk, valid)
            limit = jerk.new_tensor(self.max_jerk)
            jerk = torch.tanh(jerk) * limit
            curve = (
                F.interpolate(
                    jerk.permute(0, 1, 3, 4, 2).reshape(
                        b * (self.candidates - 1) * 6, 2, self.controls
                    ),
                    size=self.frames,
                    mode="linear",
                    align_corners=True,
                )
                .reshape(b, self.candidates - 1, 6, 2, self.frames)
                .permute(0, 1, 4, 2, 3)
            )
            residual = curve.cumsum(dim=2) * 0.04
            ramp = residual.new_ones((self.frames,))
            ramp[:5] = residual.new_tensor([0.0, 0.25, 0.5, 0.75, 1.0])
            plans = torch.cat(
                (
                    plans[:, :1],
                    plans[:, :1] + residual * ramp[None, None, :, None, None],
                ),
                dim=1,
            )
        plans = torch.stack(
            (plans[..., 0].clamp(-8.0, 4.0), plans[..., 1].clamp(-0.6, 0.6)), dim=-1
        )
        previous = (
            plans.new_zeros((b, 2))
            if previous_plan is None
            else previous_plan.mean(dim=(1, 2))
        )
        logits = self.probability(torch.cat((scene, memory, previous), dim=-1))
        return (
            plans * valid[:, None, None, 1:, None].float(),
            torch.softmax(logits, dim=-1),
            (
                jerk
                if self.candidates > 1
                else plans.new_zeros((b, 0, self.controls, 6, 2))
            ),
        )
