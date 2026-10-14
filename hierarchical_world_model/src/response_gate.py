"""Small learned switch between HiQR and rule response candidates."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

# Sixteen instantaneous causal features, five persistent driver parameters,
# and ten summaries of the latest one second of realised local motion.
RESPONSE_GATE_FEATURE_DIM = 31
# The learned selector reuses observation-only HiQR tokens, not rule authority.
ATTENTION_RESPONSE_FEATURE_DIM = 128 + RESPONSE_GATE_FEATURE_DIM - 2
RESPONSE_ACTION_SEMANTICS = (
    "continuous longitudinal authority in both response modes; "
    "new lateral commitment requires MAN; committed lateral target retained; "
    "executed longitudinal responses release smoothly unless stronger braking is requested"
)
RESPONSE_SUPERVISION = (
    "remaining-horizon executed online costs with fixed world randomness; "
    "equivalent costs have no unique mode or authority label"
)
MODE_TRACK = 0
MODE_LONGITUDINAL = 1
MODE_MANEUVER = 2
LONGITUDINAL_RESPONSE_THRESHOLD_MPS2 = 0.2
MANEUVER_RESPONSE_THRESHOLD_RPS = 0.02


@dataclass(frozen=True)
class ResponseGateDecision:
    logits: torch.Tensor
    mode: torch.Tensor
    gate: torch.Tensor

    def longitudinal_authority(self) -> torch.Tensor:
        """Execute the hard choice; estimate its gradient from response probability.

        The straight-through estimator trains entry/release from rollout error.
        It changes neither the forward action nor the discrete lateral target.
        """
        active = self.mode.ne(MODE_TRACK).to(self.gate.dtype)
        if torch.is_grad_enabled() and self.logits.requires_grad:
            probability = 1 - self.logits.softmax(-1)[..., MODE_TRACK]
            active = active + (probability - probability.detach())
        return active * self.gate


def response_gate_features(context: Any, theta: torch.Tensor) -> torch.Tensor:
    """Build causal, per-NPC local features without hidden future inputs."""
    current = context.current
    npc = current[:, 1:]
    batch = len(current)
    if theta.ndim == 2:
        theta = theta[None].expand(batch, -1, -1)
    relations = context.interaction_relations
    if relations is None:
        leader_index = torch.full(
            (batch, 6), -1, dtype=torch.long, device=current.device
        )
        gap = torch.full((batch, 6), 1.0e3, dtype=current.dtype, device=current.device)
        ttc = torch.full_like(gap, 20.0)
    else:
        leader_index = relations.leader_index[:, 1:]
        gap = relations.leader_gap_m[:, 1:]
        ttc = relations.leader_ttc_s[:, 1:]
    leader = torch.gather(
        current, 1, leader_index.clamp_min(0)[..., None].expand(-1, -1, 6)
    )
    base = context.base_actions[:, 0]
    planned = (
        npc[..., :2]
        if context.planned_current_xy is None
        else context.planned_current_xy
    )
    authority = (
        torch.zeros((batch, 6), dtype=current.dtype, device=current.device)
        if context.influence_authority is None
        else context.influence_authority
    )
    phase = (
        torch.zeros_like(authority)
        if context.reaction_phase is None
        else context.reaction_phase.to(current.dtype)
    )
    speed = torch.linalg.vector_norm(npc[..., 2:4], dim=-1)
    leader_speed = torch.linalg.vector_norm(leader[..., 2:4], dim=-1)
    dynamic = torch.stack(
        (
            speed / 45.0,
            npc[..., 4] / 8.0,
            npc[..., 5] / 8.0,
            gap.clamp_max(100.0) / 100.0,
            (speed - leader_speed) / 20.0,
            leader[..., 4] / 8.0,
            base[..., 0] / 8.0,
            base[..., 1] / 0.6,
            (planned[..., 0] - npc[..., 0]).clamp(-20.0, 20.0) / 20.0,
            (planned[..., 1] - npc[..., 1]).clamp(-8.0, 8.0) / 8.0,
            authority,
            ttc.clamp_max(20.0) / 20.0,
            phase / 3.0,
            npc[..., 3] / 10.0,
            context.current_valid[:, 1:].to(current.dtype),
            (leader_index.ge(0)).to(current.dtype),
        ),
        dim=-1,
    )
    history = context.history[:, -25:]
    history_valid = context.history_valid[:, -25:]
    npc_history = history[:, :, 1:]
    gather_index = leader_index.clamp_min(0)[:, None, :, None].expand(
        -1, history.shape[1], -1, history.shape[-1]
    )
    leader_history = torch.gather(history, 2, gather_index)
    leader_history_valid = torch.gather(
        history_valid,
        2,
        leader_index.clamp_min(0)[:, None].expand(-1, history.shape[1], -1),
    )
    pair_valid = (
        history_valid[:, :, 1:] & leader_history_valid & leader_index.ge(0)[:, None]
    )
    dimensions = context.vehicle_dimensions
    if dimensions is None:
        pair_length = history.new_full((batch, 6), 4.8)
    else:
        dimensions = torch.as_tensor(
            dimensions, dtype=history.dtype, device=history.device
        )
        leader_length = torch.gather(dimensions[..., 0], 1, leader_index.clamp_min(0))
        pair_length = 0.5 * (dimensions[:, 1:, 0] + leader_length)
    history_gap = leader_history[..., 0] - npc_history[..., 0] - pair_length[:, None]
    history_closing = npc_history[..., 2] - leader_history[..., 2]
    history_dy = leader_history[..., 1] - npc_history[..., 1]
    history_relative_vy = leader_history[..., 3] - npc_history[..., 3]

    def masked_mean(value: torch.Tensor) -> torch.Tensor:
        return torch.where(pair_valid, value, torch.zeros_like(value)).sum(
            dim=1
        ) / pair_valid.sum(dim=1).clamp_min(1)

    def masked_min(value: torch.Tensor) -> torch.Tensor:
        result = torch.where(
            pair_valid, value, torch.full_like(value, float("inf"))
        ).amin(dim=1)
        return torch.where(torch.isfinite(result), result, torch.zeros_like(result))

    def masked_max(value: torch.Tensor) -> torch.Tensor:
        result = torch.where(
            pair_valid, value, torch.full_like(value, -float("inf"))
        ).amax(dim=1)
        return torch.where(torch.isfinite(result), result, torch.zeros_like(result))

    if history.shape[1] >= 2:
        leader_acceleration_change = (
            leader_history[:, 1:, :, 4] - leader_history[:, :-1, :, 4]
        ).abs()
        change_valid = pair_valid[:, 1:] & pair_valid[:, :-1]
        sharpest_change = torch.where(
            change_valid,
            leader_acceleration_change,
            torch.zeros_like(leader_acceleration_change),
        ).amax(dim=1)
    else:
        sharpest_change = history_gap.new_zeros((batch, 6))
    temporal = torch.stack(
        (
            masked_mean(history_gap) / 100.0,
            masked_min(history_gap) / 100.0,
            masked_mean(history_closing) / 20.0,
            masked_max(history_closing) / 20.0,
            masked_mean(history_dy) / 8.0,
            masked_mean(history_relative_vy.abs()) / 10.0,
            masked_mean(leader_history[..., 4]) / 8.0,
            masked_min(leader_history[..., 4]) / 8.0,
            masked_mean(npc_history[..., 4]) / 8.0,
            sharpest_change / 8.0,
        ),
        dim=-1,
    )
    return torch.cat(
        (
            dynamic,
            theta[..., :5] / theta.new_tensor((60.0, 12.0, 4.0, 6.0, 8.0)),
            temporal,
        ),
        dim=-1,
    )


class ResponseGate(nn.Module):
    def __init__(
        self,
        input_dim: int = RESPONSE_GATE_FEATURE_DIM,
        hidden_dim: int = 64,
        mode_count: int = 3,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.mode_count = int(mode_count)
        if self.input_dim not in (
            RESPONSE_GATE_FEATURE_DIM,
            ATTENTION_RESPONSE_FEATURE_DIM,
        ):
            raise ValueError(
                "response gate requires local or attention-encoded features"
            )
        if self.mode_count not in (2, 3):
            raise ValueError("response gate mode_count must be 2 or 3")
        self.backbone = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        )
        self.mode_head = nn.Linear(self.hidden_dim, self.mode_count)
        self.gate_head = nn.Linear(self.hidden_dim, 1)

    def features(self, context: Any, theta: torch.Tensor) -> torch.Tensor:
        local = response_gate_features(context, theta)
        if self.input_dim == RESPONSE_GATE_FEATURE_DIM:
            return local
        # Exclude handcrafted authority and phase; they must not define the label.
        local = torch.cat((local[..., :10], local[..., 11:12], local[..., 13:]), -1)
        return torch.cat((context.observation_tokens[:, 1:], local), -1)

    def forward(self, features: torch.Tensor) -> ResponseGateDecision:
        if features.shape[-1] != self.input_dim:
            raise ValueError(f"response-gate features must end in {self.input_dim}")
        hidden = self.backbone(features)
        logits = self.mode_head(hidden)
        learned_gate = torch.sigmoid(self.gate_head(hidden).squeeze(-1))
        raw_mode = logits.argmax(dim=-1)
        mode = (
            torch.where(
                raw_mode.eq(1),
                torch.full_like(raw_mode, MODE_MANEUVER),
                raw_mode,
            )
            if self.mode_count == 2
            else raw_mode
        )
        # L1 is a genuine binary switch: an active decision grants the full
        # rule candidate. Only the three-mode L2 model learns authority.
        gate = torch.ones_like(learned_gate) if self.mode_count == 2 else learned_gate
        return ResponseGateDecision(logits, mode, gate)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "mode_count": self.mode_count,
            },
            path,
        )

    @classmethod
    def load(
        cls, path: str | Path, *, device: str | torch.device = "cpu"
    ) -> "ResponseGate":
        payload = torch.load(path, map_location=device)
        model = cls(
            payload["input_dim"],
            payload["hidden_dim"],
            payload["mode_count"],
        )
        model.load_state_dict(payload["state_dict"])
        return model.to(device).eval()
