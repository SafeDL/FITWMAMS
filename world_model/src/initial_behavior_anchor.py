"""First-second behavior anchors from the frozen 76-dimensional Flow."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from normalizing_flow.src.features import (
    SLOT_NAMES,
    trajectory_feature_index,
)

from .schema import FLOW_ACTION_SUMMARY_FEATURES


BEHAVIOR_ANCHOR_SECONDS = 1.0


def behavior_anchor_from_flow_feature(
    feature_row: np.ndarray,
    slot_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract the six per-slot first-second summaries from a legacy Flow row."""
    feature = np.asarray(feature_row, dtype=np.float32).reshape(-1)
    valid = np.asarray(slot_mask, dtype=bool).reshape(-1)
    if feature.size != 76 or valid.shape != (len(SLOT_NAMES),):
        raise ValueError("behavior anchors require one 76-D Flow row and a six-slot mask")
    anchor = np.zeros((len(SLOT_NAMES), len(FLOW_ACTION_SUMMARY_FEATURES)), dtype=np.float32)
    for slot_index, slot_name in enumerate(SLOT_NAMES):
        if valid[slot_index]:
            anchor[slot_index] = [
                feature[trajectory_feature_index(slot_name, name)]
                for name in FLOW_ACTION_SUMMARY_FEATURES
            ]
    return anchor, valid


def summarize_highd_actions(
    actions: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Recompute Flow's six-value behavior summary from a one-second action curve."""
    if actions.ndim != 4 or actions.shape[-1] != 2 or valid.shape != actions.shape[:-1]:
        raise ValueError("actions must be [batch, frames, agents, 2] with matching validity")
    mask = valid.to(dtype=actions.dtype)
    count = mask.sum(dim=1).clamp_min(1.0)
    acceleration = actions[..., 0]
    lateral_acceleration = actions[..., 1]
    mean_acceleration = (acceleration * mask).sum(dim=1) / count
    mean_lateral_acceleration = (lateral_acceleration * mask).sum(dim=1) / count
    minimum_acceleration = acceleration.masked_fill(~valid, float("inf")).amin(dim=1)
    minimum_acceleration = torch.where(valid.any(dim=1), minimum_acceleration, torch.zeros_like(minimum_acceleration))
    final_index = (valid.long() * torch.arange(actions.shape[1], device=actions.device).view(1, -1, 1)).argmax(dim=1)
    final_acceleration = acceleration.gather(1, final_index.unsqueeze(1)).squeeze(1)
    final_acceleration = torch.where(valid.any(dim=1), final_acceleration, torch.zeros_like(final_acceleration))
    dt = 1.0 / float(actions.shape[1])
    return torch.stack((
        (acceleration * mask).sum(dim=1) * dt,
        (lateral_acceleration * mask).sum(dim=1) * dt,
        mean_acceleration,
        minimum_acceleration,
        final_acceleration,
        mean_lateral_acceleration,
    ), dim=-1)


class BehaviorAnchorEncoder(nn.Module):
    """Mask-aware per-vehicle and scene-level encoding of a behavior anchor."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.per_agent = nn.Sequential(
            nn.Linear(len(FLOW_ACTION_SUMMARY_FEATURES), hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, anchor: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if anchor.ndim != 3 or anchor.shape[-1] != len(FLOW_ACTION_SUMMARY_FEATURES):
            raise ValueError("behavior anchor must be [batch, agents, 6]")
        if valid.shape != anchor.shape[:-1]:
            raise ValueError("behavior anchor validity must align with anchors")
        agents = self.per_agent(anchor) * valid[..., None].to(dtype=anchor.dtype)
        scene = agents.sum(dim=1) / valid.sum(dim=1, keepdim=True).clamp_min(1).to(dtype=anchor.dtype)
        return agents, scene
