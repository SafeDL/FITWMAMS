"""Structured-state behavior-cloning controller used as the learned T3 PNC."""

from __future__ import annotations

import torch
from torch import nn


class StructuredBCPNC(nn.Module):
    """Encode the causal multi-agent history and predict the ego control."""

    def __init__(
        self,
        feature_dim: int = 12,
        hidden_dim: int = 96,
        heads: int = 4,
        layers: int = 2,
    ) -> None:
        super().__init__()
        self.temporal = nn.GRU(feature_dim, hidden_dim, batch_first=True)
        layer = nn.TransformerEncoderLayer(
            hidden_dim,
            heads,
            dim_feedforward=hidden_dim * 3,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.interaction = nn.TransformerEncoder(layer, layers)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(
        self, features: torch.Tensor, history_valid: torch.Tensor
    ) -> torch.Tensor:
        batch, time, agents, feature_dim = features.shape
        temporal = features.transpose(1, 2).reshape(batch * agents, time, feature_dim)
        _, hidden = self.temporal(temporal)
        encoded = hidden[-1].reshape(batch, agents, -1)
        interacted = self.interaction(
            encoded, src_key_padding_mask=~history_valid[:, -1]
        )
        return self.head(interacted[:, 0])
