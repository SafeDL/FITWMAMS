"""Shared stochastic behavior-cloning baseline (A0)."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from npc_behavior_benchmark.data.dataset import policy_features
from .interface import NPCPolicy, PolicyAction, PolicyObservation, RandomKey


class SharedBCModel(nn.Module):
    def __init__(
        self,
        feature_dim: int = 12,
        hidden_dim: int = 96,
        heads: int = 4,
        layers: int = 2,
        style_dim: int = 0,
    ) -> None:
        super().__init__()
        self.style_dim = int(style_dim)
        self.temporal = nn.GRU(feature_dim, hidden_dim, batch_first=True)
        encoder_layer = nn.TransformerEncoderLayer(
            hidden_dim,
            heads,
            dim_feedforward=hidden_dim * 3,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.interaction = nn.TransformerEncoder(encoder_layer, layers)
        self.style_projection = (
            nn.Linear(self.style_dim, hidden_dim, bias=False)
            if self.style_dim
            else None
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 4),
        )

    def forward(
        self,
        features: torch.Tensor,
        history_valid: torch.Tensor,
        style: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, time, agents, feature_dim = features.shape
        temporal_input = features.transpose(1, 2).reshape(
            batch * agents, time, feature_dim
        )
        _, hidden = self.temporal(temporal_input)
        encoded = hidden[-1].reshape(batch, agents, -1)
        if self.style_projection is not None:
            if style is None:
                style = encoded.new_zeros((batch, agents, self.style_dim))
            if style.shape != (batch, agents, self.style_dim):
                raise ValueError(
                    f"style must have shape {(batch, agents, self.style_dim)}, got {tuple(style.shape)}"
                )
            encoded = encoded + self.style_projection(style)
        current_valid = history_valid[:, -1]
        interacted = self.interaction(encoded, src_key_padding_mask=~current_valid)
        output = self.head(interacted[:, 1:])
        mean = output[..., :2]
        log_std = output[..., 2:].clamp(-4.0, 1.5)
        return mean, log_std


def gaussian_nll(
    mean: torch.Tensor, log_std: torch.Tensor, target: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    normalized = (target - mean) * torch.exp(-log_std)
    loss = 0.5 * normalized.square() + log_std
    mask = valid[..., None].to(loss.dtype)
    return (loss * mask).sum() / (mask.sum().clamp_min(1.0) * loss.shape[-1])


@dataclass
class _Snapshot:
    history: np.ndarray
    history_valid: np.ndarray
    rng_state: dict[str, Any]
    decision_index: int


class SharedBCPolicy(NPCPolicy):
    def __init__(
        self,
        model: SharedBCModel,
        feature_mean: np.ndarray,
        feature_std: np.ndarray,
        *,
        device: str = "cpu",
        deterministic: bool = False,
    ) -> None:
        self.model = model.to(device).eval()
        self.device = torch.device(device)
        self.feature_mean = np.asarray(feature_mean, np.float32)
        self.feature_std = np.asarray(feature_std, np.float32)
        self.deterministic = bool(deterministic)

    def reset(
        self,
        history: np.ndarray,
        history_valid: np.ndarray,
        agent_ids: np.ndarray,
        map_context: Mapping[str, np.ndarray],
        random_key: RandomKey,
        behavior_condition: Mapping[str, Any] | None = None,
    ) -> None:
        del behavior_condition
        self.history = np.asarray(history, np.float32).copy()
        self.history_valid = np.asarray(history_valid, bool).copy()
        self.agent_ids = np.asarray(agent_ids, np.int64).copy()
        self.map_context = {
            key: np.asarray(value).copy() for key, value in map_context.items()
        }
        self.rng = np.random.default_rng(random_key.seed())
        self.decision_index = 0

    @torch.no_grad()
    def act(self, observation: PolicyObservation) -> PolicyAction:
        observation.validate()
        if not np.array_equal(observation.agent_ids, self.agent_ids):
            raise ValueError("agent identity/order changed without reset")
        feature = policy_features(
            observation.history_states,
            observation.history_valid,
            observation.lengths_m,
            observation.widths_m,
            observation.map_polylines,
            observation.map_polyline_valid,
        )
        feature = (feature - self.feature_mean) / self.feature_std
        x = torch.from_numpy(feature[None].astype(np.float32)).to(self.device)
        valid = torch.from_numpy(observation.history_valid[None]).to(self.device)
        mean, log_std = self.model(x, valid)
        mean_np = mean[0].cpu().numpy()
        if self.deterministic:
            action = mean_np
        else:
            action = mean_np + np.exp(
                log_std[0].cpu().numpy()
            ) * self.rng.standard_normal(mean_np.shape)
        self.history = np.asarray(observation.history_states).copy()
        self.history_valid = np.asarray(observation.history_valid).copy()
        self.decision_index += 1
        return PolicyAction(
            action.astype(np.float32), diagnostics={"nn_evaluations": 1, "replan": True}
        )

    def snapshot(self) -> _Snapshot:
        return _Snapshot(
            self.history.copy(),
            self.history_valid.copy(),
            copy.deepcopy(self.rng.bit_generator.state),
            self.decision_index,
        )

    def restore(self, snapshot: _Snapshot) -> None:
        self.history = snapshot.history.copy()
        self.history_valid = snapshot.history_valid.copy()
        self.rng.bit_generator.state = copy.deepcopy(snapshot.rng_state)
        self.decision_index = int(snapshot.decision_index)
