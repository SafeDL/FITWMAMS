"""Naturalness discriminator network and differentiable scoring helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .features import build_future_features_torch, denormalize_torch, normalize_torch


@dataclass
class NaturalnessDiscriminatorConfig:
    history_steps: int
    num_actors: int
    state_features: int
    context_dim: int
    relative_dim: int
    horizon_steps: int
    future_feature_dim: int
    summary_dim: int
    hidden_dim: int = 128
    dropout: float = 0.1
    use_summary_features: bool = True


class HistoryEncoder(nn.Module):
    """Independent history encoder mirroring the Stage 1 condition shape."""

    def __init__(self, cfg: NaturalnessDiscriminatorConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.state_proj = nn.Linear(cfg.state_features, cfg.hidden_dim)
        self.actor_type = nn.Parameter(torch.zeros(1, cfg.num_actors, cfg.hidden_dim))
        self.time_pos = nn.Parameter(torch.zeros(1, cfg.history_steps, cfg.hidden_dim))
        self.temporal = nn.GRU(cfg.hidden_dim, cfg.hidden_dim, batch_first=True)
        self.rel_proj = nn.Linear(cfg.relative_dim, cfg.hidden_dim)
        self.rel_temporal = nn.GRU(cfg.hidden_dim, cfg.hidden_dim, batch_first=True)
        self.context_proj = nn.Sequential(
            nn.Linear(cfg.context_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )
        self.fuse = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim * 3),
            nn.Linear(cfg.hidden_dim * 3, cfg.hidden_dim),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, context_states: torch.Tensor, context_features: torch.Tensor, relative_history: torch.Tensor) -> torch.Tensor:
        b, steps, actors, features = context_states.shape
        if steps != self.cfg.history_steps or actors != self.cfg.num_actors or features != self.cfg.state_features:
            raise ValueError(f"Unexpected context_states shape {tuple(context_states.shape)}")
        seq = context_states.permute(0, 2, 1, 3).reshape(b * actors, steps, features)
        x = self.state_proj(seq) + self.time_pos
        _, h = self.temporal(x)
        actor_tokens = h[-1].reshape(b, actors, -1) + self.actor_type
        scene_token = actor_tokens.mean(dim=1)
        rel = self.rel_proj(relative_history) + self.time_pos
        _, rel_h = self.rel_temporal(rel)
        context_token = self.context_proj(context_features)
        return self.fuse(torch.cat([scene_token, rel_h[-1], context_token], dim=-1))


class FutureEncoder(nn.Module):
    def __init__(self, cfg: NaturalnessDiscriminatorConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.proj = nn.Sequential(
            nn.Linear(cfg.future_feature_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
        )
        self.temporal = nn.GRU(cfg.hidden_dim, cfg.hidden_dim, batch_first=True)
        self.pool = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim * 2),
            nn.Linear(cfg.hidden_dim * 2, cfg.hidden_dim),
            nn.SiLU(),
        )

    def forward(self, future_action_features: torch.Tensor) -> torch.Tensor:
        if future_action_features.shape[1] != self.cfg.horizon_steps:
            raise ValueError(f"Expected horizon={self.cfg.horizon_steps}, got {future_action_features.shape[1]}")
        x = self.proj(future_action_features)
        y, h = self.temporal(x)
        pooled = torch.cat([h[-1], y.mean(dim=1)], dim=-1)
        return self.pool(pooled)


class NaturalnessDiscriminator(nn.Module):
    """D_psi(history, future) -> naturalness logit."""

    def __init__(self, cfg: NaturalnessDiscriminatorConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.history_encoder = HistoryEncoder(cfg)
        self.future_encoder = FutureEncoder(cfg)
        if cfg.use_summary_features and cfg.summary_dim > 0:
            self.summary_mlp = nn.Sequential(
                nn.Linear(cfg.summary_dim, cfg.hidden_dim),
                nn.SiLU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
                nn.SiLU(),
            )
            fusion_dim = cfg.hidden_dim * 3
        else:
            self.summary_mlp = None
            fusion_dim = cfg.hidden_dim * 2
        self.head = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim // 2, 1),
        )

    def forward(
        self,
        context_states: torch.Tensor,
        context_features: torch.Tensor,
        relative_history: torch.Tensor,
        future_action_features: torch.Tensor,
        summary_features: torch.Tensor,
    ) -> torch.Tensor:
        hist = self.history_encoder(context_states, context_features, relative_history)
        fut = self.future_encoder(future_action_features)
        pieces = [hist, fut]
        if self.summary_mlp is not None:
            pieces.append(self.summary_mlp(summary_features))
        return self.head(torch.cat(pieces, dim=-1)).squeeze(-1)


def build_discriminator_from_schema(schema: dict, config: dict) -> NaturalnessDiscriminator:
    model_cfg = config.get("model", {})
    cfg = NaturalnessDiscriminatorConfig(
        history_steps=int(schema["history_steps"]),
        num_actors=int(schema["num_actors"]),
        state_features=len(schema["state_features"]),
        context_dim=len(schema["context_keys"]),
        relative_dim=len(schema.get("relative_history_keys", [])),
        horizon_steps=int(schema["horizon_steps"]),
        future_feature_dim=len(schema["future_feature_keys"]),
        summary_dim=len(schema.get("summary_feature_keys", [])),
        hidden_dim=int(model_cfg.get("hidden_dim", 128)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        use_summary_features=bool(model_cfg.get("use_summary_features", True)),
    )
    return NaturalnessDiscriminator(cfg)


def _stat(stats: dict[str, Any], key: str) -> dict[str, Any]:
    if key not in stats:
        raise KeyError(f"Missing normalizer stats for {key}")
    return stats[key]


def score_naturalness(
    model: NaturalnessDiscriminator,
    context_states: torch.Tensor,
    context_features: torch.Tensor,
    relative_history: torch.Tensor,
    future_actions: torch.Tensor,
    *,
    ego_length: torch.Tensor | None = None,
    adv_length: torch.Tensor | None = None,
    schema: dict,
    config: dict,
    discriminator_stats: dict[str, Any] | None = None,
    stage1_stats: dict[str, Any] | None = None,
    inputs_normalized: bool = True,
    actions_normalized: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score raw or normalized future actions with differentiable feature construction."""
    raw_context = context_states
    raw_actions = future_actions
    if inputs_normalized and stage1_stats is not None:
        raw_context = denormalize_torch(context_states, **_stat(stage1_stats, "context_states"))
    if actions_normalized and stage1_stats is not None:
        raw_actions = denormalize_torch(future_actions, **_stat(stage1_stats, "actions"))
    if ego_length is None:
        ego_length = torch.full((context_states.shape[0],), 4.8, dtype=context_states.dtype, device=context_states.device)
    if adv_length is None:
        adv_length = torch.full((context_states.shape[0],), 4.8, dtype=context_states.dtype, device=context_states.device)
    future_features, summary_features = build_future_features_torch(
        raw_actions,
        raw_context,
        relative_history,
        ego_length,
        adv_length,
        schema,
        config,
    )
    if discriminator_stats is not None:
        ff = _stat(discriminator_stats, "future_action_features")
        sf = _stat(discriminator_stats, "summary_features")
        future_features = normalize_torch(future_features, ff["mean"], ff["std"])
        summary_features = normalize_torch(summary_features, sf["mean"], sf["std"])
    logits = model(context_states, context_features, relative_history, future_features, summary_features)
    return logits, torch.sigmoid(logits)


def naturalness_guidance_objective(*args, **kwargs) -> torch.Tensor:
    logits, _ = score_naturalness(*args, **kwargs)
    return F.logsigmoid(logits)
