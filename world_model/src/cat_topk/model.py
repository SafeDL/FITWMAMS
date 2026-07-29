"""Shared START/ROLL stochastic background-traffic policy."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from world_model.src.core.schema import (
    AGENT_NAMES,
    AGENT_STATE_FEATURES,
    DEFAULT_EGO_LENGTH_M,
    DEFAULT_OTHER_LENGTH_M,
    FLOW_ACTION_SUMMARY_FEATURES,
    RELATION_FEATURES,
    SLOT_NAMES,
    START_MODE_INDEX,
)

@dataclass
class WorldModelConfig:
    history_steps: int
    horizon_steps: int
    state_dim: int = len(AGENT_STATE_FEATURES)
    action_dim: int = 2
    flow_summary_dim: int = len(FLOW_ACTION_SUMMARY_FEATURES)
    relation_feature_dim: int = len(RELATION_FEATURES)
    num_agents: int = len(AGENT_NAMES)
    num_slots: int = len(SLOT_NAMES)
    hidden_dim: int = 128
    history_layers: int = 2
    interaction_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.1
    use_start_flow_summary: bool = False
    use_relation_features: bool = False
    # 已存档比较 checkpoint 保留这两个字段；活动 CAT-K 推理不使用它们。
    min_log_std: float = -5.0
    max_log_std: float = 2.0


class SharedStartRollWorldModel(nn.Module):
    """活动 CAT-K 与已存档比较器共用的 START/ROLL 编码器。"""

    def __init__(self, cfg: WorldModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        hidden = int(cfg.hidden_dim)
        self.state_proj = nn.Sequential(
            nn.Linear(cfg.state_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(hidden, hidden),
        )
        self.agent_embed = nn.Embedding(cfg.num_agents, hidden)
        self.time_embed = nn.Embedding(cfg.history_steps, hidden)
        self.mode_embed = nn.Embedding(2, hidden)
        self.primary_embed = nn.Embedding(cfg.num_slots, hidden)
        self.flow_summary_proj = nn.Sequential(
            nn.Linear(cfg.flow_summary_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.relation_proj = nn.Sequential(
            nn.Linear(cfg.relation_feature_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )

        hist_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=max(1, int(cfg.num_heads)),
            dim_feedforward=hidden * 4,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.history_encoder = nn.TransformerEncoder(hist_layer, num_layers=max(1, int(cfg.history_layers)))

        interaction_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=max(1, int(cfg.num_heads)),
            dim_feedforward=hidden * 4,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.interaction_encoder = nn.TransformerEncoder(
            interaction_layer,
            num_layers=max(1, int(cfg.interaction_layers)),
        )
        self.horizon_embed = nn.Embedding(cfg.horizon_steps, hidden)
        self.action_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        # 仅用于严格加载已存档比较 checkpoint；活动 CAT-K 推理不使用这两个头。
        self.mean = nn.Linear(hidden, cfg.action_dim)
        self.log_std = nn.Linear(hidden, cfg.action_dim)

    def encode_agent_context(
        self,
        history_states: torch.Tensor,
        history_valid: torch.Tensor,
        current_states: torch.Tensor,
        current_valid: torch.Tensor,
        mode_index: torch.Tensor,
        primary_slot_index: torch.Tensor,
        flow_action_summary: torch.Tensor | None = None,
        relation_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return temporally encoded tokens for ego and all background slots.

        Args:
            history_states: `[B, H, 7, state_dim]`, normalized.
            history_valid: `[B, H, 7]`.
            current_states: `[B, 7, state_dim]`, normalized.
            current_valid: `[B, 7]`.
            mode_index: START=0 or ROLL=1.
            primary_slot_index: `[B]` in slot index space.
            flow_action_summary: optional `[B, 6, flow_summary_dim]`, normalized.
            relation_features: optional `[B, 6, relation_feature_dim]`, normalized.
        """
        b, h, a, _ = history_states.shape
        if h != self.cfg.history_steps or a != self.cfg.num_agents:
            raise ValueError(
                "history_states shape mismatch: expected "
                f"[B,{self.cfg.history_steps},{self.cfg.num_agents},{self.cfg.state_dim}], "
                f"got {tuple(history_states.shape)}"
            )
        device = history_states.device
        mode_index = mode_index.long().clamp(0, 1)
        primary_slot_index = primary_slot_index.long().clamp(0, self.cfg.num_slots - 1)

        x = self.state_proj(history_states)
        agent_ids = torch.arange(self.cfg.num_agents, device=device).view(1, 1, -1)
        time_ids = torch.arange(self.cfg.history_steps, device=device).view(1, -1, 1)
        x = x + self.agent_embed(agent_ids) + self.time_embed(time_ids)
        x = x + self.mode_embed(mode_index).view(b, 1, 1, -1)
        tokens = x.reshape(b, h * a, self.cfg.hidden_dim)
        padding = ~history_valid.bool().reshape(b, h * a)
        all_padded = padding.all(dim=1)
        # Defensive fallback; START samples should always mark the last frame valid.
        # Keep this branchless to avoid synchronizing GPU execution every forward.
        padding[:, -self.cfg.num_agents :] = (
            padding[:, -self.cfg.num_agents :] & ~all_padded.view(b, 1)
        )
        encoded = self.history_encoder(tokens, src_key_padding_mask=padding)
        encoded = encoded.reshape(b, h, a, self.cfg.hidden_dim)
        valid_f = history_valid.float().unsqueeze(-1)
        denom = valid_f.sum(dim=1).clamp_min(1.0)
        temporal = (encoded * valid_f).sum(dim=1) / denom

        current = self.state_proj(current_states)
        scene_context = (
            temporal
            + current
            + self.mode_embed(mode_index).view(b, 1, -1)
            + self.primary_embed(primary_slot_index).view(b, 1, -1)
        )
        if self.cfg.use_start_flow_summary and flow_action_summary is not None:
            flow_context = self.flow_summary_proj(flow_action_summary)
            start_mask = (mode_index == START_MODE_INDEX).float().view(b, 1, 1)
            scene_context[:, 1:, :] = scene_context[:, 1:, :] + start_mask * flow_context
        if self.cfg.use_relation_features and relation_features is not None:
            scene_context[:, 1:, :] = scene_context[:, 1:, :] + self.relation_proj(relation_features)

        agent_padding = ~current_valid.bool()
        all_agents_padded = agent_padding.all(dim=1)
        agent_padding[:, 0] = agent_padding[:, 0] & ~all_agents_padded
        interaction = self.interaction_encoder(scene_context, src_key_padding_mask=agent_padding)
        return interaction

    def encode_context(
        self,
        history_states: torch.Tensor,
        history_valid: torch.Tensor,
        current_states: torch.Tensor,
        current_valid: torch.Tensor,
        mode_index: torch.Tensor,
        primary_slot_index: torch.Tensor,
        flow_action_summary: torch.Tensor | None = None,
        relation_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return per-background-slot condition tokens.

        已存档比较器直接使用逐槽位 token；活动 CAT-K 还通过场景级交互图使用 ego token。
        """
        return self.encode_agent_context(
            history_states,
            history_valid,
            current_states,
            current_valid,
            mode_index,
            primary_slot_index,
            flow_action_summary,
            relation_features,
        )[:, 1:, :]


class NominalCATKDecoder(SharedStartRollWorldModel):
    """候选 ``0`` 使用的可训练名义 CAT-K 解码器。"""

    def __init__(
        self,
        cfg: WorldModelConfig,
        *,
        num_candidates: int = 8,
        candidate_ce_weight: float = 0.05,
    ) -> None:
        super().__init__(cfg)
        self.num_candidates = max(2, int(num_candidates))
        self.candidate_ce_weight = float(candidate_ce_weight)
        hidden = int(cfg.hidden_dim)
        self.candidate_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, self.num_candidates * cfg.action_dim),
        )
        self.candidate_logit = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.num_candidates),
        )

    def forward_candidates(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        context = self.encode_context(
            batch["history_states"],
            batch["history_valid"],
            batch["current_states"],
            batch["current_valid"],
            batch["mode_index"],
            batch["primary_slot_index"],
            batch.get("flow_action_summary"),
            batch.get("relation_features"),
        )
        b = int(context.shape[0])
        device = context.device
        t_ids = torch.arange(self.cfg.horizon_steps, device=device)
        horizon = self.horizon_embed(t_ids).view(1, self.cfg.horizon_steps, 1, -1)
        slot = context.view(b, 1, self.cfg.num_slots, -1)
        hidden = self.action_head(slot + horizon)
        candidates = self.candidate_head(hidden).view(
            b,
            self.cfg.horizon_steps,
            self.cfg.num_slots,
            self.num_candidates,
            self.cfg.action_dim,
        )
        candidates = candidates.permute(0, 3, 1, 2, 4).contiguous()
        logits = self.candidate_logit(context)
        return candidates, logits

    def _candidate_probabilities(
        self,
        logits: torch.Tensor,
        current_valid: torch.Tensor,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        slot_valid = current_valid[:, 1:].float()
        pooled_logits = (logits * slot_valid.unsqueeze(-1)).sum(dim=1) / slot_valid.sum(
            dim=1,
            keepdim=True,
        ).clamp_min(1.0)
        probabilities = F.softmax(pooled_logits / max(float(temperature), 1.0e-3), dim=-1)
        return pooled_logits, probabilities

    def select_map_actions_st(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """以直通 MAP 选择名义分支，使其在训练中保留梯度。"""
        candidates, logits = self.forward_candidates(batch)
        pooled_logits, probabilities = self._candidate_probabilities(logits, batch["current_valid"])
        choice = torch.argmax(pooled_logits, dim=-1)
        gather = choice.view(-1, 1, 1, 1, 1).expand(
            -1,
            1,
            self.cfg.horizon_steps,
            self.cfg.num_slots,
            self.cfg.action_dim,
        )
        hard_actions = candidates.gather(1, gather).squeeze(1)
        soft_actions = (probabilities.view(-1, self.num_candidates, 1, 1, 1) * candidates).sum(dim=1)
        actions = hard_actions + (soft_actions - soft_actions.detach())
        return actions, choice

    def _losses_from_candidates(
        self,
        candidates: torch.Tensor,
        logits: torch.Tensor,
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        target = batch["target_actions"].unsqueeze(1)
        valid = batch["target_valid"].float().unsqueeze(1).unsqueeze(-1)
        denom = valid.sum(dim=(2, 3, 4)).clamp_min(1.0)
        per_candidate = ((candidates - target).pow(2) * valid).sum(dim=(2, 3, 4)) / denom
        best_loss, best_idx = per_candidate.min(dim=1)
        weight = batch.get("sample_weight")
        if weight is not None:
            sample_weight = weight.float()
            mse_loss = (best_loss * sample_weight).sum() / sample_weight.sum().clamp_min(1.0e-6)
        else:
            mse_loss = best_loss.mean()
        slot_valid = batch["target_valid"].any(dim=1).float()
        pooled_logits = (logits * slot_valid.unsqueeze(-1)).sum(dim=1) / slot_valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        ce = F.cross_entropy(pooled_logits, best_idx)
        loss = mse_loss + self.candidate_ce_weight * ce
        gather = best_idx.view(-1, 1, 1, 1, 1).expand(
            -1,
            1,
            self.cfg.horizon_steps,
            self.cfg.num_slots,
            self.cfg.action_dim,
        )
        selected = candidates.gather(1, gather).squeeze(1)
        return {
            "loss": loss,
            "action_mse_norm": best_loss.mean().detach(),
            "candidate_ce": ce.detach(),
            "pred_actions_normalized": selected,
        }

    @torch.no_grad()
    def sample_actions_with_xi(
        self,
        batch: dict[str, torch.Tensor],
        *,
        candidate_index: torch.Tensor | None = None,
        deterministic: bool = False,
        temperature: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        """根据显式的 ``Xi_world`` 选择一个 CAT-K 背景车动作分支。

        ``candidate_index`` 是离散世界模型随机变量。省略它时，方法按候选
        概率采样；设定 ``deterministic=True`` 时选择最高概率候选。环境接口
        默认关闭连续动作噪声，因此测试空间中的世界模型随机性仅为候选分支。
        """
        candidates, logits = self.forward_candidates(batch)
        pooled_logits, probs = self._candidate_probabilities(
            logits,
            batch["current_valid"],
            temperature=temperature,
        )
        if candidate_index is not None:
            choice = candidate_index.to(device=candidates.device, dtype=torch.long).reshape(-1)
            if len(choice) != int(candidates.shape[0]):
                raise ValueError(
                    "candidate_index batch size mismatch: "
                    f"expected {int(candidates.shape[0])}, got {len(choice)}"
                )
            if torch.any((choice < 0) | (choice >= self.num_candidates)):
                raise ValueError(f"candidate_index must lie in [0, {self.num_candidates - 1}]")
        elif deterministic:
            choice = torch.argmax(pooled_logits, dim=-1)
        else:
            choice = torch.multinomial(probs, num_samples=1, generator=generator).squeeze(1)
        gather = choice.view(-1, 1, 1, 1, 1).expand(
            -1,
            1,
            self.cfg.horizon_steps,
            self.cfg.num_slots,
            self.cfg.action_dim,
        )
        action = candidates.gather(1, gather).squeeze(1)
        return {
            "actions": action,
            "candidate_index": choice,
            "candidate_probabilities": probs,
        }

    @torch.no_grad()
    def sample_actions(
        self,
        batch: dict[str, torch.Tensor],
        *,
        deterministic: bool = False,
        temperature: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """返回单个候选动作的统一采样接口。"""
        sampled = self.sample_actions_with_xi(
            batch,
            deterministic=deterministic,
            temperature=temperature,
            generator=generator,
        )
        return sampled["actions"]


class RelativeInteractionGraphAttention(nn.Module):
    """One deterministic relative-geometry message-passing block.

    The edge values are reconstructed from the existing normalized state tensor.
    They are an internal fixed transform, not additional model input or a new
    test-space random variable.
    """

    def __init__(
        self,
        hidden_dim: int,
        *,
        state_mean: np.ndarray,
        state_std: np.ndarray,
        dropout: float,
    ) -> None:
        super().__init__()
        self.register_buffer("state_mean", torch.as_tensor(state_mean, dtype=torch.float32))
        self.register_buffer("state_std", torch.as_tensor(state_std, dtype=torch.float32))
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.edge_embedding = nn.Sequential(
            nn.Linear(8, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.edge_bias = nn.Linear(hidden_dim, 1, bias=False)
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def _edge_features(self, current_states: torch.Tensor, current_valid: torch.Tensor) -> torch.Tensor:
        raw = current_states * self.state_std.view(1, 1, -1) + self.state_mean.view(1, 1, -1)
        source = raw.unsqueeze(2)
        target = raw.unsqueeze(1)
        rel_x = target[..., 0] - source[..., 0]
        rel_y = target[..., 1] - source[..., 1]
        rel_vx = target[..., 2] - source[..., 2]
        rel_vy = target[..., 3] - source[..., 3]
        gap = torch.clamp(torch.abs(rel_x) - 0.5 * (DEFAULT_EGO_LENGTH_M + DEFAULT_OTHER_LENGTH_M), min=0.0)
        closing = torch.clamp(torch.where(rel_x >= 0.0, -rel_vx, rel_vx), min=0.0)
        ttc = torch.clamp(gap / closing.clamp_min(1.0e-3), min=0.0, max=10.0)
        drac = torch.clamp(closing.square() / (2.0 * gap.clamp_min(1.0e-3)), min=0.0, max=12.0)
        edge_valid = (current_valid.bool().unsqueeze(2) & current_valid.bool().unsqueeze(1)).float()
        return torch.stack((rel_x, rel_y, rel_vx, rel_vy, gap, closing, ttc, drac), dim=-1) * edge_valid.unsqueeze(-1)

    def forward(
        self,
        agent_context: torch.Tensor,
        current_states: torch.Tensor,
        current_valid: torch.Tensor,
    ) -> torch.Tensor:
        edge = self.edge_embedding(self._edge_features(current_states, current_valid))
        scale = math.sqrt(float(agent_context.shape[-1]))
        scores = torch.matmul(self.query(agent_context), self.key(agent_context).transpose(-1, -2)) / scale
        scores = scores + self.edge_bias(edge).squeeze(-1)
        key_padding = ~current_valid.bool().unsqueeze(1)
        scores = scores.masked_fill(key_padding, torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores, dim=-1)
        messages = torch.matmul(attention, self.value(agent_context))
        messages = messages + (attention.unsqueeze(-1) * edge).sum(dim=2)
        updated = self.norm(agent_context + self.dropout(self.output(messages)))
        return torch.where(current_valid.bool().unsqueeze(-1), updated, torch.zeros_like(updated))


class CATKResidualDynamics(SharedStartRollWorldModel):
    """Internal CAT-K residual decoder."""

    def __init__(
        self,
        cfg: WorldModelConfig,
        *,
        num_candidates: int = 8,
        state_norm_mean: np.ndarray | list[float] | tuple[float, ...] | None = None,
        state_norm_std: np.ndarray | list[float] | tuple[float, ...] | None = None,
        action_norm_mean: np.ndarray | list[float] | tuple[float, ...] | None = None,
        action_norm_std: np.ndarray | list[float] | tuple[float, ...] | None = None,
        responsibility_temperature: float = 1.0,
        energy_weight: float = 0.01,
        diversity_weight: float = 0.02,
        diversity_margin_normalized: float = 0.10,
        smoothness_weight: float = 0.01,
        map_action_weight: float = 0.0,
        probability_entropy_weight: float = 0.0,
        min_probability_entropy: float = 1.50,
        jerk_control_points: int = 5,
        max_jerk_longitudinal_mps3: float = 8.0,
        max_jerk_lateral_mps3: float = 5.0,
    ) -> None:
        super().__init__(cfg)
        self.num_candidates = max(2, int(num_candidates))
        self.responsibility_temperature = max(float(responsibility_temperature), 1.0e-3)
        self.energy_weight = max(float(energy_weight), 0.0)
        self.diversity_weight = max(float(diversity_weight), 0.0)
        self.diversity_margin_normalized = max(float(diversity_margin_normalized), 0.0)
        self.smoothness_weight = max(float(smoothness_weight), 0.0)
        self.map_action_weight = max(float(map_action_weight), 0.0)
        self.probability_entropy_weight = max(float(probability_entropy_weight), 0.0)
        self.min_probability_entropy = max(float(min_probability_entropy), 0.0)
        self.jerk_control_points = max(2, int(jerk_control_points))
        hidden = int(cfg.hidden_dim)
        state_mean = np.zeros(cfg.state_dim, dtype=np.float32) if state_norm_mean is None else np.asarray(state_norm_mean, dtype=np.float32)
        state_std = np.ones(cfg.state_dim, dtype=np.float32) if state_norm_std is None else np.asarray(state_norm_std, dtype=np.float32)
        action_mean = np.zeros(cfg.action_dim, dtype=np.float32) if action_norm_mean is None else np.asarray(action_norm_mean, dtype=np.float32)
        action_std = np.ones(cfg.action_dim, dtype=np.float32) if action_norm_std is None else np.asarray(action_norm_std, dtype=np.float32)
        if state_mean.shape != (cfg.state_dim,) or state_std.shape != (cfg.state_dim,):
            raise ValueError("state normalization statistics do not match state_dim")
        if action_mean.shape != (cfg.action_dim,) or action_std.shape != (cfg.action_dim,):
            raise ValueError("action normalization statistics do not match action_dim")
        self.register_buffer("action_mean", torch.as_tensor(action_mean, dtype=torch.float32))
        self.register_buffer("action_std", torch.as_tensor(action_std, dtype=torch.float32))
        self.register_buffer(
            "jerk_limits",
            torch.as_tensor([max_jerk_longitudinal_mps3, max_jerk_lateral_mps3], dtype=torch.float32),
        )
        self.graph_attention = RelativeInteractionGraphAttention(
            hidden,
            state_mean=state_mean,
            state_std=state_std,
            dropout=cfg.dropout,
        )
        self.scene_pool = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.LayerNorm(hidden))
        self.intent_tokens = nn.Embedding(self.num_candidates, hidden)
        self.intent_decoder = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.jerk_controls = nn.Linear(hidden, cfg.action_dim * self.jerk_control_points)
        self.intent_logits = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.num_candidates),
        )
        knot = torch.linspace(0.0, 1.0, self.jerk_control_points)
        time = torch.linspace(0.0, 1.0, cfg.horizon_steps).unsqueeze(-1)
        width = 1.0 / max(self.jerk_control_points - 1, 1)
        basis = torch.clamp(1.0 - torch.abs(time - knot.view(1, -1)) / width, min=0.0)
        self.register_buffer("jerk_spline_basis", basis / basis.sum(dim=-1, keepdim=True).clamp_min(1.0e-6))

    def _scene_context(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        agents = self.encode_agent_context(
            batch["history_states"],
            batch["history_valid"],
            batch["current_states"],
            batch["current_valid"],
            batch["mode_index"],
            batch["primary_slot_index"],
            batch.get("flow_action_summary"),
            batch.get("relation_features"),
        )
        graph_context = self.graph_attention(agents, batch["current_states"], batch["current_valid"])
        slots = graph_context[:, 1:, :]
        valid = batch["current_valid"][:, 1:].float().unsqueeze(-1)
        pooled = (slots * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        return slots, self.scene_pool(pooled)

    def forward_candidates(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        slot_context, scene_context = self._scene_context(batch)
        b = int(slot_context.shape[0])
        token = self.intent_tokens.weight.view(1, self.num_candidates, 1, -1)
        decoded = self.intent_decoder(
            slot_context.unsqueeze(1) + scene_context.view(b, 1, 1, -1) + token
        )
        controls = self.jerk_controls(decoded).view(
            b,
            self.num_candidates,
            self.cfg.num_slots,
            self.cfg.action_dim,
            self.jerk_control_points,
        )
        jerk = torch.einsum("tc,bnsdc->bntsd", self.jerk_spline_basis, controls)
        jerk = torch.tanh(jerk) * self.jerk_limits.view(1, 1, 1, 1, -1)
        raw_current = self.graph_attention.state_mean.view(1, 1, -1) + (
            batch["current_states"][:, 1:, :] * self.graph_attention.state_std.view(1, 1, -1)
        )
        reference = raw_current[:, None, None, :, 4:6]
        delta = torch.cumsum(jerk, dim=2) / 25.0
        raw_actions = reference + delta
        raw_actions = torch.stack(
            (
                torch.clamp(raw_actions[..., 0], -8.0, 4.0),
                torch.clamp(raw_actions[..., 1], -4.0, 4.0),
            ),
            dim=-1,
        )
        candidates = (raw_actions - self.action_mean.view(1, 1, 1, 1, -1)) / self.action_std.view(1, 1, 1, 1, -1)
        return candidates, self.intent_logits(scene_context)

    def _candidate_terms(
        self,
        candidates: torch.Tensor,
        logits: torch.Tensor,
        target: torch.Tensor,
        valid: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        mask = valid.float().unsqueeze(1).unsqueeze(-1)
        denom = mask.sum(dim=(2, 3, 4)).clamp_min(1.0)
        distance = ((candidates - target.unsqueeze(1)).square() * mask).sum(dim=(2, 3, 4)) / denom
        log_probs = F.log_softmax(logits, dim=-1)
        responsibilities = F.softmax(log_probs - distance / self.responsibility_temperature, dim=-1)
        mixture = -self.responsibility_temperature * torch.logsumexp(
            log_probs - distance / self.responsibility_temperature,
            dim=-1,
        )
        pair_mask = valid.float().unsqueeze(1).unsqueeze(1).unsqueeze(-1)
        pair_sq = (candidates.unsqueeze(2) - candidates.unsqueeze(1)).square()
        pair_distance = (pair_sq * pair_mask).sum(dim=(3, 4, 5)) / pair_mask.sum(dim=(3, 4, 5)).clamp_min(1.0)
        responsibility_pair = responsibilities.unsqueeze(2) * responsibilities.unsqueeze(1)
        expected_distance = (responsibilities * distance).sum(dim=-1)
        energy = expected_distance - 0.5 * (responsibility_pair * pair_distance).sum(dim=(1, 2))
        off_diagonal = 1.0 - torch.eye(self.num_candidates, device=candidates.device, dtype=candidates.dtype).view(1, self.num_candidates, self.num_candidates)
        pair_norm = torch.sqrt(pair_distance.clamp_min(0.0) + 1.0e-6)
        has_valid_action = valid.reshape(valid.shape[0], -1).any(dim=1).to(dtype=candidates.dtype)
        diversity = (
            responsibility_pair
            * off_diagonal
            * F.relu(self.diversity_margin_normalized - pair_norm)
        ).sum(dim=(1, 2)) * has_valid_action
        raw_candidates = candidates * self.action_std.view(1, 1, 1, 1, -1) + self.action_mean.view(1, 1, 1, 1, -1)
        jerk = (raw_candidates[:, :, 1:] - raw_candidates[:, :, :-1]) * 25.0
        smooth = (responsibilities * jerk.square().mean(dim=(2, 3, 4))).sum(dim=-1)
        return {
            "distance": distance,
            "responsibilities": responsibilities,
            "mixture": mixture,
            "energy": energy,
            "diversity": diversity,
            "smooth": smooth,
            "pairwise_distance": pair_norm,
        }

    @staticmethod
    def _weighted_mean(values: torch.Tensor, weights: torch.Tensor | None) -> torch.Tensor:
        if weights is None:
            return values.mean()
        weights = weights.float()
        return (values * weights).sum() / weights.sum().clamp_min(1.0e-6)

    def _losses_from_candidates(
        self,
        candidates: torch.Tensor,
        logits: torch.Tensor,
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        terms = self._candidate_terms(candidates, logits, batch["target_actions"], batch["target_valid"])
        weights = batch.get("sample_weight")
        mixture = self._weighted_mean(terms["mixture"], weights)
        energy = self._weighted_mean(terms["energy"], weights)
        diversity = self._weighted_mean(terms["diversity"], weights)
        smooth = self._weighted_mean(terms["smooth"], weights)
        probs = F.softmax(logits, dim=-1)
        map_choice = torch.argmax(logits, dim=-1)
        map_gather = map_choice.view(-1, 1, 1, 1, 1).expand(
            -1,
            1,
            self.cfg.horizon_steps,
            self.cfg.num_slots,
            self.cfg.action_dim,
        )
        map_hard_actions = candidates.gather(1, map_gather).squeeze(1)
        map_soft_actions = (probs.view(-1, self.num_candidates, 1, 1, 1) * candidates).sum(dim=1)
        map_actions = map_hard_actions + (map_soft_actions - map_soft_actions.detach())
        map_mask = batch["target_valid"].float().unsqueeze(-1)
        map_distance = ((map_actions - batch["target_actions"]).square() * map_mask).sum(
            dim=(1, 2, 3)
        ) / map_mask.sum(dim=(1, 2, 3)).clamp_min(1.0)
        map_action = self._weighted_mean(map_distance, weights)
        probability_entropy_per_sample = -(probs * probs.clamp_min(1.0e-8).log()).sum(dim=-1)
        entropy_penalty = self._weighted_mean(F.relu(self.min_probability_entropy - probability_entropy_per_sample), weights)
        loss = (
            mixture
            + self.energy_weight * energy
            + self.diversity_weight * diversity
            + self.smoothness_weight * smooth
            + self.map_action_weight * map_action
            + self.probability_entropy_weight * entropy_penalty
        )
        responsibilities = terms["responsibilities"]
        expected = (responsibilities.view(-1, self.num_candidates, 1, 1, 1) * candidates).sum(dim=1)
        entropy = probability_entropy_per_sample.mean()
        responsibility_entropy = -(
            responsibilities * responsibilities.clamp_min(1.0e-8).log()
        ).sum(dim=-1).mean()
        effective = torch.exp(-(responsibilities * responsibilities.clamp_min(1.0e-8).log()).sum(dim=-1)).mean()
        off_diagonal = ~torch.eye(self.num_candidates, device=candidates.device, dtype=torch.bool)
        pairwise = terms["pairwise_distance"][:, off_diagonal].mean()
        result = {
            "loss": loss,
            "action_mse_norm": terms["distance"].min(dim=-1).values.mean().detach(),
            "mixture_nll": mixture.detach(),
            "energy_score": energy.detach(),
            "diversity_penalty": diversity.detach(),
            "smoothness_penalty": smooth.detach(),
            "map_action_mse_norm": map_action.detach(),
            "probability_entropy_penalty": entropy_penalty.detach(),
            "candidate_entropy": entropy.detach(),
            "responsibility_entropy": responsibility_entropy.detach(),
            "effective_candidates": effective.detach(),
            "probability_responsibility_l1": torch.abs(probs - responsibilities).mean().detach(),
            "pairwise_trajectory_distance_norm": pairwise.detach(),
            "pred_actions_normalized": expected,
            "candidate_actions_normalized": candidates,
            "candidate_responsibilities": responsibilities,
            "candidate_probabilities": probs,
            "candidate_logits": logits,
        }
        return result

    def p_losses(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        candidates, logits = self.forward_candidates(batch)
        return self._losses_from_candidates(candidates, logits, batch)

    def select_map_actions_st(
        self,
        candidates: torch.Tensor,
        probabilities: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Use the environment's MAP Xi branch while retaining probability gradients."""
        choice = torch.argmax(probabilities, dim=-1)
        gather = choice.view(-1, 1, 1, 1, 1).expand(
            -1,
            1,
            self.cfg.horizon_steps,
            self.cfg.num_slots,
            self.cfg.action_dim,
        )
        hard_actions = candidates.gather(1, gather).squeeze(1)
        soft_actions = (probabilities.view(-1, self.num_candidates, 1, 1, 1) * candidates).sum(dim=1)
        actions = hard_actions + (soft_actions - soft_actions.detach())
        return actions, choice

    @torch.no_grad()
    def sample_actions_with_xi(
        self,
        batch: dict[str, torch.Tensor],
        *,
        candidate_index: torch.Tensor | None = None,
        deterministic: bool = False,
        temperature: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        candidates, logits = self.forward_candidates(batch)
        probs = F.softmax(logits / max(float(temperature), 1.0e-3), dim=-1)
        if candidate_index is not None:
            choice = candidate_index.to(device=candidates.device, dtype=torch.long).reshape(-1)
            if len(choice) != int(candidates.shape[0]):
                raise ValueError("candidate_index batch size mismatch")
            if torch.any((choice < 0) | (choice >= self.num_candidates)):
                raise ValueError(f"candidate_index must lie in [0, {self.num_candidates - 1}]")
        elif deterministic:
            choice = torch.argmax(logits, dim=-1)
        else:
            choice = torch.multinomial(probs, num_samples=1, generator=generator).squeeze(1)
        actions = candidates.gather(
            1,
            choice.view(-1, 1, 1, 1, 1).expand(-1, 1, self.cfg.horizon_steps, self.cfg.num_slots, self.cfg.action_dim),
        ).squeeze(1)
        return {
            "actions": actions,
            "candidate_index": choice,
            "candidate_probabilities": probs,
        }

    @torch.no_grad()
    def sample_actions(
        self,
        batch: dict[str, torch.Tensor],
        *,
        deterministic: bool = False,
        temperature: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        return self.sample_actions_with_xi(
            batch,
            deterministic=deterministic,
            temperature=temperature,
            generator=generator,
        )["actions"]


class CATKTopKWorldModel(CATKResidualDynamics):
    """最终 CAT-K：一个名义分支与七个可学习残差意图。"""

    def __init__(
        self,
        cfg: WorldModelConfig,
        *,
        nominal_logit_margin: float = 0.05,
        **kwargs,
    ) -> None:
        super().__init__(cfg, **kwargs)
        self.nominal_logit_margin = max(float(nominal_logit_margin), 0.0)
        self.nominal_decoder = NominalCATKDecoder(
            cfg,
            num_candidates=self.num_candidates,
        )

    def forward_candidates(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        residual_candidates, residual_logits = super().forward_candidates(batch)
        if self.training:
            nominal_actions, _nominal_choice = self.nominal_decoder.select_map_actions_st(batch)
        else:
            nominal_actions = self.nominal_decoder.sample_actions(batch, deterministic=True)
        candidates = residual_candidates.clone()
        candidates[:, 0] = nominal_actions
        logits = residual_logits.clone()
        logits[:, 0] = logits[:, 1:].max(dim=-1).values + self.nominal_logit_margin
        return candidates, logits

def build_model_from_schema(schema: dict[str, Any], config: dict[str, Any]) -> SharedStartRollWorldModel:
    model_cfg = dict(config.get("model", {}))
    model_type = str(model_cfg.get("type", "catk_topk")).lower()
    cfg = WorldModelConfig(
        history_steps=int(schema["history_steps"]),
        horizon_steps=int(schema["horizon_steps"]),
        hidden_dim=int(model_cfg.get("hidden_dim", 128)),
        history_layers=int(model_cfg.get("history_layers", 2)),
        interaction_layers=int(model_cfg.get("interaction_layers", 2)),
        num_heads=int(model_cfg.get("num_heads", 4)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        use_start_flow_summary=bool(model_cfg.get("use_start_flow_summary", False)),
        relation_feature_dim=len(schema.get("relation_features", RELATION_FEATURES)),
        use_relation_features=bool(model_cfg.get("use_relation_features", False)),
        min_log_std=float(model_cfg.get("min_log_std", -5.0)),
        max_log_std=float(model_cfg.get("max_log_std", 2.0)),
    )
    if model_type != "catk_topk":
        raise ValueError("The active world model type is 'catk_topk'.")
    kwargs = {
        "num_candidates": int(model_cfg.get("num_candidates", 8)),
        "state_norm_mean": np.asarray(schema["normalization"]["state"]["mean"], dtype=np.float32),
        "state_norm_std": np.asarray(schema["normalization"]["state"]["std"], dtype=np.float32),
        "action_norm_mean": np.asarray(schema["normalization"]["action"]["mean"], dtype=np.float32),
        "action_norm_std": np.asarray(schema["normalization"]["action"]["std"], dtype=np.float32),
        "responsibility_temperature": float(model_cfg.get("responsibility_temperature", 1.0)),
        "energy_weight": float(model_cfg.get("energy_weight", 0.01)),
        "diversity_weight": float(model_cfg.get("diversity_weight", 0.02)),
        "diversity_margin_normalized": float(model_cfg.get("diversity_margin_normalized", 0.10)),
        "smoothness_weight": float(model_cfg.get("smoothness_weight", 0.01)),
        "map_action_weight": float(model_cfg.get("map_action_weight", 0.0)),
        "probability_entropy_weight": float(model_cfg.get("probability_entropy_weight", 0.0)),
        "min_probability_entropy": float(model_cfg.get("min_probability_entropy", 1.50)),
        "jerk_control_points": int(model_cfg.get("jerk_control_points", 5)),
        "max_jerk_longitudinal_mps3": float(model_cfg.get("max_jerk_longitudinal_mps3", 8.0)),
        "max_jerk_lateral_mps3": float(model_cfg.get("max_jerk_lateral_mps3", 5.0)),
    }
    kwargs["nominal_logit_margin"] = float(model_cfg.get("nominal_logit_margin", 0.05))
    return CATKTopKWorldModel(cfg, **kwargs)


def model_config_payload(model: SharedStartRollWorldModel) -> dict[str, Any]:
    cfg = model.cfg
    if not isinstance(model, CATKTopKWorldModel):
        raise TypeError(f"Unsupported checkpoint model={model.__class__.__name__}")
    payload = {
        "model_type": "catk_topk",
        "history_steps": int(cfg.history_steps),
        "horizon_steps": int(cfg.horizon_steps),
        "state_dim": int(cfg.state_dim),
        "action_dim": int(cfg.action_dim),
        "flow_summary_dim": int(cfg.flow_summary_dim),
        "relation_feature_dim": int(cfg.relation_feature_dim),
        "num_agents": int(cfg.num_agents),
        "num_slots": int(cfg.num_slots),
        "hidden_dim": int(cfg.hidden_dim),
        "history_layers": int(cfg.history_layers),
        "interaction_layers": int(cfg.interaction_layers),
        "num_heads": int(cfg.num_heads),
        "dropout": float(cfg.dropout),
        "use_start_flow_summary": bool(cfg.use_start_flow_summary),
        "use_relation_features": bool(cfg.use_relation_features),
        "min_log_std": float(cfg.min_log_std),
        "max_log_std": float(cfg.max_log_std),
    }
    payload.update(
        {
            "num_candidates": int(model.num_candidates),
            "state_norm_mean": model.graph_attention.state_mean.detach().cpu().tolist(),
            "state_norm_std": model.graph_attention.state_std.detach().cpu().tolist(),
            "action_norm_mean": model.action_mean.detach().cpu().tolist(),
            "action_norm_std": model.action_std.detach().cpu().tolist(),
            "responsibility_temperature": float(model.responsibility_temperature),
            "energy_weight": float(model.energy_weight),
            "diversity_weight": float(model.diversity_weight),
            "diversity_margin_normalized": float(model.diversity_margin_normalized),
            "smoothness_weight": float(model.smoothness_weight),
            "map_action_weight": float(model.map_action_weight),
            "probability_entropy_weight": float(model.probability_entropy_weight),
            "min_probability_entropy": float(model.min_probability_entropy),
            "jerk_control_points": int(model.jerk_control_points),
            "max_jerk_longitudinal_mps3": float(model.jerk_limits[0].detach().cpu()),
            "max_jerk_lateral_mps3": float(model.jerk_limits[1].detach().cpu()),
        }
    )
    payload["nominal_logit_margin"] = float(model.nominal_logit_margin)
    return payload


def load_checkpoint(path: str, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    payload = torch.load(path, map_location=device)
    model_config = dict(payload["model_config"])
    model_type = str(model_config.pop("model_type", "catk_topk")).lower()
    if model_type != "catk_topk":
        raise ValueError(f"Unsupported checkpoint model_type={model_type!r}")
    state_dict = payload["state_dict"]
    num_candidates = int(model_config.pop("num_candidates", 8))
    if any(key.startswith("nominal_decoder.") for key in state_dict):
        state_norm_mean = model_config.pop("state_norm_mean", None)
        state_norm_std = model_config.pop("state_norm_std", None)
        action_norm_mean = model_config.pop("action_norm_mean", None)
        action_norm_std = model_config.pop("action_norm_std", None)
        responsibility_temperature = float(model_config.pop("responsibility_temperature", 1.0))
        energy_weight = float(model_config.pop("energy_weight", 0.01))
        diversity_weight = float(model_config.pop("diversity_weight", 0.02))
        diversity_margin_normalized = float(model_config.pop("diversity_margin_normalized", 0.10))
        smoothness_weight = float(model_config.pop("smoothness_weight", 0.01))
        map_action_weight = float(model_config.pop("map_action_weight", 0.0))
        probability_entropy_weight = float(model_config.pop("probability_entropy_weight", 0.0))
        min_probability_entropy = float(model_config.pop("min_probability_entropy", 1.50))
        jerk_control_points = int(model_config.pop("jerk_control_points", 5))
        max_jerk_longitudinal_mps3 = float(model_config.pop("max_jerk_longitudinal_mps3", 8.0))
        max_jerk_lateral_mps3 = float(model_config.pop("max_jerk_lateral_mps3", 5.0))
        nominal_logit_margin = float(model_config.pop("nominal_logit_margin", 0.05))
        cfg = WorldModelConfig(**model_config)
        model = CATKTopKWorldModel(
            cfg,
            num_candidates=num_candidates,
            state_norm_mean=state_norm_mean,
            state_norm_std=state_norm_std,
            action_norm_mean=action_norm_mean,
            action_norm_std=action_norm_std,
            responsibility_temperature=responsibility_temperature,
            energy_weight=energy_weight,
            diversity_weight=diversity_weight,
            diversity_margin_normalized=diversity_margin_normalized,
            smoothness_weight=smoothness_weight,
            map_action_weight=map_action_weight,
            probability_entropy_weight=probability_entropy_weight,
            min_probability_entropy=min_probability_entropy,
            jerk_control_points=jerk_control_points,
            max_jerk_longitudinal_mps3=max_jerk_longitudinal_mps3,
            max_jerk_lateral_mps3=max_jerk_lateral_mps3,
            nominal_logit_margin=nominal_logit_margin,
        )
    elif "candidate_head.0.weight" in state_dict:
        candidate_ce_weight = float(model_config.pop("candidate_ce_weight", 0.05))
        cfg = WorldModelConfig(**model_config)
        model = NominalCATKDecoder(
            cfg,
            num_candidates=num_candidates,
            candidate_ce_weight=candidate_ce_weight,
        )
    else:
        raise ValueError("Checkpoint state_dict does not match a supported CAT-K model")
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model, payload


def numpy_batch_to_torch(arrays: dict[str, np.ndarray], idx: np.ndarray, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "history_states": torch.from_numpy(arrays["history_states_normalized"][idx]).float().to(device),
        "history_valid": torch.from_numpy(arrays["history_valid"][idx]).bool().to(device),
        "current_states": torch.from_numpy(arrays["current_states_normalized"][idx]).float().to(device),
        "current_valid": torch.from_numpy(arrays["current_valid"][idx]).bool().to(device),
        "mode_index": torch.from_numpy(arrays["mode_index"][idx]).long().to(device),
        "primary_slot_index": torch.from_numpy(arrays["primary_slot_index"][idx]).long().to(device),
        "flow_action_summary": torch.from_numpy(arrays["flow_action_summary_normalized"][idx]).float().to(device),
        "relation_features": torch.from_numpy(arrays["relation_features_normalized"][idx]).float().to(device),
        "target_actions": torch.from_numpy(arrays["target_actions_normalized"][idx]).float().to(device),
        "target_valid": torch.from_numpy(arrays["target_valid"][idx]).bool().to(device),
        "sample_weight": torch.from_numpy(arrays["sample_weight"][idx]).float().to(device),
    }
