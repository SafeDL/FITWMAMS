"""Shared START/ROLL stochastic background-traffic policy."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .schema import (
    AGENT_NAMES,
    AGENT_STATE_FEATURES,
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
    min_log_std: float = -5.0
    max_log_std: float = 2.0


class SharedStartRollWorldModel(nn.Module):
    """A shared stochastic policy for START and ROLL conditions.

    The model follows the practical TrafficBots/VBD pattern at highD scale:
    a temporal scene encoder, an inter-agent Transformer, and a Gaussian action
    head. START and ROLL share all parameters and differ only through mode
    embeddings and the available history mask.
    """

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
        self.mean = nn.Linear(hidden, cfg.action_dim)
        self.log_std = nn.Linear(hidden, cfg.action_dim)

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
        return interaction[:, 1:, :]

    def decode_gaussian(self, slot_context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b = int(slot_context.shape[0])
        device = slot_context.device
        t_ids = torch.arange(self.cfg.horizon_steps, device=device)
        horizon = self.horizon_embed(t_ids).view(1, self.cfg.horizon_steps, 1, -1)
        slot = slot_context.view(b, 1, self.cfg.num_slots, -1)
        decoded = self.action_head(slot + horizon)
        mean = self.mean(decoded)
        log_std = torch.clamp(self.log_std(decoded), self.cfg.min_log_std, self.cfg.max_log_std)
        return mean, log_std

    def forward(
        self,
        history_states: torch.Tensor,
        history_valid: torch.Tensor,
        current_states: torch.Tensor,
        current_valid: torch.Tensor,
        mode_index: torch.Tensor,
        primary_slot_index: torch.Tensor,
        flow_action_summary: torch.Tensor | None = None,
        relation_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return Gaussian mean/log_std over future slot actions."""
        slot_context = self.encode_context(
            history_states,
            history_valid,
            current_states,
            current_valid,
            mode_index,
            primary_slot_index,
            flow_action_summary,
            relation_features,
        )
        return self.decode_gaussian(slot_context)

    def sample(
        self,
        batch: dict[str, torch.Tensor],
        *,
        deterministic: bool = False,
        temperature: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std = self(
            batch["history_states"],
            batch["history_valid"],
            batch["current_states"],
            batch["current_valid"],
            batch["mode_index"],
            batch["primary_slot_index"],
            batch.get("flow_action_summary"),
            batch.get("relation_features"),
        )
        if deterministic:
            action = mean
        else:
            noise = torch.randn(mean.shape, device=mean.device, dtype=mean.dtype, generator=generator)
            action = mean + noise * log_std.exp() * max(float(temperature), 0.0)
        return action, mean, log_std


class TopKStartRollWorldModel(SharedStartRollWorldModel):
    """CAT-K/SMART-style candidate policy with best-of-K closed-loop supervision."""

    def __init__(
        self,
        cfg: WorldModelConfig,
        *,
        num_candidates: int = 8,
        candidate_ce_weight: float = 0.05,
        branch_noise_std: float = 0.0,
    ) -> None:
        super().__init__(cfg)
        self.num_candidates = max(2, int(num_candidates))
        self.candidate_ce_weight = float(candidate_ce_weight)
        self.branch_noise_std = float(branch_noise_std)
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

    def p_losses(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        candidates, logits = self.forward_candidates(batch)
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
        add_branch_noise: bool = False,
    ) -> dict[str, torch.Tensor]:
        """根据显式的 ``Xi_world`` 选择一个 CAT-K 背景车动作分支。

        ``candidate_index`` 是离散世界模型随机变量。省略它时，方法按候选
        概率采样；设定 ``deterministic=True`` 时选择最高概率候选。环境接口
        默认关闭连续动作噪声，因此测试空间中的世界模型随机性仅为候选分支。
        """
        candidates, logits = self.forward_candidates(batch)
        slot_valid = batch["current_valid"][:, 1:].float()
        pooled_logits = (logits * slot_valid.unsqueeze(-1)).sum(dim=1) / slot_valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        probs = F.softmax(pooled_logits / max(float(temperature), 1.0e-3), dim=-1)
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
        if add_branch_noise and self.branch_noise_std > 0.0:
            noise = torch.randn(action.shape, device=action.device, dtype=action.dtype, generator=generator)
            action = action + noise * self.branch_noise_std * max(float(temperature), 0.0)
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
        """兼容旧评测接口的动作采样包装。"""
        sampled = self.sample_actions_with_xi(
            batch,
            deterministic=deterministic,
            temperature=temperature,
            generator=generator,
            add_branch_noise=not deterministic,
        )
        return sampled["actions"]


def gaussian_nll(
    actions: torch.Tensor,
    mean: torch.Tensor,
    log_std: torch.Tensor,
    valid: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    valid_f = valid.float().unsqueeze(-1)
    nll = 0.5 * ((actions - mean) / log_std.exp()).pow(2) + log_std
    per_sample = (nll * valid_f).sum(dim=(1, 2, 3)) / valid_f.sum(dim=(1, 2, 3)).clamp_min(1.0)
    if sample_weight is not None:
        weight = sample_weight.float()
        return (per_sample * weight).sum() / weight.sum().clamp_min(1.0e-6)
    return per_sample.mean()


def masked_action_mse(actions: torch.Tensor, mean: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    valid_f = valid.float().unsqueeze(-1)
    return (((actions - mean).pow(2) * valid_f).sum() / valid_f.sum().clamp_min(1.0)).mean()


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
    if model_type == "catk_topk":
        return TopKStartRollWorldModel(
            cfg,
            num_candidates=int(model_cfg.get("num_candidates", 8)),
            candidate_ce_weight=float(model_cfg.get("candidate_ce_weight", 0.05)),
            branch_noise_std=float(model_cfg.get("branch_noise_std", 0.0)),
        )
    if model_type == "gaussian_baseline":
        return SharedStartRollWorldModel(cfg)
    raise ValueError(
        "Unsupported world-model type. Use 'catk_topk' or the retained "
        "comparison baseline 'gaussian_baseline'."
    )


def model_config_payload(model: SharedStartRollWorldModel) -> dict[str, Any]:
    cfg = model.cfg
    if isinstance(model, TopKStartRollWorldModel):
        model_type = "catk_topk"
    else:
        model_type = "gaussian_baseline"
    payload = {
        "model_type": model_type,
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
    if isinstance(model, TopKStartRollWorldModel):
        payload.update(
            {
                "num_candidates": int(model.num_candidates),
                "candidate_ce_weight": float(model.candidate_ce_weight),
                "branch_noise_std": float(model.branch_noise_std),
            }
        )
    return payload


def load_checkpoint(path: str, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    payload = torch.load(path, map_location=device)
    model_config = dict(payload["model_config"])
    model_type = str(model_config.pop("model_type", "catk_topk")).lower()
    if model_type == "catk_topk":
        num_candidates = int(model_config.pop("num_candidates", 8))
        candidate_ce_weight = float(model_config.pop("candidate_ce_weight", 0.05))
        branch_noise_std = float(model_config.pop("branch_noise_std", 0.0))
        cfg = WorldModelConfig(**model_config)
        model = TopKStartRollWorldModel(
            cfg,
            num_candidates=num_candidates,
            candidate_ce_weight=candidate_ce_weight,
            branch_noise_std=branch_noise_std,
        )
    elif model_type == "gaussian_baseline":
        cfg = WorldModelConfig(**model_config)
        model = SharedStartRollWorldModel(cfg)
    else:
        raise ValueError(f"Unsupported checkpoint model_type={model_type!r}")
    model.load_state_dict(payload["state_dict"], strict=True)
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
