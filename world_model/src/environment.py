"""面向 ADS 测试的 CAT-K 背景交通环境动力学接口。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .model import TopKStartRollWorldModel, load_checkpoint
from .rollout import (
    build_relation_features_from_current,
    build_start_condition_from_flow_feature,
    integrate_background_actions,
    normalize_relation_features,
    normalize_states,
    unnormalize_actions,
)
from .schema import AGENT_STATE_FEATURES, ROLL_MODE_INDEX, SLOT_NAMES, START_MODE_INDEX


@dataclass(frozen=True)
class WorldSamplingConfig:
    """固定 ``Xi_world`` 的候选分支采样语义。"""

    candidate_selection: str = "categorical"
    candidate_temperature: float = 1.0
    world_seed: int = 123

    def __post_init__(self) -> None:
        if self.candidate_selection not in {"categorical", "argmax"}:
            raise ValueError("candidate_selection must be 'categorical' or 'argmax'")
        if float(self.candidate_temperature) <= 0.0:
            raise ValueError("candidate_temperature must be positive")


@dataclass
class BackgroundChunk:
    """一个一秒背景交通 rollout chunk 及其 ``Xi_world`` 取值。"""

    mode: str
    actions_mps2: np.ndarray
    background_states: np.ndarray
    background_valid: np.ndarray
    candidate_index: int
    candidate_probabilities: np.ndarray


class CATKBackgroundEnvironment:
    """将固定 CAT-K checkpoint 作为条件背景交通动力学。

    初始场景由 Flow 样本 ``(feature_row, slot_mask, primary_slot)`` 唯一给出。
    ``start`` 不接收 ego future；``roll`` 只接收已经发生的 ego 状态历史。
    ADS 身份、网络特征、未来动作和风险标签均不属于此接口。
    """

    def __init__(
        self,
        model: TopKStartRollWorldModel,
        schema: dict[str, Any],
        *,
        device,
        sampling: WorldSamplingConfig | None = None,
    ) -> None:
        self.model = model
        self.schema = dict(schema)
        self.device = device
        self.sampling = sampling or WorldSamplingConfig()
        self.model.eval()
        self._generator = self._make_generator(self.sampling.world_seed)
        self._started = False
        self._candidate_indices: list[int] = []
        self._background_history: np.ndarray | None = None
        self._background_valid: np.ndarray | None = None
        self._start_condition: dict[str, np.ndarray] | None = None

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path,
        *,
        device,
        sampling: WorldSamplingConfig | None = None,
    ) -> "CATKBackgroundEnvironment":
        model, payload = load_checkpoint(str(checkpoint), device)
        if not isinstance(model, TopKStartRollWorldModel):
            raise TypeError(
                "CATKBackgroundEnvironment requires a TopKStartRollWorldModel, "
                f"got {model.__class__.__name__}"
            )
        schema = dict(payload.get("schema", {}))
        if not schema:
            raise KeyError("Checkpoint is missing the dataset schema")
        return cls(model, schema, device=device, sampling=sampling)

    @property
    def horizon_steps(self) -> int:
        return int(self.schema["horizon_steps"])

    @property
    def xi_world_candidate_indices(self) -> tuple[int, ...]:
        """当前 episode 已实际使用的离散世界模型 latent。"""
        return tuple(self._candidate_indices)

    def reset_from_flow_sample(
        self,
        feature_row: np.ndarray,
        slot_mask: np.ndarray,
        *,
        primary_slot_index: int,
        world_seed: int | None = None,
    ) -> dict[str, np.ndarray]:
        """使用完整 Flow 场景样本初始化环境，不执行背景车 rollout。"""
        if world_seed is not None:
            self._generator = self._make_generator(int(world_seed))
        else:
            self._generator = self._make_generator(self.sampling.world_seed)
        self._candidate_indices = []
        self._started = False
        self._background_history = None
        self._background_valid = None
        self._start_condition = build_start_condition_from_flow_feature(
            feature_row,
            slot_mask,
            primary_slot_index=int(primary_slot_index),
            schema=self.schema,
        )
        return self.initial_state()

    def initial_state(self) -> dict[str, np.ndarray]:
        """返回 Flow 样本对应的初始物理状态，不包含任何未来 ego 信息。"""
        if self._start_condition is None:
            raise RuntimeError("Call reset_from_flow_sample before reading the initial state")
        return {
            "current_states": self._start_condition["current_states"].copy(),
            "current_valid": self._start_condition["current_valid"].copy(),
            "slot_mask": self._start_condition["slot_mask"].copy(),
            "primary_slot_index": self._start_condition["primary_slot_index"].copy(),
        }

    def start(self, *, candidate_index: int | None = None) -> BackgroundChunk:
        """生成 START 第一秒背景车行为；不接收 ego future。"""
        if self._start_condition is None:
            raise RuntimeError("Call reset_from_flow_sample before start")
        if self._started:
            raise RuntimeError("START has already been generated; call roll for later chunks")
        condition = self._start_condition
        chunk = self._predict_chunk(
            mode_index=START_MODE_INDEX,
            history_states=condition["history_states"],
            history_valid=condition["history_valid"],
            current_states=condition["current_states"],
            current_valid=condition["current_valid"],
            primary_slot_index=int(condition["primary_slot_index"]),
            flow_action_summary=condition["flow_action_summary_normalized"],
            relation_features=condition["relation_features_normalized"],
            candidate_index=candidate_index,
            coordinate_origin_xy=np.zeros(2, dtype=np.float32),
        )
        self._background_history = chunk.background_states.copy()
        self._background_valid = chunk.background_valid.copy()
        self._started = True
        return chunk

    def roll(
        self,
        ego_history_states: np.ndarray,
        ego_history_valid: np.ndarray,
        *,
        candidate_index: int | None = None,
    ) -> BackgroundChunk:
        """根据已经发生的 ego 历史生成下一秒背景车行为。

        ``ego_history_states`` 的 shape 必须为 ``[history_steps, 6]``，最后一帧
        是当前 ego 状态。坐标必须与此前返回的背景车状态位于同一全局局部坐标系。
        """
        if not self._started or self._background_history is None or self._background_valid is None:
            raise RuntimeError("Call start before roll")
        ego_history = np.asarray(ego_history_states, dtype=np.float32)
        ego_valid = np.asarray(ego_history_valid, dtype=bool).reshape(-1)
        expected = (self.horizon_steps, len(AGENT_STATE_FEATURES))
        if tuple(ego_history.shape) != expected:
            raise ValueError(f"ego_history_states must have shape {expected}, got {tuple(ego_history.shape)}")
        if tuple(ego_valid.shape) != (self.horizon_steps,):
            raise ValueError(
                "ego_history_valid must have shape "
                f"({self.horizon_steps},), got {tuple(ego_valid.shape)}"
            )
        if not bool(ego_valid[-1]):
            raise ValueError("The current ego state, i.e. the final history frame, must be valid")

        history_global = np.zeros(
            (self.horizon_steps, 1 + len(SLOT_NAMES), len(AGENT_STATE_FEATURES)),
            dtype=np.float32,
        )
        valid = np.zeros((self.horizon_steps, 1 + len(SLOT_NAMES)), dtype=bool)
        history_global[:, 0] = ego_history
        history_global[:, 1:] = self._background_history
        valid[:, 0] = ego_valid
        valid[:, 1:] = self._background_valid
        origin_xy = ego_history[-1, :2].copy()
        history_local = history_global.copy()
        history_local[:, :, 0] -= origin_xy[0]
        history_local[:, :, 1] -= origin_xy[1]
        history_local[~valid] = 0.0
        current_local = history_local[-1]
        current_valid = valid[-1]
        relation = build_relation_features_from_current(
            current_local,
            current_valid,
            primary_slot_index=int(self._start_condition["primary_slot_index"]),
        )
        relation_valid = current_valid[1:]
        flow_summary = np.zeros(
            (len(SLOT_NAMES), len(self.schema["flow_action_summary_features"])),
            dtype=np.float32,
        )
        chunk = self._predict_chunk(
            mode_index=ROLL_MODE_INDEX,
            history_states=history_local,
            history_valid=valid,
            current_states=current_local,
            current_valid=current_valid,
            primary_slot_index=int(self._start_condition["primary_slot_index"]),
            flow_action_summary=flow_summary,
            relation_features=normalize_relation_features(relation, relation_valid, self.schema),
            candidate_index=candidate_index,
            coordinate_origin_xy=origin_xy,
        )
        self._background_history = chunk.background_states.copy()
        self._background_valid = chunk.background_valid.copy()
        return chunk

    def metadata(self) -> dict[str, Any]:
        """导出环境不确定性与采样设置，便于与 ADS 结果共同存档。"""
        return {
            "model": "catk_topk",
            "candidate_selection": self.sampling.candidate_selection,
            "candidate_temperature": float(self.sampling.candidate_temperature),
            "world_seed": int(self.sampling.world_seed),
            "xi_world_candidate_indices": list(self._candidate_indices),
        }

    def _predict_chunk(
        self,
        *,
        mode_index: int,
        history_states: np.ndarray,
        history_valid: np.ndarray,
        current_states: np.ndarray,
        current_valid: np.ndarray,
        primary_slot_index: int,
        flow_action_summary: np.ndarray,
        relation_features: np.ndarray,
        candidate_index: int | None,
        coordinate_origin_xy: np.ndarray,
    ) -> BackgroundChunk:
        import torch

        batch = {
            "history_states": torch.from_numpy(
                normalize_states(history_states, history_valid, self.schema)[None]
            ).float().to(self.device),
            "history_valid": torch.from_numpy(history_valid[None]).bool().to(self.device),
            "current_states": torch.from_numpy(
                normalize_states(current_states, current_valid, self.schema)[None]
            ).float().to(self.device),
            "current_valid": torch.from_numpy(current_valid[None]).bool().to(self.device),
            "mode_index": torch.as_tensor([mode_index], dtype=torch.long, device=self.device),
            "primary_slot_index": torch.as_tensor([primary_slot_index], dtype=torch.long, device=self.device),
            "flow_action_summary": torch.from_numpy(flow_action_summary[None]).float().to(self.device),
            "relation_features": torch.from_numpy(relation_features[None]).float().to(self.device),
        }
        xi = None
        if candidate_index is not None:
            xi = torch.as_tensor([candidate_index], dtype=torch.long, device=self.device)
        sampled = self.model.sample_actions_with_xi(
            batch,
            candidate_index=xi,
            deterministic=self.sampling.candidate_selection == "argmax",
            temperature=float(self.sampling.candidate_temperature),
            generator=self._generator,
            add_branch_noise=False,
        )
        action_raw = unnormalize_actions(sampled["actions"].detach().cpu().numpy(), self.schema)[0]
        states_local, valid = integrate_background_actions(
            current_states,
            current_valid,
            action_raw,
            dt=1.0 / float(self.schema["fps"]),
        )
        states_global = states_local.copy()
        states_global[:, :, 0] += float(coordinate_origin_xy[0])
        states_global[:, :, 1] += float(coordinate_origin_xy[1])
        selected = int(sampled["candidate_index"].detach().cpu().item())
        self._candidate_indices.append(selected)
        return BackgroundChunk(
            mode="START" if mode_index == START_MODE_INDEX else "ROLL",
            actions_mps2=action_raw.astype(np.float32),
            background_states=states_global.astype(np.float32),
            background_valid=valid.astype(bool),
            candidate_index=selected,
            candidate_probabilities=sampled["candidate_probabilities"].detach().cpu().numpy()[0].astype(np.float32),
        )

    def _make_generator(self, seed: int):
        import torch

        if str(self.device).startswith("cuda"):
            generator = torch.Generator(device=self.device)
        else:
            generator = torch.Generator()
        generator.manual_seed(int(seed))
        return generator
