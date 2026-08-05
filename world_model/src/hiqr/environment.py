"""Causal HiQR environments with auditable hierarchical random controls."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import torch

from .model import HierarchicalInteractionQueryRefineWorldModel


def _as_numpy(value: Any, dtype: np.dtype[Any]) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype)


def _clone(value: torch.Tensor | None) -> torch.Tensor | None:
    return None if value is None else value.detach().clone()


def _tensor(value: Any, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.as_tensor(value, device=device, dtype=dtype)


@dataclass(frozen=True)
class HiQRWorldRandomness:
    """Explicit START and response-level hierarchy innovations.

    ``scene_standard_normal`` and ``agent_standard_normal`` may hold the first
    ROLL innovation or a stream whose leading dimension is the one-based ROLL
    response index.  A seed derives the complete independent stream without
    storing all samples.
    """

    seed: int | None = None
    start_scene_standard_normal: np.ndarray | torch.Tensor | None = None
    start_agent_standard_normal: np.ndarray | torch.Tensor | None = None
    scene_standard_normal: np.ndarray | torch.Tensor | None = None
    agent_standard_normal: np.ndarray | torch.Tensor | None = None

    @classmethod
    def from_value(
        cls, value: HiQRWorldRandomness | Mapping[str, Any] | int
    ) -> HiQRWorldRandomness:
        if isinstance(value, cls):
            return value
        if isinstance(value, (int, np.integer)):
            return cls(seed=int(value))
        return cls(**dict(value))

    @staticmethod
    def _normal(
        shape: tuple[int, ...],
        seed: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        generator = torch.Generator(device=device).manual_seed(int(seed))
        return torch.randn(shape, generator=generator, device=device, dtype=dtype)

    @staticmethod
    def _explicit(
        value: np.ndarray | torch.Tensor | None,
        shape: tuple[int, ...],
        name: str,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if value is None:
            return None
        resolved = _tensor(value, device=device, dtype=dtype)
        if tuple(resolved.shape) != shape:
            raise ValueError(f"{name} must have shape {list(shape)}")
        return resolved

    @staticmethod
    def _response_stream(
        value: np.ndarray | torch.Tensor | None,
        shape: tuple[int, ...],
        name: str,
        response_index: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if value is None:
            return None
        resolved = _tensor(value, device=device, dtype=dtype)
        if tuple(resolved.shape) == shape:
            return resolved if response_index == 1 else None
        if resolved.ndim == len(shape) + 1 and tuple(resolved.shape[1:]) == shape:
            if response_index > resolved.shape[0]:
                raise ValueError(
                    f"{name} stream has {resolved.shape[0]} responses, "
                    f"but response {response_index} was requested"
                )
            return resolved[response_index - 1]
        raise ValueError(f"{name} must have shape {list(shape)} or [responses, *shape]")

    def resolve_start(
        self,
        *,
        scene_dim: int,
        agents: int,
        residual_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scene = self._explicit(
            self.start_scene_standard_normal,
            (scene_dim,),
            "start_scene_standard_normal",
            device,
            dtype,
        )
        residual = self._explicit(
            self.start_agent_standard_normal,
            (agents, residual_dim),
            "start_agent_standard_normal",
            device,
            dtype,
        )
        if scene is not None and residual is not None:
            return scene, residual
        if self.seed is None:
            raise ValueError(
                "a stochastic HiQR world requires a seed or explicit START noise"
            )
        return (
            (
                scene
                if scene is not None
                else self._normal((scene_dim,), int(self.seed), device, dtype)
            ),
            (
                residual
                if residual is not None
                else self._normal(
                    (agents, residual_dim), int(self.seed) + 97, device, dtype
                )
            ),
        )

    def resolve_response(
        self,
        *,
        response_index: int,
        scene_dim: int,
        agents: int,
        residual_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if int(response_index) < 1:
            raise ValueError("ROLL hierarchy innovations begin after START")
        scene = self._response_stream(
            self.scene_standard_normal,
            (scene_dim,),
            "scene_standard_normal",
            int(response_index),
            device,
            dtype,
        )
        residual = self._response_stream(
            self.agent_standard_normal,
            (agents, residual_dim),
            "agent_standard_normal",
            int(response_index),
            device,
            dtype,
        )
        if scene is not None and residual is not None:
            return scene, residual
        if self.seed is None:
            raise ValueError(
                "a stochastic HiQR response requires a seed or explicit response noise"
            )
        base = int(self.seed) + 1_000_003 * int(response_index)
        return (
            (
                scene
                if scene is not None
                else self._normal((scene_dim,), base, device, dtype)
            ),
            (
                residual
                if residual is not None
                else self._normal((agents, residual_dim), base + 97, device, dtype)
            ),
        )


@dataclass(frozen=True)
class HiQRFlowStartMetadata:
    """Static Flow/map metadata; primary risk remains audit-only for HiQR."""

    slot_valid: np.ndarray
    map_polylines: np.ndarray
    map_polyline_valid: np.ndarray
    primary_slot_index: int

    @classmethod
    def from_value(
        cls, value: HiQRFlowStartMetadata | Mapping[str, Any]
    ) -> HiQRFlowStartMetadata:
        return value if isinstance(value, cls) else cls(**dict(value))

    def validate(self) -> None:
        slots = _as_numpy(self.slot_valid, np.bool_)
        maps = _as_numpy(self.map_polylines, np.float32)
        map_valid = _as_numpy(self.map_polyline_valid, np.bool_)
        if slots.shape != (6,):
            raise ValueError("Flow START metadata.slot_valid must have shape [6]")
        if maps.ndim != 3 or maps.shape[-1] != 6 or map_valid.shape != maps.shape[:2]:
            raise ValueError(
                "Flow START map tensors must be [polylines, points, 6] "
                "with matching validity"
            )
        if (
            not 0 <= int(self.primary_slot_index) < 6
            or not slots[int(self.primary_slot_index)]
        ):
            raise ValueError("primary_slot_index must identify a valid background slot")

    def audit_dict(self) -> dict[str, Any]:
        return {
            "slot_valid": _as_numpy(self.slot_valid, np.bool_).copy(),
            "primary_slot_index": int(self.primary_slot_index),
        }


@dataclass(frozen=True)
class BatchedHiQRWorldSnapshot:
    """A response-boundary snapshot with no ADS future data."""

    metadata: tuple[HiQRFlowStartMetadata, ...]
    states: torch.Tensor
    valid: torch.Tensor
    history: torch.Tensor
    history_valid: torch.Tensor
    interaction_state: torch.Tensor
    previous_buffer: torch.Tensor | None
    previous_current: torch.Tensor | None
    map_inputs: tuple[torch.Tensor, torch.Tensor]
    active_plan: torch.Tensor | None
    plan_frame_index: int
    has_planned: bool
    deterministic: bool
    scene_randomness_controls: tuple[HiQRWorldRandomness, ...] | None
    agent_randomness_controls: tuple[HiQRWorldRandomness, ...] | None
    physics_step_index: int
    response_index: int
    traces: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class HiQRWorldSnapshot:
    """Single-world wrapper around one batched snapshot row."""

    batch_snapshot: BatchedHiQRWorldSnapshot

    @property
    def response_index(self) -> int:
        return self.batch_snapshot.response_index

    @property
    def physics_step_index(self) -> int:
        return self.batch_snapshot.physics_step_index

    @property
    def states(self) -> torch.Tensor:
        return self.batch_snapshot.states


class BatchedHiQRWorldModelEnvironment:
    """Vectorized independent HiQR worlds with causal ADS separation."""

    def __init__(
        self,
        model: HierarchicalInteractionQueryRefineWorldModel,
        *,
        device: str | torch.device = "cpu",
    ) -> None:
        self.model = model.to(device).eval()
        self.device = torch.device(device)
        self._metadata: tuple[HiQRFlowStartMetadata, ...] | None = None
        self._states: torch.Tensor | None = None
        self._valid: torch.Tensor | None = None
        self._history: torch.Tensor | None = None
        self._history_valid: torch.Tensor | None = None
        self._interaction_state: torch.Tensor | None = None
        self._previous_buffer: torch.Tensor | None = None
        self._previous_current: torch.Tensor | None = None
        self._map_inputs: tuple[torch.Tensor, torch.Tensor] | None = None
        self._active_plan: torch.Tensor | None = None
        self._scene_randomness_controls: tuple[HiQRWorldRandomness, ...] | None = None
        self._agent_randomness_controls: tuple[HiQRWorldRandomness, ...] | None = None
        self._traces: list[dict[str, Any]] = []
        self._deterministic = True
        self._has_planned = False
        self._plan_frame_index = 0
        self.physics_step_index = 0
        self.response_index = 0

    @property
    def batch_size(self) -> int:
        if self._states is None:
            raise RuntimeError("Call reset_from_flow_batch before accessing batch_size")
        return int(self._states.shape[0])

    def _require_state(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if (
            self._states is None
            or self._valid is None
            or self._interaction_state is None
        ):
            raise RuntimeError("Call reset_from_flow_batch before stepping")
        return self._states, self._valid, self._interaction_state

    @staticmethod
    def _default_metadata(
        slots: np.ndarray,
        maps: np.ndarray,
        map_valid: np.ndarray,
        primary: int,
    ) -> HiQRFlowStartMetadata:
        return HiQRFlowStartMetadata(
            slot_valid=slots,
            map_polylines=maps,
            map_polyline_valid=map_valid,
            primary_slot_index=primary,
        )

    @torch.no_grad()
    def reset_from_flow_batch(
        self,
        flow_features: np.ndarray | torch.Tensor,
        slot_valid: np.ndarray | torch.Tensor,
        map_polylines: np.ndarray | torch.Tensor,
        map_polyline_valid: np.ndarray | torch.Tensor,
        *,
        primary_slot_index: np.ndarray | torch.Tensor,
        flow_metadata: (
            Sequence[HiQRFlowStartMetadata | Mapping[str, Any]] | None
        ) = None,
        deterministic: bool = False,
        world_randomness: (
            Sequence[HiQRWorldRandomness | Mapping[str, Any] | int] | None
        ) = None,
    ) -> dict[str, Any]:
        features = _tensor(flow_features, device=self.device, dtype=torch.float32)
        slots = _tensor(slot_valid, device=self.device, dtype=torch.bool)
        maps = _tensor(map_polylines, device=self.device, dtype=torch.float32)
        map_valid = _tensor(map_polyline_valid, device=self.device, dtype=torch.bool)
        if (
            features.ndim != 2
            or features.shape[1] != 76
            or slots.shape != (features.shape[0], 6)
            or maps.ndim != 4
            or maps.shape[0] != features.shape[0]
            or maps.shape[-1] != 6
            or map_valid.shape != maps.shape[:3]
        ):
            raise ValueError(
                "batched Flow inputs must align as [batch, 76]/[batch, 6]/map tensors"
            )
        if not len(features):
            raise ValueError("HiQR batch must contain at least one Flow START")
        primary = _tensor(primary_slot_index, device=self.device, dtype=torch.long)
        if primary.shape != (features.shape[0],):
            raise ValueError("primary_slot_index must have shape [batch]")
        if flow_metadata is not None and len(flow_metadata) != features.shape[0]:
            raise ValueError(
                "flow_metadata must contain one metadata object per Flow row"
            )
        if not deterministic and (
            world_randomness is None or len(world_randomness) != features.shape[0]
        ):
            raise ValueError(
                "stochastic HiQR batch requires one random control per row"
            )

        slots_np = slots.detach().cpu().numpy()
        maps_np = maps.detach().cpu().numpy()
        map_valid_np = map_valid.detach().cpu().numpy()
        metadata = tuple(
            (
                HiQRFlowStartMetadata.from_value(flow_metadata[row])
                if flow_metadata is not None
                else self._default_metadata(
                    slots_np[row],
                    maps_np[row],
                    map_valid_np[row],
                    int(primary[row]),
                )
            )
            for row in range(features.shape[0])
        )
        for item in metadata:
            item.validate()
        if (
            torch.any(primary < 0)
            or torch.any(primary >= 6)
            or not torch.all(
                slots[torch.arange(features.shape[0], device=self.device), primary]
            )
        ):
            raise ValueError(
                "primary_slot_index must identify a valid slot in every row"
            )

        current, valid, raw_b0 = self.model.flow_condition_to_scene(features, slots)
        ego_mask = torch.zeros_like(valid)
        ego_mask[:, 0] = True
        initial = self.model.initialize_start(
            current,
            valid,
            ego_mask,
            maps,
            map_valid,
            raw_b0,
            valid[:, 1:],
        )
        self._metadata = metadata
        self._states, self._valid = current, valid
        self._history, self._history_valid = current[:, None], valid[:, None]
        self._interaction_state = initial
        self._previous_buffer = None
        self._previous_current = None
        self._map_inputs = (maps, map_valid)
        self._active_plan = None
        self._deterministic = bool(deterministic)
        self._has_planned = False
        self._plan_frame_index = 0
        self.physics_step_index = 0
        self.response_index = 0
        controls = (
            None
            if deterministic
            else tuple(
                HiQRWorldRandomness.from_value(item) for item in world_randomness
            )
        )
        self._scene_randomness_controls = controls
        self._agent_randomness_controls = controls
        self._traces = [
            {
                "flow_metadata": item.audit_dict(),
                "b0_lifecycle": "interaction_state_initialization_only",
                "world_randomness": (
                    {"mode": "deterministic_prior_mean"}
                    if deterministic
                    else {
                        "scene_seed": self._scene_randomness_controls[row].seed,
                        "agent_seed": self._agent_randomness_controls[row].seed,
                        "response_innovations": [],
                    }
                ),
                "planning_steps": 0,
                "physics_steps": 0,
            }
            for row, item in enumerate(metadata)
        ]
        return self.observe()

    def observe(self) -> dict[str, Any]:
        states, valid, _ = self._require_state()
        assert self._metadata is not None
        return {
            "agent_states": states.detach().clone(),
            "agent_valid": valid.detach().clone(),
            "response_index": self.response_index,
            "physics_step_index": self.physics_step_index,
            "flow_metadata": [item.audit_dict() for item in self._metadata],
            "world_randomness": [
                deepcopy(trace["world_randomness"]) for trace in self._traces
            ],
        }

    def snapshot(self) -> BatchedHiQRWorldSnapshot:
        states, valid, interaction = self._require_state()
        if self._active_plan is not None and self._plan_frame_index not in (
            0,
            self.model.cfg.execute_frames,
        ):
            raise RuntimeError(
                "HiQR snapshots are valid only at 5 Hz response boundaries"
            )
        assert self._metadata is not None
        assert self._history is not None and self._history_valid is not None
        assert self._map_inputs is not None
        return BatchedHiQRWorldSnapshot(
            metadata=deepcopy(self._metadata),
            states=states.detach().clone(),
            valid=valid.detach().clone(),
            history=self._history.detach().clone(),
            history_valid=self._history_valid.detach().clone(),
            interaction_state=interaction.detach().clone(),
            previous_buffer=_clone(self._previous_buffer),
            previous_current=_clone(self._previous_current),
            map_inputs=tuple(item.detach().clone() for item in self._map_inputs),
            active_plan=_clone(self._active_plan),
            plan_frame_index=self._plan_frame_index,
            has_planned=self._has_planned,
            deterministic=self._deterministic,
            scene_randomness_controls=deepcopy(self._scene_randomness_controls),
            agent_randomness_controls=deepcopy(self._agent_randomness_controls),
            physics_step_index=self.physics_step_index,
            response_index=self.response_index,
            traces=tuple(deepcopy(self._traces)),
        )

    def restore(self, snapshot: BatchedHiQRWorldSnapshot) -> dict[str, Any]:
        if not isinstance(snapshot, BatchedHiQRWorldSnapshot):
            raise TypeError("restore requires BatchedHiQRWorldSnapshot")
        if snapshot.states.device != self.device:
            raise ValueError(
                "snapshot device must match destination environment device"
            )
        if snapshot.states.shape[0] != len(snapshot.metadata):
            raise ValueError("snapshot metadata does not align with its tensor batch")
        self._metadata = deepcopy(snapshot.metadata)
        self._states, self._valid = (
            snapshot.states.detach().clone(),
            snapshot.valid.detach().clone(),
        )
        self._history = snapshot.history.detach().clone()
        self._history_valid = snapshot.history_valid.detach().clone()
        self._interaction_state = snapshot.interaction_state.detach().clone()
        self._previous_buffer = _clone(snapshot.previous_buffer)
        self._previous_current = _clone(snapshot.previous_current)
        self._map_inputs = tuple(item.detach().clone() for item in snapshot.map_inputs)
        self._active_plan = _clone(snapshot.active_plan)
        self._plan_frame_index = int(snapshot.plan_frame_index)
        self._has_planned = bool(snapshot.has_planned)
        self._deterministic = bool(snapshot.deterministic)
        self._scene_randomness_controls = deepcopy(snapshot.scene_randomness_controls)
        self._agent_randomness_controls = deepcopy(snapshot.agent_randomness_controls)
        self.physics_step_index = int(snapshot.physics_step_index)
        self.response_index = int(snapshot.response_index)
        self._traces = list(deepcopy(snapshot.traces))
        return self.observe()

    def _stack_start_noise(self) -> tuple[torch.Tensor, torch.Tensor]:
        states, _, _ = self._require_state()
        assert self._scene_randomness_controls is not None
        assert self._agent_randomness_controls is not None
        scene = [
            control.resolve_start(
                scene_dim=self.model.cfg.scene_latent_dim,
                agents=states.shape[1],
                residual_dim=self.model.cfg.agent_residual_dim,
                device=self.device,
                dtype=states.dtype,
            )[0]
            for control in self._scene_randomness_controls
        ]
        agent = [
            control.resolve_start(
                scene_dim=self.model.cfg.scene_latent_dim,
                agents=states.shape[1],
                residual_dim=self.model.cfg.agent_residual_dim,
                device=self.device,
                dtype=states.dtype,
            )[1]
            for control in self._agent_randomness_controls
        ]
        return torch.stack(scene), torch.stack(agent)

    def _stack_response_noise(self) -> tuple[torch.Tensor, torch.Tensor]:
        states, _, _ = self._require_state()
        assert self._scene_randomness_controls is not None
        assert self._agent_randomness_controls is not None
        scene = [
            control.resolve_response(
                response_index=self.response_index,
                scene_dim=self.model.cfg.scene_latent_dim,
                agents=states.shape[1],
                residual_dim=self.model.cfg.agent_residual_dim,
                device=self.device,
                dtype=states.dtype,
            )[0]
            for control in self._scene_randomness_controls
        ]
        agent = [
            control.resolve_response(
                response_index=self.response_index,
                scene_dim=self.model.cfg.scene_latent_dim,
                agents=states.shape[1],
                residual_dim=self.model.cfg.agent_residual_dim,
                device=self.device,
                dtype=states.dtype,
            )[1]
            for control in self._agent_randomness_controls
        ]
        return torch.stack(scene), torch.stack(agent)

    def resample_future_innovations(
        self,
        randomness: Sequence[HiQRWorldRandomness | Mapping[str, Any] | int],
        *,
        level: Literal["scene", "residual"] = "scene",
    ) -> dict[str, Any]:
        """Replace every unexecuted hierarchy stream from this response onward."""
        states, _, _ = self._require_state()
        if self._deterministic or self._active_plan is None:
            raise RuntimeError(
                "hierarchical resampling requires a stochastic world after START"
            )
        if (
            self._plan_frame_index != self.model.cfg.execute_frames
            or self.response_index < 1
        ):
            raise RuntimeError(
                "hierarchical resampling is valid only at completed response boundaries"
            )
        if len(randomness) != states.shape[0]:
            raise ValueError("one random control is required for each batch row")
        controls = tuple(HiQRWorldRandomness.from_value(item) for item in randomness)
        if level == "scene":
            self._scene_randomness_controls = controls
        self._agent_randomness_controls = controls
        scene, agent = self._stack_response_noise()
        for row, control in enumerate(controls):
            self._traces[row]["world_randomness"].setdefault(
                "branch_resampling", []
            ).append(
                {
                    "response_index": self.response_index,
                    "level": level,
                    "scene_seed": self._scene_randomness_controls[row].seed,
                    "agent_seed": self._agent_randomness_controls[row].seed,
                    "scene_standard_normal": scene[row].detach().cpu().tolist(),
                    "agent_standard_normal": agent[row].detach().cpu().tolist(),
                }
            )
        return self.observe()

    def branch_from_snapshot(
        self,
        snapshot: BatchedHiQRWorldSnapshot,
        randomness: Sequence[HiQRWorldRandomness | Mapping[str, Any] | int],
        *,
        level: Literal["scene", "residual"] = "scene",
    ) -> dict[str, Any]:
        self.restore(snapshot)
        return self.resample_future_innovations(randomness, level=level)

    @torch.no_grad()
    def _plan_if_needed(self) -> bool:
        states, valid, interaction = self._require_state()
        assert self._map_inputs is not None
        if (
            self._active_plan is not None
            and self._plan_frame_index < self.model.cfg.execute_frames
        ):
            return False
        start_mode = not self._has_planned
        scene_noise: torch.Tensor | None = None
        agent_noise: torch.Tensor | None = None
        if not self._deterministic:
            if start_mode:
                scene_noise, agent_noise = self._stack_start_noise()
            else:
                scene_noise, agent_noise = self._stack_response_noise()
        ego_mask = torch.zeros_like(valid)
        ego_mask[:, 0] = True
        out = self.model.plan_step(
            None if start_mode else self._history,
            None if start_mode else self._history_valid,
            states,
            valid,
            ego_mask,
            *self._map_inputs,
            interaction_state=interaction,
            previous_buffer=self._previous_buffer,
            previous_current=self._previous_current,
            deterministic=self._deterministic,
            scene_standard_normal=scene_noise,
            agent_standard_normal=agent_noise,
        )
        self._active_plan = out["background_future_actions"]
        self._plan_frame_index = 0
        self._previous_buffer = self._active_plan
        self._previous_current = states
        self._interaction_state = out["interaction_state"]
        self._has_planned = True
        for row, trace in enumerate(self._traces):
            trace["planning_steps"] += 1
            if scene_noise is not None and agent_noise is not None:
                trace["world_randomness"]["response_innovations"].append(
                    {
                        "response_index": self.response_index,
                        "kind": "start" if start_mode else "roll",
                        "scene_standard_normal": scene_noise[row]
                        .detach()
                        .cpu()
                        .tolist(),
                        "agent_standard_normal": agent_noise[row]
                        .detach()
                        .cpu()
                        .tolist(),
                    }
                )
        return True

    @torch.no_grad()
    def step(
        self,
        ads_actions: np.ndarray | torch.Tensor,
        ego_valid: np.ndarray | torch.Tensor | None = None,
    ) -> dict[str, Any]:
        states, valid, _ = self._require_state()
        actions = _tensor(ads_actions, device=self.device, dtype=states.dtype)
        if actions.shape != (states.shape[0], 2):
            raise ValueError("ads_actions must have shape [batch, 2]")
        ego_mask = (
            torch.ones(states.shape[0], dtype=torch.bool, device=self.device)
            if ego_valid is None
            else _tensor(ego_valid, device=self.device, dtype=torch.bool)
        )
        if ego_mask.shape != (states.shape[0],):
            raise ValueError("ego_valid must have shape [batch]")
        self._states, self._valid = states.clone(), valid.clone()
        self._valid[:, 0] = ego_mask
        self._states[~ego_mask, 0] = 0.0
        planner_updated = self._plan_if_needed()
        assert self._active_plan is not None
        frame = self._plan_frame_index
        physical = self._states.new_zeros((self.batch_size, self._states.shape[1], 2))
        physical[:, 0] = actions
        physical[:, 1:] = self._active_plan[:, frame]
        self._states = self.model.dynamics.step(
            self._states,
            physical,
            self._valid,
            self.model.cfg.simulation_dt_s,
        )
        assert self._history is not None and self._history_valid is not None
        self._history = torch.cat((self._history, self._states[:, None]), dim=1)[
            :, -self.model.cfg.plan_frames :
        ]
        self._history_valid = torch.cat(
            (self._history_valid, self._valid[:, None]), dim=1
        )[:, -self.model.cfg.plan_frames :]
        self._plan_frame_index += 1
        self.physics_step_index += 1
        if self._plan_frame_index == self.model.cfg.execute_frames:
            self.response_index += 1
        for trace in self._traces:
            trace["physics_steps"] = self.physics_step_index
        observation = self.observe()
        observation.update(
            {
                "background_state": self._states[:, 1:].detach().clone(),
                "applied_ego_action": actions.detach().clone(),
                "applied_background_actions": physical[:, 1:].detach().clone(),
                "background_future_actions": self._active_plan.detach().clone(),
                "planner_updated": planner_updated,
                "executed_plan_frame": frame,
                "trace": deepcopy(self._traces),
            }
        )
        return observation

    @torch.no_grad()
    def advance_response(
        self,
        ads_actions: np.ndarray | torch.Tensor,
        ego_valid: np.ndarray | torch.Tensor | None = None,
    ) -> dict[str, Any]:
        states, _, _ = self._require_state()
        actions = _tensor(ads_actions, device=self.device, dtype=states.dtype)
        if (
            actions.ndim != 3
            or actions.shape[0] != states.shape[0]
            or actions.shape[2] != 2
            or not 1 <= actions.shape[1] <= self.model.cfg.execute_frames
        ):
            raise ValueError("ads_actions must have shape [batch, ticks, 2]")
        if (
            self._active_plan is not None
            and 0 < self._plan_frame_index < self.model.cfg.execute_frames
        ):
            raise RuntimeError("advance_response must start on a response boundary")
        if ego_valid is None:
            valid = torch.ones(actions.shape[:2], dtype=torch.bool, device=self.device)
        else:
            valid = _tensor(ego_valid, device=self.device, dtype=torch.bool)
            if valid.ndim == 0:
                valid = valid.expand(actions.shape[:2])
        if valid.shape != actions.shape[:2]:
            raise ValueError("ego_valid must be bool or have shape [batch, ticks]")
        ticks = [
            self.step(actions[:, tick], valid[:, tick])
            for tick in range(actions.shape[1])
        ]
        output = ticks[-1]
        state_frames = torch.stack([item["agent_states"] for item in ticks], dim=1)
        output.update(
            {
                "agent_state_frames": state_frames,
                "background_states": state_frames[:, :, 1:],
                "applied_ego_actions": torch.stack(
                    [item["applied_ego_action"] for item in ticks], dim=1
                ),
                "applied_background_action_frames": torch.stack(
                    [item["applied_background_actions"] for item in ticks], dim=1
                ),
                "planning_updates": torch.as_tensor(
                    [item["planner_updated"] for item in ticks], device=self.device
                ),
            }
        )
        return output


class HiQRWorldModelEnvironment:
    """Single-world facade over the vectorized causal environment."""

    def __init__(
        self,
        model: HierarchicalInteractionQueryRefineWorldModel,
        *,
        device: str | torch.device = "cpu",
    ) -> None:
        self.model = model.to(device).eval()
        self.device = torch.device(device)
        self._batch = BatchedHiQRWorldModelEnvironment(self.model, device=self.device)

    @property
    def response_index(self) -> int:
        return self._batch.response_index

    @property
    def physics_step_index(self) -> int:
        return self._batch.physics_step_index

    @property
    def trace(self) -> dict[str, Any]:
        return {} if not self._batch._traces else self._batch._traces[0]

    @torch.no_grad()
    def reset_from_flow(
        self,
        C0: np.ndarray | torch.Tensor,
        B0: np.ndarray | torch.Tensor,
        metadata: HiQRFlowStartMetadata | Mapping[str, Any],
        *,
        deterministic: bool = True,
        world_randomness: HiQRWorldRandomness | Mapping[str, Any] | int | None = None,
    ) -> dict[str, Any]:
        item = HiQRFlowStartMetadata.from_value(metadata)
        item.validate()
        c0 = _as_numpy(C0, np.float32).reshape(-1)
        b0 = _as_numpy(B0, np.float32)
        if c0.shape != (40,) or b0.shape != (6, 6):
            raise ValueError("reset_from_flow requires C0[40] and B0[6, 6]")
        self._batch.reset_from_flow_batch(
            np.concatenate((c0, b0.reshape(-1)))[None],
            _as_numpy(item.slot_valid, np.bool_)[None],
            _as_numpy(item.map_polylines, np.float32)[None],
            _as_numpy(item.map_polyline_valid, np.bool_)[None],
            primary_slot_index=np.asarray([item.primary_slot_index]),
            flow_metadata=[item],
            deterministic=deterministic,
            world_randomness=None if deterministic else [world_randomness],
        )
        return self.observe()

    def observe(self) -> dict[str, Any]:
        observation = self._batch.observe()
        return {
            "agent_states": observation["agent_states"][0].cpu().numpy().copy(),
            "agent_valid": observation["agent_valid"][0]
            .cpu()
            .numpy()
            .astype(bool, copy=True),
            "response_index": observation["response_index"],
            "physics_step_index": observation["physics_step_index"],
            "flow_metadata": observation["flow_metadata"][0],
            "world_randomness": observation["world_randomness"][0],
        }

    def snapshot(self) -> HiQRWorldSnapshot:
        return HiQRWorldSnapshot(self._batch.snapshot())

    def restore(self, snapshot: HiQRWorldSnapshot) -> dict[str, Any]:
        if not isinstance(snapshot, HiQRWorldSnapshot):
            raise TypeError("restore requires HiQRWorldSnapshot")
        self._batch.restore(snapshot.batch_snapshot)
        return self.observe()

    def resample_future_innovation(
        self,
        randomness: HiQRWorldRandomness | Mapping[str, Any] | int,
        *,
        level: Literal["scene", "residual"] = "scene",
    ) -> dict[str, Any]:
        self._batch.resample_future_innovations([randomness], level=level)
        return self.observe()

    def branch_from_snapshot(
        self,
        snapshot: HiQRWorldSnapshot,
        randomness: HiQRWorldRandomness | Mapping[str, Any] | int,
        *,
        level: Literal["scene", "residual"] = "scene",
    ) -> dict[str, Any]:
        self.restore(snapshot)
        return self.resample_future_innovation(randomness, level=level)

    @torch.no_grad()
    def step(
        self,
        ads_action: np.ndarray | torch.Tensor,
        ego_valid: bool = True,
    ) -> dict[str, Any]:
        output = self._batch.step(
            _tensor(ads_action, device=self.device, dtype=torch.float32)[None],
            torch.as_tensor([ego_valid], device=self.device),
        )
        observation = self.observe()
        observation.update(
            {
                "background_state": output["background_state"][0].cpu().numpy(),
                "applied_ego_action": output["applied_ego_action"][0].cpu().numpy(),
                "applied_background_actions": output["applied_background_actions"][0]
                .cpu()
                .numpy(),
                "background_future_actions": output["background_future_actions"][0]
                .cpu()
                .numpy(),
                "planner_updated": output["planner_updated"],
                "executed_plan_frame": output["executed_plan_frame"],
                "trace": output["trace"][0],
            }
        )
        return observation

    @torch.no_grad()
    def advance_response(
        self,
        ads_actions: np.ndarray | torch.Tensor,
        ego_valid: bool | np.ndarray | torch.Tensor = True,
    ) -> dict[str, Any]:
        actions = _tensor(ads_actions, device=self.device, dtype=torch.float32)
        if actions.ndim != 2 or actions.shape[1] != 2:
            raise ValueError("ads_actions must have shape [ticks, 2]")
        valid = _tensor(ego_valid, device=self.device, dtype=torch.bool)
        if valid.ndim == 0:
            valid = valid.expand(actions.shape[0])
        if valid.shape != (actions.shape[0],):
            raise ValueError("ego_valid must be bool or have shape [ticks]")
        output = self._batch.advance_response(actions[None], valid[None])
        observation = self.observe()
        observation.update(
            {
                "agent_state_frames": output["agent_state_frames"][0].cpu().numpy(),
                "background_states": output["background_states"][0].cpu().numpy(),
                "applied_ego_actions": output["applied_ego_actions"][0].cpu().numpy(),
                "applied_background_action_frames": output[
                    "applied_background_action_frames"
                ][0]
                .cpu()
                .numpy(),
                "planning_updates": output["planning_updates"]
                .cpu()
                .numpy()
                .astype(bool),
                "trace": output["trace"][0],
            }
        )
        return observation
