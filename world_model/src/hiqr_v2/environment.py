"""Causal, replayable online environments for the isolated HiQR-v2 schema.

The environment uses the shared Flow START audit contract while its learned
recurrent state is the observation-only ``FilterState``.  A replay snapshot
also captures the active one-second scene mode and action plan because exact
branch replay requires transient context.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import torch

from world_model.src.hiqr.environment import HiQRFlowStartMetadata, HiQRWorldRandomness

from .filter import FilterState
from .model import HiQRV2WorldModel


def _as_numpy(value: Any, dtype: np.dtype[Any]) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype)


def _tensor(value: Any, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.as_tensor(value, device=device, dtype=dtype)


def _clone(value: torch.Tensor | None) -> torch.Tensor | None:
    return None if value is None else value.detach().clone()


def _clone_filter(value: FilterState) -> FilterState:
    return FilterState(
        value.global_hidden.detach().clone(), value.agent_hidden.detach().clone()
    )


@dataclass(frozen=True)
class BatchedHiQRV2WorldSnapshot:
    """A 5 Hz boundary snapshot that contains no ADS future actions."""

    metadata: tuple[HiQRFlowStartMetadata, ...]
    states: torch.Tensor
    valid: torch.Tensor
    history: torch.Tensor
    history_valid: torch.Tensor
    filter_state: FilterState
    previous_current: torch.Tensor | None
    slow_scene: torch.Tensor | None
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
class HiQRV2WorldSnapshot:
    """Single-world wrapper around a batched V2 snapshot."""

    batch_snapshot: BatchedHiQRV2WorldSnapshot

    @property
    def response_index(self) -> int:
        return self.batch_snapshot.response_index

    @property
    def physics_step_index(self) -> int:
        return self.batch_snapshot.physics_step_index


class BatchedHiQRV2WorldModelEnvironment:
    """Vectorized causal worlds with explicit, replayable V2 innovations."""

    def __init__(
        self, model: HiQRV2WorldModel, *, device: str | torch.device = "cpu"
    ) -> None:
        self.model = model.to(device).eval()
        self.device = torch.device(device)
        self._metadata: tuple[HiQRFlowStartMetadata, ...] | None = None
        self._states: torch.Tensor | None = None
        self._valid: torch.Tensor | None = None
        self._history: torch.Tensor | None = None
        self._history_valid: torch.Tensor | None = None
        self._filter_state: FilterState | None = None
        self._previous_current: torch.Tensor | None = None
        self._slow_scene: torch.Tensor | None = None
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

    def _require_state(self) -> tuple[torch.Tensor, torch.Tensor, FilterState]:
        if self._states is None or self._valid is None or self._filter_state is None:
            raise RuntimeError("Call reset_from_flow_batch before stepping")
        return self._states, self._valid, self._filter_state

    @torch.no_grad()
    def reset_from_flow_batch(
        self,
        flow_features: np.ndarray | torch.Tensor,
        slot_valid: np.ndarray | torch.Tensor,
        map_polylines: np.ndarray | torch.Tensor,
        map_polyline_valid: np.ndarray | torch.Tensor,
        *,
        primary_slot_index: np.ndarray | torch.Tensor,
        flow_metadata: Sequence[HiQRFlowStartMetadata | Mapping[str, Any]],
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
            or not len(features)
        ):
            raise ValueError(
                "Flow inputs must align as [batch,76]/[batch,6]/map tensors"
            )
        primary = _tensor(primary_slot_index, device=self.device, dtype=torch.long)
        if (
            primary.shape != (features.shape[0],)
            or len(flow_metadata) != features.shape[0]
        ):
            raise ValueError(
                "Flow primary slots and metadata must have one row per world"
            )
        if not deterministic and (
            world_randomness is None or len(world_randomness) != features.shape[0]
        ):
            raise ValueError(
                "stochastic HiQR-v2 worlds require one random control per row"
            )
        metadata = tuple(
            HiQRFlowStartMetadata.from_value(item) for item in flow_metadata
        )
        slots_np, maps_np, map_valid_np = (
            slots.cpu().numpy(),
            maps.cpu().numpy(),
            map_valid.cpu().numpy(),
        )
        for row, item in enumerate(metadata):
            item.validate()
            if (
                not np.array_equal(_as_numpy(item.slot_valid, np.bool_), slots_np[row])
                or not np.array_equal(
                    _as_numpy(item.map_polylines, np.float32), maps_np[row]
                )
                or not np.array_equal(
                    _as_numpy(item.map_polyline_valid, np.bool_), map_valid_np[row]
                )
                or int(item.primary_slot_index) != int(primary[row])
            ):
                raise ValueError("Flow tensors do not match their audit metadata")
            if (
                self.model.flow_schema_sha256 is not None
                and item.flow_schema_sha256 != self.model.flow_schema_sha256
            ):
                raise ValueError("Flow metadata schema hash differs from HiQR-v2")
        if (
            torch.any(primary < 0)
            or torch.any(primary >= 6)
            or not torch.all(
                slots[torch.arange(len(features), device=self.device), primary]
            )
        ):
            raise ValueError("primary_slot_index must identify a valid background slot")
        current, valid, raw_b0 = self.model.flow_condition_to_scene(features, slots)
        ego_mask = torch.zeros_like(valid)
        ego_mask[:, 0] = True
        self._metadata = metadata
        self._states, self._valid = current, valid
        self._history, self._history_valid = current[:, None], valid[:, None]
        self._filter_state = self.model.initialize_start(
            current, valid, ego_mask, maps, map_valid, raw_b0, valid[:, 1:]
        )
        self._previous_current = self._slow_scene = None
        self._map_inputs, self._active_plan = (maps, map_valid), None
        controls = (
            None
            if deterministic
            else tuple(
                HiQRWorldRandomness.from_value(item) for item in world_randomness or ()
            )
        )
        self._scene_randomness_controls = controls
        self._agent_randomness_controls = controls
        self._deterministic, self._has_planned, self._plan_frame_index = (
            bool(deterministic),
            False,
            0,
        )
        self.physics_step_index, self.response_index = 0, 0
        self._traces = [
            {
                "flow_metadata": item.audit_dict(),
                "persistent_state": "observation_only_filter",
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
                deepcopy(row["world_randomness"]) for row in self._traces
            ],
        }

    def failure_report(self) -> dict[str, Any]:
        result = self.observe()
        return {
            key: result[key]
            for key in (
                "flow_metadata",
                "world_randomness",
                "response_index",
                "physics_step_index",
            )
        }

    def snapshot(self) -> BatchedHiQRV2WorldSnapshot:
        states, valid, filter_state = self._require_state()
        if self._active_plan is not None and self._plan_frame_index not in (
            0,
            int(self.model.cfg.execute_frames),
        ):
            raise RuntimeError(
                "HiQR-v2 snapshots are valid only at 5 Hz response boundaries"
            )
        assert (
            self._metadata is not None
            and self._history is not None
            and self._history_valid is not None
            and self._map_inputs is not None
        )
        return BatchedHiQRV2WorldSnapshot(
            metadata=deepcopy(self._metadata),
            states=states.detach().clone(),
            valid=valid.detach().clone(),
            history=self._history.detach().clone(),
            history_valid=self._history_valid.detach().clone(),
            filter_state=_clone_filter(filter_state),
            previous_current=_clone(self._previous_current),
            slow_scene=_clone(self._slow_scene),
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

    def restore(self, snapshot: BatchedHiQRV2WorldSnapshot) -> dict[str, Any]:
        if not isinstance(snapshot, BatchedHiQRV2WorldSnapshot):
            raise TypeError("restore requires BatchedHiQRV2WorldSnapshot")
        if snapshot.states.device != self.device or snapshot.states.shape[0] != len(
            snapshot.metadata
        ):
            raise ValueError("snapshot is not compatible with this HiQR-v2 environment")
        self._metadata = deepcopy(snapshot.metadata)
        self._states, self._valid = (
            snapshot.states.detach().clone(),
            snapshot.valid.detach().clone(),
        )
        self._history, self._history_valid, self._filter_state = (
            snapshot.history.detach().clone(),
            snapshot.history_valid.detach().clone(),
            _clone_filter(snapshot.filter_state),
        )
        self._previous_current = _clone(snapshot.previous_current)
        self._slow_scene = _clone(snapshot.slow_scene)
        self._map_inputs, self._active_plan = tuple(
            item.detach().clone() for item in snapshot.map_inputs
        ), _clone(snapshot.active_plan)
        self._plan_frame_index, self._has_planned, self._deterministic = (
            int(snapshot.plan_frame_index),
            bool(snapshot.has_planned),
            bool(snapshot.deterministic),
        )
        self._scene_randomness_controls, self._agent_randomness_controls = deepcopy(
            snapshot.scene_randomness_controls
        ), deepcopy(snapshot.agent_randomness_controls)
        self.physics_step_index, self.response_index = int(
            snapshot.physics_step_index
        ), int(snapshot.response_index)
        self._traces = list(deepcopy(snapshot.traces))
        return self.observe()

    def _stack_noise(self, *, start: bool) -> tuple[torch.Tensor, torch.Tensor]:
        states, _, _ = self._require_state()
        assert (
            self._scene_randomness_controls is not None
            and self._agent_randomness_controls is not None
        )
        method = "resolve_start" if start else "resolve_response"
        scene_rows = [
            getattr(control, method)(
                **({} if start else {"response_index": self.response_index}),
                scene_dim=self.model.cfg.scene_latent_dim,
                agents=states.shape[1],
                residual_dim=self.model.cfg.agent_residual_dim,
                device=self.device,
                dtype=states.dtype,
            )
            for control in self._scene_randomness_controls
        ]
        agent_rows = [
            getattr(control, method)(
                **({} if start else {"response_index": self.response_index}),
                scene_dim=self.model.cfg.scene_latent_dim,
                agents=states.shape[1],
                residual_dim=self.model.cfg.agent_residual_dim,
                device=self.device,
                dtype=states.dtype,
            )
            for control in self._agent_randomness_controls
        ]
        return torch.stack([row[0] for row in scene_rows]), torch.stack(
            [row[1] for row in agent_rows]
        )

    def resample_future_innovations(
        self,
        randomness: Sequence[HiQRWorldRandomness | Mapping[str, Any] | int],
        *,
        level: Literal["scene", "residual"] = "scene",
    ) -> dict[str, Any]:
        """Replace the unexecuted stream at a response boundary for AMS branches."""
        states, _, _ = self._require_state()
        if (
            self._deterministic
            or self._active_plan is None
            or self._plan_frame_index != int(self.model.cfg.execute_frames)
            or self.response_index < 1
        ):
            raise RuntimeError(
                "resampling requires a completed stochastic HiQR-v2 response"
            )
        if len(randomness) != states.shape[0]:
            raise ValueError("one random control is required per batch row")
        controls = tuple(HiQRWorldRandomness.from_value(item) for item in randomness)
        if level == "scene":
            self._scene_randomness_controls = controls
            # A scene-level branch starts a new slow mode at the next response
            # boundary.  Residual-only branches retain the current mode.
            self._slow_scene = None
        elif level != "residual":
            raise ValueError("level must be scene or residual")
        self._agent_randomness_controls = controls
        for row, (trace, control) in enumerate(zip(self._traces, controls)):
            trace["world_randomness"].setdefault("branch_resampling", []).append(
                {
                    "response_index": self.response_index,
                    "level": level,
                    "scene_seed": self._scene_randomness_controls[row].seed,
                    "agent_seed": control.seed,
                }
            )
        return self.observe()

    def branch_from_snapshot(
        self,
        snapshot: BatchedHiQRV2WorldSnapshot,
        randomness: Sequence[HiQRWorldRandomness | Mapping[str, Any] | int],
        *,
        level: Literal["scene", "residual"] = "scene",
    ) -> dict[str, Any]:
        self.restore(snapshot)
        return self.resample_future_innovations(randomness, level=level)

    @torch.no_grad()
    def _plan_if_needed(self) -> bool:
        states, valid, filter_state = self._require_state()
        assert self._map_inputs is not None
        if self._active_plan is not None and self._plan_frame_index < int(
            self.model.cfg.execute_frames
        ):
            return False
        start = not self._has_planned
        scene_noise = agent_noise = None
        if not self._deterministic:
            scene_noise, agent_noise = self._stack_noise(start=start)
        ego_mask = torch.zeros_like(valid)
        ego_mask[:, 0] = True
        out = self.model.plan_step(
            None if start else self._history,
            None if start else self._history_valid,
            states,
            valid,
            ego_mask,
            *self._map_inputs,
            filter_state=filter_state,
            previous_current=self._previous_current,
            slow_scene=self._slow_scene,
            response_index=self.response_index,
            deterministic=self._deterministic,
            scene_standard_normal=scene_noise,
            agent_standard_normal=agent_noise,
        )
        next_plan = out["background_future_actions"]
        self._filter_state, self._slow_scene = out["filter_state"], out["slow_scene"]
        self._active_plan, self._plan_frame_index = next_plan, 0
        self._previous_current = states.detach().clone()
        self._has_planned = True
        for row, trace in enumerate(self._traces):
            trace["planning_steps"] += 1
            if scene_noise is not None and agent_noise is not None:
                trace["world_randomness"]["response_innovations"].append(
                    {
                        "response_index": self.response_index,
                        "kind": "start" if start else "roll",
                        "scene_refreshed": bool(out["scene_refreshed"]),
                        "scene_standard_normal": scene_noise[row].cpu().tolist(),
                        "agent_standard_normal": agent_noise[row].cpu().tolist(),
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
            raise ValueError("ads_actions must have shape [batch,2]")
        ego = (
            torch.ones(states.shape[0], dtype=torch.bool, device=self.device)
            if ego_valid is None
            else _tensor(ego_valid, device=self.device, dtype=torch.bool)
        )
        if ego.shape != (states.shape[0],):
            raise ValueError("ego_valid must have shape [batch]")
        self._states, self._valid = states.clone(), valid.clone()
        self._valid[:, 0] = ego
        self._states[~ego, 0] = 0.0
        planner_updated = self._plan_if_needed()
        assert self._active_plan is not None
        frame = self._plan_frame_index
        physical = self._states.new_zeros((self.batch_size, self._states.shape[1], 2))
        physical[:, 0], physical[:, 1:] = actions, self._active_plan[:, frame]
        self._states = self.model.dynamics.step(
            self._states, physical, self._valid, self.model.cfg.simulation_dt_s
        )
        assert self._history is not None and self._history_valid is not None
        self._history = torch.cat((self._history, self._states[:, None]), dim=1)[
            :, -int(self.model.cfg.plan_frames) :
        ]
        self._history_valid = torch.cat(
            (self._history_valid, self._valid[:, None]), dim=1
        )[:, -int(self.model.cfg.plan_frames) :]
        self._plan_frame_index += 1
        self.physics_step_index += 1
        if self._plan_frame_index == int(self.model.cfg.execute_frames):
            self.response_index += 1
        for trace in self._traces:
            trace["physics_steps"] = self.physics_step_index
        result = self.observe()
        result.update(
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
        return result

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
            or not 1 <= actions.shape[1] <= int(self.model.cfg.execute_frames)
        ):
            raise ValueError("ads_actions must have shape [batch,ticks,2]")
        if self._active_plan is not None and 0 < self._plan_frame_index < int(
            self.model.cfg.execute_frames
        ):
            raise RuntimeError("advance_response must start on a response boundary")
        valid = (
            torch.ones(actions.shape[:2], dtype=torch.bool, device=self.device)
            if ego_valid is None
            else _tensor(ego_valid, device=self.device, dtype=torch.bool)
        )
        if valid.ndim == 0:
            valid = valid.expand(actions.shape[:2])
        if valid.shape != actions.shape[:2]:
            raise ValueError("ego_valid must be bool or [batch,ticks]")
        rows = [
            self.step(actions[:, tick], valid[:, tick])
            for tick in range(actions.shape[1])
        ]
        output = rows[-1]
        output.update(
            {
                "agent_state_frames": torch.stack(
                    [row["agent_states"] for row in rows], dim=1
                ),
                "background_states": torch.stack(
                    [row["background_state"] for row in rows], dim=1
                ),
                "applied_ego_actions": torch.stack(
                    [row["applied_ego_action"] for row in rows], dim=1
                ),
                "applied_background_action_frames": torch.stack(
                    [row["applied_background_actions"] for row in rows], dim=1
                ),
                "planning_updates": torch.as_tensor(
                    [row["planner_updated"] for row in rows], device=self.device
                ),
            }
        )
        return output


class HiQRV2WorldModelEnvironment:
    """Single-world facade for :class:`BatchedHiQRV2WorldModelEnvironment`."""

    def __init__(
        self, model: HiQRV2WorldModel, *, device: str | torch.device = "cpu"
    ) -> None:
        self.model = model.to(device).eval()
        self.device = torch.device(device)
        self._batch = BatchedHiQRV2WorldModelEnvironment(self.model, device=self.device)

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
        c0, b0 = _as_numpy(C0, np.float32).reshape(-1), _as_numpy(B0, np.float32)
        if c0.shape != (40,) or b0.shape != (6, 6):
            raise ValueError("reset_from_flow requires C0[40] and B0[6,6]")
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
        row = self._batch.observe()
        return {
            "agent_states": row["agent_states"][0].cpu().numpy().copy(),
            "agent_valid": row["agent_valid"][0].cpu().numpy().astype(bool, copy=True),
            "response_index": row["response_index"],
            "physics_step_index": row["physics_step_index"],
            "flow_metadata": row["flow_metadata"][0],
            "world_randomness": row["world_randomness"][0],
        }

    def failure_report(self) -> dict[str, Any]:
        row = self._batch.failure_report()
        return {
            "flow_metadata": row["flow_metadata"][0],
            "world_randomness": row["world_randomness"][0],
            "response_index": row["response_index"],
            "physics_step_index": row["physics_step_index"],
        }

    def snapshot(self) -> HiQRV2WorldSnapshot:
        return HiQRV2WorldSnapshot(self._batch.snapshot())

    def restore(self, snapshot: HiQRV2WorldSnapshot) -> dict[str, Any]:
        if not isinstance(snapshot, HiQRV2WorldSnapshot):
            raise TypeError("restore requires HiQRV2WorldSnapshot")
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
        snapshot: HiQRV2WorldSnapshot,
        randomness: HiQRWorldRandomness | Mapping[str, Any] | int,
        *,
        level: Literal["scene", "residual"] = "scene",
    ) -> dict[str, Any]:
        self.restore(snapshot)
        return self.resample_future_innovation(randomness, level=level)

    @torch.no_grad()
    def step(
        self, ads_action: np.ndarray | torch.Tensor, ego_valid: bool = True
    ) -> dict[str, Any]:
        row = self._batch.step(
            _tensor(ads_action, device=self.device, dtype=torch.float32)[None],
            torch.as_tensor([ego_valid], device=self.device),
        )
        out = self.observe()
        out.update(
            {
                key: (
                    row[key][0].cpu().numpy()
                    if isinstance(row[key], torch.Tensor)
                    else row[key]
                )
                for key in (
                    "background_state",
                    "applied_ego_action",
                    "applied_background_actions",
                    "background_future_actions",
                )
            }
        )
        out.update(
            {
                "planner_updated": row["planner_updated"],
                "executed_plan_frame": row["executed_plan_frame"],
                "trace": row["trace"][0],
            }
        )
        return out

    @torch.no_grad()
    def advance_response(
        self,
        ads_actions: np.ndarray | torch.Tensor,
        ego_valid: bool | np.ndarray | torch.Tensor = True,
    ) -> dict[str, Any]:
        actions = _tensor(ads_actions, device=self.device, dtype=torch.float32)
        if actions.ndim != 2 or actions.shape[1] != 2:
            raise ValueError("ads_actions must have shape [ticks,2]")
        valid = _tensor(ego_valid, device=self.device, dtype=torch.bool)
        if valid.ndim == 0:
            valid = valid.expand(actions.shape[0])
        row = self._batch.advance_response(actions[None], valid[None])
        out = self.observe()
        out.update(
            {
                "agent_state_frames": row["agent_state_frames"][0].cpu().numpy(),
                "background_states": row["background_states"][0].cpu().numpy(),
                "applied_ego_actions": row["applied_ego_actions"][0].cpu().numpy(),
                "applied_background_action_frames": row[
                    "applied_background_action_frames"
                ][0]
                .cpu()
                .numpy(),
                "planning_updates": row["planning_updates"].cpu().numpy().astype(bool),
                "trace": row["trace"][0],
            }
        )
        return out
