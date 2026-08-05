"""Online QR-WM environment with ADS-driven ego physics."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .model import QueryRefineWorldModel


def _as_numpy(value: Any, dtype) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype)


@dataclass(frozen=True)
class WorldRandomness:
    """Explicit START latent randomness for one independent QR world.

    QR-WM samples only its behavior latent at START.  All later response
    updates are deterministic conditional on that latent, the current world
    state, and the ego state actually observed at that response.  Supplying a
    seed or a standard-normal tensor therefore fully controls one world trace.
    """

    seed: int | None = None
    behavior_standard_normal: np.ndarray | torch.Tensor | None = None

    @classmethod
    def from_value(cls, value: "WorldRandomness | Mapping[str, Any] | int") -> "WorldRandomness":
        if isinstance(value, cls):
            return value
        if isinstance(value, (int, np.integer)):
            return cls(seed=int(value))
        return cls(**dict(value))

    def resolve(
        self, *, agents: int, latent_dim: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Return the realised standard-normal START perturbation [agents, latent]."""
        if self.behavior_standard_normal is not None:
            value = torch.as_tensor(
                _as_numpy(self.behavior_standard_normal, np.float32), device=device, dtype=dtype
            )
            if value.shape != (agents, latent_dim):
                raise ValueError(
                    "behavior_standard_normal must have shape "
                    "[agents, behavior_latent_dim] for one QR world"
                )
            return value
        if self.seed is None:
            raise ValueError(
                "a stochastic QR world requires WorldRandomness(seed=...) or "
                "behavior_standard_normal; use deterministic=True for the prior mean"
            )
        generator = torch.Generator(device=device).manual_seed(int(self.seed))
        return torch.randn((agents, latent_dim), generator=generator, device=device, dtype=dtype)

    def audit_dict(self, realised_noise: torch.Tensor) -> dict[str, Any]:
        """Preserve both the user control and realised noise for exact replay."""
        return {
            "seed": None if self.seed is None else int(self.seed),
            "behavior_standard_normal": realised_noise.detach().cpu().numpy().tolist(),
        }


def _single_world_noise(
    randomness: WorldRandomness | Mapping[str, Any] | int | None,
    *,
    deterministic: bool,
    agents: int,
    latent_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor | None, dict[str, Any]]:
    if deterministic:
        if randomness is not None:
            raise ValueError("deterministic QR world must not receive stochastic WorldRandomness")
        return None, {"mode": "deterministic_prior_mean"}
    if randomness is None:
        raise ValueError(
            "stochastic QR world requires an explicit WorldRandomness; "
            "implicit global RNG is not permitted by the environment API"
        )
    control = WorldRandomness.from_value(randomness)
    noise = control.resolve(agents=agents, latent_dim=latent_dim, device=device, dtype=dtype)
    return noise, control.audit_dict(noise)


def _batched_world_noise(
    randomness: Sequence[WorldRandomness | Mapping[str, Any] | int] | None,
    *,
    batch_size: int,
    deterministic: bool,
    agents: int,
    latent_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor | None, list[dict[str, Any]]]:
    if deterministic:
        if randomness is not None:
            raise ValueError("deterministic QR batch must not receive stochastic WorldRandomness")
        return None, [{"mode": "deterministic_prior_mean"} for _ in range(batch_size)]
    if randomness is None or len(randomness) != batch_size:
        raise ValueError(
            "stochastic QR batch requires one explicit WorldRandomness per independent world row"
        )
    controls = [WorldRandomness.from_value(value) for value in randomness]
    noise = torch.stack(
        [
            control.resolve(agents=agents, latent_dim=latent_dim, device=device, dtype=dtype)
            for control in controls
        ],
        dim=0,
    )
    return noise, [control.audit_dict(row) for control, row in zip(controls, noise)]


@dataclass(frozen=True)
class FlowStartMetadata:
    """Static map inputs plus Flow probability/event audit fields for one START."""

    slot_valid: np.ndarray
    map_polylines: np.ndarray
    map_polyline_valid: np.ndarray
    lane_graph_edges: np.ndarray
    primary_slot_index: int
    event_structure: Any
    mask_pattern: int
    event_structure_id: int
    event_structure_log_prob: float
    conditional_log_prob: float
    log_prob: float

    @classmethod
    def from_value(cls, value: "FlowStartMetadata | Mapping[str, Any]") -> "FlowStartMetadata":
        if isinstance(value, cls):
            return value
        return cls(**dict(value))

    def validate(self) -> None:
        slot_valid = _as_numpy(self.slot_valid, bool)
        if slot_valid.shape != (6,):
            raise ValueError("Flow START metadata.slot_valid must have shape [6]")
        maps = _as_numpy(self.map_polylines, np.float32)
        map_valid = _as_numpy(self.map_polyline_valid, bool)
        edges = _as_numpy(self.lane_graph_edges, np.int64)
        if maps.ndim != 3 or maps.shape[-1] != 6 or map_valid.shape != maps.shape[:2]:
            raise ValueError("Flow START metadata map tensors must be [polylines, points, 6] with matching validity")
        if edges.ndim != 2 or edges.shape[-1] != 3:
            raise ValueError("Flow START metadata.lane_graph_edges must have shape [edges, 3]")
        primary = int(self.primary_slot_index)
        if not 0 <= primary < 6 or not slot_valid[primary]:
            raise ValueError("Flow START metadata.primary_slot_index must identify one background slot")
        mask_pattern = sum(int(value) << index for index, value in enumerate(slot_valid))
        if int(self.mask_pattern) != mask_pattern:
            raise ValueError("Flow START metadata.mask_pattern must match slot_valid")
        density = np.asarray(
            (self.event_structure_log_prob, self.conditional_log_prob, self.log_prob), np.float32,
        )
        if not np.isfinite(density).all() or not np.isclose(
            density[2], density[0] + density[1], atol=1.0e-4,
        ):
            raise ValueError("Flow START metadata must retain log_prob=event_structure_log_prob+conditional_log_prob")

    def audit_dict(self) -> dict[str, Any]:
        return {
            key: deepcopy(value)
            for key, value in self.__dict__.items()
            if key not in {"map_polylines", "map_polyline_valid", "lane_graph_edges"}
        }


class QRWorldModelEnvironment:
    """25 Hz joint environment with 5 Hz QR-WM background replanning.

    ``step`` accepts only the physical ADS control ``[acceleration, yaw_rate]``.
    The control is consumed by the environment's ego dynamics and never passed
    to QR-WM.  QR-WM plans background actions on every response boundary from
    the joint state/history that has actually occurred so far.
    """

    def __init__(self, model: QueryRefineWorldModel, *, device: str | torch.device = "cpu") -> None:
        self.model = model.to(device).eval()
        self.device = torch.device(device)
        self._metadata: FlowStartMetadata | None = None
        self._states: torch.Tensor | None = None
        self._valid: torch.Tensor | None = None
        self._history: torch.Tensor | None = None
        self._history_valid: torch.Tensor | None = None
        self._behavior: torch.Tensor | None = None
        self._memory: torch.Tensor | None = None
        self._anchor_actions: torch.Tensor | None = None
        self._previous_buffer: torch.Tensor | None = None
        self._previous_current: torch.Tensor | None = None
        self._map_inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        self._active_plan: torch.Tensor | None = None
        self._plan_frame_index = 0
        self._has_planned = False
        self.physics_step_index = 0
        self.response_index = 0
        self.trace: dict[str, Any] = {}

    def reset_from_flow(
        self,
        C0: np.ndarray | torch.Tensor,
        B0: np.ndarray | torch.Tensor,
        metadata: FlowStartMetadata | Mapping[str, Any],
        *,
        deterministic: bool = True,
        world_randomness: WorldRandomness | Mapping[str, Any] | int | None = None,
    ) -> dict[str, Any]:
        """Initialize an episode from raw Flow C0, B0, and auditable metadata."""
        self._metadata = FlowStartMetadata.from_value(metadata)
        self._metadata.validate()
        c0, b0 = _as_numpy(C0, np.float32).reshape(-1), _as_numpy(B0, np.float32)
        if c0.shape != (40,) or b0.shape != (6, 6):
            raise ValueError("reset_from_flow requires C0[40] and B0[6, 6] in raw Flow coordinates")
        flow = torch.as_tensor(np.concatenate((c0, b0.reshape(-1)))[None], device=self.device)
        slot_valid = torch.as_tensor(_as_numpy(self._metadata.slot_valid, bool)[None], device=self.device)
        self._map_inputs = (
            torch.as_tensor(_as_numpy(self._metadata.map_polylines, np.float32)[None], device=self.device),
            torch.as_tensor(_as_numpy(self._metadata.map_polyline_valid, bool)[None], device=self.device),
            torch.as_tensor(_as_numpy(self._metadata.lane_graph_edges, np.int64)[None], device=self.device),
        )
        behavior_noise, randomness_audit = _single_world_noise(
            world_randomness,
            deterministic=deterministic,
            agents=7,
            latent_dim=self.model.cfg.behavior_latent_dim,
            device=self.device,
            dtype=flow.dtype,
        )
        with torch.no_grad():
            current, valid, raw_anchor = self.model.flow_condition_to_scene(flow, slot_valid)
            ego_mask = torch.zeros_like(valid); ego_mask[:, 0] = True
            start = self.model.initialize_start(
                current, valid, ego_mask, *self._map_inputs, raw_anchor, slot_valid,
                deterministic=deterministic, behavior_standard_normal=(
                    None if behavior_noise is None else behavior_noise[None]
                ),
            )
        self._states, self._valid = current, valid
        self._history, self._history_valid = current[:, None], valid[:, None]
        self._behavior = start["behavior_latent"]
        self._memory = start["scene_memory"]
        self._anchor_actions = start["start_anchor_actions"]
        self._previous_buffer = self._previous_current = None
        self._active_plan = None
        self._plan_frame_index = 0
        self._has_planned = False
        self.physics_step_index = 0
        self.response_index = 0
        self.trace = {
            "flow_metadata": self._metadata.audit_dict(), "b0_lifecycle": "START-only",
            "ego_condition": "25 Hz ADS actions applied only by environment ego dynamics",
            "response_steps": 0, "physics_steps": 0, "planning_steps": 0,
            "world_randomness": randomness_audit,
        }
        return self.observe()

    def observe(self) -> dict[str, Any]:
        if self._states is None or self._valid is None or self._metadata is None:
            raise RuntimeError("Call reset_from_flow before observe")
        return {
            "agent_states": self._states[0].detach().cpu().numpy().copy(),
            "agent_valid": self._valid[0].detach().cpu().numpy().astype(bool, copy=True),
            "response_index": self.response_index, "physics_step_index": self.physics_step_index,
            "flow_metadata": self._metadata.audit_dict(),
            "world_randomness": deepcopy(self.trace["world_randomness"]),
        }

    @torch.no_grad()
    def _plan_if_needed(self) -> bool:
        """Create the next background plan exactly at a 5 Hz response boundary."""
        required = (
            self._states, self._valid, self._history, self._history_valid,
            self._behavior, self._memory, self._metadata,
        )
        if any(value is None for value in required):
            raise RuntimeError("Call reset_from_flow before step")
        if self._map_inputs is None:
            raise RuntimeError("Flow START map inputs are unavailable")
        if self._active_plan is not None and self._plan_frame_index < self.model.cfg.execute_frames:
            return False
        ego_mask = torch.zeros_like(self._valid); ego_mask[:, 0] = True
        out = self.model.plan_step(
            self._history, self._history_valid, self._states, self._valid, ego_mask, *self._map_inputs,
            self._behavior, previous_buffer=self._previous_buffer,
            previous_current=self._previous_current, previous_memory=self._memory,
            start_anchor_actions=self._anchor_actions if not self._has_planned else None,
            start_mode=not self._has_planned,
        )
        self._active_plan = out["background_future_actions"]
        self._plan_frame_index = 0
        self._previous_buffer, self._previous_current = self._active_plan, self._states
        self._memory = out["scene_memory"]
        self._has_planned = True
        self.trace["planning_steps"] += 1
        return True

    @torch.no_grad()
    def step(self, ads_action: np.ndarray | torch.Tensor, ego_valid: bool = True) -> dict[str, Any]:
        """Advance ego and background by one 0.04-second physical tick.

        ``ads_action`` has shape ``[2]`` in the shared
        ``[longitudinal_acceleration, yaw_rate]`` control coordinates.  It is
        assembled into the joint physical control only after QR-WM has (if
        needed) generated a background plan, so it cannot enter the network.
        """
        if self._states is None or self._valid is None:
            raise RuntimeError("Call reset_from_flow before step")
        action = torch.as_tensor(ads_action, dtype=self._states.dtype, device=self.device)
        if tuple(action.shape) != (2,):
            raise ValueError("ads_action must have shape [2]")
        self._states = self._states.clone()
        self._valid = self._valid.clone()
        self._valid[:, 0] = bool(ego_valid)
        if not ego_valid:
            self._states[:, 0] = 0.0
        planner_updated = self._plan_if_needed()
        if self._active_plan is None:
            raise RuntimeError("QR-WM failed to create a background action plan")
        executed_plan_frame = self._plan_frame_index
        physical = self._states.new_zeros((1, self._states.shape[1], 2))
        physical[:, 0] = action
        physical[:, 1:] = self._active_plan[:, executed_plan_frame]
        self._states = self.model.dynamics.step(
            self._states, physical, self._valid, self.model.cfg.simulation_dt_s
        )
        self._history = torch.cat((self._history, self._states[:, None]), dim=1)[:, -25:]
        self._history_valid = torch.cat((self._history_valid, self._valid[:, None]), dim=1)[:, -25:]
        self._plan_frame_index += 1
        self.physics_step_index += 1
        self.trace["physics_steps"] = self.physics_step_index
        if self._plan_frame_index == self.model.cfg.execute_frames:
            self.response_index += 1
            self.trace["response_steps"] = self.response_index
        observation = self.observe()
        observation.update({
            "background_state": self._states[0, 1:].cpu().numpy(),
            "applied_ego_action": action.cpu().numpy(),
            "applied_background_actions": physical[0, 1:].cpu().numpy(),
            "background_future_actions": self._active_plan[0].cpu().numpy(),
            "planner_updated": planner_updated, "executed_plan_frame": executed_plan_frame,
            "trace": deepcopy(self.trace),
        })
        return observation

    @torch.no_grad()
    def advance_response(
        self, ads_actions: np.ndarray | torch.Tensor, ego_valid: bool | np.ndarray | torch.Tensor = True,
    ) -> dict[str, Any]:
        """Advance one response prefix using one to five ADS controls.

        Normal operation supplies five controls (0.2 s).  A final four-tick
        prefix is also valid for the audited 150-state highD window, whose
        physical horizon is 5.96 s rather than an invented terminal S150.
        """
        if self._active_plan is not None and 0 < self._plan_frame_index < self.model.cfg.execute_frames:
            raise RuntimeError("advance_response must start on a response boundary")
        actions = torch.as_tensor(ads_actions, dtype=torch.float32, device=self.device)
        if actions.ndim != 2 or actions.shape[1] != 2 or not 1 <= actions.shape[0] <= self.model.cfg.execute_frames:
            raise ValueError(
                "ads_actions must have shape [ticks, 2] with "
                f"1 <= ticks <= {self.model.cfg.execute_frames}"
            )
        ticks_count = int(actions.shape[0])
        valid = torch.as_tensor(ego_valid, dtype=torch.bool, device=self.device)
        if valid.ndim == 0:
            valid = valid.expand(ticks_count)
        if tuple(valid.shape) != (ticks_count,):
            raise ValueError("ego_valid must be a bool or have shape [ticks]")
        ticks = [self.step(actions[index], bool(valid[index])) for index in range(ticks_count)]
        observation = ticks[-1]
        states = np.stack([tick["agent_states"] for tick in ticks])
        observation.update({
            "agent_state_frames": states,
            "background_states": states[:, 1:],
            "applied_ego_actions": np.stack([tick["applied_ego_action"] for tick in ticks]),
            "applied_background_action_frames": np.stack(
                [tick["applied_background_actions"] for tick in ticks]
            ),
            "planning_updates": np.asarray([tick["planner_updated"] for tick in ticks], dtype=bool),
        })
        return observation


class BatchedQRWorldModelEnvironment:
    """Vectorized causal QR-WM environments for independent Flow world samples.

    Every row keeps independent scene state, history, latent, memory and action
    buffer.  The rows are only batched for neural-network and dynamics calls;
    no ego state or future information is shared across them.

    Its public step observations mirror :class:`QRWorldModelEnvironment` with
    a leading batch dimension.  This keeps high-throughput Flow evaluation
    auditable without exposing mutable private planning buffers.
    """

    def __init__(self, model: QueryRefineWorldModel, *, device: str | torch.device = "cpu") -> None:
        self.model = model.to(device).eval()
        self.device = torch.device(device)
        self._states: torch.Tensor | None = None
        self._valid: torch.Tensor | None = None
        self._history: torch.Tensor | None = None
        self._history_valid: torch.Tensor | None = None
        self._behavior: torch.Tensor | None = None
        self._memory: torch.Tensor | None = None
        self._anchor_actions: torch.Tensor | None = None
        self._previous_buffer: torch.Tensor | None = None
        self._previous_current: torch.Tensor | None = None
        self._map_inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        self._active_plan: torch.Tensor | None = None
        self._plan_frame_index = 0
        self._has_planned = False
        self.physics_step_index = 0
        self.response_index = 0

    @torch.no_grad()
    def reset_from_flow_batch(
        self,
        flow_features: np.ndarray | torch.Tensor,
        slot_valid: np.ndarray | torch.Tensor,
        map_polylines: np.ndarray | torch.Tensor,
        map_polyline_valid: np.ndarray | torch.Tensor,
        lane_graph_edges: np.ndarray | torch.Tensor,
        *,
        deterministic: bool = False,
        world_randomness: Sequence[WorldRandomness | Mapping[str, Any] | int] | None = None,
    ) -> None:
        """Initialize one independent Flow condition per batch row."""

        flow = torch.as_tensor(flow_features, dtype=torch.float32, device=self.device)
        slots = torch.as_tensor(slot_valid, dtype=torch.bool, device=self.device)
        maps = torch.as_tensor(map_polylines, dtype=torch.float32, device=self.device)
        map_valid = torch.as_tensor(map_polyline_valid, dtype=torch.bool, device=self.device)
        edges = torch.as_tensor(lane_graph_edges, dtype=torch.long, device=self.device)
        if flow.ndim != 2 or flow.shape[1] != 76 or slots.shape != (len(flow), 6):
            raise ValueError("batched Flow reset requires features [batch, 76] and slot_valid [batch, 6]")
        if maps.ndim != 4 or maps.shape[0] != len(flow) or map_valid.shape != maps.shape[:3]:
            raise ValueError("batched Flow map tensors must align with the Flow batch")
        if edges.ndim != 3 or edges.shape[0] != len(flow) or edges.shape[-1] != 3:
            raise ValueError("batched Flow lane_graph_edges must be [batch, edges, 3]")
        behavior_noise, randomness_audit = _batched_world_noise(
            world_randomness,
            batch_size=len(flow),
            deterministic=deterministic,
            agents=7,
            latent_dim=self.model.cfg.behavior_latent_dim,
            device=self.device,
            dtype=flow.dtype,
        )
        current, valid, raw_anchor = self.model.flow_condition_to_scene(flow, slots)
        ego_mask = torch.zeros_like(valid); ego_mask[:, 0] = True
        start = self.model.initialize_start(
            current, valid, ego_mask, maps, map_valid, edges, raw_anchor, slots,
            deterministic=deterministic, behavior_standard_normal=behavior_noise,
        )
        self._states, self._valid = current, valid
        self._history, self._history_valid = current[:, None], valid[:, None]
        self._behavior, self._memory = start["behavior_latent"], start["scene_memory"]
        self._anchor_actions = start["start_anchor_actions"]
        self._previous_buffer = self._previous_current = None
        self._active_plan = None
        self._plan_frame_index = 0
        self._has_planned = False
        self.physics_step_index = 0
        self._map_inputs = (maps, map_valid, edges)
        self.world_randomness_audit = randomness_audit
        self.response_index = 0

    def observe(self) -> dict[str, Any]:
        """Return the current batched state and episode counters.

        State tensors are cloned so an integration caller cannot mutate the
        causal environment state between 25 Hz ticks.
        """
        if self._states is None or self._valid is None:
            raise RuntimeError("Call reset_from_flow_batch before observe")
        return {
            "agent_states": self._states.clone(),
            "agent_valid": self._valid.clone(),
            "response_index": self.response_index,
            "physics_step_index": self.physics_step_index,
            "world_randomness": deepcopy(self.world_randomness_audit),
        }

    @torch.no_grad()
    def _plan_if_needed(self) -> bool:
        """Create a shared-tensor background plan when the 5 Hz boundary is due."""
        required = (self._states, self._valid, self._history, self._history_valid, self._behavior, self._memory)
        if any(value is None for value in required) or self._map_inputs is None:
            raise RuntimeError("Call reset_from_flow_batch before step")
        if self._active_plan is not None and self._plan_frame_index < self.model.cfg.execute_frames:
            return False
        ego_mask = torch.zeros_like(self._valid); ego_mask[:, 0] = True
        out = self.model.plan_step(
            self._history, self._history_valid, self._states, self._valid, ego_mask, *self._map_inputs, self._behavior,
            previous_buffer=self._previous_buffer, previous_current=self._previous_current,
            previous_memory=self._memory, start_anchor_actions=self._anchor_actions if not self._has_planned else None,
            start_mode=not self._has_planned,
        )
        self._active_plan = out["background_future_actions"]
        self._plan_frame_index = 0
        self._previous_buffer, self._previous_current, self._memory = self._active_plan, self._states, out["scene_memory"]
        self._has_planned = True
        return True

    @torch.no_grad()
    def step(
        self, ads_actions: np.ndarray | torch.Tensor, ego_valid: np.ndarray | torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Advance every batch row by one 0.04-second joint physical tick.

        Inputs are ADS controls ``[batch, 2]``.  The returned observation
        includes the joint post-step state, applied ego/background controls,
        active background plan, and the 5 Hz planning boundary flag.  Controls
        never enter QR-WM.
        """
        if self._states is None or self._valid is None:
            raise RuntimeError("Call reset_from_flow_batch before step")
        actions = torch.as_tensor(ads_actions, dtype=self._states.dtype, device=self.device)
        if actions.shape != (self._states.shape[0], 2):
            raise ValueError("ads_actions must have shape [batch, 2]")
        valid_ego = (
            torch.ones(len(actions), dtype=torch.bool, device=self.device)
            if ego_valid is None
            else torch.as_tensor(ego_valid, dtype=torch.bool, device=self.device)
        )
        if valid_ego.shape != (len(actions),):
            raise ValueError("ego_valid must have shape [batch]")
        self._states, self._valid = self._states.clone(), self._valid.clone()
        self._valid[:, 0] = valid_ego
        self._states[:, 0] *= valid_ego[:, None].float()
        planner_updated = self._plan_if_needed()
        if self._active_plan is None:
            raise RuntimeError("QR-WM failed to create a background action plan")
        executed_plan_frame = self._plan_frame_index
        physical = self._states.new_zeros((len(self._states), self._states.shape[1], 2))
        physical[:, 0] = actions
        physical[:, 1:] = self._active_plan[:, executed_plan_frame]
        self._states = self.model.dynamics.step(
            self._states, physical, self._valid, self.model.cfg.simulation_dt_s
        )
        self._history = torch.cat((self._history, self._states[:, None]), dim=1)[:, -25:]
        self._history_valid = torch.cat((self._history_valid, self._valid[:, None]), dim=1)[:, -25:]
        self._plan_frame_index += 1
        self.physics_step_index += 1
        if self._plan_frame_index == self.model.cfg.execute_frames:
            self.response_index += 1
        observation = self.observe()
        observation.update({
            "applied_ego_action": actions.clone(),
            "applied_background_actions": physical[:, 1:].clone(),
            "background_future_actions": self._active_plan.clone(),
            "planner_updated": planner_updated,
            "executed_plan_frame": executed_plan_frame,
        })
        return observation

    @torch.no_grad()
    def advance_response(
        self, ads_actions: np.ndarray | torch.Tensor, ego_valid: np.ndarray | torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Advance a response prefix from ``[batch, ticks, 2]`` controls.

        The result is strictly equivalent to repeatedly calling :meth:`step`
        and additionally stacks the per-tick observation fields.
        """
        if self._states is None:
            raise RuntimeError("Call reset_from_flow_batch before advance_response")
        if self._active_plan is not None and 0 < self._plan_frame_index < self.model.cfg.execute_frames:
            raise RuntimeError("advance_response must start on a response boundary")
        actions = torch.as_tensor(ads_actions, dtype=self._states.dtype, device=self.device)
        if (
            actions.ndim != 3
            or actions.shape[0] != self._states.shape[0]
            or actions.shape[2] != 2
            or not 1 <= actions.shape[1] <= self.model.cfg.execute_frames
        ):
            raise ValueError(
                "ads_actions must have shape [batch, ticks, 2] with "
                f"1 <= ticks <= {self.model.cfg.execute_frames}"
            )
        ticks_count = int(actions.shape[1])
        expected = (self._states.shape[0], ticks_count)
        if ego_valid is None:
            valid = torch.ones(expected, dtype=torch.bool, device=self.device)
        else:
            valid = torch.as_tensor(ego_valid, dtype=torch.bool, device=self.device)
            if valid.shape == (self._states.shape[0],):
                valid = valid[:, None].expand(-1, ticks_count)
        if tuple(valid.shape) != expected:
            raise ValueError("ego_valid must have shape [batch] or [batch, ticks]")
        ticks = [self.step(actions[:, index], valid[:, index]) for index in range(ticks_count)]
        observation = ticks[-1]
        states = torch.stack([tick["agent_states"] for tick in ticks], dim=1)
        observation.update({
            "agent_state_frames": states,
            "background_states": states[:, :, 1:],
            "applied_ego_actions": torch.stack(
                [tick["applied_ego_action"] for tick in ticks], dim=1
            ),
            "applied_background_action_frames": torch.stack(
                [tick["applied_background_actions"] for tick in ticks], dim=1
            ),
            "planning_updates": torch.as_tensor(
                [tick["planner_updated"] for tick in ticks], dtype=torch.bool, device=self.device,
            ),
        })
        return observation
