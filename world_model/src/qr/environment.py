"""Online QR-WM environment driven by observed ego states."""

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
    """A 0.2-second closed-loop environment conditioned on observed ego state."""

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
        self.response_index = 0
        self.trace = {
            "flow_metadata": self._metadata.audit_dict(), "b0_lifecycle": "START-only",
            "ego_condition": "observed state only", "response_steps": 0,
            "world_randomness": randomness_audit,
        }
        return self.observe()

    def observe(self) -> dict[str, Any]:
        if self._states is None or self._valid is None or self._metadata is None:
            raise RuntimeError("Call reset_from_flow before observe")
        return {
            "agent_states": self._states[0].detach().cpu().numpy().copy(),
            "agent_valid": self._valid[0].detach().cpu().numpy().astype(bool, copy=True),
            "response_index": self.response_index, "flow_metadata": self._metadata.audit_dict(),
            "world_randomness": deepcopy(self.trace["world_randomness"]),
        }

    @torch.no_grad()
    def step(self, ego_state: np.ndarray | torch.Tensor, ego_valid: bool = True) -> dict[str, Any]:
        """Advance one response interval from the currently observed ego state."""
        required = (
            self._states, self._valid, self._history, self._history_valid,
            self._behavior, self._memory, self._metadata,
        )
        if any(value is None for value in required):
            raise RuntimeError("Call reset_from_flow before step")
        observed = torch.as_tensor(ego_state, dtype=self._states.dtype, device=self.device)
        if tuple(observed.shape) != (6,):
            raise ValueError("ego_state must have shape [6]")
        if self._map_inputs is None:
            raise RuntimeError("Flow START map inputs are unavailable")
        self._states = self._states.clone()
        self._valid = self._valid.clone()
        self._states[:, 0] = observed
        self._valid[:, 0] = bool(ego_valid)
        self._history[:, -1] = self._states
        self._history_valid[:, -1] = self._valid
        ego_mask = torch.zeros_like(self._valid); ego_mask[:, 0] = True
        out = self.model.plan_step(
            self._history, self._history_valid, self._states, self._valid, ego_mask, *self._map_inputs,
            self._behavior, previous_buffer=self._previous_buffer,
            previous_current=self._previous_current, previous_memory=self._memory,
            start_anchor_actions=self._anchor_actions if self.response_index == 0 else None,
            start_mode=self.response_index == 0,
        )
        before, current, frames = self._states, self._states, []
        plan = out["background_future_actions"]
        for frame in range(self.model.cfg.execute_frames):
            physical = current.new_zeros((1, current.shape[1], 2))
            physical[:, 1:] = plan[:, frame]
            current = self.model.dynamics.step(current, physical, self._valid, self.model.cfg.simulation_dt_s)
            current[:, 0] = observed
            current[:, 0] *= float(ego_valid)
            frames.append(current)
        appended = torch.stack(frames, dim=1)
        self._states = current
        self._history = torch.cat((self._history, appended), dim=1)[:, -25:]
        valid_frames = self._valid[:, None].expand(-1, len(frames), -1)
        self._history_valid = torch.cat((self._history_valid, valid_frames), dim=1)[:, -25:]
        self._previous_buffer, self._previous_current = plan, before
        self._memory = out["scene_memory"]
        self.response_index += 1
        self.trace["response_steps"] = self.response_index
        observation = self.observe()
        observation.update({
            "background_states": appended[0, :, 1:].cpu().numpy(),
            "observed_ego_state": observed.cpu().numpy(),
            "applied_background_actions": plan[0, : self.model.cfg.execute_frames].cpu().numpy(),
            "background_future_actions": plan[0].cpu().numpy(), "trace": deepcopy(self.trace),
        })
        return observation


class BatchedQRWorldModelEnvironment:
    """Vectorized causal QR-WM environments for independent Flow world samples.

    Every row keeps independent scene state, history, latent, memory and action
    buffer.  The rows are only batched for neural-network and dynamics calls;
    no ego state or future information is shared across them.
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
        self._map_inputs = (maps, map_valid, edges)
        self.world_randomness_audit = randomness_audit
        self.response_index = 0

    @torch.no_grad()
    def step(self, ego_states: np.ndarray | torch.Tensor, ego_valid: np.ndarray | torch.Tensor | None = None) -> torch.Tensor:
        """Causally advance every batch row by one response interval."""

        required = (self._states, self._valid, self._history, self._history_valid, self._behavior, self._memory)
        if any(value is None for value in required) or self._map_inputs is None:
            raise RuntimeError("Call reset_from_flow_batch before step")
        observed = torch.as_tensor(ego_states, dtype=self._states.dtype, device=self.device)
        if observed.shape != (self._states.shape[0], 6):
            raise ValueError("ego_states must have shape [batch, 6]")
        valid_ego = torch.ones(len(observed), dtype=torch.bool, device=self.device) if ego_valid is None else torch.as_tensor(ego_valid, dtype=torch.bool, device=self.device)
        if valid_ego.shape != (len(observed),):
            raise ValueError("ego_valid must have shape [batch]")
        states, valid = self._states.clone(), self._valid.clone()
        states[:, 0], valid[:, 0] = observed, valid_ego
        history, history_valid = self._history.clone(), self._history_valid.clone()
        history[:, -1], history_valid[:, -1] = states, valid
        ego_mask = torch.zeros_like(valid); ego_mask[:, 0] = True
        out = self.model.plan_step(
            history, history_valid, states, valid, ego_mask, *self._map_inputs, self._behavior,
            previous_buffer=self._previous_buffer, previous_current=self._previous_current,
            previous_memory=self._memory, start_anchor_actions=self._anchor_actions if self.response_index == 0 else None,
            start_mode=self.response_index == 0,
        )
        before, current, frames = states, states, []
        plan = out["background_future_actions"]
        for frame in range(self.model.cfg.execute_frames):
            physical = current.new_zeros((len(current), current.shape[1], 2))
            physical[:, 1:] = plan[:, frame]
            current = self.model.dynamics.step(current, physical, valid, self.model.cfg.simulation_dt_s)
            current[:, 0] = observed * valid_ego[:, None]
            frames.append(current)
        appended = torch.stack(frames, dim=1)
        valid_frames = valid[:, None].expand(-1, len(frames), -1)
        self._states, self._valid = current, valid
        self._history = torch.cat((history, appended), dim=1)[:, -25:]
        self._history_valid = torch.cat((history_valid, valid_frames), dim=1)[:, -25:]
        self._previous_buffer, self._previous_current, self._memory = plan, before, out["scene_memory"]
        self.response_index += 1
        return appended[:, :, 1:]
