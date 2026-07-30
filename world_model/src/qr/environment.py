"""Online QR-WM environment driven by observed ego states."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch

from .model import QueryRefineWorldModel


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
        if np.asarray(self.slot_valid, bool).shape != (6,):
            raise ValueError("Flow START metadata.slot_valid must have shape [6]")
        maps = np.asarray(self.map_polylines)
        map_valid = np.asarray(self.map_polyline_valid)
        edges = np.asarray(self.lane_graph_edges)
        if maps.ndim != 3 or maps.shape[-1] != 6 or map_valid.shape != maps.shape[:2]:
            raise ValueError("Flow START metadata map tensors must be [polylines, points, 6] with matching validity")
        if edges.ndim != 2 or edges.shape[-1] != 3:
            raise ValueError("Flow START metadata.lane_graph_edges must have shape [edges, 3]")
        if not 0 <= int(self.primary_slot_index) < 6:
            raise ValueError("Flow START metadata.primary_slot_index must identify one background slot")
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
    ) -> dict[str, Any]:
        """Initialize an episode from raw Flow C0, B0, and auditable metadata."""
        self._metadata = FlowStartMetadata.from_value(metadata)
        self._metadata.validate()
        c0, b0 = np.asarray(C0, np.float32).reshape(-1), np.asarray(B0, np.float32)
        if c0.shape != (40,) or b0.shape != (6, 6):
            raise ValueError("reset_from_flow requires C0[40] and B0[6, 6] in raw Flow coordinates")
        flow = torch.as_tensor(np.concatenate((c0, b0.reshape(-1)))[None], device=self.device)
        slot_valid = torch.as_tensor(np.asarray(self._metadata.slot_valid, bool)[None], device=self.device)
        self._map_inputs = (
            torch.as_tensor(np.asarray(self._metadata.map_polylines, np.float32)[None], device=self.device),
            torch.as_tensor(np.asarray(self._metadata.map_polyline_valid, bool)[None], device=self.device),
            torch.as_tensor(np.asarray(self._metadata.lane_graph_edges, np.int64)[None], device=self.device),
        )
        with torch.no_grad():
            current, valid, raw_anchor = self.model.flow_condition_to_scene(flow, slot_valid)
            ego_mask = torch.zeros_like(valid); ego_mask[:, 0] = True
            start = self.model.initialize_start(
                current, valid, ego_mask, *self._map_inputs, raw_anchor, slot_valid,
                deterministic=deterministic,
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
        }
        return self.observe()

    def observe(self) -> dict[str, Any]:
        if self._states is None or self._valid is None or self._metadata is None:
            raise RuntimeError("Call reset_from_flow before observe")
        return {
            "agent_states": self._states[0].detach().cpu().numpy().copy(),
            "agent_valid": self._valid[0].detach().cpu().numpy().astype(bool, copy=True),
            "response_index": self.response_index, "flow_metadata": self._metadata.audit_dict(),
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
