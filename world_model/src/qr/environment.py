"""Online QR-WM environment driven by incremental ADS controls."""

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
    """A 0.2-second closed-loop environment; ADS supplies only the next chunk."""

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
        self._anchor_controls: torch.Tensor | None = None
        self._previous_buffer: torch.Tensor | None = None
        self._previous_current: torch.Tensor | None = None
        self._previous_ego: torch.Tensor | None = None
        self._map_inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        self.response_index = 0
        self.trace: dict[str, Any] = {}

    def reset_from_flow(
        self,
        C0: np.ndarray | torch.Tensor,
        B0: np.ndarray | torch.Tensor,
        metadata: FlowStartMetadata | Mapping[str, Any],
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
            start = self.model.initialize_start(current, valid, ego_mask, *self._map_inputs, raw_anchor, slot_valid)
        self._states, self._valid = current, valid
        self._history, self._history_valid = current[:, None], valid[:, None]
        self._behavior = start["behavior_latent"]
        self._memory = start["scene_memory"]
        self._anchor_controls = start["start_anchor_controls"]
        self._previous_buffer = self._previous_current = self._previous_ego = None
        self.response_index = 0
        self.trace = {
            "flow_metadata": self._metadata.audit_dict(), "b0_lifecycle": "START-only",
            "online_ego_tail_policy": "hold_last_control", "response_steps": 0,
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
    def step(self, ego_action: np.ndarray | torch.Tensor) -> dict[str, Any]:
        """Apply the next five 25-Hz ADS controls, then replan from feedback."""
        required = (
            self._states, self._valid, self._history, self._history_valid,
            self._behavior, self._memory, self._metadata,
        )
        if any(value is None for value in required):
            raise RuntimeError("Call reset_from_flow before step")
        action = torch.as_tensor(ego_action, dtype=self._states.dtype, device=self.device)
        expected = (self.model.cfg.execute_frames, 2)
        if tuple(action.shape) != expected:
            raise ValueError(f"ego_action must have shape {expected}")
        tail = action[-1:].expand(self.model.cfg.plan_frames - len(action), -1)
        horizon = torch.cat((action, tail), dim=0)[None]
        if self._map_inputs is None:
            raise RuntimeError("Flow START map inputs are unavailable")
        ego_mask = torch.zeros_like(self._valid); ego_mask[:, 0] = True
        ego_states = self.model._integrate_ego_plan(self._states, self._valid, horizon)
        out = self.model.plan_step(
            self._history, self._history_valid, self._states, self._valid, ego_mask, *self._map_inputs,
            self._behavior, horizon, ego_states, previous_buffer=self._previous_buffer,
            previous_current=self._previous_current, previous_memory=self._memory,
            previous_ego_control=self._previous_ego,
            start_anchor_controls=self._anchor_controls if self.response_index == 0 else None,
            start_mode=self.response_index == 0,
        )
        before, current, frames = self._states, self._states, []
        plan = out["refined_buffer"]
        for frame in range(self.model.cfg.execute_frames):
            physical = current.new_zeros((1, current.shape[1], 2))
            physical[:, 0] = action[frame]
            physical[:, 1:] = plan[:, frame]
            current = self.model.dynamics.step(current, physical, self._valid, self.model.cfg.simulation_dt_s)
            frames.append(current)
        appended = torch.stack(frames, dim=1)
        self._states = current
        self._history = torch.cat((self._history, appended), dim=1)[:, -25:]
        valid_frames = self._valid[:, None].expand(-1, len(frames), -1)
        self._history_valid = torch.cat((self._history_valid, valid_frames), dim=1)[:, -25:]
        self._previous_buffer, self._previous_current = plan, before
        self._previous_ego, self._memory = action[-1:], out["scene_memory"]
        self.response_index += 1
        self.trace["response_steps"] = self.response_index
        observation = self.observe()
        observation.update({
            "background_states": appended[0, :, 1:].cpu().numpy(),
            "applied_ego_controls": action.cpu().numpy(),
            "applied_background_controls": plan[0, : self.model.cfg.execute_frames].cpu().numpy(),
            "control_buffer": plan[0].cpu().numpy(), "trace": deepcopy(self.trace),
        })
        return observation
