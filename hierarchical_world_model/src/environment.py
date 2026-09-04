"""Offline causal response-boundary environment with exact branch replay.

This module is retained for model/loss unit tests and differentiable offline
evaluation.  It is not the execution backend for ADS risk estimates; those
must use :mod:`highway`, which owns the HighwayEnv road,
collision and IDM dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from world_model.src.hiqr.filter import FilterState

from .model import DiffusionGuidedHiQR
from .randomness import WorldExogenousState
from .reaction_controller import ReactionController, ReactionControllerContext, make_reaction_controller


@dataclass(frozen=True)
class WorldSnapshot:
    states: torch.Tensor
    valid: torch.Tensor
    history: torch.Tensor
    history_valid: torch.Tensor
    reference_index: int
    motion_generator_state: torch.Tensor
    filter_global: torch.Tensor | None
    filter_agents: torch.Tensor | None
    slow_scene: torch.Tensor | None
    slow_scene_noise: torch.Tensor | None
    agent_noise_state: torch.Tensor | None
    agent_style_state: torch.Tensor | None
    previous_current: torch.Tensor | None
    committed_ego_controls: torch.Tensor
    intervention_memory: torch.Tensor | None
    lateral_intervention_memory: torch.Tensor | None
    response_innovations: torch.Tensor | None
    response_agent_innovations: torch.Tensor | None


class ClosedLoopWorld:
    """Execute ADS and background actions in synchronized 25 Hz responses."""

    def __init__(
        self,
        model: DiffusionGuidedHiQR,
        *,
        device: str | torch.device = "cpu",
        controller: ReactionController | str | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        if isinstance(controller, str):
            controller = make_reaction_controller(
                controller, adapter_logit=self.model.decoder.intervention_logit
            )
        self.controller = None if controller is None else controller.to(self.device)
        self.states: torch.Tensor | None = None
        self.valid: torch.Tensor | None = None
        self.history: torch.Tensor | None = None
        self.history_valid: torch.Tensor | None = None
        self.reference: torch.Tensor | None = None
        self.reference_base: torch.Tensor | None = None
        self.reference_index = 0
        self.generator = torch.Generator(device=self.device)
        self.map_polylines: torch.Tensor | None = None
        self.map_polyline_valid: torch.Tensor | None = None
        self.filter_state: FilterState | None = None
        self.slow_scene: torch.Tensor | None = None
        self.slow_scene_noise: torch.Tensor | None = None
        self.agent_noise_state: torch.Tensor | None = None
        self.agent_style_state: torch.Tensor | None = None
        self.previous_current: torch.Tensor | None = None
        self.committed_ego_controls: torch.Tensor | None = None
        self.intervention_memory: torch.Tensor | None = None
        self.lateral_intervention_memory: torch.Tensor | None = None
        self.response_innovations: torch.Tensor | None = None
        self.response_agent_innovations: torch.Tensor | None = None

    def _controller_context(self, response: object) -> ReactionControllerContext:
        assert self.history is not None and self.history_valid is not None
        assert self.states is not None and self.valid is not None
        assert self.committed_ego_controls is not None
        return ReactionControllerContext(
            history=self.history, history_valid=self.history_valid,
            current=self.states, current_valid=self.valid,
            committed_ego_controls=self.committed_ego_controls,
            base_actions=response.actions, reference_actions=response.reference_actions,
            intervention_trigger=response.intervention_trigger,
            intervention_memory=response.intervention_memory,
            lateral_intervention_memory=response.lateral_intervention_memory,
            agent_style_state=response.agent_style_state,
            response_field_gain=response.response_field_gain,
            response_sensitivity_bounds=self.model.response_sensitivity_bounds,
            adapter_gain=torch.sigmoid(self.model.decoder.intervention_logit), cfg=self.model.cfg,
            reaction_enabled=None,
        )

    @torch.no_grad()
    def reset(
        self,
        initial_states: torch.Tensor,
        valid: torch.Tensor,
        soft_reference: torch.Tensor,
        map_polylines: torch.Tensor,
        map_polyline_valid: torch.Tensor,
        *,
        motion_seed: int | None = None,
        exogenous_state: WorldExogenousState | None = None,
    ) -> dict[str, torch.Tensor | int]:
        states = torch.as_tensor(
            initial_states, dtype=torch.float32, device=self.device
        )
        present = torch.as_tensor(valid, dtype=torch.bool, device=self.device)
        reference = torch.as_tensor(
            soft_reference, dtype=torch.float32, device=self.device
        )
        if states.ndim == 2:
            states = states[None]
            present = present[None]
            reference = reference[None]
        if states.shape[1:] != (7, 6) or present.shape != states.shape[:2]:
            raise ValueError("initial state contract is [batch,7,6]/[batch,7]")
        if reference.ndim != 4 or reference.shape[2:] != (6, 2):
            raise ValueError("soft_reference must be [batch,frames,6,2]")
        self.states = states.clone()
        self.valid = present.clone()
        self.history = states[:, None].clone()
        self.history_valid = present[:, None].clone()
        self.reference = reference.clone()
        self.reference_base = states[:, 1:, :2].clone()
        self.reference_index = 0
        if exogenous_state is not None:
            exogenous_state.validate(
                response_steps=exogenous_state.response_steps,
                scene_dim=self.model.cfg.scene_latent_dim,
                agent_dim=self.model.cfg.agent_latent_dim,
            )
            if exogenous_state.batch_size != states.shape[0]:
                raise ValueError("exogenous state and initial states have different batch sizes")
            self.response_innovations = torch.as_tensor(
                exogenous_state.scene_innovations,
                dtype=states.dtype,
                device=self.device,
            )
            self.response_agent_innovations = torch.as_tensor(
                exogenous_state.agent_response_innovations,
                dtype=states.dtype,
                device=self.device,
            )
        else:
            if motion_seed is None:
                raise ValueError("motion_seed is required when exogenous_state is absent")
            self.generator.manual_seed(int(motion_seed))
            self.response_innovations = None
            self.response_agent_innovations = None
        self.map_polylines = torch.as_tensor(
            map_polylines, dtype=torch.float32, device=self.device
        )
        self.map_polyline_valid = torch.as_tensor(
            map_polyline_valid, dtype=torch.bool, device=self.device
        )
        if self.map_polylines.ndim == 3:
            self.map_polylines = self.map_polylines[None]
            self.map_polyline_valid = self.map_polyline_valid[None]
        self.filter_state = None
        self.slow_scene = None
        self.slow_scene_noise = None
        self.agent_noise_state = None
        self.agent_style_state = None
        self.previous_current = None
        self.committed_ego_controls = torch.zeros(
            (states.shape[0], 1, 2), dtype=states.dtype, device=self.device
        )
        self.intervention_memory = None
        self.lateral_intervention_memory = None
        return self.observe()

    def _require(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.states is None or self.valid is None:
            raise RuntimeError("reset the world before stepping")
        return self.states, self.valid

    def observe(self) -> dict[str, torch.Tensor | int]:
        states, valid = self._require()
        return {
            "agent_states": states.detach().clone(),
            "agent_valid": valid.detach().clone(),
            "reference_index": self.reference_index,
        }

    def snapshot(self) -> WorldSnapshot:
        states, valid = self._require()
        assert self.history is not None and self.history_valid is not None
        return WorldSnapshot(
            states.detach().clone(),
            valid.detach().clone(),
            self.history.detach().clone(),
            self.history_valid.detach().clone(),
            self.reference_index,
            self.generator.get_state().clone(),
            None
            if self.filter_state is None
            else self.filter_state.global_hidden.detach().clone(),
            None
            if self.filter_state is None
            else self.filter_state.agent_hidden.detach().clone(),
            None if self.slow_scene is None else self.slow_scene.detach().clone(),
            None
            if self.slow_scene_noise is None
            else self.slow_scene_noise.detach().clone(),
            None
            if self.agent_noise_state is None
            else self.agent_noise_state.detach().clone(),
            None
            if self.agent_style_state is None
            else self.agent_style_state.detach().clone(),
            None
            if self.previous_current is None
            else self.previous_current.detach().clone(),
            self.committed_ego_controls.detach().clone(),
            None
            if self.intervention_memory is None
            else self.intervention_memory.detach().clone(),
            None
            if self.lateral_intervention_memory is None
            else self.lateral_intervention_memory.detach().clone(),
            None
            if self.response_innovations is None
            else self.response_innovations.detach().clone(),
            None
            if self.response_agent_innovations is None
            else self.response_agent_innovations.detach().clone(),
        )

    def restore(self, snapshot: WorldSnapshot) -> dict[str, torch.Tensor | int]:
        self.states = snapshot.states.detach().clone().to(self.device)
        self.valid = snapshot.valid.detach().clone().to(self.device)
        self.history = snapshot.history.detach().clone().to(self.device)
        self.history_valid = snapshot.history_valid.detach().clone().to(self.device)
        self.reference_index = int(snapshot.reference_index)
        # Generator state is a CPU ByteTensor even when this world executes on
        # CUDA; moving it to the model device breaks ``Generator.set_state``.
        self.generator.set_state(snapshot.motion_generator_state.cpu())
        self.filter_state = (
            None
            if snapshot.filter_global is None
            else FilterState(
                snapshot.filter_global.detach().clone().to(self.device),
                snapshot.filter_agents.detach().clone().to(self.device),
            )
        )
        self.slow_scene = (
            None
            if snapshot.slow_scene is None
            else snapshot.slow_scene.detach().clone().to(self.device)
        )
        self.slow_scene_noise = (
            None
            if snapshot.slow_scene_noise is None
            else snapshot.slow_scene_noise.detach().clone().to(self.device)
        )
        self.agent_noise_state = (
            None
            if snapshot.agent_noise_state is None
            else snapshot.agent_noise_state.detach().clone().to(self.device)
        )
        self.agent_style_state = (
            None
            if snapshot.agent_style_state is None
            else snapshot.agent_style_state.detach().clone().to(self.device)
        )
        self.previous_current = (
            None
            if snapshot.previous_current is None
            else snapshot.previous_current.detach().clone().to(self.device)
        )
        self.committed_ego_controls = snapshot.committed_ego_controls.detach().clone().to(
            self.device
        )
        self.intervention_memory = (
            None
            if snapshot.intervention_memory is None
            else snapshot.intervention_memory.detach().clone().to(self.device)
        )
        self.lateral_intervention_memory = (
            None
            if snapshot.lateral_intervention_memory is None
            else snapshot.lateral_intervention_memory.detach().clone().to(self.device)
        )
        self.response_innovations = (
            None
            if snapshot.response_innovations is None
            else snapshot.response_innovations.detach().clone().to(self.device)
        )
        self.response_agent_innovations = (
            None
            if snapshot.response_agent_innovations is None
            else snapshot.response_agent_innovations.detach().clone().to(self.device)
        )
        return self.observe()

    def _preview(self) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.reference is not None and self.reference_base is not None
        start = self.reference_index
        stop = start + self.model.cfg.preview_frames
        preview = self.reference[:, start:stop]
        if preview.shape[1] < self.model.cfg.preview_frames:
            padding = preview[:, -1:].expand(
                -1, self.model.cfg.preview_frames - preview.shape[1], -1, -1
            )
            preview = torch.cat((preview, padding), dim=1)
        base = self.reference_base if start == 0 else self.reference[:, start - 1]
        return preview, base

    @torch.no_grad()
    def advance_response(
        self, ego_actions: torch.Tensor
    ) -> dict[str, torch.Tensor | int]:
        states, valid = self._require()
        assert self.history is not None and self.history_valid is not None
        actions = torch.as_tensor(ego_actions, dtype=states.dtype, device=self.device)
        if actions.ndim == 2:
            actions = actions[None]
        expected = (states.shape[0], self.model.cfg.execute_frames, 2)
        if actions.shape != expected:
            raise ValueError(f"ego_actions must have shape {expected}")
        preview, base = self._preview()
        refresh = self.slow_scene is None or (
            self.reference_index % self.model.cfg.scene_refresh_responses == 0
        )
        if self.response_innovations is None:
            scene_noise = torch.randn(
                (states.shape[0], self.model.cfg.scene_latent_dim),
                generator=self.generator,
                device=self.device,
                dtype=states.dtype,
            )
            agent_noise = torch.randn(
                (states.shape[0], 7, self.model.cfg.agent_latent_dim),
                generator=self.generator,
                device=self.device,
                dtype=states.dtype,
            )
        else:
            if self.reference_index >= self.response_agent_innovations.shape[1]:
                raise RuntimeError("world exogenous agent innovations are exhausted")
            assert self.response_agent_innovations is not None
            scene_noise = (
                self.response_innovations[:, self.reference_index // self.model.cfg.scene_refresh_responses]
                if refresh
                else torch.zeros(
                    (states.shape[0], self.model.cfg.scene_latent_dim),
                    dtype=states.dtype,
                    device=self.device,
                )
            )
            agent_noise = self.response_agent_innovations[:, self.reference_index]
        assert self.map_polylines is not None and self.map_polyline_valid is not None
        response = self.model(
            self.history,
            self.history_valid,
            states,
            valid,
            preview,
            base,
            self.map_polylines,
            self.map_polyline_valid,
            filter_state=self.filter_state,
            previous_current=self.previous_current,
            slow_scene=self.slow_scene,
            slow_scene_noise=self.slow_scene_noise,
            agent_noise_state=self.agent_noise_state,
            agent_style_state=self.agent_style_state,
            committed_ego_controls=self.committed_ego_controls,
            intervention_memory=self.intervention_memory,
            lateral_intervention_memory=self.lateral_intervention_memory,
            response_index=self.reference_index // self.model.cfg.execute_frames,
            scene_standard_normal=scene_noise,
            agent_standard_normal=agent_noise,
            apply_intervention_adapter=self.controller is None,
            apply_explicit_ego_response=True,
        )
        self.filter_state = response.filter_state
        self.slow_scene = response.slow_scene
        self.slow_scene_noise = response.slow_scene_noise
        self.agent_noise_state = response.agent_noise_state
        self.agent_style_state = response.agent_style_state
        self.intervention_memory = response.intervention_memory
        self.lateral_intervention_memory = response.lateral_intervention_memory
        self.previous_current = states.detach().clone()
        base_actions = response.actions
        controller_output = None
        if self.controller is not None:
            controller_output = self.controller(self._controller_context(response))
            response_actions = controller_output.actions
        else:
            response_actions = base_actions
        frames: list[torch.Tensor] = []
        for frame in range(self.model.cfg.execute_frames):
            controls = torch.cat(
                (actions[:, frame, None], response_actions[:, frame]), dim=1
            )
            states = self.model.dynamics.step(
                states, controls, valid, self.model.cfg.dt_s
            )
            frames.append(states)
        self.committed_ego_controls = torch.cat(
            (self.committed_ego_controls, actions[:, : self.model.cfg.execute_frames]),
            dim=1,
        )[:, -self.model.cfg.intervention_trigger_history_frames - 1 :]
        self.states = states
        new_frames = torch.stack(frames, dim=1)
        frame_valid = valid[:, None].expand(-1, len(frames), -1)
        self.history = torch.cat((self.history, new_frames), dim=1)[
            :, -self.model.cfg.history_frames :
        ]
        self.history_valid = torch.cat((self.history_valid, frame_valid), dim=1)[
            :, -self.model.cfg.history_frames :
        ]
        self.reference_index += self.model.cfg.execute_frames
        result = self.observe()
        result.update(
            {
                "agent_state_frames": new_frames,
                "background_actions": response_actions,
                "base_background_actions": base_actions,
                "response_mean": response.mean,
                "response_std": response.std,
                "rebased_preview": response.rebased_preview,
                "controller_alpha": None if controller_output is None else controller_output.alpha,
                "controller_delta_ax": None if controller_output is None else controller_output.delta_ax,
                "controller_active": None if controller_output is None else controller_output.active,
            }
        )
        return result
