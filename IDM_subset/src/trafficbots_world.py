"""TrafficBots background policy executed with an IDM ego in HighwayEnv."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from hierarchical_world_model.src.highway import HighwayEnvTraffic
from normalizing_flow.src.features import SLOT_NAMES, feature_index
from normalizing_flow.src.sampling import load_checkpoint_and_dataset
from world_model.trafficbots.data import adapt_highd_batch
from world_model.trafficbots.evaluation import load_checkpoint as load_trafficbots_checkpoint
from world_model.src.core.evaluation_scope import (
    evaluation_scope_contract,
    scoped_slot_mask,
)

from .trafficbots_randomness import TrafficBotsExogenousState


TRAFFICBOTS_IDM_DYNAMICS_CONTRACT = "kinematic_unicycle_on_highwayenv_road"


@dataclass(frozen=True)
class TrafficBotsInitialWorld:
    """A realized common-prior S0 and its TrafficBots map representation."""

    initial_states: np.ndarray
    initial_valid: np.ndarray
    map_polylines: np.ndarray
    map_polyline_valid: np.ndarray
    exogenous_state: TrafficBotsExogenousState


@dataclass(frozen=True)
class TrafficBotsIDMRollout:
    states: np.ndarray
    initial_valid: np.ndarray
    ego_actions: np.ndarray
    background_actions: np.ndarray


def _initial_states_from_c0(c0: np.ndarray, slot_mask: np.ndarray) -> np.ndarray:
    """Restore `[batch,7,6]` states without importing diffusion code."""
    values = np.asarray(c0, np.float32)
    slots = np.asarray(slot_mask, bool)
    if values.ndim != 2 or values.shape[1] != 40 or slots.shape != (len(values), 6):
        raise ValueError("C0/slot mask must have shapes [batch,40]/[batch,6]")
    states = np.zeros((len(values), 7, 6), np.float32)
    states[:, 0, 2:] = values[
        :,
        [
            feature_index(None, "ego_vx_mps"),
            feature_index(None, "ego_vy_left_mps"),
            feature_index(None, "ego_ax_mps2"),
            feature_index(None, "ego_ay_left_mps2"),
        ],
    ]
    for slot, name in enumerate(SLOT_NAMES):
        active = slots[:, slot]
        states[active, slot + 1] = np.stack(
            (
                values[active, feature_index(name, "rel_x_m")],
                values[active, feature_index(name, "rel_y_left_m")],
                states[active, 0, 2]
                + values[active, feature_index(name, "rel_vx_mps")],
                states[active, 0, 3]
                + values[active, feature_index(name, "rel_vy_left_mps")],
                values[active, feature_index(name, "other_ax_mps2")],
                values[active, feature_index(name, "other_ay_left_mps2")],
            ),
            axis=-1,
        )
    return states


def _straight_lane_map(initial_states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    batch = len(initial_states)
    x = np.linspace(-200.0, 200.0, 8, dtype=np.float32)
    offsets = np.arange(-3, 5, dtype=np.float32) * 3.6
    polylines = np.zeros((batch, 8, 8, 6), np.float32)
    polylines[..., 0] = x[None, None]
    polylines[..., 1] = initial_states[:, None, 0, 1, None] + offsets[None, :, None]
    polylines[..., 2] = 1.0
    polylines[..., 4] = 3.6
    return polylines, np.ones((batch, 8, 8), bool)


class TrafficBotsInitialSampler:
    """Use the common Flow only as an external `(M,C0)` benchmark source."""

    def __init__(
        self,
        *,
        flow_checkpoint: str | Path,
        flow_output_dir: str | Path,
        repo_root: str | Path,
        device: str | torch.device,
    ) -> None:
        self.device = torch.device(device)
        self.flow, _, _, _ = load_checkpoint_and_dataset(
            flow_checkpoint,
            flow_output_dir,
            repo_root=repo_root,
            device=self.device,
        )
        self.flow.eval()
        self.evaluation_scope = evaluation_scope_contract()

    @torch.no_grad()
    def compose(
        self, exogenous_state: TrafficBotsExogenousState
    ) -> TrafficBotsInitialWorld:
        exogenous_state.validate()
        c0, slot_mask = self.flow.sample_initial_conditions_from_base_randomness(
            exogenous_state.scenario_uniform,
            exogenous_state.c0_base_latent,
        )
        slot_mask = np.asarray(scoped_slot_mask(slot_mask), bool)
        states = _initial_states_from_c0(c0, slot_mask)
        valid = np.concatenate(
            (np.ones((len(states), 1), bool), np.asarray(slot_mask, bool)),
            axis=1,
        )
        polylines, map_valid = _straight_lane_map(states)
        return TrafficBotsInitialWorld(
            initial_states=states,
            initial_valid=valid,
            map_polylines=polylines,
            map_polyline_valid=map_valid,
            exogenous_state=exogenous_state,
        )


def _realized_pose_motion(
    states: torch.Tensor,
    previous_yaw: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    velocity = states[..., 2:4]
    speed = torch.linalg.vector_norm(velocity, dim=-1)
    measured_yaw = torch.atan2(velocity[..., 1], velocity[..., 0])
    if previous_yaw is None:
        yaw = torch.where(speed >= 1.0, measured_yaw, torch.zeros_like(measured_yaw))
    else:
        yaw = torch.where(speed >= 1.0, measured_yaw, previous_yaw)
    acceleration_xy = states[..., 4:6]
    longitudinal = (
        acceleration_xy[..., 0] * torch.cos(yaw)
        + acceleration_xy[..., 1] * torch.sin(yaw)
    )
    lateral = (
        -acceleration_xy[..., 0] * torch.sin(yaw)
        + acceleration_xy[..., 1] * torch.cos(yaw)
    )
    yaw_rate = lateral / speed.clamp_min(0.5)
    pose = torch.cat((states[..., :2], yaw.unsqueeze(-1)), dim=-1)
    motion = torch.stack((speed, longitudinal, yaw_rate), dim=-1)
    return pose, motion


def _categorical_from_uniform(
    probabilities: torch.Tensor,
    uniform: np.ndarray,
    valid: torch.Tensor,
) -> torch.Tensor:
    draw = torch.as_tensor(uniform, dtype=probabilities.dtype, device=probabilities.device)
    if tuple(draw.shape) != tuple(probabilities.shape[:-1]):
        raise ValueError("destination uniform does not match predictor batch/agent shape")
    cumulative = probabilities.cumsum(dim=-1)
    cumulative[..., -1] = 1.0
    destination = (draw.unsqueeze(-1) > cumulative).sum(dim=-1)
    destination = destination.clamp_max(probabilities.shape[-1] - 1).long()
    destination = torch.where(valid, destination, torch.zeros_like(destination))
    return destination


class TrafficBotsHighwayEnvWorld:
    """Vectorized TrafficBots inference with per-scene HighwayEnv plants."""

    def __init__(
        self,
        module: Any,
        *,
        idm_config: dict[str, Any],
        device: str | torch.device,
    ) -> None:
        self.device = torch.device(device)
        self.module = module.to(self.device).eval()
        self.idm_config = dict(idm_config)
        self.traffic: list[HighwayEnvTraffic] = []
        self.batch: dict[str, Any] | None = None
        self.pose: torch.Tensor | None = None
        self.motion: torch.Tensor | None = None
        self.attributes: torch.Tensor | None = None
        self.mp_tokens: dict[str, torch.Tensor] | None = None
        self.tl_tokens: dict[str, torch.Tensor] | None = None
        self.latent: torch.Tensor | None = None
        self.latent_valid: torch.Tensor | None = None
        self.destination: torch.Tensor | None = None
        self.valid: torch.Tensor | None = None
        self.states: torch.Tensor | None = None
        self.step_index = 0

    @torch.no_grad()
    def reset(self, sample: TrafficBotsInitialWorld) -> torch.Tensor:
        states = np.asarray(sample.initial_states, np.float32)
        valid = np.asarray(sample.initial_valid, bool)
        if states.shape != (len(states), 7, 6) or valid.shape != (len(states), 7):
            raise ValueError("initial TrafficBots world must be [batch,7,6]/[batch,7]")
        pseudo_states = np.repeat(states[:, None], 150, axis=1)
        pseudo_valid = np.repeat(valid[:, None], 150, axis=1)
        adapted = adapt_highd_batch(
            pseudo_states,
            pseudo_valid,
            sample.map_polylines,
            sample.map_polyline_valid,
        )
        self.batch = self.module._move(
            {name: torch.from_numpy(value.copy()) for name, value in adapted.items()},
            self.device,
        )
        self.mp_tokens, self.tl_tokens = self.module._tokens(self.batch)
        self.latent_valid = self.batch["agent/valid"][..., 0]
        latent = torch.as_tensor(
            sample.exogenous_state.personality_latent,
            dtype=torch.float32,
            device=self.device,
        )
        if latent.shape[:2] != self.latent_valid.shape:
            raise ValueError("personality latent does not match padded agent slots")
        self.latent = latent
        destination_distribution = self.module._dest_distribution(
            self.batch, self.mp_tokens
        )
        self.destination = _categorical_from_uniform(
            destination_distribution.distribution.probs,
            sample.exogenous_state.destination_uniform,
            self.latent_valid,
        )
        self.valid = self.latent_valid.clone()
        self.attributes = torch.cat(
            (self.batch["agent/size"], self.batch["agent/type"].float()), dim=-1
        )
        self.traffic = []
        realized = []
        for index in range(len(states)):
            traffic = HighwayEnvTraffic(dt_s=0.04, seed=index)
            realized.append(
                traffic.reset(states[index], valid[index], idm_config=self.idm_config)
            )
            self.traffic.append(traffic)
        realized_tensor = torch.from_numpy(np.stack(realized)).to(self.device)
        padded = torch.zeros((len(states), 8, 6), dtype=torch.float32, device=self.device)
        padded[:, :7] = realized_tensor
        self.pose, self.motion = _realized_pose_motion(padded)
        self.states = realized_tensor
        self.step_index = 0
        self.module.model.init()
        return self.states.detach().clone()

    @torch.no_grad()
    def step(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if any(
            value is None
            for value in (
                self.batch,
                self.pose,
                self.motion,
                self.attributes,
                self.mp_tokens,
                self.tl_tokens,
                self.latent,
                self.latent_valid,
                self.destination,
                self.valid,
            )
        ):
            raise RuntimeError("reset the TrafficBots HighwayEnv world before stepping")
        if self.step_index >= 149:
            raise RuntimeError("TrafficBots IDM rollout horizon is exhausted")
        assert self.batch is not None and self.pose is not None and self.motion is not None
        assert self.attributes is not None and self.mp_tokens is not None
        assert self.tl_tokens is not None and self.latent is not None
        assert self.latent_valid is not None and self.destination is not None
        assert self.valid is not None
        action_distribution, _ = self.module.model(
            self.valid,
            self.pose,
            self.motion,
            self.attributes,
            self.batch["agent/type"],
            self.latent,
            self.latent_valid,
            self.destination,
            self.latent_valid,
            self.step_index == 0,
            self.batch["tl_stop/state"][:, :, self.step_index],
            self.tl_tokens,
            self.mp_tokens,
        )
        controls = self.module.plant.process_action(action_distribution.mean)
        realized: list[np.ndarray] = []
        ego_actions: list[np.ndarray] = []
        background_actions: list[np.ndarray] = []
        for index, traffic in enumerate(self.traffic):
            transition = traffic.step(
                controls[index, 1:7].detach().cpu().numpy(), ego_action=None
            )
            realized.append(transition.states)
            ego_actions.append(transition.ego_action)
            background_actions.append(transition.background_actions)
        self.states = torch.from_numpy(np.stack(realized)).to(self.device)
        padded = torch.zeros(
            (len(realized), 8, 6), dtype=self.states.dtype, device=self.device
        )
        padded[:, :7] = self.states
        self.pose, self.motion = _realized_pose_motion(padded, self.pose[..., 2])
        self.step_index += 1
        return (
            self.states.detach().clone(),
            torch.from_numpy(np.stack(ego_actions)).to(self.device),
            torch.from_numpy(np.stack(background_actions)).to(self.device),
        )


def build_trafficbots_idm_world(
    *,
    config: dict[str, Any],
    checkpoint: str | Path,
    idm_config: dict[str, Any],
    device: str | torch.device,
) -> TrafficBotsHighwayEnvWorld:
    module = load_trafficbots_checkpoint(config, checkpoint)
    return TrafficBotsHighwayEnvWorld(
        module, idm_config=idm_config, device=device
    )


@torch.no_grad()
def rollout_trafficbots_idm(
    initial_sampler: TrafficBotsInitialSampler,
    world: TrafficBotsHighwayEnvWorld,
    exogenous_state: TrafficBotsExogenousState,
    *,
    steps: int = 149,
) -> TrafficBotsIDMRollout:
    sample = initial_sampler.compose(exogenous_state)
    initial = world.reset(sample)
    state_frames = [initial.cpu().numpy()]
    ego_actions: list[np.ndarray] = []
    background_actions: list[np.ndarray] = []
    for _ in range(int(steps)):
        states, ego, background = world.step()
        state_frames.append(states.cpu().numpy())
        ego_actions.append(ego.cpu().numpy())
        background_actions.append(background.cpu().numpy())
    return TrafficBotsIDMRollout(
        states=np.stack(state_frames, axis=1).astype(np.float32),
        initial_valid=sample.initial_valid.copy(),
        ego_actions=np.stack(ego_actions, axis=1).astype(np.float32),
        background_actions=np.stack(background_actions, axis=1).astype(np.float32),
    )
