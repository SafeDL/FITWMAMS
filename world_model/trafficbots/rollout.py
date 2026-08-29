"""Causal TrafficBots rollout with a common external ego plant."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from world_model.src.core.dynamics import KinematicTrafficDynamics

from .data import DT_S


@dataclass(frozen=True)
class Rollout:
    states: torch.Tensor                 # [B,149,7,6]
    background_actions: torch.Tensor     # [B,149,6,2]
    ego_actions: torch.Tensor            # [B,149,2]
    reference_actions: None = None
    latent_sample: torch.Tensor | None = None
    destination_sample: torch.Tensor | None = None


def logged_ego_controls(states: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Recover causal controls from adjacent logged ego states for external replay."""
    source, target = states[:, :-1, 0], states[:, 1:, 0]
    speed, next_speed = torch.linalg.vector_norm(source[..., 2:4], dim=-1), torch.linalg.vector_norm(target[..., 2:4], dim=-1)
    # ``atan2`` already handles a zero longitudinal component.  Clamping it
    # would incorrectly flip headings for the opposite carriageway.
    heading = torch.atan2(source[..., 3], source[..., 2])
    next_heading = torch.atan2(target[..., 3], target[..., 2])
    yaw_rate = torch.atan2(torch.sin(next_heading - heading), torch.cos(next_heading - heading)) / DT_S
    controls = torch.stack(((next_speed - speed) / DT_S, yaw_rate), -1)
    controls[~(valid[:, :-1, 0] & valid[:, 1:, 0])] = 0.0
    return controls


def _pose_motion_from_external(state: torch.Tensor, control: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    speed = torch.linalg.vector_norm(state[..., 2:4], dim=-1)
    yaw = torch.atan2(state[..., 3], state[..., 2])
    pose = torch.cat((state[..., :2], yaw.unsqueeze(-1)), -1)
    motion = torch.stack((speed, control[..., 0], control[..., 1]), -1)
    return pose, motion


def _canonical_background(pose: torch.Tensor, motion: torch.Tensor) -> torch.Tensor:
    speed, acceleration, yaw_rate = motion.unbind(-1)
    yaw = pose[..., 2]
    velocity = torch.stack((speed * torch.cos(yaw), speed * torch.sin(yaw)), -1)
    cart_acc = torch.stack((acceleration * torch.cos(yaw) - speed * yaw_rate * torch.sin(yaw), acceleration * torch.sin(yaw) + speed * yaw_rate * torch.cos(yaw)), -1)
    return torch.cat((pose[..., :2], velocity, cart_acc), -1)


class TrafficBotsHighDRollout:
    """Runs deterministic, stochastic, oracle, and paired-CRN trajectories."""
    def __init__(self, module) -> None:
        self.module = module
        self.external_ego_dynamics = KinematicTrafficDynamics()

    @torch.no_grad()
    def run(
        self,
        batch: dict[str, Any],
        *,
        deterministic: bool,
        oracle: bool = False,
        ego_controls: torch.Tensor | None = None,
        latent_sample: torch.Tensor | None = None,
        destination_sample: torch.Tensor | None = None,
    ) -> Rollout:
        device = next(self.module.parameters()).device
        batch = self.module._move(batch, device)
        canonical, canonical_valid = batch["canonical/states"], batch["canonical/valid"]
        if bool(canonical_valid[..., 2].any()):
            raise ValueError(
                "TrafficBots evaluation batch must apply the follower-excluded "
                "scope before rollout (same_rear/canonical agent 2 is still valid)"
            )
        if ego_controls is None:
            ego_controls = logged_ego_controls(canonical, canonical_valid)
        else:
            ego_controls = ego_controls.to(device)
        mp_tokens, tl_tokens = self.module._tokens(batch)
        latent_distribution = self.module._latent(batch, mp_tokens, tl_tokens, posterior=oracle)
        latent_valid = batch["agent/valid"][..., 0]
        latent = latent_distribution.sample(deterministic) if latent_sample is None else latent_sample.to(device)
        destination_distribution = self.module._dest_distribution(batch, mp_tokens)
        destination = batch["agent/dest"] if oracle else destination_distribution.sample(deterministic)
        if destination_sample is not None:
            destination = destination_sample.to(device)
        valid = batch["agent/valid"][..., 0].clone()
        pose = torch.cat((batch["agent/pos"][..., 0, :2], batch["agent/yaw_bbox"][..., 0, :]), -1)
        motion = torch.cat((batch["agent/spd"][..., 0, :], batch["agent/acc"][..., 0, :], batch["agent/yaw_rate"][..., 0, :]), -1)
        attributes = torch.cat((batch["agent/size"], batch["agent/type"].float()), -1)
        external_ego = canonical[:, 0, 0].clone()
        self.module.model.init(); navi_updated = True
        output, background_actions, executed_ego = [], [], []
        for step in range(149):
            action_distribution, _ = self.module.model(valid, pose, motion, attributes, batch["agent/type"], latent, latent_valid, destination, latent_valid, navi_updated, batch["tl_stop/state"][:, :, step], tl_tokens, mp_tokens)
            navi_updated = False
            controls = self.module.plant.process_action(action_distribution.mean)
            next_pose, next_motion = self.module.plant.update(pose, motion, controls)
            external_ego = self.external_ego_dynamics.step(external_ego, ego_controls[:, step], canonical_valid[:, step, 0], DT_S)
            ego_pose, ego_motion = _pose_motion_from_external(external_ego, ego_controls[:, step])
            next_pose[:, 0], next_motion[:, 0] = ego_pose, ego_motion
            pose, motion = next_pose, next_motion
            current = torch.cat((external_ego[:, None], _canonical_background(pose[:, 1:], motion[:, 1:])), 1)
            output.append(current); background_actions.append(controls[:, 1:]); executed_ego.append(ego_controls[:, step])
        # HPTR retains one invalid padded agent internally.  The public highD
        # protocol is exactly ego plus the six canonical background slots.
        return Rollout(
            torch.stack(output, 1)[:, :, :7],
            torch.stack(background_actions, 1)[:, :, :6],
            torch.stack(executed_ego, 1),
            latent_sample=latent,
            destination_sample=destination,
        )
