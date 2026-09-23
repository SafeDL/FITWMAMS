"""Closed-loop fixed-PNC adapter for TrafficBots without changing its source tree."""

from __future__ import annotations

from typing import Any

import torch

from interactive_behavior_world_model.evaluation.pnc import (
    fixed_pnc_action,
    initial_pnc_targets,
    structured_bc_pnc_action,
)
from interactive_behavior_world_model.evaluation.rollout import _torch_features
from reproduction.models.trafficbots.data import DT_S
from reproduction.models.trafficbots.rollout import (
    Rollout,
    TrafficBotsHighDRollout,
    _canonical_background,
    _pose_motion_from_external,
)


class TrafficBotsPNCRollout(TrafficBotsHighDRollout):
    """TrafficBots rollout whose ego is a causal fixed 5-Hz PNC.

    This deliberately lives outside ``reproduction/models/trafficbots/rollout.py``:
    that upstream-compatible file may contain user work.  The background
    rollout remains identical to the common-backend TrafficBots adapter.
    """

    @torch.no_grad()
    def run_pnc(
        self,
        batch: dict[str, Any],
        *,
        deterministic: bool,
        pnc_id: str,
        lengths: torch.Tensor,
        widths: torch.Tensor,
        map_polylines: torch.Tensor,
        map_valid: torch.Tensor,
        pnc_history: torch.Tensor,
        pnc_history_valid: torch.Tensor,
        pnc_model: torch.nn.Module | None = None,
        pnc_checkpoint: dict | None = None,
        latent_sample: torch.Tensor | None = None,
        destination_sample: torch.Tensor | None = None,
        steps: int = 149,
    ) -> Rollout:
        device = next(self.module.parameters()).device
        batch = self.module._move(batch, device)
        canonical, canonical_valid = batch["canonical/states"], batch["canonical/valid"]
        if self.require_follower_excluded and bool(canonical_valid[..., 2].any()):
            raise ValueError(
                "TrafficBots evaluation batch must not exclude the common background scope"
            )
        lengths, widths = lengths.to(device), widths.to(device)
        lines, line_valid = map_polylines.to(device), map_valid.to(device).bool()
        history, history_valid = (
            pnc_history.to(device),
            pnc_history_valid.to(device).bool(),
        )
        lane_valid = line_valid.any(-1)
        lane_centers = (lines[..., 1] * line_valid).sum(-1) / line_valid.sum(
            -1
        ).clamp_min(1)
        lane_widths = (lines[..., 4] * line_valid).sum(-1) / line_valid.sum(
            -1
        ).clamp_min(1)

        mp_tokens, tl_tokens = self.module._tokens(batch)
        latent_distribution = self.module._latent(
            batch, mp_tokens, tl_tokens, posterior=False
        )
        latent_valid = batch["agent/valid"][..., 0]
        latent = (
            latent_distribution.sample(deterministic)
            if latent_sample is None
            else latent_sample.to(device)
        )
        destination_distribution = self.module._dest_distribution(batch, mp_tokens)
        destination = (
            destination_distribution.sample(deterministic)
            if destination_sample is None
            else destination_sample.to(device)
        )
        valid = batch["agent/valid"][..., 0].clone()
        # HPTR has one permanently padded internal agent.  PNC geometry uses
        # only the public ego plus six backgrounds, matching its state tensor.
        public_valid = valid[:, :7]
        pose = torch.cat(
            (batch["agent/pos"][..., 0, :2], batch["agent/yaw_bbox"][..., 0, :]), -1
        )
        motion = torch.cat(
            (
                batch["agent/spd"][..., 0, :],
                batch["agent/acc"][..., 0, :],
                batch["agent/yaw_rate"][..., 0, :],
            ),
            -1,
        )
        attributes = torch.cat((batch["agent/size"], batch["agent/type"].float()), -1)
        target_speed, target_lane_y = initial_pnc_targets(
            canonical[:, 0], lane_centers, lane_valid
        )
        ego_state = canonical[:, 0, 0].clone()
        self.module.model.init()
        navi_updated = True
        output, background_actions, requested_background_actions, executed_ego = (
            [],
            [],
            [],
            [],
        )
        ego_control = torch.zeros((len(canonical), 2), device=device)

        for step in range(int(steps)):
            action_distribution, _ = self.module.model(
                valid,
                pose,
                motion,
                attributes,
                batch["agent/type"],
                latent,
                latent_valid,
                destination,
                latent_valid,
                navi_updated,
                batch["tl_stop/state"][:, :, step],
                tl_tokens,
                mp_tokens,
            )
            navi_updated = False
            requested_controls = self.module.plant.process_action(
                action_distribution.mean
            )
            controls = self._common_bounds(requested_controls, motion[..., 0])
            canonical_state = _canonical_background(pose, motion)
            next_canonical = self.external_ego_dynamics.step(
                canonical_state, controls, valid, DT_S
            )
            next_pose, next_motion = _pose_motion_from_external(
                next_canonical, controls
            )

            if step % 5 == 0:
                current = torch.cat((ego_state[:, None], canonical_state[:, 1:7]), 1)
                if pnc_id in {"structured_bc", "structured_bc_lane_stable"}:
                    if pnc_model is None or pnc_checkpoint is None:
                        raise ValueError(
                            "structured_bc PNC requires its frozen checkpoint"
                        )
                    raw_feature = _torch_features(
                        history,
                        history_valid,
                        lengths,
                        widths,
                        lane_centers,
                        lane_widths,
                        lane_valid,
                    )
                    ego_control = structured_bc_pnc_action(
                        pnc_model,
                        pnc_checkpoint,
                        raw_feature,
                        history_valid,
                        current,
                        public_valid,
                        target_lane_y=(
                            target_lane_y
                            if pnc_id == "structured_bc_lane_stable"
                            else None
                        ),
                        learned_yaw_weight=(
                            0.25 if pnc_id == "structured_bc_lane_stable" else 1.0
                        ),
                    )
                else:
                    ego_control = fixed_pnc_action(
                        current,
                        public_valid,
                        lengths,
                        target_speed,
                        target_lane_y,
                        pnc_id,
                        widths=widths,
                    )
            # The same plant and bounds apply to the ego and every background
            # agent.  Its action is recomputed only at the formal 5-Hz clock.
            ego_control = self._common_bounds(
                ego_control, torch.linalg.vector_norm(ego_state[:, 2:4], dim=-1)
            )
            ego_state = self.external_ego_dynamics.step(
                ego_state, ego_control, canonical_valid[:, step, 0], DT_S
            )
            ego_pose, ego_motion = _pose_motion_from_external(ego_state, ego_control)
            next_pose[:, 0], next_motion[:, 0] = ego_pose, ego_motion
            pose, motion = next_pose, next_motion
            current = torch.cat(
                (
                    ego_state[:, None],
                    _canonical_background(pose[:, 1:7], motion[:, 1:7]),
                ),
                1,
            )
            history = torch.cat((history[:, 1:], current[:, None]), 1)
            history_valid = torch.cat((history_valid[:, 1:], public_valid[:, None]), 1)
            output.append(current)
            background_actions.append(controls[:, 1:])
            requested_background_actions.append(requested_controls[:, 1:])
            executed_ego.append(ego_control)
        return Rollout(
            torch.stack(output, 1)[:, :, :7],
            torch.stack(background_actions, 1)[:, :, :6],
            torch.stack(executed_ego, 1),
            reference_actions=torch.stack(requested_background_actions, 1)[:, :, :6],
            latent_sample=latent,
            destination_sample=destination,
        )
