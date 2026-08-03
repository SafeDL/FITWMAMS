"""Standalone RAMP-WM model and closed-loop training/inference rollouts."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import torch
import torch.nn as nn

from world_model.src.core.dynamics import DynamicsConfig, KinematicTrafficDynamics
from world_model.src.core import ContinuousTrafficMemory
from world_model.src.core.initial_behavior_anchor import BehaviorAnchorControlPlan
from world_model.src.relations.relational_encoder import (
    RelationalEncoderConfig,
    RelationalTrafficEncoder,
)
from .config import RAMPConfig
from .joint_plan_decoder import JointPlanDecoder
from .losses import candidate_energy, masked_mean, mixture_loss


class RAMPWorldModel(nn.Module):
    """Continuous-memory, scene-joint candidate planner; no latent state/duration."""

    model_type = "ramp_relational_memory_planning"

    def __init__(self, cfg: RAMPConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = RelationalTrafficEncoder(
            RelationalEncoderConfig(
                hidden_dim=cfg.hidden_dim,
                temporal_layers=cfg.temporal_layers,
                dropout=cfg.dropout,
            )
        )
        self.memory = ContinuousTrafficMemory(cfg.hidden_dim)
        self.decoder = JointPlanDecoder(
            cfg.hidden_dim,
            cfg.num_candidates,
            cfg.jerk_controls,
            cfg.plan_frames,
            (cfg.max_longitudinal_jerk, cfg.max_yaw_jerk),
        )
        self.start_anchor = (
            BehaviorAnchorControlPlan(cfg.plan_frames) if cfg.use_start_anchor else None
        )
        self.dynamics = KinematicTrafficDynamics(
            DynamicsConfig(
                acceleration_min_mps2=cfg.min_acceleration,
                acceleration_max_mps2=cfg.max_acceleration,
            )
        )

    @staticmethod
    def _ego_mask(batch: dict[str, torch.Tensor]) -> torch.Tensor:
        n = batch["agent_states"].shape[2]
        return torch.nn.functional.one_hot(
            batch["ego_index"].long().clamp(0, n - 1), n
        ).bool()

    @staticmethod
    def _lane_candidates(
        states: torch.Tensor,
        polylines: torch.Tensor,
        polyline_valid: torch.Tensor,
        top_r: int = 3,
    ) -> torch.Tensor:
        b, agents, _ = states.shape
        lanes = polylines.shape[1]
        if lanes == 0:
            return torch.empty((b, agents, 0), dtype=torch.long, device=states.device)
        d = (
            (polylines[:, None, :, :, :2] - states[:, :, None, None, :2])
            .square()
            .sum(-1)
        )
        d = d.masked_fill(~polyline_valid[:, None], float("inf")).amin(-1)
        count = min(top_r, lanes)
        choice = d.argsort(-1)[..., :count]
        lane_exists = (
            polyline_valid.any(2)[:, None].expand(-1, agents, -1).gather(2, choice)
        )
        return choice.masked_fill(~lane_exists, -1)

    def encode_step(
        self,
        history: torch.Tensor,
        history_valid: torch.Tensor,
        current: torch.Tensor,
        current_valid: torch.Tensor,
        ego_mask: torch.Tensor,
        batch: dict[str, torch.Tensor],
    ):
        lanes = self._lane_candidates(
            current, batch["map_polylines"], batch["map_polyline_valid"]
        )
        agents, scene = self.encoder(
            history,
            history_valid,
            current,
            current_valid,
            ego_mask,
            batch["map_polylines"],
            batch["map_polyline_valid"],
            lanes,
            batch.get("lane_graph_edges"),
        )
        return agents, scene

    def _integrate_plan(
        self, current: torch.Tensor, plan: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        """Integrate all candidates independently, holding ego outside the decoder."""
        b, candidates, frames, backgrounds, _ = plan.shape
        state = (
            current[:, 1:]
            .unsqueeze(1)
            .expand(-1, candidates, -1, -1)
            .reshape(b * candidates, backgrounds, 6)
        )
        mask = (
            valid[:, 1:]
            .unsqueeze(1)
            .expand(-1, candidates, -1)
            .reshape(b * candidates, backgrounds)
        )
        generated = []
        for frame in range(frames):
            control = plan[:, :, frame].reshape(b * candidates, backgrounds, 2)
            state = self.dynamics.step(state, control, mask, self.cfg.simulation_dt_s)
            generated.append(state.reshape(b, candidates, backgrounds, 6))
        return torch.stack(generated, dim=2)

    def _project_plan_jerk(
        self, plans: torch.Tensor, current: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        """Hard-limit every planned control change from the observed state.

        The nominal decoder and the START anchor both emit acceleration
        curves, while residual candidates are already parameterized by jerk.
        Applying the same projection after their composition enforces the
        stated jerk bounds at every 25 Hz frame *and* across a re-planning
        boundary.  The initial control is reconstructed only from the current
        already-observed background state, so it introduces no future input.
        """
        initial = self.dynamics.controls_from_highd_actions(
            current[:, 1:, 4:6], current[:, 1:]
        )
        previous = initial[:, None].expand(-1, plans.shape[1], -1, -1)
        limit = (
            plans.new_tensor((self.cfg.max_longitudinal_jerk, self.cfg.max_yaw_jerk))
            * self.cfg.simulation_dt_s
        )
        projected: list[torch.Tensor] = []
        for frame in range(plans.shape[2]):
            value = previous + (plans[:, :, frame] - previous).clamp(
                min=-limit, max=limit
            )
            projected.append(value)
            previous = value
        return torch.stack(projected, dim=2) * valid[:, None, None, 1:, None].float()

    def _target_controls(
        self, batch: dict[str, torch.Tensor], response: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        stride, frames = self.cfg.physics_steps_per_response, self.cfg.plan_frames
        actions = batch["actions_highd"]
        available = min(
            (actions.shape[1] // stride) - response, (frames + stride - 1) // stride
        )
        target = actions.new_zeros((actions.shape[0], frames, 6, 2))
        valid = torch.zeros(
            (actions.shape[0], frames, 6), dtype=torch.bool, device=actions.device
        )
        if available:
            highd = (
                actions[:, response * stride : (response + available) * stride]
                .reshape(actions.shape[0], available, stride, 6, 2)
                .mean(2)
            )
            states = batch["agent_states"][
                :,
                24 + response * stride : 24 + (response + available) * stride : stride,
                1:,
            ]
            controls = self.dynamics.controls_from_highd_actions(
                highd, states
            ).repeat_interleave(stride, 1)[:, :frames]
            masks = batch["agent_valid"][
                :,
                24 + response * stride : 24 + response * stride + controls.shape[1],
                1:,
            ]
            target[:, : controls.shape[1]] = controls
            valid[:, : controls.shape[1]] = masks
        return target, valid

    def _target_plan_states(
        self, batch: dict[str, torch.Tensor], response: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        start = 25 + response * self.cfg.physics_steps_per_response
        stop = min(start + self.cfg.plan_frames, batch["agent_states"].shape[1])
        states = batch["agent_states"].new_zeros(
            (batch["agent_states"].shape[0], self.cfg.plan_frames, 7, 6)
        )
        valid = torch.zeros(states.shape[:-1], dtype=torch.bool, device=states.device)
        states[:, : stop - start] = batch["agent_states"][:, start:stop]
        valid[:, : stop - start] = batch["agent_valid"][:, start:stop]
        return states, valid

    def plan_step(
        self,
        history: torch.Tensor,
        history_valid: torch.Tensor,
        current: torch.Tensor,
        current_valid: torch.Tensor,
        ego_mask: torch.Tensor,
        batch: dict[str, torch.Tensor],
        memory: torch.Tensor | None,
        previous_plan: torch.Tensor | None,
        previous_current: torch.Tensor | None = None,
        response_index: int = 0,
    ):
        agents, scene = self.encode_step(
            history, history_valid, current, current_valid, ego_mask, batch
        )
        delta = (
            torch.zeros_like(current)
            if previous_current is None
            else current - previous_current
        )
        memory_next = self.memory(scene, agents, previous_plan, delta, memory)
        plans, probabilities, jerk = self.decoder(
            agents, scene, memory_next, current, current_valid, previous_plan
        )
        # B0 is an explicit START-only input.  It is removed after the first
        # second, so ROLL cannot reach either its raw values or its summary.
        anchor = batch.get("behavior_anchor_raw")
        if self.start_anchor is not None and anchor is not None and response_index < 5:
            anchor_valid = batch.get(
                "behavior_anchor_valid", current_valid[:, 1:]
            ).bool()
            highd = self.start_anchor(
                batch["agent_states"][:, 24], anchor, anchor_valid
            )
            controls = self.dynamics.controls_from_highd_actions(
                highd, batch["agent_states"][:, 24:25, 1:]
            )
            start = response_index * self.cfg.execute_frames
            suffix = controls.new_zeros((controls.shape[0], self.cfg.plan_frames, 6, 2))
            count = min(self.cfg.plan_frames, controls.shape[1] - start)
            if count > 0:
                suffix[:, :count] = controls[:, start : start + count]
            plans = torch.stack(
                (
                    plans[..., 0] + suffix[:, None, ..., 0],
                    plans[..., 1] + suffix[:, None, ..., 1],
                ),
                dim=-1,
            )
            plans = torch.stack(
                (
                    plans[..., 0].clamp(
                        self.cfg.min_acceleration, self.cfg.max_acceleration
                    ),
                    plans[..., 1].clamp(-0.6, 0.6),
                ),
                dim=-1,
            )
        if self.cfg.hard_jerk_projection:
            plans = self._project_plan_jerk(plans, current, current_valid)
        return {
            "candidate_control_plans": plans,
            "candidate_probabilities": probabilities,
            "jerks": jerk,
            "continuous_memory_next": memory_next,
            "agent_context": agents,
            "scene_context": scene,
        }

    def _closed_loop(
        self,
        batch: dict[str, torch.Tensor],
        response_steps: int,
        *,
        deterministic: bool,
        world_uniforms: torch.Tensor | None = None,
        tbptt_steps: int = 0,
        active_candidates: int | None = None,
    ) -> dict[str, Any]:
        states, valid = batch["agent_states"], batch["agent_valid"]
        b = states.shape[0]
        stride = self.cfg.physics_steps_per_response
        ego_mask = self._ego_mask(batch)
        history, history_valid = states[:, :25].clone(), valid[:, :25].clone()
        current, current_valid = history[:, -1], history_valid[:, -1]
        memory = None
        previous_plan = None
        previous_current = None
        previous_scene = None
        pred_frames: list[torch.Tensor] = []
        all_plans = []
        all_plan_states = []
        all_probs = []
        all_selected = []
        all_memory = []
        loss_terms: list[dict[str, torch.Tensor]] = []
        for response in range(response_steps):
            out = self.plan_step(
                history,
                history_valid,
                current,
                current_valid,
                ego_mask,
                batch,
                memory,
                previous_plan,
                previous_current,
                response,
            )
            plans, probs = (
                out["candidate_control_plans"],
                out["candidate_probabilities"],
            )
            active = (
                self.cfg.num_candidates
                if active_candidates is None
                else int(active_candidates)
            )
            plans, probs = plans[:, :active], probs[:, :active]
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
            plan_states = self._integrate_plan(current, plans, current_valid)
            if deterministic:
                selected = probs.argmax(-1)
            elif world_uniforms is not None:
                # ``searchsorted`` treats the last dimension as the sorted
                # candidate axis.  Preserve a singleton candidate dimension
                # for batched inverse-CDF sampling, then remove it again.
                selected = (
                    torch.searchsorted(
                        probs.cumsum(-1),
                        world_uniforms[:, response, None].clamp(0, 1 - 1e-7),
                    )
                    .squeeze(-1)
                    .clamp_max(active - 1)
                )
            else:
                selected = torch.multinomial(probs, 1).squeeze(-1)
            selected_plan = plans[torch.arange(b, device=plans.device), selected]
            target_controls, control_valid = self._target_controls(batch, response)
            target_plan, plan_valid = self._target_plan_states(batch, response)
            candidate_valid = control_valid & plan_valid[:, :, 1:]
            energy = candidate_energy(
                plans, target_controls, plan_states, target_plan, candidate_valid
            )
            mixture, calibration = mixture_loss(
                energy, probs, self.cfg.mixture_temperature
            )
            execute = selected_plan[:, :stride]
            generated = current.clone()
            physical = []
            for step in range(stride):
                all_control = current.new_zeros((b, 7, 2))
                all_control[:, 1:] = execute[:, step]
                generated = self.dynamics.step(
                    generated, all_control, current_valid, self.cfg.simulation_dt_s
                )
                observed_ego = states[:, 25 + response * stride + step, :1]
                observed_valid = valid[:, 25 + response * stride + step, :1]
                generated = torch.cat((observed_ego, generated[:, 1:]), dim=1)
                frame_valid = torch.cat((observed_valid, current_valid[:, 1:]), dim=1)
                physical.append(generated)
                current_valid = frame_valid
            predicted = torch.stack(physical, 1)
            target_prefix = states[
                :, 25 + response * stride : 25 + (response + 1) * stride
            ]
            target_valid = valid[
                :, 25 + response * stride : 25 + (response + 1) * stride, 1:
            ]
            prefix_position = masked_mean(
                (predicted[:, :, 1:, :2] - target_prefix[:, :, 1:, :2]).abs().mean(-1),
                target_valid,
            )
            prefix_velocity = masked_mean(
                (predicted[:, :, 1:, 2:4] - target_prefix[:, :, 1:, 2:4])
                .abs()
                .mean(-1),
                target_valid,
            )
            prefix_control = masked_mean(
                (selected_plan[:, :stride] - target_controls[:, :stride])
                .abs()
                .mean(-1),
                control_valid[:, :stride],
            )
            overlap = plans.new_zeros(())
            if previous_plan is not None and previous_scene is not None:
                overlap_frames = min(
                    self.cfg.plan_frames - stride, self.cfg.plan_frames
                )
                relation_change = (
                    (out["scene_context"] - previous_scene).norm(dim=-1).detach()
                )
                weight = torch.exp(-relation_change).mean()
                overlap = (
                    previous_plan[:, stride : stride + overlap_frames]
                    - selected_plan[:, :overlap_frames]
                ).abs().mean() * weight
            # Stage A deliberately exposes only candidate 0.  ``std`` over
            # an empty residual-candidate axis is NaN, so key this diagnostic
            # off the active plan tensor rather than architectural capacity.
            diversity = (
                torch.relu(0.04 - plan_states[:, 1:, :, :, :2].std(dim=1).mean())
                if plans.shape[1] > 1
                else plans.new_zeros(())
            )
            smooth = (selected_plan[:, 1:] - selected_plan[:, :-1]).abs().mean()
            # Explicit scene-joint supervision: candidates must preserve
            # pairwise future geometry, not merely fit independent controls.
            agent_count = plan_states.shape[3]
            pair = torch.triu(
                torch.ones(
                    (agent_count, agent_count), device=plans.device, dtype=torch.bool
                ),
                diagonal=1,
            )
            pair_valid = (
                candidate_valid[:, :, :, None]
                & candidate_valid[:, :, None, :]
                & pair[None, None]
            )
            predicted_relative = (
                plan_states[..., :, None, :4] - plan_states[..., None, :, :4]
            )
            target_relative = target_plan[:, None, :, 1:, None, :4] - target_plan[
                :, None, :, 1:, None, :4
            ].transpose(3, 4)
            # target_relative is [B,1,T,6,6,4], matching candidate axis by broadcast.
            joint = (
                predicted_relative.sub(target_relative).abs().mean(dim=-1)
                * pair_valid[:, None].float()
            ).sum()
            joint = joint / (
                pair_valid.float().sum().clamp_min(1.0) * max(plans.shape[1], 1)
            )
            loss_terms.append(
                {
                    "prefix_position": prefix_position,
                    "prefix_velocity": prefix_velocity,
                    "prefix_control": prefix_control,
                    "mixture": mixture,
                    "probability": calibration,
                    "overlap": overlap,
                    "diversity": diversity,
                    "smooth": smooth,
                    "joint": joint,
                    "energy": energy.mean(),
                }
            )
            pred_frames.extend(physical)
            all_plans.append(plans)
            all_plan_states.append(plan_states)
            all_probs.append(probs)
            all_selected.append(selected)
            all_memory.append(out["continuous_memory_next"])
            previous_plan, previous_current, previous_scene = (
                selected_plan,
                current,
                out["scene_context"],
            )
            current = predicted[:, -1]
            history = torch.cat((history[:, stride:], predicted), 1)
            history_valid = torch.cat(
                (
                    history_valid[:, stride:],
                    current_valid[:, None].expand(-1, stride, -1),
                ),
                1,
            )
            memory = out["continuous_memory_next"]
            if (
                tbptt_steps
                and (response + 1) % tbptt_steps == 0
                and response + 1 < response_steps
            ):
                history, current, memory, previous_plan = (
                    history.detach(),
                    current.detach(),
                    memory.detach(),
                    previous_plan.detach(),
                )
        return {
            "predicted_states": torch.stack(pred_frames, 1),
            "target_states": states[:, 25 : 25 + response_steps * stride],
            "target_valid": valid[:, 25 : 25 + response_steps * stride],
            "candidate_control_plans": torch.stack(all_plans, 1),
            "predicted_candidate_states": torch.stack(all_plan_states, 1),
            "candidate_probabilities": torch.stack(all_probs, 1),
            "selected_candidate_index": torch.stack(all_selected, 1),
            "continuous_memory": torch.stack(all_memory, 1),
            "loss_terms": loss_terms,
        }

    def forward_training(
        self,
        batch: dict[str, torch.Tensor],
        *,
        response_steps: int | None = None,
        tbptt_steps: int = 5,
        active_candidates: int | None = None,
    ) -> dict[str, torch.Tensor]:
        rollout = self._closed_loop(
            batch,
            response_steps or self.cfg.response_steps,
            deterministic=True,
            tbptt_steps=tbptt_steps,
            active_candidates=active_candidates,
        )
        means = {
            key: torch.stack([item[key] for item in rollout["loss_terms"]]).mean()
            for key in rollout["loss_terms"][0]
        }
        total = (
            self.cfg.position_weight * means["prefix_position"]
            + self.cfg.velocity_weight * means["prefix_velocity"]
            + self.cfg.control_weight * means["prefix_control"]
            + self.cfg.mixture_weight * means["mixture"]
            + self.cfg.probability_weight * means["probability"]
            + self.cfg.overlap_weight * means["overlap"]
            + self.cfg.diversity_weight * means["diversity"]
            + self.cfg.smoothness_weight * means["smooth"]
            + self.cfg.joint_weight * means["joint"]
        )
        return {
            "loss": total,
            **means,
            **{key: value for key, value in rollout.items() if key != "loss_terms"},
        }

    @torch.no_grad()
    def rollout_roll_mode(
        self,
        batch: dict[str, torch.Tensor],
        *,
        seed: int = 123,
        deterministic: bool = True,
    ) -> dict[str, Any]:
        generator = torch.Generator(device=batch["agent_states"].device).manual_seed(
            int(seed)
        )
        uniforms = (
            None
            if deterministic
            else torch.rand(
                (batch["agent_states"].shape[0], self.cfg.response_steps),
                generator=generator,
                device=batch["agent_states"].device,
            )
        )
        return self._closed_loop(
            batch,
            self.cfg.response_steps,
            deterministic=deterministic,
            world_uniforms=uniforms,
        )

    def checkpoint_payload(self) -> dict[str, Any]:
        return {
            "model_type": self.model_type,
            "model_config": asdict(self.cfg),
            "state_dict": self.state_dict(),
        }
