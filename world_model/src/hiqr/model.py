"""HiQR-WM: Flow-initialized hierarchical interaction world model."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as functional

from world_model.src.core.dynamics import DynamicsConfig, KinematicTrafficDynamics
from world_model.src.core.initial_behavior_anchor import start_state_from_flow_tensor

from .config import HiQRWorldModelConfig
from .decoder import AdaptiveJointPlanContinuationDecoder
from .encoder import UnifiedRelationalQueryEncoder
from .interaction_state import HierarchicalStochasticInteractionState, masked_mean

BUFFER_MASK_NAMES = ("carried", "appended", "refinable", "valid")


class HierarchicalInteractionQueryRefineWorldModel(nn.Module):
    """HiQR-WM with a single persistent hierarchical interaction state."""

    model_type = "hierarchical_interaction_query_refine_world_model"

    def __init__(self, cfg: HiQRWorldModelConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or HiQRWorldModelConfig()
        self.encoder = UnifiedRelationalQueryEncoder(self.cfg)
        self.interaction_state = HierarchicalStochasticInteractionState(self.cfg)
        self.decoder = AdaptiveJointPlanContinuationDecoder(self.cfg)
        self.dynamics = KinematicTrafficDynamics(
            DynamicsConfig(
                acceleration_min_mps2=self.cfg.min_acceleration,
                acceleration_max_mps2=self.cfg.max_acceleration,
            )
        )
        self.flow_schema_sha256: str | None = None

    @staticmethod
    def _ego_mask(batch: dict[str, torch.Tensor]) -> torch.Tensor:
        agents = int(batch["agent_states"].shape[2])
        return functional.one_hot(
            batch["ego_index"].long().clamp(0, agents - 1), agents
        ).bool()

    @staticmethod
    def flow_condition_to_scene(
        flow_condition: torch.Tensor,
        slot_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode frozen Flow ``C0+B0`` coordinates without changing the contract."""
        return start_state_from_flow_tensor(flow_condition, slot_valid)[:3]

    def initialize_start(
        self,
        current: torch.Tensor,
        current_valid: torch.Tensor,
        ego_mask: torch.Tensor,
        map_polylines: torch.Tensor,
        map_polyline_valid: torch.Tensor,
        raw_b0: torch.Tensor,
        behavior_anchor_valid: torch.Tensor,
        primary_slot_index: torch.Tensor,
    ) -> torch.Tensor:
        """Create h0; B0 does not appear in the plan decoder or ROLL API."""
        _, scene, _, _ = self.encoder(
            None,
            None,
            current,
            current_valid,
            ego_mask,
            map_polylines,
            map_polyline_valid,
            mode="start",
        )
        event_slot_valid = current_valid[:, 1:].bool()
        anchor_valid = behavior_anchor_valid.bool()
        if anchor_valid.shape != event_slot_valid.shape:
            raise ValueError("behavior_anchor_valid must have shape [batch, 6]")
        if torch.any(anchor_valid & ~event_slot_valid):
            raise ValueError("behavior_anchor_valid cannot exceed current valid slots")
        primary = primary_slot_index.long()
        if primary.shape != (current.shape[0],):
            raise ValueError("primary_slot_index must have shape [batch]")
        if (
            torch.any(primary < 0)
            or torch.any(primary >= 6)
            or torch.any(
                ~event_slot_valid[
                    torch.arange(
                        event_slot_valid.shape[0], device=current_valid.device
                    ),
                    primary,
                ]
            )
        ):
            raise ValueError("primary_slot_index must identify a valid event slot")
        return self.interaction_state.initialize(
            scene, raw_b0, anchor_valid, event_slot_valid, primary
        )

    def _integrate_background_actions(
        self, current: torch.Tensor, actions: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        state = current[:, 1:]
        values: list[torch.Tensor] = []
        for frame in range(actions.shape[1]):
            state = self.dynamics.step(
                state, actions[:, frame], valid[:, 1:], self.cfg.simulation_dt_s
            )
            values.append(state)
        return torch.stack(values, dim=1)

    def plan_step(
        self,
        history: torch.Tensor | None,
        history_valid: torch.Tensor | None,
        current: torch.Tensor,
        current_valid: torch.Tensor,
        ego_mask: torch.Tensor,
        map_polylines: torch.Tensor,
        map_polyline_valid: torch.Tensor,
        *,
        interaction_state: torch.Tensor,
        previous_buffer: torch.Tensor | None = None,
        previous_current: torch.Tensor | None = None,
        deterministic: bool = True,
        use_posterior: bool = False,
        posterior_future: torch.Tensor | None = None,
        posterior_future_valid: torch.Tensor | None = None,
        scene_standard_normal: torch.Tensor | None = None,
        agent_standard_normal: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Plan one causal response from a realized prefix and h_k."""
        mode = "start" if previous_buffer is None else "roll"
        agents, scene, _, _ = self.encoder(
            history,
            history_valid,
            current,
            current_valid,
            ego_mask,
            map_polylines,
            map_polyline_valid,
            mode=mode,
        )
        delta = (
            torch.zeros_like(current)
            if previous_current is None
            else current - previous_current
        )
        g, z, terms = self.interaction_state.sample(
            agents,
            scene,
            interaction_state,
            current,
            current_valid,
            deterministic=deterministic,
            use_posterior=use_posterior,
            posterior_future=posterior_future,
            posterior_future_valid=posterior_future_valid,
            scene_standard_normal=scene_standard_normal,
            agent_standard_normal=agent_standard_normal,
        )
        next_state = self.interaction_state.update(
            interaction_state, scene, g, z, delta, current_valid
        )
        decoded = self.decoder(agents, next_state, g, z, current_valid, previous_buffer)
        actions = decoded["background_future_actions"]
        predicted_states = self._integrate_background_actions(
            current, actions, current_valid
        )
        return {
            **decoded,
            "interaction_state": next_state,
            "scene_latent": g,
            "agent_residual": z,
            "hierarchical_terms": terms,
            "background_future_states": predicted_states,
        }

    def _target_plan(
        self, batch: dict[str, torch.Tensor], response: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        states, valid, recorded = (
            batch["agent_states"],
            batch["agent_valid"],
            batch["actions_highd"],
        )
        start, horizon = int(response) * int(self.cfg.execute_frames), int(
            self.cfg.plan_frames
        )
        anchor = self.cfg.anchor_state_index
        first_future = self.cfg.first_future_state_index
        count = min(horizon, int(recorded.shape[1]) - start)
        batch_size, agents = states.shape[0], states.shape[2]
        target = states.new_zeros((batch_size, horizon, agents, 6))
        target_valid = torch.zeros(
            (batch_size, horizon, agents), dtype=torch.bool, device=states.device
        )
        actions = states.new_zeros((batch_size, horizon, agents - 1, 2))
        action_valid = torch.zeros(
            (batch_size, horizon, agents - 1), dtype=torch.bool, device=states.device
        )
        if count > 0:
            target[:, :count] = states[
                :, first_future + start : first_future + start + count
            ]
            target_valid[:, :count] = valid[
                :, first_future + start : first_future + start + count
            ]
            source_current = states[:, anchor + start : anchor + start + count, 1:]
            actions[:, :count] = self.dynamics.controls_from_highd_actions(
                recorded[:, start : start + count], source_current
            )
            action_valid[:, :count] = valid[
                :, first_future + start : first_future + start + count, 1:
            ]
        return target, target_valid, actions, action_valid

    @staticmethod
    def _interaction_loss(
        predicted: torch.Tensor, target: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        predicted_bg, target_bg = predicted[:, :, 1:], target[:, :, 1:]
        ego, target_ego = predicted[:, :, :1], target[:, :, :1]
        gap = (
            (predicted_bg[..., 0] - ego[..., 0]).abs()
            - (target_bg[..., 0] - target_ego[..., 0]).abs()
        ).abs()
        velocity = (
            (predicted_bg[..., 2] - ego[..., 2])
            - (target_bg[..., 2] - target_ego[..., 2])
        ).abs()
        return masked_mean(gap + 0.25 * velocity, valid)

    @staticmethod
    def _physical_loss(predicted: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        background, ego = predicted[:, :, 1:], predicted[:, :, :1]
        dx, dy = (background[..., 0] - ego[..., 0]).abs(), (
            background[..., 1] - ego[..., 1]
        ).abs()
        ego_barrier = functional.softplus((4.5 - dx) / 0.75) * functional.softplus(
            (1.0 - dy) / 0.25
        )
        pair_dx = (background[..., None, :, 0] - background[..., :, None, 0]).abs()
        pair_dy = (background[..., None, :, 1] - background[..., :, None, 1]).abs()
        pair_barrier = functional.softplus(
            (4.5 - pair_dx) / 0.75
        ) * functional.softplus((1.0 - pair_dy) / 0.25)
        count = background.shape[2]
        upper = torch.triu(
            torch.ones((count, count), dtype=torch.bool, device=background.device),
            diagonal=1,
        )
        pair_valid = valid[..., :, None] & valid[..., None, :] & upper
        return masked_mean(ego_barrier, valid) + masked_mean(pair_barrier, pair_valid)

    @staticmethod
    def _gap_ttc_loss(
        predicted: torch.Tensor, target: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        background, ego = predicted[:, :, 1:], predicted[:, :, :1]
        target_bg, target_ego = target[:, :, 1:], target[:, :, :1]
        pred_gap = (background[..., 0] - ego[..., 0]).abs().clamp_min(0.5)
        target_gap = (target_bg[..., 0] - target_ego[..., 0]).abs().clamp_min(0.5)
        pred_closing = (ego[..., 2] - background[..., 2]).clamp_min(0.0)
        target_closing = (target_ego[..., 2] - target_bg[..., 2]).clamp_min(0.0)
        pred_ttc = torch.where(
            pred_closing > 1.0e-3,
            pred_gap / pred_closing.clamp_min(1.0e-3),
            pred_gap.new_full(pred_gap.shape, 10.0),
        ).clamp_max(10.0)
        target_ttc = torch.where(
            target_closing > 1.0e-3,
            target_gap / target_closing.clamp_min(1.0e-3),
            target_gap.new_full(target_gap.shape, 10.0),
        ).clamp_max(10.0)
        pred_drac = torch.where(
            pred_closing > 1.0e-3,
            pred_closing.square() / (2.0 * pred_gap).clamp_min(0.1),
            pred_gap.new_zeros(pred_gap.shape),
        )
        target_drac = torch.where(
            target_closing > 1.0e-3,
            target_closing.square() / (2.0 * target_gap).clamp_min(0.1),
            target_gap.new_zeros(target_gap.shape),
        )
        return masked_mean(
            (pred_gap - target_gap).abs() / 10.0
            + (pred_ttc - target_ttc).abs() / 10.0
            + (pred_drac - target_drac).abs() / 8.0,
            valid,
        )

    def _lane_consistency_loss(
        self,
        predicted_background: torch.Tensor,
        current: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        """Discourage an implausible multi-lane jump inside one one-second plan."""
        lateral_shift = (predicted_background[..., 1] - current[:, None, 1:, 1]).abs()
        return masked_mean(
            functional.relu(lateral_shift - 1.25 * float(self.cfg.lane_width_m)), valid
        )

    def _available_frames(self, batch: dict[str, torch.Tensor]) -> int:
        return min(
            max(
                0,
                int(batch["agent_states"].shape[1]) - self.cfg.first_future_state_index,
            ),
            int(batch["actions_highd"].shape[1]),
        )

    def _rollout(
        self,
        batch: dict[str, torch.Tensor],
        *,
        response_steps: int | None,
        deterministic: bool,
        use_posterior: bool,
        tbptt_steps: int = 0,
        scene_standard_normal: torch.Tensor | None = None,
        agent_standard_normal: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        states, valid = batch["agent_states"], batch["agent_valid"]
        requested = (
            self.cfg.response_steps if response_steps is None else int(response_steps)
        )
        total_frames = min(
            self.cfg.rollout_frames_for_responses(requested),
            self._available_frames(batch),
        )
        if total_frames < 1:
            raise ValueError(
                "HiQR rollout batch does not contain a supervised future transition"
            )
        steps = (total_frames + self.cfg.execute_frames - 1) // self.cfg.execute_frames
        ego_mask = self._ego_mask(batch)
        if not torch.all(ego_mask[:, 0]):
            raise ValueError(
                "HiQR-WM requires the fixed [ego, six background slots] schema"
            )
        current, current_valid = (
            states[:, self.cfg.anchor_state_index],
            valid[:, self.cfg.anchor_state_index],
        )
        if "behavior_anchor_raw" not in batch or "behavior_anchor_valid" not in batch:
            raise ValueError("HiQR training requires its prepared B0 sidecar")
        hidden = self.initialize_start(
            current,
            current_valid,
            ego_mask,
            batch["map_polylines"],
            batch["map_polyline_valid"],
            batch["behavior_anchor_raw"],
            batch["behavior_anchor_valid"],
            batch["primary_slot_index"],
        )
        history = history_valid = None
        previous_buffer = previous_current = None
        predicted_frames: list[torch.Tensor] = []
        plans: list[torch.Tensor] = []
        masks: dict[str, list[torch.Tensor]] = {name: [] for name in BUFFER_MASK_NAMES}
        term_rows: list[dict[str, torch.Tensor]] = []
        hierarchical_rows: list[dict[str, torch.Tensor]] = []
        scene_latents: list[torch.Tensor] = []
        residuals: list[torch.Tensor] = []
        gates: list[torch.Tensor] = []
        for response in range(steps):
            target, target_valid, target_actions, target_action_valid = (
                self._target_plan(batch, response)
            )
            step_current, step_valid = current, current_valid
            scene_noise = (
                None
                if scene_standard_normal is None
                else scene_standard_normal[:, response]
            )
            agent_noise = (
                None
                if agent_standard_normal is None
                else agent_standard_normal[:, response]
            )
            out = self.plan_step(
                history,
                history_valid,
                current,
                current_valid,
                ego_mask,
                batch["map_polylines"],
                batch["map_polyline_valid"],
                interaction_state=hidden,
                previous_buffer=previous_buffer,
                previous_current=previous_current,
                deterministic=deterministic,
                use_posterior=use_posterior,
                posterior_future=target,
                posterior_future_valid=target_valid,
                scene_standard_normal=scene_noise,
                agent_standard_normal=agent_noise,
            )
            plan = out["background_future_actions"]
            execute_count = min(
                int(self.cfg.execute_frames),
                total_frames - response * int(self.cfg.execute_frames),
            )
            generated = current.clone()
            frames: list[torch.Tensor] = []
            frame_valid: list[torch.Tensor] = []
            for frame in range(execute_count):
                physical = current.new_zeros((current.shape[0], current.shape[1], 2))
                physical[:, 1:] = plan[:, frame]
                generated = self.dynamics.step(
                    generated, physical, current_valid, self.cfg.simulation_dt_s
                )
                target_frame = (
                    self.cfg.first_future_state_index
                    + response * int(self.cfg.execute_frames)
                    + frame
                )
                generated = torch.where(
                    ego_mask[..., None], states[:, target_frame], generated
                )
                current_valid = torch.where(
                    ego_mask, valid[:, target_frame], current_valid
                )
                frames.append(generated)
                frame_valid.append(current_valid)
            predicted = torch.stack(frames, dim=1)
            generated_valid = torch.stack(frame_valid, dim=1)
            execute_valid = target_valid[:, :execute_count, 1:]
            full_valid = target_valid[:, :, 1:]
            position = masked_mean(
                (predicted[:, :, 1:, :2] - target[:, :execute_count, 1:, :2])
                .abs()
                .mean(dim=-1),
                execute_valid,
            )
            velocity = masked_mean(
                (predicted[:, :, 1:, 2:4] - target[:, :execute_count, 1:, 2:4])
                .abs()
                .mean(dim=-1),
                execute_valid,
            )
            action = masked_mean(
                (plan[:, :execute_count] - target_actions[:, :execute_count])
                .abs()
                .mean(dim=-1),
                target_action_valid[:, :execute_count],
            )
            plan_position = masked_mean(
                (out["background_future_states"][..., :2] - target[:, :, 1:, :2])
                .abs()
                .mean(dim=-1),
                full_valid,
            )
            plan_action = masked_mean(
                (plan - target_actions).abs().mean(dim=-1), target_action_valid
            )
            overlap = (
                plan.new_zeros(())
                if previous_buffer is None
                else (
                    previous_buffer[:, self.cfg.execute_frames :]
                    - plan[:, : -self.cfg.execute_frames]
                )
                .abs()
                .mean()
            )
            jerk = (
                plan.new_zeros(())
                if plan.shape[1] < 2
                else (plan[:, 1:] - plan[:, :-1]).abs().mean()
            )
            lane = self._lane_consistency_loss(
                out["background_future_states"], step_current, full_valid
            )
            target_all = target[:, :execute_count]
            term_rows.append(
                {
                    "position": position,
                    "velocity": velocity,
                    "action": action,
                    "plan_position": plan_position,
                    "plan_action": plan_action,
                    "continuation": overlap,
                    "interaction": self._interaction_loss(
                        predicted, target_all, execute_valid
                    ),
                    "physical": self._physical_loss(predicted, execute_valid),
                    "jerk": jerk,
                    "lane": lane,
                    "gap_ttc": self._gap_ttc_loss(predicted, target_all, execute_valid),
                }
            )
            hierarchical_rows.append(out["hierarchical_terms"])
            scene_latents.append(out["scene_latent"])
            residuals.append(out["agent_residual"])
            gates.append(out["continuation_gate"])
            predicted_frames.extend(frames)
            plans.append(plan)
            for name, value in out["background_future_action_masks"].items():
                masks[name].append(value)
            previous_buffer, previous_current, hidden = (
                plan,
                step_current,
                out["interaction_state"],
            )
            current = predicted[:, -1]
            history = torch.cat(
                (step_current[:, None] if history is None else history, predicted),
                dim=1,
            )[:, -self.cfg.plan_frames :]
            history_valid = torch.cat(
                (
                    step_valid[:, None] if history_valid is None else history_valid,
                    generated_valid,
                ),
                dim=1,
            )[:, -self.cfg.plan_frames :]
            if (
                tbptt_steps
                and (response + 1) % int(tbptt_steps) == 0
                and response + 1 < steps
            ):
                (
                    history,
                    history_valid,
                    current,
                    previous_buffer,
                    previous_current,
                    hidden,
                ) = (
                    history.detach(),
                    history_valid.detach(),
                    current.detach(),
                    previous_buffer.detach(),
                    previous_current.detach(),
                    hidden.detach(),
                )
        terms = {
            key: torch.stack([row[key] for row in term_rows]).mean()
            for key in term_rows[0]
        }
        terms.update(
            {
                key: torch.stack([row[key] for row in hierarchical_rows]).mean()
                for key in hierarchical_rows[0]
            }
        )
        return {
            "predicted_states": torch.stack(predicted_frames, dim=1),
            "target_states": states[
                :,
                self.cfg.first_future_state_index : self.cfg.first_future_state_index
                + total_frames,
            ],
            "target_valid": valid[
                :,
                self.cfg.first_future_state_index : self.cfg.first_future_state_index
                + total_frames,
            ],
            "background_future_actions": torch.stack(plans, dim=1),
            "background_future_action_masks": {
                name: torch.stack(masks[name], dim=1) for name in BUFFER_MASK_NAMES
            },
            "scene_latent": torch.stack(scene_latents, dim=1),
            "agent_residual": torch.stack(residuals, dim=1),
            "continuation_gate": torch.stack(gates, dim=1),
            "terms": terms,
        }

    def rollout_reconstruction(
        self,
        batch: dict[str, torch.Tensor],
        *,
        response_steps: int | None = None,
        deterministic: bool = True,
        scene_standard_normal: torch.Tensor | None = None,
        agent_standard_normal: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        return self._rollout(
            batch,
            response_steps=response_steps,
            deterministic=deterministic,
            use_posterior=False,
            scene_standard_normal=scene_standard_normal,
            agent_standard_normal=agent_standard_normal,
        )

    def supervised_terms(
        self,
        batch: dict[str, torch.Tensor],
        *,
        response_steps: int | None = None,
    ) -> dict[str, torch.Tensor]:
        rollout = self._rollout(
            batch,
            response_steps=response_steps,
            deterministic=False,
            use_posterior=True,
        )
        return rollout["terms"]

    def forward_training(
        self,
        batch: dict[str, torch.Tensor],
        *,
        response_steps: int | None = None,
        tbptt_steps: int = 0,
    ) -> dict[str, torch.Tensor]:
        rollout = self._rollout(
            batch,
            response_steps=response_steps,
            deterministic=False,
            use_posterior=True,
            tbptt_steps=tbptt_steps,
        )
        terms = rollout["terms"]
        weights = {
            "position": self.cfg.position_weight,
            "velocity": self.cfg.velocity_weight,
            "action": self.cfg.action_weight,
            "plan_position": self.cfg.plan_position_weight,
            "plan_action": self.cfg.plan_action_weight,
            "continuation": self.cfg.continuation_weight,
            "interaction": self.cfg.interaction_weight,
            "physical": self.cfg.physical_weight,
            "jerk": self.cfg.jerk_weight,
            "lane": self.cfg.lane_weight,
            "gap_ttc": self.cfg.gap_ttc_weight,
            "scene_kl": self.cfg.scene_kl_weight,
            "agent_kl": self.cfg.agent_kl_weight,
            "diversity_floor": self.cfg.diversity_weight,
        }
        loss = sum(terms[name] * float(weight) for name, weight in weights.items())
        return {"loss": loss, **terms}

    def checkpoint_payload(self) -> dict[str, Any]:
        return {
            "model_type": self.model_type,
            "model_config": asdict(self.cfg),
            "state_dict": self.state_dict(),
            "flow_interface": {
                "flow_schema_sha256": self.flow_schema_sha256,
                "flow_coordinate_dim": 76,
                "b0_usage": "interaction_state_initialization_only",
                "event_structure": "slot_mask_plus_primary_risk_slot",
            },
        }
