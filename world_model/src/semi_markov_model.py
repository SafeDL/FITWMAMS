"""Semi-Markov Relational Traffic World Model."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn

from .anchor_residual_controller import AnchorResidualController
from .dynamics import DynamicsConfig, KinematicTrafficDynamics
from .initial_behavior_anchor import (
    BEHAVIOR_ANCHOR_SECONDS,
    FrozenLegacyFlowSchema,
    summarize_first_second_states,
)
from .start_mode import StartModeControl
from .intent_response_decoder import IntentResponseDecoder, IntentResponseDecoderConfig
from .relational_encoder import RelationalEncoderConfig, RelationalTrafficEncoder
from .semi_markov_state import SemiMarkovConfig, SemiMarkovLatentState


@dataclass(frozen=True)
class SemiMarkovWorldModelConfig:
    hidden_dim: int = 128
    temporal_layers: int = 1
    dropout: float = 0.1
    num_latent_states: int = 12
    max_duration_response_steps: int = 30
    response_interval_s: float = 0.2
    simulation_dt_s: float = 0.04
    beta_latent: float = 0.05
    lambda_roll: float = 1.0
    position_weight: float = 1.0
    velocity_weight: float = 0.25
    control_weight: float = 0.15
    boundary_change_threshold: float = 0.05
    boundary_supervision_weight: float = 0.25
    prototype_weight: float = 1.0
    state_bootstrap_weight: float = 5.0
    prior_roll_weight: float = 1.0
    late_roll_weight: float = 1.0
    prior_control_weight: float = 1.0
    late_prior_control_weight: float = 0.0
    reference_control_scale: float = 1.0
    use_conflict_zones: bool = False
    learn_duration: bool = True
    variant: str = "m0"
    anchor_loss_weight: float = 1.0
    start_roll_weight: float = 1.0
    cold_start_history: bool = False

    @property
    def physics_steps_per_response(self) -> int:
        return max(1, int(round(self.response_interval_s / self.simulation_dt_s)))

    @property
    def behavior_anchor_response_steps(self) -> int:
        return max(1, int(round(BEHAVIOR_ANCHOR_SECONDS / self.response_interval_s)))


class SemiMarkovRelationalWorldModel(nn.Module):
    """A graph world model with state persistence decoupled from response rate."""

    model_type = "semi_markov_relational"

    def __init__(self, cfg: SemiMarkovWorldModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        if cfg.variant not in {"m0", "m1"}:
            raise ValueError("model.variant must be one of 'm0' or 'm1'")
        if abs(cfg.behavior_anchor_response_steps * cfg.response_interval_s - BEHAVIOR_ANCHOR_SECONDS) > 1.0e-6:
            raise ValueError("response_interval_s must divide the one-second behavior anchor exactly")
        self.encoder = RelationalTrafficEncoder(RelationalEncoderConfig(
            hidden_dim=cfg.hidden_dim, temporal_layers=cfg.temporal_layers, dropout=cfg.dropout,
            use_conflict_zones=cfg.use_conflict_zones,
        ))
        self.latent = SemiMarkovLatentState(SemiMarkovConfig(
            num_states=cfg.num_latent_states, hidden_dim=cfg.hidden_dim,
            max_duration_steps=cfg.max_duration_response_steps,
            boundary_supervision_weight=cfg.boundary_supervision_weight,
            prototype_weight=cfg.prototype_weight,
            state_bootstrap_weight=cfg.state_bootstrap_weight,
        ))
        self.start_mode = None
        self.anchor_residual = None
        self.frozen_flow_schema: FrozenLegacyFlowSchema | None = None
        if self.uses_behavior_anchor:
            self.start_mode = StartModeControl(physics_steps=cfg.behavior_anchor_response_steps * cfg.physics_steps_per_response)
            self.anchor_residual = AnchorResidualController(cfg.hidden_dim, cfg.physics_steps_per_response)
        self.decoder = IntentResponseDecoder(IntentResponseDecoderConfig(
            hidden_dim=cfg.hidden_dim, reference_control_scale=cfg.reference_control_scale,
        ))
        self.dynamics = KinematicTrafficDynamics(DynamicsConfig())

    @property
    def response_steps(self) -> int:
        return 125 // self.cfg.physics_steps_per_response

    @property
    def uses_behavior_anchor(self) -> bool:
        return self.cfg.variant == "m1"

    def set_frozen_flow_schema(self, schema: FrozenLegacyFlowSchema) -> None:
        self.frozen_flow_schema = schema

    def _batch_behavior_anchor(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        if not self.uses_behavior_anchor:
            return None, None, None
        cached = {name: batch.get(name) for name in ("behavior_anchor_raw", "behavior_anchor_std", "behavior_anchor_valid")}
        if all(value is not None for value in cached.values()):
            anchor_raw, anchor_std, anchor_valid = cached["behavior_anchor_raw"], cached["behavior_anchor_std"], cached["behavior_anchor_valid"].bool()
            if anchor_raw.shape[-2:] != (6, 6) or anchor_std.shape != anchor_raw.shape or anchor_valid.shape != anchor_raw.shape[:-1]:
                raise ValueError("cached frozen Flow behavior anchors have incompatible shapes")
        else:
            # Unit-level graph tests intentionally do not own a persistent
            # highD cache.  Formal M1 training/evaluation always provides the
            # sidecar so it never recomputes logged B0 in a batch.
            if self.frozen_flow_schema is not None:
                raise RuntimeError("M1 with a frozen Flow schema requires cached behavior_anchor_raw/std/valid tensors")
            frames = self.cfg.behavior_anchor_response_steps * self.cfg.physics_steps_per_response
            valid = batch["agent_valid"][:, 24 : 25 + frames, 1:]
            anchor_raw, anchor_valid = summarize_first_second_states(batch["agent_states"][:, 24 : 25 + frames, 1:], valid)
            anchor_std = anchor_raw
        return anchor_raw, anchor_std, anchor_valid

    def _start_mode_controls(self, initial_states: torch.Tensor, anchor_raw: torch.Tensor | None, valid: torch.Tensor) -> torch.Tensor | None:
        if anchor_raw is None or self.start_mode is None:
            return None
        return self.start_mode(initial_states, anchor_raw, valid[:, 1:])

    def _realized_prefix_anchor(
        self,
        initial_states: torch.Tensor,
        generated_states: list[torch.Tensor],
        valid: torch.Tensor,
    ) -> torch.Tensor:
        """Summarize a generated START prefix in Flow model coordinates.

        A prefix is intentionally not a completed one-second anchor.  It is
        nevertheless represented by the same six physical quantities, then
        passed through the *frozen* Flow transform before the controller sees
        it.  This prevents the former invalid subtraction of normalized B0
        from metres/seconds squared.  The formal completion metric still uses
        :func:`summarize_first_second_states` on all 26 points.
        """
        prefix = torch.stack((initial_states, *generated_states), dim=1) if generated_states else initial_states[:, None]
        realized_raw = torch.stack((
            prefix[:, -1, 1:, 2] - initial_states[:, 1:, 2],
            prefix[:, -1, 1:, 3] - initial_states[:, 1:, 3],
            prefix[:, :, 1:, 4].mean(dim=1),
            prefix[:, :, 1:, 4].amin(dim=1),
            prefix[:, -1, 1:, 4],
            prefix[:, :, 1:, 5].mean(dim=1),
        ), dim=-1)
        background_valid = valid[:, 1:]
        if self.frozen_flow_schema is not None:
            return self.frozen_flow_schema.standardize(realized_raw, background_valid)
        return realized_raw * background_valid[..., None].to(dtype=realized_raw.dtype)

    def _anchor_residual_controls(
        self, agent_context: torch.Tensor, scene_context: torch.Tensor, latent_context: torch.Tensor,
        anchor_std: torch.Tensor | None, initial_states: torch.Tensor, generated_states: list[torch.Tensor],
        start_actions: torch.Tensor | None, response: int, valid: torch.Tensor,
    ) -> torch.Tensor | None:
        if self.anchor_residual is None or start_actions is None:
            return None
        if anchor_std is None:
            return None
        realized = self._realized_prefix_anchor(initial_states, generated_states, valid)
        remaining = anchor_std - realized
        start_controls = torch.zeros((valid.shape[0], start_actions.shape[1], valid.shape[1], 2), dtype=start_actions.dtype, device=start_actions.device)
        start_controls[:, :, 1:] = start_actions
        return self.anchor_residual(
            agent_context, scene_context, latent_context, anchor_std, realized, remaining, start_controls,
            1.0 - response * self.cfg.response_interval_s, valid,
        )

    @staticmethod
    def _ego_mask(batch: dict[str, torch.Tensor]) -> torch.Tensor:
        n = batch["agent_states"].shape[2]
        index = batch["ego_index"].long().clamp(0, n - 1)
        return torch.nn.functional.one_hot(index, num_classes=n).bool()

    @staticmethod
    def _lane_candidates(states: torch.Tensor, map_polylines: torch.Tensor, map_valid: torch.Tensor, top_r: int = 3) -> torch.Tensor:
        b, n, _ = states.shape
        m = map_polylines.shape[1]
        if m == 0:
            return torch.empty((b, n, 0), dtype=torch.long, device=states.device)
        displacement = map_polylines[:, None, :, :, :2] - states[:, :, None, None, :2]
        point_distance = displacement.square().sum(dim=-1).masked_fill(~map_valid[:, None], float("inf"))
        dist = point_distance.amin(dim=-1)
        r = min(int(top_r), m)
        candidates = dist.argsort(dim=-1)[..., :r]
        invalid_lanes = map_valid.any(dim=2)[:, None, :].expand(b, n, m).gather(2, candidates)
        return candidates.masked_fill(~invalid_lanes, -1)

    def encode_step(
        self,
        history_states: torch.Tensor,
        history_valid: torch.Tensor,
        current_states: torch.Tensor,
        current_valid: torch.Tensor,
        ego_mask: torch.Tensor,
        map_polylines: torch.Tensor,
        map_polyline_valid: torch.Tensor,
        lane_graph_edges: torch.Tensor | None = None,
        conflict_zone_features: torch.Tensor | None = None,
        conflict_zone_valid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        candidates = self._lane_candidates(current_states, map_polylines, map_polyline_valid)
        agents, scene = self.encoder(
            history_states, history_valid, current_states, current_valid, ego_mask,
            map_polylines, map_polyline_valid, candidates, lane_graph_edges,
            conflict_zone_features, conflict_zone_valid,
        )
        return agents, scene, candidates

    def _teacher_contexts(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        states, valid = batch["agent_states"], batch["agent_valid"]
        ego = self._ego_mask(batch)
        contexts: list[torch.Tensor] = []
        agents: list[torch.Tensor] = []
        stride = self.cfg.physics_steps_per_response
        for response in range(self.response_steps):
            end = 25 + response * stride
            history_states = states[:, end - 25 : end]
            history_valid = valid[:, end - 25 : end]
            if response == 0 and self.cfg.cold_start_history:
                history_states = states[:, 24:25].expand(-1, 25, -1, -1)
                history_valid = valid[:, 24:25].expand(-1, 25, -1)
            a, scene, _ = self.encode_step(
                history_states, history_valid, states[:, end - 1], valid[:, end - 1], ego,
                batch["map_polylines"], batch["map_polyline_valid"], batch.get("lane_graph_edges"),
                batch.get("conflict_zone_features"), batch.get("conflict_zone_valid"),
            )
            contexts.append(scene)
            agents.append(a)
        return torch.stack(agents, dim=1), torch.stack(contexts, dim=1)

    def _initial_history(
        self, states: torch.Tensor, valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cfg.cold_start_history:
            return (
                states[:, 24:25].expand(-1, 25, -1, -1).clone(),
                valid[:, 24:25].expand(-1, 25, -1).clone(),
            )
        return states[:, :25].clone(), valid[:, :25].clone()

    def _elapsed_from_boundaries(self, boundaries: torch.Tensor) -> torch.Tensor:
        values = [torch.ones_like(boundaries[:, 0])]
        for response in range(1, boundaries.shape[1]):
            values.append((1.0 - boundaries[:, response]) * (values[-1] + 1.0) + boundaries[:, response])
        return torch.stack(values, dim=1)

    def _integrate_response(
        self,
        state: torch.Tensor,
        controls: torch.Tensor,
        valid: torch.Tensor,
        ego_future: torch.Tensor,
        ego_valid: torch.Tensor,
        ego_index: int = 0,
        start_actions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Integrate a response residual plus an optional 25 Hz anchor plan."""
        outputs: list[torch.Tensor] = []
        current = state
        ego_mask = torch.zeros_like(valid)
        ego_mask[:, int(ego_index)] = True
        for physical in range(self.cfg.physics_steps_per_response):
            step_controls = controls[:, physical] if controls.ndim == 4 else controls
            if start_actions is not None:
                if start_actions.shape[1] != self.cfg.physics_steps_per_response:
                    raise ValueError("START controls must span one response interval")
                # B0 contains only background slots; the ego is replayed as
                # an external observation and deliberately receives no plan.
                start_highd = torch.zeros_like(current[..., :2])
                background = torch.ones(current.shape[1], dtype=torch.bool, device=current.device)
                background[int(ego_index)] = False
                start_highd[:, background] = start_actions[:, physical]
                start_controls = self.dynamics.controls_from_highd_actions(start_highd, current)
                step_controls = step_controls + start_controls
            next_state = self.dynamics.step(current, step_controls, valid, self.cfg.simulation_dt_s)
            next_state = torch.where(ego_mask[..., None], ego_future[:, physical, None, :], next_state)
            next_valid = torch.where(ego_mask, ego_valid[:, physical, None], valid)
            current = next_state * next_valid[..., None].float()
            valid = next_valid
            outputs.append(current)
        return current, outputs

    def _target_controls(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        # Mean Cartesian actions over a response interval, projected using the
        # logged response-start state.  The ego target is deliberately unused.
        b = batch["actions_highd"].shape[0]
        actions = batch["actions_highd"].reshape(b, self.response_steps, self.cfg.physics_steps_per_response, -1, 2).mean(dim=2)
        states = batch["agent_states"][:, 24 : 24 + self.response_steps * self.cfg.physics_steps_per_response : self.cfg.physics_steps_per_response, 1:]
        return self.dynamics.controls_from_highd_actions(actions, states)

    def _roll_mode_training(
        self,
        batch: dict[str, torch.Tensor],
        *,
        ego_mask: torch.Tensor,
        background_mask: torch.Tensor,
        response_steps: int | None = None,
        tbptt_response_steps: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Differentiable ROLL-mode generation used by the environment.

        The forward path uses hard discrete prior states and stage durations,
        matching ROLL.  The state one-hot uses a straight-through estimator so
        the same closed-loop ``L_roll`` can still train the prior without
        reverting to a soft mixture of behavioural states.
        """
        states, valid = batch["agent_states"], batch["agent_valid"]
        b = states.shape[0]
        rollout_steps = self.response_steps if response_steps is None else int(response_steps)
        if not 1 <= rollout_steps <= self.response_steps:
            raise ValueError("response_steps must be within the five-second response horizon")
        truncate_every = max(0, int(tbptt_response_steps))
        history, history_valid = self._initial_history(states, valid)
        current, current_valid = history[:, -1], history_valid[:, -1]
        anchor_raw, anchor_std, _anchor_valid = self._batch_behavior_anchor(batch)
        start_mode = self._start_mode_controls(states[:, 24], anchor_raw, valid[:, 24])
        state_prob = torch.zeros((b, self.cfg.num_latent_states), dtype=states.dtype, device=states.device)
        elapsed = torch.zeros((b,), dtype=states.dtype, device=states.device)
        remaining = torch.zeros((b,), dtype=torch.long, device=states.device)
        outputs: list[torch.Tensor] = []
        controls_out: list[torch.Tensor] = []
        ego_index = int(batch["ego_index"][0].item())
        for response in range(rollout_steps):
            agent_context, scene_context, _ = self.encode_step(
                history, history_valid, current, current_valid, ego_mask,
                batch["map_polylines"], batch["map_polyline_valid"], batch.get("lane_graph_edges"),
                batch.get("conflict_zone_features"), batch.get("conflict_zone_valid"),
            )
            starts = torch.ones_like(remaining, dtype=torch.bool) if not self.cfg.learn_duration else remaining <= 0
            proposal = torch.softmax(self.latent.prior_logits(scene_context, state_prob), dim=-1)
            hard = torch.nn.functional.one_hot(proposal.argmax(dim=-1), self.cfg.num_latent_states).to(dtype=proposal.dtype)
            sampled = hard + proposal - proposal.detach()
            state_prob = torch.where(starts[:, None], sampled, state_prob)
            # Draw one external uniform at each newly started phase and invert
            # the same discrete hazard distribution as ``rollout_roll_mode``.
            # The forward path is consequently a genuine hard semi-Markov
            # sample, rather than the old (and different) ``hazard >= 0.5``
            # rule.  Durations remain supervised by their complete-phase NLL;
            # they do not need a fictitious gradient through the draw.
            if self.cfg.learn_duration:
                duration = torch.full_like(remaining, self.cfg.max_duration_response_steps)
                unresolved = starts.clone()
                # Validation losses should be repeatable.  Training samples
                # the same exogenous duration variable that the world uses.
                uniform = torch.rand((b,), dtype=states.dtype, device=states.device) if self.training else torch.full_like(
                    elapsed, 0.5, dtype=states.dtype,
                )
                for age in range(1, self.cfg.max_duration_response_steps + 1):
                    age_tensor = torch.full_like(remaining, age)
                    hazard = torch.sigmoid(self.latent.hazard_logits(scene_context, hard, age_tensor))
                    chosen = unresolved & (uniform <= hazard)
                    duration = torch.where(chosen, torch.full_like(duration, age), duration)
                    uniform = torch.where(
                        unresolved,
                        (uniform - hazard) / (1.0 - hazard).clamp_min(1.0e-8),
                        uniform,
                    )
                    unresolved = unresolved & ~chosen
            else:
                duration = torch.ones_like(remaining)
            remaining = torch.where(starts, duration, remaining)
            elapsed = torch.where(starts, torch.ones_like(elapsed), elapsed + 1.0)
            start = response * self.cfg.physics_steps_per_response
            active_anchor = response < self.cfg.behavior_anchor_response_steps
            start_slice = start_mode[:, start : start + self.cfg.physics_steps_per_response] if active_anchor and start_mode is not None else None
            controls = self._anchor_residual_controls(
                agent_context, scene_context, self.latent.state_embedding(state_prob), anchor_std, states[:, 24], outputs,
                start_slice, response, current_valid & background_mask,
            ) if active_anchor else None
            if controls is None:
                decoded = self.decoder(
                    agent_context, scene_context, self.latent.state_embedding(state_prob), elapsed,
                    current_valid & background_mask,
                    self.dynamics.controls_from_highd_actions(current[..., 4:6], current),
                )
                controls = decoded["controls"]
            controls_out.append((controls.mean(dim=1) if controls.ndim == 4 else controls)[:, 1:])
            ego_future = states[:, 25 + start : 25 + start + self.cfg.physics_steps_per_response, ego_index]
            ego_future_valid = valid[:, 25 + start : 25 + start + self.cfg.physics_steps_per_response, ego_index]
            current, physical = self._integrate_response(
                current, controls, current_valid, ego_future, ego_future_valid, ego_index=ego_index,
                start_actions=start_slice,
            )
            outputs.extend(physical)
            # ROLL cannot inspect a future background membership mask. Preserve
            # generated background membership from
            # the current graph; only the externally observed ego validity is
            # allowed to advance from the logged replay stream.  (A sequence
            # cache has no highD entries after the initial history, but does
            # contain exits, so copying this mask was a real future leak.)
            observed_ego_valid = valid[:, 25 + start + self.cfg.physics_steps_per_response - 1]
            current_valid = torch.where(ego_mask, observed_ego_valid, current_valid)
            current = current * current_valid[..., None].float()
            history = torch.cat((history[:, self.cfg.physics_steps_per_response :], *[entry.unsqueeze(1) for entry in physical]), dim=1)
            history_valid = torch.cat((history_valid[:, self.cfg.physics_steps_per_response :], *[current_valid.unsqueeze(1) for _ in physical]), dim=1)
            # Retain the generated physical state while cutting gradients into
            # earlier response blocks.  This is genuine TBPTT rather than a
            # shorter independent trajectory: later graph encodings still see
            # the model's own generated history.
            if truncate_every and (response + 1) % truncate_every == 0 and response + 1 < rollout_steps:
                current = current.detach()
                history = history.detach()
                history_valid = history_valid.detach()
            remaining = remaining - 1
        return torch.stack(outputs, dim=1), torch.stack(controls_out, dim=1)

    def _boundary_target(self, batch: dict[str, torch.Tensor], target_controls: torch.Tensor) -> torch.Tensor:
        """Observed behaviour-change proxy for posterior state boundaries.

        This is derived only from natural-driving controls and lane-relative
        physical motion—not from EVT, ADS, or risk labels.  A scene starts a
        fresh state at response zero; later boundaries identify substantial
        joint control changes or an observed cross-lane displacement.
        """
        b, responses, _agents, _control_dim = target_controls.shape
        target = torch.zeros((b, responses), dtype=target_controls.dtype, device=target_controls.device)
        target[:, 0] = 1.0
        if responses <= 1:
            return target
        valid = batch["agent_valid"][:, 24 : 24 + responses * self.cfg.physics_steps_per_response : self.cfg.physics_steps_per_response, 1:]
        delta_control = torch.linalg.vector_norm(target_controls[:, 1:] - target_controls[:, :-1], dim=-1)
        pair_valid = valid[:, 1:] & valid[:, :-1]
        mean_change = (delta_control * pair_valid.float()).sum(dim=-1) / pair_valid.float().sum(dim=-1).clamp_min(1.0)
        response_states = batch["agent_states"][:, 24 : 24 + responses * self.cfg.physics_steps_per_response : self.cfg.physics_steps_per_response, 1:]
        lane_displacement = (response_states[:, 1:, :, 1] - response_states[:, :-1, :, 1]).abs()
        lane_change = ((lane_displacement > 0.75) & pair_valid).any(dim=-1)
        target[:, 1:] = ((mean_change >= float(self.cfg.boundary_change_threshold)) | lane_change).to(target.dtype)
        return target

    def _interaction_descriptor(self, batch: dict[str, torch.Tensor], target_controls: torch.Tensor) -> torch.Tensor:
        """Scale-bounded observed descriptor used for latent codebook commitment."""
        responses = target_controls.shape[1]
        valid = batch["agent_valid"][:, 24 : 24 + responses * self.cfg.physics_steps_per_response : self.cfg.physics_steps_per_response, 1:]
        states = batch["agent_states"][:, 24 : 24 + responses * self.cfg.physics_steps_per_response : self.cfg.physics_steps_per_response]
        mask = valid.float()
        denom = mask.sum(dim=-1).clamp_min(1.0)
        accel = (target_controls[..., 0] * mask).sum(dim=-1) / denom / 3.0
        yaw = (target_controls[..., 1] * mask).sum(dim=-1) / denom / 0.20
        relative_vx = (states[:, :, 1:, 2] - states[:, :, :1, 2]).mul(mask).sum(dim=-1) / denom / 10.0
        relative_x = (states[:, :, 1:, 0] - states[:, :, :1, 0]).mul(mask).sum(dim=-1) / denom / 40.0
        return torch.tanh(torch.stack((accel, yaw, relative_vx, relative_x), dim=-1))

    def _state_bootstrap_target(self, descriptor: torch.Tensor, boundary_target: torch.Tensor) -> torch.Tensor:
        """Quantize physical interaction descriptors, preserving a code per phase."""
        centers = self.latent.descriptor_centroids.to(dtype=descriptor.dtype)
        distance = (descriptor.unsqueeze(-2) - centers.view(1, 1, centers.shape[0], -1)).square().sum(dim=-1)
        instantaneous = distance.argmin(dim=-1)
        phases = [instantaneous[:, 0]]
        for response in range(1, instantaneous.shape[1]):
            phases.append(torch.where(boundary_target[:, response].bool(), instantaneous[:, response], phases[-1]))
        return torch.stack(phases, dim=1)

    @staticmethod
    def _masked_l1(
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        importance: torch.Tensor | None = None,
    ) -> torch.Tensor:
        weight = mask.float() if importance is None else mask.float() * importance.to(dtype=pred.dtype)
        weight = weight.unsqueeze(-1)
        return (torch.abs(pred - target) * weight).sum() / weight.sum().clamp_min(1.0)

    def _response_endpoint_l1(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Increasingly weight response endpoints within a closed-loop rollout.

        A flat frame-average underweights the 5-second endpoint: it contributes
        just one of 125 physics frames even though every integration error has
        accumulated there.  Response endpoints retain the model's multi-rate
        semantics and make the long-horizon part of the specified ``L_roll``
        explicit without introducing a separate training objective.
        """
        endpoint = self.cfg.physics_steps_per_response - 1
        pred, target, mask = pred[:, endpoint::self.cfg.physics_steps_per_response], target[:, endpoint::self.cfg.physics_steps_per_response], mask[:, endpoint::self.cfg.physics_steps_per_response]
        response_weight = torch.linspace(0.25, 1.0, pred.shape[1], dtype=pred.dtype, device=pred.device)
        weight = mask.float().unsqueeze(-1) * response_weight.view(1, -1, 1, 1)
        return (torch.abs(pred - target) * weight).sum() / weight.sum().clamp_min(1.0)

    @staticmethod
    def _late_response_l1(
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """L1 over response controls with linearly increasing horizon weight."""
        weights = torch.linspace(0.25, 1.0, pred.shape[1], dtype=pred.dtype, device=pred.device)
        weight = mask.float().unsqueeze(-1) * weights.view(1, -1, 1, 1)
        return (torch.abs(pred - target) * weight).sum() / weight.sum().clamp_min(1.0)

    def forward_training(
        self,
        batch: dict[str, torch.Tensor],
        teacher_forcing_ratio: float = 1.0,
        *,
        rollout_response_steps: int | None = None,
        tbptt_response_steps: int = 0,
    ) -> dict[str, torch.Tensor]:
        """Compute the three loss groups with optional random-length TBPTT.

        The posterior and duration objectives continue to observe the complete
        six-second sequence.  Only the closed-loop reconstruction path is
        randomly unfolded from one to five seconds during training, as the
        specification requires.  Evaluation leaves ``rollout_response_steps``
        unset and therefore always uses the complete five-second horizon.
        """
        states, valid = batch["agent_states"], batch["agent_valid"]
        b, _, n, _ = states.shape
        rollout_steps = self.response_steps if rollout_response_steps is None else int(rollout_response_steps)
        if not 1 <= rollout_steps <= self.response_steps:
            raise ValueError("rollout_response_steps must be within the five-second response horizon")
        truncate_every = max(0, int(tbptt_response_steps))
        ego_mask = self._ego_mask(batch)
        background_mask = ~ego_mask
        _, teacher_scene = self._teacher_contexts(batch)
        anchor_raw, anchor_std, anchor_valid = self._batch_behavior_anchor(batch)
        start_mode = self._start_mode_controls(states[:, 24], anchor_raw, valid[:, 24])
        target_controls = self._target_controls(batch)
        boundary_target = self._boundary_target(batch, target_controls)
        descriptor = self._interaction_descriptor(batch, target_controls)
        state_target = self._state_bootstrap_target(descriptor, boundary_target)
        latent_terms = self.latent.training_terms(
            teacher_scene, teacher_scene, boundary_target, descriptor, state_target,
            force_stepwise=not self.cfg.learn_duration,
        )
        q_state = latent_terms["posterior_state_probs"]
        q_boundary = latent_terms["posterior_boundary_probs"]
        elapsed = self._elapsed_from_boundaries(q_boundary)

        history, history_valid = self._initial_history(states, valid)
        current = history[:, -1]
        current_valid = history_valid[:, -1]
        predicted_frames: list[torch.Tensor] = []
        predicted_controls: list[torch.Tensor] = []
        response_controls: list[torch.Tensor] = []
        for response in range(rollout_steps):
            agent_context, scene_context, _ = self.encode_step(
                history, history_valid, current, current_valid, ego_mask,
                batch["map_polylines"], batch["map_polyline_valid"], batch.get("lane_graph_edges"),
                batch.get("conflict_zone_features"), batch.get("conflict_zone_valid"),
            )
            active_anchor = response < self.cfg.behavior_anchor_response_steps
            start = response * self.cfg.physics_steps_per_response
            start_slice = start_mode[:, start : start + self.cfg.physics_steps_per_response] if active_anchor and start_mode is not None else None
            controls = self._anchor_residual_controls(
                agent_context, scene_context, self.latent.state_embedding(q_state[:, response]), anchor_std, states[:, 24], predicted_frames,
                start_slice, response, current_valid & background_mask,
            ) if active_anchor else None
            if controls is None:
                decoded = self.decoder(
                    agent_context, scene_context, self.latent.state_embedding(q_state[:, response]), elapsed[:, response],
                    current_valid & background_mask, self.dynamics.controls_from_highd_actions(current[..., 4:6], current),
                )
                controls = decoded["controls"]
                response_gate = decoded["response_gate"]
            else:
                response_gate = torch.zeros((*controls.shape[:1], controls.shape[2], 1), dtype=controls.dtype, device=controls.device)
            ego_future = states[:, 25 + start : 25 + start + self.cfg.physics_steps_per_response, 0]
            ego_future_valid = valid[:, 25 + start : 25 + start + self.cfg.physics_steps_per_response, 0]
            next_current, physical_outputs = self._integrate_response(
                current, controls, current_valid, ego_future, ego_future_valid,
                start_actions=start_slice,
            )
            predicted_frames.extend(physical_outputs)
            predicted_controls.append((controls.mean(dim=1) if controls.ndim == 4 else controls)[:, 1:])
            response_controls.append(response_gate)
            true_physical = states[:, 25 + start : 25 + start + self.cfg.physics_steps_per_response]
            true_physical_valid = valid[:, 25 + start : 25 + start + self.cfg.physics_steps_per_response]
            true_next = states[:, 25 + start + self.cfg.physics_steps_per_response - 1]
            true_valid = valid[:, 25 + start + self.cfg.physics_steps_per_response - 1]
            # Scheduled sampling affects only background history.  Ego remains
            # the logged, already-observed physical state at every update.
            use_teacher = torch.rand((b, 1, 1), device=states.device) < float(teacher_forcing_ratio)
            teacher_background = background_mask[:, :, None] & use_teacher
            current = torch.where(teacher_background, true_next, next_current)
            current = torch.where(ego_mask[:, :, None], true_next, current)
            current_valid = torch.where(teacher_background.squeeze(-1), true_valid, current_valid)
            current_valid = torch.where(ego_mask, true_valid, current_valid)
            # Teacher forcing must update the *history* as well as the current
            # state.  Previously only ``current`` was replaced; the next graph
            # encoder then saw generated frames followed by a logged current
            # state, creating a non-physical discontinuity during stage one.
            # For free-running backgrounds, retain generated frames; ego always
            # uses the logged state that is already available at that time.
            history_frames: list[torch.Tensor] = []
            history_masks: list[torch.Tensor] = []
            for physical_index, generated in enumerate(physical_outputs):
                observed = true_physical[:, physical_index]
                observed_valid = true_physical_valid[:, physical_index]
                frame = torch.where(teacher_background, observed, generated)
                frame = torch.where(ego_mask[:, :, None], observed, frame)
                generated_valid = current_valid.clone()
                generated_valid = torch.where(ego_mask, observed_valid, generated_valid)
                frame_valid = torch.where(teacher_background.squeeze(-1), observed_valid, generated_valid)
                frame_valid = torch.where(ego_mask, observed_valid, frame_valid)
                history_frames.append(frame.unsqueeze(1))
                history_masks.append(frame_valid.unsqueeze(1))
            history = torch.cat((history[:, self.cfg.physics_steps_per_response :], *history_frames), dim=1)
            history_valid = torch.cat((history_valid[:, self.cfg.physics_steps_per_response :], *history_masks), dim=1)
            if truncate_every and (response + 1) % truncate_every == 0 and response + 1 < rollout_steps:
                current = current.detach()
                history = history.detach()
                history_valid = history_valid.detach()

        predicted = torch.stack(predicted_frames, dim=1)
        rollout_frames = rollout_steps * self.cfg.physics_steps_per_response
        target = states[:, 25 : 25 + rollout_frames]
        target_valid = valid[:, 25 : 25 + rollout_frames] & background_mask[:, None]
        pos_loss = self._masked_l1(predicted[..., :2], target[..., :2], target_valid)
        vel_loss = self._masked_l1(predicted[..., 2:4], target[..., 2:4], target_valid)
        controls_pred = torch.stack(predicted_controls, dim=1)
        control_mask = valid[:, 24 : 24 + rollout_frames : self.cfg.physics_steps_per_response, 1:]
        rollout_target_controls = target_controls[:, :rollout_steps]
        control_loss = self._masked_l1(controls_pred, rollout_target_controls, control_mask)
        anchor_loss = controls_pred.new_zeros(())
        if self.uses_behavior_anchor and rollout_frames >= self.cfg.behavior_anchor_response_steps * self.cfg.physics_steps_per_response:
            anchor_frames = self.cfg.behavior_anchor_response_steps * self.cfg.physics_steps_per_response
            generated_anchor, generated_valid = summarize_first_second_states(
                torch.cat((states[:, 24:25, 1:], predicted[:, :anchor_frames, 1:]), dim=1),
                valid[:, 24 : 25 + anchor_frames, 1:],
            )
            anchor_mask = generated_valid & anchor_valid
            normalized_generated = self.frozen_flow_schema.standardize(generated_anchor, anchor_mask) if self.frozen_flow_schema else generated_anchor
            normalized_target = anchor_std
            anchor_loss = self._masked_l1(normalized_generated, normalized_target, anchor_mask)
        prior_predicted, prior_controls = self._roll_mode_training(
            batch, ego_mask=ego_mask, background_mask=background_mask,
            response_steps=rollout_steps, tbptt_response_steps=truncate_every,
        )
        prior_control_loss = self._masked_l1(prior_controls, rollout_target_controls, control_mask)
        late_prior_control_loss = self._late_response_l1(prior_controls, rollout_target_controls, control_mask)
        start_prior_roll = controls_pred.new_zeros(())
        start_prior_endpoint = controls_pred.new_zeros(())
        start_prior_control = controls_pred.new_zeros(())
        if self.uses_behavior_anchor:
            start_frames = min(
                rollout_frames,
                self.cfg.behavior_anchor_response_steps * self.cfg.physics_steps_per_response,
            )
            start_responses = max(1, start_frames // self.cfg.physics_steps_per_response)
            start_prior_roll = self._masked_l1(
                prior_predicted[:, :start_frames, ..., :4], target[:, :start_frames, ..., :4], target_valid[:, :start_frames],
            )
            start_prior_endpoint = self._masked_l1(
                prior_predicted[:, start_frames - 1, ..., :4], target[:, start_frames - 1, ..., :4], target_valid[:, start_frames - 1],
            )
            start_prior_control = self._masked_l1(
                prior_controls[:, :start_responses], rollout_target_controls[:, :start_responses], control_mask[:, :start_responses],
            )
        recon = (
            self.cfg.position_weight * pos_loss + self.cfg.velocity_weight * vel_loss
            + self.cfg.control_weight * (
                control_loss + self.cfg.prior_control_weight * prior_control_loss
                + self.cfg.late_prior_control_weight * late_prior_control_loss
            )
        )
        first_step_frames = self.cfg.physics_steps_per_response
        first_target = target[:, :first_step_frames]
        first_valid = target_valid[:, :first_step_frames]
        first_recon = self._masked_l1(predicted[:, :first_step_frames, ..., :4], first_target[..., :4], first_valid)
        roll = self._masked_l1(predicted[..., :4], target[..., :4], target_valid)
        prior_roll = self._masked_l1(prior_predicted[..., :4], target[..., :4], target_valid)
        endpoint_roll = self._response_endpoint_l1(predicted[..., :4], target[..., :4], target_valid)
        prior_endpoint_roll = self._response_endpoint_l1(prior_predicted[..., :4], target[..., :4], target_valid)
        latent_loss = (
            latent_terms["latent_kl"] + latent_terms["duration_nll"] + latent_terms["censor_nll"]
            + self.cfg.boundary_supervision_weight * latent_terms["posterior_boundary_nll"]
            + self.cfg.prototype_weight * latent_terms["prototype_reconstruction"]
            + self.cfg.state_bootstrap_weight * latent_terms["state_bootstrap_nll"]
        )
        total = recon + self.cfg.anchor_loss_weight * anchor_loss + self.cfg.beta_latent * latent_loss + self.cfg.lambda_roll * (
            roll + self.cfg.prior_roll_weight * prior_roll
            + self.cfg.late_roll_weight * (endpoint_roll + self.cfg.prior_roll_weight * prior_endpoint_roll)
        ) + self.cfg.start_roll_weight * (
            start_prior_roll + start_prior_endpoint + self.cfg.control_weight * start_prior_control
        )
        return {
            "loss": total, "recon_loss": recon.detach(), "roll_loss": roll.detach(), "prior_roll_loss": prior_roll.detach(),
            "endpoint_roll_loss": endpoint_roll.detach(), "prior_endpoint_roll_loss": prior_endpoint_roll.detach(), "first_step_recon": first_recon.detach(),
            "position_l1": pos_loss.detach(), "velocity_l1": vel_loss.detach(), "control_l1": control_loss.detach(),
            "anchor_loss": anchor_loss.detach(),
            "prior_control_loss": prior_control_loss.detach(), "late_prior_control_loss": late_prior_control_loss.detach(),
            "start_prior_roll_loss": start_prior_roll.detach(), "start_prior_endpoint_loss": start_prior_endpoint.detach(),
            "start_prior_control_loss": start_prior_control.detach(),
            "latent_loss": latent_loss.detach(), "latent_kl": latent_terms["latent_kl"].detach(),
            "duration_nll": latent_terms["duration_nll"].detach(), "censor_nll": latent_terms["censor_nll"].detach(),
            "posterior_boundary_nll": latent_terms["posterior_boundary_nll"].detach(),
            "boundary_target_rate": latent_terms["boundary_target_rate"].detach(),
            "prototype_reconstruction": latent_terms["prototype_reconstruction"].detach(),
            "state_bootstrap_nll": latent_terms["state_bootstrap_nll"].detach(),
            "switch_rate": latent_terms["switch_rate"].detach(), "posterior_state_probs": q_state.detach(),
            "posterior_boundary_probs": q_boundary.detach(), "predicted_states": predicted.detach(),
            "boundary_target": boundary_target.detach(),
            "state_target": state_target.detach(),
            "prior_logits": latent_terms["prior_logits"].detach(),
            "posterior_raw_state_probs": latent_terms["posterior_raw_state_probs"].detach(),
            "target_states": target.detach(), "target_valid": target_valid.detach(), "response_gate": torch.stack(response_controls, dim=1).detach(),
            "rollout_response_steps": torch.as_tensor(float(rollout_steps), device=states.device),
        }

    @torch.no_grad()
    def rollout_roll_mode(
        self,
        batch: dict[str, torch.Tensor],
        *,
        seed: int = 123,
        deterministic: bool = True,
        deterministic_duration: bool = False,
    ) -> dict[str, torch.Tensor | list[list[int]]]:
        """Five-second ROLL-mode generation for logged-ego reconstruction.

        The posterior is never consulted here. At each response update the
        model only sees generated background history and the ego state already
        observed at that update; the subsequent logged ego segment is used
        solely as the physical replay input for the next time interval.
        """
        states, valid = batch["agent_states"], batch["agent_valid"]
        b, _, n, _ = states.shape
        ego_mask = self._ego_mask(batch)
        background_mask = ~ego_mask
        history, history_valid = self._initial_history(states, valid)
        current, current_valid = history[:, -1], history_valid[:, -1]
        anchor_raw, anchor_std, _anchor_valid = self._batch_behavior_anchor(batch)
        start_mode = self._start_mode_controls(states[:, 24], anchor_raw, valid[:, 24])
        generator = torch.Generator(device=states.device)
        generator.manual_seed(int(seed))
        previous = torch.zeros((b, self.cfg.num_latent_states), device=states.device)
        remaining = torch.zeros((b,), dtype=torch.long, device=states.device)
        elapsed = torch.zeros((b,), dtype=torch.long, device=states.device)
        latent_states: list[list[int]] = [[] for _ in range(b)]
        latent_durations: list[list[int]] = [[] for _ in range(b)]
        predicted: list[torch.Tensor] = []
        controls_out: list[torch.Tensor] = []
        for response in range(self.response_steps):
            agents, scene, _ = self.encode_step(
                history, history_valid, current, current_valid, ego_mask,
                batch["map_polylines"], batch["map_polyline_valid"], batch.get("lane_graph_edges"),
                batch.get("conflict_zone_features"), batch.get("conflict_zone_valid"),
            )
            starts = torch.ones_like(remaining, dtype=torch.bool) if not self.cfg.learn_duration else remaining <= 0
            logits = self.latent.prior_logits(scene, previous)
            probabilities = torch.softmax(logits, dim=-1)
            if deterministic:
                sampled = probabilities.argmax(dim=-1)
            else:
                sampled = torch.multinomial(probabilities, 1, generator=generator).squeeze(-1)
            for item in torch.nonzero(starts, as_tuple=False).flatten().tolist():
                one = torch.nn.functional.one_hot(sampled[item], num_classes=self.cfg.num_latent_states).float().view(1, -1)
                if deterministic_duration:
                    # The deterministic path is useful for reconstruction
                    # diagnostics: select the most likely discrete duration
                    # rather than retaining a hidden uniform draw.  Sampling
                    # remains the default and is what the closed-loop world
                    # environment uses for its explicit Xi_world trace.
                    survival = 1.0
                    duration_probabilities: list[float] = []
                    for age in range(1, self.cfg.max_duration_response_steps + 1):
                        hazard = 1.0 if not self.cfg.learn_duration else float(torch.sigmoid(
                            self.latent.hazard_logits(scene[item : item + 1], one, torch.tensor([age], device=states.device))
                        )[0].item())
                        duration_probabilities.append(survival * hazard)
                        survival *= 1.0 - hazard
                    duration = max(range(len(duration_probabilities)), key=duration_probabilities.__getitem__) + 1
                else:
                    # A duration is sampled by inverse hazards from one
                    # independent uniform.  ``torch.rand`` keeps replay
                    # reproducible while preserving the model's stochastic
                    # semi-Markov semantics.
                    u = float(torch.rand((), generator=generator, device=states.device).item())
                    duration = 1 if not self.cfg.learn_duration else self.cfg.max_duration_response_steps
                    rem = u
                    for age in range(1, self.cfg.max_duration_response_steps + 1):
                        if not self.cfg.learn_duration:
                            break
                        hazard = torch.sigmoid(self.latent.hazard_logits(scene[item : item + 1], one, torch.tensor([age], device=states.device)))[0]
                        if rem <= float(hazard.item()):
                            duration = age
                            break
                        rem = (rem - float(hazard.item())) / max(1.0 - float(hazard.item()), 1.0e-8)
                remaining[item] = int(duration)
                elapsed[item] = 1
                latent_states[item].append(int(sampled[item]))
                latent_durations[item].append(int(duration))
            previous = torch.where(starts[:, None], torch.nn.functional.one_hot(sampled, self.cfg.num_latent_states).float(), previous)
            elapsed = torch.where(starts, elapsed, elapsed + 1)
            start = response * self.cfg.physics_steps_per_response
            active_anchor = response < self.cfg.behavior_anchor_response_steps
            start_slice = start_mode[:, start : start + self.cfg.physics_steps_per_response] if active_anchor and start_mode is not None else None
            controls = self._anchor_residual_controls(
                agents, scene, self.latent.state_embedding(previous), anchor_std, states[:, 24], predicted,
                start_slice, response, current_valid & background_mask,
            ) if active_anchor else None
            if controls is None:
                decoded = self.decoder(
                    agents, scene, self.latent.state_embedding(previous), elapsed, current_valid & background_mask,
                    self.dynamics.controls_from_highd_actions(current[..., 4:6], current),
                )
                controls = decoded["controls"]
            ego_index = int(batch["ego_index"][0].item())
            ego_future = states[:, 25 + start : 25 + start + self.cfg.physics_steps_per_response, ego_index]
            ego_future_valid = valid[:, 25 + start : 25 + start + self.cfg.physics_steps_per_response, ego_index]
            current, physical = self._integrate_response(
                current, controls, current_valid, ego_future, ego_future_valid, ego_index=ego_index,
                start_actions=start_slice,
            )
            predicted.extend(physical)
            controls_out.append(controls.mean(dim=1) if controls.ndim == 4 else controls)
            # Generated background membership does not use logged future
            # validity, which remains a target/evaluation mask;
            # ego validity is the sole externally supplied state signal.
            observed_ego_valid = valid[:, 25 + start + self.cfg.physics_steps_per_response - 1]
            current_valid = torch.where(ego_mask, observed_ego_valid, current_valid)
            history = torch.cat((history[:, self.cfg.physics_steps_per_response :], *[entry.unsqueeze(1) for entry in physical]), dim=1)
            history_valid = torch.cat((history_valid[:, self.cfg.physics_steps_per_response :], *[current_valid.unsqueeze(1) for _ in physical]), dim=1)
            remaining = remaining - 1
        target_valid = valid[:, 25:150] & background_mask[:, None]
        predicted_tensor = torch.stack(predicted, dim=1)
        first_second_states = torch.cat((states[:, 24:25], predicted_tensor[:, :25]), dim=1)
        first_second_valid = torch.cat((valid[:, 24:25], target_valid[:, :25]), dim=1)
        roll_anchor_raw, roll_anchor_valid = summarize_first_second_states(
            first_second_states[:, :, 1:], first_second_valid[:, :, 1:]
        )
        return {
            "predicted_states": predicted_tensor, "target_states": states[:, 25:150],
            "target_valid": target_valid, "controls": torch.stack(controls_out, dim=1),
            "latent_states": latent_states, "latent_durations": latent_durations,
            "first_second_states": first_second_states, "roll_mode_anchor_raw": roll_anchor_raw,
            "roll_mode_anchor_valid": roll_anchor_valid,
        }

    def config_payload(self) -> dict[str, Any]:
        return asdict(self.cfg)
