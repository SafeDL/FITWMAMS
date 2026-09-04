"""Diffusion-guided, observation-filtered HiQR response model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import nn

from world_model.src.core.dynamics import DynamicsConfig, KinematicTrafficDynamics
from world_model.src.hiqr.encoder import UnifiedRelationalQueryEncoder
from world_model.src.hiqr.filter import (
    FilterState,
    ObservedHierarchicalInteractionFilter,
)

from .config import WorldModelConfig
from .reference import (
    preview_features,
    rebase_soft_preview,
    response_relevance,
    soft_reference_controls,
)
from .reaction_controller import ReactionControllerContext, apply_handcrafted_response
from .stochastic import CausalInteractionResponseField, GraphCoupledLatentTransition


@dataclass(frozen=True)
class ResponseDistribution:
    mean: torch.Tensor
    std: torch.Tensor
    actions: torch.Tensor
    states: torch.Tensor
    rebased_preview: torch.Tensor
    reference_actions: torch.Tensor
    filter_state: FilterState
    slow_scene: torch.Tensor
    slow_scene_noise: torch.Tensor
    agent_noise_state: torch.Tensor
    agent_style_state: torch.Tensor
    agent_latent: torch.Tensor
    agent_latent_log_prob: torch.Tensor
    graph_latent_message: torch.Tensor
    response_field_gain: torch.Tensor | None
    intervention_trigger: torch.Tensor
    intervention_memory: torch.Tensor
    lateral_intervention_memory: torch.Tensor
    scene_refreshed: bool


class DiffusionGuidedJerkDecoder(nn.Module):
    """Decode coordinated residual jerk inside a modifiable soft plan."""

    def __init__(self, cfg: WorldModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        hidden = cfg.hidden_dim
        self.agent = nn.Sequential(
            nn.Linear(3 * hidden + cfg.agent_latent_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.scene = nn.Sequential(
            nn.Linear(2 * hidden + cfg.scene_latent_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.knot = nn.Parameter(torch.empty(cfg.jerk_knots, hidden))
        layer = nn.TransformerEncoderLayer(
            hidden,
            cfg.num_heads,
            2 * hidden,
            cfg.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.coordination = nn.TransformerEncoder(
            layer, num_layers=cfg.decoder_layers
        )
        self.residual_jerk = nn.Linear(hidden, 2)
        self.response_gain = nn.Linear(3 * hidden, 1)
        self.scene_innovation = nn.Linear(
            cfg.scene_latent_dim, cfg.jerk_knots * 2, bias=False
        )
        self.agent_innovation = nn.Linear(
            cfg.agent_latent_dim, cfg.jerk_knots * 2, bias=False
        )
        # A zero-impact-on-natural-replay response adapter.  Its action
        # correction is gated by a causal, large ego-control deviation and is
        # the only trainable component in the intervention-only pilot.
        self.intervention_logit = nn.Parameter(
            torch.tensor(float(cfg.intervention_adapter_initial_logit))
        )
        self.response_field = CausalInteractionResponseField(
            hidden, cfg.agent_latent_dim
        )
        self.behavior_mode = nn.Sequential(
            nn.Linear(cfg.agent_latent_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2),
        )
        nn.init.normal_(self.knot, std=0.02)
        nn.init.zeros_(self.residual_jerk.weight)
        nn.init.zeros_(self.residual_jerk.bias)
        nn.init.zeros_(self.response_gain.weight)
        initial_fraction = (
            (cfg.causal_response_initial_gain - cfg.causal_response_min_gain)
            / (cfg.causal_response_max_gain - cfg.causal_response_min_gain)
        )
        nn.init.constant_(
            self.response_gain.bias,
            float(torch.logit(torch.tensor(initial_fraction))),
        )
        nn.init.orthogonal_(self.scene_innovation.weight)
        nn.init.orthogonal_(self.agent_innovation.weight)
        nn.init.zeros_(self.behavior_mode[-1].weight)
        nn.init.zeros_(self.behavior_mode[-1].bias)

    def forward(
        self,
        agents: torch.Tensor,
        preview_agents: torch.Tensor,
        preview_scene: torch.Tensor,
        filter_state: FilterState,
        scene_latent: torch.Tensor,
        agent_latent: torch.Tensor,
        current: torch.Tensor,
        current_valid: torch.Tensor,
        soft_actions: torch.Tensor,
        ego_acceleration: torch.Tensor,
        intervention_trigger: torch.Tensor,
        lateral_intervention_trigger: torch.Tensor,
        relevance: torch.Tensor,
        scene_standard_normal: torch.Tensor,
        agent_standard_normal: torch.Tensor,
        agent_style_state: torch.Tensor,
        stochastic: bool,
        response_field_gain: torch.Tensor | None = None,
        response_sensitivity_bounds: torch.Tensor | None = None,
        apply_intervention_adapter: bool = True,
        apply_explicit_ego_response: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        batch, backgrounds = current.shape[0], current.shape[1] - 1
        if backgrounds != 6:
            raise ValueError("the response layer requires six background slots")
        base = self.agent(
            torch.cat(
                (
                    agents[:, 1:],
                    filter_state.agent_hidden[:, 1:],
                    preview_agents,
                    agent_latent[:, 1:],
                ),
                dim=-1,
            )
        )
        scene = self.scene(
            torch.cat(
                (filter_state.global_hidden, preview_scene, scene_latent), dim=-1
            )
        )
        token = base[:, None] + scene[:, None, None] + self.knot[None, :, None]
        valid = current_valid[:, None, 1:].expand(-1, cfg.jerk_knots, -1)
        padding = ~valid.reshape(batch, cfg.jerk_knots * backgrounds)
        padding = torch.where(
            padding.all(dim=1, keepdim=True), torch.zeros_like(padding), padding
        )
        encoded = self.coordination(
            token.reshape(batch, cfg.jerk_knots * backgrounds, -1),
            src_key_padding_mask=padding,
        ).reshape(batch, cfg.jerk_knots, backgrounds, -1)
        limits = encoded.new_tensor(
            (cfg.max_residual_jerk_mps3, cfg.max_residual_yaw_acceleration_rps2)
        )
        residual_knots = torch.tanh(self.residual_jerk(encoded)) * limits
        gain_features = torch.cat(
            (agents[:, 1:], filter_state.agent_hidden[:, 1:], preview_agents), dim=-1
        )
        response_gain = (
            cfg.causal_response_min_gain
            + torch.sigmoid(self.response_gain(gain_features)).squeeze(-1)
            * (cfg.causal_response_max_gain - cfg.causal_response_min_gain)
        )
        # The released decoder historically contains this explicit
        # ego-acceleration actuator in addition to its learned nominal action
        # head.  Controller-mode rollouts request the latter as their base
        # action and apply exactly one post-HIQR controller afterwards.  Keep
        # the default intact for checkpoint-compatible direct model calls.
        causal_longitudinal = (
            cfg.causal_response_scale
            * response_gain
            * relevance
            * ego_acceleration[:, None]
            if apply_explicit_ego_response
            else torch.zeros_like(relevance)
        )
        causal_knots = torch.stack(
            (causal_longitudinal, torch.zeros_like(causal_longitudinal)), dim=-1
        )[:, None].expand(-1, cfg.jerk_knots, -1, -1)
        scene_innovation = self.scene_innovation(scene_standard_normal).reshape(
            batch, cfg.jerk_knots, 1, 2
        )
        agent_innovation = self.agent_innovation(
            agent_standard_normal[:, 1:]
        ).reshape(batch, backgrounds, cfg.jerk_knots, 2).permute(0, 2, 1, 3)
        innovation_scale = residual_knots.new_tensor(
            (
                cfg.stochastic_longitudinal_jerk_mps3,
                cfg.stochastic_yaw_acceleration_rps2,
            )
        )
        stochastic_knots = (
            torch.tanh((scene_innovation + agent_innovation) / 2.0)
            * innovation_scale
            if stochastic
            else torch.zeros_like(residual_knots)
        )
        deterministic_knots = residual_knots + causal_knots
        residual = functional.interpolate(
            deterministic_knots.permute(0, 2, 3, 1).reshape(
                batch * backgrounds, 2, cfg.jerk_knots
            ),
            size=cfg.preview_frames,
            mode="linear",
            align_corners=True,
        ).reshape(batch, backgrounds, 2, cfg.preview_frames).permute(0, 3, 1, 2)
        stochastic_residual = functional.interpolate(
            stochastic_knots.permute(0, 2, 3, 1).reshape(
                batch * backgrounds, 2, cfg.jerk_knots
            ),
            size=cfg.preview_frames,
            mode="linear",
            align_corners=True,
        ).reshape(batch, backgrounds, 2, cfg.preview_frames).permute(0, 3, 1, 2)
        initial = KinematicTrafficDynamics.controls_from_highd_actions(
            current[:, 1:, 4:6], current[:, 1:]
        )
        soft_jerk = torch.diff(
            soft_actions,
            dim=1,
            prepend=initial[:, None],
        ) / cfg.dt_s
        deterministic_total = soft_jerk + residual
        deterministic_actions = (
            initial[:, None] + deterministic_total.cumsum(dim=1) * cfg.dt_s
        )
        sampled_actions = deterministic_actions + stochastic_residual.cumsum(
            dim=1
        ) * cfg.dt_s
        if cfg.behavior_mode_decoder_enabled and stochastic:
            # A behavior-state effect is constant over this one-second soft
            # preview.  Unlike jitter, it therefore remains identifiable as
            # a persistent driver-response branch across 25 Hz replans.
            mode_limits = sampled_actions.new_tensor(
                (
                    cfg.behavior_mode_max_acceleration_mps2,
                    cfg.behavior_mode_max_yaw_rate_rps,
                )
            )
            mode = torch.tanh(self.behavior_mode(agent_style_state[:, 1:]))
            sampled_actions = sampled_actions + mode[:, None] * mode_limits

        if cfg.intervention_adapter_enabled and apply_intervention_adapter:
            adapter_context = ReactionControllerContext(
                history=current[:, None], history_valid=current_valid[:, None],
                current=current, current_valid=current_valid,
                committed_ego_controls=ego_acceleration[:, None, None].expand(-1, 1, 2),
                base_actions=deterministic_actions, reference_actions=soft_actions,
                intervention_trigger=intervention_trigger,
                intervention_memory=intervention_trigger,
                lateral_intervention_memory=lateral_intervention_trigger,
                agent_style_state=agent_style_state,
                response_field_gain=response_field_gain,
                response_sensitivity_bounds=response_sensitivity_bounds,
                adapter_gain=torch.sigmoid(self.intervention_logit), cfg=cfg,
                reaction_enabled=None,
            )
            deterministic_actions = apply_handcrafted_response(deterministic_actions, adapter_context)
            sampled_actions = apply_handcrafted_response(sampled_actions, adapter_context)

        def bound(actions: torch.Tensor) -> torch.Tensor:
            return torch.stack(
                (
                    actions[..., 0].clamp(
                        cfg.min_acceleration_mps2, cfg.max_acceleration_mps2
                    ),
                    actions[..., 1].clamp(
                        -cfg.max_yaw_rate_rps, cfg.max_yaw_rate_rps
                    ),
                ),
                dim=-1,
            )
        mask = current_valid[:, None, 1:, None].float()
        return (
            bound(sampled_actions) * mask,
            bound(deterministic_actions) * mask,
            soft_actions * mask,
        )


class DiffusionGuidedHiQR(nn.Module):
    """Couple a diffusion soft plan to HiQR priors and joint jerk decoding."""

    model_type = "diffusion_guided_hiqr"

    def __init__(self, cfg: WorldModelConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or WorldModelConfig()
        hiqr_cfg = self.cfg.hiqr_config()
        self.encoder = UnifiedRelationalQueryEncoder(hiqr_cfg)
        self.filter = ObservedHierarchicalInteractionFilter(hiqr_cfg)
        self.preview_agent = nn.Sequential(
            nn.Linear(12, self.cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.cfg.hidden_dim, self.cfg.hidden_dim),
        )
        self.preview_scene = nn.Sequential(
            nn.Linear(self.cfg.hidden_dim, self.cfg.hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(self.cfg.hidden_dim),
        )
        self.decoder = DiffusionGuidedJerkDecoder(self.cfg)
        self.latent_transition = GraphCoupledLatentTransition(
            self.cfg.agent_latent_dim,
            self.cfg.hidden_dim,
            self.cfg.latent_flow_persistence,
        )
        initial_std = torch.tensor(
            (
                self.cfg.initial_acceleration_std_mps2,
                self.cfg.initial_yaw_rate_std_rps,
            )
        )
        self.log_std = nn.Parameter(initial_std.log())
        self.register_buffer(
            "matched_response_bounds", torch.full((2, 2), float("nan"))
        )
        self.register_buffer(
            "response_sensitivity_bounds", torch.full((2, 2), float("nan"))
        )
        self.dynamics = KinematicTrafficDynamics(
            DynamicsConfig(
                acceleration_min_mps2=self.cfg.min_acceleration_mps2,
                acceleration_max_mps2=self.cfg.max_acceleration_mps2,
            )
        )

    def set_matched_response_bounds(self, values: torch.Tensor) -> None:
        """Install highD-derived P10/P90 acceleration-response bounds."""
        bounds = torch.as_tensor(values, dtype=self.matched_response_bounds.dtype)
        if bounds.shape != (2, 2) or not torch.isfinite(bounds).all():
            raise ValueError("matched response bounds must be finite [2,2]")
        self.matched_response_bounds.copy_(bounds.to(self.matched_response_bounds))

    def set_response_sensitivity_bounds(self, values: torch.Tensor) -> None:
        """Install train-split P10/P90 response sensitivity per ego m/s²."""
        bounds = torch.as_tensor(values, dtype=self.response_sensitivity_bounds.dtype)
        if bounds.shape != (2, 2) or not torch.isfinite(bounds).all():
            raise ValueError("response sensitivity bounds must be finite [2,2]")
        self.response_sensitivity_bounds.copy_(
            bounds.to(self.response_sensitivity_bounds)
        )

    def _preview_context(
        self,
        current: torch.Tensor,
        current_valid: torch.Tensor,
        rebased: torch.Tensor,
        reference_base: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.cfg.use_soft_plan:
            agents = current.new_zeros((len(current), 6, self.cfg.hidden_dim))
            return agents, current.new_zeros((len(current), self.cfg.hidden_dim))
        summary = preview_features(current[:, 1:], rebased, dt_s=self.cfg.dt_s)
        summary_scale = summary.new_tensor(
            (100.0, 12.0, 100.0, 12.0, 100.0, 12.0, 20.0, 5.0)
        )
        summary = summary / summary_scale
        first_velocity = (
            rebased[:, 0] - current[:, 1:, :2]
        ) / self.cfg.dt_s
        deviation = torch.cat(
            (
                current[:, 1:, :2] - reference_base,
                current[:, 1:, 2:4] - first_velocity,
            ),
            dim=-1,
        )
        deviation = deviation / deviation.new_tensor((20.0, 5.0, 20.0, 5.0))
        agents = self.preview_agent(torch.cat((summary, deviation), dim=-1))
        valid = current_valid[:, 1:, None].float()
        agents = agents * valid
        scene = self.preview_scene(
            (agents * valid).sum(1) / valid.sum(1).clamp_min(1.0)
        )
        return agents, scene

    def _integrate(
        self,
        current: torch.Tensor,
        actions: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        state = current[:, 1:]
        frames = []
        for action in actions.unbind(dim=1):
            state = self.dynamics.step(state, action, valid[:, 1:], self.cfg.dt_s)
            frames.append(state)
        return torch.stack(frames, dim=1)

    def forward(
        self,
        history: torch.Tensor,
        history_valid: torch.Tensor,
        current: torch.Tensor,
        current_valid: torch.Tensor,
        soft_reference: torch.Tensor,
        reference_base: torch.Tensor,
        map_polylines: torch.Tensor,
        map_polyline_valid: torch.Tensor,
        *,
        filter_state: FilterState | None = None,
        previous_current: torch.Tensor | None = None,
        slow_scene: torch.Tensor | None = None,
        slow_scene_noise: torch.Tensor | None = None,
        agent_noise_state: torch.Tensor | None = None,
        agent_style_state: torch.Tensor | None = None,
        committed_ego_controls: torch.Tensor | None = None,
        intervention_memory: torch.Tensor | None = None,
        lateral_intervention_memory: torch.Tensor | None = None,
        response_index: int = 0,
        scene_standard_normal: torch.Tensor | None = None,
        agent_standard_normal: torch.Tensor | None = None,
        deterministic: bool = False,
        apply_intervention_adapter: bool | None = None,
        apply_explicit_ego_response: bool | None = None,
    ) -> ResponseDistribution:
        if not 1 <= history.shape[1] <= self.cfg.history_frames:
            raise ValueError("history must contain one to 25 observed frames")
        ego_mask = torch.zeros_like(current_valid)
        ego_mask[:, 0] = True
        agents, scene, _, _ = self.encoder(
            history,
            history_valid,
            current,
            current_valid,
            ego_mask,
            map_polylines,
            map_polyline_valid,
            mode="roll",
        )
        if filter_state is None:
            filter_state = self.filter.initialize(
                scene,
                agents,
                history[:, 0, 1:],
                history_valid[:, 0, 1:],
            )
        observed = self.filter.observe(
            filter_state,
            agents,
            scene,
            current,
            history[:, -2] if previous_current is None and history.shape[1] > 1
            else previous_current,
            current_valid,
        )
        rebased = rebase_soft_preview(
            current[:, 1:],
            soft_reference[:, : self.cfg.preview_frames],
            reference_base,
            dt_s=self.cfg.dt_s,
            velocity_horizon_s=self.cfg.velocity_rebase_horizon_s,
            endpoint_offset_weight=self.cfg.soft_anchor_final_weight,
        )
        preview_agents, preview_scene = self._preview_context(
            current, current_valid, rebased, reference_base
        )
        conditioned_agents = agents.clone()
        conditioned_agents[:, 1:] += preview_agents
        conditioned_scene = scene + preview_scene
        scene_mean = self.filter.prior_scene(observed, conditioned_scene)
        refresh = slow_scene is None or (
            int(response_index) % self.cfg.scene_refresh_responses == 0
        )
        stochastic = self.cfg.stochastic_latents and not deterministic
        scene_noise = (
            torch.randn_like(scene_mean)
            if scene_standard_normal is None
            else scene_standard_normal.to(scene_mean)
        )
        if refresh:
            slow_scene_noise = scene_noise
            slow_scene = scene_mean + (
                self.cfg.scene_noise_scale * scene_noise if stochastic else 0.0
            )
        assert slow_scene is not None
        if slow_scene_noise is None:
            slow_scene_noise = torch.zeros_like(scene_noise)
        agent_mean = self.filter.prior_agents(
            observed, conditioned_agents, slow_scene
        )
        raw_agent_noise = (
            torch.randn_like(agent_mean)
            if agent_standard_normal is None
            else agent_standard_normal.to(agent_mean)
        )
        if self.cfg.graph_coupled_latent_enabled:
            transition = self.latent_transition(
                agent_noise_state,
                raw_agent_noise,
                conditioned_agents,
                slow_scene,
                current,
                current_valid,
            )
            agent_noise = transition.state
            agent_log_prob = transition.innovation_log_prob
            graph_message = transition.graph_message
        else:
            agent_noise = raw_agent_noise
            agent_log_prob = raw_agent_noise.new_zeros(raw_agent_noise.shape[:2])
            graph_message = raw_agent_noise.new_zeros(raw_agent_noise.shape)
        # Style is a world-level exogenous variable: sample it once and retain
        # it across response boundaries and counterfactual twins.  It must not
        # be recomputed from post-intervention graph state, otherwise a shared
        # random seed can still create a spurious branch difference.
        if agent_style_state is None:
            agent_style_state = raw_agent_noise
        else:
            agent_style_state = agent_style_state.to(agent_mean)
        agent_style_state = agent_style_state * current_valid[..., None].float()
        agent_latent = agent_mean + (
            self.cfg.agent_noise_scale * agent_noise if stochastic else 0.0
        )
        agent_latent = agent_latent * current_valid[..., None].float()
        if self.cfg.use_soft_plan:
            soft_actions = soft_reference_controls(
                current[:, 1:],
                rebased,
                dt_s=self.cfg.dt_s,
                min_acceleration=self.cfg.min_acceleration_mps2,
                max_acceleration=self.cfg.max_acceleration_mps2,
                max_yaw_rate=self.cfg.max_yaw_rate_rps,
            )
        else:
            initial = self.dynamics.controls_from_highd_actions(
                current[:, 1:, 4:6], current[:, 1:]
            )
            soft_actions = initial[:, None].expand(
                -1, self.cfg.preview_frames, -1, -1
            )
        ego_control = self.dynamics.controls_from_highd_actions(
            current[:, 0, 4:6], current[:, 0]
        )
        if committed_ego_controls is not None:
            if (
                committed_ego_controls.ndim != 3
                or committed_ego_controls.shape[0] != current.shape[0]
                or committed_ego_controls.shape[-1] != 2
            ):
                raise ValueError(
                    "committed_ego_controls must have shape [batch,past_frames,2]"
                )
            history_count = min(
                self.cfg.intervention_trigger_history_frames,
                committed_ego_controls.shape[1] - 1,
            )
            if history_count:
                recent_controls = committed_ego_controls[:, -history_count - 1 :]
                deviation = recent_controls[:, -1, 0] - recent_controls[
                    :, :-1, 0
                ].mean(dim=1)
                # Preserve evidence for a newly committed lateral command
                # until the next response boundary.  A mean would dilute a
                # 0.08 rad/s command to 0.016 after four committed frames;
                # the sign-aware preceding-window extreme is a causal step
                # detector and does not react to settled yaw by itself.
                recent_yaw = recent_controls[:, :, 1]
                latest_yaw = recent_yaw[:, -1]
                preceding_yaw = recent_yaw[:, :-1]
                reference_yaw = torch.where(
                    latest_yaw >= 0.0,
                    preceding_yaw.min(dim=1).values,
                    preceding_yaw.max(dim=1).values,
                )
                lateral_deviation = latest_yaw - reference_yaw
            else:
                deviation = torch.zeros_like(ego_control[:, 0])
                lateral_deviation = torch.zeros_like(ego_control[:, 1])
        else:
            # A direct model call has no externally committed action buffer.
            # All rollout and simulation paths pass the buffer explicitly:
            # reconstructing controls from integrated state causes numerical
            # false interventions during factual replay.
            history_count = min(
                self.cfg.intervention_trigger_history_frames, history.shape[1] - 1
            )
            if history_count:
                recent_ego = history[:, -history_count - 1 : -1, 0]
                recent_controls = self.dynamics.controls_from_highd_actions(
                    recent_ego[..., 4:6], recent_ego
                )
                deviation = ego_control[:, 0] - recent_controls[..., 0].mean(dim=1)
                lateral_deviation = (
                    ego_control[:, 1] - recent_controls[..., 1].mean(dim=1)
                )
            else:
                deviation = torch.zeros_like(ego_control[:, 0])
                lateral_deviation = torch.zeros_like(ego_control[:, 1])
        intervention_trigger = torch.sign(deviation) * torch.relu(
            deviation.abs() - self.cfg.intervention_trigger_threshold_mps2
        )
        # A discontinuous/invalid ego history has no meaningful committed
        # control difference.  Suppressing the adapter there prevents a
        # missing highD ego frame from becoming a fictitious ADS intervention.
        valid_count = min(history_count + 1, history_valid.shape[1])
        ego_history_valid = history_valid[:, -valid_count:, 0].all(dim=1)
        intervention_trigger = intervention_trigger * (
            ego_history_valid & current_valid[:, 0]
        ).to(intervention_trigger)
        lateral_intervention_trigger = torch.sign(lateral_deviation) * torch.relu(
            lateral_deviation.abs()
            - self.cfg.lateral_intervention_trigger_threshold_rps
        )
        lateral_intervention_trigger = lateral_intervention_trigger * (
            ego_history_valid & current_valid[:, 0]
        ).to(lateral_intervention_trigger)
        if intervention_memory is None:
            intervention_memory = torch.zeros_like(intervention_trigger)
        else:
            intervention_memory = intervention_memory.to(intervention_trigger)
        intervention_memory = (
            self.cfg.intervention_memory_decay * intervention_memory
            + intervention_trigger
        ).clamp(-4.0, 4.0)
        if lateral_intervention_memory is None:
            lateral_intervention_memory = torch.zeros_like(lateral_intervention_trigger)
        else:
            lateral_intervention_memory = lateral_intervention_memory.to(
                lateral_intervention_trigger
            )
        lateral_intervention_memory = (
            self.cfg.lateral_intervention_memory_decay * lateral_intervention_memory
            + lateral_intervention_trigger
        ).clamp(-1.0, 1.0)
        response_field_gain = None
        if self.cfg.causal_response_field_enabled:
            response_field_gain = self.decoder.response_field(
                agents[:, 1:],
                observed.agent_hidden[:, 1:],
                preview_agents,
                agent_latent[:, 1:],
                graph_message[:, 1:],
                intervention_memory,
            )
        actions, mean, reference_actions = self.decoder(
            agents,
            preview_agents,
            preview_scene,
            observed,
            slow_scene,
            agent_latent,
            current,
            current_valid,
            soft_actions,
            # The response network only observes this control after it has
            # been integrated into ``current`` at the preceding boundary.
            ego_control[:, 0].clamp(-8.0, 8.0),
            intervention_memory,
            lateral_intervention_memory,
            response_relevance(current, current_valid),
            slow_scene_noise,
            agent_noise,
            agent_style_state,
            stochastic,
            response_field_gain,
            self.response_sensitivity_bounds,
            self.cfg.intervention_adapter_enabled if apply_intervention_adapter is None else bool(apply_intervention_adapter),
            True if apply_explicit_ego_response is None else bool(apply_explicit_ego_response),
        )
        mean = mean[:, : self.cfg.execute_frames]
        actions = actions[:, : self.cfg.execute_frames]
        minimum_std = mean.new_tensor(
            (self.cfg.min_acceleration_std_mps2, self.cfg.min_yaw_rate_std_rps)
        )
        std = torch.maximum(self.log_std.exp(), minimum_std)
        std = std[None, None, None].expand_as(mean)
        return ResponseDistribution(
            mean=mean,
            std=std * current_valid[:, None, 1:, None].float(),
            actions=actions,
            states=self._integrate(current, actions, current_valid),
            rebased_preview=rebased,
            reference_actions=reference_actions[:, : self.cfg.execute_frames],
            filter_state=observed,
            slow_scene=slow_scene,
            slow_scene_noise=slow_scene_noise,
            agent_noise_state=agent_noise,
            agent_style_state=agent_style_state,
            agent_latent=agent_latent,
            agent_latent_log_prob=agent_log_prob,
            graph_latent_message=graph_message,
            response_field_gain=response_field_gain,
            intervention_trigger=intervention_trigger,
            intervention_memory=intervention_memory,
            lateral_intervention_memory=lateral_intervention_memory,
            scene_refreshed=refresh,
        )

    def checkpoint_payload(self) -> dict[str, object]:
        return {
            "model_type": self.model_type,
            "model_config": self.cfg.to_dict(),
            "state_dict": self.state_dict(),
        }
