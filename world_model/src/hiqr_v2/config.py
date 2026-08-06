"""Configuration for the independent prior-driven HiQR-v2 model."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HiQRV2Config:
    """Architecture and objective settings for HiQR-v2.

    V2 uses the same ego-plus-six-background highD tensor contract as HiQR-v1,
    but intentionally has a new model/checkpoint schema.
    """

    hidden_dim: int = 128
    scene_latent_dim: int = 16
    agent_residual_dim: int = 16
    temporal_layers: int = 1
    attention_layers: int = 2
    decoder_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.10
    plan_frames: int = 25
    execute_frames: int = 5
    rollout_frames: int = 149
    start_reconstruction_frames: int = 25
    response_interval_s: float = 0.20
    simulation_dt_s: float = 0.04
    lane_width_m: float = 3.6
    min_acceleration: float = -8.0
    max_acceleration: float = 4.0
    max_yaw_rate: float = 0.6

    # Observation filter and two-time-scale hierarchy.
    scene_mode_responses: int = 5
    residual_noise_scale: float = 0.35
    hard_carry_frames: int = 5
    carry_replan_frames: int = 15
    emergency_ego_position_error_m: float = 1.0
    emergency_ego_velocity_error_mps: float = 2.0
    emergency_gate_sharpness: float = 4.0
    continuation_mode: str = "adaptive_5_15_5"
    # Target-aware physical barriers are admitted only after their diagnostic
    # gate; the first canonical V2 run therefore starts with them disabled.
    physical_mode: str = "off"
    filter_update_mode: str = "observed"

    # Prior closed-loop objective.
    position_weight: float = 1.0
    velocity_weight: float = 0.25
    action_weight: float = 0.20
    plan_position_weight: float = 0.25
    plan_action_weight: float = 0.10
    interaction_weight: float = 0.12
    physical_weight: float = 0.03
    jerk_weight: float = 0.02
    lane_weight: float = 0.02
    gap_ttc_weight: float = 0.05
    b0_summary_weight: float = 0.10

    # Posterior is strictly local auxiliary supervision.
    posterior_aux_weight: float = 0.10
    prior_distillation_weight: float = 0.05
    scene_kl_weight: float = 0.05
    agent_kl_weight: float = 0.05
    diversity_weight: float = 0.01

    def __post_init__(self) -> None:
        if int(self.plan_frames) != 25 or int(self.start_reconstruction_frames) != 25:
            raise ValueError("HiQR-v2 uses the fixed 25-frame START protocol")
        if int(self.execute_frames) != 5:
            raise ValueError("HiQR-v2 uses 5-frame (0.20 s) responses")
        if int(self.rollout_frames) != 149 or float(self.simulation_dt_s) != 0.04:
            raise ValueError("HiQR-v2 uses the canonical 149-transition 25 Hz protocol")
        if int(self.scene_mode_responses) < 1:
            raise ValueError("scene_mode_responses must be positive")
        if int(self.hard_carry_frames) != int(self.execute_frames):
            raise ValueError("hard_carry_frames must equal execute_frames")
        if int(self.hard_carry_frames) + int(self.carry_replan_frames) != 20:
            raise ValueError(
                "HiQR-v2 requires 5 hard-carry plus 15 replan carry frames"
            )
        if (
            float(self.emergency_ego_position_error_m) <= 0.0
            or float(self.emergency_ego_velocity_error_mps) <= 0.0
            or float(self.emergency_gate_sharpness) <= 0.0
        ):
            raise ValueError("HiQR-v2 emergency carry thresholds must be positive")
        if self.continuation_mode not in {"adaptive_5_15_5", "full_replan"}:
            raise ValueError("continuation_mode must be adaptive_5_15_5 or full_replan")
        if self.physical_mode not in {"target_aware", "off"}:
            raise ValueError("physical_mode must be target_aware or off")
        if self.filter_update_mode not in {"observed", "stateless"}:
            raise ValueError("filter_update_mode must be observed or stateless")

    @property
    def anchor_state_index(self) -> int:
        return int(self.start_reconstruction_frames) - 1

    @property
    def first_future_state_index(self) -> int:
        return int(self.start_reconstruction_frames)

    @property
    def response_steps(self) -> int:
        return (int(self.rollout_frames) + int(self.execute_frames) - 1) // int(
            self.execute_frames
        )

    def rollout_frames_for_responses(self, response_steps: int) -> int:
        return min(
            int(self.rollout_frames), int(response_steps) * int(self.execute_frames)
        )

    def rollout_seconds_for_responses(self, response_steps: int) -> float:
        return self.rollout_frames_for_responses(response_steps) * float(
            self.simulation_dt_s
        )
