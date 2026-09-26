"""Configuration for the independent prior-driven HiQR model."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HiQRConfig:
    """Architecture and objective settings for HiQR.

    The model uses a fixed ego-plus-six-background highD tensor contract and
    an independent checkpoint schema.
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

    # Accepted pre-experiment: five physical jerk knots per one-second plan.
    jerk_knots: int = 5
    max_longitudinal_jerk: float = 5.0
    max_yaw_jerk: float = 0.5

    # Accepted observation filter and two-time-scale hierarchy.
    scene_mode_responses: int = 5
    scene_noise_scale: float = 0.17
    residual_noise_scale: float = 0.06

    # Prior closed-loop objective.
    position_weight: float = 1.0
    velocity_weight: float = 0.25
    action_weight: float = 0.20
    plan_position_weight: float = 0.25
    plan_action_weight: float = 0.10
    interaction_weight: float = 0.12
    jerk_weight: float = 0.02
    lane_weight: float = 0.02
    gap_ttc_weight: float = 0.05
    b0_summary_weight: float = 0.10

    def __post_init__(self) -> None:
        if int(self.plan_frames) != 25 or int(self.start_reconstruction_frames) != 25:
            raise ValueError("HiQR uses the fixed 25-frame START protocol")
        if int(self.execute_frames) != 5:
            raise ValueError("HiQR uses 5-frame (0.20 s) responses")
        if int(self.rollout_frames) != 149 or float(self.simulation_dt_s) != 0.04:
            raise ValueError("HiQR uses the canonical 149-transition 25 Hz protocol")
        if int(self.scene_mode_responses) != 5:
            raise ValueError("HiQR holds each scene mode for five responses")
        if int(self.jerk_knots) < 2:
            raise ValueError("HiQR requires at least two jerk knots")
        if float(self.max_longitudinal_jerk) <= 0.0 or float(self.max_yaw_jerk) <= 0.0:
            raise ValueError("HiQR jerk limits must be positive")
        if (
            float(self.scene_noise_scale) < 0.0
            or float(self.residual_noise_scale) < 0.0
        ):
            raise ValueError("HiQR innovation scales must be non-negative")

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
