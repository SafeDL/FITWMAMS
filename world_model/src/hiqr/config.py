"""Configuration for the independently maintained HiQR-WM."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HiQRWorldModelConfig:
    """Architecture and objective settings for HiQR-WM.

    The fixed highD tensor layout is ``[ego, six background slots]``.  Values
    intentionally do not inherit QR's deprecated anchor/refinement settings.
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
    max_yaw_rate: float = 0.60
    position_weight: float = 1.0
    velocity_weight: float = 0.25
    action_weight: float = 0.20
    plan_position_weight: float = 0.25
    plan_action_weight: float = 0.10
    continuation_weight: float = 0.10
    gate_weight: float = 0.01
    interaction_weight: float = 0.12
    physical_weight: float = 0.03
    jerk_weight: float = 0.02
    lane_weight: float = 0.02
    gap_ttc_weight: float = 0.05
    scene_kl_weight: float = 0.02
    agent_kl_weight: float = 0.02
    diversity_weight: float = 0.01

    def __post_init__(self) -> None:
        if int(self.plan_frames) != 25 or int(self.start_reconstruction_frames) != 25:
            raise ValueError("HiQR-WM uses the fixed 25-frame START protocol")
        if int(self.execute_frames) != 5:
            raise ValueError("HiQR-WM uses 5-frame (0.20 s) responses")
        if float(self.simulation_dt_s) != 0.04:
            raise ValueError("HiQR-WM uses 25 Hz dynamics (dt=0.04 s)")

    @property
    def anchor_state_index(self) -> int:
        """Index of the last observed state before the 149 ROLL transitions."""
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
