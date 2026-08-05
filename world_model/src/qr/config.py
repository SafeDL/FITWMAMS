"""Configuration for the independent Query-Refine World Model."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QRWorldModelConfig:
    """Architecture and objective settings for QR-WM.

    The model operates on the repository's fixed highD scene tensor
    ``[ego, six background slots]``.  It intentionally carries no traffic
    light input; road information is supplied only as map polylines and lane
    topology.
    """

    hidden_dim: int = 128
    behavior_latent_dim: int = 16
    temporal_layers: int = 1
    # Shared by the relation-aware scene encoder and the joint agent-time
    # future-action refiner.  Both values are consumed by real MultiheadAttention
    # modules rather than being descriptive-only configuration.
    attention_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.10
    plan_frames: int = 25
    execute_frames: int = 5
    # A highD natural window stores S0..S149.  QR reconstructs the first 25
    # transitions from B0, then rolls the remaining 124 transitions without
    # raw B0: 1.00 s START + 4.96 s ROLL = 5.96 s total.
    rollout_frames: int = 149
    start_reconstruction_frames: int = 25
    response_interval_s: float = 0.20
    simulation_dt_s: float = 0.04
    lane_width_m: float = 3.6
    min_acceleration: float = -8.0
    max_acceleration: float = 4.0
    max_yaw_rate: float = 0.60
    refinement_iterations: int = 2
    buffer_carry_mix: float = 0.35
    start_anchor_mix: float = 0.75
    start_summary_weight: float = 0.10
    start_training_fraction: float = 0.50
    position_weight: float = 1.0
    velocity_weight: float = 0.25
    action_weight: float = 0.20
    plan_position_weight: float = 0.25
    plan_action_weight: float = 0.10
    refinement_weight: float = 0.35
    overlap_weight: float = 0.10
    interaction_weight: float = 0.12
    physical_weight: float = 0.03
    behavior_kl_weight: float = 0.02
    behavior_reconstruction_weight: float = 0.08
    diversity_weight: float = 0.01

    @property
    def response_steps(self) -> int:
        return (int(self.rollout_frames) + int(self.execute_frames) - 1) // int(self.execute_frames)

    @property
    def roll_frames(self) -> int:
        return int(self.rollout_frames) - int(self.start_reconstruction_frames)

    def rollout_frames_for_responses(self, response_steps: int) -> int:
        return min(int(self.rollout_frames), int(response_steps) * int(self.execute_frames))

    def rollout_seconds_for_responses(self, response_steps: int) -> float:
        return self.rollout_frames_for_responses(response_steps) * float(self.simulation_dt_s)
