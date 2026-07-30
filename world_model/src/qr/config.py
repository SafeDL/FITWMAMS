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
    # control refiner.  Both values are consumed by real MultiheadAttention
    # modules rather than being descriptive-only configuration.
    attention_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.10
    plan_frames: int = 25
    execute_frames: int = 5
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
    refinement_noise_levels: tuple[float, ...] = (0.0, 0.25, 0.50, 1.0)
    denoising_acceleration_std: float = 1.5
    denoising_yaw_rate_std: float = 0.15
    position_weight: float = 1.0
    velocity_weight: float = 0.25
    control_weight: float = 0.20
    plan_position_weight: float = 0.25
    plan_control_weight: float = 0.10
    refinement_weight: float = 0.35
    denoising_weight: float = 0.15
    overlap_weight: float = 0.10
    interaction_weight: float = 0.12
    physical_weight: float = 0.03
    behavior_kl_weight: float = 0.02
    behavior_reconstruction_weight: float = 0.08
    diversity_weight: float = 0.01

    @property
    def response_steps(self) -> int:
        return 125 // int(self.execute_frames)
