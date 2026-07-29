"""Configuration for the map-free FIRM-WM implementation."""

from dataclasses import dataclass


@dataclass(frozen=True)
class FIRMConfig:
    hidden_dim: int = 160
    dropout: float = 0.10
    world_latent_dim: int = 24
    action_flow_layers: int = 4
    plan_frames: int = 25
    execute_frames: int = 5
    response_interval_s: float = 0.2
    simulation_dt_s: float = 0.04
    lane_width_m: float = 3.6
    max_longitudinal_jerk: float = 5.0
    max_yaw_jerk: float = 0.50
    min_acceleration: float = -8.0
    max_acceleration: float = 4.0
    max_yaw_rate: float = 0.60
    prefix_nll_weight: float = 0.20
    roll_weight: float = 1.00
    velocity_weight: float = 0.25
    control_weight: float = 0.15
    behavior_anchor_weight: float = 0.50
    overlap_weight: float = 0.10
    interaction_weight: float = 0.15
    physical_weight: float = 0.05
    plan_horizon_control_weight: float = 0.20
    plan_horizon_state_weight: float = 0.50
    plan_horizon_state_interval_responses: int = 5
    sampled_physical_weight: float = 0.15
    sampled_physical_batch_size: int = 64
    latent_variance_weight: float = 0.005

    @property
    def physics_steps_per_response(self) -> int:
        return max(1, int(round(self.response_interval_s / self.simulation_dt_s)))

    @property
    def response_steps(self) -> int:
        return 125 // self.physics_steps_per_response
