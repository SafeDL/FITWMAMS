"""Configuration for the standalone RAMP-WM implementation."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RAMPConfig:
    hidden_dim: int = 128
    temporal_layers: int = 1
    dropout: float = 0.10
    num_candidates: int = 8
    plan_frames: int = 25
    execute_frames: int = 5
    jerk_controls: int = 5
    response_interval_s: float = 0.2
    simulation_dt_s: float = 0.04
    max_longitudinal_jerk: float = 5.0
    max_yaw_jerk: float = 0.50
    hard_jerk_projection: bool = True
    max_acceleration: float = 4.0
    min_acceleration: float = -8.0
    use_start_anchor: bool = True
    mixture_temperature: float = 0.25
    position_weight: float = 1.0
    velocity_weight: float = 0.25
    control_weight: float = 0.20
    mixture_weight: float = 0.50
    probability_weight: float = 0.10
    overlap_weight: float = 0.15
    joint_weight: float = 0.10
    diversity_weight: float = 0.02
    smoothness_weight: float = 0.01

    @property
    def physics_steps_per_response(self) -> int:
        return max(1, int(round(self.response_interval_s / self.simulation_dt_s)))

    @property
    def response_steps(self) -> int:
        return 125 // self.physics_steps_per_response
