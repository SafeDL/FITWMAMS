"""Paper-locked parameters and explicit longitudinal-adaptation choices."""
from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class ActiveInferenceConfig:
    """Parameters from Schumann et al. (2026), Table 1 unless marked adapted."""

    native_dt_s: float = 0.04
    decision_dt_s: float = 0.2
    horizon: int = 30
    particles: int = 75
    cem_iterations: int = 10
    cem_plans: int = 100
    elite_fraction: float = 0.1
    action_sample_sd: float = 5.0
    acceleration_limit: float = 8.0
    belief_acceleration_sd: float = 3.0
    prediction_acceleration_sd: float = 0.6
    looming_threshold_per_s: float = 0.00215
    norm_prediction_horizon: int = 20
    desired_speed_sd: float = 0.5
    acceleration_sd: float = 0.1
    collision_cost: float = -10000.0
    evidence_drift: float = 10 ** -5.95
    evidence_threshold: float = 1.0
    coast_acceleration_mps2: float = -0.1
    pedal_hold_s: float = 0.2
    enforce_pedal_constraint: bool = True
    safety_response_s: float = 1.0
    safety_leader_brake_mps2: float = -5.0
    perception_gap_sd_m: float = 0.02
    perception_speed_sd_mps: float = 0.02
    longitudinal_norm_acceleration_sd: float = 0.20
    seed: int = 20260916
    # This is deliberately an explicit adaptation, not a paper parameter.
    no_steering_adaptation: bool = True

    @property
    def native_ticks_per_decision(self) -> int:
        ticks = round(self.decision_dt_s / self.native_dt_s)
        if abs(ticks * self.native_dt_s - self.decision_dt_s) > 1e-10:
            raise ValueError("decision_dt_s must be an integer number of native ticks")
        return ticks

    def reduced(self, *, plans: int = 20, iterations: int = 3, particles: int = 16) -> "ActiveInferenceConfig":
        """Small deterministic setting used only by unit tests and smoke runs."""
        return replace(self, cem_plans=plans, cem_iterations=iterations, particles=particles)
