"""Single-pass, causal MA-IDM safety response around the current HiQR action.

The controller never consumes a precomputed no-intervention trajectory. HiQR
remains the action source unless the *realized* leader/follower state predicts
an unsafe gap and its action does not already brake sufficiently.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.special import ndtr

from external_model_baselines.models.bayesian_ma_idm.src.model import (
    LOWER,
    UPPER,
    load_posterior,
)

from ..reaction_controller import (
    ManeuverPhase,
    ReactionController,
    ReactionControllerContext,
    ReactionControllerOutput,
)
from ..influence_graph import DynamicInteractionRelations
from ..core.dynamics import KinematicTrafficDynamics
from ..reference import planned_lateral_position_at_x
from ..scene_geometry import (
    torch_adjacent_lane_index,
    torch_map_lane_geometry,
    torch_nearest_lane_geometry,
)
from .mobil_response import MOBILDecision, evaluate_mobil_candidates
from .driver_policy import (
    corrected_acceleration,
    causal_maneuver_features,
    exclusive_lane_probabilities,
    response_history_features,
    physical_leader_observation,
    sample_lane_choice,
)
from .lane_choice import lane_choice_features
from .lane_trajectory import (
    LearnedLaneTrajectory,
    frenet_feedback,
    limit_realized_lateral_acceleration,
    path_clearance,
)
from ..response_gate import (
    MODE_MANEUVER,
    MODE_TRACK,
    ResponseGate,
    ResponseGateDecision,
    response_gate_features,
)

DEFAULT_MA_IDM_POSTERIOR = (
    Path(__file__).resolve().parents[3]
    / "external_model_baselines/models/bayesian_ma_idm/evidence/deployment/ma_idm_all_251.npz"
)


def calibrated_mobil_parameters(fit):
    """Use the same frozen decision parameters in GT and generated worlds."""
    return {
        "mobil_politeness": fit["politeness"],
        "mobil_plan_keep_bias_mps2": 0.0,
        "mobil_onset_calibration": (fit["bias"], fit["slope"]),
        "mobil_parameter_variation": False,
        "lane_changes_on_response_only": True,
        "driver_correction": fit.get("correction"),
        "lane_style": fit.get("lane_style"),
        "lane_model": fit.get("lane_model"),
    }


def mobil_parameters_from_maneuver_latent(
    maneuver_latent: np.ndarray | torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Map persistent standard-normal traits to bounded MOBIL parameters."""
    latent = torch.as_tensor(maneuver_latent, dtype=torch.float32)
    if latent.ndim not in (2, 3) or tuple(latent.shape[-2:]) != (6, 4):
        raise ValueError("maneuver_latent must have shape [6,4] or [batch,6,4]")
    probability = torch.sigmoid(latent)
    return {
        "politeness": 0.05 + 0.55 * probability[..., 0],
        "minimum_incentive_mps2": 0.05 + 0.75 * probability[..., 1],
        "safe_braking_mps2": 2.5 + 3.5 * probability[..., 2],
        "plan_keep_bias_mps2": 0.02 + 0.28 * probability[..., 3],
    }


def sample_population_theta_from_world(
    posterior_path: str | Path,
    exogenous_state: Any,
    *,
    slots: int = 6,
    use_driver_latent: bool = True,
) -> np.ndarray:
    """Draw replayable fixed episode parameters from explicit world randomness."""
    innovations = np.asarray(exogenous_state.agent_response_innovations)
    if (
        innovations.ndim != 4
        or innovations.shape[1] < 1
        or innovations.shape[2] < slots + 1
    ):
        raise ValueError("world innovations do not cover all NPC driver slots")
    posterior = load_posterior(posterior_path)
    if use_driver_latent:
        latent = np.asarray(exogenous_state.driver_latent, np.float64)
        if latent.shape != (innovations.shape[0], slots, 5):
            raise ValueError("driver_latent must have shape [batch,slots,5]")
        means = np.asarray(posterior["population_log_mean"], np.float64)[:, :5]
        covariances = np.asarray(posterior["population_log_cov"], np.float64)[:, :5, :5]
        mean = means.mean(axis=0)
        centered = means - mean
        covariance = covariances.mean(axis=0) + centered.T @ centered / len(means)
        eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (covariance + covariance.T))
        transform = eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 1.0e-8)))
        log_theta = mean + latent @ transform.T
        return np.clip(np.exp(log_theta), LOWER[:5], UPPER[:5]).astype(np.float32)
    # Legacy W0 parameter draws reuse the fixed response-noise block.
    latent = innovations[:, 0, 1 : slots + 1].astype(np.float64)
    uniform = ndtr(latent)
    if "driver_parameter_draws" in posterior:
        draws = np.asarray(posterior["driver_parameter_draws"])
        sample_index = np.minimum(
            (uniform[..., 0] * len(draws)).astype(np.int64), len(draws) - 1
        )
        driver_index = np.minimum(
            (uniform[..., 1] * draws.shape[1]).astype(np.int64), draws.shape[1] - 1
        )
        return draws[sample_index, driver_index, :5].astype(np.float32)

    means = np.asarray(posterior["population_log_mean"], np.float64)
    covariances = np.asarray(posterior["population_log_cov"], np.float64)
    draw_index = np.minimum(
        (uniform[..., 0] * len(means)).astype(np.int64), len(means) - 1
    )
    mean = means[draw_index, :5]
    covariance = covariances[draw_index, :5, :5]
    eigenvalues, eigenvectors = np.linalg.eigh(
        0.5 * (covariance + np.swapaxes(covariance, -1, -2))
    )
    transform = eigenvectors * np.sqrt(np.maximum(eigenvalues, 1.0e-8))[..., None, :]
    log_theta = mean + np.einsum("...ij,...j->...i", transform, latent[..., 1:6])
    return np.clip(np.exp(log_theta), LOWER[:5], UPPER[:5]).astype(np.float32)


def nearest_leader_observation(
    states: torch.Tensor,
    valid: torch.Tensor,
    *,
    lane_half_width_m: float = 1.8,
    vehicle_length_m: float = 4.8,
    vehicle_lengths_m: torch.Tensor | None = None,
    vehicle_widths_m: torch.Tensor | None = None,
    target_y: torch.Tensor | None = None,
    relations: DynamicInteractionRelations | None = None,
    prediction_horizon_s: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return current gap, follower/leader speeds and leader slot per NPC."""
    if states.ndim != 3 or states.shape[1:] != (7, 6):
        raise ValueError("states must have shape [batch,7,6]")
    if valid.shape != states.shape[:2]:
        raise ValueError("valid must have shape [batch,7]")
    if vehicle_widths_m is not None:
        dimensions = torch.stack((vehicle_lengths_m, vehicle_widths_m), -1)
        gap, speed, leader_speed, leader = physical_leader_observation(
            states, valid, dimensions, horizon_s=prediction_horizon_s, target_y=target_y
        )
        return (
            gap[:, 1:].clamp_min(0.05),
            speed[:, 1:],
            leader_speed[:, 1:],
            leader[:, 1:],
        )
    if relations is not None:
        leader_index = relations.leader_index[:, 1:]
        has_leader = leader_index.ge(0)
        speed_all = torch.linalg.vector_norm(states[..., 2:4], dim=-1)
        follower_speed = speed_all[:, 1:]
        leader_speed = torch.gather(speed_all, 1, leader_index.clamp_min(0))
        return (
            relations.leader_gap_m[:, 1:],
            follower_speed,
            torch.where(has_leader, leader_speed, follower_speed),
            leader_index,
        )
    follower = states[:, 1:]
    dx = states[:, None, :, 0] - follower[:, :, None, 0]
    relative_y = states[:, None, :, 1] - follower[:, :, None, 1]
    future_relative_y = relative_y + (
        states[:, None, :, 3] - follower[:, :, None, 3]
    ) * float(prediction_horizon_s)
    lateral_overlap = torch.minimum(relative_y.abs(), future_relative_y.abs()).lt(
        float(lane_half_width_m)
    ) | (relative_y * future_relative_y <= 0.0)
    candidates = valid[:, None, :] & valid[:, 1:, None] & dx.gt(0.0) & lateral_overlap
    identity = torch.arange(6, device=states.device)[None, :, None] + 1
    candidates = candidates & torch.arange(7, device=states.device)[None, None, :].ne(
        identity
    )
    speed_all = torch.linalg.vector_norm(states[..., 2:4], dim=-1)
    follower_speed = speed_all[:, 1:]
    horizon = float(prediction_horizon_s)
    if vehicle_lengths_m is None:
        clearance_length = torch.full_like(dx, float(vehicle_length_m))
    else:
        lengths = torch.as_tensor(
            vehicle_lengths_m, dtype=states.dtype, device=states.device
        )
        if lengths.shape != states.shape[:2]:
            raise ValueError("vehicle_lengths_m must have shape [batch,7]")
        clearance_length = 0.5 * (lengths[:, 1:, None] + lengths[:, None, :])
    projected_gap = (
        dx
        - clearance_length
        + (speed_all[:, None] - follower_speed[..., None]) * horizon
        + 0.5 * (states[:, None, :, 4] - follower[..., 4, None]) * horizon * horizon
    )
    unsafe_projection = candidates & projected_gap.lt(
        torch.maximum(
            torch.full_like(follower_speed[..., None], 2.0),
            0.1 * follower_speed[..., None],
        )
    )
    # The geometrically closest prospective leader need not be the dangerous
    # one: a target-lane car pulling away can mask a slightly farther but
    # hard-braking current-lane leader. Prefer the worst projected clearance
    # whenever one is unsafe, then fall back to nearest distance.
    nearest_distance = torch.where(candidates, dx, torch.full_like(dx, float("inf")))
    risk_distance = torch.where(
        unsafe_projection, projected_gap, torch.full_like(dx, float("inf"))
    )
    leader_index = torch.where(
        unsafe_projection.any(dim=-1),
        risk_distance.argmin(dim=-1),
        nearest_distance.argmin(dim=-1),
    )
    has_leader = candidates.any(dim=-1)
    selected_dx = torch.gather(dx, 2, leader_index[..., None]).squeeze(-1)
    leader_speed = torch.gather(speed_all, 1, leader_index)
    gap = torch.where(
        has_leader,
        (
            selected_dx
            - torch.gather(clearance_length, 2, leader_index[..., None]).squeeze(-1)
        ).clamp_min(0.05),
        torch.full_like(selected_dx, 1.0e6),
    )
    leader_speed = torch.where(has_leader, leader_speed, follower_speed)
    leader_index = torch.where(
        has_leader, leader_index, torch.full_like(leader_index, -1)
    )
    return gap, follower_speed, leader_speed, leader_index


def planned_adjacent_lane_intent(
    origin_y: torch.Tensor,
    terminal_y: torch.Tensor,
    map_polylines: torch.Tensor | None,
    map_polyline_valid: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Infer a map-grounded adjacent lane from the generated terminal plan.

    A partially completed lateral manoeuvre is not automatically a command
    to snap a vehicle one nominal lane width from its *current* position.
    """
    if map_polylines is None or map_polyline_valid is None:
        displacement = terminal_y - origin_y
        intent = (displacement.abs() > 1.8) & (displacement.abs() < 5.4)
        return intent, origin_y, origin_y + torch.sign(displacement) * 3.6
    lane_y, lane_present, lane_width = torch_map_lane_geometry(
        map_polylines, map_polyline_valid
    )
    source_index, source_y, source_width, map_available = torch_nearest_lane_geometry(
        origin_y, lane_y, lane_present, lane_width
    )
    target_index, target_y, target_width, _ = torch_nearest_lane_geometry(
        terminal_y, lane_y, lane_present, lane_width
    )
    lower_index, lower_available = torch_adjacent_lane_index(
        source_index, lane_y, lane_present, lane_width, side=-1
    )
    upper_index, upper_available = torch_adjacent_lane_index(
        source_index, lane_y, lane_present, lane_width, side=1
    )
    adjacent_target = (lower_available & target_index.eq(lower_index)) | (
        upper_available & target_index.eq(upper_index)
    )
    intent = (
        map_available
        & adjacent_target
        & (origin_y - source_y).abs().lt(0.45 * source_width)
        & (terminal_y - target_y).abs().lt(0.45 * target_width)
    )
    return intent, source_y, target_y


def _context_vehicle_dimensions(
    context: ReactionControllerContext,
    *,
    fallback_length_m: float,
    fallback_width_m: float,
) -> torch.Tensor:
    dimensions = (
        torch.empty(
            (*context.current.shape[:2], 2),
            dtype=context.current.dtype,
            device=context.current.device,
        )
        if context.vehicle_dimensions is None
        else torch.as_tensor(
            context.vehicle_dimensions,
            dtype=context.current.dtype,
            device=context.current.device,
        )
    )
    if context.vehicle_dimensions is None:
        dimensions[..., 0] = float(fallback_length_m)
        dimensions[..., 1] = float(fallback_width_m)
    if dimensions.shape != (*context.current.shape[:2], 2):
        raise ValueError("vehicle_dimensions must have shape [batch,7,2]")
    return dimensions


def _target_lane_width(
    context: ReactionControllerContext,
    target_y: torch.Tensor,
    *,
    fallback_width_m: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if context.map_polylines is None or context.map_polyline_valid is None:
        return torch.full_like(target_y, float(fallback_width_m)), torch.ones_like(
            target_y, dtype=torch.bool
        )
    lane_y, lane_present, lane_width = torch_map_lane_geometry(
        context.map_polylines,
        context.map_polyline_valid,
        fallback_width_m=fallback_width_m,
    )
    _, _, width, available = torch_nearest_lane_geometry(
        target_y, lane_y, lane_present, lane_width
    )
    return width, available


class OnlineMAIDMController(ReactionController):
    """Online traffic-aware longitudinal response for every NPC slot.

    Sampled driver parameters are fixed within an episode. Every decision
    uses only the state at the current response boundary and the current HiQR
    action. No ADS action for a future tick or nominal rollout is available.
    """

    mode = "online_ma_idm"

    def __init__(
        self,
        theta: np.ndarray | torch.Tensor,
        *,
        lane_half_width_m: float = 1.8,
        vehicle_length_m: float = 4.8,
        prediction_horizon_s: float = 2.0,
        minimum_clearance_m: float = 2.0,
        speed_clearance_s: float = 0.1,
        minimum_acceleration_mps2: float = -8.0,
        maximum_acceleration_mps2: float = 4.0,
        correction_jerk_limit_mps3: float = 12.0,
        native_dt_s: float = 0.04,
        mobil_politeness: float = 0.3,
        mobil_minimum_incentive_mps2: float = 0.3,
        mobil_safe_braking_mps2: float = 4.0,
        mobil_plan_keep_bias_mps2: float = 0.1,
        mobil_onset_calibration: tuple[float, float] | None = None,
        mobil_parameter_variation: bool = True,
        maneuver_latent: np.ndarray | torch.Tensor | None = None,
        response_gate: ResponseGate | None = None,
        response_gate_release_frames: int = 5,
        autonomous_lane_changes_enabled: bool = True,
        lane_changes_on_response_only: bool = True,
        allow_passing_cutin: bool = True,
        driver_correction: dict | None = None,
        lane_style: dict | None = None,
        lane_model: dict | None = None,
    ) -> None:
        super().__init__()
        values = torch.as_tensor(theta, dtype=torch.float32)
        if values.ndim not in (2, 3) or tuple(values.shape[-2:]) != (6, 5):
            raise ValueError("theta must have shape [6,5] or [batch,6,5]")
        if (
            min(
                prediction_horizon_s,
                minimum_clearance_m,
                speed_clearance_s,
                correction_jerk_limit_mps3,
                native_dt_s,
            )
            <= 0.0
        ):
            raise ValueError("response horizon, clearance and timing must be positive")
        self.register_buffer("theta", values.clone())
        self.autonomous_lane_changes_enabled = bool(autonomous_lane_changes_enabled)
        self.lane_changes_on_response_only = bool(lane_changes_on_response_only)
        self.mobil_onset_calibration = mobil_onset_calibration
        self.mobil_parameter_variation = bool(mobil_parameter_variation)
        self.allow_passing_cutin = bool(allow_passing_cutin)
        self.driver_correction = driver_correction
        self.lane_style = lane_style
        self._driver_residual_state: torch.Tensor | None = None
        self._driver_source_y: torch.Tensor | None = None
        self.lane_trajectory = LearnedLaneTrajectory(
            correction=(
                None if driver_correction is None else driver_correction["trajectory"]
            ),
            model=lane_model,
        )
        maneuver_values = (
            None
            if maneuver_latent is None
            else torch.as_tensor(maneuver_latent, dtype=torch.float32)
        )
        if maneuver_values is not None and (
            maneuver_values.ndim not in (2, 3)
            or tuple(maneuver_values.shape[-2:]) != (6, 4)
        ):
            raise ValueError("maneuver_latent must have shape [6,4] or [batch,6,4]")
        self.register_buffer(
            "maneuver_latent",
            None if maneuver_values is None else maneuver_values.clone(),
        )
        self.lane_half_width_m = float(lane_half_width_m)
        self.vehicle_length_m = float(vehicle_length_m)
        self.prediction_horizon_s = float(prediction_horizon_s)
        self.minimum_clearance_m = float(minimum_clearance_m)
        self.speed_clearance_s = float(speed_clearance_s)
        self.minimum_acceleration_mps2 = float(minimum_acceleration_mps2)
        self.maximum_acceleration_mps2 = float(maximum_acceleration_mps2)
        self.correction_jerk_limit_mps3 = float(correction_jerk_limit_mps3)
        self.native_dt_s = float(native_dt_s)
        self.mobil_politeness = float(mobil_politeness)
        self.mobil_minimum_incentive_mps2 = float(mobil_minimum_incentive_mps2)
        self.mobil_safe_braking_mps2 = float(mobil_safe_braking_mps2)
        self.mobil_plan_keep_bias_mps2 = float(mobil_plan_keep_bias_mps2)
        self.response_gate = response_gate
        self.response_gate_release_frames = int(response_gate_release_frames)
        if self.response_gate_release_frames < 1:
            raise ValueError("response_gate_release_frames must be positive")
        if not 0.0 <= self.mobil_politeness <= 1.0:
            raise ValueError("MOBIL politeness must lie in [0,1]")
        if (
            min(
                self.mobil_minimum_incentive_mps2,
                self.mobil_safe_braking_mps2,
                self.mobil_plan_keep_bias_mps2,
            )
            < 0.0
        ):
            raise ValueError("MOBIL thresholds must be non-negative")
        self._lane_repair_latched: torch.Tensor | None = None
        self._spatial_path_latched: torch.Tensor | None = None
        self._longitudinal_response_latched: torch.Tensor | None = None
        self._disturbance_latched: torch.Tensor | None = None
        self._disturbance_leader_index: torch.Tensor | None = None
        self._passing_cutin_latched: torch.Tensor | None = None
        self._autonomous_lane_active: torch.Tensor | None = None
        self._autonomous_lane_initial_active: torch.Tensor | None = None
        self._autonomous_lane_ever: torch.Tensor | None = None
        self._autonomous_lane_target_y: torch.Tensor | None = None
        self._lane_path_coefficients: torch.Tensor | None = None
        self._lane_path_origin: torch.Tensor | None = None
        self._lane_path_distance: torch.Tensor | None = None
        self._lane_path_waiting: torch.Tensor | None = None
        self._lane_path_replans: torch.Tensor | None = None
        self._maneuver_phase: torch.Tensor | None = None
        self._maneuver_stable_frames: torch.Tensor | None = None
        self._maneuver_cooldown_frames: torch.Tensor | None = None
        self._learned_gate_latched_mode: torch.Tensor | None = None
        self._learned_gate_latched_value: torch.Tensor | None = None
        self._learned_gate_inactive_frames: torch.Tensor | None = None

    def snapshot_runtime(self) -> dict[str, torch.Tensor]:
        result = {}
        if self._driver_residual_state is not None:
            result["driver_residual_state"] = (
                self._driver_residual_state.detach().clone()
            )
        if self._driver_source_y is not None:
            result["driver_source_y"] = self._driver_source_y.detach().clone()
        if self._lane_repair_latched is not None:
            result["lane_repair_latched"] = self._lane_repair_latched.detach().clone()
        if self._spatial_path_latched is not None:
            result["spatial_path_latched"] = self._spatial_path_latched.detach().clone()
        if self._longitudinal_response_latched is not None:
            result["longitudinal_response_latched"] = (
                self._longitudinal_response_latched.detach().clone()
            )
        if self._disturbance_latched is not None:
            result["disturbance_latched"] = self._disturbance_latched.detach().clone()
        if self._disturbance_leader_index is not None:
            result["disturbance_leader_index"] = (
                self._disturbance_leader_index.detach().clone()
            )
        if self._passing_cutin_latched is not None:
            result["passing_cutin_latched"] = (
                self._passing_cutin_latched.detach().clone()
            )
        if self._autonomous_lane_active is not None:
            result["autonomous_lane_active"] = (
                self._autonomous_lane_active.detach().clone()
            )
            if self._autonomous_lane_initial_active is not None:
                result["autonomous_lane_initial_active"] = (
                    self._autonomous_lane_initial_active.detach().clone()
                )
            result["autonomous_lane_ever"] = self._autonomous_lane_ever.detach().clone()
            result["autonomous_lane_target_y"] = (
                self._autonomous_lane_target_y.detach().clone()
            )
        if self._maneuver_phase is not None:
            result["maneuver_phase"] = self._maneuver_phase.detach().clone()
            result["maneuver_stable_frames"] = (
                self._maneuver_stable_frames.detach().clone()
            )
            result["maneuver_cooldown_frames"] = (
                self._maneuver_cooldown_frames.detach().clone()
            )
        for key in ("coefficients", "origin", "distance", "waiting", "replans"):
            value = getattr(self, f"_lane_path_{key}")
            if value is not None:
                result[f"lane_path_{key}"] = value.detach().clone()
        if self._learned_gate_latched_mode is not None:
            result["learned_gate_latched_mode"] = (
                self._learned_gate_latched_mode.detach().clone()
            )
            result["learned_gate_latched_value"] = (
                self._learned_gate_latched_value.detach().clone()
            )
            result["learned_gate_inactive_frames"] = (
                self._learned_gate_inactive_frames.detach().clone()
            )
        return result

    def restore_runtime(self, state: dict[str, torch.Tensor]) -> None:
        def restored(key: str) -> torch.Tensor | None:
            value = state.get(key)
            return (
                None if value is None else value.detach().clone().to(self.theta.device)
            )

        self._lane_repair_latched = restored("lane_repair_latched")
        self._driver_residual_state = restored("driver_residual_state")
        self._driver_source_y = restored("driver_source_y")
        self._spatial_path_latched = restored("spatial_path_latched")
        self._longitudinal_response_latched = restored("longitudinal_response_latched")
        self._disturbance_latched = restored("disturbance_latched")
        self._disturbance_leader_index = restored("disturbance_leader_index")
        self._passing_cutin_latched = restored("passing_cutin_latched")
        self._autonomous_lane_active = restored("autonomous_lane_active")
        self._autonomous_lane_initial_active = restored(
            "autonomous_lane_initial_active"
        )
        self._autonomous_lane_ever = restored("autonomous_lane_ever")
        self._autonomous_lane_target_y = restored("autonomous_lane_target_y")
        self._maneuver_phase = restored("maneuver_phase")
        self._maneuver_stable_frames = restored("maneuver_stable_frames")
        self._maneuver_cooldown_frames = restored("maneuver_cooldown_frames")
        for key in ("coefficients", "origin", "distance", "waiting", "replans"):
            setattr(self, f"_lane_path_{key}", restored(f"lane_path_{key}"))
        self._learned_gate_latched_mode = restored("learned_gate_latched_mode")
        self._learned_gate_latched_value = restored("learned_gate_latched_value")
        self._learned_gate_inactive_frames = restored("learned_gate_inactive_frames")

    def _hysteretic_gate_decision(
        self, decision: ResponseGateDecision
    ) -> ResponseGateDecision:
        """Retain an entered learned response until a stable causal release."""
        shape = decision.mode.shape
        if (
            self._learned_gate_latched_mode is None
            or self._learned_gate_latched_mode.shape != shape
        ):
            self._learned_gate_latched_mode = torch.zeros_like(decision.mode)
            self._learned_gate_latched_value = torch.zeros_like(decision.gate)
            self._learned_gate_inactive_frames = torch.zeros_like(decision.mode)
        raw_active = decision.mode.ne(0)
        phase_active = (
            torch.zeros_like(raw_active)
            if self._maneuver_phase is None
            else self._maneuver_phase.eq(int(ManeuverPhase.EXECUTE))
        )
        self._learned_gate_inactive_frames = torch.where(
            raw_active | phase_active,
            torch.zeros_like(self._learned_gate_inactive_frames),
            self._learned_gate_inactive_frames + 1,
        )
        hold = (
            self._learned_gate_latched_mode.ne(0)
            & ~raw_active
            & (
                phase_active
                | self._learned_gate_inactive_frames.lt(
                    self.response_gate_release_frames
                )
            )
        )
        mode = torch.where(
            raw_active,
            decision.mode,
            torch.where(
                hold,
                self._learned_gate_latched_mode,
                torch.zeros_like(decision.mode),
            ),
        )
        gate = torch.where(
            raw_active,
            decision.gate,
            torch.where(hold, self._learned_gate_latched_value, decision.gate),
        )
        active = mode.ne(0)
        self._learned_gate_latched_mode = (
            torch.where(active, mode, torch.zeros_like(mode)).detach().clone()
        )
        self._learned_gate_latched_value = (
            torch.where(active, gate, torch.zeros_like(gate)).detach().clone()
        )
        self._learned_gate_inactive_frames = (
            self._learned_gate_inactive_frames.detach().clone()
        )
        return ResponseGateDecision(decision.logits, mode, gate)

    def _planned_lane_escape(
        self,
        context: ReactionControllerContext,
        gap: torch.Tensor,
        speed: torch.Tensor,
        leader_speed: torch.Tensor,
        leader_index: torch.Tensor,
        leader_acceleration: torch.Tensor,
        base: torch.Tensor,
        safety_clearance: torch.Tensor,
    ) -> torch.Tensor:
        """Avoid braking for a gap that becomes unsafe only after a clear lane exit.

        This is a short-horizon collision check using the fixed Diffusion lane
        intent and currently realized traffic, not a no-intervention rollout.
        If the path has already fallen behind the realized longitudinal state
        or the target lane is occupied, ordinary longitudinal safety applies.
        """
        empty = torch.zeros_like(gap, dtype=torch.bool)
        if any(
            value is None
            for value in (
                context.planned_origin_xy,
                context.planned_current_xy,
                context.planned_terminal_xy,
                context.planned_path_xy,
            )
        ):
            return empty
        npc = context.current[:, 1:]
        intent, _, target_y = planned_adjacent_lane_intent(
            context.planned_origin_xy[..., 1],
            context.planned_terminal_xy[..., 1],
            context.map_polylines,
            context.map_polyline_valid,
        )
        dimensions = _context_vehicle_dimensions(
            context,
            fallback_length_m=self.vehicle_length_m,
            fallback_width_m=self.lane_half_width_m,
        )
        npc_dimensions = dimensions[:, 1:]
        leader_width = torch.gather(dimensions[..., 1], 1, leader_index.clamp_min(0))
        pair_half_width = 0.5 * (npc_dimensions[..., 1] + leader_width)
        pair_length = 0.5 * (npc_dimensions[..., 0, None] + dimensions[:, None, :, 0])
        target_width, target_available = _target_lane_width(
            context, target_y, fallback_width_m=2.0 * self.lane_half_width_m
        )
        target_half_width = 0.5 * target_width
        # Use the reference anchor at this boundary, not the next position.
        on_plan = (npc[..., 0] - context.planned_current_xy[..., 0]).abs() < 2.0
        horizon_steps = max(1, int(round(self.prediction_horizon_s / self.native_dt_s)))
        indices = (
            torch.arange(1, horizon_steps + 1, device=npc.device)
            + int(context.response_index)
        ).clamp_max(context.planned_path_xy.shape[1] - 1)
        times = torch.arange(1, horizon_steps + 1, device=npc.device, dtype=npc.dtype)
        times = times * self.native_dt_s
        planned_y = context.planned_path_xy.index_select(1, indices)[..., 1].permute(
            0, 2, 1
        )
        leader_y = torch.gather(context.current[..., 1], 1, leader_index.clamp_min(0))
        leader_vy = torch.gather(context.current[..., 3], 1, leader_index.clamp_min(0))
        predicted_lateral_gap = planned_y - (
            leader_y[..., None] + leader_vy[..., None] * times
        )
        predicted_longitudinal_gap = (
            gap[..., None]
            + (leader_speed - speed)[..., None] * times
            + 0.5 * (leader_acceleration - base)[..., None] * times.square()
        )
        collision_risk = (
            (predicted_longitudinal_gap < safety_clearance[..., None])
            & (predicted_lateral_gap.abs() < pair_half_width[..., None])
        ).any(dim=-1)
        # During hard braking, require realized lateral clearance as well as planned clearance.
        realized_lateral_gap = (
            npc[..., 1, None]
            + npc[..., 3, None] * times
            - leader_y[..., None]
            - leader_vy[..., None] * times
        )
        collision_risk = collision_risk | leader_acceleration.le(-4.0) & (
            (predicted_longitudinal_gap < safety_clearance[..., None])
            & (realized_lateral_gap.abs() < pair_half_width[..., None])
        ).any(dim=-1)
        other = context.current[:, None]
        dx = other[..., 0] - npc[:, :, None, 0]
        entry_mask = (planned_y - target_y[..., None]).abs() < target_half_width[
            ..., None
        ]
        steps = torch.arange(1, horizon_steps + 1, device=npc.device)
        entry_step = torch.where(
            entry_mask,
            steps,
            torch.full_like(steps, horizon_steps),
        ).amin(dim=-1)
        entry_time = entry_step.to(npc.dtype) * self.native_dt_s
        target_horizon = torch.full(
            (1, 1, 7),
            self.prediction_horizon_s,
            dtype=npc.dtype,
            device=npc.device,
        )
        # Reserve the ADS's realized crossing corridor during gradual lane changes.
        target_horizon[..., 0] = max(4.0, self.prediction_horizon_s)
        future_dx = dx + target_horizon * (other[..., 2] - npc[:, :, None, 2])
        other_y = other[..., 1]
        future_other_y = other_y + target_horizon * other[..., 3]
        entry_dx = dx + entry_time[..., None] * (other[..., 2] - npc[:, :, None, 2])
        currently_in_target = (
            other_y - target_y[:, :, None]
        ).abs() < target_half_width[:, :, None]
        target_band = (
            torch.minimum(other_y, future_other_y)
            < target_y[:, :, None] + target_half_width[:, :, None]
        ) & (
            torch.maximum(other_y, future_other_y)
            > target_y[:, :, None] - target_half_width[:, :, None]
        )
        own = torch.arange(7, device=npc.device)[None, None] == (
            torch.arange(6, device=npc.device)[None, :, None] + 1
        )
        target_occupied = (
            context.current_valid[:, None]
            & ~own
            & (
                (
                    currently_in_target
                    & (entry_dx < pair_length + self.minimum_clearance_m)
                    & (entry_dx > -(pair_length + self.minimum_clearance_m))
                )
                | (
                    target_band
                    & ~currently_in_target
                    & (
                        torch.minimum(dx, future_dx)
                        < pair_length + self.minimum_clearance_m
                    )
                    & (
                        torch.maximum(dx, future_dx)
                        > -(pair_length + self.minimum_clearance_m)
                    )
                )
            )
        ).any(dim=-1)
        escape = (
            intent
            & target_available
            & on_plan
            & leader_index.ge(0)
            & ~target_occupied
            & ~collision_risk
            & (predicted_lateral_gap[..., -1].abs() >= pair_half_width)
        )
        if context.reference_retired is not None:
            escape = escape & ~context.reference_retired
        return escape

    def _departing_leader_clearance(
        self,
        context: ReactionControllerContext,
        gap: torch.Tensor,
        speed: torch.Tensor,
        leader_speed: torch.Tensor,
        leader_index: torch.Tensor,
        leader_acceleration: torch.Tensor,
        base: torch.Tensor,
        safety_clearance: torch.Tensor,
    ) -> torch.Tensor:
        """Keep a lane-keeping NPC on HiQR when its leader exits safely.

        The leader's observed lateral velocity must increase separation, and
        longitudinal clearance must remain safe until lateral separation is
        complete. Planned NPC lane changes use the stricter escape gate above
        so a slowly crossing ADS cannot be mistaken for a departing leader.
        """
        empty = torch.zeros_like(gap, dtype=torch.bool)
        if any(
            value is None
            for value in (
                context.planned_origin_xy,
                context.planned_terminal_xy,
                context.map_polylines,
                context.map_polyline_valid,
            )
        ):
            return empty
        intent, _, _ = planned_adjacent_lane_intent(
            context.planned_origin_xy[..., 1],
            context.planned_terminal_xy[..., 1],
            context.map_polylines,
            context.map_polyline_valid,
        )
        npc = context.current[:, 1:]
        dimensions = _context_vehicle_dimensions(
            context,
            fallback_length_m=self.vehicle_length_m,
            fallback_width_m=self.lane_half_width_m,
        )
        leader_width = torch.gather(dimensions[..., 1], 1, leader_index.clamp_min(0))
        pair_half_width = 0.5 * (dimensions[:, 1:, 1] + leader_width)
        leader_y = torch.gather(context.current[..., 1], 1, leader_index.clamp_min(0))
        leader_vy = torch.gather(context.current[..., 3], 1, leader_index.clamp_min(0))
        lateral_gap = leader_y - npc[..., 1]
        lateral_rate = leader_vy - npc[..., 3]
        horizon_steps = max(1, int(round(self.prediction_horizon_s / self.native_dt_s)))
        times = (
            torch.arange(
                1,
                horizon_steps + 1,
                device=npc.device,
                dtype=npc.dtype,
            )
            * self.native_dt_s
        )
        future_gap = (
            gap[..., None]
            + (leader_speed - speed)[..., None] * times
            + 0.5 * (leader_acceleration - base)[..., None] * times.square()
        )
        future_lateral_gap = lateral_gap[..., None] + lateral_rate[..., None] * times
        joint_risk = (
            (future_gap < safety_clearance[..., None])
            & (future_lateral_gap.abs() < pair_half_width[..., None])
        ).any(dim=-1)
        return (
            ~intent
            & context.current_valid[:, 1:]
            & leader_index.ge(0)
            & lateral_gap.abs().ge(0.5)
            & lateral_gap.abs().lt(pair_half_width)
            & (lateral_gap * lateral_rate > 0.2)
            & (future_lateral_gap[..., -1].abs() >= pair_half_width)
            & ~joint_risk
        )

    def _semantic_lane_repair(
        self,
        context: ReactionControllerContext,
        longitudinal_active: torch.Tensor,
        maneuver_permission: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Delay an unsafe planned change and finish it at its lane centre.

        The target comes from the generated Diffusion plan's start/end lane,
        not from a simulated nominal branch. On-plan HiQR steering is kept
        when the target gap is safe; an uncommitted change waits for a clear
        target-lane gap, then follows the same lane-level intent.
        """
        base_yaw = context.base_actions[:, 0, :, 1]
        if any(
            value is None
            for value in (
                context.planned_origin_xy,
                context.planned_current_xy,
                context.planned_terminal_xy,
            )
        ):
            empty = torch.zeros_like(base_yaw, dtype=torch.bool)
            return base_yaw, empty, empty
        origin = context.planned_origin_xy[..., 1]
        planned = context.planned_current_xy[..., 1]
        terminal = context.planned_terminal_xy[..., 1]
        npc = context.current[:, 1:]
        intent, source_y, target_y = planned_adjacent_lane_intent(
            origin, terminal, context.map_polylines, context.map_polyline_valid
        )
        replanned = (
            torch.zeros_like(intent)
            if self._autonomous_lane_ever is None
            else self._autonomous_lane_ever
        )
        if self._autonomous_lane_target_y is not None:
            target_y = torch.where(replanned, self._autonomous_lane_target_y, target_y)
        intent = intent | replanned
        source_width, source_available = _target_lane_width(
            context, source_y, fallback_width_m=2.0 * self.lane_half_width_m
        )
        target_width, target_available = _target_lane_width(
            context, target_y, fallback_width_m=2.0 * self.lane_half_width_m
        )
        target_half_width = 0.5 * target_width
        dimensions = _context_vehicle_dimensions(
            context,
            fallback_length_m=self.vehicle_length_m,
            fallback_width_m=self.lane_half_width_m,
        )
        pair_length = 0.5 * (dimensions[:, 1:, 0, None] + dimensions[:, None, :, 0])
        displaced = (npc[..., 1] - planned).abs() > 0.5
        speed = torch.linalg.vector_norm(npc[..., 2:4], dim=-1).clamp_min(0.5)
        heading = KinematicTrafficDynamics.heading_from_velocity(
            npc[..., 3], npc[..., 2]
        )
        error_y = target_y - npc[..., 1]
        settled = (error_y.abs() < 0.05) & (heading.abs() < 0.005)
        if context.planned_path_xy is None:
            path_y = target_y
            path_mode = torch.zeros_like(intent)
            spatial_mismatch = torch.zeros_like(intent)
        else:
            if (
                self._spatial_path_latched is None
                or self._spatial_path_latched.shape != intent.shape
            ):
                self._spatial_path_latched = torch.zeros_like(intent)
            progress_shift = (
                npc[..., 0] - context.planned_current_xy[..., 0]
            ).abs() > 2.0
            path_mode = (
                (self._spatial_path_latched | (longitudinal_active & progress_shift))
                & intent
                & ~settled
                & ~replanned
            )
            self._spatial_path_latched = path_mode.detach().clone()
            lookahead = torch.minimum(torch.full_like(speed, 5.0), speed * 0.25)
            spatial_y = planned_lateral_position_at_x(
                context.planned_origin_xy,
                context.planned_path_xy,
                npc[..., 0] + lookahead,
            )
            path_y = torch.where(path_mode, spatial_y, target_y)
            spatial_mismatch = (
                path_mode & progress_shift & ((path_y - planned).abs() > 0.2)
            )
        near_source = source_available & (npc[..., 1] - source_y).abs().lt(
            0.25 * source_width
        )
        other = context.current[:, None]
        dx = other[..., 0] - npc[:, :, None, 0]
        horizon_steps = max(1, int(round(self.prediction_horizon_s / self.native_dt_s)))
        if context.planned_path_xy is None:
            entry_time = torch.full_like(speed, self.prediction_horizon_s)
        else:
            indices = (
                torch.arange(1, horizon_steps + 1, device=npc.device)
                + int(context.response_index)
            ).clamp_max(context.planned_path_xy.shape[1] - 1)
            planned_future_y = context.planned_path_xy.index_select(1, indices)[
                ..., 1
            ].permute(0, 2, 1)
            entry_mask = (
                planned_future_y - target_y[..., None]
            ).abs() < target_half_width[..., None]
            steps = torch.arange(1, horizon_steps + 1, device=npc.device)
            entry_step = torch.where(
                entry_mask,
                steps,
                torch.full_like(steps, horizon_steps),
            ).amin(dim=-1)
            entry_time = entry_step.to(npc.dtype) * self.native_dt_s
        entry_dx = dx + entry_time[..., None] * (other[..., 2] - npc[:, :, None, 2])
        target_lane = (other[..., 1] - target_y[:, :, None]).abs() < target_half_width[
            :, :, None
        ]
        own = torch.arange(7, device=npc.device)[None, None] == (
            torch.arange(6, device=npc.device)[None, :, None] + 1
        )
        unsafe = (
            context.current_valid[:, None]
            & ~own
            & target_lane
            & (entry_dx < pair_length + self.minimum_clearance_m)
            & (entry_dx > -(pair_length + self.minimum_clearance_m))
        ).any(dim=-1)
        target_horizon = torch.full(
            (1, 1, 7),
            self.prediction_horizon_s,
            dtype=npc.dtype,
            device=npc.device,
        )
        target_horizon[..., 0] = max(4.0, self.prediction_horizon_s)
        future_other_y = other[..., 1] + target_horizon * other[..., 3]
        future_dx = dx + target_horizon * (other[..., 2] - npc[:, :, None, 2])
        swept_target = (
            ~target_lane
            & (
                torch.minimum(other[..., 1], future_other_y)
                < target_y[:, :, None] + target_half_width[:, :, None]
            )
            & (
                torch.maximum(other[..., 1], future_other_y)
                > target_y[:, :, None] - target_half_width[:, :, None]
            )
        )
        late_target_unsafe = unsafe | (
            context.current_valid[:, None]
            & ~own
            & swept_target
            & (torch.minimum(dx, future_dx) < pair_length + self.minimum_clearance_m)
            & (torch.maximum(dx, future_dx) > -(pair_length + self.minimum_clearance_m))
        ).any(dim=-1)
        # After clearing a temporary obstruction, finish at the mapped lane centre.
        late_clear_finish = (
            path_mode & (int(context.response_index) >= 75) & ~late_target_unsafe
        )
        path_y = torch.where(late_clear_finish, target_y, path_y)
        path_error_y = path_y - npc[..., 1]
        cap = torch.full_like(speed, 2.0)
        gain = torch.full_like(speed, 0.9)
        desired_vy = cap * torch.tanh(gain * path_error_y / cap)
        target_heading = torch.asin((desired_vy / speed).clamp(-0.95, 0.95))
        heading_error = torch.atan2(
            torch.sin(target_heading - heading),
            torch.cos(target_heading - heading),
        )
        yaw = 6.0 * heading_error
        direction = torch.sign(target_y - origin)
        scheduled = ((planned - origin) * direction > 0.2) | (
            path_mode & ((path_y - origin) * direction > 0.2)
        )
        unsafe_start = intent & scheduled & near_source & unsafe
        if (
            self._lane_repair_latched is None
            or self._lane_repair_latched.shape != intent.shape
        ):
            self._lane_repair_latched = torch.zeros_like(intent)
        repair = (
            (self._lane_repair_latched | displaced | unsafe_start | spatial_mismatch)
            & intent
            & target_available
            & (~settled | replanned)
            & context.current_valid[:, 1:]
        )
        if maneuver_permission is not None:
            repair = repair & (
                maneuver_permission | self._lane_repair_latched | replanned
            )
        self._lane_repair_latched = repair.detach().clone()
        wait = repair & near_source & unsafe
        yaw = torch.where(wait, -6.0 * heading, yaw)
        limit = torch.minimum(torch.full_like(speed, 0.6), 4.0 / speed)
        yaw = torch.maximum(torch.minimum(yaw, limit), -limit)
        if self._lane_path_coefficients is not None:
            path_yaw = self.lane_trajectory.track(
                npc,
                self._lane_path_coefficients,
                self._lane_path_origin,
                self._lane_path_distance,
            )
            yaw = torch.where(replanned, path_yaw, yaw)
            holding = replanned & ~self._autonomous_lane_active
            hold_yaw = frenet_feedback(
                npc,
                target_y,
                torch.zeros_like(target_y),
                torch.zeros_like(target_y),
                frequency=self.lane_trajectory.frequency,
                acceleration_limit=self.lane_trajectory.acceleration_limit,
                jerk_limit=self.lane_trajectory.jerk_limit,
                dt=self.native_dt_s,
            )
            yaw = torch.where(holding, hold_yaw, yaw)
        # Retain a committed intent but wait for target-lane clearance before recovery.
        execute_repair = repair & (~unsafe | near_source)
        return torch.where(execute_repair, yaw, base_yaw), execute_repair, path_mode

    def _passing_cutin_clearance(
        self, context: ReactionControllerContext
    ) -> torch.Tensor:
        """Let an NPC finish a clear pass before a visibly crossing ADS enters.

        A near-side ADS can be ahead *now* yet behind the NPC by the time its
        lateral motion reaches the occupied lane. In that case forcing an IDM
        brake can put the NPC back into the ADS cut-in corridor. This check
        uses realized ADS state, the current HiQR action and mapped lane centers.
        """
        npc = context.current[:, 1:]
        empty = torch.zeros_like(npc[..., 0], dtype=torch.bool)
        if context.map_polylines is None or context.map_polyline_valid is None:
            return empty
        ego = context.current[:, :1]
        dx = ego[..., 0] - npc[..., 0]
        lane_y, lane_present, lane_width = torch_map_lane_geometry(
            context.map_polylines, context.map_polyline_valid
        )
        source_index, source_y, _, source_available = torch_nearest_lane_geometry(
            ego[..., 1], lane_y, lane_present, lane_width
        )
        side = torch.sign(ego[..., 3])
        lower_index, lower_available = torch_adjacent_lane_index(
            source_index, lane_y, lane_present, lane_width, side=-1
        )
        upper_index, upper_available = torch_adjacent_lane_index(
            source_index, lane_y, lane_present, lane_width, side=1
        )
        choose_upper = side.gt(0.0)
        target_index = torch.where(choose_upper, upper_index, lower_index)
        target_exists = source_available & torch.where(
            choose_upper, upper_available, lower_available
        )
        safe_target_index = target_index.clamp_min(0)
        target_y = torch.gather(lane_y, 1, safe_target_index)
        target_width = torch.gather(lane_width, 1, safe_target_index)
        target_half_width = 0.5 * target_width
        npc_lane_index, _, _, npc_lane_available = torch_nearest_lane_geometry(
            npc[..., 1], lane_y, lane_present, lane_width
        )
        target_npc = (
            target_exists
            & npc_lane_available
            & npc_lane_index.eq(target_index)
            & (npc[..., 1] - target_y).abs().lt(target_half_width)
        )
        dimensions = _context_vehicle_dimensions(
            context,
            fallback_length_m=self.vehicle_length_m,
            fallback_width_m=self.lane_half_width_m,
        )
        pair_length = 0.5 * (dimensions[:, :1, 0] + dimensions[:, 1:, 0])
        pair_half_width = 0.5 * (dimensions[:, :1, 1] + dimensions[:, 1:, 1])
        directed_dy = (npc[..., 1] - ego[..., 1]) * side
        directed_vy = ego[..., 3] * side
        time_to_entry = (
            (directed_dy - pair_half_width).clamp_min(0.0) / directed_vy.clamp_min(1.0)
        ).clamp_max(self.prediction_horizon_s)
        base_ax = context.base_actions[:, 0, :, 0]
        dx_at_entry = (
            dx
            + (ego[..., 2] - npc[..., 2]) * time_to_entry
            + 0.5 * (ego[..., 4] - base_ax) * time_to_entry.square()
        )
        eligible = (
            context.current_valid[:, 1:]
            & (directed_vy > 0.08)
            & (directed_dy > pair_half_width)
            & (directed_dy < (target_y - source_y) * side + target_half_width)
            & (dx > 0.0)
            & (dx < 45.0)
            & target_npc
        )
        clear_start = eligible & (dx_at_entry < -(pair_length + 0.5))
        if (
            self._passing_cutin_latched is None
            or self._passing_cutin_latched.shape != dx.shape
        ):
            self._passing_cutin_latched = torch.zeros_like(eligible)
        # Maintain a safe pass until longitudinal clearance is achieved.
        keep_pass = (
            self._passing_cutin_latched
            & context.current_valid[:, 1:]
            & target_npc
            & (dx > -(pair_length + self.minimum_clearance_m))
            & (dx < 45.0)
            & (directed_vy > 0.08)
        )
        passing = clear_start | keep_pass
        self._passing_cutin_latched = passing.detach().clone()
        return passing

    def _autonomous_lane_response(
        self,
        context: ReactionControllerContext,
        actions: torch.Tensor,
        *,
        gap: torch.Tensor,
        speed: torch.Tensor,
        leader_speed: torch.Tensor,
        leader_index: torch.Tensor,
        leader_acceleration: torch.Tensor,
        leader_y: torch.Tensor,
        idm: torch.Tensor,
        theta: torch.Tensor,
        longitudinal_active: torch.Tensor,
        lateral_response_active: torch.Tensor,
        maneuver_permission: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, MOBILDecision | None]:
        """Plan a Train-conditioned lane path and track it in Frenet coordinates."""
        npc = context.current[:, 1:]
        active_shape = npc.shape[:2]
        execution_dimensions = _context_vehicle_dimensions(
            context,
            fallback_length_m=self.vehicle_length_m,
            fallback_width_m=self.lane_half_width_m,
        )[:, 1:]
        initial_controller_state = (
            self._autonomous_lane_active is None
            or self._autonomous_lane_active.shape != active_shape
        )
        if initial_controller_state:
            self._autonomous_lane_active = torch.zeros(
                active_shape, dtype=torch.bool, device=npc.device
            )
            self._autonomous_lane_initial_active = torch.zeros_like(
                self._autonomous_lane_active
            )
            self._autonomous_lane_ever = torch.zeros_like(self._autonomous_lane_active)
            self._autonomous_lane_target_y = torch.zeros_like(npc[..., 1])
            self._lane_path_coefficients = torch.zeros(
                (*active_shape, 8), device=npc.device
            )
            self._lane_path_origin = npc[..., 0].clone()
            self._lane_path_distance = torch.ones_like(npc[..., 0])
            self._lane_path_waiting = torch.zeros_like(self._autonomous_lane_active)
            self._lane_path_replans = torch.zeros_like(npc[..., 0], dtype=torch.long)
            self._maneuver_phase = torch.full(
                active_shape,
                int(ManeuverPhase.TRACK),
                dtype=torch.long,
                device=npc.device,
            )
            self._maneuver_stable_frames = torch.zeros(
                active_shape, dtype=torch.long, device=npc.device
            )
            self._maneuver_cooldown_frames = torch.zeros(
                active_shape, dtype=torch.long, device=npc.device
            )
        self._maneuver_cooldown_frames = (self._maneuver_cooldown_frames - 1).clamp_min(
            0
        )
        rebase_finished = self._maneuver_phase.eq(
            int(ManeuverPhase.REBASE)
        ) & self._maneuver_cooldown_frames.eq(0)
        self._maneuver_phase = torch.where(
            rebase_finished,
            torch.full_like(self._maneuver_phase, int(ManeuverPhase.TRACK)),
            self._maneuver_phase,
        )
        longitudinal_response_phase = (
            longitudinal_active
            & context.current_valid[:, 1:]
            & (
                self._maneuver_phase.eq(int(ManeuverPhase.TRACK))
                | self._maneuver_phase.eq(int(ManeuverPhase.RESPOND))
            )
        )
        self._maneuver_phase = torch.where(
            longitudinal_response_phase,
            torch.full_like(self._maneuver_phase, int(ManeuverPhase.RESPOND)),
            torch.where(
                self._maneuver_phase.eq(int(ManeuverPhase.RESPOND)),
                torch.full_like(self._maneuver_phase, int(ManeuverPhase.TRACK)),
                self._maneuver_phase,
            ),
        )
        active = (
            self._maneuver_phase.eq(int(ManeuverPhase.EXECUTE))
            & context.current_valid[:, 1:]
        )
        if context.map_polylines is None or context.map_polyline_valid is None:
            return actions, active, None

        mobil = None
        lane_choice_probability = None
        learned_draw = None
        learned_selected = torch.zeros_like(active)
        if context.interaction_relations is not None:
            mobil_parameters: dict[str, float | torch.Tensor] = {
                "politeness": self.mobil_politeness,
                "minimum_incentive_mps2": self.mobil_minimum_incentive_mps2,
                "safe_braking_mps2": self.mobil_safe_braking_mps2,
                "plan_keep_bias_mps2": self.mobil_plan_keep_bias_mps2,
            }
            if (
                self.maneuver_latent is not None
                and self.mobil_onset_calibration is None
                and self.mobil_parameter_variation
            ):
                latent = self.maneuver_latent
                if latent.ndim == 2:
                    latent = latent[None].expand(len(context.current), -1, -1)
                elif latent.shape[0] != len(context.current):
                    raise ValueError(
                        "maneuver_latent batch does not match the current world"
                    )
                mobil_parameters = mobil_parameters_from_maneuver_latent(latent)
            mobil = evaluate_mobil_candidates(
                context.current,
                context.current_valid,
                theta,
                context.interaction_relations,
                vehicle_lengths_m=(
                    None
                    if context.vehicle_dimensions is None
                    else context.vehicle_dimensions[..., 0]
                ),
                **mobil_parameters,
            )
            if self.mobil_onset_calibration is not None:
                bias, slope = self.mobil_onset_calibration
                dimensions = _context_vehicle_dimensions(
                    context,
                    fallback_length_m=self.vehicle_length_m,
                    fallback_width_m=1.8,
                )
                features = lane_choice_features(
                    context.current, dimensions[..., 0], context.interaction_relations
                )
                lane_choice_probability = exclusive_lane_probabilities(
                    mobil.incentive_mps2,
                    mobil.safe,
                    {"bias": bias, "slope": slope},
                    features=features,
                    correction=self.driver_correction,
                    style_offset=(
                        None
                        if self.lane_style is None
                        else self.maneuver_latent[..., 0].to(context.current)
                        * self.lane_style["std"]
                        + self.lane_style["mean"]
                    ),
                )
            if lane_choice_probability is not None:
                noise = (
                    torch.randn_like(lane_choice_probability[..., 0])
                    if context.policy_standard_normal is None
                    else context.policy_standard_normal[..., 0]
                )
                learned_draw = sample_lane_choice(
                    lane_choice_probability, torch.special.ndtr(noise), self.native_dt_s
                )
                learned_selected = learned_draw.ne(0)

        lane_y, lane_present, lane_width = torch_map_lane_geometry(
            context.map_polylines, context.map_polyline_valid
        )
        road_bounds = (
            (lane_y - lane_width / 2).masked_fill(~lane_present, torch.inf).amin(-1),
            (lane_y + lane_width / 2).masked_fill(~lane_present, -torch.inf).amax(-1),
        )
        source_index, source_y, source_width, source_available = (
            torch_nearest_lane_geometry(npc[..., 1], lane_y, lane_present, lane_width)
        )
        horizon = self.prediction_horizon_s
        projected_gap = (
            gap
            + (leader_speed - speed) * horizon
            + 0.5 * (leader_acceleration - actions[:, 0, :, 0]) * horizon * horizon
        )
        clearance = torch.maximum(
            torch.full_like(speed, self.minimum_clearance_m),
            self.speed_clearance_s * speed,
        )
        if context.interaction_relations is None:
            leader_lane, _, _, _ = torch_nearest_lane_geometry(
                leader_y, lane_y, lane_present, lane_width
            )
        else:
            leader_lane = torch.gather(
                context.interaction_relations.lane_index,
                1,
                leader_index.clamp_min(0),
            )
        emergency = (
            leader_index.ge(0)
            & leader_lane.eq(source_index)
            & leader_acceleration.le(-4.0)
            & projected_gap.lt(clearance)
        )
        mobil_selected = (
            torch.zeros_like(active)
            if mobil is None or lane_choice_probability is not None
            else mobil.selected
        )
        can_start = (
            context.current_valid[:, 1:]
            & ~active
            & (emergency | mobil_selected | learned_selected)
            & self._maneuver_cooldown_frames.eq(0)
            & (
                self._maneuver_phase.eq(int(ManeuverPhase.TRACK))
                | self._maneuver_phase.eq(int(ManeuverPhase.RESPOND))
            )
            & (npc[..., 1] - source_y).abs().lt(0.2 * source_width)
            & (npc[..., 3].abs().lt(0.3) | emergency)
            & source_available
        )
        if self.lane_changes_on_response_only:
            can_start = can_start & (
                longitudinal_active | lateral_response_active | emergency
            )
        if maneuver_permission is not None:
            can_start = can_start & maneuver_permission
        if not self.autonomous_lane_changes_enabled:
            can_start = torch.zeros_like(can_start)
        if not (active | can_start).any():
            return actions, active, mobil
        preferred_side = torch.zeros_like(source_index)
        if lane_choice_probability is not None:
            fallback = torch.where(
                lane_choice_probability[..., 1] >= lane_choice_probability[..., 2],
                -torch.ones_like(source_index),
                torch.ones_like(source_index),
            )
            preferred_side = torch.where(
                learned_selected,
                learned_draw,
                fallback,
            )
        elif mobil is not None:
            fallback_side = torch.where(
                mobil.incentive_mps2.argmax(dim=-1).eq(0),
                -torch.ones_like(source_index),
                torch.ones_like(source_index),
            )
            preferred_side = torch.where(
                mobil.best_side.ne(0), mobil.best_side, fallback_side
            )
        npc_dx = (npc[..., 0, None] - npc[:, None, :, 0]).abs()
        different = torch.arange(6, device=npc.device)[:, None].ne(
            torch.arange(6, device=npc.device)[None, :]
        )
        lower_slot = torch.arange(6, device=npc.device)[None, :] < (
            torch.arange(6, device=npc.device)[:, None]
        )
        active_target_index, _, _, _ = torch_nearest_lane_geometry(
            self._autonomous_lane_target_y,
            lane_y,
            lane_present,
            lane_width,
        )
        style_noise = (
            torch.zeros((*active_shape, 2), dtype=npc.dtype, device=npc.device)
            if self.maneuver_latent is None
            else self.maneuver_latent[..., 2:4]
        )
        if style_noise.ndim == 2:
            style_noise = style_noise[None].expand(len(npc), -1, -1)
        candidate_clear = None
        candidate_paths = {}
        if context.interaction_relations is not None:
            dimensions = _context_vehicle_dimensions(
                context,
                fallback_length_m=self.vehicle_length_m,
                fallback_width_m=1.8,
            )
            target_indices = context.interaction_relations.adjacent_lane_index[
                :, 1:
            ].clamp_min(0)
            targets = torch.gather(lane_y[:, None].expand(-1, 6, -1), 2, target_indices)
            candidate_clear = []
            for side_index in (0, 1):
                path, clear = self.lane_trajectory.safe_plan(
                    context.current,
                    context.current_valid,
                    dimensions,
                    targets[..., side_index],
                    style_noise,
                    actions[:, 0, :, 0],
                    road_bounds,
                )
                candidate_paths[side_index] = path
                candidate_clear.append(clear)
            candidate_clear = torch.stack(candidate_clear, -1)
        for side in (-1.0, 1.0):
            side_index = 0 if side < 0.0 else 1
            if context.interaction_relations is None:
                target_index, target_available = torch_adjacent_lane_index(
                    source_index, lane_y, lane_present, lane_width, side=int(side)
                )
            else:
                target_index = context.interaction_relations.adjacent_lane_index[
                    :, 1:, side_index
                ]
                target_available = target_index.ge(0)
            safe_target_index = target_index.clamp_min(0)
            target_y = torch.gather(
                lane_y[:, None].expand(-1, npc.shape[1], -1),
                2,
                safe_target_index[..., None],
            ).squeeze(-1)
            target_width = torch.gather(
                lane_width[:, None].expand(-1, npc.shape[1], -1),
                2,
                safe_target_index[..., None],
            ).squeeze(-1)
            if candidate_clear is None:
                other = context.current[:, None]
                dx = other[..., 0] - npc[:, :, None, 0]
                own = torch.arange(7, device=npc.device)[None, None] == (
                    torch.arange(6, device=npc.device)[None, :, None] + 1
                )
                clear = target_available.clone()
                for time in (0.0, 1.5, 3.0):
                    future_y = other[..., 1] + other[..., 3] * time
                    future_dx = dx + (other[..., 2] - npc[:, :, None, 2]) * time
                    occupied = (
                        context.current_valid[:, None]
                        & ~own
                        & (
                            (future_y - target_y[..., None]).abs()
                            < 0.5 * target_width[..., None]
                        )
                        & future_dx.gt(-18.0)
                        & future_dx.lt(12.0)
                    ).any(dim=-1)
                    clear = clear & ~occupied
            else:
                clear = candidate_clear[..., side_index] & target_available
            # Avoid simultaneous choices of, or entry into, the same corridor.
            committed_conflict = (
                active[:, None]
                & different[None]
                & npc_dx.lt(25.0)
                & active_target_index[:, None].eq(target_index[..., None])
            ).any(dim=-1)
            take = can_start & clear & ~committed_conflict
            if mobil is not None or lane_choice_probability is not None:
                take = take & preferred_side.eq(int(side))
            simultaneous_conflict = (
                take[..., None]
                & take[:, None]
                & lower_slot[None]
                & npc_dx.lt(25.0)
                & target_index[..., None].eq(target_index[:, None])
            ).any(dim=-1)
            take = take & ~simultaneous_conflict
            self._autonomous_lane_target_y = torch.where(
                take,
                target_y,
                self._autonomous_lane_target_y,
            )
            path = candidate_paths.get(side_index)
            if path is None:
                path = self.lane_trajectory.plan(
                    npc, execution_dimensions, target_y, style_noise
                )
            self._lane_path_coefficients = torch.where(
                take[..., None], path[0], self._lane_path_coefficients
            )
            self._lane_path_origin = torch.where(take, path[1], self._lane_path_origin)
            self._lane_path_distance = torch.where(
                take, path[2], self._lane_path_distance
            )
            active = active | take
            self._autonomous_lane_ever = self._autonomous_lane_ever | take
            self._maneuver_phase = torch.where(
                take,
                torch.full_like(self._maneuver_phase, int(ManeuverPhase.EXECUTE)),
                self._maneuver_phase,
            )
            self._maneuver_stable_frames = torch.where(
                take,
                torch.zeros_like(self._maneuver_stable_frames),
                self._maneuver_stable_frames,
            )
            can_start = can_start & ~take
        self._autonomous_lane_active = active.detach().clone()
        self._autonomous_lane_ever = self._autonomous_lane_ever.detach().clone()
        self._autonomous_lane_target_y = self._autonomous_lane_target_y.detach().clone()

        heading = KinematicTrafficDynamics.heading_from_velocity(
            npc[..., 3], npc[..., 2]
        )
        error_y = self._autonomous_lane_target_y - npc[..., 1]
        settled = active & error_y.abs().lt(0.25) & npc[..., 3].abs().lt(0.15)
        self._maneuver_stable_frames = torch.where(
            settled,
            self._maneuver_stable_frames + 1,
            torch.where(
                active,
                torch.zeros_like(self._maneuver_stable_frames),
                self._maneuver_stable_frames,
            ),
        )
        completed = active & self._maneuver_stable_frames.ge(10)
        self._maneuver_phase = torch.where(
            completed,
            torch.full_like(self._maneuver_phase, int(ManeuverPhase.REBASE)),
            self._maneuver_phase,
        )
        self._maneuver_cooldown_frames = torch.where(
            completed,
            torch.full_like(self._maneuver_cooldown_frames, 25),
            self._maneuver_cooldown_frames,
        )
        active = active & ~completed
        self._autonomous_lane_active = active.detach().clone()
        dimensions = _context_vehicle_dimensions(
            context, fallback_length_m=self.vehicle_length_m, fallback_width_m=1.8
        )
        safe = path_clearance(
            context.current,
            context.current_valid,
            dimensions,
            self._lane_path_coefficients[..., None, :],
            self._lane_path_origin[..., None],
            self._lane_path_distance[..., None],
            idm.clamp(-8, 4),
            road_bounds=road_bounds,
            acceleration_limit=2 * self.lane_trajectory.acceleration_limit,
        ).squeeze(-1)
        replan = active & (~safe | self._lane_path_waiting)
        path, clear = self.lane_trajectory.safe_plan(
            context.current,
            context.current_valid,
            dimensions,
            self._autonomous_lane_target_y,
            style_noise,
            idm.clamp(-8, 4),
            road_bounds,
        )
        replace_path = replan & clear
        self._lane_path_replans += replace_path.to(torch.long)
        self._lane_path_coefficients = torch.where(
            replace_path[..., None], path[0], self._lane_path_coefficients
        )
        self._lane_path_origin = torch.where(
            replace_path, path[1], self._lane_path_origin
        )
        self._lane_path_distance = torch.where(
            replace_path, path[2], self._lane_path_distance
        )
        waiting = replan & ~clear
        self._lane_path_waiting = waiting.detach().clone()
        yaw = self.lane_trajectory.track(
            npc,
            self._lane_path_coefficients,
            self._lane_path_origin,
            self._lane_path_distance,
        )
        # A blocked path keeps its target intent while dissipating lateral velocity.
        previous_lateral = -npc[..., 4] * heading.sin() + npc[..., 5] * heading.cos()
        wait_lateral = (-2 * heading * speed).clamp(-4, 4)
        wait_lateral = wait_lateral.clamp(
            previous_lateral - 12 * self.native_dt_s,
            previous_lateral + 12 * self.native_dt_s,
        )
        wait_yaw = wait_lateral / speed.clamp_min(0.5)
        yaw = torch.where(waiting, wait_yaw, yaw)
        result = actions.clone()
        result[:, 0, :, 1] = torch.where(active, yaw, result[:, 0, :, 1])
        # Use the episode driver during a committed lane change.
        lane_ax = idm.clamp(
            self.minimum_acceleration_mps2, self.maximum_acceleration_mps2
        )
        if context.previous_background_actions is not None:
            previous_ax = context.previous_background_actions[..., 0]
            max_step = self.correction_jerk_limit_mps3 * self.native_dt_s
            lane_ax = lane_ax.clamp(previous_ax - max_step, previous_ax + max_step)
        result[:, 0, :, 0] = torch.where(active, lane_ax, result[:, 0, :, 0])
        return result, active, mobil

    def forward(
        self, context: ReactionControllerContext, *, deterministic: bool = False
    ) -> ReactionControllerOutput:
        del deterministic
        batch, _, slots, _ = context.base_actions.shape
        if slots != 6:
            raise ValueError("online MA-IDM expects six NPC slots")
        theta = self.theta.to(context.current)
        if theta.ndim == 2:
            theta = theta[None].expand(batch, -1, -1)
        elif theta.shape[0] != batch:
            raise ValueError("theta batch does not match the world batch")

        learned_gate_features = None
        gate_decision = None
        if self.response_gate is not None:
            learned_gate_features = self.response_gate.features(context, theta)
            gate_decision = self._hysteretic_gate_decision(
                self.response_gate(learned_gate_features)
            )

        prospective_target = None
        if (
            self._autonomous_lane_active is not None
            and self._autonomous_lane_active.shape == (batch, 6)
        ):
            prospective_target = torch.full_like(context.current[..., 1], torch.nan)
            prospective_target[:, 1:] = torch.where(
                self._autonomous_lane_active, self._autonomous_lane_target_y, torch.nan
            )
        gap, speed, leader_speed, leader_index = nearest_leader_observation(
            context.current,
            context.current_valid,
            lane_half_width_m=self.lane_half_width_m,
            vehicle_length_m=self.vehicle_length_m,
            vehicle_lengths_m=(
                None
                if context.vehicle_dimensions is None
                else context.vehicle_dimensions[..., 0]
            ),
            vehicle_widths_m=(
                None
                if context.vehicle_dimensions is None
                else context.vehicle_dimensions[..., 1]
            ),
            target_y=prospective_target,
            relations=context.interaction_relations,
            prediction_horizon_s=self.prediction_horizon_s,
        )
        base = context.base_actions[:, 0, :, 0]
        leader_acceleration = torch.gather(
            context.current[..., 4], 1, leader_index.clamp_min(0)
        )
        horizon = self.prediction_horizon_s
        projected_gap = (
            gap
            + (leader_speed - speed) * horizon
            + 0.5 * (leader_acceleration - base) * horizon * horizon
        )
        safety_clearance = torch.maximum(
            torch.full_like(speed, self.minimum_clearance_m),
            self.speed_clearance_s * speed,
        )
        at_risk = projected_gap < safety_clearance
        planned_escape = self._planned_lane_escape(
            context,
            gap,
            speed,
            leader_speed,
            leader_index,
            leader_acceleration,
            base,
            safety_clearance,
        )
        departing_leader = self._departing_leader_clearance(
            context,
            gap,
            speed,
            leader_speed,
            leader_index,
            leader_acceleration,
            base,
            safety_clearance,
        )
        at_risk = at_risk & ~(planned_escape | departing_leader)
        # Respond to abrupt realized leader changes before clearance is critical;
        # opposite changes cancel the stimulus.
        leader_history = torch.gather(
            context.history[..., 4],
            2,
            leader_index.clamp_min(0)[:, None].expand(-1, context.history.shape[1], -1),
        )
        leader_history_valid = torch.gather(
            context.history_valid,
            2,
            leader_index.clamp_min(0)[:, None].expand(-1, context.history.shape[1], -1),
        )
        if leader_history.shape[1] >= 2:
            changes = leader_history[:, 1:] - leader_history[:, :-1]
            sharpest = torch.gather(
                changes, 1, changes.abs().argmax(dim=1, keepdim=True)
            ).squeeze(1)
            net_change = leader_acceleration - leader_history[:, 0]
            abrupt = (
                (sharpest.abs() >= 1.5)
                & (net_change.abs() >= 1.0)
                & (sharpest * net_change > 0.0)
                & leader_history_valid.all(dim=1)
            )
        else:
            sharpest = torch.zeros_like(base)
            abrupt = torch.zeros_like(base, dtype=torch.bool)
        leader_y = torch.gather(context.current[..., 1], 1, leader_index.clamp_min(0))
        # An adjacent future leader is handled by the swept-corridor gate, not as a same-lane shock.
        aligned_npc_leader = leader_index.gt(0) & (
            (leader_y - context.current[:, 1:, 1]).abs() < 0.8
        )
        influence = (
            (leader_index.eq(0) | aligned_npc_leader) & gap.lt(45.0) & abrupt
        ) & ~(planned_escape | departing_leader)
        # Suppress plan-driven acceleration when the realized ADS remains slower;
        # clearance logic still controls braking.
        plan_pursuit = torch.zeros_like(at_risk)
        if context.planned_current_xy is not None:
            plan_lag = context.planned_current_xy[..., 0] - context.current[:, 1:, 0]
            plan_pursuit = (
                leader_index.ge(0)
                & (leader_y - context.current[:, 1:, 1])
                .abs()
                .lt(self.lane_half_width_m)
                & gap.lt(45.0)
                & speed.gt(leader_speed)
                & plan_lag.gt(2.0)
                & base.gt(0.0)
                & ~planned_escape
                & ~departing_leader
            )
        active = (
            context.current_valid[:, 1:]
            & leader_index.ge(0)
            & (at_risk | influence | plan_pursuit)
        )
        innovation = (
            None
            if context.policy_standard_normal is None
            else context.policy_standard_normal[..., 1]
        )
        longitudinal_correction = (
            self.driver_correction is not None
            and self.driver_correction.get("longitudinal") is not None
        )
        driver_history = (
            response_history_features(
                context.current,
                context.history,
                context.history_valid,
                leader_index,
                _context_vehicle_dimensions(
                    context,
                    fallback_length_m=self.vehicle_length_m,
                    fallback_width_m=1.8,
                ),
            )
            if longitudinal_correction
            else None
        )
        maneuver_features = None
        if longitudinal_correction:
            lane_y, lane_present, lane_width = torch_map_lane_geometry(
                context.map_polylines, context.map_polyline_valid
            )
            _, center, width, _ = torch_nearest_lane_geometry(
                context.current[:, 1:, 1], lane_y, lane_present, lane_width
            )
            if self._driver_source_y is None:
                self._driver_source_y = center.clone()
                for past, valid in zip(
                    context.history.unbind(1), context.history_valid.unbind(1)
                ):
                    _, old_center, old_width, _ = torch_nearest_lane_geometry(
                        past[:, 1:, 1], lane_y, lane_present, lane_width
                    )
                    _, updated = causal_maneuver_features(
                        past[:, 1:], self._driver_source_y, old_center, old_width
                    )
                    self._driver_source_y = torch.where(
                        valid[:, 1:], updated, self._driver_source_y
                    )
            maneuver_features, self._driver_source_y = causal_maneuver_features(
                context.current[:, 1:], self._driver_source_y, center, width
            )
        idm, self._driver_residual_state = corrected_acceleration(
            gap,
            speed,
            leader_speed,
            theta,
            correction=self.driver_correction,
            residual_state=self._driver_residual_state,
            innovation=innovation,
            initial_innovation=context.policy_initial_standard_normal,
            history_features=driver_history,
            maneuver_features=maneuver_features,
            dt=self.native_dt_s,
        )
        learned_driver_active = None
        if longitudinal_correction:
            deviation = (
                torch.ones(len(base), device=base.device, dtype=torch.bool)
                if context.nominal_ego_velocity is None
                else torch.linalg.vector_norm(
                    context.current[:, 0, 2:4] - context.nominal_ego_velocity, dim=-1
                )
                > 0.1
            )
            connected = leader_index.eq(0)
            for _ in range(6):
                connected |= leader_index.gt(0) & torch.gather(
                    connected, 1, (leader_index - 1).clamp_min(0)
                )
            observed_response = (
                context.current_valid[:, 1:]
                & leader_index.ge(0)
                & driver_history[..., 1].gt(0)
                & driver_history[..., 0].lt(-0.2)
                & gap.lt(3 * speed)
                & (speed - leader_speed).gt(-0.5)
                & ~(planned_escape | departing_leader)
                & connected
                & deviation[:, None]
            )
            observed_response &= self.driver_correction["response_blend"] > 0
            influence = influence | observed_response
            active = active | observed_response
            learned_driver_active = observed_response
        if gate_decision is not None:
            learned_driver_active = (
                gate_decision.mode.ne(MODE_TRACK) & context.current_valid[:, 1:]
            )
            # Authority blends executable commands, as in selector training.
            # Clipping only after blending amplifies a small learned gate when
            # raw IDM asks for braking far beyond the actuator limit.
            idm_candidate = idm.clamp(
                self.minimum_acceleration_mps2, self.maximum_acceleration_mps2
            )
            desired = base + gate_decision.longitudinal_authority() * (
                context.current_valid[:, 1:] * (idm_candidate - base)
            )
            # Learned authority cannot disable the physical clearance guard.
            desired = torch.where(at_risk, torch.minimum(desired, idm), desired)
            active = learned_driver_active | (
                context.current_valid[:, 1:] & (at_risk | plan_pursuit)
            )
        else:
            proximity = (1.0 - gap / 45.0).clamp(0.0, 1.0)
            stimulus_gain = (theta[..., 2] / 2.0).clamp(0.3, 0.75)
            stimulus = sharpest * stimulus_gain * proximity
            desired = torch.where(influence, base + stimulus, base)
            if longitudinal_correction:
                blend = self.driver_correction["response_blend"]
                desired = torch.where(
                    influence, (1 - blend) * desired + blend * idm, desired
                )
            desired = torch.where(at_risk, torch.minimum(desired, idm), desired)
        desired = torch.where(
            plan_pursuit, torch.minimum(desired, torch.zeros_like(desired)), desired
        )
        desired = desired.clamp(
            self.minimum_acceleration_mps2, self.maximum_acceleration_mps2
        )
        if (
            self._longitudinal_response_latched is None
            or self._longitudinal_response_latched.shape != active.shape
        ):
            self._longitudinal_response_latched = torch.zeros_like(active)
        if (
            self._disturbance_latched is None
            or self._disturbance_latched.shape != active.shape
        ):
            self._disturbance_latched = torch.zeros_like(active)
        if (
            self._disturbance_leader_index is None
            or self._disturbance_leader_index.shape != leader_index.shape
        ):
            self._disturbance_leader_index = torch.full_like(leader_index, -1)
        continued_disturbance = (
            self._disturbance_latched
            & leader_index.eq(self._disturbance_leader_index)
            & gap.lt(45.0)
        )
        # Retain NPC shock memory; ADS uses the instantaneous fallback to avoid cross-dose inversions.
        npc_shock = influence & leader_index.gt(0)
        disturbance_response = npc_shock | continued_disturbance
        longitudinal_active = active
        if learned_driver_active is not None:
            longitudinal_active = longitudinal_active | learned_driver_active
        if context.previous_background_actions is not None:
            previous = context.previous_background_actions[..., 0]
            max_step = self.correction_jerk_limit_mps3 * self.native_dt_s
            # Coasting after positive acceleration is a release, not an urgent brake.
            new_hiqr_brake = base < previous.clamp_max(0) - max_step
            release = (
                self._longitudinal_response_latched
                & ~active
                & context.current_valid[:, 1:]
                & ((base - previous).abs() > max_step)
                & ~new_hiqr_brake
            )
            guarded = desired.clamp(previous - max_step, previous + max_step)
            # Preserve exact HiQR actions for unaffected cars.
            longitudinal_active = active | release
            if learned_driver_active is not None:
                longitudinal_active = longitudinal_active | learned_driver_active
            desired = torch.where(longitudinal_active, guarded, base)
        self._disturbance_latched = disturbance_response.detach().clone()
        self._disturbance_leader_index = (
            torch.where(
                disturbance_response,
                torch.where(npc_shock, leader_index, self._disturbance_leader_index),
                torch.full_like(leader_index, -1),
            )
            .detach()
            .clone()
        )
        actions = context.base_actions.clone()
        passing_ads = (
            self._passing_cutin_clearance(context) & leader_index.eq(0)
            if self.allow_passing_cutin
            else torch.zeros_like(active)
        )
        if self._passing_cutin_latched is not None:
            # A harmless intervening NPC does not erase an active ADS pass intent.
            self._passing_cutin_latched = self._passing_cutin_latched & ~(
                leader_index.gt(0) & (desired < base - 0.05)
            )
        actions[:, 0, :, 0] = torch.where(passing_ads, base, desired)
        maneuver_permission = (
            None if gate_decision is None else gate_decision.mode.eq(MODE_MANEUVER)
        )
        actions[:, 0, :, 1], lane_repair, path_mode = self._semantic_lane_repair(
            context, longitudinal_active, maneuver_permission
        )
        actions, autonomous_lane, mobil = self._autonomous_lane_response(
            context,
            actions,
            gap=gap,
            speed=speed,
            leader_speed=leader_speed,
            leader_index=leader_index,
            leader_acceleration=leader_acceleration,
            leader_y=leader_y,
            idm=idm,
            theta=theta,
            longitudinal_active=longitudinal_active,
            lateral_response_active=lane_repair,
            maneuver_permission=maneuver_permission,
        )
        self._longitudinal_response_latched = (
            (longitudinal_active | autonomous_lane).detach().clone()
        )
        if self._autonomous_lane_ever is not None and self._autonomous_lane_ever.any():
            tracking = self._autonomous_lane_ever & ~self._lane_path_waiting
            first_actions = actions[:, 0].clone()
            bounded_yaw = limit_realized_lateral_acceleration(
                context.current[:, 1:],
                first_actions[..., 1],
                first_actions[..., 0],
                acceleration_limit=self.lane_trajectory.acceleration_limit,
                jerk_limit=self.lane_trajectory.jerk_limit,
                dt=self.native_dt_s,
            )
            actions[:, 0, :, 1] = torch.where(
                tracking, bounded_yaw, first_actions[..., 1]
            )
        overall_active = longitudinal_active | lane_repair | autonomous_lane
        if learned_gate_features is None:
            learned_gate_features = response_gate_features(context, theta)
        delta = actions[:, 0, :, 0] - base
        return ReactionControllerOutput(
            actions=actions,
            alpha=overall_active.to(base.dtype),
            delta_ax=delta,
            active=overall_active,
            rule_action_ax=idm,
            desired_action_ax=actions[:, 0, :, 0],
            policy_active=lane_repair | autonomous_lane,
            spatial_path_active=path_mode,
            autonomous_lane_active=autonomous_lane,
            autonomous_lane_initial_active=self._autonomous_lane_initial_active,
            mobil_incentive_mps2=(None if mobil is None else mobil.incentive_mps2),
            mobil_selected_side=(None if mobil is None else mobil.best_side),
            response_mode=(None if gate_decision is None else gate_decision.mode),
            response_gate=(None if gate_decision is None else gate_decision.gate),
            response_gate_features=learned_gate_features,
            realized_leader_index=leader_index,
        )
