"""A standalone longitudinal active-inference collision-avoidance driver.

The implementation retains the paper's causal sequence: perceive, update a
particle belief, predict non-reactive target futures, extend or fully re-plan
with CEM, and execute one 0.2 s action through a 25 Hz plant.  It is an
independent NumPy implementation, not a copy of the authors' non-commercial
code.  Steering is intentionally excluded and reported as an adaptation.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .config import ActiveInferenceConfig


@dataclass
class DriverState:
    """All stochastic-driver state that must survive a simulator snapshot."""

    gap_particles: np.ndarray
    leader_speed_particles: np.ndarray
    leader_acceleration_particles: np.ndarray
    policy: np.ndarray
    evidence: float
    previous_action: float
    desired_speed: float
    last_observed_leader_speed: float
    decision_count: int
    replan_count: int
    rng: np.random.Generator


@dataclass(frozen=True)
class DecisionTrace:
    action: float
    evidence_before: float
    surprise: float
    replanned: bool
    belief_gap_mean: float
    belief_speed_mean: float


def surprise_step(previous_evidence: float, pragmatic_value: float, config: ActiveInferenceConfig) -> tuple[float, float]:
    """Equations (12)-(13): deterministic, no leak and no accumulation noise."""
    surprise = max(0.0, -float(pragmatic_value))
    return previous_evidence + config.evidence_drift * surprise, surprise


class ActiveInferenceDriver:
    """Documented longitudinal adaptation with the paper-locked planning budget.

    This class is not the official two-dimensional paper model.  Use
    :class:`OfficialPOMDP25Hz` for source-equivalent behavior.
    """

    def __init__(self, config: ActiveInferenceConfig = ActiveInferenceConfig()):
        self.config = config
        self.state: DriverState | None = None

    def reset(self, *, gap_m: float, ego_speed_mps: float, leader_speed_mps: float,
              leader_acceleration_mps2: float = 0.0, desired_speed_mps: float | None = None,
              seed: int | None = None) -> None:
        rng = np.random.default_rng(self.config.seed if seed is None else seed)
        n = self.config.particles
        self.state = DriverState(
            gap_particles=np.maximum(.05, rng.normal(gap_m, self.config.perception_gap_sd_m, n)),
            leader_speed_particles=np.maximum(0., rng.normal(leader_speed_mps, self.config.perception_speed_sd_mps, n)),
            leader_acceleration_particles=rng.normal(leader_acceleration_mps2, self.config.belief_acceleration_sd, n),
            policy=np.full(self.config.horizon, self.config.coast_acceleration_mps2),
            evidence=0.0,
            previous_action=self.config.coast_acceleration_mps2,
            desired_speed=ego_speed_mps if desired_speed_mps is None else desired_speed_mps,
            last_observed_leader_speed=leader_speed_mps,
            decision_count=0,
            replan_count=0,
            rng=rng,
        )

    def _require_state(self) -> DriverState:
        if self.state is None:
            raise RuntimeError("call reset before deciding")
        return self.state

    def _update_belief(self, gap_m: float, ego_speed_mps: float, leader_speed_mps: float) -> None:
        """Particle update using current-only looming-compatible observations."""
        state, cfg = self._require_state(), self.config
        # Prediction is based on the previous internal state; no leader future is read.
        state.gap_particles += (state.leader_speed_particles - ego_speed_mps) * cfg.decision_dt_s
        state.leader_speed_particles = np.maximum(
            0., state.leader_speed_particles + state.leader_acceleration_particles * cfg.decision_dt_s
        )
        state.leader_acceleration_particles += state.rng.normal(0., cfg.belief_acceleration_sd, cfg.particles)
        # The paper's looming threshold makes small relative-speed changes unobservable.
        looming = abs(leader_speed_mps - ego_speed_mps) / max(gap_m, .1)
        observed_speed = leader_speed_mps if looming >= cfg.looming_threshold_per_s else ego_speed_mps
        gap_residual = (state.gap_particles - gap_m) / cfg.perception_gap_sd_m
        speed_residual = (state.leader_speed_particles - observed_speed) / cfg.perception_speed_sd_mps
        log_weight = -.5 * (gap_residual ** 2 + speed_residual ** 2)
        log_weight -= np.max(log_weight)
        weight = np.exp(log_weight); weight /= np.sum(weight)
        indices = state.rng.choice(cfg.particles, size=cfg.particles, p=weight)
        state.gap_particles = np.maximum(.05, state.gap_particles[indices] + state.rng.normal(0., .01, cfg.particles))
        state.leader_speed_particles = np.maximum(0., state.leader_speed_particles[indices])
        # Equation (3)'s norm factor is important in routine highD following:
        # sample broad kinematic changes (sigma_a,0=3), then retain trajectories
        # compatible with the currently observed course.  A real deceleration
        # shifts the centre, so a norm violation is not permanently ignored.
        observed_acceleration = (leader_speed_mps - state.last_observed_leader_speed) / cfg.decision_dt_s
        candidates = state.rng.normal(observed_acceleration, cfg.belief_acceleration_sd, cfg.particles * 4)
        norm_log_weight = -.5 * ((candidates - observed_acceleration) / cfg.prediction_acceleration_sd) ** 2
        norm_log_weight -= np.max(norm_log_weight)
        norm_weight = np.exp(norm_log_weight); norm_weight /= np.sum(norm_weight)
        state.leader_acceleration_particles = state.rng.choice(candidates, cfg.particles, p=norm_weight)
        state.last_observed_leader_speed = leader_speed_mps

    def _constrain_plan(self, actions: np.ndarray, previous_action: float) -> np.ndarray:
        """Apply source-equivalent acceleration/jerk and one-foot pedal constraints."""
        cfg = self.config
        values = np.asarray(actions, float).copy()
        prior = float(previous_action)
        for step in range(len(values)):
            target = float(np.clip(values[step], -cfg.acceleration_limit, cfg.acceleration_limit))
            # a0 must occupy one 0.2 s decision interval when changing pedals.
            prior_sign = np.sign(prior - cfg.coast_acceleration_mps2)
            target_sign = np.sign(target - cfg.coast_acceleration_mps2)
            if (cfg.enforce_pedal_constraint and prior_sign and target_sign
                    and prior_sign != target_sign):
                target = cfg.coast_acceleration_mps2
            jerk_low = -30.0
            jerk_high = 15.0 if prior < 0.0 else 5.0
            target = np.clip(target, prior + jerk_low * cfg.decision_dt_s, prior + jerk_high * cfg.decision_dt_s)
            values[step] = target
            prior = target
        return values

    def _target_predictions(self) -> tuple[np.ndarray, np.ndarray]:
        """Norm-conditioned, non-reactive particle futures for the leader."""
        state, cfg = self._require_state(), self.config
        h, n = cfg.horizon, cfg.particles
        gap = np.empty((h, n)); leader_speed = np.empty((h, n))
        current_gap = state.gap_particles.copy()
        current_speed = state.leader_speed_particles.copy()
        acceleration = state.leader_acceleration_particles.copy()
        norm_center = float(np.mean(acceleration))
        # Norm conditioning means acceleration regressively returns to the observed course;
        # after a violation, particles remain kinematically plausible rather than being clipped.
        for t in range(h):
            acceleration = .80 * acceleration + state.rng.normal(0., cfg.prediction_acceleration_sd, n)
            # The original model's norm-conditioned filter prevents routine
            # driving from treating every kinematic tail as equally likely.
            # In the longitudinal adaptation, the normal course is the
            # measured leader acceleration. Once a clear hard-brake violation
            # is observed, constraints are relaxed and long-tail futures stay.
            if t < cfg.norm_prediction_horizon and norm_center > -1.0:
                norm_log_weight = -.5 * ((acceleration - norm_center) / cfg.longitudinal_norm_acceleration_sd) ** 2
                norm_log_weight -= np.max(norm_log_weight)
                weight = np.exp(norm_log_weight); weight /= np.sum(weight)
                acceleration = state.rng.choice(acceleration, n, p=weight)
            current_speed = np.maximum(0., current_speed + acceleration * cfg.decision_dt_s)
            # Express the target position in the coordinate system of the
            # current ego state.  Ego motion is applied separately in policy
            # roll-out, preventing a double subtraction of ego displacement.
            current_gap = current_gap + current_speed * cfg.decision_dt_s
            gap[t], leader_speed[t] = current_gap, current_speed
        return gap, leader_speed

    def _pragmatic_value(self, actions: np.ndarray, ego_speed_mps: float,
                          target_gap: np.ndarray, target_speed: np.ndarray) -> np.ndarray:
        """Equation (11)'s longitudinal velocity/control/collision/safety terms."""
        cfg = self.config
        h, plans = actions.shape
        particles = target_gap.shape[1]
        ego_speed = np.full((plans, particles), ego_speed_mps)
        ego_position = np.zeros((plans, particles))
        # target_gap is relative to the current ego reference.  Shift it by ego motion.
        value = np.zeros(plans)
        for t in range(h):
            action = actions[t, :, None]
            ego_position += ego_speed * cfg.decision_dt_s + .5 * action * cfg.decision_dt_s ** 2
            ego_speed = np.maximum(0., ego_speed + action * cfg.decision_dt_s)
            gap = target_gap[t][None, :] - ego_position
            speed_cost = -.5 * ((ego_speed - self._require_state().desired_speed) / cfg.desired_speed_sd) ** 2
            action_cost = -.5 * ((action - cfg.coast_acceleration_mps2) / cfg.acceleration_sd) ** 2
            collision = gap < .05
            relative_impact = np.maximum(0., ego_speed - target_speed[t][None, :])
            collision_cost = cfg.collision_cost * collision * (.2 + .8 * np.minimum(1., relative_impact / 10.))
            # Equation (11)'s safe-state preference: can a 1 s delayed maximum brake avoid impact?
            leader_stop = target_speed[t][None, :] ** 2 / (2. * abs(cfg.safety_leader_brake_mps2))
            ego_stop = ego_speed * cfg.safety_response_s + ego_speed ** 2 / (2. * cfg.acceleration_limit)
            unsafe = gap < (ego_stop - leader_stop + .5)
            safety_cost = -0.5 * abs(cfg.collision_cost) * unsafe
            value += np.mean(speed_cost + action_cost + collision_cost + safety_cost, axis=1)
        return value

    def _plan(self, ego_speed_mps: float, *, full_replan: bool) -> np.ndarray:
        """Finite CEM: 100 plans, 10 iterations, best final policy (not a mean)."""
        state, cfg = self._require_state(), self.config
        target_gap, target_speed = self._target_predictions()
        horizon = cfg.horizon if full_replan else 1
        mean = np.zeros(horizon); sd = np.full(horizon, cfg.action_sample_sd)
        best = np.full(horizon, cfg.coast_acceleration_mps2)
        for _ in range(cfg.cem_iterations):
            raw = state.rng.normal(mean[:, None], np.maximum(sd[:, None], 1e-4), (horizon, cfg.cem_plans))
            actions = np.empty_like(raw)
            constraint_start = state.previous_action if full_replan else float(state.policy[-1])
            for plan in range(cfg.cem_plans):
                actions[:, plan] = self._constrain_plan(raw[:, plan], constraint_start)
            if full_replan:
                scores = self._pragmatic_value(actions, ego_speed_mps, target_gap, target_speed)
            else:
                candidate = np.vstack((np.repeat(state.policy[1:, None], cfg.cem_plans, axis=1), actions))
                scores = self._pragmatic_value(candidate, ego_speed_mps, target_gap, target_speed)
            elite_count = max(1, int(round(cfg.cem_plans * cfg.elite_fraction)))
            elite = actions[:, np.argpartition(scores, -elite_count)[-elite_count:]]
            mean, sd = elite.mean(axis=1), elite.std(axis=1)
            best = actions[:, int(np.argmax(scores))]
        if full_replan:
            return best
        return np.r_[state.policy[1:], best[0]]

    def decide(self, *, gap_m: float, ego_speed_mps: float, leader_speed_mps: float) -> DecisionTrace:
        """Consume one 5 Hz decision; caller holds the returned action for five plant ticks."""
        state, cfg = self._require_state(), self.config
        self._update_belief(gap_m, ego_speed_mps, leader_speed_mps)
        if state.decision_count == 0:
            state.policy = self._plan(ego_speed_mps, full_replan=True)
            replanned, surprise, before = True, 0.0, state.evidence
        else:
            target_gap, target_speed = self._target_predictions()
            pragmatic = float(self._pragmatic_value(state.policy[:, None], ego_speed_mps, target_gap, target_speed)[0])
            before = state.evidence
            state.evidence, surprise = surprise_step(state.evidence, pragmatic, cfg)
            replanned = state.evidence >= cfg.evidence_threshold
            state.policy = self._plan(ego_speed_mps, full_replan=replanned)
            if replanned:
                state.evidence = 0.0
                state.replan_count += 1
        action = float(state.policy[0])
        state.previous_action = action
        state.decision_count += 1
        return DecisionTrace(action, before, surprise, replanned,
                             float(np.mean(state.gap_particles)), float(np.mean(state.leader_speed_particles)))
