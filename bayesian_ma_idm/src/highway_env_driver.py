"""Native-25-Hz MA-IDM vehicle for the local :mod:`highway_env` checkout.

The class is deliberately an adapter rather than a fork of highway-env.  It
uses one correlated seven-parameter driver draw for a vehicle's whole life,
refreshes a stochastic action at 5 Hz, and lets highway-env integrate the
plant at 25 Hz.  This matches the Zhang--Sun simulator organisation while
preserving highway-env's road and collision mechanics.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

from .model import ACCELERATION_OBSERVATION_NOISE, load_posterior, sample_driver_joint
from .reference_kernels import GPHistory, idm

def _make_vehicle_class():
    try:
        from highway_env.vehicle.behavior import IDMVehicle
        from highway_env.vehicle.kinematics import Vehicle
    except ImportError as exc:  # keeps ordinary data fitting independent of highway-env
        raise ImportError("Install the local HighwayEnv package or add HighwayEnv to PYTHONPATH") from exc

    class _MAIDMVehicle(IDMVehicle):
        """Longitudinal MA-IDM; MOBIL lane changes are disabled by default."""

        UPDATE_FRAMES = 5
        DT_S = .04
        GP_MEMORY_SECONDS = 5.0
        MAX_ACCELERATION = 12.0
        MIN_SPEED = 0.0
        MAX_SPEED = 60.0
        PLANT_INTEGRATION = "ballistic_tangent_25hz"
        idm_semantics = "equation"
        posterior: dict[str, np.ndarray] | None = None
        posterior_source: str | None = None
        simulation_mode = "ma_idm"
        fixed_parameters: np.ndarray | None = None
        source_code_gp_protocol = False
        source_code_gp_sigma = np.nan
        source_code_gp_ell = np.nan

        @classmethod
        def configure_posterior(cls, path: str | Path) -> None:
            cls.posterior = load_posterior(path)
            cls.posterior_source = str(Path(path))
            cls.simulation_mode = "ma_idm"
            cls.fixed_parameters = None
            cls.source_code_gp_protocol = False
            cls.source_code_gp_sigma = cls.source_code_gp_ell = np.nan

        @classmethod
        def configure_source_code_gp_protocol(cls, enabled: bool) -> None:
            """Match the public ring script's shared mean GP hyperparameters.

            ``simulation_ring.py`` takes posterior means of ``l`` and
            ``s2_f`` once for the whole ring and conditions its full GP draw
            on a short zero-residual context.  Keep this opt-in because the
            paper-text/highway adapter normally samples a complete joint
            driver draw per vehicle instead.
            """
            cls.source_code_gp_protocol = bool(enabled)
            if enabled:
                if cls.posterior is None or "driver_sigma_draws" not in cls.posterior:
                    raise RuntimeError("source-code GP protocol requires retained PyMC GP draws")
                cls.source_code_gp_sigma = float(np.mean(cls.posterior["driver_sigma_draws"]))
                cls.source_code_gp_ell = float(np.mean(cls.posterior["driver_ell_draws"]))
            else:
                cls.source_code_gp_sigma = cls.source_code_gp_ell = np.nan

        @classmethod
        def configure_fixed_iid(cls, theta: np.ndarray, sigma: float) -> None:
            value = np.asarray(theta, dtype=float)
            if value.shape != (5,) or not np.all(value > 0) or sigma < 0:
                raise ValueError("fixed IDM requires five positive theta values and non-negative iid sigma")
            cls.posterior = None; cls.posterior_source = None
            cls.simulation_mode = "fixed_idm"
            cls.fixed_parameters = np.r_[value, float(sigma), 1.]

        @classmethod
        def configure_idm_semantics(cls, value: str) -> None:
            if value not in {"equation", "donor_ring"}:
                raise ValueError("idm semantics must be 'equation' or 'donor_ring'")
            cls.idm_semantics = value

        def __init__(self, *args, enable_lane_change: bool = False, **kwargs):
            if self.posterior is None and self.simulation_mode != "fixed_idm":
                raise RuntimeError("Call MAIDMVehicle.configure_posterior(path) before creating highway-env vehicles")
            super().__init__(*args, enable_lane_change=enable_lane_change, **kwargs)
            self._ma_frame = 0
            self._ma_time = 0.
            self._ma_noise = 0.
            self._ma_parameters = (self.fixed_parameters.copy() if self.simulation_mode == "fixed_idm"
                                   else sample_driver_joint(self.posterior, self.road.np_random))
            if self.simulation_mode == "fixed_idm":
                self._ma_history = None
            else:
                if self.source_code_gp_protocol:
                    self._ma_parameters[5:7] = (self.source_code_gp_sigma, self.source_code_gp_ell)
                    points = max(1, int(2 * self.source_code_gp_ell / self.DT_S))
                    context = np.linspace(-points * self.DT_S, 0., points).tolist()
                    residual = [0.] * points
                else:
                    context, residual = [], []
                self._ma_history = GPHistory(context, residual, float(self._ma_parameters[5]), float(self._ma_parameters[6]),
                                             observation_noise=1.e-8, memory_seconds=self.GP_MEMORY_SECONDS)

        @classmethod
        def create_from(cls, vehicle):
            """Copy deterministic state and retain the sampled driver state."""
            copied = cls(vehicle.road, vehicle.position, heading=vehicle.heading, speed=vehicle.speed,
                         target_lane_index=vehicle.target_lane_index, target_speed=vehicle.target_speed,
                         route=vehicle.route, timer=getattr(vehicle, "timer", None), enable_lane_change=False)
            if isinstance(vehicle, cls):
                copied._ma_frame = vehicle._ma_frame; copied._ma_time = vehicle._ma_time
                copied._ma_noise = vehicle._ma_noise; copied._ma_parameters = vehicle._ma_parameters.copy()
                if vehicle._ma_history is not None:
                    copied._ma_history = GPHistory(list(vehicle._ma_history.times), list(vehicle._ma_history.residuals),
                                                    vehicle._ma_history.sigma, vehicle._ma_history.lengthscale,
                                                    vehicle._ma_history.observation_noise, vehicle._ma_history.memory_seconds)
            return copied

        def _net_gap(self, front) -> float:
            centre_distance = self.lane_distance_to(front)
            # A normal highway lane has a positive front distance.  A periodic
            # ring lane reports a negative coordinate difference when the
            # leader lies across its zero-longitude seam.
            if centre_distance <= 0 and getattr(self.lane, "length", None):
                centre_distance += self.lane.length
            return max(centre_distance - .5 * (self.LENGTH + front.LENGTH), 1.e-3)

        def _sample_longitudinal_action(self) -> float:
            front, _ = self.road.neighbour_vehicles(self, self.lane_index)
            if front is None:
                gap, leader_speed = 1.e6, self.speed
            else:
                gap, leader_speed = self._net_gap(front), front.speed
            theta = self._ma_parameters[:5]
            deterministic = float(idm(gap, self.speed, self.speed - leader_speed, theta,
                                      donor_ring_clip=self.idm_semantics == "donor_ring"))
            if self.simulation_mode == "fixed_idm" or str(self.posterior["model"].item()) == "b_idm":
                return float(np.clip(deterministic + self._ma_parameters[5] * self.road.np_random.standard_normal(),
                                     -self.MAX_ACCELERATION, self.MAX_ACCELERATION))
            correlated = self._ma_history.step(self._ma_time, self.road.np_random.standard_normal())
            iid_sigma = float(self._ma_parameters[7]) if len(self._ma_parameters) > 7 else ACCELERATION_OBSERVATION_NOISE
            iid = iid_sigma * self.road.np_random.standard_normal()
            return float(np.clip(deterministic + correlated + iid,
                                 -self.MAX_ACCELERATION, self.MAX_ACCELERATION))

        def act(self, action=None) -> None:
            if self.crashed:
                return
            self.follow_road()  # road following, but no MOBIL lane-choice policy
            if self._ma_frame % self.UPDATE_FRAMES == 0:
                self._ma_noise = self._sample_longitudinal_action()
                self._ma_time += self.UPDATE_FRAMES * self.DT_S
            steering = np.clip(self.steering_control(self.target_lane_index), -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE)
            Vehicle.act(self, {"steering": steering, "acceleration": self._ma_noise})
            self._ma_frame += 1

        def step(self, dt: float) -> None:
            """Keep highway-env mechanics while using paper Eq. (3) longitudinal integration.

            ``Vehicle.step`` advances position with forward Euler before it
            advances speed.  The paper uses a ballistic ``.5*a*dt²`` term.
            Add precisely that tangent displacement before delegating, while
            retaining highway-env's lane update and collision machinery.
            """
            if self.crashed:
                return super().step(dt)
            if self.speed + self.action["acceleration"] * dt < 0:
                self.action["acceleration"] = -self.speed / dt
            self.clip_actions()
            acceleration = float(self.action["acceleration"])
            steering = float(self.action["steering"])
            beta = np.arctan(.5 * np.tan(steering))
            tangent = np.asarray((np.cos(self.heading + beta), np.sin(self.heading + beta)))
            self.position += .5 * acceleration * dt * dt * tangent
            super().step(dt)
            self.speed = max(0., self.speed)

        @property
        def ma_parameters(self) -> np.ndarray:
            """Joint physical SI draw [v0,s0,T,alpha,beta,sigma,ell]."""
            return self._ma_parameters.copy()

    return _MAIDMVehicle


MAIDMVehicle = _make_vehicle_class()
