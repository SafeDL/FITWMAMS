"""IDM and the stochastic AR residual process (paper Eqs. 8--10)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import numpy as np


def idm(gap: np.ndarray | float, speed: np.ndarray | float, closing: np.ndarray | float,
        theta: np.ndarray) -> np.ndarray:
    """IDM acceleration, where ``closing = follower_speed - leader_speed``."""
    v0, s0, headway, alpha, beta = np.asarray(theta, float)
    s = np.maximum(np.asarray(gap, float), .15)
    v = np.maximum(np.asarray(speed, float), 0.)
    dv = np.asarray(closing, float)
    desired = s0 + v * headway + v * dv / (2. * np.sqrt(alpha * beta))
    return alpha * (1. - (v / v0) ** 4 - (desired / s) ** 2)


def ar_innovation(error_history: np.ndarray, rho: np.ndarray, sigma_eta: float,
                  z_t: float) -> tuple[float, np.ndarray]:
    """Sample/update Eq. (9); history and coefficients are newest-first.

    The stored values are *IDM residuals*, never raw historical accelerations.
    This is deliberately separate from action clipping in a downstream plant.
    """
    rho = np.asarray(rho, float)
    history = np.asarray(error_history, float)
    if history.shape != rho.shape:
        raise ValueError("residual history and rho must have equal AR order")
    error = float(rho @ history + float(sigma_eta) * float(z_t))
    return error, np.r_[error, history[:-1]] if len(history) else history.copy()


def ar_spectral_radius(rho: np.ndarray) -> float:
    """Companion-matrix spectral radius; <1 is AR stationarity."""
    rho = np.asarray(rho, float)
    if not len(rho):
        return 0.
    companion = np.zeros((len(rho), len(rho)))
    companion[0] = rho
    if len(rho) > 1:
        companion[1:, :-1] = np.eye(len(rho) - 1)
    return float(np.max(np.abs(np.linalg.eigvals(companion))))


@dataclass
class DynamicIDMState:
    """State for a 5 Hz stochastic driver embedded in a 25 Hz plant."""
    theta: np.ndarray
    rho: np.ndarray
    sigma_eta: float
    residual_history: np.ndarray = field(default_factory=lambda: np.empty(0))
    last_native_decision_time: float = -np.inf
    held_acceleration: float = 0.

    def decision(self, gap: float, speed: float, closing: float, z_t: float) -> float:
        mean = float(idm(gap, speed, closing, self.theta))
        error, self.residual_history = ar_innovation(self.residual_history, self.rho, self.sigma_eta, z_t)
        self.held_acceleration = mean + error
        return self.held_acceleration


class DynamicIDMDriver:
    """Episode-level posterior draw plus causal AR residual state."""

    def __init__(self, theta: np.ndarray, rho: np.ndarray, sigma_eta: float, *,
                 seed: int = 0, require_stationary: bool = True):
        self.theta = np.asarray(theta, float)
        self.rho = np.asarray(rho, float)
        self.sigma_eta = float(sigma_eta)
        if self.theta.shape != (5,):
            raise ValueError("theta must contain the five IDM parameters")
        if require_stationary and ar_spectral_radius(self.rho) >= 1.0:
            raise ValueError("deployment driver requires a stationary AR process")
        self.reset(seed=seed)

    @classmethod
    def from_posterior(cls, path: str | Path, *, seed: int = 0) -> "DynamicIDMDriver":
        rng = np.random.default_rng(seed)
        with np.load(path, allow_pickle=False) as posterior:
            draw = int(rng.integers(len(posterior["sigma_draws"])))
            driver = int(rng.integers(posterior["theta_draws"].shape[1]))
            theta = posterior["theta_draws"][draw, driver]
            rho = posterior["rho_map"]
            sigma = float(posterior["sigma_draws"][draw])
        instance = cls(theta, rho, sigma, seed=seed, require_stationary=True)
        instance._rng = rng
        return instance

    def reset(self, *, seed: int = 0) -> None:
        self._rng = np.random.default_rng(seed)
        self.state = DynamicIDMState(
            theta=self.theta.copy(),
            rho=self.rho.copy(),
            sigma_eta=self.sigma_eta,
            residual_history=np.zeros(len(self.rho), dtype=float),
        )

    def decision(
        self,
        *,
        gap_m: float,
        speed_mps: float,
        leader_speed_mps: float,
        standard_normal: float | None = None,
    ) -> float:
        z_t = float(self._rng.standard_normal()) if standard_normal is None else float(standard_normal)
        return self.state.decision(gap_m, speed_mps, speed_mps - leader_speed_mps, z_t)
