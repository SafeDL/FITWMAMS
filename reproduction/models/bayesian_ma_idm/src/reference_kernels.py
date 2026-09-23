"""Small, float64 reference implementations of equations in Zhang & Sun (2024)."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.linalg import cho_factor, cho_solve


def idm(gap: np.ndarray | float, speed: np.ndarray | float, closing_speed: np.ndarray | float,
        theta: np.ndarray, delta: float = 4.0, donor_ring_clip: bool = False) -> np.ndarray:
    """IDM acceleration in SI units; closing_speed is follower minus leader speed.

    ``donor_ring_clip`` reproduces the extra ``max(dynamic_gap, 0)`` in the
    author's ``Simulator/simulation_ring.py``.  It is deliberately opt-in:
    the paper's displayed Eq. (2), and therefore fitting/evaluation defaults,
    use the un-clipped equation.
    """
    v0, s0, headway, alpha, beta = np.asarray(theta, dtype=np.float64)
    s = np.maximum(np.asarray(gap, dtype=np.float64), 1.0e-3)
    v = np.maximum(np.asarray(speed, dtype=np.float64), 0.0)
    dv = np.asarray(closing_speed, dtype=np.float64)
    dynamic_gap = v * headway + v * dv / (2.0 * np.sqrt(alpha * beta))
    if donor_ring_clip:
        dynamic_gap = np.maximum(dynamic_gap, 0.0)
    desired_gap = s0 + dynamic_gap
    return alpha * (1.0 - (v / v0) ** delta - (desired_gap / s) ** 2)


def se_kernel(times_a: np.ndarray, times_b: np.ndarray, sigma: float, lengthscale: float) -> np.ndarray:
    """Squared-exponential covariance, k(t,t') = sigma² exp(-(t-t')²/(2 l²))."""
    a = np.asarray(times_a, dtype=np.float64)[:, None]
    b = np.asarray(times_b, dtype=np.float64)[None, :]
    return float(sigma) ** 2 * np.exp(-0.5 * ((a - b) / float(lengthscale)) ** 2)


def _stable_cholesky(matrix: np.ndarray) -> tuple[np.ndarray, bool]:
    """Factor a covariance, allowing only rounding-scale diagonal jitter."""
    eye = np.eye(len(matrix), dtype=np.float64)
    for jitter in (0.0, 1.0e-12, 1.0e-10):
        try:
            return cho_factor(matrix + jitter * eye, lower=True, check_finite=True), bool(jitter)
        except np.linalg.LinAlgError:
            continue
    raise np.linalg.LinAlgError("GP covariance is not positive definite")


def conditional_gp(times_past: np.ndarray, residuals_past: np.ndarray, time_new: float,
                   sigma: float, lengthscale: float, observation_noise: float = 1.0e-6) -> tuple[float, float]:
    """GP conditional mean and variance using Cholesky solves, never a matrix inverse."""
    past = np.asarray(times_past, dtype=np.float64)
    residuals = np.asarray(residuals_past, dtype=np.float64)
    if len(past) == 0:
        return 0.0, float(sigma) ** 2
    covariance = se_kernel(past, past, sigma, lengthscale)
    covariance.flat[:: len(past) + 1] += float(observation_noise) ** 2
    factor, _ = _stable_cholesky(covariance)
    cross = se_kernel(np.array([time_new]), past, sigma, lengthscale)[0]
    solution = cho_solve(factor, residuals, check_finite=True)
    mean = float(cross @ solution)
    variance = float(sigma) ** 2 - float(cross @ cho_solve(factor, cross, check_finite=True))
    if variance < -1.0e-8:
        raise FloatingPointError(f"materially negative conditional GP variance: {variance}")
    return mean, max(variance, 0.0)


@dataclass
class GPHistory:
    times: list[float]
    residuals: list[float]
    sigma: float
    lengthscale: float
    observation_noise: float = 1.0e-6
    memory_seconds: float | None = 5.0

    def step(self, timestamp: float, standard_normal: float) -> float:
        """Sample one causal GP value and append that sampled value to history."""
        if self.memory_seconds is not None and self.times:
            cutoff = float(timestamp) - float(self.memory_seconds)
            first = int(np.searchsorted(np.asarray(self.times), cutoff, side="left"))
            self.times = self.times[first:]
            self.residuals = self.residuals[first:]
        mean, variance = conditional_gp(np.asarray(self.times), np.asarray(self.residuals), timestamp,
                                        self.sigma, self.lengthscale, self.observation_noise)
        value = mean + np.sqrt(variance) * float(standard_normal)
        self.times.append(float(timestamp))
        self.residuals.append(float(value))
        return float(value)
