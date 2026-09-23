"""Stateful 5 Hz B-IDM and MA-IDM drivers for a 25 Hz external plant."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .model import load_posterior, sample_driver_joint
from .reference_kernels import GPHistory, idm


@dataclass(frozen=True)
class IDMDecision:
    requested_acceleration_mps2: float
    deterministic_acceleration_mps2: float
    residual_acceleration_mps2: float


class BayesianIDMDriver:
    """Causal online driver retaining one joint parameter draw per episode.

    The driver emits one action every 0.2 s.  A simulator must hold that action
    for five 25 Hz plant ticks.  Plant clipping is deliberately external and
    is never fed back into the stochastic residual process.
    """

    def __init__(
        self,
        model: str,
        theta: np.ndarray,
        process_sigma: float,
        *,
        lengthscale_s: float = 1.0,
        iid_sigma: float = 0.0,
        memory_s: float | None = 5.0,
        seed: int = 0,
    ) -> None:
        if model not in {"b_idm", "ma_idm"}:
            raise ValueError("model must be b_idm or ma_idm")
        self.model = model
        self.theta = np.asarray(theta, dtype=float)
        if self.theta.shape != (5,):
            raise ValueError("theta must contain the five IDM parameters")
        self.process_sigma = float(process_sigma)
        self.lengthscale_s = float(lengthscale_s)
        self.iid_sigma = float(iid_sigma)
        self.memory_s = memory_s
        self.reset(seed=seed)

    @classmethod
    def from_population_posterior(
        cls, path: str | Path, *, model: str, seed: int = 0, memory_s: float | None = 5.0
    ) -> "BayesianIDMDriver":
        rng = np.random.default_rng(seed)
        values = sample_driver_joint(load_posterior(path), rng)
        iid_sigma = float(values[7]) if len(values) > 7 and model == "ma_idm" else 0.0
        driver = cls(
            model,
            values[:5],
            float(values[5]),
            lengthscale_s=float(values[6]),
            iid_sigma=iid_sigma,
            memory_s=memory_s,
            seed=seed,
        )
        driver._rng = rng
        return driver

    def reset(self, *, seed: int = 0) -> None:
        self._rng = np.random.default_rng(seed)
        self._decision_index = 0
        self._gp = GPHistory(
            times=[],
            residuals=[],
            sigma=self.process_sigma,
            lengthscale=self.lengthscale_s,
            memory_seconds=self.memory_s,
        )

    def decision(
        self,
        *,
        gap_m: float,
        speed_mps: float,
        leader_speed_mps: float,
        process_standard_normal: float | None = None,
        iid_standard_normal: float | None = None,
    ) -> IDMDecision:
        process_z = (
            float(self._rng.standard_normal())
            if process_standard_normal is None
            else float(process_standard_normal)
        )
        if self.model == "ma_idm":
            residual = self._gp.step(self._decision_index * 0.2, process_z)
            iid_z = (
                float(self._rng.standard_normal())
                if iid_standard_normal is None
                else float(iid_standard_normal)
            )
            residual += self.iid_sigma * iid_z
        else:
            residual = self.process_sigma * process_z
        deterministic = float(
            idm(gap_m, speed_mps, speed_mps - leader_speed_mps, self.theta)
        )
        self._decision_index += 1
        return IDMDecision(deterministic + residual, deterministic, float(residual))
