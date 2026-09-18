"""Causal online driver for the documented finite-HSMM adaptation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from bayesian_ma_idm.src.reference_kernels import idm

from .hsmm import FiniteHSMM
from .online_filter import FilterState, filter_step, initialize_filter, regime_posterior


def load_hsmm_models(path: str | Path) -> list[FiniteHSMM]:
    with np.load(path, allow_pickle=False) as data:
        return [
            FiniteHSMM(
                data["means"][style],
                data["variances"][style],
                data["transition"][style],
                data["initial"][style],
                data["duration_lambda"][style],
                int(data["duration_max"]),
                data["observation_mean"][style],
                data["observation_scale"][style],
            )
            for style in range(len(data["initial"]))
        ]


@dataclass(frozen=True)
class RegimeDecision:
    requested_acceleration_mps2: float
    regime: int
    regime_probability: np.ndarray
    low_observation_likelihood: bool


class MultiRegimeIDMDriver:
    """Stateful deployment boundary with explicit observe/decide ordering."""

    def __init__(
        self,
        model: FiniteHSMM,
        theta_by_regime: np.ndarray,
        sigma_by_regime: np.ndarray,
        *,
        seed: int = 0,
    ) -> None:
        self.model = model
        self.theta = np.asarray(theta_by_regime, float)
        self.sigma = np.asarray(sigma_by_regime, float)
        if self.theta.shape != (model.states, 5) or self.sigma.shape != (model.states,):
            raise ValueError("posterior parameter dimensions do not match the HSMM states")
        self._rng = np.random.default_rng(seed)
        self.state: FilterState | None = None

    @classmethod
    def from_posterior(
        cls,
        model: FiniteHSMM,
        path: str | Path,
        *,
        seed: int = 0,
    ) -> "MultiRegimeIDMDriver":
        """Draw one joint regime-parameter table for the complete episode."""
        rng = np.random.default_rng(seed)
        with np.load(path, allow_pickle=False) as posterior:
            draw = int(rng.integers(len(posterior["theta_draws"])))
            instance = cls(
                model,
                posterior["theta_draws"][draw],
                posterior["sigma_draws"][draw],
                seed=seed,
            )
        instance._rng = rng
        return instance

    def reset(self, prefix_observations: np.ndarray, *, seed: int = 0) -> None:
        observations = np.asarray(prefix_observations, float)
        if observations.ndim != 2 or observations.shape[1] != 3 or not len(observations):
            raise ValueError("prefix_observations must have shape [time, gap/speed/closing]")
        self._rng = np.random.default_rng(seed)
        state = initialize_filter(self.model, observations[0])
        for observation in observations[1:]:
            state = filter_step(self.model, state, observation)
        self.state = state

    def observe(self, *, gap_m: float, speed_mps: float, leader_speed_mps: float) -> None:
        if self.state is None:
            raise RuntimeError("call reset with a causal prefix before observe")
        observation = np.asarray((gap_m, speed_mps, speed_mps - leader_speed_mps))
        self.state = filter_step(self.model, self.state, observation)

    def decision(
        self,
        *,
        gap_m: float,
        speed_mps: float,
        leader_speed_mps: float,
        standard_normal: float | None = None,
        regime_uniform: float | None = None,
    ) -> RegimeDecision:
        if self.state is None:
            raise RuntimeError("call reset with a causal prefix before decision")
        probability = regime_posterior(self.state)
        uniform = float(self._rng.random()) if regime_uniform is None else float(regime_uniform)
        regime = int(min(np.searchsorted(np.cumsum(probability), uniform, side="right"), len(probability) - 1))
        z_t = float(self._rng.standard_normal()) if standard_normal is None else float(standard_normal)
        deterministic = float(
            idm(gap_m, speed_mps, speed_mps - leader_speed_mps, self.theta[regime])
        )
        return RegimeDecision(
            requested_acceleration_mps2=deterministic + float(self.sigma[regime]) * z_t,
            regime=regime,
            regime_probability=probability.copy(),
            low_observation_likelihood=self.state.low_observation_likelihood,
        )
