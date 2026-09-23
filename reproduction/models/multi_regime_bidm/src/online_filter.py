"""Causal filtering for a fitted finite explicit-duration regime model."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .hsmm import FiniteHSMM, _logsumexp


def duration_pmf(model: FiniteHSMM) -> np.ndarray:
    """Return ``P(duration=d | regime)`` as ``[regime, d-1]``."""
    return np.exp(model._duration_logpmf().T)


@dataclass(frozen=True)
class FilterState:
    """Posterior mass after the latest observation; duration axis is remaining ticks."""
    mass: np.ndarray
    observations_seen: int
    low_observation_likelihood: bool = False


def _observe(model: FiniteHSMM, mass: np.ndarray, observation: np.ndarray) -> tuple[np.ndarray, bool]:
    log_likelihood = model._emission_logpdf(np.asarray(observation, float)[None, :])[0]
    log_prior = np.full_like(mass, -np.inf, dtype=float)
    positive = mass > 0
    log_prior[positive] = np.log(mass[positive])
    log_weight = log_prior + log_likelihood[:, None]
    normalizer = _logsumexp(log_weight.ravel())
    # Log normalization preserves a useful posterior even when every state is
    # implausible.  Expose that condition rather than silently inventing a new
    # emergency state.
    return np.exp(log_weight - normalizer), bool(normalizer < -30.)


def initialize_filter(model: FiniteHSMM, observation: np.ndarray) -> FilterState:
    mass = model.initial[:, None] * duration_pmf(model)
    posterior, low = _observe(model, mass, observation)
    return FilterState(posterior, observations_seen=1, low_observation_likelihood=low)


def predict_filter(model: FiniteHSMM, state: FilterState) -> np.ndarray:
    """Advance one 5 Hz tick before observing it, excluding self transitions."""
    mass = np.asarray(state.mass, float)
    if mass.shape != (model.states, model.duration_max):
        raise ValueError("filter mass does not match the fitted regime model")
    next_mass = np.zeros_like(mass)
    next_mass[:, :-1] += mass[:, 1:]
    fresh = duration_pmf(model)
    for previous in range(model.states):
        for current in range(model.states):
            if current != previous:
                next_mass[current] += mass[previous, 0] * model.transition[previous, current] * fresh[current]
    total = float(np.sum(next_mass))
    if not np.isclose(total, 1., atol=1.e-10):
        raise FloatingPointError(f"duration/regime prediction lost probability mass: {total}")
    return next_mass


def filter_step(model: FiniteHSMM, state: FilterState, observation: np.ndarray) -> FilterState:
    """Causally predict one tick, then condition on that tick's observation."""
    posterior, low = _observe(model, predict_filter(model, state), observation)
    return FilterState(posterior, observations_seen=state.observations_seen + 1,
                       low_observation_likelihood=state.low_observation_likelihood or low)


def regime_posterior(state: FilterState) -> np.ndarray:
    return np.sum(state.mass, axis=1)
