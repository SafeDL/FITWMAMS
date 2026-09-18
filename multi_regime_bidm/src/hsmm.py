"""Finite explicit-duration Gaussian HSMM used as a documented adaptation.

This is deliberately *not* called HDP-HSMM: the cited paper's continuous
observation/Poisson-duration wording is ambiguous and no author inference code
is available.  It is a train-only, at-most-three-state hard-EM approximation
with truncated Poisson durations and diagonal Gaussian emissions.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import lgamma
import numpy as np

from .style import kmeans, standardize_train


def _logsumexp(values: np.ndarray) -> float:
    maximum = float(np.max(values))
    return maximum + float(np.log(np.sum(np.exp(values - maximum))))


@dataclass
class FiniteHSMM:
    means: np.ndarray
    variances: np.ndarray
    transition: np.ndarray
    initial: np.ndarray
    duration_lambda: np.ndarray
    duration_max: int
    observation_mean: np.ndarray
    observation_scale: np.ndarray

    @property
    def states(self) -> int:
        return len(self.initial)

    def _emission_logpdf(self, observations: np.ndarray) -> np.ndarray:
        values = (np.asarray(observations, float) - self.observation_mean) / self.observation_scale
        delta = values[:, None, :] - self.means[None, :, :]
        return -.5 * np.sum(np.log(2 * np.pi * self.variances)[None, :, :] + delta ** 2 / self.variances[None, :, :], axis=2)

    def _duration_logpmf(self) -> np.ndarray:
        duration = np.arange(1, self.duration_max + 1, dtype=float)[:, None]
        lam = np.maximum(self.duration_lambda[None, :], 1.e-4)
        log_mass = duration * np.log(lam) - lam - np.vectorize(lgamma)(duration + 1.)
        return log_mass - np.asarray([_logsumexp(log_mass[:, state]) for state in range(self.states)])[None, :]

    def viterbi(self, observations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Offline HSMM MAP segmentation; never use this result in an online controller."""
        emission = self._emission_logpdf(observations)
        cumulative = np.vstack((np.zeros((1, self.states)), np.cumsum(emission, axis=0)))
        duration = self._duration_logpmf()
        count, states = len(observations), self.states
        score = np.full((count + 1, states), -np.inf)
        previous_state = np.full((count + 1, states), -1, dtype=int)
        previous_duration = np.zeros((count + 1, states), dtype=int)
        log_transition = np.full((states, states), -np.inf)
        positive = self.transition > 0
        log_transition[positive] = np.log(self.transition[positive])
        for end in range(1, count + 1):
            lengths = np.arange(1, min(self.duration_max, end) + 1, dtype=int)
            starts = end - lengths
            for state in range(states):
                segment = cumulative[end, state] - cumulative[starts, state] + duration[lengths - 1, state]
                parents = score[starts] + log_transition[:, state][None, :]
                parents[:, state] = -np.inf  # explicit-duration HSMM excludes self transitions
                parent = np.argmax(parents, axis=1)
                candidate = segment + np.max(parents, axis=1)
                initial = starts == 0
                candidate[initial] = np.log(self.initial[state]) + segment[initial]
                parent[initial] = -1
                best = int(np.argmax(candidate))
                score[end, state] = candidate[best]
                previous_state[end, state] = parent[best]
                previous_duration[end, state] = lengths[best]
        labels = np.empty(count, dtype=int); durations: list[int] = []
        state, end = int(np.argmax(score[count])), count
        while end:
            length = int(previous_duration[end, state]); start = end - length
            labels[start:end] = state; durations.append(length)
            state, end = int(previous_state[end, state]), start
        return labels, np.asarray(durations[::-1], dtype=int)


def _runs(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    change = np.r_[True, labels[1:] != labels[:-1], True]
    edge = np.flatnonzero(change)
    return labels[edge[:-1]], np.diff(edge)


def _initial_model(sequences: list[np.ndarray], states: int, duration_max: int, seed: int) -> FiniteHSMM:
    raw = np.concatenate(sequences)
    standardized, mean, scale = standardize_train(raw)
    labels, centers = kmeans(standardized, states, seed=seed)
    variance = np.vstack([np.var(standardized[labels == state], axis=0) if np.any(labels == state) else np.ones(raw.shape[1])
                          for state in range(states)])
    return FiniteHSMM(centers, np.maximum(variance, .05), np.full((states, states), 1 / max(1, states - 1)) * (1 - np.eye(states)),
                      np.full(states, 1 / states), np.full(states, 5.), duration_max, mean, scale)


def fit_hsmm(sequences: list[np.ndarray], *, states: int = 3, duration_max: int = 50,
             iterations: int = 12, seed: int = 20260915) -> tuple[FiniteHSMM, list[np.ndarray], list[np.ndarray]]:
    """Fit train sequences by hard-EM and return offline labels/durations."""
    if not sequences or states < 1 or duration_max < 2:
        raise ValueError("need sequences, a positive state count, and duration_max >= 2")
    model = _initial_model(sequences, states, duration_max, seed)
    labels_all: list[np.ndarray] = []
    durations_all: list[np.ndarray] = []
    for _ in range(iterations):
        labels_all, durations_all = zip(*(model.viterbi(sequence) for sequence in sequences))
        standardized = [(sequence - model.observation_mean) / model.observation_scale for sequence in sequences]
        values = np.concatenate(standardized); labels = np.concatenate(labels_all)
        for state in range(states):
            selected = values[labels == state]
            if len(selected):
                model.means[state] = np.mean(selected, axis=0)
                model.variances[state] = np.maximum(np.var(selected, axis=0), .02)
        initial_count = np.ones(states)
        transition_count = np.ones((states, states)) * .1
        np.fill_diagonal(transition_count, 0.)
        duration_values: list[list[int]] = [[] for _ in range(states)]
        for labels in labels_all:
            run_state, run_length = _runs(labels)
            initial_count[run_state[0]] += 1
            for state, length in zip(run_state, run_length):
                duration_values[int(state)].append(int(length))
            for before, after in zip(run_state[:-1], run_state[1:]):
                transition_count[int(before), int(after)] += 1
        model.initial = initial_count / np.sum(initial_count)
        model.transition = transition_count / np.maximum(np.sum(transition_count, axis=1, keepdims=True), 1.e-12)
        model.duration_lambda = np.asarray([np.clip(np.mean(value) if value else 5., 1., duration_max)
                                            for value in duration_values])
    labels_all, durations_all = zip(*(model.viterbi(sequence) for sequence in sequences))
    return model, list(labels_all), list(durations_all)
