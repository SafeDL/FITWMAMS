"""Leakage-aware normal/new/history event mixtures for ADS repair."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .env import WorldEvent


@dataclass(frozen=True)
class EventMixture:
    events: tuple[WorldEvent, ...]
    categories: tuple[str, ...]
    requested_ratios: tuple[float, float, float]


def build_event_mixture(
    normal: Sequence[WorldEvent],
    new_failures: Sequence[WorldEvent],
    history: Sequence[WorldEvent],
    *,
    ratios: tuple[float, float, float] = (0.5, 0.3, 0.2),
    size: int = 1000,
    seed: int = 0,
    forbidden_source_ids: set[str] | None = None,
) -> EventMixture:
    groups = (normal, new_failures, history)
    names = ("normal", "new_failure", "history")
    weights = np.asarray(ratios, np.float64)
    if len(weights) != 3 or np.any(weights < 0.0) or not np.isclose(weights.sum(), 1.0):
        raise ValueError("event mixture ratios must be three non-negative values summing to one")
    if int(size) < 1:
        raise ValueError("mixture size must be positive")
    for name, group, weight in zip(names, groups, weights):
        if weight > 0.0 and not len(group):
            raise ValueError(f"{name} event group is empty but has non-zero weight")
    forbidden = forbidden_source_ids or set()
    leaking = {
        event.source_event_id
        for group in groups for event in group
        if event.source_event_id in forbidden or event.split == "test"
    }
    if leaking:
        raise ValueError(f"repair mixture contains held-out source events: {sorted(leaking)[:5]}")
    counts = np.floor(weights * int(size)).astype(int)
    counts[np.argmax(weights)] += int(size) - int(counts.sum())
    rng = np.random.default_rng(int(seed))
    events: list[WorldEvent] = []
    categories: list[str] = []
    for name, group, count in zip(names, groups, counts):
        if count == 0:
            continue
        indices = rng.integers(0, len(group), size=int(count))
        events.extend(group[int(index)] for index in indices)
        categories.extend((name,) * int(count))
    order = rng.permutation(len(events))
    return EventMixture(
        tuple(events[int(index)] for index in order),
        tuple(categories[int(index)] for index in order),
        tuple(float(value) for value in weights),
    )

