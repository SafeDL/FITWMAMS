"""Deterministic, cacheable first-second B0 control planner."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .initial_behavior_anchor import BehaviorAnchorControlPlan


class FirstSecondNominalPlanner(BehaviorAnchorControlPlan):
    """Five-knot bounded least-squares planner with no trainable parameters.

    The parent implementation stores only fixed basis/smoothness buffers; this
    wrapper supplies stable cache identity and explicit cache I/O for offline
    dataset preparation or repeated Flow end-to-end samples.
    """

    def __init__(self, physics_steps: int = 25, knots: int = 5, *, config: dict | None = None) -> None:
        super().__init__(physics_steps=physics_steps, knots=knots)
        self.config = {"physics_steps": physics_steps, "knots": knots, "ax_bounds": [-8.0, 4.0], "ay_bounds": [-4.0, 4.0], **(config or {})}

    @property
    def config_sha256(self) -> str:
        return hashlib.sha256(json.dumps(self.config, sort_keys=True).encode()).hexdigest()

    def cache_key(self, initial_states: np.ndarray, anchor_raw: np.ndarray, *, flow_schema_sha256: str, dynamics_version: str) -> str:
        digest = hashlib.sha256()
        for value in (initial_states, anchor_raw):
            array = np.ascontiguousarray(np.asarray(value, np.float32)); digest.update(array.tobytes())
        digest.update(flow_schema_sha256.encode()); digest.update(self.config_sha256.encode()); digest.update(str(dynamics_version).encode())
        return digest.hexdigest()

    @staticmethod
    def load_cached(cache_dir: str | Path, key: str) -> np.ndarray | None:
        path = Path(cache_dir) / f"{key}.npy"
        return np.load(path) if path.exists() else None

    @staticmethod
    def save_cached(cache_dir: str | Path, key: str, plan: np.ndarray) -> Path:
        path = Path(cache_dir); path.mkdir(parents=True, exist_ok=True)
        target = path / f"{key}.npy"
        np.save(target, np.asarray(plan, np.float32))
        return target
