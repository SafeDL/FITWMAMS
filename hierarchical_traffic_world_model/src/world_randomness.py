"""Explicit base-randomness contract for the hierarchical traffic world.

The arrays in :class:`WorldExogenousState` are the complete stochastic input
to Flow, diffusion and the 25 Hz response layer.  They are intentionally
plain NumPy values so an AMS implementation can serialize and mutate blocks
without treating an integer seed as a latent variable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class WorldExogenousState:
    """Prior-distributed base variables for one or more complete worlds."""

    scenario_uniform: np.ndarray
    c0_base_latent: np.ndarray
    k_base_latent: np.ndarray
    diffusion_noise: np.ndarray
    scene_response_innovations: np.ndarray
    agent_response_innovations: np.ndarray

    @property
    def batch_size(self) -> int:
        return int(np.asarray(self.scenario_uniform).shape[0])

    @property
    def response_steps(self) -> int:
        return int(np.asarray(self.scene_response_innovations).shape[1])

    def validate(
        self,
        *,
        c0_dim: int = 40,
        k_dim: int = 72,
        horizon_frames: int = 149,
        response_steps: int | None = None,
        scene_dim: int = 16,
        agent_dim: int = 16,
    ) -> None:
        n = self.batch_size
        expected_steps = horizon_frames if response_steps is None else int(response_steps)
        arrays = {
            "scenario_uniform": (self.scenario_uniform, (n,)),
            "c0_base_latent": (self.c0_base_latent, (n, c0_dim)),
            "k_base_latent": (self.k_base_latent, (n, k_dim)),
            "diffusion_noise": (self.diffusion_noise, (n, horizon_frames, 12)),
            "scene_response_innovations": (
                self.scene_response_innovations,
                (n, expected_steps, scene_dim),
            ),
            "agent_response_innovations": (
                self.agent_response_innovations,
                (n, expected_steps, 7, agent_dim),
            ),
        }
        for name, (value, shape) in arrays.items():
            array = np.asarray(value)
            if array.shape != shape:
                raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
            if not np.isfinite(array).all():
                raise ValueError(f"{name} contains a non-finite value")
        uniform = np.asarray(self.scenario_uniform)
        if np.any((uniform < 0.0) | (uniform >= 1.0)):
            raise ValueError("scenario_uniform values must lie in [0, 1)")

    @classmethod
    def sample(
        cls,
        n: int,
        *,
        seed: int,
        response_steps: int = 149,
        scene_dim: int = 16,
        agent_dim: int = 16,
    ) -> "WorldExogenousState":
        if int(n) < 1 or int(response_steps) < 1:
            raise ValueError("n and response_steps must both be positive")
        generator = np.random.default_rng(int(seed))
        state = cls(
            scenario_uniform=generator.random(int(n), dtype=np.float64),
            c0_base_latent=generator.standard_normal((int(n), 40), dtype=np.float32),
            k_base_latent=generator.standard_normal((int(n), 72), dtype=np.float32),
            diffusion_noise=generator.standard_normal((int(n), 149, 12), dtype=np.float32),
            scene_response_innovations=generator.standard_normal(
                (int(n), int(response_steps), int(scene_dim)), dtype=np.float32
            ),
            agent_response_innovations=generator.standard_normal(
                (int(n), int(response_steps), 7, int(agent_dim)), dtype=np.float32
            ),
        )
        state.validate(response_steps=response_steps, scene_dim=scene_dim, agent_dim=agent_dim)
        return state

    def save(self, path: str | Path) -> None:
        self.validate(
            response_steps=self.response_steps,
            scene_dim=np.asarray(self.scene_response_innovations).shape[-1],
            agent_dim=np.asarray(self.agent_response_innovations).shape[-1],
        )
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **self.as_dict())

    @classmethod
    def load(cls, path: str | Path) -> "WorldExogenousState":
        with np.load(path) as values:
            state = cls(**{name: np.asarray(values[name]) for name in values.files})
        state.validate(
            response_steps=state.response_steps,
            scene_dim=np.asarray(state.scene_response_innovations).shape[-1],
            agent_dim=np.asarray(state.agent_response_innovations).shape[-1],
        )
        return state

    def as_dict(self) -> dict[str, np.ndarray]:
        return {
            name: np.asarray(getattr(self, name)).copy()
            for name in (
                "scenario_uniform",
                "c0_base_latent",
                "k_base_latent",
                "diffusion_noise",
                "scene_response_innovations",
                "agent_response_innovations",
            )
        }

    def select(self, indices: slice | np.ndarray) -> "WorldExogenousState":
        """Return a batch subset while preserving every randomness block."""
        return WorldExogenousState(
            **{name: values[indices].copy() for name, values in self.as_dict().items()}
        )

    @classmethod
    def concatenate(
        cls,
        states: list["WorldExogenousState"],
    ) -> "WorldExogenousState":
        """Combine compatible batches for vectorized world evaluation."""
        if not states:
            raise ValueError("states must not be empty")
        values = [state.as_dict() for state in states]
        keys = values[0].keys()
        combined = {
            name: np.concatenate([item[name] for item in values], axis=0)
            for name in keys
        }
        result = cls(**combined)
        result.validate(
            response_steps=result.response_steps,
            scene_dim=combined["scene_response_innovations"].shape[-1],
            agent_dim=combined["agent_response_innovations"].shape[-1],
        )
        return result

    def pcn_mutate(self, block: str, *, beta: float, seed: int) -> "WorldExogenousState":
        """Return a prior-preserving pCN proposal for one Gaussian block.

        The categorical scenario uniform is deliberately not treated as
        Gaussian.  AMS may replace that block with an independent U(0,1)
        proposal while preserving its prior exactly.
        """
        if not 0.0 < float(beta) <= 1.0:
            raise ValueError("beta must lie in (0, 1]")
        values = self.as_dict()
        if block == "scenario":
            values["scenario_uniform"] = np.random.default_rng(int(seed)).random(
                self.batch_size, dtype=np.float64
            )
        elif block in {
            "c0_base_latent",
            "k_base_latent",
            "diffusion_noise",
            "scene_response_innovations",
            "agent_response_innovations",
        }:
            current = values[block]
            innovation = np.random.default_rng(int(seed)).standard_normal(
                current.shape, dtype=np.float32
            )
            values[block] = (
                np.sqrt(1.0 - float(beta) ** 2) * current + float(beta) * innovation
            ).astype(np.float32)
        else:
            raise ValueError(f"unknown exogenous block {block!r}")
        return WorldExogenousState(**values)
