"""Explicit prior variables for TrafficBots-driven IDM worlds."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np


TRAFFICBOTS_RNG_SCHEMA = "trafficbots_idm_rng_v1"
TRAFFICBOTS_RANDOM_BLOCKS = (
    "scenario_uniform",
    "c0_base_latent",
    "personality_latent",
    "destination_uniform",
)


def _block_rng(seed: int, block: str) -> np.random.Generator:
    material = f"{TRAFFICBOTS_RNG_SCHEMA}:{int(seed)}:{block}".encode("utf-8")
    child_seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "little")
    return np.random.default_rng(child_seed)


@dataclass(frozen=True)
class TrafficBotsExogenousState:
    """Independent base variables for shared-C0 TrafficBots simulation.

    The initial slot pattern and C0 use the common external highD Flow prior.
    TrafficBots itself receives only the realized S0.  Personality is its
    standard-normal prior and ``destination_uniform`` is transformed through
    the S0-conditioned categorical destination predictor.
    """

    scenario_uniform: np.ndarray
    c0_base_latent: np.ndarray
    personality_latent: np.ndarray
    destination_uniform: np.ndarray

    @property
    def batch_size(self) -> int:
        return int(np.asarray(self.scenario_uniform).shape[0])

    def validate(self, *, latent_dim: int = 16) -> None:
        n = self.batch_size
        expected = {
            "scenario_uniform": (n,),
            "c0_base_latent": (n, 40),
            "personality_latent": (n, 8, int(latent_dim)),
            "destination_uniform": (n, 8),
        }
        for name, shape in expected.items():
            value = np.asarray(getattr(self, name))
            if value.shape != shape:
                raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
            if not np.isfinite(value).all():
                raise ValueError(f"{name} contains a non-finite value")
        for name in ("scenario_uniform", "destination_uniform"):
            value = np.asarray(getattr(self, name))
            if np.any((value < 0.0) | (value >= 1.0)):
                raise ValueError(f"{name} values must lie in [0, 1)")

    @classmethod
    def sample(
        cls,
        n: int,
        *,
        seed: int,
        latent_dim: int = 16,
    ) -> "TrafficBotsExogenousState":
        if int(n) < 1 or int(latent_dim) < 1:
            raise ValueError("n and latent_dim must be positive")
        state = cls(
            scenario_uniform=_block_rng(seed, "scenario_uniform").random(
                int(n), dtype=np.float64
            ),
            c0_base_latent=_block_rng(seed, "c0_base_latent").standard_normal(
                (int(n), 40), dtype=np.float32
            ),
            personality_latent=_block_rng(seed, "personality_latent").standard_normal(
                (int(n), 8, int(latent_dim)), dtype=np.float32
            ),
            destination_uniform=_block_rng(seed, "destination_uniform").random(
                (int(n), 8), dtype=np.float64
            ),
        )
        state.validate(latent_dim=latent_dim)
        return state

    def as_dict(self) -> dict[str, np.ndarray]:
        return {
            name: np.asarray(getattr(self, name)).copy()
            for name in TRAFFICBOTS_RANDOM_BLOCKS
        }

    def replace_arrays(
        self, values: dict[str, np.ndarray]
    ) -> "TrafficBotsExogenousState":
        state = TrafficBotsExogenousState(
            **{
                name: np.asarray(values[name]).copy()
                for name in TRAFFICBOTS_RANDOM_BLOCKS
            }
        )
        state.validate(latent_dim=self.personality_latent.shape[-1])
        return state

    def select(self, indices: slice | np.ndarray) -> "TrafficBotsExogenousState":
        return TrafficBotsExogenousState(
            **{name: value[indices].copy() for name, value in self.as_dict().items()}
        )

    @classmethod
    def concatenate(
        cls, states: list["TrafficBotsExogenousState"]
    ) -> "TrafficBotsExogenousState":
        if not states:
            raise ValueError("states must not be empty")
        result = cls(
            **{
                name: np.concatenate(
                    [getattr(state, name) for state in states], axis=0
                )
                for name in TRAFFICBOTS_RANDOM_BLOCKS
            }
        )
        result.validate(latent_dim=states[0].personality_latent.shape[-1])
        return result

    def pcn_mutate(
        self, block: str, *, beta: float, seed: int
    ) -> "TrafficBotsExogenousState":
        if not 0.0 < float(beta) <= 1.0:
            raise ValueError("beta must lie in (0, 1]")
        values = self.as_dict()
        rng = np.random.default_rng(int(seed))
        if block == "scenario":
            values["scenario_uniform"] = rng.random(self.batch_size, dtype=np.float64)
        elif block == "destination_uniform":
            # An independent refresh preserves U(0,1); Gaussian pCN does not.
            values[block] = rng.random((self.batch_size, 8), dtype=np.float64)
        elif block in {"c0_base_latent", "personality_latent"}:
            current = values[block]
            innovation = rng.standard_normal(current.shape, dtype=np.float32)
            values[block] = (
                np.sqrt(1.0 - float(beta) ** 2) * current
                + float(beta) * innovation
            ).astype(np.float32)
        else:
            raise ValueError(f"unknown TrafficBots exogenous block {block!r}")
        return self.replace_arrays(values)

    def save(self, path: str | Path) -> None:
        self.validate(latent_dim=self.personality_latent.shape[-1])
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            **self.as_dict(),
            rng_schema=np.asarray(TRAFFICBOTS_RNG_SCHEMA),
        )

    @classmethod
    def load(cls, path: str | Path) -> "TrafficBotsExogenousState":
        with np.load(path) as values:
            required = {*TRAFFICBOTS_RANDOM_BLOCKS, "rng_schema"}
            if not required.issubset(values.files):
                raise ValueError("not a TrafficBots IDM exogenous-state archive")
            if str(values["rng_schema"]) != TRAFFICBOTS_RNG_SCHEMA:
                raise ValueError("unsupported TrafficBots IDM RNG schema")
            state = cls(
                **{
                    name: np.asarray(values[name]).copy()
                    for name in TRAFFICBOTS_RANDOM_BLOCKS
                }
            )
        state.validate(latent_dim=state.personality_latent.shape[-1])
        return state
