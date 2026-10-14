"""Explicit base-randomness contract for the hierarchical traffic world.

The arrays in :class:`WorldExogenousState` are the complete stochastic input
to Flow, diffusion and the 25 Hz response layer.  They are intentionally
plain NumPy values so an AMS implementation can serialize and mutate blocks
without treating an integer seed as a latent variable.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np

DIFFUSION_HORIZON_STEPS = 149
RNG_SCHEMA_NAME = "world_rng"
RNG_SCHEMA_VERSION = 8
WORLD_RNG_NAMESPACE = 1
WORLD_RANDOM_BLOCKS = (
    "scenario_uniform",
    "c0_base_latent",
    "k_base_latent",
    "diffusion_noise",
    "driver_latent",
    "maneuver_latent",
    "scene_innovations",
    "agent_response_innovations",
    "policy_response_innovations",
    "policy_calibration_innovations",
)
WORLD_RANDOM_STREAMS = {
    "scenario_uniform": 1,
    "c0_base_latent": 2,
    "k_base_latent": 3,
    "diffusion_noise": 4,
    "driver_latent": 5,
    "maneuver_latent": 6,
    "scene_innovations": 7,
    "agent_response_innovations": 8,
    "policy_response_innovations": 9,
    "policy_calibration_innovations": 10,
}


def named_generator(
    seed: int, *, namespace: int, stream: int, version: int = 1
) -> np.random.Generator:
    """Create a stable independent named stream from integer identifiers."""
    sequence = np.random.SeedSequence(
        (int(seed), int(namespace), int(version), int(stream))
    )
    return np.random.default_rng(sequence)


@dataclass(frozen=True)
class WorldExogenousState:
    """Prior-distributed base variables for one or more complete worlds."""

    scenario_uniform: np.ndarray
    c0_base_latent: np.ndarray
    k_base_latent: np.ndarray
    diffusion_noise: np.ndarray
    # Per-NPC persistent style draws, separate from frame noise.
    driver_latent: np.ndarray
    maneuver_latent: np.ndarray
    scene_innovations: np.ndarray
    agent_response_innovations: np.ndarray
    policy_response_innovations: np.ndarray
    # Independent response-calibration and residual-initialization randomness.
    policy_calibration_innovations: np.ndarray
    # Archive metadata; the runtime keeps a single response index.
    scene_refresh_responses: int = 25

    @property
    def batch_size(self) -> int:
        return int(np.asarray(self.scenario_uniform).shape[0])

    @property
    def response_steps(self) -> int:
        """The response horizon is owned by the per-response agent process."""
        return int(np.asarray(self.agent_response_innovations).shape[1])

    def validate(
        self,
        *,
        c0_dim: int = 26,
        k_dim: int = 36,
        horizon_frames: int | None = None,
        response_steps: int | None = None,
        scene_refresh_responses: int | None = None,
        scene_dim: int = 16,
        agent_dim: int = 16,
    ) -> None:
        n = self.batch_size
        if response_steps is None and horizon_frames is None:
            response_steps = self.response_steps
        if horizon_frames is None:
            horizon_frames = DIFFUSION_HORIZON_STEPS
        expected_diffusion_steps = int(horizon_frames)
        expected_response_steps = (
            self.response_steps if response_steps is None else int(response_steps)
        )
        refresh = (
            self.scene_refresh_responses
            if scene_refresh_responses is None
            else int(scene_refresh_responses)
        )
        if expected_diffusion_steps < 1 or expected_response_steps < 1 or refresh < 1:
            raise ValueError(
                "response_steps and scene_refresh_responses must be positive"
            )
        if expected_response_steps != self.response_steps:
            raise ValueError(
                "response_steps must equal agent_response_innovations.shape[1]"
            )
        expected_scene_steps = math.ceil(expected_response_steps / refresh)
        arrays = {
            "scenario_uniform": (self.scenario_uniform, (n,)),
            "c0_base_latent": (self.c0_base_latent, (n, c0_dim)),
            "k_base_latent": (self.k_base_latent, (n, k_dim)),
            "diffusion_noise": (
                self.diffusion_noise,
                (n, expected_diffusion_steps, 12),
            ),
            "driver_latent": (self.driver_latent, (n, 6, 5)),
            "maneuver_latent": (self.maneuver_latent, (n, 6, 4)),
            "scene_innovations": (
                self.scene_innovations,
                (n, expected_scene_steps, scene_dim),
            ),
            "agent_response_innovations": (
                self.agent_response_innovations,
                (n, expected_response_steps, 7, agent_dim),
            ),
            "policy_response_innovations": (
                self.policy_response_innovations,
                (n, expected_response_steps, 6, 2),
            ),
            "policy_calibration_innovations": (
                self.policy_calibration_innovations,
                (n, expected_response_steps, 6, 2),
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
        scene_refresh_responses: int = 25,
        scene_dim: int = 16,
        agent_dim: int = 16,
    ) -> "WorldExogenousState":
        if int(n) < 1 or int(response_steps) < 1:
            raise ValueError("n and response_steps must both be positive")
        if int(scene_refresh_responses) < 1:
            raise ValueError("scene_refresh_responses must be positive")

        # Keep named streams independent as the schema grows.
        def rng(block: str) -> np.random.Generator:
            version = 4 if block == "policy_calibration_innovations" else 2
            return named_generator(
                seed,
                namespace=WORLD_RNG_NAMESPACE,
                stream=WORLD_RANDOM_STREAMS[block],
                version=version,
            )

        scene_steps = math.ceil(int(response_steps) / int(scene_refresh_responses))
        state = cls(
            scenario_uniform=rng("scenario_uniform").random(int(n), dtype=np.float64),
            c0_base_latent=rng("c0_base_latent").standard_normal(
                (int(n), 26), dtype=np.float32
            ),
            k_base_latent=rng("k_base_latent").standard_normal(
                (int(n), 36), dtype=np.float32
            ),
            diffusion_noise=rng("diffusion_noise").standard_normal(
                (int(n), DIFFUSION_HORIZON_STEPS, 12), dtype=np.float32
            ),
            driver_latent=rng("driver_latent").standard_normal(
                (int(n), 6, 5), dtype=np.float32
            ),
            maneuver_latent=rng("maneuver_latent").standard_normal(
                (int(n), 6, 4), dtype=np.float32
            ),
            scene_innovations=rng("scene_innovations").standard_normal(
                (int(n), scene_steps, int(scene_dim)), dtype=np.float32
            ),
            agent_response_innovations=rng(
                "agent_response_innovations"
            ).standard_normal(
                (int(n), int(response_steps), 7, int(agent_dim)), dtype=np.float32
            ),
            policy_response_innovations=rng(
                "policy_response_innovations"
            ).standard_normal((int(n), int(response_steps), 6, 2), dtype=np.float32),
            policy_calibration_innovations=rng(
                "policy_calibration_innovations"
            ).standard_normal((int(n), int(response_steps), 6, 2), dtype=np.float32),
            scene_refresh_responses=int(scene_refresh_responses),
        )
        return state

    def save(self, path: str | Path) -> None:
        self.validate(
            response_steps=self.response_steps,
            scene_refresh_responses=self.scene_refresh_responses,
            scene_dim=np.asarray(self.scene_innovations).shape[-1],
            agent_dim=np.asarray(self.agent_response_innovations).shape[-1],
        )
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            **self.as_dict(),
            response_steps=np.asarray(self.response_steps, dtype=np.int64),
            scene_refresh_responses=np.asarray(
                self.scene_refresh_responses, dtype=np.int64
            ),
            schema_name=np.asarray(RNG_SCHEMA_NAME),
            schema_version=np.asarray(RNG_SCHEMA_VERSION, dtype=np.int64),
        )

    @classmethod
    def load(cls, path: str | Path) -> "WorldExogenousState":
        with np.load(path) as values:
            required = {*WORLD_RANDOM_BLOCKS, "schema_name", "schema_version"}
            if not required.issubset(values.files):
                raise ValueError("not a current world exogenous-state archive")
            if (
                str(values["schema_name"]) != RNG_SCHEMA_NAME
                or int(values["schema_version"]) != RNG_SCHEMA_VERSION
            ):
                raise ValueError("unsupported world RNG schema")
            refresh = (
                int(values["scene_refresh_responses"])
                if "scene_refresh_responses" in values
                else 25
            )
            state = cls(
                scenario_uniform=np.asarray(values["scenario_uniform"]),
                c0_base_latent=np.asarray(values["c0_base_latent"]),
                k_base_latent=np.asarray(values["k_base_latent"]),
                diffusion_noise=np.asarray(values["diffusion_noise"]),
                driver_latent=np.asarray(values["driver_latent"]),
                maneuver_latent=np.asarray(values["maneuver_latent"]),
                scene_innovations=np.asarray(values["scene_innovations"]),
                agent_response_innovations=np.asarray(
                    values["agent_response_innovations"]
                ),
                policy_response_innovations=np.asarray(
                    values["policy_response_innovations"]
                ),
                policy_calibration_innovations=np.asarray(
                    values["policy_calibration_innovations"]
                ),
                scene_refresh_responses=refresh,
            )
            if (
                "response_steps" in values
                and int(values["response_steps"]) != state.response_steps
            ):
                raise ValueError(
                    "response_steps metadata disagrees with agent innovations"
                )
        state.validate(
            response_steps=state.response_steps,
            scene_refresh_responses=state.scene_refresh_responses,
            scene_dim=np.asarray(state.scene_innovations).shape[-1],
            agent_dim=np.asarray(state.agent_response_innovations).shape[-1],
        )
        return state

    def as_dict(self) -> dict[str, np.ndarray]:
        return {
            name: np.asarray(getattr(self, name)).copy() for name in WORLD_RANDOM_BLOCKS
        }

    def replace_arrays(self, values: dict[str, np.ndarray]) -> "WorldExogenousState":
        """Rebuild this state after a generic subset-simulation update."""
        return WorldExogenousState(
            **{name: np.asarray(values[name]).copy() for name in WORLD_RANDOM_BLOCKS},
            scene_refresh_responses=self.scene_refresh_responses,
        )

    def select(self, indices: slice | np.ndarray) -> "WorldExogenousState":
        """Return a batch subset while preserving every randomness block."""
        return WorldExogenousState(
            **{name: values[indices].copy() for name, values in self.as_dict().items()},
            scene_refresh_responses=self.scene_refresh_responses,
        )

    @classmethod
    def concatenate(
        cls,
        states: list["WorldExogenousState"],
    ) -> "WorldExogenousState":
        """Combine compatible batches for vectorized world evaluation."""
        if not states:
            raise ValueError("states must not be empty")
        refresh = states[0].scene_refresh_responses
        if any(state.scene_refresh_responses != refresh for state in states[1:]):
            raise ValueError("states use different scene refresh intervals")
        values = [state.as_dict() for state in states]
        keys = values[0].keys()
        combined = {
            name: np.concatenate([item[name] for item in values], axis=0)
            for name in keys
        }
        return cls(**combined, scene_refresh_responses=refresh)

    def mutate(
        self,
        block: str,
        *,
        beta: float,
        seed: int,
        actor_mask: np.ndarray | None = None,
        time_range: tuple[int, int] | None = None,
    ) -> "WorldExogenousState":
        """Return a prior-preserving pCN proposal for one Gaussian block.

        The categorical scenario uniform is deliberately not treated as
        Gaussian.  AMS may replace that block with an independent U(0,1)
        proposal while preserving its prior exactly.
        """
        if not 0.0 < float(beta) <= 1.0:
            raise ValueError("beta must lie in (0, 1]")
        values = self.as_dict()
        if block == "scenario":
            if actor_mask is not None or time_range is not None:
                raise ValueError("scenario refresh cannot be actor/time masked")
            values["scenario_uniform"] = np.random.default_rng(int(seed)).random(
                self.batch_size, dtype=np.float64
            )
            block = "scenario_uniform"
        else:
            aliases = {
                "driver": "driver_latent",
                "maneuver": "maneuver_latent",
                "response": "agent_response_innovations",
                "policy": "policy_response_innovations",
            }
            block = aliases.get(block, block)
        if block != "scenario_uniform" and block in {
            "c0_base_latent",
            "k_base_latent",
            "diffusion_noise",
            "driver_latent",
            "maneuver_latent",
            "scene_innovations",
            "agent_response_innovations",
            "policy_response_innovations",
            "policy_calibration_innovations",
        }:
            current = values[block]
            innovation = np.random.default_rng(int(seed)).standard_normal(
                current.shape, dtype=np.float32
            )
            proposal = (
                np.sqrt(1.0 - float(beta) ** 2) * current + float(beta) * innovation
            ).astype(np.float32)
            selection = np.ones(current.shape, dtype=bool)
            if time_range is not None:
                if block not in {
                    "diffusion_noise",
                    "scene_innovations",
                    "agent_response_innovations",
                    "policy_response_innovations",
                    "policy_calibration_innovations",
                }:
                    raise ValueError(f"block {block!r} has no time axis")
                start, stop = (int(value) for value in time_range)
                if not 0 <= start <= stop <= current.shape[1]:
                    raise ValueError("time_range lies outside the selected block")
                selection[:] = False
                selection[:, start:stop] = True
            if actor_mask is not None:
                actors = np.asarray(actor_mask, dtype=bool)
                if actors.ndim == 1:
                    actors = np.broadcast_to(actors, (self.batch_size, 6))
                if actors.shape != (self.batch_size, 6):
                    raise ValueError("actor_mask must have shape [6] or [batch,6]")
                actor_selection = np.zeros(current.shape, dtype=bool)
                if block in {"driver_latent", "maneuver_latent"}:
                    actor_selection |= actors[..., None]
                elif block in {
                    "policy_response_innovations",
                    "policy_calibration_innovations",
                }:
                    actor_selection |= actors[:, None, :, None]
                elif block == "agent_response_innovations":
                    expanded = np.zeros(
                        (self.batch_size, current.shape[1], 7, current.shape[3]), bool
                    )
                    expanded[:, :, 1:, :] = actors[:, None, :, None]
                    actor_selection |= expanded
                elif block == "diffusion_noise":
                    actor_selection |= np.repeat(actors[:, None, :], 2, axis=-1)
                else:
                    raise ValueError(f"block {block!r} cannot be mutated per actor")
                selection &= actor_selection
            values[block] = np.where(selection, proposal, current)
        elif block != "scenario_uniform":
            raise ValueError(f"unknown exogenous block {block!r}")
        return WorldExogenousState(
            **values, scene_refresh_responses=self.scene_refresh_responses
        )

    def pcn_mutate(
        self, block: str, *, beta: float, seed: int
    ) -> "WorldExogenousState":
        """Backward-compatible whole-block pCN proposal."""
        return self.mutate(block, beta=beta, seed=seed)
