"""Empirical test-context worlds with a fixed logged long-horizon ``K``.

This is deliberately separate from :mod:`randomness`: it defines the
reconstruction-backed evaluation population used when an experiment keeps the
held-out highD context ``(C0, M, K_GT)`` fixed and samples only the diffusion
and reactive response variables.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import numpy as np

from normalizing_flow.src.scenario import ScenarioBatch
from hierarchical_world_model.src.core.evaluation_scope import scoped_slot_mask
from .randomness import named_generator

EMPIRICAL_K_RNG_SCHEMA_NAME = "empirical_fixed_k_world"
EMPIRICAL_K_RNG_SCHEMA_VERSION = 4
EMPIRICAL_K_RNG_NAMESPACE = 2
EMPIRICAL_K_RANDOM_BLOCKS = (
    "test_row_uniform",
    "diffusion_noise",
    "driver_latent",
    "maneuver_latent",
    "scene_innovations",
    "agent_response_innovations",
    "policy_response_innovations",
    "policy_calibration_innovations",
)


@dataclass(frozen=True)
class EmpiricalKWorldState:
    """Random variables conditional on one uniformly sampled held-out row."""

    test_row_uniform: np.ndarray
    diffusion_noise: np.ndarray
    driver_latent: np.ndarray
    maneuver_latent: np.ndarray
    scene_innovations: np.ndarray
    agent_response_innovations: np.ndarray
    policy_response_innovations: np.ndarray
    policy_calibration_innovations: np.ndarray
    scene_refresh_responses: int = 25

    @property
    def batch_size(self) -> int:
        return int(np.asarray(self.test_row_uniform).shape[0])

    @property
    def response_steps(self) -> int:
        return int(np.asarray(self.agent_response_innovations).shape[1])

    def validate(
        self,
        *,
        response_steps: int | None = None,
        scene_refresh_responses: int | None = None,
        scene_dim: int = 16,
        agent_dim: int = 16,
    ) -> None:
        n = self.batch_size
        steps = self.response_steps if response_steps is None else int(response_steps)
        refresh = (
            self.scene_refresh_responses
            if scene_refresh_responses is None
            else int(scene_refresh_responses)
        )
        if steps != self.response_steps or steps < 1 or refresh < 1:
            raise ValueError("invalid empirical-K response-time contract")
        expected = {
            "test_row_uniform": (self.test_row_uniform, (n,)),
            "diffusion_noise": (self.diffusion_noise, (n, 149, 12)),
            "driver_latent": (self.driver_latent, (n, 6, 5)),
            "maneuver_latent": (self.maneuver_latent, (n, 6, 4)),
            "scene_innovations": (
                self.scene_innovations,
                (n, math.ceil(steps / refresh), int(scene_dim)),
            ),
            "agent_response_innovations": (
                self.agent_response_innovations,
                (n, steps, 7, int(agent_dim)),
            ),
            "policy_response_innovations": (
                self.policy_response_innovations,
                (n, steps, 6, 2),
            ),
            "policy_calibration_innovations": (
                self.policy_calibration_innovations,
                (n, steps, 6, 2),
            ),
        }
        for name, (value, shape) in expected.items():
            array = np.asarray(value)
            if array.shape != shape or not np.isfinite(array).all():
                raise ValueError(f"{name} must be finite with shape {shape}")
        uniform = np.asarray(self.test_row_uniform)
        if np.any((uniform < 0.0) | (uniform >= 1.0)):
            raise ValueError("test_row_uniform must lie in [0, 1)")

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
    ) -> "EmpiricalKWorldState":
        if int(n) < 1 or int(response_steps) < 1 or int(scene_refresh_responses) < 1:
            raise ValueError(
                "n, response_steps and scene_refresh_responses must be positive"
            )

        def rng(name: str) -> np.random.Generator:
            return named_generator(
                seed,
                namespace=EMPIRICAL_K_RNG_NAMESPACE,
                stream=EMPIRICAL_K_RANDOM_BLOCKS.index(name) + 1,
                version=2,
            )

        scenes = math.ceil(int(response_steps) / int(scene_refresh_responses))
        result = cls(
            test_row_uniform=rng("test_row_uniform").random(int(n), dtype=np.float64),
            diffusion_noise=rng("diffusion_noise").standard_normal(
                (int(n), 149, 12), dtype=np.float32
            ),
            driver_latent=rng("driver_latent").standard_normal(
                (int(n), 6, 5), dtype=np.float32
            ),
            maneuver_latent=rng("maneuver_latent").standard_normal(
                (int(n), 6, 4), dtype=np.float32
            ),
            scene_innovations=rng("scene_innovations").standard_normal(
                (int(n), scenes, int(scene_dim)), dtype=np.float32
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
        return result

    def as_dict(self) -> dict[str, np.ndarray]:
        return {
            name: np.asarray(getattr(self, name)).copy()
            for name in EMPIRICAL_K_RANDOM_BLOCKS
        }

    def select(self, indices: slice | np.ndarray) -> "EmpiricalKWorldState":
        return EmpiricalKWorldState(
            **{name: value[indices].copy() for name, value in self.as_dict().items()},
            scene_refresh_responses=self.scene_refresh_responses,
        )

    @classmethod
    def concatenate(
        cls, states: list["EmpiricalKWorldState"]
    ) -> "EmpiricalKWorldState":
        if not states:
            raise ValueError("states must not be empty")
        refresh = states[0].scene_refresh_responses
        if any(item.scene_refresh_responses != refresh for item in states):
            raise ValueError("empirical-K states have incompatible scene refresh")
        values = [item.as_dict() for item in states]
        return cls(
            **{
                name: np.concatenate([item[name] for item in values], axis=0)
                for name in EMPIRICAL_K_RANDOM_BLOCKS
            },
            scene_refresh_responses=refresh,
        )

    def replace_arrays(self, values: dict[str, np.ndarray]) -> "EmpiricalKWorldState":
        return EmpiricalKWorldState(
            **{
                name: np.asarray(values[name]).copy()
                for name in EMPIRICAL_K_RANDOM_BLOCKS
            },
            scene_refresh_responses=self.scene_refresh_responses,
        )

    def pcn_mutate(
        self, block: str, *, beta: float, seed: int
    ) -> "EmpiricalKWorldState":
        if not 0.0 < float(beta) <= 1.0:
            raise ValueError("beta must lie in (0, 1]")
        values = self.as_dict()
        rng = np.random.default_rng(int(seed))
        if block == "test_context":
            values["test_row_uniform"] = rng.random(self.batch_size, dtype=np.float64)
        elif block in {
            "diffusion_noise",
            "driver_latent",
            "maneuver_latent",
            "scene_innovations",
            "agent_response_innovations",
            "policy_response_innovations",
            "policy_calibration_innovations",
        }:
            current = values[block]
            innovation = rng.standard_normal(current.shape, dtype=np.float32)
            values[block] = (
                np.sqrt(1.0 - float(beta) ** 2) * current + float(beta) * innovation
            ).astype(np.float32)
        else:
            raise ValueError(f"unknown empirical-K block {block!r}")
        return self.replace_arrays(values)

    def save(self, path: str | Path) -> None:
        self.validate(
            response_steps=self.response_steps,
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
            schema_name=np.asarray(EMPIRICAL_K_RNG_SCHEMA_NAME),
            schema_version=np.asarray(EMPIRICAL_K_RNG_SCHEMA_VERSION, dtype=np.int64),
        )

    @classmethod
    def load(cls, path: str | Path) -> "EmpiricalKWorldState":
        with np.load(path) as values:
            if (
                str(values["schema_name"]) != EMPIRICAL_K_RNG_SCHEMA_NAME
                or int(values["schema_version"]) != EMPIRICAL_K_RNG_SCHEMA_VERSION
            ):
                raise ValueError("not an empirical fixed-K world archive")
            result = cls(
                **{
                    name: np.asarray(values[name]) for name in EMPIRICAL_K_RANDOM_BLOCKS
                },
                scene_refresh_responses=int(values["scene_refresh_responses"]),
            )
        result.validate(
            response_steps=result.response_steps,
            scene_dim=np.asarray(result.scene_innovations).shape[-1],
            agent_dim=np.asarray(result.agent_response_innovations).shape[-1],
        )
        return result


class EmpiricalKContextSampler:
    """Bind a hierarchy sampler to immutable test contexts and logged ``K_GT``."""

    test_space = "empirical_test_fixed_k_gt"

    def __init__(self, base_sampler: Any, bundle: Any, test_rows: np.ndarray) -> None:
        self.base_sampler = base_sampler
        self.response = base_sampler.response
        self.device = base_sampler.device
        self.rows = np.asarray(test_rows, np.int64)
        if not len(self.rows):
            raise ValueError("the empirical fixed-K population is empty")
        self._c0 = np.asarray(bundle.flow_arrays["features"])[
            np.asarray(bundle.flow_row_for_sequence)[self.rows]
        ].astype(np.float32)
        self._slot_mask = np.asarray(bundle.flow_arrays["slot_mask"])[
            np.asarray(bundle.flow_row_for_sequence)[self.rows]
        ].astype(bool)
        self._k = np.asarray(bundle.flow_arrays["trajectory_constraint"])[
            np.asarray(bundle.flow_row_for_sequence)[self.rows]
        ].astype(np.float32)
        self._vehicle_dimensions = np.asarray(bundle.arrays["vehicle_dimensions"])[
            self.rows
        ].astype(np.float32)

    @property
    def context_contract(self) -> dict[str, Any]:
        return {
            "test_space": self.test_space,
            "sampling": "uniform_over_recording-isolated_test_sequences",
            "fixed_variables": ["M", "C0", "K_GT"],
            "sampled_variables": [
                "diffusion_noise",
                "driver_latent",
                "maneuver_latent",
                "scene_innovations",
                "agent_response_innovations",
                "policy_response_innovations",
                "policy_calibration_innovations",
            ],
            "test_sequences": int(len(self.rows)),
        }

    def sample_world_exogenous(
        self, n: int, *, seed: int, response_steps: int = 149
    ) -> EmpiricalKWorldState:
        return EmpiricalKWorldState.sample(
            n,
            seed=seed,
            response_steps=response_steps,
            scene_refresh_responses=self.response.cfg.scene_refresh_responses,
            scene_dim=self.response.cfg.scene_latent_dim,
            agent_dim=self.response.cfg.agent_latent_dim,
        )

    def _indices(self, world: EmpiricalKWorldState) -> np.ndarray:
        return np.minimum(
            (np.asarray(world.test_row_uniform) * len(self.rows)).astype(np.int64),
            len(self.rows) - 1,
        )

    def compose_exogenous(self, world: EmpiricalKWorldState) -> Any:
        index = self._indices(world)
        c0 = self._c0[index].copy()
        slots = np.asarray(scoped_slot_mask(self._slot_mask[index]), bool)
        c0, slots = self.base_sampler._scope_condition(c0, slots)
        # Dataset-level normalized inactive values can encode stale values for
        # invalid slots. The fixed-context protocol uses zeros only for those
        # invalid slots; every valid canonical slot, including same_rear,
        # remains available to inference.
        scenario = ScenarioBatch(
            c0=c0,
            slot_mask=slots,
            trajectory_constraint=self._k[index].copy(),
            trajectory_constraint_valid=np.broadcast_to(
                slots[..., None], self._k[index].shape
            ).copy(),
            vehicle_dimensions=self._vehicle_dimensions[index].copy(),
            c0_normalized_reference=np.zeros_like(c0, dtype=np.float32),
        )
        return self.base_sampler._compose(
            scenario,
            scenario_seed=0,
            motion_seed=0,
            exogenous_state=world,
        )

    def create_world(self, *args: Any, **kwargs: Any) -> Any:
        return self.base_sampler.create_world(*args, **kwargs)
