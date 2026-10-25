"""Explicit replayable randomness for traffic-world evaluation."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from hierarchical_world_model.src.core.evaluation_scope import scoped_slot_mask
from normalizing_flow.src.scenario import ScenarioBatch

from .randomness import DIFFUSION_HORIZON_STEPS, named_generator

EMPIRICAL_K_RNG_SCHEMA_NAME = "empirical_fixed_k_world"
EMPIRICAL_K_RNG_SCHEMA_VERSION = 6
EMPIRICAL_K_RNG_NAMESPACE = 2
EMPIRICAL_K_RANDOM_BLOCKS = (
    "test_row_uniform",
    "diffusion_noise",
    "driver_latent",
    "maneuver_latent",
    "policy_response_innovations",
    "policy_calibration_innovations",
)
EMPIRICAL_K_RANDOM_STREAMS = {
    "test_row_uniform": 1,
    "diffusion_noise": 2,
    "driver_latent": 3,
    "maneuver_latent": 4,
    "policy_response_innovations": 7,
    "policy_calibration_innovations": 8,
}


@dataclass(frozen=True)
class EmpiricalKWorldState:
    test_row_uniform: np.ndarray
    diffusion_noise: np.ndarray
    driver_latent: np.ndarray
    maneuver_latent: np.ndarray
    policy_response_innovations: np.ndarray
    policy_calibration_innovations: np.ndarray

    @property
    def batch_size(self):
        return len(self.test_row_uniform)

    @property
    def response_steps(self):
        return self.policy_response_innovations.shape[1]

    def validate(self):
        n, steps = self.batch_size, self.response_steps
        expected = {
            "test_row_uniform": (n,),
            "diffusion_noise": (n, DIFFUSION_HORIZON_STEPS, 12),
            "driver_latent": (n, 6, 5),
            "maneuver_latent": (n, 6, 4),
            "policy_response_innovations": (n, steps, 6, 2),
            "policy_calibration_innovations": (n, steps, 6, 2),
        }
        for name, shape in expected.items():
            value = np.asarray(getattr(self, name))
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"{name} must be finite with shape {shape}")
        if steps < 1 or np.any(
            (self.test_row_uniform < 0) | (self.test_row_uniform >= 1)
        ):
            raise ValueError("invalid response horizon or categorical uniform")

    @classmethod
    def sample(cls, n, *, seed, response_steps=DIFFUSION_HORIZON_STEPS):
        n, steps = int(n), int(response_steps)
        if n < 1 or steps < 1:
            raise ValueError("batch and response horizon must be positive")

        def rng(name):
            version = 2
            return named_generator(
                seed,
                namespace=EMPIRICAL_K_RNG_NAMESPACE,
                stream=EMPIRICAL_K_RANDOM_STREAMS[name],
                version=version,
            )

        return cls(
            test_row_uniform=rng("test_row_uniform").random((n,), dtype=np.float64),
            diffusion_noise=rng("diffusion_noise").standard_normal(
                (n, DIFFUSION_HORIZON_STEPS, 12), dtype=np.float32
            ),
            driver_latent=rng("driver_latent").standard_normal(
                (n, 6, 5), dtype=np.float32
            ),
            maneuver_latent=rng("maneuver_latent").standard_normal(
                (n, 6, 4), dtype=np.float32
            ),
            policy_response_innovations=rng(
                "policy_response_innovations"
            ).standard_normal((n, steps, 6, 2), dtype=np.float32),
            policy_calibration_innovations=rng(
                "policy_calibration_innovations"
            ).standard_normal((n, steps, 6, 2), dtype=np.float32),
        )

    def as_dict(self):
        return {
            name: np.asarray(getattr(self, name)).copy()
            for name in EMPIRICAL_K_RANDOM_BLOCKS
        }

    def replace_arrays(self, values):
        return type(self)(
            **{
                name: np.asarray(values[name]).copy()
                for name in EMPIRICAL_K_RANDOM_BLOCKS
            }
        )

    def select(self, indices):
        return self.replace_arrays(
            {name: value[indices] for name, value in self.as_dict().items()}
        )

    @classmethod
    def concatenate(cls, states):
        return cls(
            **{
                name: np.concatenate([getattr(s, name) for s in states])
                for name in EMPIRICAL_K_RANDOM_BLOCKS
            }
        )

    def save(self, path):
        self.validate()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            **self.as_dict(),
            schema_name=np.asarray(EMPIRICAL_K_RNG_SCHEMA_NAME),
            schema_version=np.asarray(EMPIRICAL_K_RNG_SCHEMA_VERSION),
        )

    @classmethod
    def load(cls, path):
        with np.load(path) as values:
            if (
                str(values["schema_name"]) != EMPIRICAL_K_RNG_SCHEMA_NAME
                or int(values["schema_version"]) != EMPIRICAL_K_RNG_SCHEMA_VERSION
            ):
                raise ValueError("unsupported world RNG schema")
            state = cls(
                **{name: values[name].copy() for name in EMPIRICAL_K_RANDOM_BLOCKS}
            )
        state.validate()
        return state

    def mutate(self, block: str, *, beta: float, seed: int) -> "EmpiricalKWorldState":
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


class EmpiricalKContextSampler:
    """Bind a hierarchy sampler to immutable test contexts and logged ``K_GT``."""

    test_space = "empirical_test_fixed_k_gt"

    def __init__(self, base_sampler: Any, bundle: Any, test_rows: np.ndarray) -> None:
        self.base_sampler = base_sampler
        self.cfg = base_sampler.cfg
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
                "policy_response_innovations",
                "policy_calibration_innovations",
            ],
            "test_sequences": len(self.rows),
        }

    def sample_world_exogenous(
        self, n: int, *, seed: int, response_steps: int = 149
    ) -> EmpiricalKWorldState:
        return EmpiricalKWorldState.sample(
            n,
            seed=seed,
            response_steps=response_steps,
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
