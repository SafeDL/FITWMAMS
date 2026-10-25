"""Explicit replayable randomness for traffic-world evaluation."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

DIFFUSION_HORIZON_STEPS = 149


def named_generator(seed, *, namespace, stream, version=1):
    return np.random.default_rng(
        np.random.SeedSequence((int(seed), int(namespace), int(version), int(stream)))
    )


RNG_SCHEMA_NAME = "world_rng"
RNG_SCHEMA_VERSION = 10
WORLD_RNG_NAMESPACE = 1
WORLD_RANDOM_BLOCKS = (
    "scenario_uniform",
    "c0_base_latent",
    "k_base_latent",
    "diffusion_noise",
    "driver_latent",
    "maneuver_latent",
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
    "policy_response_innovations": 9,
    "policy_calibration_innovations": 10,
}


@dataclass(frozen=True)
class WorldExogenousState:
    scenario_uniform: np.ndarray
    c0_base_latent: np.ndarray
    k_base_latent: np.ndarray
    diffusion_noise: np.ndarray
    driver_latent: np.ndarray
    maneuver_latent: np.ndarray
    policy_response_innovations: np.ndarray
    policy_calibration_innovations: np.ndarray

    @property
    def batch_size(self):
        return len(self.scenario_uniform)

    @property
    def response_steps(self):
        return self.policy_response_innovations.shape[1]

    def validate(self):
        n, steps = self.batch_size, self.response_steps
        expected = {
            "scenario_uniform": (n,),
            "c0_base_latent": (n, 26),
            "k_base_latent": (n, 36),
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
            (self.scenario_uniform < 0) | (self.scenario_uniform >= 1)
        ):
            raise ValueError("invalid response horizon or categorical uniform")

    @classmethod
    def sample(cls, n, *, seed, response_steps=DIFFUSION_HORIZON_STEPS):
        n, steps = int(n), int(response_steps)
        if n < 1 or steps < 1:
            raise ValueError("batch and response horizon must be positive")

        def rng(name):
            version = 4 if name == "policy_calibration_innovations" else 2
            return named_generator(
                seed,
                namespace=WORLD_RNG_NAMESPACE,
                stream=WORLD_RANDOM_STREAMS[name],
                version=version,
            )

        return cls(
            scenario_uniform=rng("scenario_uniform").random((n,), dtype=np.float64),
            c0_base_latent=rng("c0_base_latent").standard_normal(
                (n, 26), dtype=np.float32
            ),
            k_base_latent=rng("k_base_latent").standard_normal(
                (n, 36), dtype=np.float32
            ),
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
            name: np.asarray(getattr(self, name)).copy() for name in WORLD_RANDOM_BLOCKS
        }

    def replace_arrays(self, values):
        return type(self)(
            **{name: np.asarray(values[name]).copy() for name in WORLD_RANDOM_BLOCKS}
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
                for name in WORLD_RANDOM_BLOCKS
            }
        )

    def save(self, path):
        self.validate()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            **self.as_dict(),
            schema_name=np.asarray(RNG_SCHEMA_NAME),
            schema_version=np.asarray(RNG_SCHEMA_VERSION),
        )

    @classmethod
    def load(cls, path):
        with np.load(path) as values:
            if (
                str(values["schema_name"]) != RNG_SCHEMA_NAME
                or int(values["schema_version"]) != RNG_SCHEMA_VERSION
            ):
                raise ValueError("unsupported world RNG schema")
            state = cls(**{name: values[name].copy() for name in WORLD_RANDOM_BLOCKS})
        state.validate()
        return state

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
        if block != "scenario_uniform" and block in {
            "c0_base_latent",
            "k_base_latent",
            "diffusion_noise",
            "driver_latent",
            "maneuver_latent",
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
                elif block == "diffusion_noise":
                    actor_selection |= np.repeat(actors[:, None, :], 2, axis=-1)
                else:
                    raise ValueError(f"block {block!r} cannot be mutated per actor")
                selection &= actor_selection
            values[block] = np.where(selection, proposal, current)
        elif block != "scenario_uniform":
            raise ValueError(f"unknown exogenous block {block!r}")
        return WorldExogenousState(**values)
