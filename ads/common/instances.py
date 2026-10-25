"""Materialise the existing full prior without conditioning on logged events."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from ads.common.world import WorldEvent
from hierarchical_world_model.src.randomness import (
    WORLD_RANDOM_BLOCKS,
    WorldExogenousState,
)

PURPOSES = {"development": 11, "search": 23, "repair": 37, "final_test": 101}
SCHEMA = "ads_world_instance_v2"
SOURCE_KINDS = ("tail_generated", "natural_generated", "empirical_natural")


def file_hash(path: str | Path) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    if path.is_dir():
        for entry in sorted(p for p in path.rglob("*") if p.is_file()):
            digest.update(str(entry.relative_to(path)).encode())
            digest.update(file_hash(entry).encode())
        return digest.hexdigest()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_record(path: str | Path | None) -> dict[str, object] | None:
    """Describe an immutable file input by resolved path, size and digest."""
    if path is None:
        return None
    value = Path(path).resolve()
    if not value.is_file():
        raise FileNotFoundError(value)
    return {
        "path": str(value),
        "bytes": int(value.stat().st_size),
        "sha256": file_hash(value),
    }


def json_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def array_hash(values: dict[str, np.ndarray], assets: dict[str, str]) -> str:
    digest = hashlib.sha256(json_hash(assets).encode())
    for name, value in sorted(values.items()):
        array = np.ascontiguousarray(value)
        digest.update(name.encode())
        digest.update(array.dtype.str.encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class GeneratedInstance:
    event: WorldEvent
    root_sample_id: str
    parent_id: str | None
    purpose: str
    assets: dict[str, str]
    mutation_block: str | None = None
    source_kind: str = "tail_generated"
    source_metadata: dict = field(default_factory=dict)

    @property
    def instance_hash(self) -> str:
        return array_hash(self.arrays(), self.assets)

    def arrays(self) -> dict[str, np.ndarray]:
        event = self.event
        result = {
            name: np.asarray(getattr(event, name))
            for name in (
                "initial_states",
                "valid",
                "soft_reference",
                "map_polylines",
                "map_polyline_valid",
                "vehicle_dimensions",
            )
        }
        result.update(
            {"rng_" + k: v for k, v in event.exogenous_state.as_dict().items()}
        )
        result["initial_history"] = (
            np.empty((0, 7, 6), np.float32)
            if event.initial_history is None
            else np.asarray(event.initial_history)
        )
        result["initial_history_valid"] = (
            np.empty((0, 7), bool)
            if event.initial_history_valid is None
            else np.asarray(event.initial_history_valid)
        )
        return result

    def validate(self) -> None:
        if self.purpose not in PURPOSES:
            raise ValueError(f"unknown instance purpose {self.purpose}")
        self.event.exogenous_state.validate()
        if self.event.exogenous_state.batch_size != 1:
            raise ValueError("a stored instance contains exactly one world")
        if self.event.event_id != self.instance_hash:
            raise ValueError("instance content hash mismatch")
        if self.source_kind not in SOURCE_KINDS:
            raise ValueError("unknown instance source kind")
        if (
            self.source_kind != "empirical_natural"
            and self.event.initial_history is not None
        ):
            raise ValueError("generated instances use the existing cold start")
        history, mask = (
            self.arrays()["initial_history"],
            self.arrays()["initial_history_valid"],
        )
        if history.shape != (len(history), 7, 6) or mask.shape != (len(history), 7):
            raise ValueError("history and per-agent validity differ")
        if self.source_kind == "empirical_natural" and not len(history):
            raise ValueError("empirical instances require their actual history")
        for name, value in self.arrays().items():
            if not np.isfinite(value).all():
                raise FloatingPointError(f"non-finite generated field: {name}")
        if self.event.initial_states.shape != (7, 6) or self.event.valid.shape != (7,):
            raise ValueError("invalid generated state dimensions")
        if self.event.vehicle_dimensions.shape != (7, 2) or np.any(
            self.event.vehicle_dimensions <= 0
        ):
            raise ValueError("invalid generated vehicle dimensions")

    def save(self, path: str | Path) -> None:
        self.validate()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "schema": SCHEMA,
            "instance_hash": self.instance_hash,
            "root_sample_id": self.root_sample_id,
            "parent_id": self.parent_id,
            "purpose": self.purpose,
            "assets": self.assets,
            "mutation_block": self.mutation_block,
            "source_kind": self.source_kind,
            "source_metadata": self.source_metadata,
            "source_event_id": self.event.source_event_id,
        }
        # Exclusive creation keeps a partial or previous result from being overwritten.
        with path.open("xb") as stream:
            np.savez_compressed(
                stream,
                metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
                **self.arrays(),
            )

    @classmethod
    def load(cls, path: str | Path) -> GeneratedInstance:
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["metadata"]))
            if meta["schema"] != SCHEMA:
                raise ValueError("unsupported generated instance schema")
            event = WorldEvent(
                event_id=meta["instance_hash"],
                source_event_id=meta["source_event_id"],
                **{
                    k: data[k].copy()
                    for k in (
                        "initial_states",
                        "valid",
                        "soft_reference",
                        "map_polylines",
                        "map_polyline_valid",
                        "vehicle_dimensions",
                    )
                },
                exogenous_state=WorldExogenousState(
                    **{k: data["rng_" + k].copy() for k in WORLD_RANDOM_BLOCKS}
                ),
                split="test" if meta["purpose"] == "final_test" else "train",
                initial_history=data["initial_history"].copy()
                if len(data["initial_history"])
                else None,
                initial_history_valid=data["initial_history_valid"].copy()
                if len(data["initial_history_valid"])
                else None,
            )
        result = cls(
            event,
            meta["root_sample_id"],
            meta["parent_id"],
            meta["purpose"],
            meta["assets"],
            meta["mutation_block"],
            meta["source_kind"],
            meta["source_metadata"],
        )
        result.validate()
        return result


class UsageLedger:
    """Reject cross-use of final-test content and its independent seed roots."""

    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = Path(path)
        with sqlite3.connect(self.path) as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS usage (identity TEXT, group_name TEXT, PRIMARY KEY(identity, group_name))"
            )

    def register(self, instance: GeneratedInstance, purpose: str) -> None:
        if purpose not in PURPOSES:
            raise ValueError("unknown data purpose")
        if instance.purpose == "final_test" and purpose != "final_test":
            raise ValueError(
                "final-test instances cannot enter development or training"
            )
        if instance.purpose == "development" and purpose != "development":
            raise ValueError("development evaluation instances cannot enter learning")
        group = (
            "test"
            if purpose == "final_test"
            else "validation"
            if instance.purpose == "development"
            else "learning"
        )
        identities = [
            "instance:" + instance.instance_hash,
            "root:" + instance.root_sample_id,
        ]
        if instance.parent_id is not None:
            identities.append("instance:" + instance.parent_id)
        with sqlite3.connect(self.path) as db:
            db.execute("BEGIN IMMEDIATE")
            for identity in identities:
                old = {
                    r[0]
                    for r in db.execute(
                        "SELECT group_name FROM usage WHERE identity=?", (identity,)
                    )
                }
                if old - {group}:
                    raise ValueError("final-test/development data leakage")
                db.execute(
                    "INSERT OR IGNORE INTO usage VALUES (?,?)", (identity, group)
                )


class InstanceStore(Sequence[GeneratedInstance]):
    def __init__(self, directory: str | Path, *, purpose: str, ledger: UsageLedger):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.purpose, self.ledger = purpose, ledger
        self.paths = sorted(self.directory.glob("*.npz"))

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        instance = GeneratedInstance.load(self.paths[index])
        self.ledger.register(instance, self.purpose)
        return instance

    def add(self, instance: GeneratedInstance) -> None:
        self.ledger.register(instance, self.purpose)
        path = self.directory / (instance.instance_hash + ".npz")
        if path.exists():
            saved = GeneratedInstance.load(path)
            if (
                saved.root_sample_id != instance.root_sample_id
                or saved.parent_id != instance.parent_id
            ):
                raise ValueError("same concrete instance has inconsistent lineage")
            return
        instance.save(path)
        self.paths.append(path)


class GeneratedSource:
    def __init__(
        self,
        sampler,
        assets: dict[str, str],
        *,
        master_seed: int,
        purpose: str,
        stream: int = 0,
        source_kind: str = "tail_generated",
    ):
        if purpose not in PURPOSES:
            raise ValueError("unknown generation purpose")
        self.sampler, self.assets = sampler, assets
        self.master_seed, self.purpose, self.stream = master_seed, purpose, stream
        if source_kind not in ("tail_generated", "natural_generated"):
            raise ValueError("GeneratedSource requires a generated distribution")
        self.source_kind = source_kind
        self.index = 0
        self.generation_elapsed_s = 0.0

    def _materialise(self, omega, roots, parents, blocks) -> list[GeneratedInstance]:
        started = time.perf_counter()
        sample = self.sampler.compose_exogenous(omega)
        # This is exactly the existing HierarchicalWorldSampler.create_world map.
        x = np.linspace(-200, 200, 8, dtype=np.float32)
        offsets = np.asarray(self.sampler.generated_road["lane_centers_m"], np.float32)
        results = []
        for i in range(omega.batch_size):
            maps = np.zeros((8, 8, 6), np.float32)
            maps[: len(offsets), :, 0] = x
            maps[: len(offsets), :, 1] = (
                sample.initial_states[i, 0, 1] + offsets[:, None]
            )
            maps[: len(offsets), :, 2] = 1
            maps[: len(offsets), :, 4] = self.sampler.generated_road["lane_width_m"]
            valid = np.zeros((8, 8), bool)
            valid[: len(offsets)] = True
            event = WorldEvent(
                "",
                "",
                sample.initial_states[i].copy(),
                sample.initial_valid[i].copy(),
                sample.soft_plan[i].copy(),
                maps,
                valid,
                omega.select(slice(i, i + 1)),
                vehicle_dimensions=sample.vehicle_dimensions[i].copy(),
                split="test" if self.purpose == "final_test" else "train",
            )
            instance = GeneratedInstance(
                event,
                roots[i],
                parents[i],
                self.purpose,
                dict(self.assets),
                blocks[i],
                self.source_kind,
            )
            instance = replace(
                instance, event=replace(event, event_id=instance.instance_hash)
            )
            instance.validate()
            results.append(instance)
        self.generation_elapsed_s += time.perf_counter() - started
        return results

    def prior_identity(self, index):
        parts = (self.master_seed, PURPOSES[self.purpose], self.stream, index)
        seed = int(np.random.SeedSequence(parts).generate_state(1, dtype=np.uint64)[0])
        return seed, json_hash({"seed": seed, "assets": self.assets})

    def sample(self, n: int) -> list[GeneratedInstance]:
        states, roots = [], []
        for _ in range(n):
            seed, root = self.prior_identity(self.index)
            roots.append(root)
            states.append(WorldExogenousState.sample(1, seed=seed))
            self.index += 1
        return self._materialise(
            WorldExogenousState.concatenate(states), roots, [None] * n, [None] * n
        )

    def mutation(
        self, parent: GeneratedInstance, omega: WorldExogenousState, block: str
    ) -> GeneratedInstance:
        if parent.purpose == "final_test" or parent.assets != self.assets:
            raise ValueError("cannot mutate test data or a different frozen world")
        return self._materialise(
            omega, [parent.root_sample_id], [parent.instance_hash], [block]
        )[0]


def generate_to_store(
    source: GeneratedSource, store: InstanceStore, n: int, batch_size: int = 32
) -> None:
    for start in range(0, n, batch_size):
        for instance in source.sample(min(batch_size, n - start)):
            store.add(instance)
