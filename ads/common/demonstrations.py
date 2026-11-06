"""One validated expert execution cache shared by native nuPlan supervised planners."""

import fcntl
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ads.common import expert
from ads.common.instances import file_hash, json_hash


class DemonstrationCache:
    def __init__(self, directory, factory, *, algorithm, candidate_budget, ledger=None):
        self.directory, self.factory = Path(directory), factory
        self.algorithm, self.budget = algorithm, candidate_budget
        self.ledger = ledger
        self.directory.mkdir(parents=True, exist_ok=True)

    def get(self, instance):
        if instance.purpose in ("final_test", "development"):
            raise ValueError("expert cannot label final-test or development instances")
        if self.ledger is not None:
            self.ledger.register(instance, "repair")
        config = {
            "schema": "shared_highway_expert_v2",
            "budget": self.budget,
            "assets": instance.assets,
            "implementation": file_hash(expert.__file__),
        }
        key = json_hash({"instance": instance.instance_hash, "teacher": config})
        stem = self.directory / key
        with stem.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            metadata = stem.with_suffix(".json")
            new_candidates, new_elapsed = 0, 0.0
            if metadata.exists():
                report = json.loads(metadata.read_text())
            else:
                started = time.perf_counter()
                execution, report = expert.successful_expert(
                    instance, self.factory, budget=self.budget
                )
                np.savez_compressed(
                    stem.with_suffix(".npz"),
                    states=(
                        execution.states
                        if execution
                        else np.empty((0, 7, 6), np.float32)
                    ),
                    controls=(
                        execution.controls
                        if execution
                        else np.empty((0, 2), np.float32)
                    ),
                    candidate=np.asarray(
                        json.dumps(asdict(execution.candidate) if execution else {})
                    ),
                )
                report.update(
                    teacher_config=config,
                    trajectory_sha256=file_hash(stem.with_suffix(".npz")),
                )
                metadata.write_text(
                    json.dumps(report, indent=2, allow_nan=False) + "\n"
                )
                new_candidates, new_elapsed = (
                    report["candidates"],
                    time.perf_counter() - started,
                )
            if file_hash(stem.with_suffix(".npz")) != report["trajectory_sha256"]:
                raise ValueError("cached expert trajectory changed")
            samples = []
            if report["qualified_demonstration"]:
                with np.load(stem.with_suffix(".npz"), allow_pickle=False) as data:
                    from ads.nuplan.teacher import demonstration

                    samples = demonstration(instance, data["states"], self.algorithm)
        return samples, {
            **report,
            "samples": len(samples),
            "new_candidates": new_candidates,
            "new_elapsed_s": new_elapsed,
            "cache_path": str(metadata.resolve()),
        }
