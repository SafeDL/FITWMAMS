"""Request a correction for every eligible primary-search failure, resumably."""

import json
import multiprocessing
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import yaml

from ads.common.demonstrations import DemonstrationCache
from ads.common.instances import GeneratedInstance, UsageLedger, file_hash
from ads.common.reporting import write_json
from ads.common.setup import REPO_ROOT
from ads.common.world import FrozenWorldFactory
from ads.diffusion_planner.label_tracking import check as check_labels

CONFIG = REPO_ROOT / "ads/configs/safety_enhanced.yaml"
EXPERT_EXECUTION_BUDGET = 128


class CountingWorld:
    def __init__(self, world, counter):
        self.world, self.counter = world, counter

    def __getattr__(self, name):
        return getattr(self.world, name)

    def advance_response(self, action):
        result = self.world.advance_response(action)
        self.counter.steps += 1
        return result


class CountingFactory:
    def __init__(self, factory):
        self.factory, self.steps = factory, 0

    def __call__(self, event):
        return CountingWorld(self.factory(event), self)


def initialize(settings):
    global _settings, _output, _factory, _ledger
    _settings = settings
    _output = REPO_ROOT / settings["output_dir"]
    torch.set_num_threads(settings["runtime"]["torch_threads"])
    _factory = CountingFactory(
        FrozenWorldFactory(
            REPO_ROOT / settings["world_config"],
            device=settings["runtime"]["world_device"],
        )
    )
    _ledger = UsageLedger(_output / "usage.sqlite")


def execute(job):
    identity, path, expected_assets = job
    instance = GeneratedInstance.load(path)
    if instance.instance_hash != identity or instance.assets != expected_assets:
        raise ValueError("Expert input identity or frozen world changed")
    _ledger.register(instance, "repair")
    from ads.common import expert
    from ads.common.instances import json_hash

    candidate_budget = min(
        _settings["repair"]["expert_candidates"], EXPERT_EXECUTION_BUDGET - 1
    )
    teacher = {
        "schema": "shared_highway_expert_v2",
        "budget": candidate_budget,
        "assets": instance.assets,
        "implementation": file_hash(expert.__file__),
    }
    protocol = {
        "teacher": teacher,
        "total_execution_budget": EXPERT_EXECUTION_BUDGET,
        "tracker_sha256": file_hash(Path(__file__).with_name("label_tracking.py")),
        "tracking_trace_required": True,
    }
    destination = _output / "experts/inputs" / (identity + ".json")
    if destination.exists():
        row = json.loads(destination.read_text())
        if row.get("expert_protocol") != protocol:
            raise ValueError(
                "Completed expert input uses a different execution protocol"
            )
        if (
            row["instance_hash"] != identity
            or file_hash(row["trajectory_path"]) != row["trajectory_sha256"]
        ):
            raise ValueError("Completed expert input provenance changed")
        tracking = row["label_tracking"]
        if tracking is not None and (
            file_hash(tracking["trajectory_path"]) != tracking["trajectory_sha256"]
        ):
            raise ValueError("Completed expert tracking evidence changed")
        return {**row, "resumed": True}
    # Copy matching immutable cache records; never add outputs to the historical run.
    historical = REPO_ROOT / _settings["historical_run"] / "experts/cache"
    cache_root = _output / "experts/cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    key = json_hash({"instance": identity, "teacher": teacher})
    reused = False
    if (
        not (cache_root / (key + ".json")).exists()
        and (historical / (key + ".json")).exists()
    ):
        metadata = json.loads((historical / (key + ".json")).read_text())
        if (
            metadata["teacher_config"] != teacher
            or file_hash(historical / (key + ".npz")) != metadata["trajectory_sha256"]
        ):
            raise ValueError(
                "Historical expert cache does not match its declared identity"
            )
        for suffix in (".npz", ".json"):
            shutil.copy2(historical / (key + suffix), cache_root / (key + suffix))
        reused = True
    cache = DemonstrationCache(
        cache_root,
        _factory,
        algorithm="diffusion_planner",
        candidate_budget=candidate_budget,
        ledger=_ledger,
    )
    before = _factory.steps
    _, report = cache.get(instance)
    candidate_steps = _factory.steps - before
    tracking = None
    if report["qualified_demonstration"]:
        with np.load(cache_root / (key + ".npz"), allow_pickle=False) as data:
            tracking = check_labels(
                instance,
                data["states"],
                _factory,
                _ledger,
                trace_path=destination.with_suffix(".tracked.npz"),
            )
    tracking_calls = int(tracking is not None)
    total_executions = report["candidates"] + tracking_calls
    if total_executions > EXPERT_EXECUTION_BUDGET:
        raise AssertionError("Initial expert and tracking exceeded the shared budget")
    row = {
        **report,
        "expert_protocol": protocol,
        "expert_success": report["qualified_demonstration"],
        "label_tracking": tracking,
        "qualified_demonstration": bool(tracking and tracking["success"]),
        "samples": report["samples"] if tracking and tracking["success"] else 0,
        "new_candidate_physics_steps": candidate_steps,
        "label_tracking_new_executions": tracking_calls,
        "expert_execution_budget": EXPERT_EXECUTION_BUDGET,
        "candidate_and_tracking_executions": total_executions,
        "new_world_executions": report["new_candidates"] + tracking_calls,
        "root_sample_id": instance.root_sample_id,
        "instance_path": str(Path(path).resolve()),
        "trajectory_path": str((cache_root / (key + ".npz")).resolve()),
        "new_physics_steps": _factory.steps - before,
        "historical_cache_reuse": reused,
        "resumed": False,
    }
    write_json(destination, row)
    return row


def prepare(settings):
    output = REPO_ROOT / settings["output_dir"]
    source = json.loads((output / "sources.json").read_text())
    identities = sorted(
        {
            identity
            for method in source["methods"].values()
            for identity in method["eligible_failure_inputs"]
        }
    )
    jobs = [
        (identity, source["paths"][identity], source["assets"])
        for identity in identities
    ]
    reports = []
    with ProcessPoolExecutor(
        max_workers=settings["runtime"].get(
            "expert_workers", settings["runtime"]["workers"]
        ),
        mp_context=multiprocessing.get_context("spawn"),
        initializer=initialize,
        initargs=(settings,),
    ) as executor:
        for index, result in enumerate(executor.map(execute, jobs, chunksize=1), 1):
            reports.append(result)
            print(
                f"Failure experts: {index}/{len(jobs)}, qualified {sum(r['qualified_demonstration'] for r in reports)}",
                flush=True,
            )
    report = {
        "status": "complete",
        "sources_sha256": file_hash(output / "sources.json"),
        "inputs": len(reports),
        "protected_roots": source["protected_roots"],
        "qualified_inputs": sum(r["qualified_demonstration"] for r in reports),
        "expert_success_inputs": sum(r["expert_success"] for r in reports),
        "label_tracking_executions": sum(
            r["label_tracking_new_executions"] for r in reports
        ),
        "new_candidates_total": sum(r["new_candidates"] for r in reports),
        "new_world_executions_total": sum(r["new_world_executions"] for r in reports),
        "expert_execution_budget": EXPERT_EXECUTION_BUDGET,
        "new_physics_steps_total": sum(r["new_physics_steps"] for r in reports),
        "reports": reports,
    }
    write_json(output / "experts/report.json", report)
    return report


if __name__ == "__main__":
    prepare(yaml.safe_load(CONFIG.read_text()))
