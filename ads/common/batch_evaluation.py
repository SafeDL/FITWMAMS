"""Parallel, resumable policy execution with per-instance provenance and cost."""

import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from ads.common.evaluation import WorldEvaluator
from ads.common.instances import GeneratedInstance, UsageLedger, file_hash
from ads.common.reporting import write_json
from ads.common.setup import REPO_ROOT, load_policy
from ads.common.world import FrozenWorldFactory
from tools.evt import GPDTailModel


def initialize(settings, algorithm, policy_path, ledger_path, purpose):
    global _evaluator, _policy_hash
    torch.set_num_threads(settings["runtime"]["torch_threads"])
    model = load_policy(algorithm, policy_path, device=settings["runtime"]["device"])
    factory = FrozenWorldFactory(
        REPO_ROOT / settings["world_config"], device=settings["runtime"]["world_device"]
    )
    evt = GPDTailModel.from_dict(
        json.loads((REPO_ROOT / settings["evt_model"]).read_text())
    )
    _policy_hash = file_hash(policy_path)
    _evaluator = WorldEvaluator(
        model,
        algorithm,
        factory,
        evt,
        ledger=UsageLedger(ledger_path),
        purpose=purpose,
        seed=settings["master_seed"],
    )


def execute(job):
    path, directory, save_trace = job
    instance = GeneratedInstance.load(path)
    target = Path(directory) / (instance.instance_hash + ".json")
    if target.exists():
        entry = json.loads(target.read_text())
        if (
            entry["policy_sha256"] != _policy_hash
            or entry["record"]["instance_hash"] != instance.instance_hash
        ):
            raise ValueError("Saved evaluation belongs to another policy/instance")
        if (
            entry.get("trajectory")
            and file_hash(entry["trajectory"]) != entry["trajectory_sha256"]
        ):
            raise ValueError("Saved successful trajectory changed")
        return {**entry, "cache_hit": True}
    result = _evaluator(instance)
    entry = {"policy_sha256": _policy_hash, "record": result.record, "cache_hit": False}
    if save_trace and not result.failure and not result.record["stalled"]:
        trace = target.with_suffix(".npz")
        np.savez_compressed(trace, states=result.states, controls=result.controls)
        entry.update(
            trajectory=str(trace.resolve()), trajectory_sha256=file_hash(trace)
        )
    write_json(target, entry)
    return entry


def evaluate_paths(
    paths,
    policy_path,
    directory,
    settings,
    *,
    algorithm="diffusion_planner",
    purpose="development",
    workers=4,
    ledger_path=None,
    save_trace=False,
):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    ledger_path = ledger_path or directory / "usage.sqlite"
    jobs = [(str(path), str(directory), save_trace) for path in paths]
    entries = []
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=initialize,
        initargs=(settings, algorithm, str(policy_path), str(ledger_path), purpose),
    ) as executor:
        for index, entry in enumerate(executor.map(execute, jobs, chunksize=1), 1):
            entries.append(entry)
            if index % 25 == 0 or index == len(jobs):
                print(
                    f"{directory.name}: {index}/{len(jobs)}, collisions {sum(e['record']['collision'] for e in entries)}",
                    flush=True,
                )
    write_json(
        directory / "summary.json",
        {
            "policy_sha256": file_hash(policy_path),
            "purpose": purpose,
            "instances": len(entries),
            "collision_count": sum(e["record"]["collision"] for e in entries),
            "offroad_count": sum(e["record"]["offroad"] for e in entries),
            "stalled_count": sum(e["record"]["stalled"] for e in entries),
            "execution_s": sum(
                e["record"]["elapsed_s"] for e in entries if not e["cache_hit"]
            ),
            "new_executions": sum(not e["cache_hit"] for e in entries),
            "cache_hits": sum(e["cache_hit"] for e in entries),
            "mean_progress_m": float(
                np.mean([e["record"]["progress_m"] for e in entries])
            ),
            "records": [entry["record"] for entry in entries],
        },
    )
    return entries
