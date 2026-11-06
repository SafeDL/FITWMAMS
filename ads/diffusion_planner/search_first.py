"""A single fresh 5000-execution search under the current frozen world source."""

import argparse
import fcntl
import json
import multiprocessing
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import psutil
import torch
import yaml

from ads.common.evaluation import (
    EpisodeResult,
    WorldEvaluator,
    annotate_dependencies,
    summarise_trace,
)
from ads.common.evaluation_sampling import named_stream
from ads.common.instances import (
    GeneratedInstance,
    GeneratedSource,
    InstanceStore,
    UsageLedger,
    file_hash,
    json_hash,
)
from ads.common.reporting import write_json
from ads.common.setup import REPO_ROOT, load_policy, load_world
from ads.common.world import FrozenWorldFactory
from ads.diffusion_planner.checkpoints import link_checkpoint
from ads.nuplan.assets import official_path
from ads.search.engine import SearchConfig, SearchEngine
from tools.evt import GPDTailModel


class ReplayableEvaluator:
    def __init__(self, evaluator, directory, policy_hash, *, executor=None):
        self.evaluator, self.directory, self.policy_hash = (
            evaluator,
            directory,
            policy_hash,
        )
        self.calls = 0
        self.new_calls, self.cache_hits = 0, 0
        self.executor = executor
        self.directory.mkdir(parents=True, exist_ok=True)

    def __call__(self, instance):
        if self.executor is not None:
            return self.many([instance])[0]
        self.calls += 1
        path = self.directory / f"{self.calls:05d}.json"
        trace = path.with_suffix(".npz")
        if path.exists():
            self.cache_hits += 1
            self.last_cache_hit = True
            metadata = json.loads(path.read_text())
            if (
                metadata["policy_sha256"] != self.policy_hash
                or metadata["record"]["instance_hash"] != instance.instance_hash
                or file_hash(trace) != metadata["trace_sha256"]
            ):
                raise ValueError(
                    "Search resume diverged from its recorded evaluation sequence"
                )
            with np.load(trace, allow_pickle=False) as data:
                names = [
                    name.removeprefix("diagnostic_")
                    for name in data.files
                    if name.startswith("diagnostic_")
                ]
                diagnostics = [
                    {name: data["diagnostic_" + name][i] for name in names}
                    for i in range(len(data["controls"]))
                ]
                return EpisodeResult(
                    instance,
                    metadata["record"],
                    data["states"],
                    data["controls"],
                    diagnostics,
                )
        self.new_calls += 1
        self.last_cache_hit = False
        result = self.evaluator(instance)
        diagnostics = {
            "diagnostic_" + name: np.stack([d[name] for d in result.diagnostics])
            for name in (result.diagnostics[0] if result.diagnostics else [])
        }
        np.savez_compressed(
            trace, states=result.states, controls=result.controls, **diagnostics
        )
        write_json(
            path,
            {
                "policy_sha256": self.policy_hash,
                "record": result.record,
                "trace_sha256": file_hash(trace),
            },
        )
        return result

    def many(self, instances):
        if self.executor is None:
            return [self(instance) for instance in instances]
        jobs = [
            (self.calls + index + 1, instance)
            for index, instance in enumerate(instances)
        ]
        self.calls += len(jobs)
        results = []
        for instance, (record, states, controls, diagnostics, cached) in zip(
            instances, self.executor.map(execute_search_case, jobs, chunksize=1)
        ):
            self.cache_hits += int(cached)
            self.new_calls += int(not cached)
            results.append(
                EpisodeResult(instance, record, states, controls, diagnostics)
            )
        return results


def initialize_search_evaluator(settings, initial, ledger_path, directory):
    global _search_replay
    torch.set_num_threads(settings["runtime"]["torch_threads"])
    evaluator = WorldEvaluator(
        load_policy("diffusion_planner", initial, device=settings["runtime"]["device"]),
        "diffusion_planner",
        FrozenWorldFactory(
            REPO_ROOT / settings["world_config"],
            device=settings["runtime"]["world_device"],
        ),
        GPDTailModel.from_dict(
            json.loads((REPO_ROOT / settings["evt_model"]).read_text())
        ),
        ledger=UsageLedger(ledger_path),
        seed=settings["master_seed"],
    )
    _search_replay = ReplayableEvaluator(evaluator, directory, file_hash(initial))


def execute_search_case(job):
    index, instance = job
    _search_replay.calls = index - 1
    result = _search_replay(instance)
    return (
        result.record,
        result.states,
        result.controls,
        result.diagnostics,
        _search_replay.last_cache_hit,
    )


def initialize_generation_stream():
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.default_stream())
    torch.cuda.set_stream(stream)


def reused_prior(settings, source, assets):
    old = REPO_ROOT / settings["historical_run"] / "search"
    protocol = json.loads((old / "protocol.json").read_text())
    if protocol["assets"] != assets:
        raise ValueError("Historical prior and current frozen world assets differ")
    directory = old.parent / "common_prior"
    replay = ReplayableEvaluator(
        None, directory / "executions", file_hash(official_path("diffusion_planner"))
    )
    factory = FrozenWorldFactory(REPO_ROOT / settings["world_config"])
    evt = GPDTailModel.from_dict(
        json.loads((REPO_ROOT / settings["evt_model"]).read_text())
    )
    results = []
    controller = None
    for index in range(settings["search"]["initial_samples"]):
        metadata_path = directory / "executions" / f"{index + 1:05d}.json"
        metadata = json.loads(metadata_path.read_text())
        instance = GeneratedInstance.load(
            directory
            / "search_instances"
            / (metadata["record"]["instance_hash"] + ".npz")
        )
        if (
            instance.assets != assets
            or instance.root_sample_id != source.prior_identity(index)[1]
        ):
            raise ValueError("Historical common-prior identity changed")
        cached = replay(instance)
        result = summarise_trace(
            instance,
            cached.states,
            cached.controls,
            cached.diagnostics,
            cached.record,
            evt_model=evt,
            reward=cached.record["task_reward"],
            elapsed_s=cached.record["elapsed_s"],
        )
        for key in ("policy_stream_id", "policy_random_mode"):
            result.record[key] = cached.record[key]
        result.record["cached_prior_source"] = {
            "metadata_path": str(metadata_path.resolve()),
            "metadata_sha256": file_hash(metadata_path),
            "trace_sha256": metadata["trace_sha256"],
        }
        if controller is None:
            controller = factory(instance.event).controller
        annotate_dependencies(result, controller)
        results.append(result)
    source.index = len(results)
    return results


def launch_search(settings, output):
    root = REPO_ROOT / settings["reuse_search"]
    root.mkdir(parents=True, exist_ok=True)
    settings_path = root / "worker_settings.json"
    if settings_path.exists():
        if json.loads(settings_path.read_text()) != settings:
            raise ValueError("Frozen search worker settings changed")
    else:
        write_json(settings_path, settings)
    source_manifest = {
        str(path.relative_to(REPO_ROOT)): file_hash(path)
        for name in (
            "hierarchical_world_model/src",
            "normalizing_flow/src",
            "diffusion/src",
            "process_highD/src",
            "HighwayEnv/highway_env",
        )
        for path in sorted((REPO_ROOT / name).rglob("*.py"))
    }
    manifest_path = root / "world_source_manifest.json"
    if (
        manifest_path.exists()
        and json.loads(manifest_path.read_text()) != source_manifest
    ):
        raise ValueError("World source differs from the frozen search")
    write_json(manifest_path, source_manifest)
    snapshot = root / "source_snapshot"
    for name in source_manifest:
        destination = snapshot / name
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((REPO_ROOT / name).read_bytes())
    for path in (REPO_ROOT / "ads").rglob("*.py"):
        destination = snapshot / path.relative_to(REPO_ROOT)
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(path.read_bytes())
    destination = snapshot / settings_path.name
    if not destination.exists():
        destination.write_bytes(settings_path.read_bytes())
    design = Path(settings.get("design_doc", "doc/search_budget_feedback.md"))
    destination = snapshot / design
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((REPO_ROOT / design).read_bytes())
    registry_path = output / "search_processes.json"
    registry = json.loads(registry_path.read_text()) if registry_path.exists() else {}
    for method in settings["methods"] + settings.get("search_controls", []):
        summary = (
            root
            / "diffusion_planner"
            / str(settings["master_seed"])
            / method
            / "round_1/search_summary.json"
        )
        if summary.exists():
            continue
        entry = registry.get(method)
        if entry and psutil.pid_exists(entry["pid"]):
            process = psutil.Process(entry["pid"])
            if (
                process.create_time() == entry["created"]
                and process.status() != psutil.STATUS_ZOMBIE
            ):
                continue
        with (output / f"search_{method}.log").open("a") as stream:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-u",
                    "-m",
                    "ads.diffusion_planner.search_first",
                    "--config",
                    str(settings_path.resolve()),
                    "--method",
                    method,
                ],
                stdout=stream,
                stderr=subprocess.STDOUT,
                cwd=REPO_ROOT,
            )
        registry[method] = {
            "pid": process.pid,
            "created": psutil.Process(process.pid).create_time(),
        }
        write_json(registry_path, registry)


def run(settings, method):
    torch.set_num_threads(settings["runtime"]["torch_threads"])
    root = REPO_ROOT / settings["reuse_search"]
    directory = (
        root / "diffusion_planner" / str(settings["master_seed"]) / method / "round_1"
    )
    if (directory / "search_summary.json").exists():
        return json.loads((directory / "search_summary.json").read_text())
    directory.mkdir(parents=True, exist_ok=True)
    sampler, _, _, assets = load_world(settings)
    manifest = json.loads((root / "world_source_manifest.json").read_text())
    if json_hash(manifest) != assets["world_implementation"] or any(
        file_hash(REPO_ROOT / name) != digest for name, digest in manifest.items()
    ):
        raise ValueError("World execution source changed after its full snapshot")
    protocol = {
        "assets": assets,
        "settings": settings,
        "torch": torch.__version__,
        "schema": "interaction_guided_search_current_world",
        "search_sources": {
            str(path.relative_to(REPO_ROOT)): file_hash(path)
            for directory in ("ads/search", "ads/common")
            for path in sorted((REPO_ROOT / directory).glob("*.py"))
        },
    }
    protocol_path = root / "protocol.json"
    # Each method has a separate worker; publish the identical protocol atomically.
    with (root / "protocol.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
            raise ValueError("Frozen search protocol changed")
        write_json(protocol_path, protocol)
    initial = directory.parent.parent / "baseline/policy.pth"
    with (root / "initial.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        initial.parent.mkdir(parents=True, exist_ok=True)
        if initial.exists():
            if file_hash(initial) != file_hash(official_path("diffusion_planner")):
                raise ValueError("Search initial checkpoint changed")
        else:
            link_checkpoint(initial, official_path("diffusion_planner"))
    ledger = UsageLedger(root / "usage.sqlite")
    store = InstanceStore(
        directory / "search_instances", purpose="search", ledger=ledger
    )
    source = GeneratedSource(
        sampler,
        assets,
        master_seed=settings["master_seed"],
        purpose="search",
        stream=named_stream(
            "formal",
            "search",
            "diffusion_planner",
            settings["master_seed"],
            "common_prior",
            0,
        ),
    )

    def record(result):
        store.add(result.instance)

    initial_results = reused_prior(settings, source, assets)
    started = time.perf_counter()
    with (
        ThreadPoolExecutor(
            max_workers=settings["runtime"]["search_workers"],
            initializer=(
                initialize_generation_stream
                if torch.device(settings["runtime"]["device"]).type == "cuda"
                else None
            ),
        ) as generation_executor,
        ProcessPoolExecutor(
            max_workers=settings["runtime"]["search_workers"],
            mp_context=multiprocessing.get_context("spawn"),
            initializer=initialize_search_evaluator,
            initargs=(
                settings,
                str(initial),
                root / "usage.sqlite",
                directory / "executions",
            ),
        ) as executor,
    ):
        evaluator = ReplayableEvaluator(
            None, directory / "executions", file_hash(initial), executor=executor
        )
        engine = SearchEngine(
            source,
            evaluator,
            config=SearchConfig(**settings["search"]),
            seed=named_stream(
                "formal", "search_rng", settings["master_seed"], method, 0
            ),
            progress=lambda done, total: print(f"{method}: {done}/{total}", flush=True),
            on_evaluation=record,
            evaluation_batch_size=settings["runtime"]["search_workers"],
            generation_executor=generation_executor,
        )
        evaluator.calls = len(initial_results)
        evaluator.cache_hits = len(initial_results)
        result = engine.run(method, initial=initial_results)
    with (directory / "search.jsonl").open("w") as stream:
        for row in result.records:
            stream.write(json.dumps(row.record, allow_nan=False) + "\n")
    result.archive.save(directory / "archive")
    report = {
        **result.summary(),
        "policy_sha256": file_hash(initial),
        "wall_elapsed_s": time.perf_counter() - started,
        "new_execution_calls": evaluator.new_calls,
        "resumed_execution_calls": evaluator.cache_hits,
        "parallel_workers": settings["runtime"]["search_workers"],
    }
    if report["evaluations"] > settings["search"]["budget"]:
        raise AssertionError("Search exceeded its execution cap")
    write_json(directory / "search_summary.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--method",
        choices=("monte_carlo", "subset", "guided"),
        required=True,
    )
    args = parser.parse_args(argv)
    run(yaml.safe_load(args.config.read_text()), args.method)


if __name__ == "__main__":
    main()
