"""Reproducible ADS search/repair campaign and sealed IID evaluation.

python -m ads.assets
python -m ads.risk_stratified_repair run
python -m ads.risk_stratified_repair evaluate
"""

from __future__ import annotations

import argparse
import fcntl
import gc
import json
import os
import time
from pathlib import Path

import torch
import yaml

from ads.common.curriculum import (
    build_repair_banks,
    load_curriculum,
    persist_curriculum,
)
from ads.common.evaluation import WorldEvaluator
from ads.common.evaluation_sampling import (
    final_sampling_spec,
    named_stream,
    source_stream,
    verify_final_roots,
    verify_sampling_spec,
)
from ads.common.instances import (
    GeneratedInstance,
    GeneratedSource,
    InstanceStore,
    UsageLedger,
    file_hash,
    generate_to_store,
    json_hash,
)
from ads.common.repair import RepairEnvironment, train_discrete
from ads.common.reporting import (
    paired_summary,
    repair_method_comparisons,
    shared_world_seed_summary,
    write_json,
)
from ads.common.setup import REPO_ROOT, load_policy, load_world
from ads.search.archive import FailureArchive
from ads.search.engine import SearchConfig, SearchEngine
from ads.tasks import (
    NATIVE_PLANNERS,
    POLICY_SUFFIX,
    SUPERVISED_ALGORITHMS,
)

ALGORITHMS = tuple(POLICY_SUFFIX)
METHODS = ("monte_carlo", "subset", "guided")


def registered_protocol(current, saved, revision):
    """Retain the sampling protocol while recording an explicit code correction."""
    if current == saved:
        return saved
    if (
        dict(current, ads_implementation=saved["ads_implementation"]) != saved
        or revision is None
    ):
        raise ValueError(
            "campaign configuration/assets changed or code revision is missing"
        )
    before, after = revision["base_files"], revision["revised_files"]
    changes = {
        name: {"before": before.get(name), "after": after.get(name)}
        for name in sorted(before.keys() | after.keys())
        if before.get(name) != after.get(name)
    }
    if (
        json_hash(before) != saved["ads_implementation"]
        or json_hash(after) != current["ads_implementation"]
        or changes != revision["changes"]
    ):
        raise ValueError("campaign code differs from the registered correction")
    return saved


class Campaign:
    def __init__(self, config_path):
        self.config = yaml.safe_load(Path(config_path).read_text())
        self.config.setdefault("run_kind", "formal")
        for algorithm in self.config["algorithms"]:
            if algorithm in NATIVE_PLANNERS:
                from ads.nuplan.assets import official_path

                checkpoint = official_path(algorithm)
                if not checkpoint.exists():
                    raise FileNotFoundError(
                        f"Install official {algorithm} weights before creating a campaign: {checkpoint}"
                    )
                provenance = checkpoint.parent / "official.json"
                expected_hash = json.loads(provenance.read_text())["checkpoint_sha256"]
                if file_hash(checkpoint) != expected_hash:
                    raise ValueError(
                        f"official {algorithm} checkpoint differs from its installation manifest"
                    )
        self.output = REPO_ROOT / self.config["output_dir"]
        self.output.mkdir(parents=True, exist_ok=True)
        if not set(self.config["algorithms"]).issubset(ALGORITHMS) or not set(
            self.config["methods"]
        ).issubset(METHODS):
            raise ValueError("invalid experiment algorithm or search method")
        if self.config["rounds"] != 2:
            raise ValueError("the registered protocol uses exactly two repair rounds")
        torch.set_num_threads(self.config["runtime"]["torch_threads"])
        self.sampler, self.factory, self.evt, self.assets = load_world(self.config)
        self.ledger = UsageLedger(self.output / "usage.sqlite")
        self.device = self.config["runtime"]["device"]
        protocol = {
            "settings": self.config,
            "assets": self.assets,
            "torch": torch.__version__,
            "schema": "method_c_branching_v2",
            "ads_implementation": json_hash(
                {
                    str(path.relative_to(REPO_ROOT)): file_hash(path)
                    for path in sorted((REPO_ROOT / "ads").rglob("*.py"))
                }
            ),
        }
        path = self.output / "protocol.json"
        with (self.output / "protocol.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if path.exists():
                revision_path = self.output / "implementation_revision.json"
                revision = (
                    json.loads(revision_path.read_text())
                    if revision_path.exists()
                    else None
                )
                protocol = registered_protocol(
                    protocol, json.loads(path.read_text()), revision
                )
            else:
                write_json(path, protocol)
        self.protocol_hash = json_hash(protocol)

    def stream(self, *parts):
        return named_stream(*parts)

    def source(self, purpose, *parts):
        return GeneratedSource(
            self.sampler,
            self.assets,
            master_seed=self.config["master_seed"],
            purpose=purpose,
            stream=source_stream(self.config, self.protocol_hash, purpose, *parts),
        )

    def pool(self, purpose, name, n, *stream):
        directory = self.output / "instances" / name
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "pool.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            return self._load_or_generate_pool(purpose, name, n, *stream)

    def _load_or_generate_pool(self, purpose, name, n, *stream):
        store = InstanceStore(
            self.output / "instances" / name, purpose=purpose, ledger=self.ledger
        )
        if len(store) > n:
            raise ValueError("stored cohort exceeds the frozen configured size")
        if len(store) < n:
            needed = n - len(store)
            source = self.source(purpose, *stream)
            source.index = len(store)
            generate_to_store(
                source,
                store,
                needed,
                self.config["runtime"]["generation_batch_size"],
            )
            with (self.output / "generation.jsonl").open("a") as stream_file:
                stream_file.write(
                    json.dumps(
                        {
                            "purpose": purpose,
                            "pool": name,
                            "new_instances": needed,
                            "generation_elapsed_s": source.generation_elapsed_s,
                        }
                    )
                    + "\n"
                )
        store.paths.sort()
        return list(store)

    def checkpoint(self, directory, algorithm):
        return Path(directory) / ("policy" + POLICY_SUFFIX[algorithm])

    def save_policy(self, model, algorithm, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = self.checkpoint(directory, algorithm)
        model.save(path)
        if algorithm == "dqn":
            model.save_replay_buffer(directory / "replay_buffer.pkl")
        return path

    def evaluate_instances(self, model, algorithm, instances, purpose):
        evaluate = WorldEvaluator(
            model,
            algorithm,
            self.factory,
            self.evt,
            ledger=self.ledger,
            purpose=purpose,
        )
        return [evaluate(instance) for instance in instances]

    def teacher_samples(self, model, entries, *, cache_name):
        from ads.common.demonstrations import DemonstrationCache

        cache = DemonstrationCache(
            self.output / "demonstrations" / cache_name,
            self.factory,
            algorithm=model.algorithm,
            cfg=getattr(model, "cfg", None),
            candidate_budget=self.config["teacher"]["candidate_budget"],
        )
        samples, reports = [], []
        for index, (instance, domain) in enumerate(entries):
            self.ledger.register(instance, "repair")
            produced, report = cache.get(instance)
            for sample in produced:
                sample["domain"] = domain
                sample["scene_id"] = instance.root_sample_id
            samples.extend(produced)
            reports.append(report)
            print(
                f"  teacher {index + 1}/{len(entries)}: {report.get('samples', 0)} labels, {report['new_candidates']} new rollouts",
                flush=True,
            )
        return samples, reports

    def train(
        self,
        algorithm,
        directory,
        banks,
        *,
        base,
        seed,
        round_index,
        replay=None,
        teacher_history=None,
    ):
        directory = Path(directory)
        started = time.perf_counter()
        settings = self.config["repair"]
        report = {
            "algorithm": algorithm,
            "seed": seed,
            "round": round_index,
            "baseline": False,
            "base_sha256": file_hash(base),
            "protocol_hash": self.protocol_hash,
            "run_kind": self.config["run_kind"],
            "curriculum_sha256": persist_curriculum(
                directory, banks, self.ledger, teacher_history=teacher_history or []
            ),
        }
        env = None
        training_seed = named_stream("training", algorithm, seed, round_index)

        def progress(done, total):
            value = {
                "algorithm": algorithm,
                "seed": seed,
                "round": round_index,
                "baseline": False,
                "completed": done,
                "target": total,
                "unit": "gradient_updates"
                if algorithm in SUPERVISED_ALGORITHMS
                else "decisions",
                "elapsed_s": time.perf_counter() - started,
            }
            write_json(directory / "progress.json", value)
            print(f"  {algorithm} training {done}/{total} {value['unit']}", flush=True)

        if algorithm in ("dqn", "ppo"):
            env = RepairEnvironment(
                algorithm,
                banks,
                self.factory,
                round_index=round_index,
                seed=training_seed,
                ledger=self.ledger,
            )
            model, learning = train_discrete(
                algorithm,
                env,
                steps=settings["rl_steps"],
                seed=training_seed,
                device=self.device,
                base=base,
                replay=replay,
                progress=progress,
            )
        elif algorithm == "carl":
            from ads.carl.experiment import repair_and_select

            model = load_policy(algorithm, base, device=self.device)
            env = RepairEnvironment(
                algorithm,
                banks,
                self.factory,
                round_index=round_index,
                seed=training_seed,
                ledger=self.ledger,
            )
            model, learning = repair_and_select(
                self,
                model,
                env,
                base=base,
                directory=directory,
                steps=settings["rl_steps"],
                seed=training_seed,
                progress=progress,
            )
        else:
            from ads.common.expert import select_teacher_instances

            if algorithm in NATIVE_PLANNERS:
                from ads.nuplan.train import train
            else:
                from ads.plant2.train import train

            model = load_policy(algorithm, base, device=self.device)
            limit = settings["plant_normal_teacher_instances"]
            limit_new = self.config["repair"]["plant_new_teacher_instances"]
            # Entries already arrive in domain/ancestor-balanced order.
            selected = select_teacher_instances(
                banks.get("new_failure", []),
                banks.get("success_neighbours", {}),
                limit_new,
            )
            selected_manifest = []
            for instance, domain in selected:
                saved = directory / "teacher_inputs" / (instance.instance_hash + ".npz")
                if not saved.exists():
                    instance.save(saved)
                selected_manifest.append(
                    {"instance_hash": instance.instance_hash, "domain": domain}
                )
            write_json(directory / "teacher_selection.json", selected_manifest)
            if (
                algorithm == "diffusion_planner"
                and "capability_preserving_repair" in self.config
            ):
                from ads.diffusion_planner.experiment import repair_and_select
                from ads.diffusion_planner.supervision import prepare_materials

                samples, reports = prepare_materials(
                    self,
                    model,
                    base,
                    directory,
                    banks,
                    selected,
                    teacher_history or [],
                    round_index,
                )

                model, learning = repair_and_select(
                    self,
                    model,
                    samples,
                    base=base,
                    directory=directory,
                    updates=settings["plant_updates"],
                    seed=training_seed,
                    round_index=round_index,
                    progress=progress,
                )
            else:
                normal, normal_reports = self.teacher_samples(
                    model, banks["normal"][:limit], cache_name="normal"
                )
                new, new_reports = self.teacher_samples(
                    model, selected, cache_name="failures"
                )
                history, history_reports = self.teacher_samples(
                    model, teacher_history or [], cache_name="failures"
                )
                samples = {"normal": normal, "new_failure": new, "history": history}
                reports = normal_reports + new_reports + history_reports
                learning = train(
                    model,
                    samples,
                    updates=settings["plant_updates"],
                    seed=training_seed,
                    round_index=round_index,
                    batch_size=self.config["repair"]["plant_batch_size"],
                    micro_batch_size=self.config["repair"]
                    .get("micro_batch_sizes", {})
                    .get(algorithm, self.config["repair"]["plant_micro_batch_size"]),
                    learning_rate=self.config["repair"]["plant_learning_rate"],
                    progress=progress,
                )
            report["teacher"] = {
                "instances": len(reports),
                "new_candidate_executions": sum(r["new_candidates"] for r in reports),
                "new_elapsed_s": sum(r["new_elapsed_s"] for r in reports),
                "new_reference_executions": sum(
                    r.get("reference_new_executions", 0) for r in reports
                ),
                "reference_execution_s": sum(
                    r.get("reference_new_elapsed_s", 0.0) for r in reports
                ),
                "qualified_demonstrations": sum(
                    r["qualified_demonstration"] for r in reports
                ),
                "reports": reports,
            }
        path = self.save_policy(model, algorithm, directory)
        report.update(
            learning=learning,
            policy_sha256=file_hash(path),
            elapsed_s=time.perf_counter() - started,
        )
        write_json(directory / "training.json", report)
        del model, env
        gc.collect()
        torch.cuda.empty_cache()
        return path

    def baseline(self, algorithm, seed):
        """Import the agreed pi0; this protocol never performs initial training."""
        from ads.nuplan.assets import official_path

        directory = self.output / algorithm / str(seed) / "baseline"
        directory.mkdir(parents=True, exist_ok=True)
        path = self.checkpoint(directory, algorithm)
        report_path = directory / "training.json"
        if report_path.exists():
            if file_hash(path) != json.loads(report_path.read_text())["policy_sha256"]:
                raise ValueError("initial policy checkpoint changed")
            return path
        if algorithm in ("dqn", "ppo", "plant2"):
            source = (
                REPO_ROOT
                / f"results/risk_stratified_repair_history/formal/{algorithm}/{seed}/baseline/policy{POLICY_SUFFIX[algorithm]}"
            )
        elif algorithm == "carl":
            source = REPO_ROOT / "ads/carl/artifacts/official.pth"
        else:
            source = official_path(algorithm)
        model = load_policy(algorithm, source, device=self.device)
        model.save(path)
        curriculum = persist_curriculum(directory, {}, self.ledger)
        write_json(
            report_path,
            {
                "algorithm": algorithm,
                "seed": seed,
                "baseline": True,
                "protocol_hash": self.protocol_hash,
                "run_kind": self.config["run_kind"],
                "policy_sha256": file_hash(path),
                "source_path": str(source),
                "base_sha256": file_hash(source),
                "curriculum_sha256": curriculum,
                "round": 0,
                "elapsed_s": 0.0,
                "learning": {
                    "actual_steps": 0,
                    "gradient_updates": 0,
                    "elapsed_s": 0.0,
                    "initialization": "imported_initial_policy_without_training",
                },
            },
        )
        del model
        return path

    def load_saved_search(self, algorithm, seed, method, round_number):
        """Validate the policy, execution ledger and persisted repair curriculum."""
        directory = (
            self.output / algorithm / str(seed) / method / f"round_{round_number}"
        )
        if (directory / "training.json").exists():
            raise ValueError("saved search already has a completed training report")
        previous = (
            self.output / algorithm / str(seed) / "baseline"
            if round_number == 1
            else directory.parent / f"round_{round_number - 1}"
        )
        base = self.checkpoint(previous, algorithm)
        base_hash = file_hash(base)
        metadata = json.loads((previous / "training.json").read_text())
        summary = json.loads((directory / "search_summary.json").read_text())
        if base_hash != metadata["policy_sha256"]:
            raise ValueError("saved-search base policy changed")
        if "policy_sha256" in summary:
            if summary["policy_sha256"] != base_hash:
                raise ValueError("completed search used another policy")
        elif round_number != 1:
            raise ValueError(
                "later-round training requires recorded search policy provenance"
            )
        records = (directory / "search.jsonl").read_text().splitlines()
        if (
            len(records) != summary["evaluations"]
            or not 0 < len(records) <= self.config["search"]["budget"]
        ):
            raise ValueError("saved search is incomplete or over budget")
        FailureArchive.load(directory / "archive")
        banks, history = load_curriculum(directory, self.ledger)
        entries = [
            entry
            for name in ("normal", "new_failure", "history")
            for entry in banks[name]
        ]
        entries += list(banks["success_neighbours"].values()) + history
        if any(instance.assets != self.assets for instance, _ in entries):
            raise ValueError("saved curriculum uses another frozen world")
        return directory, previous, base, banks, history

    def train_saved_search(self, algorithm, seed, method, round_number):
        """Continue optimization without repeating the completed scene search."""
        directory, previous, base, banks, history = self.load_saved_search(
            algorithm, seed, method, round_number
        )
        print(
            f"Reusing completed search; training {algorithm} from its saved curriculum",
            flush=True,
        )
        return self.train(
            algorithm,
            directory,
            banks,
            base=base,
            replay=previous / "replay_buffer.pkl"
            if algorithm == "dqn" and round_number > 1
            else None,
            seed=seed,
            round_index=round_number - 1,
            teacher_history=history,
        )

    def run_branch(
        self, algorithm, seed, method, initial, *, through_round=None, search_only=False
    ):
        branch = self.output / algorithm / str(seed) / method
        checkpoint, replay, archive = initial, None, None
        teacher_history = []
        rounds = self.config["rounds"] if through_round is None else through_round
        if not 1 <= rounds <= self.config["rounds"]:
            raise ValueError("invalid requested repair round")
        if search_only and through_round is None:
            raise ValueError("search preparation requires an explicit repair round")
        for round_index in range(rounds):
            directory = branch / f"round_{round_index + 1}"
            final_path = self.checkpoint(directory, algorithm)
            report_path = directory / "training.json"
            if search_only and round_index + 1 != rounds and not report_path.exists():
                raise ValueError("train preceding rounds before preparing this search")
            if not report_path.exists() and all(
                (directory / name).exists()
                for name in (
                    "search_summary.json",
                    "inputs/curriculum.json",
                    "archive/archive.json",
                )
            ):
                if search_only:
                    self.load_saved_search(algorithm, seed, method, round_index + 1)
                    return directory / "search_summary.json"
                self.train_saved_search(algorithm, seed, method, round_index + 1)
            if report_path.exists():
                if (
                    file_hash(final_path)
                    != json.loads(report_path.read_text())["policy_sha256"]
                ):
                    raise ValueError("repair checkpoint changed")
                checkpoint = final_path
                replay = directory / "replay_buffer.pkl" if algorithm == "dqn" else None
                archive = FailureArchive.load(directory / "archive")
                if algorithm in SUPERVISED_ALGORITHMS:
                    teacher_history += [
                        (
                            GeneratedInstance.load(
                                directory
                                / "teacher_inputs"
                                / (row["instance_hash"] + ".npz")
                            ),
                            row["domain"],
                        )
                        for row in json.loads(
                            (directory / "teacher_selection.json").read_text()
                        )
                    ]
                continue
            directory.mkdir(parents=True, exist_ok=True)
            old_endpoints = archive.endpoints() if archive else []
            model = load_policy(algorithm, checkpoint, device=self.device)
            evaluator = WorldEvaluator(
                model, algorithm, self.factory, self.evt, ledger=self.ledger
            )
            print(
                f"Searching {algorithm}/{seed}/{method}, round {round_index + 1}",
                flush=True,
            )
            search_store = InstanceStore(
                directory / "search_instances", purpose="search", ledger=self.ledger
            )
            attempts = directory / "search_attempts"
            attempts.mkdir(exist_ok=True)
            journal = attempts / f"{time.time_ns()}.jsonl"

            def record_execution(item, store=search_store, path=journal):
                store.add(item.instance)
                with path.open("a") as stream:
                    stream.write(json.dumps(item.record, allow_nan=False) + "\n")

            engine = SearchEngine(
                self.source("search", algorithm, seed, "common_prior", round_index),
                evaluator,
                config=SearchConfig(**self.config["search"]),
                seed=self.stream("search_rng", seed, method, round_index),
                progress=lambda done, total: print(
                    f"  executed {done}/{total}", flush=True
                ),
                on_evaluation=record_execution,
            )
            result = engine.run(
                method,
                archive=archive,
            )
            archive = result.archive
            persistence_started = time.perf_counter()
            with (directory / "search.jsonl").open("w") as stream:
                for item in result.records:
                    stream.write(json.dumps(item.record, allow_nan=False) + "\n")
            summary = result.summary()
            summary["policy_sha256"] = file_hash(checkpoint)
            summary["execution_journal"] = str(journal.relative_to(directory))
            summary["persistence_elapsed_s"] = time.perf_counter() - persistence_started
            summary["implementation_sha256"] = {
                name: file_hash(REPO_ROOT / name)
                for name in (
                    "ads/search/engine.py",
                    "ads/search/archive.py",
                    "ads/common/evaluation.py",
                )
            }
            if method == "guided":
                summary["domain_sampling_rule"] = "weighted_fair_branch_allocation"
                summary["response_mutation_scope"] = (
                    "frozen_previous_level_vehicle_time_union"
                )
            write_json(directory / "search_summary.json", summary)
            archive.save(directory / "archive")
            normal = self.pool(
                "repair",
                f"normal_{seed}_{round_index}",
                self.config["repair"]["normal_instances"],
                "normal",
                seed,
                round_index,
            )
            banks = build_repair_banks(
                algorithm,
                normal,
                result.records,
                old_endpoints,
                engine.known,
                limit=self.config["repair"]["plant_new_teacher_instances"],
            )
            del model, evaluator, result, engine
            gc.collect()
            torch.cuda.empty_cache()
            if search_only:
                persist_curriculum(
                    directory, banks, self.ledger, teacher_history=teacher_history
                )
                return directory / "search_summary.json"
            checkpoint = self.train(
                algorithm,
                directory,
                banks,
                base=checkpoint,
                replay=replay,
                seed=seed,
                round_index=round_index,
                teacher_history=teacher_history,
            )
            if algorithm in SUPERVISED_ALGORITHMS:
                teacher_history += [
                    (
                        GeneratedInstance.load(
                            directory
                            / "teacher_inputs"
                            / (row["instance_hash"] + ".npz")
                        ),
                        row["domain"],
                    )
                    for row in json.loads(
                        (directory / "teacher_selection.json").read_text()
                    )
                ]
            replay = directory / "replay_buffer.pkl" if algorithm == "dqn" else None
        return checkpoint

    def verify_runtime(self):
        runtime_report = json.loads(
            (
                REPO_ROOT / "results/risk_stratified_repair_history/runtime_check.json"
            ).read_text()
        )
        if (
            runtime_report["status"] != "passed"
            or runtime_report["assets"] != self.assets
            or runtime_report["target_torch"] != torch.__version__
        ):
            raise ValueError(
                "pass the current cross-runtime world replay check before training"
            )

    def run(self):
        self.verify_runtime()
        if (self.output / "frozen_policies.json").exists():
            self.verify_freeze()
            print("All registered final policies are already frozen.", flush=True)
            return
        policies, repair_checkpoints = {}, {}
        for algorithm in self.config["algorithms"]:
            for seed in self.config["seeds"]:
                initial = self.baseline(algorithm, seed)
                policies[f"{algorithm}/{seed}/initial"] = {
                    "path": str(initial.relative_to(REPO_ROOT)),
                    "sha256": file_hash(initial),
                }
                for method in self.config["methods"]:
                    final = self.run_branch(algorithm, seed, method, initial)
                    policies[f"{algorithm}/{seed}/{method}"] = {
                        "path": str(final.relative_to(REPO_ROOT)),
                        "sha256": file_hash(final),
                    }
                    for round_index in range(1, self.config["rounds"] + 1):
                        directory = (
                            self.output
                            / algorithm
                            / str(seed)
                            / method
                            / f"round_{round_index}"
                        )
                        checkpoint = self.checkpoint(directory, algorithm)
                        repair_checkpoints[
                            f"{algorithm}/{seed}/{method}/round_{round_index}"
                        ] = {
                            "path": str(checkpoint.relative_to(REPO_ROOT)),
                            "sha256": file_hash(checkpoint),
                        }
        from ads.search.comparison import common_discovery_partition

        for algorithm in self.config["algorithms"]:
            for seed in self.config["seeds"]:
                common_discovery_partition(self, algorithm, seed)
        from ads.common.cohorts import prepare_evaluation_assets

        evaluation_assets = prepare_evaluation_assets(self)
        write_json(
            self.output / "frozen_policies.json",
            {
                "evaluation_assets": evaluation_assets,
                "protocol_hash": self.protocol_hash,
                "implementation_revision_sha256": (
                    file_hash(self.output / "implementation_revision.json")
                    if (self.output / "implementation_revision.json").exists()
                    else None
                ),
                "policies": policies,
                "repair_checkpoints": repair_checkpoints,
                "run_kind": self.config["run_kind"],
                "evaluation_sampling": final_sampling_spec(
                    json.loads((self.output / "protocol.json").read_text()),
                    evaluation_assets,
                ),
            },
        )

    def verify_freeze(self):
        path = self.output / "frozen_policies.json"
        if not path.exists():
            raise ValueError(
                "finish and freeze every registered policy before creating final-test instances"
            )
        payload = json.loads(path.read_text())
        if payload["protocol_hash"] != self.protocol_hash:
            raise ValueError("evaluation protocol differs from the frozen campaign")
        revision = self.output / "implementation_revision.json"
        if payload.get("implementation_revision_sha256") != (
            file_hash(revision) if revision.exists() else None
        ):
            raise ValueError("implementation correction changed after policy freeze")
        expected = {
            f"{algorithm}/{seed}/{arm}"
            for algorithm in self.config["algorithms"]
            for seed in self.config["seeds"]
            for arm in ("initial", *self.config["methods"])
        }
        if set(payload["policies"]) != expected:
            raise ValueError(
                "freeze does not contain the complete registered policy set"
            )
        if self.config["run_kind"] == "formal":
            expected_repairs = {
                f"{algorithm}/{seed}/{method}/round_{round_index}"
                for algorithm in self.config["algorithms"]
                for seed in self.config["seeds"]
                for method in self.config["methods"]
                for round_index in range(1, self.config["rounds"] + 1)
            }
            if set(payload.get("repair_checkpoints", {})) != expected_repairs:
                raise ValueError(
                    "freeze does not contain every registered repair checkpoint"
                )
        verify_sampling_spec(
            payload, json.loads((self.output / "protocol.json").read_text())
        )
        for entry in list(payload["policies"].values()) + list(
            payload.get("repair_checkpoints", {}).values()
        ):
            if file_hash(REPO_ROOT / entry["path"]) != entry["sha256"]:
                raise ValueError("a policy changed after final-test freeze")
        from ads.common.cohorts import verify_evaluation_assets

        verify_evaluation_assets(payload["evaluation_assets"])
        return payload

    def final_cohorts(self, frozen=None):
        from ads.common.cohorts import final_cohort
        from ads.common.evaluation_sampling import COHORTS

        frozen = self.verify_freeze() if frozen is None else frozen
        return {cohort: final_cohort(self, cohort, frozen) for cohort in COHORTS}

    def evaluate(self):
        frozen = self.verify_freeze()
        cohorts = self.final_cohorts(frozen)
        all_rows = {
            key: self.evaluate_one(key, frozen=frozen, instances=cohorts)
            for key in frozen["policies"]
        }
        summary = {
            "run_kind": self.config["run_kind"],
            "external_evaluation_data": False,
            "cohorts": {},
        }
        replicates = self.config["evaluation"]["bootstrap_replicates"]
        for cohort, instances in cohorts.items():
            rows = {key: value[cohort] for key, value in all_rows.items()}
            result = {
                "instances": len(instances),
                "per_seed": {},
                "shared_world_aggregate": {},
            }
            for algorithm in self.config["algorithms"]:
                for method in self.config["methods"]:
                    pairs = []
                    for seed in self.config["seeds"]:
                        before, after = (
                            rows[f"{algorithm}/{seed}/initial"],
                            rows[f"{algorithm}/{seed}/{method}"],
                        )
                        result["per_seed"][f"{algorithm}/{seed}/{method}"] = (
                            paired_summary(
                                before, after, bootstrap_replicates=replicates
                            )
                        )
                        pairs.append((before, after))
                    result["shared_world_aggregate"][f"{algorithm}/{method}"] = (
                        shared_world_seed_summary(
                            pairs, bootstrap_replicates=replicates
                        )
                    )
            result["between_repair_methods"] = repair_method_comparisons(
                rows,
                self.config["algorithms"],
                self.config["seeds"],
                self.config["methods"],
                bootstrap_replicates=replicates,
            )
            summary["cohorts"][cohort] = result
        summary["archive_diagnostics"] = self.archive_diagnostics(frozen)
        write_json(self.output / "evaluation" / "summary.json", summary)
        from ads.report import export_campaign

        export_campaign(self.output, self.config)

    def evaluate_one(self, key, *, frozen=None, instances=None):
        frozen = self.verify_freeze() if frozen is None else frozen
        instances = self.final_cohorts(frozen) if instances is None else instances
        entry = frozen["policies"][key]
        algorithm, seed, arm = key.split("/")
        model, rows = None, {}
        for cohort, worlds in instances.items():
            protocol = json.loads((self.output / "protocol.json").read_text())
            verify_final_roots(worlds, protocol, cohort, frozen["evaluation_assets"])
            path = (
                self.output / "evaluation" / cohort / algorithm / seed / (arm + ".json")
            )
            if path.exists():
                saved = json.loads(path.read_text())
                if (
                    saved["policy_sha256"] != entry["sha256"]
                    or saved["protocol_hash"] != self.protocol_hash
                ):
                    raise ValueError("evaluation policy/protocol hash mismatch")
                expected = {instance.instance_hash for instance in worlds}
                if {
                    row["instance_hash"] for row in saved["records"]
                } != expected or len(saved["records"]) != len(expected):
                    raise ValueError("cached evaluation differs from sealed instances")
                rows[cohort] = saved["records"]
                continue
            if model is None:
                model = load_policy(
                    algorithm, REPO_ROOT / entry["path"], device=self.device
                )
            evaluate = WorldEvaluator(
                model,
                algorithm,
                self.factory,
                self.evt,
                ledger=self.ledger,
                purpose="final_test",
                seed=int(seed),
            )
            records = []
            attempts = path.parent / (arm + ".attempts")
            attempts.mkdir(parents=True, exist_ok=True)
            journal = attempts / f"{time.time_ns()}.jsonl"
            for index, instance in enumerate(worlds, 1):
                record = evaluate(instance).record
                with journal.open("a") as stream:
                    stream.write(
                        json.dumps(
                            {
                                "policy_sha256": entry["sha256"],
                                "protocol_hash": self.protocol_hash,
                                "record": record,
                            }
                        )
                        + "\n"
                    )
                records.append(record)
                if index % 32 == 0 or index == len(worlds):
                    write_json(
                        path.with_suffix(".progress.json"),
                        {
                            "completed": index,
                            "target": len(worlds),
                            "policy_sha256": entry["sha256"],
                        },
                    )
                    print(f"{cohort} {key}: {index}/{len(worlds)}", flush=True)
            write_json(
                path,
                {
                    "policy_sha256": entry["sha256"],
                    "protocol_hash": self.protocol_hash,
                    "execution_journal": str(journal.relative_to(self.output)),
                    "records": records,
                },
            )
            rows[cohort] = records
        del model
        gc.collect()
        torch.cuda.empty_cache()
        return rows

    def archive_diagnostics(self, frozen, *, keys=None):
        from ads.common.archive_diagnostics import (
            archive_role_outcomes,
            collect_archive_records,
        )

        summaries = {}
        for algorithm in self.config["algorithms"]:
            for seed in self.config["seeds"]:
                for method in self.config["methods"]:
                    if keys is not None and f"{algorithm}/{seed}/{method}" not in keys:
                        continue
                    branch = self.output / algorithm / str(seed) / method
                    output = self.output / "diagnostic" / algorithm / str(seed) / method
                    checkpoints = [frozen["policies"][f"{algorithm}/{seed}/initial"]]
                    journals = []
                    for round_index in range(1, self.config["rounds"] + 1):
                        directory = branch / f"round_{round_index}"
                        training = json.loads((directory / "training.json").read_text())
                        checkpoint = self.checkpoint(directory, algorithm)
                        entry = {
                            "path": str(checkpoint.relative_to(REPO_ROOT)),
                            "sha256": training["policy_sha256"],
                        }
                        frozen_entry = frozen.get("repair_checkpoints", {}).get(
                            f"{algorithm}/{seed}/{method}/round_{round_index}"
                        )
                        if frozen_entry is not None and frozen_entry != entry:
                            raise ValueError("intermediate policy differs from freeze")
                        if file_hash(checkpoint) != entry["sha256"]:
                            raise ValueError(
                                "intermediate policy differs from training evidence"
                            )
                        checkpoints.append(entry)
                        journals.append(
                            (directory / "search.jsonl", training["base_sha256"])
                        )
                    rounds, identities = {}, set()
                    for round_index in range(1, self.config["rounds"] + 1):
                        archive = FailureArchive.load(
                            branch / f"round_{round_index}" / "archive"
                        )
                        instances = [result.instance for result in archive.endpoints()]
                        identities.update(
                            instance.instance_hash for instance in instances
                        )
                        evaluated, costs = [], {}
                        for arm, entry in zip(
                            ("before", "after"),
                            checkpoints[round_index - 1 : round_index + 1],
                        ):

                            def execute(missing, entry=entry, algorithm=algorithm):
                                model = load_policy(
                                    algorithm,
                                    REPO_ROOT / entry["path"],
                                    device=self.device,
                                )
                                records = [
                                    result.record
                                    for result in self.evaluate_instances(
                                        model, algorithm, missing, "search"
                                    )
                                ]
                                del model
                                gc.collect()
                                torch.cuda.empty_cache()
                                return records

                            payload = collect_archive_records(
                                instances,
                                output / f"round_{round_index}" / (arm + ".json"),
                                policy_sha256=entry["sha256"],
                                protocol_hash=self.protocol_hash,
                                journals=journals,
                                caches=[
                                    output / "initial.json",
                                    output / (method + ".json"),
                                ]
                                + sorted(output.glob("round_*/*.json")),
                                evaluate=execute,
                            )
                            evaluated.append(payload["records"])
                            costs[arm] = payload["cost"]
                        comparison = (
                            paired_summary(
                                *evaluated,
                                diagnostic=True,
                                bootstrap_replicates=self.config["evaluation"][
                                    "bootstrap_replicates"
                                ],
                            )
                            if instances
                            else {"cohort_kind": "search_archive_diagnostic"}
                        )
                        rounds[f"round_{round_index}"] = {
                            **comparison,
                            "instances": len(instances),
                            "policy_before_sha256": checkpoints[round_index - 1][
                                "sha256"
                            ],
                            "policy_after_sha256": checkpoints[round_index]["sha256"],
                            "domain_outcomes": archive_role_outcomes(
                                archive, *evaluated
                            ),
                            "execution_costs": costs,
                        }
                    summaries[f"{algorithm}/{seed}/{method}"] = {
                        "schema": "ads_archive_diagnostics_v2",
                        "cohort_kind": "search_archive_diagnostic",
                        "instances": len(identities),
                        "rounds": rounds,
                    }
        return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("run", "evaluate", "all"))
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "ads/configs/risk_stratified_repair.yaml",
    )
    args = parser.parse_args()
    settings = yaml.safe_load(args.config.read_text())
    output = REPO_ROOT / settings["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    with (output / "campaign.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = {"stage": args.stage, "pid": os.getpid(), "status": "running"}
        write_json(output / "status.json", status)
        try:
            campaign = Campaign(args.config)
            if args.stage in ("run", "all"):
                campaign.run()
            if args.stage in ("evaluate", "all"):
                campaign.evaluate()
        except Exception as error:
            from ads.common.diagnostics import capture_failure

            try:
                diagnostic = capture_failure(error, output)
            except Exception as diagnostic_error:  # noqa: BLE001 -- preserve the original experiment error
                diagnostic = f"capture failed: {diagnostic_error}"
            write_json(
                output / "status.json",
                {
                    **status,
                    "status": "failed",
                    "error": str(error),
                    "diagnostic": diagnostic,
                },
            )
            raise
        write_json(output / "status.json", {**status, "status": "complete"})


if __name__ == "__main__":
    main()
