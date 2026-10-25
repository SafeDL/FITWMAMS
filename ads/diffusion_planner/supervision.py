"""Preserve successful driving; request corrective experts for failures."""

import fcntl
import json
import time
from pathlib import Path

import numpy as np

from ads.common.demonstrations import DemonstrationCache
from ads.common.evaluation import WorldEvaluator
from ads.common.instances import file_hash
from ads.common.randomness import PolicyRandomContext
from ads.common.reporting import write_json
from ads.nuplan.teacher import demonstration


def samples_from_trace(instance, path, expected_hash):
    path = Path(path)
    if file_hash(path) != expected_hash:
        raise ValueError("Successful supervision trajectory changed")
    with np.load(path, allow_pickle=False) as data:
        return demonstration(instance, data["states"], "diffusion_planner")


def reference_execution(campaign, model, instance, policy_hash):
    context = PolicyRandomContext(
        "diffusion_planner",
        campaign.config["master_seed"],
        "search",
        instance.instance_hash,
    )
    directory = campaign.output / "repair_references" / policy_hash / context.identity
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (instance.instance_hash + ".json")
    with path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            entry = json.loads(path.read_text())
            if (
                entry["policy_sha256"] != policy_hash
                or entry["record"]["instance_hash"] != instance.instance_hash
                or entry["record"]["policy_stream_id"] != context.identity
            ):
                raise ValueError(
                    "Reference supervision belongs to another policy/instance"
                )
            return entry, 0, 0.0, path
        evaluate = WorldEvaluator(
            model,
            "diffusion_planner",
            campaign.factory,
            campaign.evt,
            ledger=campaign.ledger,
            purpose="repair",
            seed=campaign.config["master_seed"],
        )
        started = time.perf_counter()
        result = evaluate(instance)
        elapsed = time.perf_counter() - started
        entry = {"policy_sha256": policy_hash, "record": result.record}
        if not result.failure and not result.record["stalled"]:
            trace = path.with_suffix(".npz")
            np.savez_compressed(trace, states=result.states, controls=result.controls)
            entry.update(
                trajectory=str(trace.resolve()), trajectory_sha256=file_hash(trace)
            )
        write_json(path, entry)
        return entry, 1, elapsed, path


def prepare_materials(
    campaign, model, base, directory, banks, selected, teacher_history, round_index
):
    directory = Path(directory)
    policy_hash = file_hash(base)
    summary = json.loads((directory / "search_summary.json").read_text())
    if summary["policy_sha256"] != policy_hash:
        raise ValueError("Corrective materials require the policy used for search")
    known = {
        row["instance_hash"]: row
        for row in map(
            json.loads, (directory / "search.jsonl").read_text().splitlines()
        )
        if row["execution_kind"] != "branch_check"
    }
    history_sources = {}
    if teacher_history:
        previous = directory.parent / f"round_{round_index}" / "training.json"
        history_sources = {
            row["instance_hash"]: row
            for row in json.loads(previous.read_text())["teacher"]["reports"]
        }
    limit = campaign.config["repair"]["plant_normal_teacher_instances"]
    categories = {
        "normal": banks["normal"][:limit],
        "new_failure": selected,
        "history": teacher_history,
    }
    caches = {
        name: DemonstrationCache(
            campaign.output / "demonstrations" / name,
            campaign.factory,
            algorithm="diffusion_planner",
            candidate_budget=campaign.config["teacher"]["candidate_budget"],
        )
        for name in ("normal", "failures")
    }
    material, reports = {}, []
    for category, entries in categories.items():
        material[category] = []
        for index, (instance, domain) in enumerate(entries):
            if instance.purpose in ("development", "final_test"):
                raise ValueError("Evaluation instances cannot provide repair labels")
            campaign.ledger.register(instance, "repair")
            if category == "history":
                previous = history_sources[instance.instance_hash]
                report = {
                    **previous,
                    "new_candidates": 0,
                    "new_elapsed_s": 0.0,
                    "reference_new_executions": 0,
                    "reference_new_elapsed_s": 0.0,
                    "historical_reuse": True,
                }
                produced = (
                    samples_from_trace(
                        instance, report["trajectory_path"], report["trajectory_sha256"]
                    )
                    if report["qualified_demonstration"]
                    else []
                )
            else:
                record = known.get(instance.instance_hash)
                failed = record is not None and (
                    record["safety_failure"] or record["stalled"]
                )
                new_references, reference_seconds = 0, 0.0
                reference = None
                if not failed:
                    reference, new_references, reference_seconds, path = (
                        reference_execution(campaign, model, instance, policy_hash)
                    )
                    failed = (
                        reference["record"]["safety_failure"]
                        or reference["record"]["stalled"]
                    )
                if failed:
                    cache = caches["normal" if category == "normal" else "failures"]
                    produced, report = cache.get(instance)
                    report.update(
                        label_source="expert",
                        trajectory_path=str(
                            Path(report["cache_path"]).with_suffix(".npz")
                        ),
                    )
                else:
                    produced = samples_from_trace(
                        instance,
                        reference["trajectory"],
                        reference["trajectory_sha256"],
                    )
                    report = {
                        "instance_hash": instance.instance_hash,
                        "label_source": "policy_success",
                        "reference_policy_sha256": policy_hash,
                        "qualified_demonstration": True,
                        "cache_path": str(path.resolve()),
                        "trajectory_path": reference["trajectory"],
                        "trajectory_sha256": reference["trajectory_sha256"],
                        "candidates": 0,
                        "new_candidates": 0,
                        "new_elapsed_s": 0.0,
                    }
                report.update(
                    reference_new_executions=new_references,
                    reference_new_elapsed_s=reference_seconds,
                )
            for sample in produced:
                sample.update(
                    domain=domain,
                    scene_id=instance.root_sample_id,
                    label_source=report["label_source"],
                )
            report["samples"] = len(produced)
            reports.append(report)
            material[category].extend(produced)
            print(
                f"  {category} material {index + 1}/{len(entries)}: "
                f"{report['label_source']}, {len(produced)} labels",
                flush=True,
            )
    write_json(directory / "material_sources.json", reports)
    return material, reports
