"""Execute the full-data, single-cycle capability-preserving DP protocol."""

import fcntl
import gc
import json
import multiprocessing
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import psutil
import torch
import yaml

from ads.common.admission import safe_task_change
from ads.common.batch_evaluation import evaluate_paths
from ads.common.instances import GeneratedInstance, UsageLedger, file_hash
from ads.common.reporting import paired_summary, write_json
from ads.common.setup import REPO_ROOT, load_policy
from ads.common.world import FrozenWorldFactory
from ads.diffusion_planner.aggregation import (
    collect_query,
    correct_query,
    select_query_records,
)
from ads.diffusion_planner.capability_materials import KeepSamples, generate_pool
from ads.diffusion_planner.checkpoints import (
    link_checkpoint,
    model_destination,
    save_checkpoint,
)
from ads.diffusion_planner.expert_materials import CONFIG
from ads.diffusion_planner.native_coverage import audit as audit_native_coverage
from ads.diffusion_planner.native_export import export_verified
from ads.diffusion_planner.prepare_data import load_training_samples
from ads.diffusion_planner.repair import normalized_schedule, train
from ads.diffusion_planner.sampling import phase_budget
from ads.diffusion_planner.search_first import launch_search
from ads.diffusion_planner.sources import import_search
from ads.diffusion_planner.summarize import plot_losses
from ads.diffusion_planner.summarize import report as summarize
from ads.nuplan.assets import official_path
from ads.nuplan.teacher import demonstration
from ads.search.comparison import compare_with_history
from tools.evt import GPDTailModel


def initialize_suffix_worker(settings, ledger_path):
    global _suffix_factory, _suffix_ledger
    torch.set_num_threads(settings["runtime"]["torch_threads"])
    _suffix_factory = FrozenWorldFactory(
        REPO_ROOT / settings["world_config"], device=settings["runtime"]["world_device"]
    )
    _suffix_ledger = UsageLedger(ledger_path)


def suffix_job(job):
    instance_path, snapshot_path, snapshot_hash, directory, policy_hash, budget = job
    if file_hash(snapshot_path) != snapshot_hash:
        raise ValueError("Query snapshot changed before expert execution")
    instance = GeneratedInstance.load(instance_path)
    snapshot = torch.load(snapshot_path, map_location="cpu", weights_only=False)
    return correct_query(
        instance,
        snapshot,
        _suffix_factory,
        Path(directory),
        policy_hash=policy_hash,
        ledger=_suffix_ledger,
        budget=budget,
    )


def wait_search(settings, output):
    registry = json.loads((output / "search_processes.json").read_text())
    root = (
        REPO_ROOT
        / settings["reuse_search"]
        / "diffusion_planner"
        / str(settings["master_seed"])
    )
    while True:
        pending = []
        for method in settings["methods"] + settings.get("search_controls", []):
            path = root / method / "round_1/search_summary.json"
            if path.exists():
                if (
                    not 0
                    < json.loads(path.read_text())["evaluations"]
                    <= settings["search"]["budget"]
                ):
                    raise ValueError(
                        "Search completed with a different execution budget"
                    )
                continue
            entry = registry[method]
            try:
                process = psutil.Process(entry["pid"])
                alive = (
                    process.is_running()
                    and process.status() != psutil.STATUS_ZOMBIE
                    and process.create_time() == entry["created"]
                )
            except psutil.NoSuchProcess:
                alive = False
            if not alive:
                raise RuntimeError(
                    f"Search worker {method} stopped before its completion report; inspect search_{method}.log"
                )
            pending.append(method)
        if not pending:
            return
        print("Waiting for live search workers: " + ", ".join(pending), flush=True)
        time.sleep(30)


def validate(settings, output, policy, material):
    identity = file_hash(policy)
    initial = identity == file_hash(official_path("diffusion_planner"))
    records = {}
    for name, paths in material["development"].items():
        directory = (
            output / "initial" / name
            if initial
            else output / "validation" / identity / name
        )
        records[name] = [
            row["record"]
            for row in evaluate_paths(
                paths,
                policy,
                directory,
                settings,
                workers=settings["runtime"]["workers"],
                ledger_path=output / "usage.sqlite",
            )
        ]
    return {"records": records, "native": {"status": "manual_by_user"}}


def candidate_decision(original, previous, candidate):
    checks, summaries = {}, {}
    for reference, before in (("original", original), ("previous", previous)):
        for name in ("tail_generated", "natural_generated"):
            summaries[reference + "/" + name] = paired_summary(
                before["records"][name], candidate["records"][name]
            )
            checks[reference + "/" + name] = safe_task_change(
                before["records"][name], candidate["records"][name]
            )
    native_diagnostics = {"status": "manual_by_user"}
    danger_before_rows, danger_after_rows = [], []
    for name in ("tail_generated", "natural_generated"):
        baseline = {r["instance_hash"]: r for r in original["records"][name]}
        danger_before_rows.extend(r for r in baseline.values() if r["safety_failure"])
        danger_after_rows.extend(
            r
            for r in candidate["records"][name]
            if baseline[r["instance_hash"]]["safety_failure"]
        )
    danger = (
        paired_summary(danger_before_rows, danger_after_rows, diagnostic=True)
        if danger_before_rows
        else {"instances": 0}
    )
    tail_before = sum(r["collision"] for r in original["records"]["tail_generated"])
    tail_after = sum(r["collision"] for r in candidate["records"]["tail_generated"])
    danger_before = sum(r["safety_failure"] for r in danger_before_rows)
    danger_after = sum(r["safety_failure"] for r in danger_after_rows)
    original_success = {
        r["instance_hash"]
        for name in ("tail_generated", "natural_generated")
        for r in original["records"][name]
        if not r["collision"]
    }
    regressions = sum(
        r["collision"] and r["instance_hash"] in original_success
        for name in ("tail_generated", "natural_generated")
        for r in candidate["records"][name]
    )
    passed = all(all(group.values()) for group in checks.values())
    improved = tail_after < tail_before or danger_after < danger_before
    return {
        "continue": passed,
        "eligible": passed and improved,
        "checks": checks,
        "native_diagnostics": native_diagnostics,
        "tail_or_existing_failures_improved": improved,
        "summaries": summaries,
        "dangerous": danger,
        "rank": [
            tail_after,
            sum(r["collision"] for r in candidate["records"]["natural_generated"]),
            regressions,
            sum(
                r["offroad"]
                for name in ("tail_generated", "natural_generated")
                for r in candidate["records"][name]
            ),
        ],
    }


def corrections(source, method, output):
    samples, reports = [], []
    for identity in source["methods"][method]["eligible_failure_inputs"]:
        report = json.loads(
            (output / "experts/inputs" / (identity + ".json")).read_text()
        )
        instance = GeneratedInstance.load(source["paths"][identity])
        if (
            instance.root_sample_id in source["protected_roots"]
            or file_hash(report["trajectory_path"]) != report["trajectory_sha256"]
        ):
            raise ValueError("Expert input or protected ancestry changed")
        if report["qualified_demonstration"]:
            with np.load(report["trajectory_path"], allow_pickle=False) as data:
                produced = demonstration(instance, data["states"], "diffusion_planner")
            for sample in produced:
                critical_frame = source["records"][identity]["critical_frame"]
                sample.update(
                    scene_id=instance.root_sample_id,
                    label_source="expert",
                    domain="correction",
                    purpose="repair",
                    critical_window=max(0, critical_frame - 25)
                    <= sample["frame_id"]
                    <= critical_frame,
                )
            samples.extend(produced)
        reports.append(report)
    return samples, reports


def aggregate_queries(
    settings,
    output,
    directory,
    source,
    model,
    trained_samples,
    success_samples=(),
    candidate_path=None,
):
    factory = FrozenWorldFactory(
        REPO_ROOT / settings["world_config"], device=settings["runtime"]["world_device"]
    )
    evt = GPDTailModel.from_dict(
        json.loads((REPO_ROOT / settings["evt_model"]).read_text())
    )
    ledger = UsageLedger(output / "usage.sqlite")
    visited = set(model.repair_state["exposure"]["correction"]["instance_visits"])
    identities = {sample["instance_id"] for sample in trained_samples} & visited
    visited_keep = set(model.repair_state["exposure"]["keep"]["instance_visits"])
    neighbours = set(source["success_neighbours"].values())
    identities |= (
        {
            sample["instance_id"]
            for sample in (
                success_samples.metadata
                if hasattr(success_samples, "metadata")
                else success_samples
            )
        }
        & visited_keep
        & neighbours
    )
    current = (
        evaluate_paths(
            [source["paths"][identity] for identity in sorted(identities)],
            candidate_path,
            directory / "queries/current_policy",
            settings,
            purpose="repair",
            workers=settings["runtime"]["workers"],
            ledger_path=output / "usage.sqlite",
        )
        if identities
        else []
    )
    selected = select_query_records(
        [entry["record"] for entry in current],
        source["records"],
    )
    roots = [row["root_sample_id"] for row in selected]
    samples, reports, jobs = [], [], []
    for index, row in enumerate(selected):
        instance = GeneratedInstance.load(source["paths"][row["instance_hash"]])
        identity = row["instance_hash"]
        collection_path = directory / "queries/collections" / (identity + ".json")
        snapshot_path = collection_path.with_suffix(".snapshot.pt")
        if collection_path.exists():
            collection = json.loads(collection_path.read_text())
            snapshot = (
                torch.load(snapshot_path, weights_only=False)
                if snapshot_path.exists()
                else None
            )
        else:
            snapshot, collection = collect_query(
                instance, model, factory, evt, ledger, seed=settings["master_seed"]
            )
            collection_path.parent.mkdir(parents=True, exist_ok=True)
            if snapshot is not None:
                torch.save(snapshot, snapshot_path)
                collection["snapshot_sha256"] = file_hash(snapshot_path)
            write_json(collection_path, collection)
        if snapshot is not None:
            if file_hash(snapshot_path) != collection["snapshot_sha256"]:
                raise ValueError("Query snapshot changed")
            jobs.append(
                (
                    source["paths"][identity],
                    str(snapshot_path),
                    collection["snapshot_sha256"],
                    str(directory / "queries/experts"),
                    file_hash(candidate_path),
                    settings["repair"]["expert_candidates"],
                )
            )
        print(f"Offset query collection: {index + 1}/{len(roots)}", flush=True)
    with ProcessPoolExecutor(
        max_workers=settings["runtime"]["workers"],
        mp_context=multiprocessing.get_context("spawn"),
        initializer=initialize_suffix_worker,
        initargs=(settings, output / "usage.sqlite"),
    ) as executor:
        for index, (produced, report) in enumerate(
            executor.map(suffix_job, jobs, chunksize=1), 1
        ):
            samples.extend(produced)
            reports.append(report)
            print(f"Offset experts: {index}/{len(jobs)}", flush=True)
    write_json(
        directory / "queries/report.json",
        {
            "selected_ancestors": roots,
            "selected_instances": [row["instance_hash"] for row in selected],
            "training_recheck_instances": len(current),
            "current_failures": sum(
                entry["record"]["safety_failure"] for entry in current
            ),
            "too_early_for_query": sum(
                entry["record"]["safety_failure"]
                and entry["record"]["critical_frame"] < 30
                for entry in current
            ),
            "qualified_queries": len({s["query_id"] for s in samples}),
            "samples": len(samples),
            "reports": reports,
        },
    )
    return samples


def repair_method(settings, output, source, material, method, native, keep, original):
    directory = output / "repairs" / method
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "result.json").exists():
        return json.loads((directory / "result.json").read_text())
    corrected, expert_reports = corrections(source, method, output)
    model = load_policy(
        "diffusion_planner",
        official_path("diffusion_planner"),
        device=settings["runtime"]["device"],
    )
    accepted, accepted_validation, accepted_rank, candidates = (
        official_path("diffusion_planner"),
        original,
        None,
        [],
    )
    first_updates = None
    query_report = None
    for phase in range(2):
        phase_dir = directory / "candidates" / f"phase_{phase + 1}"
        phase_dir.mkdir(parents=True, exist_ok=True)
        candidate_path, partial = (
            phase_dir / "policy.pth",
            phase_dir / "in_progress.pth",
        )
        prior_done = (model.repair_state or {}).get("gradient_updates", 0)
        minimum = settings["repair"]["minimum_phase_updates"]
        if phase and (query_report or {}).get("current_failures", 0):
            minimum = max(minimum, int(np.ceil(first_updates / 2)))
        plan_path = phase_dir / "plan.json"
        if plan_path.exists():
            plan = json.loads(plan_path.read_text())
        else:
            plan = phase_budget(
                native,
                keep,
                corrected,
                minimum=minimum,
                exposure=(model.repair_state or {}).get("samplers"),
            )
            plan.update(
                phase=phase + 1,
                start_update=prior_done,
                end_update=prior_done + plan["updates"],
                correction_entries=len(corrected),
                data_policy="all_valid_entries_no_truncation",
                start_training_elapsed_s=(model.repair_state or {}).get(
                    "training_elapsed_s", 0.0
                ),
                schedule=normalized_schedule(
                    plan["updates"],
                    settings["repair"]["reference_updates"],
                    settings["repair"]["learning_rate"],
                    settings["repair"]["ema_decay"],
                ),
            )
            write_json(plan_path, plan)
        updates, cumulative = plan["updates"], plan["end_update"]
        if phase == 0:
            first_updates = updates
        optimization = phase_dir / "optimization.json"
        if candidate_path.exists():
            model = load_policy(
                "diffusion_planner",
                candidate_path,
                device=settings["runtime"]["device"],
            )
            if model.repair_state["gradient_updates"] != cumulative:
                raise ValueError("Completed phase has an inconsistent update count")
        else:
            if partial.exists():
                model = load_policy(
                    "diffusion_planner", partial, device=settings["runtime"]["device"]
                )
            done = (model.repair_state or {}).get("gradient_updates", 0)
            if not plan["start_update"] <= done <= cumulative:
                raise ValueError(
                    "Partial repair checkpoint lies outside the phase plan"
                )
            learning = {"gradient_updates": updates}
            if done < cumulative:
                learning = train(
                    model,
                    {"native": native, "keep": keep, "new_failure": corrected},
                    updates=cumulative - done,
                    phase_updates=updates,
                    phase_offset=done - plan["start_update"],
                    seed=settings["master_seed"],
                    micro_batch_size=settings["repair"]["micro_batch_size"],
                    learning_rate=plan["schedule"]["learning_rate"],
                    ema_decay=plan["schedule"]["ema_decay"],
                    retention_coefficient=settings["repair"]["retention_coefficient"],
                    protected_roots=source["protected_roots"],
                    checkpoint=lambda policy, count, destination=partial: (
                        save_checkpoint(policy, output, destination)
                    ),
                    progress=lambda count, total, losses, stage=phase + 1, end=cumulative: (
                        print(
                            f"Repair {method}, phase {stage}: {count}/{end}, correction {losses['correction_loss']:.5g}",
                            flush=True,
                        )
                    ),
                )
            if not learning["gradient_updates"]:
                write_json(optimization, learning)
                break
            save_checkpoint(model, output, candidate_path)
        learning = {
            "gradient_updates": updates,
            "cumulative_updates": cumulative,
            "batch_size": 128,
            "exposure": model.repair_state["exposure"],
            "losses": model.repair_state["losses"][plan["start_update"] : cumulative],
            "candidate_sha256": file_hash(candidate_path),
            "plan": plan,
            "elapsed_s": model.repair_state["training_elapsed_s"]
            - plan["start_training_elapsed_s"],
        }
        for name, exposure in learning["exposure"].items():
            if exposure["unvisited_entries"]:
                raise AssertionError(
                    f"{name} has unvisited entries after its full-data phase"
                )
        if model.repair_state["exposure"]["correction"]["minimum_critical_visits"] < (
            2 if any(s.get("critical_window") for s in corrected) else 0
        ):
            raise AssertionError("Critical corrections did not complete two visits")
        write_json(optimization, learning)
        plot_losses(
            [
                {
                    "method": method,
                    "phase": phase_dir.name,
                    **{k: v for k, v in row.items() if isinstance(v, (int, float))},
                }
                for row in learning["losses"]
            ],
            phase_dir,
        )
        evaluated = validate(settings, output, candidate_path, material)
        decision = candidate_decision(original, accepted_validation, evaluated)
        decision.update(
            updates=cumulative,
            phase=phase + 1,
            policy_sha256=file_hash(candidate_path),
            policy=str(candidate_path.resolve()),
        )
        write_json(phase_dir / "admission.json", decision)
        candidates.append(decision)
        if decision["eligible"] and (
            accepted_rank is None or decision["rank"] < accepted_rank
        ):
            accepted, accepted_validation, accepted_rank = (
                candidate_path,
                evaluated,
                decision["rank"],
            )
        if not decision["continue"]:
            model = load_policy(
                "diffusion_planner", accepted, device=settings["runtime"]["device"]
            )
            break
        if phase == 0:
            corrected += aggregate_queries(
                settings,
                output,
                directory,
                source,
                model,
                corrected,
                keep,
                candidate_path=candidate_path,
            )
            query_report = json.loads((directory / "queries/report.json").read_text())
    final = directory / "policy.pth"
    link_checkpoint(final, Path(accepted))
    result = {
        "method": method,
        "accepted_update": accepted_rank is not None,
        "selected_policy": str(final.resolve()),
        "policy_sha256": file_hash(final),
        "selected_source": str(Path(accepted).resolve()),
        "candidates": candidates,
        "expert_inputs": len(expert_reports),
        "qualified_initial_inputs": sum(
            r["qualified_demonstration"] for r in expert_reports
        ),
        "native_scenes": len(native),
        "keep_label_frames": len(keep),
        "full_data_learning": bool(candidates),
        "data_policy": "all_valid_entries_no_truncation",
    }
    write_json(directory / "result.json", result)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def final_evaluation(settings, output, source, results):
    policies = {
        "initial": str(official_path("diffusion_planner")),
        **{r["method"]: r["selected_policy"] for r in results},
    }
    freeze = {
        "settings": settings,
        "sources_sha256": file_hash(output / "sources.json"),
        "materials_sha256": file_hash(output / "materials.json"),
        "native_materials_sha256": file_hash(settings["native_materials"]),
        "world_assets": json.loads((output / "materials.json").read_text())["assets"],
        "final_namespace": settings["run_kind"],
        "policies": {
            name: {"path": path, "sha256": file_hash(path)}
            for name, path in policies.items()
        },
        "ads_sources": {
            str(p.relative_to(REPO_ROOT)): file_hash(p)
            for p in sorted((REPO_ROOT / "ads").rglob("*.py"))
        },
    }
    freeze_path = output / "freeze.json"
    if freeze_path.exists() and json.loads(freeze_path.read_text()) != freeze:
        raise ValueError("Final evaluation freeze changed")
    write_json(freeze_path, freeze)
    cohorts = {}
    for kind in ("tail_generated", "natural_generated"):
        paths, actual_assets = generate_pool(
            settings,
            output,
            name="final_" + kind,
            kind=kind,
            purpose="final_test",
            count=settings["evaluation"]["instances"],
        )
        if actual_assets != freeze["world_assets"][kind]:
            raise ValueError("Final generated assets differ from the frozen study")
        cohorts[kind] = paths
    records, comparisons = {}, {}
    for name, policy in policies.items():
        records[name] = {}
        for kind, paths in cohorts.items():
            records[name][kind] = [
                r["record"]
                for r in evaluate_paths(
                    paths,
                    policy,
                    output / "final" / name / kind,
                    settings,
                    purpose="final_test",
                    workers=settings["runtime"]["workers"],
                    ledger_path=output / "usage.sqlite",
                )
            ]
    for name in settings["methods"]:
        comparisons[name] = {
            kind: paired_summary(records["initial"][kind], records[name][kind])
            for kind in cohorts
        }
    between_methods = {
        "guided_vs_"
        + reference: {
            kind: paired_summary(records[reference][kind], records["guided"][kind])
            for kind in cohorts
        }
        for reference in ("monte_carlo", "subset")
        if reference in records
    }
    diagnostics = {}
    for result in results:
        method = result["method"]
        identities = set(source["methods"][method]["eligible_failure_inputs"])
        identities.update(
            source["success_neighbours"][identity]
            for identity in list(identities)
            if identity in source["success_neighbours"]
        )
        identities = sorted(identities)
        entries = evaluate_paths(
            [source["paths"][identity] for identity in identities],
            result["selected_policy"],
            output / "repair_diagnostics" / method,
            settings,
            purpose="search",
            workers=settings["runtime"]["workers"],
            ledger_path=output / "usage.sqlite",
        )
        diagnostics[method] = (
            paired_summary(
                [source["records"][identity] for identity in identities],
                [entry["record"] for entry in entries],
                diagnostic=True,
            )
            if entries
            else {"instances": 0}
        )
    write_json(
        output / "repair_diagnostics.json",
        {
            "used_for_model_selection": False,
            "prior_probability_estimate": False,
            "methods": diagnostics,
        },
    )
    write_json(
        output / "final_results.json",
        {
            "cohort_instances": settings["evaluation"]["instances"],
            "different_instances": sum(len(p) for p in cohorts.values()),
            "policy_executions": sum(
                len(v) for c in records.values() for v in c.values()
            ),
            "comparisons": comparisons,
            "between_repair_methods": between_methods,
        },
    )
    for name, policy in policies.items():
        role = "enhanced" if name == "guided" else name
        destination = output / "manual_nuplan" / role / "model.pth"
        sample = load_training_samples(settings["native_materials"], purpose="replay")[
            0
        ]
        actual = model_destination(output, destination)
        export_verified(policy, actual, sample)
        link_checkpoint(destination, actual)
        for filename in ("handoff.json", "model.pth.source.json"):
            shutil.copy2(actual.parent / filename, destination.parent / filename)
        (output / "manual_nuplan/native_metrics" / role).mkdir(
            parents=True, exist_ok=True
        )
    manual = output / "manual_nuplan"
    original_manifest = (
        REPO_ROOT
        / "results/ads/assets/nuplan_evaluation/common_scenarios.json"
    )
    manifest = json.loads(original_manifest.read_text())
    scenario_path = original_manifest.parent / manifest["path"]
    if file_hash(scenario_path) != manifest["sha256"]:
        raise ValueError("Manual nuPlan common scenario manifest changed")
    shutil.copy2(scenario_path, manual / manifest["path"])
    write_json(manual / "common_scenarios.json", manifest)
    preparation = original_manifest.parent / "preparation/config"
    shutil.copytree(preparation, manual / "preparation/config", dirs_exist_ok=True)
    original_readme = (original_manifest.parent / "README.md").read_text()
    command = original_readme.split("```bash\n", 1)[1].split("```", 1)[0]
    command = "\n".join(
        "ADS_PAIR=" + str(manual.resolve()) if line.startswith("ADS_PAIR=") else line
        for line in command.splitlines()
    ).replace(
        str(original_manifest.parent / "preparation/config"),
        str(manual / "preparation/config"),
    )
    (manual / "README.md").write_text(
        "# nuPlan 人工配对评价\n\n原始与安全增强版权重位于 initial、enhanced；highway-env 与 nuPlan 使用相同部署参数。\n\n"
        "依次将 ADS_ROLE 设置为 initial 和待比较方法，使用以下相同命令：\n\n```bash\n"
        + command
        + "\n```\n\n"
        "ADS_CHECKPOINT 为 `$ADS_PAIR/$ADS_ROLE/model.pth`，ADS_ARGS 为 `$ADS_PAIR/$ADS_ROLE/args.json`。使用 tread、同一场景清单及 seed 20260926；结果分别保存到 native_metrics/$ADS_ROLE。\n\n"
        "scenarios.json 保留已有共同场景清单；各模型 handoff.json 记录官方加载器、权重及输出一致性。本工程不自动执行城市闭环，也不将其指标并入 highD 分母或用于已冻结模型重选。\n"
    )
    return comparisons


def run(settings):
    output = REPO_ROOT / settings["output_dir"]
    torch.set_num_threads(settings["runtime"]["torch_threads"])
    launch_search(settings, output)
    wait_search(settings, output)
    compare_with_history(settings, output)
    audit_native_coverage(settings["native_materials"], output / "native_coverage.json")
    source = import_search(settings, REPO_ROOT, output)
    write_json(
        output / "status.json",
        {"status": "running", "stage": "initial_materials", "pid": os.getpid()},
    )
    processes = []
    for module, log in (
        ("expert_materials", "experts.log"),
        ("capability_materials", "materials.log"),
    ):
        report_path = output / (
            "experts/report.json" if module == "expert_materials" else "materials.json"
        )
        if (
            report_path.exists()
            and json.loads(report_path.read_text())["status"] == "complete"
        ):
            report = json.loads(report_path.read_text())
            if module == "expert_materials" and report["sources_sha256"] != file_hash(
                output / "sources.json"
            ):
                raise ValueError("Completed expert pool belongs to another search")
            continue
        stream = (output / log).open("a")
        process = subprocess.Popen(
            [sys.executable, "-u", "-m", "ads.diffusion_planner." + module],
            stdout=stream,
            stderr=subprocess.STDOUT,
            cwd=REPO_ROOT,
        )
        processes.append((process, stream, module))
    write_json(
        output / "preparation_processes.json",
        {
            module: {"pid": process.pid, "command": process.args}
            for process, _, module in processes
        },
    )
    for process, stream, module in processes:
        code = process.wait()
        stream.close()
        if code:
            raise RuntimeError(
                f"Preparation {module} exited with code {code}; inspect its log"
            )
    material = json.loads((output / "materials.json").read_text())
    protocol = {
        "settings": settings,
        "sources_sha256": file_hash(output / "sources.json"),
        "materials_sha256": file_hash(output / "materials.json"),
        "native_materials_sha256": file_hash(settings["native_materials"]),
        "experts_sha256": file_hash(output / "experts/report.json"),
        "ads_sources": {
            str(p.relative_to(REPO_ROOT)): file_hash(p)
            for p in sorted((REPO_ROOT / "ads").rglob("*.py"))
        },
    }
    protocol_path = output / "training_protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("Training protocol changed after material preparation")
    write_json(protocol_path, protocol)
    for name, digest in protocol["ads_sources"].items():
        destination = output / "learning_source_snapshot" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and file_hash(destination) != digest:
            raise ValueError("Frozen training source snapshot differs")
        if not destination.exists():
            destination.write_bytes((REPO_ROOT / name).read_bytes())
    configuration = "ads/configs/safety_enhanced.yaml"
    destination = output / "learning_source_snapshot" / configuration
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        destination.write_bytes((REPO_ROOT / configuration).read_bytes())
    native, keep = (
        load_training_samples(settings["native_materials"]),
        KeepSamples(material),
    )
    original = validate(settings, output, official_path("diffusion_planner"), material)
    results = []
    for method in settings["methods"]:
        write_json(
            output / "status.json",
            {
                "status": "running",
                "stage": "repair",
                "method": method,
                "pid": os.getpid(),
            },
        )
        results.append(
            repair_method(
                settings, output, source, material, method, native, keep, original
            )
        )
    write_json(
        output / "status.json",
        {"status": "running", "stage": "final_evaluation", "pid": os.getpid()},
    )
    final_evaluation(settings, output, source, results)
    summarize(settings, output)
    write_json(
        output / "status.json",
        {
            "status": "generated_experiments_complete_manual_nuplan_pending",
            "results": results,
        },
    )


if __name__ == "__main__":
    settings = yaml.safe_load(CONFIG.read_text())
    output = REPO_ROOT / settings["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    with (output / "study.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write_json(
            output / "status.json",
            {"status": "running", "stage": "waiting_for_fresh_search"},
        )
        try:
            run(settings)
        except Exception as error:
            write_json(
                output / "status.json", {"status": "failed", "error": str(error)}
            )
            raise
