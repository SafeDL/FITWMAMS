"""Audit the executed protocol and report search, repair, inference and export costs."""

import json
import math
import sqlite3
from collections import Counter
from pathlib import Path

import matplotlib
import torch

from ads.common.instances import GeneratedInstance, file_hash
from ads.common.reporting import write_json
from ads.common.setup import REPO_ROOT, load_policy
from ads.diffusion_planner.repair import TRAINABLE_PREFIXES
from ads.nuplan.assets import official_path


def execution_records(directory):
    records = {}
    for path in sorted(directory.rglob("*.json")):
        entry = json.loads(path.read_text())
        if "record" in entry:
            rows = [entry["record"]]
        elif path.name == "summary.json":
            rows = entry.get("records", [])
        else:
            continue
        # Partial summaries can overlap with retained per-instance training records.
        for row in rows:
            key = (path.parent, entry["policy_sha256"], row["instance_hash"])
            records[key] = row
    return list(records.values())


def plot_losses(loss_rows, output):
    import csv

    if not loss_rows:
        return
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    columns = sorted({key for row in loss_rows for key in row})
    with (output / "training_losses.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(loss_rows)
    figures = [
        (
            "training_losses.png",
            (
                "native_supervised_loss",
                "native_teacher_loss",
                "retention_loss",
                "correction_ego_loss",
                "correction_neighbor_loss",
                "total_loss",
            ),
            2,
        )
    ]
    for filename, keys, height in figures:
        figure, axes = plt.subplots(height, 3, figsize=(14, height * 3.5))
        for axis, key in zip(axes.flat, keys):
            for phase in sorted({row["phase"] for row in loss_rows}):
                rows = [r for r in loss_rows if r["phase"] == phase]
                x, values = [r["update"] for r in rows], [r[key] for r in rows]
                axis.plot(x, values, alpha=0.15)
                window = min(50, len(values))
                mean = [
                    sum(values[max(0, i - window + 1) : i + 1]) / min(window, i + 1)
                    for i in range(len(values))
                ]
                axis.plot(x, mean, label=phase)
            if key.endswith("_ratio"):
                axis.axhline(1, color="black", linewidth=0.7, linestyle="--")
            axis.set(xlabel="Cumulative updates", ylabel=key)
            axis.legend()
        figure.tight_layout()
        figure.savefig(output / filename, dpi=180)
        plt.close(figure)


def report(settings, output):
    source = json.loads((output / "sources.json").read_text())
    material = json.loads((output / "materials.json").read_text())
    native = json.loads(Path(settings["native_materials"]).read_text())
    experts = json.loads((output / "experts/report.json").read_text())
    final = json.loads((output / "final_results.json").read_text())
    checks = {}
    for purpose, total in (("replay", 13066), ("development", 3140)):
        coverage = native["coverage"][purpose]
        checks["native_full_" + purpose] = (
            coverage["candidates"] == total
            and coverage["scenes"] + coverage["excluded"] == total
        )
    sessions = [
        {r["source"]["session"] for r in native["scenes"][p]}
        for p in ("replay", "development")
    ]
    checks["native_session_isolation"] = not bool(sessions[0] & sessions[1])
    protected = set(source["protected_roots"])
    checks["protected_ancestry_recorded"] = len(protected) == 32
    with sqlite3.connect(output / "usage.sqlite") as db:
        leaked = db.execute(
            "SELECT roles.root FROM roles JOIN protected_roots ON roles.root=protected_roots.root WHERE roles.purpose='repair'"
        ).fetchall()
    checks["no_protected_learning"] = not leaked
    requested = {
        identity
        for m in source["methods"].values()
        for identity in m["eligible_failure_inputs"]
    }
    checks["all_failures_requested_experts"] = requested == {
        r["instance_hash"] for r in experts["reports"]
    }
    keep_counts = Counter()
    for entry in material["keep"]:
        instance = GeneratedInstance.load(entry["instance_path"])
        keep_counts[instance.source_kind] += 1
        if instance.root_sample_id in protected:
            raise AssertionError("Protected root entered the success bank")
        if instance.source_kind == "empirical_natural" and instance.source_metadata[
            "source_split"
        ] not in ("train", "val"):
            raise AssertionError("Empirical Test entered retention training")
    checks["all_successes_used"] = material[
        "all_qualified_successes_included"
    ] and material["keep_count"] == sum(keep_counts.values())
    checks["two_fresh_final_cohorts"] = final["different_instances"] == 10000 and final[
        "policy_executions"
    ] == 10000 * (len(settings["methods"]) + 1)
    costs = {
        "search": {
            "current_protocol_executions": sum(
                m["executions"] for m in source["methods"].values()
            )
        },
        "initial_experts": {
            "distinct_inputs": experts["inputs"],
            "expert_success_inputs": experts["expert_success_inputs"],
            "usable_inputs": experts["qualified_inputs"],
            "new_candidate_executions": experts["new_candidates_total"],
            "label_tracking_executions": experts["label_tracking_executions"],
            "new_physics_steps": experts["new_physics_steps_total"],
        },
    }
    reuse_path = output / "preparation_reuse.json"
    if reuse_path.exists():
        reuse = json.loads(reuse_path.read_text())
        prior = Path(reuse["prepared_output"])
        learning = json.loads(
            (prior / "phase_one_training_acceptance.json").read_text()
        )
        previous_records = execution_records(
            prior / "validation" / learning["policy_sha256"]
        )
        costs["scope"] = (
            "One shared search/expert preparation, rejected development attempt, and bounded repair; preparation is counted once."
        )
        costs["preparation_reuse"] = reuse
        costs["rejected_development_attempt"] = {
            "updates": learning["completed_updates"],
            "training_wall_s": learning["wall_elapsed_s"],
            "training_timing_source": learning["wall_timing_source"],
            "development_executions": len(previous_records),
            "development_physics_steps": sum(
                r["physics_steps"] for r in previous_records
            ),
            "development_execution_s": sum(r["elapsed_s"] for r in previous_records),
            "final_executions": 0,
        }
    search_root = REPO_ROOT / settings["reuse_search"]
    search_methods = settings["methods"] + settings.get("search_controls", [])
    costs["search"]["methods"] = {}
    for method in search_methods:
        directory = (
            search_root
            / "diffusion_planner"
            / str(settings["master_seed"])
            / method
            / "round_1"
        )
        summary = json.loads((directory / "search_summary.json").read_text())
        costs["search"]["methods"][method] = {
            key: summary[key]
            for key in (
                "evaluations",
                "new_execution_calls",
                "resumed_execution_calls",
                "parallel_workers",
                "wall_elapsed_s",
            )
        }
    costs["search"]["search_only_control_executions"] = sum(
        costs["search"]["methods"][method]["evaluations"]
        for method in settings.get("search_controls", [])
    )
    costs["search"]["total_with_controls"] = sum(
        values["evaluations"] for values in costs["search"]["methods"].values()
    )
    transition = output / "parallel_execution_transition.json"
    if transition.exists():
        costs["search"]["parallel_execution_transition"] = json.loads(
            transition.read_text()
        )
    parallel_check = output / "parallel_acceptance/report.json"
    if parallel_check.exists():
        costs["parallel_execution_verification"] = json.loads(
            parallel_check.read_text()
        )
    generation_transition = output / "generation_execution_transition.json"
    if generation_transition.exists():
        transition = json.loads(generation_transition.read_text())
        costs["search"]["generation_execution_transition"] = {
            "manifest_sha256": file_hash(generation_transition),
            "completed_calls_lost": transition["completed_calls_lost"],
            "completed_executions_preserved": {
                method: values["completed_executions"]
                for method, values in transition["methods"].items()
            },
        }
    generation_checks = [
        output / name
        for name in (
            "parallel_generation_prototype.json",
            "parallel_generation_acceptance.json",
        )
    ]
    costs["generation_verification"] = {
        path.name: json.loads(path.read_text())
        for path in generation_checks
        if path.exists()
    }
    for role in (
        "initial",
        "validation",
        "dangerous_validation",
        "success_neighbours",
        "repair_diagnostics",
        "final",
    ):
        entries = execution_records(output / role)
        costs[role] = {
            "full_executions": len(entries),
            "new_executions": 0 if role == "initial" else len(entries),
            "reused_execution_records": len(entries) if role == "initial" else 0,
            "physics_steps": sum(r["physics_steps"] for r in entries),
            "execution_s": sum(r["elapsed_s"] for r in entries),
        }
    costs["native_offline_validation"] = {}
    costs["native_offline_validation"]["status"] = "manual_by_user_not_executed"
    costs["queries"], costs["training"] = {}, {}
    baseline = load_policy(
        "diffusion_planner", official_path("diffusion_planner"), device="cpu"
    )
    before = baseline.network.state_dict()
    for method in settings["methods"]:
        directory = output / "repairs" / method
        result = json.loads((directory / "result.json").read_text())
        checks[method + "/selected_weights"] = (
            file_hash(directory / "policy.pth") == result["policy_sha256"]
        )
        after = load_policy(
            "diffusion_planner", directory / "policy.pth", device="cpu"
        ).network.state_dict()
        checks[method + "/frozen_parameters"] = all(
            torch.equal(value, after[name])
            for name, value in before.items()
            if not name.startswith(TRAINABLE_PREFIXES)
        )
        for path in sorted((directory / "candidates").glob("*/optimization.json")):
            learning = json.loads(path.read_text())
            if not learning["gradient_updates"]:
                continue
            checks[
                method + "/" + path.parent.name + "/batch_and_constraint"
            ] = learning["batch_size"] == 128 and all(
                max(r[field]["actual_normalized_dots"], default=0) <= 1e-9
                for r in learning["losses"]
                for field in ("constraint", "ema_constraint")
            )
            schedule = learning["plan"]["schedule"]
            checks[method + "/" + path.parent.name + "/reference_clock"] = math.isclose(
                sum(row["learning_rate"] for row in learning["losses"]),
                schedule["reference_learning_rate_sum"],
                rel_tol=1e-10,
            ) and math.isclose(
                sum(math.log(row["ema_decay"]) for row in learning["losses"]),
                math.log(schedule["reference_ema_retention"]),
                rel_tol=1e-10,
            )
        optimizations = [
            json.loads(p.read_text())
            for p in sorted((directory / "candidates").glob("*/optimization.json"))
        ]
        used = [r for r in optimizations if r["gradient_updates"]]
        if used:
            exposure = used[-1]["exposure"]
            for category in ("native", "keep"):
                checks[method + "/" + category + "_entry_coverage"] = (
                    exposure[category]["unvisited_entries"] == 0
                )
            eligible = set(source["methods"][method]["eligible_failure_inputs"])
            qualified_ids = {
                r["instance_hash"]
                for r in experts["reports"]
                if r["instance_hash"] in eligible and r["qualified_demonstration"]
            }
            checks[method + "/qualified_corrections_visited"] = qualified_ids <= set(
                exposure["correction"]["instance_visits"]
            )
            correction = exposure["correction"]
            checks[method + "/all_correction_instances_visited"] = (
                correction["visited_instances"] == correction["available_instances"]
            )
            checks[method + "/all_critical_entries_visited"] = (
                correction["visited_critical_frames"]
                == correction["available_critical_frames"]
            )
            checks[method + "/all_correction_entries_visited"] = (
                correction["unvisited_entries"] == 0
            )
        checks[method + "/update_budget"] = all(
            r["gradient_updates"] == r["plan"]["updates"]
            for r in optimizations
            if r["gradient_updates"]
        )
        costs["training"][method] = {
            "updates": sum(
                json.loads(p.read_text())["gradient_updates"]
                for p in (directory / "candidates").glob("*/optimization.json")
            ),
            "elapsed_s": sum(
                json.loads(p.read_text()).get("elapsed_s", 0)
                for p in (directory / "candidates").glob("*/optimization.json")
            ),
            "retention_projection_s": sum(
                step["constraint"]["elapsed_s"]
                for learning in optimizations
                for step in learning.get("losses", [])
            ),
            "ema_projection_s": sum(
                step["ema_constraint"]["elapsed_s"]
                for learning in optimizations
                for step in learning.get("losses", [])
            ),
        }
        query_path = directory / "queries/report.json"
        if query_path.exists():
            queries = json.loads(query_path.read_text())
            roots = queries["selected_ancestors"]
            checks[method + "/query_ancestry"] = not bool(
                set(roots) & protected
            ) and len(queries["selected_instances"]) == len(
                set(queries["selected_instances"])
            )
            collection_reports = [
                json.loads(p.read_text())
                for p in (directory / "queries/collections").glob("*.json")
            ]
            collections = [r["collection"] for r in collection_reports]
            costs["queries"][method] = {
                "training_recheck_executions": queries["training_recheck_instances"],
                "collection_full_executions": len(collections),
                "collection_physics_steps": sum(
                    r["physics_steps"] for r in collections
                ),
                "snapshot_replay_suffix_executions": sum(
                    "snapshot_replay" in r for r in collection_reports
                ),
                "snapshot_replay_physics_steps": sum(
                    r.get("snapshot_replay", {}).get("physics_steps", 0)
                    for r in collection_reports
                ),
                "suffix_candidate_executions": sum(
                    r["candidates"] for r in queries["reports"]
                ),
                "suffix_physics_steps": sum(
                    r["suffix_physics_steps"] for r in queries["reports"]
                ),
                "label_tracking_suffix_executions": sum(
                    r["label_tracking"] is not None for r in queries["reports"]
                ),
                "label_tracking_suffix_physics_steps": sum(
                    r["label_tracking"]["physics_steps"]
                    for r in queries["reports"]
                    if r["label_tracking"] is not None
                ),
            }
        handoff = json.loads(
            (output / "manual_nuplan" / method / "handoff.json").read_text()
        )
        checks[method + "/official_native_export"] = (
            handoff["native_parameters_equal"]
            and handoff["output_max_abs_difference"] == 0
            and handoff["weights_only_load"]
        )
    legacy = (
        output
        / "preparation_with_previous_source_identity/legacy_preparation_stop.json"
    )
    if legacy.exists():
        costs["superseded_preparation"] = json.loads(legacy.read_text())
        costs["superseded_preparation"]["in_flight_work_bound"] = {
            "additional_candidate_executions_at_most": 4 * 128,
            "additional_physics_steps_at_most": 4 * 128 * 149,
        }
    numerical_trial = output / "pre_projection_rounding_fix/costs.json"
    if numerical_trial.exists():
        costs["superseded_projection_trial"] = json.loads(numerical_trial.read_text())
    write_json(output / "costs.json", costs)
    write_json(
        output / "checkpoint_storage.json",
        {
            str(p.relative_to(output)): {
                "actual_path": str(p.resolve()),
                "sha256": file_hash(p),
            }
            for p in output.rglob("*.pth")
            if p.is_file()
        },
    )
    write_json(
        output / "acceptance.json",
        {
            "checks": checks,
            "passed": all(checks.values()),
            "city_closed_loop": "manual_by_user",
        },
    )
    if not all(checks.values()):
        raise AssertionError(
            "Protocol acceptance failed: "
            + ", ".join(k for k, value in checks.items() if not value)
        )
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    figure, axis = plt.subplots(figsize=(6.5, 4))
    for method in settings["methods"]:
        curve = source["methods"][method]["discovery"]["curve"]
        axis.plot(
            [r["executions"] for r in curve],
            [r["distinct_collisions"] for r in curve],
            label=method,
        )
    axis.set(
        xlabel="Closed-loop executions",
        ylabel="Distinct primary collision instances",
        xlim=(0, 5000),
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "discovery_efficiency.png", dpi=180)
    plt.close(figure)
    loss_rows = []
    for method in settings["methods"]:
        for path in sorted(
            (output / "repairs" / method / "candidates").glob("*/optimization.json")
        ):
            learning = json.loads(path.read_text())
            loss_rows.extend(
                {
                    "method": method,
                    "phase": path.parent.name,
                    **{k: v for k, v in row.items() if isinstance(v, (int, float))},
                }
                for row in learning.get("losses", [])
            )
    plot_losses(loss_rows, output)
    lines = [
        "# 全量材料与能力保持修复结果",
        "",
        "DP，单种子 20260926；仅新增本文搜索和一次两阶段全量修复，历史方法为近似参照。",
        "",
        "| 方法 | 实际搜索次数 | 去重碰撞输入 | 碰撞祖先 | 可申请纠正的失效输入 |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in settings["methods"]:
        values = source["methods"][method]
        lines.append(
            f"| {method} | {values['executions']} | {values['collision_inputs']} | {values['discovery']['collision_ancestors']} | {len(values['eligible_failure_inputs'])} |"
        )
    lines += [
        "",
        "| 方法 | 最终集合 | 初始碰撞率 | 修复后碰撞率 | 配对差（百分点） | 95% 区间（百分点） |",
        "|---|---|---:|---:|---:|---|",
    ]
    for method, cohorts in final["comparisons"].items():
        for cohort, values in cohorts.items():
            low, high = values["collision_rate_difference_ci95"]
            lines.append(
                f"| {method} | {cohort} | {values['collision_rate_before']:.2%} | {values['collision_rate_after']:.2%} | {values['collision_rate_difference_after_minus_before'] * 100:.3f} | [{low * 100:.3f}, {high * 100:.3f}] |"
            )
    lines += [
        "",
        "| 修复方法比较 | 最终集合 | 本文减参照的碰撞率差（百分点） | 95% 区间（百分点） |",
        "|---|---|---:|---|",
    ]
    for pair, cohorts in final["between_repair_methods"].items():
        for cohort, values in cohorts.items():
            low, high = values["collision_rate_difference_ci95"]
            lines.append(
                f"| {pair} | {cohort} | {values['collision_rate_difference_after_minus_before'] * 100:.3f} | [{low * 100:.3f}, {high * 100:.3f}] |"
            )
    lines += [
        "",
        "经验自然 Test 未新增评价。nuPlan 城市闭环由用户手动执行，原生权重在 `manual_nuplan/`。",
        "",
        "逐例结果、接纳检查、材料覆盖与成本分别见 `final/`、`repairs/`、`acceptance.json`、`costs.json`。专家成本随发现输入数变化，完整流程不作等总成本比较。",
        "",
    ]
    (output / "results.md").write_text("\n".join(lines))
    return checks
