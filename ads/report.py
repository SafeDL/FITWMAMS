"""Export discovery curves and costs from existing campaign executions."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from ads.common.teacher_costs import teacher_generation_costs
from ads.common.teacher_coverage import FIELDS as TEACHER_COVERAGE_FIELDS
from ads.common.teacher_coverage import teacher_coverage


def discovery_curve(records):
    collisions, failures, roots, domains = set(), set(), set(), set()
    curve = []
    for index, row in enumerate(records, 1):
        discovery = row.get("execution_kind") not in ("branch_check", "archive_replay")
        if discovery and row["safety_failure"]:
            failures.add(row["instance_hash"])
            if row.get("common_branch_id") is not None:
                domains.add(row["common_branch_id"])
        if discovery and row["collision"]:
            collisions.add(row["instance_hash"])
            roots.add(row["root_sample_id"])
        curve.append(
            {
                "evaluations": index,
                "distinct_collisions": len(collisions),
                "distinct_instance_failures": len(failures),
                "failure_roots": len(roots),
                "common_failure_branches": len(domains)
                if row.get("common_partition_resolved", False)
                else None,
            }
        )
    return curve


def export_evaluation(output, reports, config):
    summary = json.loads((output / "evaluation/summary.json").read_text())
    seed_rows, aggregate_rows = [], []
    fig, axes = plt.subplots(
        len(summary["cohorts"]),
        2,
        figsize=(13, 7 * len(summary["cohorts"])),
        squeeze=False,
    )
    for row, (cohort, result) in enumerate(summary["cohorts"].items()):
        groups = (
            ("versus_initial", result),
            ("between_repair_methods", result["between_repair_methods"]),
        )
        for ax, (scope, group) in zip(axes[row], groups):
            for key, value in group["per_seed"].items():
                algorithm, seed, comparison = key.split("/")
                item = {
                    "cohort": cohort,
                    "scope": scope,
                    "algorithm": algorithm,
                    "seed": int(seed),
                    "comparison": comparison,
                }
                item.update(
                    {
                        name: value
                        for name, value in value.items()
                        if isinstance(value, (int, float, bool, str))
                    }
                )
                item["collision_ci95_low"], item["collision_ci95_high"] = value[
                    "collision_rate_difference_ci95"
                ]
                seed_rows.append(item)
            labels = []
            for index, (key, value) in enumerate(
                group["shared_world_aggregate"].items()
            ):
                algorithm, comparison = key.split("/")
                low, high = value["paired_collision_difference_ci95"]
                mean = value["mean_collision_difference"]
                aggregate_rows.append(
                    {
                        "cohort": cohort,
                        "scope": scope,
                        "algorithm": algorithm,
                        "comparison": comparison,
                        "instances": value["instances"],
                        "independent_units": value["independent_units"],
                        "collision_difference": mean,
                        "ci95_low": low,
                        "ci95_high": high,
                        "bootstrap_unit": value["bootstrap_unit"],
                        "ci_method": value["collision_ci_method"],
                        "ci_is_degenerate": value["collision_ci_is_degenerate"],
                        "discordant_instances": value["collision_discordant_instances"],
                    }
                )
                ax.hlines(index, low, high, color="tab:blue")
                ax.plot(mean, index, "o", color="tab:blue", markersize=4)
                marker = " *" if value["collision_ci_is_degenerate"] else ""
                labels.append(f"{algorithm}: {comparison}{marker}")
            ax.axvline(0, color="grey", linestyle="--", linewidth=1)
            ax.set(
                yticks=range(len(labels)),
                yticklabels=labels,
                xlabel="Collision rate difference (after minus reference)",
                title=cohort + ": " + scope,
            )
            ax.invert_yaxis()
            ax.grid(axis="x", alpha=0.2)
    pd.DataFrame(seed_rows).to_csv(reports / "repair_per_seed.csv", index=False)
    pd.DataFrame(aggregate_rows).to_csv(reports / "repair_comparisons.csv", index=False)
    diagnostics = []
    for key, entry in summary["archive_diagnostics"].items():
        for round_name, value in entry["rounds"].items():
            item = {"policy": key, "round": round_name}
            item.update(
                {
                    name: value
                    for name, value in value.items()
                    if isinstance(value, (int, float, bool, str))
                }
            )
            diagnostics.append(item)
    pd.DataFrame(diagnostics).to_csv(reports / "archive_diagnostics.csv", index=False)
    fig.text(
        0.01,
        0.002,
        "* Degenerate cluster bootstrap: no sampled variation; not a bound on unseen discordance.",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.015, 1, 1))
    fig.savefig(reports / "repair.png", dpi=180)
    fig.savefig(reports / "repair.pdf")
    plt.close(fig)


def export_campaign(output, config):
    output = Path(output)
    reports = output / "reports"
    reports.mkdir(exist_ok=True)
    incidents = output / "engineering_incidents.json"
    if incidents.exists():
        (reports / incidents.name).write_bytes(incidents.read_bytes())
    curves, costs = [], []
    for algorithm in config["algorithms"]:
        for seed in config["seeds"]:
            for method in config["methods"]:
                for round_index in range(1, config["rounds"] + 1):
                    directory = (
                        output / algorithm / str(seed) / method / f"round_{round_index}"
                    )
                    records = [
                        json.loads(line)
                        for line in (directory / "search.jsonl")
                        .read_text()
                        .splitlines()
                    ]
                    identity = {
                        "algorithm": algorithm,
                        "seed": seed,
                        "method": method,
                        "round": round_index,
                    }
                    curves.extend(
                        {**identity, **row} for row in discovery_curve(records)
                    )
                    search = json.loads((directory / "search_summary.json").read_text())
                    if config["search"].get("fixed_budget", False) and not (
                        len(records)
                        == search["evaluations"]
                        == config["search"]["budget"]
                    ):
                        raise ValueError(f"unequal actual search budget: {directory}")
                    training = json.loads((directory / "training.json").read_text())
                    costs.append(
                        {
                            **identity,
                            "cost_scope": "completed_registered_attempt",
                            "search_executions": search["evaluations"],
                            "search_execution_s": search["execution_elapsed_s"],
                            "generation_s": search.get("generation_elapsed_s"),
                            "search_total_s": search.get("total_search_elapsed_s"),
                            "search_persistence_s": search.get("persistence_elapsed_s"),
                            "training_s": training["learning"]["elapsed_s"],
                            "repair_total_s": training["elapsed_s"],
                            "teacher_new_executions": training.get("teacher", {}).get(
                                "new_candidate_executions", 0
                            ),
                            "teacher_new_s": training.get("teacher", {}).get(
                                "new_elapsed_s", 0
                            ),
                            "reference_new_executions": training.get("teacher", {}).get(
                                "new_reference_executions", 0
                            ),
                            "reference_new_s": training.get("teacher", {}).get(
                                "reference_execution_s", 0
                            ),
                        }
                    )
    frame = pd.DataFrame(curves)
    frame.to_csv(reports / "discovery.csv", index=False)
    pd.DataFrame(costs).to_csv(reports / "costs.csv", index=False)
    pd.DataFrame(
        teacher_generation_costs(output, config),
        columns=(
            "cache",
            "instance_hash",
            "cache_kind",
            "candidate_executions",
            "generation_s",
            "qualified_demonstration",
            "training_consumers",
            "unique_label_frames",
            "cost_scope",
        ),
    ).to_csv(reports / "teacher_generation.csv", index=False)
    pd.DataFrame(
        teacher_coverage(output, config), columns=TEACHER_COVERAGE_FIELDS
    ).to_csv(reports / "teacher_coverage.csv", index=False)
    fig, axes = plt.subplots(
        len(config["algorithms"]),
        2,
        figsize=(13, 3 * len(config["algorithms"])),
        squeeze=False,
    )
    for row, algorithm in enumerate(config["algorithms"]):
        for col, metric in enumerate(("distinct_collisions", "failure_roots")):
            ax = axes[row, col]
            for method in config["methods"]:
                selected = frame[
                    (frame.algorithm == algorithm)
                    & (frame.method == method)
                    & (frame["round"] == 1)
                ]
                grouped = selected.groupby("evaluations")[metric]
                mean, std = grouped.mean(), grouped.std(ddof=0)
                ax.plot(mean.index, mean, label=method)
                ax.fill_between(mean.index, mean - std, mean + std, alpha=0.15)
            ax.set(
                xlabel="Closed-loop executions",
                ylabel=metric.replace("_", " "),
                title=algorithm,
            )
            ax.grid(alpha=0.2)
    axes[0, 0].legend()
    fig.suptitle(f"{config['run_kind']}: round 1 discovery at actual execution costs")
    fig.tight_layout()
    fig.savefig(reports / "discovery.png", dpi=180)
    fig.savefig(reports / "discovery.pdf")
    plt.close(fig)
    export_evaluation(output, reports, config)
