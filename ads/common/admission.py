"""Paired highway development admission shared by supervised and PPO repair."""

import json
from pathlib import Path

from ads.common.batch_evaluation import evaluate_paths
from ads.common.instances import file_hash
from ads.common.reporting import paired_summary, write_json
from ads.search.archive import FailureArchive


def development_results(
    policy_path, campaign, *, algorithm, official, material_root, development
):
    material_root = Path(material_root)
    identity = file_hash(policy_path)
    records = {}
    for name, paths in development.items():
        # Initial official outputs were already collected during material preparation.
        if identity == file_hash(official):
            entries = [
                json.loads(
                    (
                        material_root / "initial" / name / (Path(p).stem + ".json")
                    ).read_text()
                )
                for p in paths
            ]
        else:
            entries = evaluate_paths(
                paths,
                policy_path,
                material_root / "validation" / identity / name,
                campaign.config,
                workers=4,
                ledger_path=material_root / "usage.sqlite",
                algorithm=algorithm,
            )
        if any(row["policy_sha256"] != identity for row in entries):
            raise ValueError("Development results belong to another deployment policy")
        records[name] = [row["record"] for row in entries]
    return records


def safe_task_change(before, after, minimum_progress_ratio=0.98):
    if len(before) != len(after):
        raise ValueError("Candidate validation cohorts differ")
    checks = {}
    for field in ("collision", "offroad", "stalled"):
        checks[field] = sum(row[field] for row in after) <= sum(
            row[field] for row in before
        )
    checks["progress"] = sum(
        row["progress_m"] for row in after
    ) >= minimum_progress_ratio * sum(row["progress_m"] for row in before)
    return checks


def admission(
    candidates,
    base,
    campaign,
    directory,
    *,
    algorithm,
    official,
    material_root,
    development,
    training_instances,
):
    directory = Path(directory)

    def validation(path):
        return development_results(
            path,
            campaign,
            algorithm=algorithm,
            official=official,
            material_root=material_root,
            development=development,
        )

    references = {"official": validation(official)}
    if file_hash(base) == file_hash(official):
        references["previous"] = references["official"]
    else:
        references["previous"] = validation(base)
    archive = FailureArchive.load(directory / "archive")
    trained = set(training_instances)
    endpoints = sorted(
        archive.endpoints(limit=len(archive.endpoints())),
        key=lambda result: result.instance.instance_hash in trained,
    )[:64]
    paths = [
        directory / "archive" / (result.instance.instance_hash + ".npz")
        for result in endpoints
    ]
    write_json(
        directory / "admission_endpoints.json",
        [
            {
                "instance_hash": result.instance.instance_hash,
                "root_sample_id": result.instance.root_sample_id,
                "used_for_training": result.instance.instance_hash in trained,
            }
            for result in endpoints
        ],
    )

    def archive_results(path):
        if not paths:
            return []
        return [
            row["record"]
            for row in evaluate_paths(
                paths,
                path,
                directory / "development_archive" / file_hash(path),
                campaign.config,
                algorithm=algorithm,
                purpose="search",
                workers=4,
                ledger_path=campaign.ledger.path,
            )
        ]

    archive_before = {
        "official": archive_results(official),
        "previous": archive_results(base),
    }
    reports, eligible = [], []
    for update, candidate in candidates:
        results = validation(candidate)
        archive_after = archive_results(candidate)
        checks, summaries = {}, {}
        for reference, cohorts in references.items():
            for name in cohorts:
                checks[reference + "/" + name] = safe_task_change(
                    cohorts[name],
                    results[name],
                    campaign.config["capability_preserving_repair"][
                        "minimum_progress_ratio"
                    ],
                )
                summaries[reference + "/" + name] = paired_summary(
                    cohorts[name], results[name]
                )
            improved_tail = sum(
                row["collision"] for row in results["tail_generated"]
            ) < sum(row["collision"] for row in cohorts["tail_generated"])
            old = {row["instance_hash"]: row for row in archive_before[reference]}
            repaired = sum(
                old[row["instance_hash"]]["safety_failure"]
                and not row["safety_failure"]
                for row in archive_after
            )
            regressed = sum(
                not old[row["instance_hash"]]["safety_failure"]
                and row["safety_failure"]
                for row in archive_after
            )
            checks[reference + "/improvement"] = {
                "tail_or_archive_improved": improved_tail
                or (repaired > 0 and regressed == 0)
            }
            summaries[reference + "/archive"] = {
                "endpoints": len(paths),
                "repaired": repaired,
                "regressed": regressed,
            }
        passed = all(all(group.values()) for group in checks.values())
        report = {
            "decision_steps" if algorithm == "carl" else "updates": update,
            "policy": str(candidate),
            "sha256": file_hash(candidate),
            "highway_passed": passed,
            "checks": checks,
            "comparisons": summaries,
            "nuplan_validation": "pending_manual_by_user",
        }
        reports.append(report)
        if passed:
            eligible.append(
                (
                    sum(row["collision"] for row in results["tail_generated"]),
                    -sum(row["progress_m"] for row in results["natural_generated"]),
                    update,
                    candidate,
                )
            )
    selected = min(eligible)[-1] if eligible else Path(base)
    report = {
        "algorithm": algorithm,
        "selected_policy": str(selected),
        "selected_sha256": file_hash(selected),
        "accepted_update": bool(eligible),
        "candidates": reports,
        "selection_scope": "paired highway development and archive only",
        "nuplan_validation": "manual_by_user_no_automatic_simulation",
    }
    write_json(directory / "admission.json", report)
    return selected, report
