"""Method-blind geometric coverage of executed collision states; no world calls."""

import json
from pathlib import Path

import numpy as np

from ads.common.reporting import write_json


def motion_catalog(journals, radius):
    pooled = {}
    for rows in journals.values():
        for row in rows:
            value = np.asarray(row.get("descriptor", [])[:4], float)
            if (
                row.get("collision", False)
                and value.shape == (4,)
                and np.isfinite(value).all()
                and row.get("critical_slot", -1) > 0
            ):
                pooled[row["instance_hash"]] = row
    keys = sorted(pooled)
    if not keys:
        return {
            "representatives": [],
            "methods": {
                name: {"covered_states": 0, "assignments": {}} for name in journals
            },
        }
    values = np.asarray([pooled[key]["descriptor"][:4] for key in keys], float)
    selected = [0]
    distance = np.linalg.norm(values - values[0], axis=1)
    while float(distance.max()) > radius:
        index = int(np.argmax(distance))
        selected.append(index)
        distance = np.minimum(distance, np.linalg.norm(values - values[index], axis=1))
    labels = np.linalg.norm(values[:, None] - values[selected][None], axis=-1).argmin(
        axis=1
    )
    assignments = dict(zip(keys, labels.tolist()))
    methods = {}
    for name, rows in journals.items():
        actual = {
            row["instance_hash"]: assignments[row["instance_hash"]]
            for row in rows
            if row["instance_hash"] in assignments
        }
        methods[name] = {
            "covered_states": len(set(actual.values())),
            "assignments": actual,
        }
    return {"representatives": [keys[index] for index in selected], "methods": methods}


def common_discovery_partition(campaign, algorithm, seed):
    target = campaign.output / "common_discovery" / algorithm / f"{seed}.json"
    journals, summaries, directories = {}, {}, {}
    methods = list(campaign.config["methods"]) + list(
        campaign.config.get("search_controls", [])
    )
    for method in methods:
        directory = campaign.output / algorithm / str(seed) / method / "round_1"
        directories[method] = directory
        journals[method] = [
            json.loads(line)
            for line in (directory / "search.jsonl").read_text().splitlines()
        ]
        summaries[method] = json.loads((directory / "search_summary.json").read_text())
    radii = [summary["collision_coverage"]["radius"] for summary in summaries.values()]
    if len(set(radii)) != 1:
        raise ValueError("Methods do not share the same prior motion-state calibration")
    radius = radii[0]
    report = {
        "schema": "method_blind_motion_catalog",
        "radius": radius,
        "world_executions": 0,
        "resolved": True,
        "coverage_interpretation": "empirical_motion_states_not_topological_domains",
        **motion_catalog(journals, radius),
        "radius_sensitivity": (
            {
                str(factor): motion_catalog(journals, radius * factor)
                for factor in (0.5, 2.0)
            }
            if radius
            else {}
        ),
    }
    for method, rows in journals.items():
        assignments = report["methods"][method]["assignments"]
        for row in rows:
            row["common_partition_resolved"] = True
            row["common_branch_id"] = (
                f"motion.{assignments[row['instance_hash']]}"
                if row["instance_hash"] in assignments
                else None
            )
        (directories[method] / "search.jsonl").write_text(
            "".join(json.dumps(row, allow_nan=False) + "\n" for row in rows)
        )
    write_json(target, report)
    return report


def compare_with_history(settings, output):
    """Recompute a common physical-deduplicated catalog without new world calls."""
    import hashlib
    from collections import Counter

    historical = Path(settings["historical_run"]) / "search"
    current = Path(settings["reuse_search"])
    journals, directories, summaries = {}, {}, {}
    for name, root, method in (
        ("monte_carlo", historical, "monte_carlo"),
        ("subset", historical, "subset"),
        ("guided", current, "guided"),
    ):
        directory = (
            root
            / "diffusion_planner"
            / str(settings["master_seed"])
            / method
            / "round_1"
        )
        directories[name] = directory
        rows = [
            json.loads(line)
            for line in (directory / "search.jsonl").read_text().splitlines()
        ]
        for row in rows:
            if not row.get("collision") or row.get("trajectory_hash"):
                continue
            trace = directory / "executions" / f"{row['evaluation_index']:05d}.npz"
            digest = hashlib.sha256()
            with np.load(trace, allow_pickle=False) as data:
                for key in ("states", "controls"):
                    value = np.ascontiguousarray(data[key])
                    digest.update(str((value.shape, value.dtype.str)).encode())
                    digest.update(value.tobytes())
            row["trajectory_hash"] = digest.hexdigest()
        journals[name] = rows
        summaries[name] = json.loads((directory / "search_summary.json").read_text())
    radius = summaries["guided"]["collision_coverage"]["radius"]
    if any(
        not np.isclose(summary["collision_coverage"]["radius"], radius)
        for summary in summaries.values()
    ):
        raise ValueError("Historical common prior has a different coverage resolution")
    reports = {}
    for factor in (1.0, 0.5, 2.0):
        catalog = motion_catalog(journals, radius * factor)
        for name, rows in journals.items():
            assignments = catalog["methods"][name]["assignments"]
            physical = {}
            for row in rows:
                if row.get("collision") and row["instance_hash"] in assignments:
                    physical.setdefault(row["trajectory_hash"], row)
            counts = Counter(
                assignments[row["instance_hash"]] for row in physical.values()
            )
            total = sum(counts.values())
            roots = Counter(row["root_sample_id"] for row in physical.values())
            catalog["methods"][name].update(
                executions=len(rows),
                collision_inputs=len(
                    {r["instance_hash"] for r in rows if r.get("collision")}
                ),
                physical_collision_trajectories=total,
                coverage_D2=(
                    total * total / sum(n * n for n in counts.values()) if total else 0
                ),
                collision_ancestors=len(roots),
                maximum_ancestor_fraction=max(roots.values(), default=0)
                / max(total, 1),
                effective_ancestors=(
                    total * total / sum(n * n for n in roots.values()) if total else 0
                ),
            )
        reports[str(factor)] = catalog
    payload = {
        "radius": radius,
        "world_executions_added": 0,
        "historical_methods_rerun": False,
        "comparison_scope": "historical approximate reference; all collision diversity uses physical deduplication",
        "catalogs": reports,
    }
    write_json(output / "search_comparison.json", payload)
    return payload
