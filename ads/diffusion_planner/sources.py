"""Import the complete first search and protect shared repair-validation ancestry."""

import json
from collections import Counter

import numpy as np

from ads.common.instances import GeneratedInstance, UsageLedger, file_hash, json_hash
from ads.common.reporting import write_json


def executed_success_neighbours(records, paths):
    neighbours = {}
    for identity, row in records.items():
        parent = records.get(row["parent_id"])
        if (
            parent is None
            or row["safety_failure"] == parent["safety_failure"]
            or row["execution_kind"] == "branch_check"
        ):
            continue
        child = GeneratedInstance.load(paths[identity])
        previous = GeneratedInstance.load(paths[parent["instance_hash"]])
        if child.mutation_block is None or not np.array_equal(
            child.event.valid, previous.event.valid
        ):
            continue
        failure, success = (row, parent) if row["safety_failure"] else (parent, row)
        neighbours.setdefault(failure["instance_hash"], success["instance_hash"])
    return neighbours


def import_search(settings, repo_root, output):
    source = repo_root / settings["reuse_search"]
    protocol = json.loads((source / "protocol.json").read_text())
    manifest_path = source / "world_source_manifest.json"
    if not manifest_path.exists():
        raise ValueError("Source search lacks a verifiable frozen world implementation")
    manifest = json.loads(manifest_path.read_text())
    if json_hash(manifest) != protocol["assets"]["world_implementation"] or any(
        file_hash(repo_root / name) != digest for name, digest in manifest.items()
    ):
        raise ValueError("Search and current frozen execution source differ")
    initial = (
        source
        / "diffusion_planner"
        / str(settings["master_seed"])
        / "baseline/policy.pth"
    )
    methods, by_hash, paths, discovery = {}, {}, {}, {}
    journal_hashes, execution_counts = {}, {}
    shared_inputs = 0
    for method in settings["methods"]:
        directory = initial.parent.parent / method / "round_1"
        summary = json.loads((directory / "search_summary.json").read_text())
        records = [
            json.loads(line)
            for line in (directory / "search.jsonl").read_text().splitlines()
        ]
        seen_collisions, seen_ancestors, curve = set(), set(), []
        for index, row in enumerate(records, 1):
            if (
                row["execution_kind"] not in ("branch_check", "archive_replay")
                and row["collision"]
            ):
                seen_collisions.add(row["instance_hash"])
                seen_ancestors.add(row["root_sample_id"])
            if index % 50 == 0 or index == len(records):
                curve.append(
                    {
                        "executions": index,
                        "distinct_collisions": len(seen_collisions),
                        "collision_ancestors": len(seen_ancestors),
                    }
                )
        discovery[method] = {"curve": curve, "collision_ancestors": len(seen_ancestors)}
        if not 0 < len(records) == summary["evaluations"] <= settings["search"][
            "budget"
        ] or summary["policy_sha256"] != file_hash(initial):
            raise ValueError(
                "Search count or initial policy differs from the recorded protocol"
            )
        unique = {
            row["instance_hash"]: row
            for row in records
            if row["execution_kind"] not in ("branch_check", "archive_replay")
        }
        for identity, row in unique.items():
            if identity in by_hash:
                previous = by_hash[identity]
                if any(
                    previous[field] != row[field]
                    for field in (
                        "collision",
                        "offroad",
                        "stalled",
                        "event_risk",
                        "progress_m",
                        "physics_steps",
                        "policy_stream_id",
                    )
                ):
                    raise ValueError(
                        "Common pi0 input has inconsistent replay outcomes across methods"
                    )
                shared_inputs += 1
            path = directory / "search_instances" / (identity + ".npz")
            instance = GeneratedInstance.load(path)
            if (
                instance.assets != protocol["assets"]
                or instance.root_sample_id != row["root_sample_id"]
            ):
                raise ValueError("Search instance asset or ancestor mismatch")
            if row["policy_random_mode"] != "search":
                raise ValueError(
                    "Search did not use the shared fixed policy random stream"
                )
            by_hash.setdefault(identity, row)
            paths.setdefault(identity, str(path.resolve()))
        methods[method] = unique
        journal_hashes[method] = file_hash(directory / "search.jsonl")
        execution_counts[method] = len(records)
    historical = repo_root / settings["historical_run"] / "experts/report.json"
    protected = json.loads(historical.read_text())["protected_roots"]
    ledger = UsageLedger(output / "usage.sqlite")
    ledger.protect_roots(protected)
    neighbours = executed_success_neighbours(by_hash, paths)
    payload = {
        "schema": "capability_repair_sources",
        "seed": settings["master_seed"],
        "initial_policy": str(initial.resolve()),
        "initial_sha256": file_hash(initial),
        "source_protocol_sha256": file_hash(source / "protocol.json"),
        "search_journals": journal_hashes,
        "shared_primary_replays_verified": shared_inputs,
        "assets": protocol["assets"],
        "protected_roots": protected,
        "paths": paths,
        "records": by_hash,
        "success_neighbours": neighbours,
        "methods": {},
    }
    for method, records in methods.items():
        failures = [r for r in records.values() if r["safety_failure"]]
        eligible = [r for r in failures if r["root_sample_id"] not in protected]
        payload["methods"][method] = {
            "executions": execution_counts[method],
            "collision_inputs": sum(r["collision"] for r in failures),
            "failure_inputs": len(failures),
            "eligible_failure_inputs": [r["instance_hash"] for r in eligible],
            "root_counts": dict(Counter(r["root_sample_id"] for r in eligible)),
            "discovery": discovery[method],
        }
    path = output / "sources.json"
    if path.exists() and json_hash(json.loads(path.read_text())) != json_hash(payload):
        raise ValueError(
            "Frozen failure pool or dangerous development selection changed"
        )
    write_json(path, payload)
    return payload
