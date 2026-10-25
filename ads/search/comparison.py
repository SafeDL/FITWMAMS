"""Method-blind, common post-hoc branch analysis on the shared first-round pi0.

Diagnostic calls are appended to the owning method's real execution ledger and
cannot influence its already completed search or repair. Unknown graph edges
leave coverage unresolved, rather than create artificial extra domains.
"""

import json
import time

import numpy as np

from ads.common.evaluation import EpisodeResult, WorldEvaluator
from ads.common.instances import GeneratedInstance, InstanceStore
from ads.common.reporting import write_json
from ads.common.setup import load_policy
from ads.search.branches import HillValleyPartitioner, midpoint


def common_discovery_partition(campaign, algorithm, seed):
    target = campaign.output / "common_discovery" / algorithm / f"{seed}.json"
    if target.exists():
        return json.loads(target.read_text())
    started = time.perf_counter()
    methods = campaign.config["methods"]
    journals, summaries, directories, populations, known = {}, {}, {}, {}, {}
    for method in methods:
        directory = campaign.output / algorithm / str(seed) / method / "round_1"
        directories[method] = directory
        rows = [
            json.loads(line)
            for line in (directory / "search.jsonl").read_text().splitlines()
        ]
        journals[method] = rows
        summaries[method] = json.loads((directory / "search_summary.json").read_text())
        results = []
        for row in rows:
            instance = GeneratedInstance.load(
                directory / "search_instances" / (row["instance_hash"] + ".npz")
            )
            result = EpisodeResult(
                instance, row, np.empty((0, 7, 6)), np.empty((0, 2)), []
            )
            known[instance.instance_hash] = result
            if row["execution_kind"] not in ("branch_check", "archive_replay"):
                results.append(result)
        populations[method] = results
    # Equal identities are the same function evaluation under the common pi0 and
    # policy stream. Only actual failures define this terminal-set partition;
    # safe points must not become graph bridges between failure regions.
    pooled = {
        r.instance.instance_hash: r
        for results in populations.values()
        for r in results
        if r.failure
    }
    source = campaign.source("search", "common_partition", algorithm, seed)
    policy = None

    def check(left, right):
        nonlocal policy
        if campaign.config["search"].get("fixed_budget", False) and all(
            len(journals[method]) >= campaign.config["search"]["budget"]
            for method in methods
        ):
            return None
        check_started = time.perf_counter()
        generation_before = source.generation_elapsed_s
        instance = source.mutation(
            left.instance, midpoint(left.instance, right.instance), "branch_check"
        )
        if instance.instance_hash in known:
            return known[instance.instance_hash]
        available = [
            method
            for method in methods
            if len(journals[method]) < campaign.config["search"]["budget"]
            and summaries[method]["execution_costs"].get("branch_check", 0)
            < campaign.config["search"]["branch_check_budget"]
        ]
        if not available:
            return None
        owner = min(
            available,
            key=lambda name: (
                summaries[name]["execution_costs"].get("branch_check", 0),
                name,
            ),
        )
        if policy is None:
            path = campaign.checkpoint(
                campaign.output / algorithm / str(seed) / "baseline", algorithm
            )
            policy = load_policy(algorithm, path, device=campaign.device)
        result = WorldEvaluator(
            policy,
            algorithm,
            campaign.factory,
            campaign.evt,
            ledger=campaign.ledger,
            seed=seed,
        )(instance)
        result.record.update(
            execution_kind="branch_check",
            branch_id="common_diagnostic",
            independent_sampling_root=False,
            diagnostic_endpoints=[
                left.instance.instance_hash,
                right.instance.instance_hash,
            ],
            evaluation_index=len(journals[owner]) + 1,
            search_batch=-1,
            common_posthoc=True,
        )
        store = InstanceStore(
            directories[owner] / "search_instances",
            purpose="search",
            ledger=campaign.ledger,
        )
        store.add(instance)
        journals[owner].append(result.record)
        summary = summaries[owner]
        summary["evaluations"] += 1
        summary["execution_costs"]["branch_check"] = (
            summary["execution_costs"].get("branch_check", 0) + 1
        )
        summary["unused_budget"] -= 1
        summary["execution_elapsed_s"] += result.record["elapsed_s"]
        summary["generation_elapsed_s"] += (
            source.generation_elapsed_s - generation_before
        )
        summary["total_search_elapsed_s"] += time.perf_counter() - check_started
        known[instance.instance_hash] = result
        return result

    partition, evidence = HillValleyPartitioner(
        graph_size=campaign.config["search"]["graph_size"], seed=seed
    ).split(list(pooled.values()), check)
    resolved = evidence is not None and evidence["status"] != "incomplete"
    report = {
        "schema": "method_blind_common_failure_partition_v2",
        "partition_population": "executed_ego_failures_only",
        "unique_partition_instances": len(pooled),
        "resolved": resolved,
        "evidence": evidence,
        "partition": partition.as_dict() if partition else None,
        "coverage_interpretation": "common empirical partition; no topology completeness claim",
        "analysis_elapsed_s": time.perf_counter() - started,
        "generation_elapsed_s": source.generation_elapsed_s,
    }
    for method, results in populations.items():
        labels = (
            partition.predict([r.instance for r in results])
            if partition
            else np.zeros(len(results), int)
        )
        by_hash = {
            r.instance.instance_hash: int(label) for r, label in zip(results, labels)
        }
        for row in journals[method]:
            row["common_partition_resolved"] = resolved
            if (
                row["instance_hash"] in by_hash
                and row["execution_kind"] != "branch_check"
            ):
                row["common_branch_id"] = (
                    f"common.{by_hash[row['instance_hash']]}" if resolved else None
                )
        (directories[method] / "search.jsonl").write_text(
            "".join(json.dumps(row, allow_nan=False) + "\n" for row in journals[method])
        )
        summaries[method]["common_partition_resolved"] = resolved
        write_json(directories[method] / "search_summary.json", summaries[method])
    write_json(target, report)
    return report
