"""Reuse only hash-matched executions by the exact checkpoint being diagnosed."""

import json

from ads.common.instances import json_hash
from ads.common.reporting import write_json


def collect_archive_records(
    instances, output, *, policy_sha256, protocol_hash, journals, caches, evaluate
):
    instances = sorted(instances, key=lambda instance: instance.instance_hash)
    cohort = {instance.instance_hash: instance for instance in instances}
    cohort_hash = json_hash(list(cohort))

    def validate(row):
        instance = cohort[row["instance_hash"]]
        if (
            row["event_id"] != instance.instance_hash
            or row["root_sample_id"] != instance.root_sample_id
            or row["assets_hash"] != json_hash(instance.assets)
            or row["purpose"] == "final_test"
        ):
            raise ValueError(
                "archive execution does not match the full generated instance"
            )

    if output.exists():
        payload = json.loads(output.read_text())
        if (
            payload["policy_sha256"] != policy_sha256
            or payload["protocol_hash"] != protocol_hash
            or payload["cohort_hash"] != cohort_hash
            or len(payload["records"]) != len(cohort)
            or {row["instance_hash"] for row in payload["records"]} != set(cohort)
        ):
            raise ValueError(
                "cached archive evaluation uses different policy, protocol or instances"
            )
        for row in payload["records"]:
            validate(row)
        return payload

    known, origins = {}, {}
    for path, checkpoint_hash in journals:
        if checkpoint_hash != policy_sha256:
            continue
        for line in path.read_text().splitlines():
            row = json.loads(line)
            identity = row["instance_hash"]
            if identity in cohort:
                validate(row)
                known[identity], origins[identity] = row, "search"
    for path in caches:
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        if payload["policy_sha256"] != policy_sha256:
            continue
        if payload.get("protocol_hash", protocol_hash) != protocol_hash:
            raise ValueError("archive cache uses a different protocol")
        for row in payload["records"]:
            identity = row["instance_hash"]
            if identity in cohort and identity not in known:
                validate(row)
                known[identity], origins[identity] = row, "cache"

    missing = [
        instance for identity, instance in cohort.items() if identity not in known
    ]
    executed = evaluate(missing) if missing else []
    if len(executed) != len(missing) or {row["instance_hash"] for row in executed} != {
        instance.instance_hash for instance in missing
    }:
        raise ValueError("new archive executions do not cover the missing instances")
    for row in executed:
        validate(row)
        known[row["instance_hash"]] = row
    payload = {
        "schema": "ads_archive_evaluation_v2",
        "policy_sha256": policy_sha256,
        "protocol_hash": protocol_hash,
        "cohort_hash": cohort_hash,
        "records": [known[identity] for identity in cohort],
        "cost": {
            "new_executions": len(executed),
            "new_execution_elapsed_s": sum(row["elapsed_s"] for row in executed),
            "reused_search_records": sum(
                origin == "search" for origin in origins.values()
            ),
            "reused_cached_records": sum(
                origin == "cache" for origin in origins.values()
            ),
        },
    }
    write_json(output, payload)
    return payload


def archive_role_outcomes(archive, before, after):
    before = {row["instance_hash"]: row["safety_failure"] for row in before}
    after = {row["instance_hash"]: row["safety_failure"] for row in after}
    domains = {}
    for domain, failures in archive.domains.items():
        identities = {result.instance.instance_hash for result in failures}
        failed = {identity for identity in identities if before[identity]}
        neighbours = {
            archive.neighbours[identity].instance.instance_hash
            for identity in identities
            if identity in archive.neighbours
        }
        safe = {identity for identity in neighbours if not before[identity]}
        domains[domain] = {
            "pre_repair_failures": len(failed),
            "already_safe_representatives": len(identities - failed),
            "persistent_failures": sum(after[identity] for identity in failed),
            "repaired_failures": sum(not after[identity] for identity in failed),
            "pre_repair_success_neighbours": len(safe),
            "regressed_success_neighbours": sum(after[identity] for identity in safe),
            "neighbours_already_failed_before_repair": len(neighbours - safe),
        }
    return domains
