"""Persist the exact ordered scene banks used to reset each repair learner."""

import json
from pathlib import Path

from ads.common.instances import GeneratedInstance, InstanceStore, file_hash
from ads.common.reporting import write_json
from ads.search.archive import FailureArchive
from ads.tasks import SUPERVISED_ALGORITHMS


def build_repair_banks(algorithm, normal, records, previous, known, *, limit=32):
    records = [
        r
        for r in records
        if r.record.get("execution_kind") not in ("branch_check", "archive_replay")
    ]
    previous_ids = {r.instance.instance_hash for r in previous}
    failures = [
        r for r in records if r.failure and r.instance.instance_hash not in previous_ids
    ]
    # Select distinct learning roots before truncating the candidate pool.
    # This pool is separate from the bounded online search archive.
    fresh = FailureArchive(capacity=limit, root_limit=1)
    fresh.add(
        [
            r
            for r in records
            if not r.failure or r.instance.instance_hash not in previous_ids
        ],
        known,
    )
    selected = fresh.repair_representatives(limit=len(failures))
    banks = {
        "normal": [(i, "prior") for i in normal],
        "new_failure": [(r.instance, r.domain) for r in selected],
        "history": [(r.instance, r.domain) for r in previous],
    }
    if algorithm in SUPERVISED_ALGORITHMS:
        domains = {r.instance.instance_hash: r.domain for r in selected}
        banks["success_neighbours"] = {
            key: (value.instance, domains[key])
            for key, value in fresh.neighbours.items()
            if key in domains
        }
    else:
        banks["new_failure"] += [
            (r.instance, r.domain) for r in fresh.neighbours.values()
        ]
    return banks


def persist_curriculum(directory, banks, ledger, *, teacher_history=()):
    directory = Path(directory)
    store = InstanceStore(
        directory / "inputs" / "instances", purpose="repair", ledger=ledger
    )
    payload = {"schema": "ads_curriculum_v1", "categories": {}}
    for name in ("normal", "new_failure", "history"):
        entries = banks.get(name, [])
        payload["categories"][name] = []
        for instance, domain in entries:
            store.add(instance)
            payload["categories"][name].append(
                {"instance_hash": instance.instance_hash, "domain": domain}
            )
    payload["success_neighbours"] = {}
    for failure_hash, (instance, domain) in banks.get("success_neighbours", {}).items():
        store.add(instance)
        payload["success_neighbours"][failure_hash] = {
            "instance_hash": instance.instance_hash,
            "domain": domain,
        }
    payload["teacher_history"] = []
    for instance, domain in teacher_history:
        store.add(instance)
        payload["teacher_history"].append(
            {"instance_hash": instance.instance_hash, "domain": domain}
        )
    path = directory / "inputs" / "curriculum.json"
    if path.exists():
        if json.loads(path.read_text()) != payload:
            raise ValueError(
                "stored training curriculum differs from the requested scene banks"
            )
    else:
        write_json(path, payload)
    return file_hash(path)


def load_curriculum(directory, ledger):
    directory = Path(directory)
    payload = json.loads((directory / "inputs/curriculum.json").read_text())

    def load(entry):
        instance = GeneratedInstance.load(
            directory / "inputs/instances" / (entry["instance_hash"] + ".npz")
        )
        ledger.register(instance, "repair")
        return instance, entry["domain"]

    banks = {
        name: [load(entry) for entry in entries]
        for name, entries in payload["categories"].items()
    }
    banks["success_neighbours"] = {
        key: load(entry) for key, entry in payload["success_neighbours"].items()
    }
    return banks, [load(entry) for entry in payload["teacher_history"]]
