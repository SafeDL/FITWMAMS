"""Build the successful 6500-scene bank and two independent development cohorts."""

import json
import time

import numpy as np
import torch
import yaml

from ads.common.evaluation_sampling import named_stream
from ads.common.instances import (
    GeneratedInstance,
    GeneratedSource,
    InstanceStore,
    UsageLedger,
    file_hash,
)
from ads.common.reporting import write_json
from ads.common.setup import REPO_ROOT, load_policy, load_world
from ads.diffusion_planner.expert_materials import CONFIG
from ads.nuplan.assets import official_path
from ads.nuplan.teacher import demonstration


def verify_initial(source, output):
    official = load_policy(
        "diffusion_planner", official_path("diffusion_planner"), device="cpu"
    )
    searched = load_policy("diffusion_planner", source["initial_policy"], device="cpu")
    for key, value in official.network.state_dict().items():
        if not torch.equal(value, searched.network.state_dict()[key]):
            raise ValueError(
                "Imported search did not use the official initial deployment"
            )
    write_json(
        output / "initial_identity.json",
        {
            "official_sha256": file_hash(official_path("diffusion_planner")),
            "searched_sha256": source["initial_sha256"],
            "strict_state_equality": True,
            "tensors": len(official.network.state_dict()),
        },
    )


def generate_pool(settings, output, *, name, kind, purpose, count):
    started = time.perf_counter()
    sampler, _, _, assets = load_world(settings, distribution=kind)
    ledger = UsageLedger(output / "usage.sqlite")
    store = InstanceStore(output / "instances" / name, purpose=purpose, ledger=ledger)
    if len(store) > count:
        raise ValueError("Generated cohort exceeds the frozen requested size")
    source = GeneratedSource(
        sampler,
        assets,
        master_seed=settings["master_seed"],
        purpose=purpose,
        stream=named_stream(settings["run_kind"], name),
        source_kind=kind,
    )
    source.index = len(store)
    before = len(store)
    for start in range(before, count, settings["runtime"]["generation_batch_size"]):
        for instance in source.sample(
            min(settings["runtime"]["generation_batch_size"], count - start)
        ):
            store.add(instance)
        print(f"Generate {name}: {len(store)}/{count}", flush=True)
    path = output / "generation" / (name + ".json")
    previous = json.loads(path.read_text()) if path.exists() else {}
    write_json(
        path,
        {
            "instances": count,
            "assets": assets,
            "new_instances_total": previous.get("new_instances_total", 0)
            + count
            - before,
            "elapsed_s_total": previous.get("elapsed_s_total", 0.0)
            + time.perf_counter()
            - started,
        },
    )
    del sampler
    torch.cuda.empty_cache()
    return sorted(store.paths), assets


def qualified(entry):
    row = entry["record"]
    return not (row["safety_failure"] or row["stalled"]) and bool(
        entry.get("trajectory")
    )


def prepare(settings):
    output = REPO_ROOT / settings["output_dir"]
    torch.set_num_threads(settings["runtime"]["torch_threads"])
    source = json.loads((output / "sources.json").read_text())
    verify_initial(source, output)
    old = REPO_ROOT / settings["historical_run"]
    previous = json.loads((old / "retention_materials.json").read_text())
    if previous["assets"]["tail_generated"] != source["assets"]:
        raise ValueError(
            "Historical success/development world differs from the new search"
        )
    protected = set(source["protected_roots"])
    ledger = UsageLedger(output / "usage.sqlite")
    entries = {}
    for entry in previous["keep"]:
        instance = GeneratedInstance.load(entry["instance_path"])
        if instance.root_sample_id not in protected and qualified(entry):
            if (
                entry["policy_sha256"] != source["initial_sha256"]
                or file_hash(entry["trajectory"]) != entry["trajectory_sha256"]
            ):
                raise ValueError("Historical success changed")
            entries[instance.instance_hash] = entry
    search_roots = [old / "search", REPO_ROOT / settings["reuse_search"]]
    for root in search_roots:
        protocol = json.loads((root / "protocol.json").read_text())
        if protocol["assets"] != source["assets"]:
            raise ValueError("Success search comes from different frozen assets")
        for directory in sorted(
            (root / "diffusion_planner" / str(settings["master_seed"])).glob(
                "*/round_1"
            )
        ):
            # Search success traces are training material; development and final traces are excluded.
            for record_path in sorted((directory / "executions").glob("*.json")):
                row = json.loads(record_path.read_text())
                record = row["record"]
                identity = record["instance_hash"]
                if (
                    record["safety_failure"]
                    or record["stalled"]
                    or record["root_sample_id"] in protected
                    or identity in entries
                ):
                    continue
                instance_path = directory / "search_instances" / (identity + ".npz")
                trajectory = record_path.with_suffix(".npz")
                if (
                    row["policy_sha256"] != source["initial_sha256"]
                    or file_hash(trajectory) != row["trace_sha256"]
                ):
                    raise ValueError("Search success trace or policy changed")
                entries[identity] = {
                    "record": record,
                    "policy_sha256": row["policy_sha256"],
                    "trajectory": str(trajectory.resolve()),
                    "trajectory_sha256": row["trace_sha256"],
                    "instance_path": str(instance_path.resolve()),
                    "historical_reuse": root == search_roots[0],
                }
    # Reused initial priors are stored as instances but their traces remain in the verified historical search.
    development = previous["development"]
    for name, paths in development.items():
        if len(paths) != settings["materials"]["development_instances"]:
            raise ValueError("Development cohort has a different frozen size")
        destination = output / "initial" / name
        destination.mkdir(parents=True, exist_ok=True)
        for path in paths:
            instance = GeneratedInstance.load(path)
            ledger.register(instance, "development")
            record_path = old / "initial" / name / (instance.instance_hash + ".json")
            entry = json.loads(record_path.read_text())
            if (
                entry["policy_sha256"] != source["initial_sha256"]
                or entry["record"]["instance_hash"] != instance.instance_hash
            ):
                raise ValueError("Initial development provenance differs")
            target = destination / record_path.name
            if not target.exists():
                target.write_bytes(record_path.read_bytes())
    for entry in entries.values():
        ledger.register(GeneratedInstance.load(entry["instance_path"]), "repair")
    report = {
        "status": "complete",
        "keep": [entries[key] for key in sorted(entries)],
        "development": development,
        "assets": previous["assets"],
        "keep_count": len(entries),
        "protected_roots": sorted(protected),
        "historical_materials_sha256": file_hash(old / "materials.json"),
        "all_qualified_successes_included": True,
        "new_world_executions": 0,
    }
    write_json(output / "materials.json", report)
    print(
        f"All successful highway inputs: {len(entries)} (no 6500 truncation)",
        flush=True,
    )
    return report


class KeepSamples:
    """Decode only requested frames while all successful frames remain in the pool."""

    def __init__(self, report):
        self.entries, self.metadata, self.indices = report["keep"], [], []
        protected = set(report["protected_roots"])
        for index, entry in enumerate(self.entries):
            instance = GeneratedInstance.load(entry["instance_path"])
            if (
                instance.root_sample_id in protected
                or instance.purpose in {"development", "final_test"}
                or (
                    instance.source_kind == "empirical_natural"
                    and instance.source_metadata["source_split"] not in {"train", "val"}
                )
                or not qualified(entry)
                or file_hash(entry["trajectory"]) != entry["trajectory_sha256"]
            ):
                raise ValueError("Keep material purpose, safety or provenance changed")
            with np.load(entry["trajectory"], allow_pickle=False) as data:
                count = len(data["states"])
            for frame in range(0, count - 3, 5):
                self.indices.append((index, frame))
                self.metadata.append(
                    {
                        "scene_id": instance.root_sample_id,
                        "instance_id": instance.instance_hash,
                        "frame_id": frame,
                        "domain": "keep",
                        "label_source": "policy_success",
                        "purpose": "repair",
                        "source_kind": instance.source_kind,
                        "policy_random_mode": entry["record"]["policy_random_mode"],
                        "policy_stream_id": entry["record"]["policy_stream_id"],
                        "reference_risk": float(entry["record"]["event_risk"]),
                    }
                )

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        entry_index, frame = self.indices[index]
        entry = self.entries[entry_index]
        instance = GeneratedInstance.load(entry["instance_path"])
        with np.load(entry["trajectory"], allow_pickle=False) as data:
            produced = demonstration(
                instance, data["states"], "diffusion_planner", frames=[frame]
            )[0]
        produced.update(self.metadata[index])
        return produced


if __name__ == "__main__":
    prepare(yaml.safe_load(CONFIG.read_text()))
