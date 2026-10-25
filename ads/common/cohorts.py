"""Explicit construction of the three sealed final cohorts; never used in repair."""

import json

import numpy as np
import pandas as pd

from ads.common.evaluation_sampling import COHORTS, final_source, verify_final_roots
from ads.common.instances import InstanceStore, file_hash, generate_to_store, json_hash
from ads.common.natural import (
    empirical_instances,
    empirical_manifest,
    recorded_history_sources,
)
from ads.common.setup import REPO_ROOT, load_world


def prepare_evaluation_assets(campaign):
    settings = campaign.config["evaluation"]
    natural_sampler, _, _, natural_assets = load_world(
        campaign.config, distribution="natural_generated"
    )
    del natural_sampler
    dataset = REPO_ROOT / settings["natural_dataset"]
    segments = REPO_ROOT / settings["natural_segments"]
    with np.load(dataset, allow_pickle=False) as data:
        arrays = {
            key: data[key] for key in ("segment_id", "split_index", "is_evt_tail")
        }
    frame = empirical_manifest(
        pd.read_csv(segments),
        arrays,
        count=settings["instances"],
        seed=campaign.config["master_seed"],
    )
    directory = campaign.output / "evaluation_sources"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "empirical_manifest.csv"
    contents = frame.to_csv(index=False)
    if path.exists() and path.read_text() != contents:
        raise ValueError("sealed empirical selection changed")
    path.write_text(contents)
    roots = [
        json_hash(
            {
                "namespace": "sealed_empirical_natural",
                "seed": campaign.config["master_seed"],
                "source_event_id": event,
            }
        )
        for event in frame.source_event_id
    ]
    files = {
        "natural_dataset": dataset,
        "natural_segments": segments,
        "empirical_manifest": path,
        "natural_flow_checkpoint": REPO_ROOT / settings["natural_flow_checkpoint"],
    }
    files.update(
        recorded_history_sources(
            frame.recording_id.unique(), REPO_ROOT / "process_highD/config.yaml"
        )
    )
    cache = REPO_ROOT / settings["sequence_cache"]
    files.update({"sequence_" + p.stem: p for p in cache.glob("*.npy")})
    return {
        "natural_world": natural_assets,
        "empirical_roots": roots,
        "files": {
            key: {"path": str(path.relative_to(REPO_ROOT)), "sha256": file_hash(path)}
            for key, path in files.items()
        },
        "tail_flow_source_reuse_in_empirical": int(
            frame.source_used_by_tail_flow.sum()
        ),
        "independence": "held out of ADS search/repair; some source events fitted the frozen tail Flow",
    }


def verify_evaluation_assets(assets):
    for entry in assets["files"].values():
        if file_hash(REPO_ROOT / entry["path"]) != entry["sha256"]:
            raise ValueError("final evaluation source changed after policy freeze")


def final_cohort(campaign, cohort, frozen):
    if cohort not in COHORTS:
        raise ValueError("unknown final cohort")
    protocol = json.loads((campaign.output / "protocol.json").read_text())
    directory = campaign.output / "instances" / cohort
    directory.mkdir(parents=True, exist_ok=True)
    import fcntl

    with (directory / "pool.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        store = InstanceStore(directory, purpose="final_test", ledger=campaign.ledger)
        count = campaign.config["evaluation"]["instances"]
        if len(store) > count:
            raise ValueError("final cohort exceeds the sealed size")
        if len(store) < count:
            if cohort == "empirical_natural":
                assets = frozen["evaluation_assets"]
                for instance in empirical_instances(
                    REPO_ROOT / assets["files"]["empirical_manifest"]["path"],
                    REPO_ROOT / campaign.config["evaluation"]["sequence_cache"],
                    {
                        **campaign.assets,
                        "empirical_sources": json_hash(assets["files"]),
                    },
                    history_config=REPO_ROOT
                    / assets["files"]["history_preprocessing"]["path"],
                    seed=campaign.config["master_seed"],
                ):
                    store.add(instance)
            else:
                source = final_source(protocol, cohort, frozen["evaluation_assets"])
                if cohort == "tail_generated":
                    source.sampler = campaign.sampler
                else:
                    source.sampler, _, _, current_assets = load_world(
                        campaign.config, distribution=cohort
                    )
                    if current_assets != frozen["evaluation_assets"]["natural_world"]:
                        raise ValueError("natural generator changed after freeze")
                source.index = len(store)
                needed = count - len(store)
                generate_to_store(
                    source,
                    store,
                    needed,
                    campaign.config["runtime"]["generation_batch_size"],
                )
                with (campaign.output / "generation.jsonl").open("a") as stream:
                    stream.write(
                        json.dumps(
                            {
                                "purpose": "final_test",
                                "pool": cohort,
                                "new_instances": needed,
                                "generation_elapsed_s": source.generation_elapsed_s,
                            }
                        )
                        + "\n"
                    )
        store.paths.sort()
        instances = list(store)
        verify_final_roots(instances, protocol, cohort, frozen["evaluation_assets"])
        return instances
