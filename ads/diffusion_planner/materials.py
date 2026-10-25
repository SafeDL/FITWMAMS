"""Build highway retention and validation scenes before any DP repair learning."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from ads.common.batch_evaluation import evaluate_paths
from ads.common.evaluation_sampling import named_stream
from ads.common.instances import (
    GeneratedSource,
    InstanceStore,
    UsageLedger,
    file_hash,
    generate_to_store,
)
from ads.common.natural import empirical_instances, empirical_manifest
from ads.common.reporting import write_json
from ads.common.setup import REPO_ROOT, load_world
from ads.nuplan.assets import official_path
from ads.nuplan.teacher import demonstration

ROOT = REPO_ROOT / "results/risk_stratified_repair/materials/diffusion_planner"


def generated_pool(sampler, assets, *, purpose, name, count, settings, kind):
    ledger = UsageLedger(ROOT / "usage.sqlite")
    store = InstanceStore(ROOT / "instances" / name, purpose=purpose, ledger=ledger)
    if len(store) > count:
        raise ValueError("Highway material pool exceeds its fixed size")
    source = GeneratedSource(
        sampler,
        assets,
        master_seed=settings["master_seed"],
        purpose=purpose,
        stream=named_stream("capability_materials_v1", name),
        source_kind=kind,
    )
    source.index = len(store)
    generate_to_store(
        source, store, count - len(store), settings["runtime"]["generation_batch_size"]
    )
    return sorted(store.paths)


def prepare(config_path, workers=4):
    settings = yaml.safe_load(Path(config_path).read_text())
    protocol = settings["capability_preserving_repair"]
    empirical_count = protocol["keep_candidates"] // 2
    torch.set_num_threads(settings["runtime"]["torch_threads"])
    ROOT.mkdir(parents=True, exist_ok=True)
    tail, _, _, tail_assets = load_world(settings)
    natural, _, _, natural_assets = load_world(
        settings, distribution="natural_generated"
    )
    policy = official_path("diffusion_planner")
    keep = generated_pool(
        natural,
        natural_assets,
        purpose="repair",
        name="keep_natural_generated",
        count=protocol["keep_candidates"] - empirical_count,
        settings=settings,
        kind="natural_generated",
    )
    sampling = settings["evaluation"]
    manifest = ROOT / "empirical_keep.csv"
    with np.load(REPO_ROOT / sampling["natural_dataset"], allow_pickle=False) as data:
        arrays = {
            key: data[key] for key in ("segment_id", "split_index", "is_evt_tail")
        }
    frame = empirical_manifest(
        pd.read_csv(REPO_ROOT / sampling["natural_segments"]),
        arrays,
        count=empirical_count,
        seed=named_stream("highway_retention_empirical"),
        split_indices=(0, 1),
    )
    if manifest.exists() and manifest.read_text() != frame.to_csv(index=False):
        raise ValueError("Empirical retention source selection changed")
    manifest.write_text(frame.to_csv(index=False))
    store = InstanceStore(
        ROOT / "instances/keep_empirical",
        purpose="repair",
        ledger=UsageLedger(ROOT / "usage.sqlite"),
    )
    if len(store) < empirical_count:
        for instance in empirical_instances(
            manifest,
            REPO_ROOT / sampling["sequence_cache"],
            tail_assets,
            history_config=REPO_ROOT / "process_highD/config.yaml",
            purpose="repair",
            namespace="highway_retention_empirical",
            seed=settings["master_seed"],
        ):
            store.add(instance)
    keep += sorted(store.paths)
    development = {
        "tail_generated": generated_pool(
            tail,
            tail_assets,
            purpose="development",
            name="development_tail",
            count=protocol["development_tail"],
            settings=settings,
            kind="tail_generated",
        ),
        "natural_generated": generated_pool(
            natural,
            natural_assets,
            purpose="development",
            name="development_natural",
            count=protocol["development_natural"],
            settings=settings,
            kind="natural_generated",
        ),
    }
    write_json(
        ROOT / "instances.json",
        {
            "keep": [str(p) for p in keep],
            "development": {
                name: [str(p) for p in paths] for name, paths in development.items()
            },
            "assets": {"tail": tail_assets, "natural": natural_assets},
            "nuplan_validation": "manual_by_user",
        },
    )
    # Release generator GPU state before spawning inference workers.
    del tail, natural
    torch.cuda.empty_cache()
    retain = evaluate_paths(
        keep,
        policy,
        ROOT / "initial/keep",
        settings,
        purpose="repair",
        workers=workers,
        ledger_path=ROOT / "usage.sqlite",
        save_trace=True,
    )
    for name, paths in development.items():
        evaluate_paths(
            paths,
            policy,
            ROOT / "initial" / name,
            settings,
            workers=workers,
            ledger_path=ROOT / "usage.sqlite",
        )
    report = {
        "status": "passed",
        "keep_candidates": len(keep),
        "qualified_keep_instances": sum(bool(e.get("trajectory")) for e in retain),
        "development_tail": protocol["development_tail"],
        "development_natural": protocol["development_natural"],
        "nuplan_simulation_performed": False,
    }
    write_json(ROOT / "materials.json", report)
    print(f"Highway material preparation complete: {ROOT}", flush=True)
    return report


def load_keep_samples(limit=None):
    from ads.common.instances import GeneratedInstance

    manifest = json.loads((ROOT / "instances.json").read_text())
    expected_policy = file_hash(official_path("diffusion_planner"))
    samples, scenes = [], 0
    for path in manifest["keep"]:
        instance = GeneratedInstance.load(path)
        entry = json.loads(
            (ROOT / "initial/keep" / (instance.instance_hash + ".json")).read_text()
        )
        if not entry.get("trajectory"):
            continue
        if (
            entry["policy_sha256"] != expected_policy
            or entry["record"]["instance_hash"] != instance.instance_hash
            or entry["record"]["safety_failure"]
            or entry["record"]["stalled"]
            or file_hash(entry["trajectory"]) != entry["trajectory_sha256"]
        ):
            raise ValueError("Highway retention provenance or success check differs")
        with np.load(entry["trajectory"], allow_pickle=False) as data:
            produced = demonstration(instance, data["states"], "diffusion_planner")
        for sample in produced:
            sample.update(
                scene_id=instance.root_sample_id,
                domain="keep",
                source=instance.source_kind,
            )
        samples.extend(produced)
        scenes += 1
        if limit is not None and scenes == limit:
            break
    if not samples:
        raise ValueError(
            "Official policy supplied no qualified highway retention material"
        )
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "ads/configs/diffusion_planner_pilot.yaml",
    )
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    prepare(args.config, args.workers)


if __name__ == "__main__":
    main()
