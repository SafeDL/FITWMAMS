"""Capability-preserving DP training and paired highway candidate admission."""

import json
from pathlib import Path

import numpy as np

from ads.common.admission import admission
from ads.common.instances import GeneratedInstance, file_hash
from ads.common.reporting import write_json
from ads.common.setup import REPO_ROOT, load_policy
from ads.diffusion_planner.prepare_data import load_training_samples
from ads.diffusion_planner.repair import train
from ads.nuplan.assets import official_path
from ads.nuplan.teacher import demonstration

ROOT = REPO_ROOT / "results/ads/history/two_round_repair/materials/diffusion_planner"


def load_keep_samples():
    manifest = json.loads((ROOT / "instances.json").read_text())
    expected_policy = file_hash(official_path("diffusion_planner"))
    samples = []
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
    if not samples:
        raise ValueError(
            "Official policy supplied no qualified highway retention material"
        )
    return samples


def repair_and_select(
    campaign, model, banks, *, base, directory, updates, seed, round_index, progress
):
    directory = Path(directory)
    manifest = json.loads((ROOT / "instances.json").read_text())
    if manifest["assets"]["tail"] != campaign.assets:
        raise ValueError("Retention and search use different frozen worlds")
    native_path = campaign.config["capability_preserving_repair"]["native_materials"]
    native, keep = load_training_samples(native_path), load_keep_samples()
    candidates = []

    def save(policy, update):
        path = directory / "candidates" / str(update) / "policy.pth"
        policy.save(path)
        candidates.append((update, path))
        write_json(path.parent / "training_state.json", policy.repair_state)

    report = train(
        model,
        {**banks, "native": native, "keep": keep},
        updates=updates,
        seed=seed,
        round_index=round_index,
        micro_batch_size=campaign.config["repair"]["plant_micro_batch_size"],
        learning_rate=campaign.config["repair"]["plant_learning_rate"],
        retention_coefficient=campaign.config["capability_preserving_repair"][
            "retention_coefficient"
        ],
        checkpoint=save,
        progress=lambda done, total, losses: progress(done, total),
    )
    report["retention_materials"] = {
        "native_sha256": file_hash(native_path),
        "highway_sha256": file_hash(ROOT / "instances.json"),
        "native_label_frames": len(native),
        "highway_label_frames": len(keep),
    }
    write_json(directory / "optimization.json", report)
    if not candidates:
        report["admission"] = {
            "accepted_update": False,
            "reason": "no_qualified_demonstrations",
            "selected_policy": str(base),
        }
        return load_policy("diffusion_planner", base, device=campaign.device), report
    selected, outcome = admission(
        candidates,
        base,
        campaign,
        directory,
        algorithm="diffusion_planner",
        official=official_path("diffusion_planner"),
        material_root=ROOT,
        development=manifest["development"],
        training_instances=[
            row["instance_hash"]
            for row in json.loads((directory / "teacher_selection.json").read_text())
        ],
    )
    report["admission"] = outcome
    return load_policy("diffusion_planner", selected, device=campaign.device), report
