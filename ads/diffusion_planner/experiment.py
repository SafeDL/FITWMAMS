"""Capability-preserving DP training and paired highway candidate admission."""

import json
from pathlib import Path

from ads.common.admission import admission
from ads.common.instances import file_hash
from ads.common.reporting import write_json
from ads.common.setup import load_policy
from ads.diffusion_planner.materials import ROOT, load_keep_samples
from ads.diffusion_planner.prepare_data import load_training_samples
from ads.diffusion_planner.repair import train
from ads.nuplan.assets import official_path


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
