"""Run one complete DP pilot, preserving final cohorts for the formal experiment."""

import argparse
import fcntl
import json
import os
from pathlib import Path

import torch
import yaml

from ads.common.curriculum import load_curriculum
from ads.common.expert import select_teacher_instances
from ads.common.instances import file_hash
from ads.common.reporting import write_json
from ads.common.setup import REPO_ROOT, load_policy
from ads.diffusion_planner.materials import ROOT, load_keep_samples, prepare
from ads.diffusion_planner.prepare_data import load_training_samples
from ads.diffusion_planner.repair import train
from ads.risk_stratified_repair import Campaign


def engineering(campaign, initial, seed):
    directory = campaign.output / "engineering"
    report_path = directory / "report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text())
        if report["initial_sha256"] != file_hash(initial):
            raise ValueError("Engineering check used another initial policy")
        return report
    search = campaign.output / "diffusion_planner" / str(seed) / "guided/round_1"
    banks, _ = load_curriculum(search, campaign.ledger)
    selected = select_teacher_instances(
        banks.get("new_failure", []), banks.get("success_neighbours", {}), limit=8
    )
    model = load_policy("diffusion_planner", initial, device=campaign.device)
    corrected, teacher_reports = campaign.teacher_samples(
        model, selected, cache_name="failures"
    )
    for sample in corrected:
        sample["label_source"] = "expert"
    if not corrected:
        raise RuntimeError(
            "Pilot search supplied no qualified corrections for engineering validation"
        )
    native = load_training_samples()[:8]
    keep = load_keep_samples(limit=8)
    report = train(
        model,
        {"native": native, "keep": keep, "new_failure": corrected},
        updates=32,
        seed=seed,
        checkpoint=lambda policy, update: policy.save(directory / "policy.pth"),
        progress=lambda done, total, losses: print(
            f"Engineering DP: {done}/{total}, native={losses['native_loss']:.4f}, retain={losses['retention_loss']:.6f}",
            flush=True,
        ),
    )
    native_path = directory / "native/model.pth"
    model.export_native(native_path)
    readback = load_policy("diffusion_planner", native_path, device="cpu")
    for name, value in model.network.state_dict().items():
        if not torch.equal(value.detach().cpu(), readback.network.state_dict()[name]):
            raise AssertionError(f"Native engineering export differs: {name}")
    report.update(
        initial_sha256=file_hash(initial),
        input_instances=len(selected),
        native_retention_scenes=8,
        highway_retention_scenes=8,
        teacher_reports=teacher_reports,
        native_export_sha256=file_hash(native_path),
        native_export_readback_identical=True,
        interpretation="engineering validation; weights are not reused for pilot repair",
    )
    write_json(report_path, report)
    return report


def run(config_path):
    if not (ROOT / "materials.json").exists():
        prepare(config_path)
    campaign = Campaign(config_path)
    campaign.verify_runtime()
    if (
        campaign.config["algorithms"] != ["diffusion_planner"]
        or campaign.config["methods"] != ["guided"]
        or campaign.config["run_kind"] != "pilot"
    ):
        raise ValueError("This entry executes the single-ADS guided pilot only")
    seed = campaign.config["master_seed"]
    initial = campaign.baseline("diffusion_planner", seed)
    first = campaign.output / "diffusion_planner" / str(seed) / "guided/round_1"
    if not (first / "training.json").exists():
        campaign.run_branch(
            "diffusion_planner",
            seed,
            "guided",
            initial,
            through_round=1,
            search_only=True,
        )
        engineering(campaign, initial, seed)
    final = campaign.run_branch("diffusion_planner", seed, "guided", initial)
    admissions = [
        json.loads((first.parent / f"round_{index}/admission.json").read_text())
        for index in (1, 2)
        if (first.parent / f"round_{index}/admission.json").exists()
    ]
    exported = campaign.output / "handoff/model.pth"
    model = load_policy("diffusion_planner", final, device="cpu")
    model.export_native(exported)
    restored = load_policy("diffusion_planner", exported, device="cpu")
    for name, value in model.network.state_dict().items():
        if not torch.equal(value, restored.network.state_dict()[name]):
            raise AssertionError(f"Pilot export differs: {name}")
    report = {
        "status": "highway_pilot_passed"
        if any(r["accepted_update"] for r in admissions)
        else "no_accepted_enhancement",
        "algorithm": "diffusion_planner",
        "seed": seed,
        "rounds": 2,
        "initial_policy": str(initial),
        "final_policy": str(final),
        "final_sha256": file_hash(final),
        "native_export": str(exported),
        "native_export_sha256": file_hash(exported),
        "native_export_readback_identical": True,
        "admission_reports": admissions,
        "nuplan_validation": "manual_by_user",
        "final_cohorts_generated": False,
        "formal_experiment_completed": False,
    }
    write_json(campaign.output / "pilot_summary.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "ads/configs/diffusion_planner_pilot.yaml",
    )
    args = parser.parse_args()
    settings = yaml.safe_load(args.config.read_text())
    output = REPO_ROOT / settings["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    with (output / "pilot.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = {"pid": os.getpid(), "config": str(args.config), "status": "running"}
        write_json(output / "status.json", status)
        try:
            report = run(args.config)
        except Exception as error:
            write_json(
                output / "status.json",
                {**status, "status": "failed", "error": str(error)},
            )
            raise
        write_json(output / "status.json", {**status, "status": report["status"]})


if __name__ == "__main__":
    main()
