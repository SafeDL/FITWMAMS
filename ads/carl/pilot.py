"""Run CaRL's guided two-round pilot without opening final evaluation cohorts."""

import argparse
import fcntl
import json
import os
from pathlib import Path

import torch
import yaml

from ads.carl.materials import (
    COMMON_INPUTS,
    OFFICIAL,
    ROOT,
    load_keep_samples,
    load_native_samples,
    prepare_highway,
)
from ads.carl.policy import CaRLPolicy
from ads.carl.train import train
from ads.common.batch_evaluation import evaluate_paths
from ads.common.curriculum import load_curriculum
from ads.common.evaluation_sampling import named_stream
from ads.common.instances import file_hash
from ads.common.repair import RepairEnvironment
from ads.common.reporting import write_json
from ads.common.setup import REPO_ROOT, load_policy
from ads.risk_stratified_repair import Campaign


def export_and_verify(model, path):
    model.export_native(path)
    restored = CaRLPolicy.load(path, device="cpu")
    for name, value in model.network.state_dict().items():
        if not torch.equal(value.detach().cpu(), restored.network.state_dict()[name]):
            raise AssertionError(f"Native CaRL export differs: {name}")
    return {
        "native_export": str(path),
        "native_export_sha256": file_hash(path),
        "native_export_readback_identical": True,
        "native_state_tensors": len(model.network.state_dict()),
    }


def engineering(campaign, initial, seed):
    directory = campaign.output / "engineering"
    report_path = directory / "report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text())
        if report["initial_sha256"] != file_hash(initial):
            raise ValueError("Engineering check used another initial CaRL policy")
        return report
    first = campaign.output / "carl" / str(seed) / "guided/round_1"
    banks, _ = load_curriculum(first, campaign.ledger)
    selected = {name: banks.get(name, [])[:8] for name in ("normal", "new_failure")}
    native_path = campaign.config["capability_preserving_repair"]["native_materials"]
    native = load_native_samples(native_path)[:8]
    keep = load_keep_samples(ROOT / "highway.json", official_hash=file_hash(OFFICIAL))
    roots = set(list(dict.fromkeys(sample["scene_id"] for sample in keep))[:8])
    keep = [sample for sample in keep if sample["scene_id"] in roots]
    model = load_policy("carl", initial, device=campaign.device)
    engineering_seed = named_stream("engineering", "carl", seed)
    env = RepairEnvironment(
        "carl",
        selected,
        campaign.factory,
        round_index=0,
        seed=engineering_seed,
        ledger=campaign.ledger,
    )
    report = train(
        model,
        env,
        steps=1024,
        seed=engineering_seed,
        settings=campaign.config["carl"],
        retention={"native": native, "keep": keep},
        retention_coefficient=campaign.config["capability_preserving_repair"][
            "retention_coefficient"
        ],
    )
    checkpoint = directory / "policy.pth"
    model.save(checkpoint)
    report.update(export_and_verify(model, directory / "native/model_best.pth"))
    inputs = json.loads(COMMON_INPUTS.read_text())
    development = {}
    for name, paths in inputs["development"].items():
        entries = evaluate_paths(
            paths[:16],
            checkpoint,
            directory / "development" / name,
            campaign.config,
            algorithm="carl",
            workers=4,
            ledger_path=campaign.ledger.path,
        )
        development[name] = {
            "instances": len(entries),
            "new_executions": sum(not row["cache_hit"] for row in entries),
            "collisions": sum(row["record"]["collision"] for row in entries),
        }
    report.update(
        initial_sha256=file_hash(initial),
        training_input_instances={name: len(rows) for name, rows in selected.items()},
        native_retention_scenes=len(native),
        highway_retention_scenes=len(roots),
        development=development,
        interpretation="engineering validation; weights are not reused for pilot repair",
    )
    write_json(report_path, report)
    return report


def run(config_path):
    settings = yaml.safe_load(Path(config_path).read_text())
    if (
        settings["algorithms"] != ["carl"]
        or settings["methods"] != ["guided"]
        or settings["run_kind"] != "pilot"
    ):
        raise ValueError("This entry executes the single-ADS CaRL guided pilot only")
    if not (ROOT / "highway.json").exists():
        prepare_highway(config_path)
    campaign = Campaign(config_path)
    campaign.verify_runtime()
    seed = campaign.config["master_seed"]
    initial = campaign.baseline("carl", seed)
    first = campaign.output / "carl" / str(seed) / "guided/round_1"
    if not (first / "training.json").exists():
        campaign.run_branch(
            "carl", seed, "guided", initial, through_round=1, search_only=True
        )
        engineering(campaign, initial, seed)
    final = campaign.run_branch("carl", seed, "guided", initial)
    admissions = [
        json.loads((first.parent / f"round_{index}/admission.json").read_text())
        for index in range(1, campaign.config["rounds"] + 1)
    ]
    model = load_policy("carl", final, device="cpu")
    report = {
        "status": "highway_pilot_passed"
        if any(row["accepted_update"] for row in admissions)
        else "no_accepted_enhancement",
        "algorithm": "carl",
        "seed": seed,
        "rounds": campaign.config["rounds"],
        "initial_policy": str(initial),
        "final_policy": str(final),
        "final_sha256": file_hash(final),
        **export_and_verify(model, campaign.output / "handoff/model_best.pth"),
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
        default=Path(__file__).resolve().parents[1] / "configs/carl_pilot.yaml",
    )
    config = parser.parse_args().config
    settings = yaml.safe_load(config.read_text())
    directory = REPO_ROOT / settings["output_dir"]
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "pilot.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write_json(
            directory / "pilot_process.json",
            {"pid": os.getpid(), "config": str(config)},
        )
        run(config)


if __name__ == "__main__":
    main()
