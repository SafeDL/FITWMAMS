"""Local on-policy CaRL repair with native retention and paired admission."""

import json
from pathlib import Path

from ads.carl.materials import (
    COMMON_INPUTS,
    OFFICIAL,
    ROOT,
    load_keep_samples,
    load_native_samples,
)
from ads.carl.train import train
from ads.common.admission import admission
from ads.common.instances import file_hash
from ads.common.reporting import write_json
from ads.common.setup import load_policy


def repair_and_select(campaign, model, env, *, base, directory, steps, seed, progress):
    directory = Path(directory)
    manifest = json.loads(COMMON_INPUTS.read_text())
    prepared = json.loads((ROOT / "highway.json").read_text())
    protocol_path = ROOT / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    official_hash = file_hash(OFFICIAL)
    if (
        prepared["status"] != "passed"
        or prepared["protocol_sha256"] != file_hash(protocol_path)
        or prepared["input_manifest_sha256"] != file_hash(COMMON_INPUTS)
        or prepared["policy_sha256"] != official_hash
        or protocol["assets"] != manifest["assets"]
        or manifest["assets"]["tail"] != campaign.assets
    ):
        raise ValueError("CaRL material and search protocols differ")
    native_path = campaign.config["capability_preserving_repair"]["native_materials"]
    native = load_native_samples(native_path)
    keep = load_keep_samples(ROOT / "highway.json", official_hash=official_hash)
    candidates = []

    def save(policy, collected):
        path = directory / "candidates" / str(collected) / "policy.pth"
        policy.save(path)
        candidates.append((collected, path))
        write_json(path.parent / "training_state.json", policy.repair_state)

    report = train(
        model,
        env,
        steps=steps,
        seed=seed,
        settings=campaign.config["carl"],
        retention={"native": native, "keep": keep},
        retention_coefficient=campaign.config["capability_preserving_repair"][
            "retention_coefficient"
        ],
        checkpoint=save,
        progress=progress,
    )
    report["retention_materials"] = {
        "native_sha256": file_hash(native_path),
        "highway_sha256": file_hash(ROOT / "highway.json"),
        "native_observations": len(native),
        "highway_observations": len(keep),
    }
    write_json(directory / "optimization.json", report)
    event_instances = {
        instance.event.event_id: instance.instance_hash for instance in env.instances
    }
    selected, outcome = admission(
        candidates,
        base,
        campaign,
        directory,
        algorithm="carl",
        official=OFFICIAL,
        material_root=ROOT,
        development=manifest["development"],
        training_instances=[
            event_instances[event] for event in report["event_decisions"]
        ],
    )
    report["admission"] = outcome
    return load_policy("carl", selected, device=campaign.device), report
