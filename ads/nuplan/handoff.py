"""Package a frozen policy pair for manual, separate nuPlan evaluation."""

import argparse
import fcntl
import json
import shutil
from pathlib import Path

from ads.common.instances import file_hash
from ads.common.reporting import write_json
from ads.common.setup import REPO_ROOT
from ads.nuplan.assets import PUBLISHED
from ads.nuplan.export import export_policy

PLANNERS = ("carl", *PUBLISHED)
FILENAMES = {
    "carl": "model_best.pth",
    "diffusion_planner": "model.pth",
    "dtpp": "base_model.pth",
    "flow_planner": "model.pth",
    "str2": "checkpoint-297000",
}


def prepare_handoff(campaign, algorithm, method, scenarios):
    if algorithm not in PLANNERS or method not in campaign.config["methods"]:
        raise ValueError(
            "manual nuPlan handoff requires a registered native policy pair"
        )
    frozen = campaign.verify_freeze()
    root = REPO_ROOT / campaign.config["manual_nuplan"]["output_dir"]
    root.mkdir(parents=True, exist_ok=True)
    scenarios = Path(scenarios).resolve()
    digest = file_hash(scenarios)
    with (root / "handoff.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        common = root / "common_scenarios.json"
        if common.exists():
            manifest = json.loads(common.read_text())
            if (
                digest != manifest["sha256"]
                or file_hash(root / manifest["path"]) != digest
            ):
                raise ValueError(
                    "manual comparisons must use the same frozen scenario manifest"
                )
        else:
            path = root / ("scenarios" + scenarios.suffix)
            shutil.copyfile(scenarios, path)
            manifest = {"path": path.name, "sha256": digest}
            write_json(common, manifest)
        outputs = []
        for seed in campaign.config["seeds"]:
            directory = root / algorithm / method / str(seed)
            directory.mkdir(parents=True, exist_ok=True)
            pair = {}
            for role, arm in (("initial", "initial"), ("repaired", method)):
                source = frozen["policies"][f"{algorithm}/{seed}/{arm}"]
                destination = directory / role / FILENAMES[algorithm]
                pair[role] = {
                    "source": source,
                    "native_path": str(destination.relative_to(root)),
                }
            record = {
                "schema": "nuplan_manual_policy_pair_v1",
                "protocol_hash": frozen["protocol_hash"],
                "algorithm": algorithm,
                "comparison": method,
                "seed": seed,
                "common_scenarios": manifest,
                "policies": pair,
                "native_metrics": {
                    role: str((directory / "native_metrics" / role).relative_to(root))
                    for role in pair
                },
                "automatic_evaluation": False,
                "included_in_highd_statistics": False,
                "used_for_model_selection": False,
            }
            path = directory / "handoff.json"
            if path.exists():
                saved = json.loads(path.read_text())
                for role, entry in pair.items():
                    entry["native_sha256"] = file_hash(root / entry["native_path"])
                if saved != record:
                    raise ValueError(
                        "manual handoff differs from the frozen policy pair"
                    )
            else:
                for role, entry in pair.items():
                    destination = root / entry["native_path"]
                    export_policy(
                        algorithm, REPO_ROOT / entry["source"]["path"], destination
                    )
                    entry["native_sha256"] = file_hash(destination)
                    (root / record["native_metrics"][role]).mkdir(
                        parents=True, exist_ok=True
                    )
                write_json(path, record)
            outputs.append(path)
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("algorithm", choices=PLANNERS)
    parser.add_argument("method", choices=("monte_carlo", "subset", "guided"))
    parser.add_argument("scenarios", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "ads/configs/risk_stratified_repair.yaml",
    )
    args = parser.parse_args()
    from ads.risk_stratified_repair import Campaign

    for path in prepare_handoff(
        Campaign(args.config), args.algorithm, args.method, args.scenarios
    ):
        print(path)


if __name__ == "__main__":
    main()
