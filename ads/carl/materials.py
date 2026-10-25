"""Capture CaRL's actual successful observations for fixed-teacher retention."""

import argparse
import fcntl
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import yaml

from ads.carl.policy import CaRLPolicy
from ads.common.batch_evaluation import evaluate_paths
from ads.common.evaluation import WorldEvaluator
from ads.common.instances import GeneratedInstance, UsageLedger, file_hash, json_hash
from ads.common.reporting import write_json
from ads.common.setup import REPO_ROOT, load_world
from ads.common.world import FrozenWorldFactory
from tools.evt import GPDTailModel

OBSERVATION_KEYS = ("bev_semantics", "measurements", "value_measurements")
ROOT = REPO_ROOT / "results/risk_stratified_repair/materials/carl"
COMMON_INPUTS = ROOT.parent / "diffusion_planner/instances.json"
OFFICIAL = Path(__file__).parent / "artifacts/official.pth"


class RecordingCaRLPolicy(CaRLPolicy):
    def reset(self, context):
        self.observations = []
        super().reset(context)

    def predict(self, observation, deterministic=True):
        self.observations.append(
            {key: observation[key].copy() for key in OBSERVATION_KEYS}
        )
        return super().predict(observation, deterministic=deterministic)


def capture_instance(
    policy, instance, factory, evt, directory, *, policy_hash, ledger, seed=20260926
):
    if instance.purpose != "repair" or instance.source_kind not in (
        "natural_generated",
        "empirical_natural",
    ):
        raise ValueError("CaRL retention requires natural training instances")
    if instance.source_kind == "empirical_natural" and (
        instance.source_metadata["source_split"] not in ("train", "val")
    ):
        raise ValueError("Empirical Test cannot supply CaRL retention material")
    ledger.register(instance, "repair")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (instance.instance_hash + ".json")
    with path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            saved = json.loads(path.read_text())
            if (
                saved["policy_sha256"] != policy_hash
                or saved["record"]["instance_hash"] != instance.instance_hash
                or saved["record"]["assets_hash"] != json_hash(instance.assets)
                or saved["seed"] != seed
            ):
                raise ValueError(
                    "CaRL retention execution belongs to another policy or instance"
                )
            if saved["qualified_keep"] and (
                file_hash(saved["observations_path"]) != saved["observations_sha256"]
            ):
                raise ValueError("CaRL recorded retention observations changed")
            return saved, 0
        result = WorldEvaluator(
            policy, "carl", factory, evt, ledger=ledger, purpose="repair", seed=seed
        )(instance)
        saved = {
            "policy_sha256": policy_hash,
            "seed": seed,
            "record": result.record,
            "qualified_keep": not result.failure and not result.record["stalled"],
            "decision_observations": len(policy.observations),
            "source": "recorded before each native policy decision",
        }
        if saved["qualified_keep"]:
            arrays = {
                key: np.stack([obs[key] for obs in policy.observations])
                for key in OBSERVATION_KEYS
            }
            arrays["decision_physics_steps"] = (
                np.arange(len(policy.observations), dtype=np.int32) * 5
            )
            target = path.with_suffix(".npz")
            np.savez_compressed(target, **arrays)
            saved.update(
                observations_path=str(target.resolve()),
                observations_sha256=file_hash(target),
            )
        write_json(path, saved)
        return saved, 1


def load_keep_samples(report_path, *, official_hash):
    report = json.loads(Path(report_path).read_text())
    if report["policy_sha256"] != official_hash:
        raise ValueError("CaRL retention must use the fixed official policy")
    samples = []
    for entry in report["executions"]:
        if (
            entry["policy_sha256"] != official_hash
            or entry["record"]["purpose"] != "repair"
        ):
            raise ValueError(
                "CaRL retention entry has a different policy or data purpose"
            )
        if not entry["qualified_keep"]:
            continue
        if entry["record"]["safety_failure"] or entry["record"]["stalled"]:
            raise ValueError("Unsuccessful CaRL execution cannot supply retention")
        if file_hash(entry["observations_path"]) != entry["observations_sha256"]:
            raise ValueError("Recorded CaRL observations changed")
        with np.load(entry["observations_path"], allow_pickle=False) as data:
            arrays = {key: data[key] for key in OBSERVATION_KEYS}
        for index in range(entry["decision_observations"]):
            samples.append(
                {
                    "scene_id": entry["record"]["root_sample_id"],
                    "instance_hash": entry["record"]["instance_hash"],
                    "observation": {key: value[index] for key, value in arrays.items()},
                }
            )
    return samples


def load_native_samples(report_path):
    report = json.loads(Path(report_path).read_text())
    if report["status"] != "passed":
        raise ValueError("CaRL native material preparation is incomplete")
    samples = []
    for entry in report["scenes"]["replay"]:
        if entry["source"]["purpose"] != "replay":
            raise ValueError("Native development data cannot supply CaRL retention")
        if file_hash(entry["path"]) != entry["sha256"]:
            raise ValueError("CaRL native material changed")
        with np.load(entry["path"], allow_pickle=False) as data:
            observation = {key: data[key] for key in OBSERVATION_KEYS}
        samples.append({"scene_id": entry["scene_id"], "observation": observation})
    return samples


def initialize_capture(settings):
    global _policy, _factory, _evt, _ledger, _seed, _policy_hash
    torch.set_num_threads(settings["runtime"]["torch_threads"])
    _policy = RecordingCaRLPolicy.load(OFFICIAL, device=settings["runtime"]["device"])
    _factory = FrozenWorldFactory(
        REPO_ROOT / settings["world_config"], device=settings["runtime"]["world_device"]
    )
    _evt = GPDTailModel.from_dict(
        json.loads((REPO_ROOT / settings["evt_model"]).read_text())
    )
    _ledger = UsageLedger(ROOT / "usage.sqlite")
    _seed, _policy_hash = settings["master_seed"], file_hash(OFFICIAL)


def capture_path(path):
    return capture_instance(
        _policy,
        GeneratedInstance.load(path),
        _factory,
        _evt,
        ROOT / "initial/keep",
        policy_hash=_policy_hash,
        ledger=_ledger,
        seed=_seed,
    )


def prepare_highway(config_path):
    settings = yaml.safe_load(Path(config_path).read_text())
    common = json.loads(COMMON_INPUTS.read_text())
    expected = settings["capability_preserving_repair"]["keep_candidates"]
    if len(common["keep"]) != expected or len(set(common["keep"])) != expected:
        raise ValueError("CaRL requires the complete fixed highway candidate pool")
    torch.set_num_threads(settings["runtime"]["torch_threads"])
    tail, _, _, tail_assets = load_world(settings)
    natural, _, _, natural_assets = load_world(
        settings, distribution="natural_generated"
    )
    assets = {"tail": tail_assets, "natural": natural_assets}
    if common["assets"] != assets:
        raise ValueError("CaRL retention and fixed inputs use different frozen worlds")
    del tail, natural
    torch.cuda.empty_cache()
    ROOT.mkdir(parents=True, exist_ok=True)
    policy_hash = file_hash(OFFICIAL)
    protocol = {
        "common_inputs_sha256": file_hash(COMMON_INPUTS),
        "official_policy_sha256": policy_hash,
        "assets": assets,
        "master_seed": settings["master_seed"],
        "runtime": settings["runtime"],
        "capture_implementation_sha256": file_hash(__file__),
    }
    header = ROOT / "protocol.json"
    if header.exists() and json.loads(header.read_text()) != protocol:
        raise ValueError("CaRL highway material protocol changed")
    write_json(header, protocol)
    executions, new_executions = [], 0
    with ProcessPoolExecutor(
        max_workers=4,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=initialize_capture,
        initargs=(settings,),
    ) as executor:
        for index, (entry, new) in enumerate(
            executor.map(capture_path, common["keep"]), 1
        ):
            executions.append(entry)
            new_executions += new
            if index % 25 == 0 or index == expected:
                print(f"CaRL retention: {index}/{expected}", flush=True)
    development = {}
    for kind, paths in common["development"].items():
        if (
            len(paths)
            != settings["capability_preserving_repair"][
                "development_tail"
                if kind == "tail_generated"
                else "development_natural"
            ]
        ):
            raise ValueError("CaRL development pool has a different size")
        entries = evaluate_paths(
            paths,
            OFFICIAL,
            ROOT / "initial" / kind,
            settings,
            algorithm="carl",
            purpose="development",
            workers=4,
            ledger_path=ROOT / "usage.sqlite",
        )
        development[kind] = {
            "recorded_executions": len(entries),
            "new_executions_this_invocation": sum(
                not row["cache_hit"] for row in entries
            ),
            "summary": str((ROOT / "initial" / kind / "summary.json").resolve()),
        }
    report = {
        "status": "passed",
        "protocol_sha256": file_hash(header),
        "policy_sha256": policy_hash,
        "input_manifest_sha256": file_hash(COMMON_INPUTS),
        "keep_candidates": expected,
        "qualified_keep_instances": sum(
            entry["qualified_keep"] for entry in executions
        ),
        "recorded_keep_executions": len(executions),
        "new_keep_executions_this_invocation": new_executions,
        "recorded_keep_execution_s": sum(e["record"]["elapsed_s"] for e in executions),
        "executions": executions,
        "development": development,
        "nuplan_simulation_performed": False,
    }
    write_json(ROOT / "highway.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    prepare_highway(parser.parse_args().config)
