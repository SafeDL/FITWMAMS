"""Load and fingerprint the published generator; never refit or repartition it."""

import json
from importlib import import_module
from pathlib import Path

import torch
import yaml

from ads.common.instances import file_hash, json_hash
from ads.common.world import FrozenWorldFactory
from hierarchical_world_model.src.composition import (
    DEFAULT_WORLD_SELECTION,
    HierarchicalWorldSampler,
)
from hierarchical_world_model.src.stochastic_drivers.idm_population import (
    MODEL_PATH as DRIVER_MODEL_PATH,
)
from hierarchical_world_model.src.stochastic_drivers.lane_trajectory import (
    MODEL_PATH as LANE_MODEL_PATH,
)
from tools.evt import GPDTailModel

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_world(settings, *, distribution="tail_generated"):
    world_path = REPO_ROOT / settings["world_config"]
    world_cfg = yaml.safe_load(world_path.read_text())
    paths = {
        k: REPO_ROOT / world_cfg["paths"][k]
        for k in ("flow_checkpoint", "diffusion_checkpoint", "diffusion_contract")
    }
    if distribution == "natural_generated":
        paths["flow_checkpoint"] = (
            REPO_ROOT / settings["evaluation"]["natural_flow_checkpoint"]
        )
    flow = torch.load(paths["flow_checkpoint"], map_location="cpu", weights_only=False)
    metrics = flow.get("metrics", {})
    valid_scope = (
        metrics.get("fit_scope") == "natural_train_val"
        if distribution == "natural_generated"
        else metrics.get("fit_scope") == "all_evt_tail_events"
        and metrics.get("num_fit_events") == 2964
    )
    if not valid_scope:
        raise ValueError(f"invalid frozen Flow fit scope for {distribution}")
    del flow
    if world_cfg["evaluation_scope"]["excluded_background_slots"]:
        raise ValueError(
            "risk-stratified repair uses the full existing highD background scope"
        )
    sampler = HierarchicalWorldSampler(
        **paths,
        repo_root=REPO_ROOT,
        device=settings["runtime"]["device"],
        ddim_steps=settings["ddim_steps"],
        excluded_slots=(),
        flow_population="natural"
        if distribution == "natural_generated"
        else "evt_tail",
    )
    files = {
        **paths,
        "world_config": world_path,
        "world_selection": DEFAULT_WORLD_SELECTION,
        "driver_model": DRIVER_MODEL_PATH,
        "lane_model": LANE_MODEL_PATH,
        "interaction_errors": REPO_ROOT
        / world_cfg["takeover"]["error_trajectories_path"],
        "evt_model": REPO_ROOT / settings["evt_model"],
    }
    assets = {name: file_hash(path) for name, path in files.items()}
    assets["generation_contract"] = json_hash(
        {
            "ddim_steps": settings["ddim_steps"],
            "excluded_slots": [],
            "fit_events": metrics["num_fit_events"],
            "distribution": distribution,
        }
    )
    implementation = {}
    for directory in (
        "hierarchical_world_model/src",
        "normalizing_flow/src",
        "diffusion/src",
        "process_highD/src",
        "HighwayEnv/highway_env",
    ):
        for path in sorted((REPO_ROOT / directory).rglob("*.py")):
            implementation[str(path.relative_to(REPO_ROOT))] = file_hash(path)
    assets["world_implementation"] = json_hash(implementation)
    factory = FrozenWorldFactory(world_path, device=settings["runtime"]["world_device"])
    evt = GPDTailModel.from_dict(
        json.loads((REPO_ROOT / settings["evt_model"]).read_text())
    )
    return sampler, factory, evt, assets


def load_policy(algorithm, path, *, device):
    if algorithm in ("dqn", "ppo"):
        from stable_baselines3 import DQN, PPO

        from ads.common.discrete_policy import DiscretePolicy

        return DiscretePolicy(
            (DQN if algorithm == "dqn" else PPO).load(path, device=device), algorithm
        )
    if algorithm == "carl":
        from ads.carl.policy import CaRLPolicy

        return CaRLPolicy.load(path, device=device)
    if algorithm == "plant2":
        from ads.plant2.policy import PlanTPolicy

        return PlanTPolicy.load(path, device=device)
    native_classes = {
        "diffusion_planner": "DiffusionPlannerPolicy",
        "dtpp": "DTPPPolicy",
        "flow_planner": "FlowPlannerPolicy",
        "str2": "STR2Policy",
    }
    if algorithm in native_classes:
        return getattr(
            import_module(f"ads.{algorithm}.policy"), native_classes[algorithm]
        ).load(path, device=device)
    raise ValueError("unknown ADS algorithm")
