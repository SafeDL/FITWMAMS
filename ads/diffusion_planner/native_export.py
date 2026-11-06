"""Read enhanced EMA weights through the official nuPlan planner loader."""

from types import SimpleNamespace

import numpy as np
import torch
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from ads.common.instances import file_hash
from ads.common.randomness import PolicyRandomContext
from ads.common.reporting import write_json
from ads.common.setup import load_policy
from ads.diffusion_planner.vendor.diffusion_planner.planner.planner import (
    DiffusionPlanner,
)
from ads.diffusion_planner.vendor.diffusion_planner.utils.config import Config


def export_verified(source, destination, sample):
    model = load_policy("diffusion_planner", source, device="cpu")
    model.export_native(destination)
    # The official loader uses torch.load's default weights-only mode in Torch 2.6.
    torch.load(destination, map_location="cpu", weights_only=True)
    config = Config(str(destination.parent / "args.json"), None)
    config.device = "cpu"
    native = DiffusionPlanner(
        config,
        str(destination),
        TrajectorySampling(num_poses=20, time_horizon=2),
        TrajectorySampling(num_poses=80, time_horizon=8),
        enable_ema=True,
        device="cpu",
    )
    native.initialize(SimpleNamespace(map_api=None, route_roadblock_ids=[]))
    expected = model.network.state_dict()
    actual = native._planner.state_dict()
    if expected.keys() != actual.keys() or any(
        not torch.equal(expected[key], actual[key]) for key in expected
    ):
        raise AssertionError(
            "Official native loader changed enhanced deployment parameters"
        )
    features = {key: value.copy() for key, value in sample["features"].items()}
    features["ego_current_state"] = np.array((0, 0, 1, 0, 0, 0, 0, 0, 0, 0), np.float32)
    inputs = model.inputs([features])
    context = PolicyRandomContext(
        "diffusion_planner",
        mode="final_test",
        instance_hash="native_export_verification",
    )
    model.network.eval()
    with context.step(0, "cpu"), torch.no_grad():
        _, first = model.network(inputs)
    with context.step(0, "cpu"), torch.no_grad():
        _, second = native._planner(inputs)
    difference = float((first["prediction"] - second["prediction"]).abs().max())
    if difference != 0:
        raise AssertionError(f"Official native output differs by {difference}")
    report = {
        "source_sha256": file_hash(source),
        "export_sha256": file_hash(destination),
        "official_loader": "DiffusionPlanner.initialize",
        "weights_only_load": True,
        "native_parameters_equal": True,
        "output_max_abs_difference": difference,
        "city_closed_loop": "manual_by_user",
        "used_for_model_selection": False,
    }
    write_json(destination.parent / "handoff.json", report)
    return report
