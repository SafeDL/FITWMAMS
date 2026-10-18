"""Shared-split factual evaluation on the actual traffic execution backend."""

from pathlib import Path

import numpy as np
import torch

from .core.utils import load_json
from .data import ego_controls, artifact_identity
from .highway import HighwayEnvClosedLoopWorld
from .randomness import WORLD_RANDOM_BLOCKS, WorldExogenousState
from .stochastic_drivers.idm_population import MODEL_PATH
from .stochastic_drivers.raw_data import DRIVER_DATA_DIR, LABEL_PROTOCOL
from .stochastic_drivers.response_data import RESPONSE_DATA_DIR, PROTOCOL

ROOT = Path(__file__).resolve().parents[2]
MOBIL_PATH = ROOT / "results/hierarchical_world_model/model_assets/mobil.json"


def load_shared_components():
    """Load the frozen raw-highD Train/Val NPC population and policy."""
    population = dict(np.load(MODEL_PATH))
    fit = load_json(MOBIL_PATH)
    raw_data = DRIVER_DATA_DIR
    driver_population = {
        name: population[name].tolist() for name in ("log_mean", "log_cov")
    }
    if (
        Path(fit["driver_parameter_source"]).resolve() != MODEL_PATH.resolve()
        or fit["driver_population"] != driver_population
        or (
            fit["fit_split"],
            fit["selection_split"],
            fit["test_used_for_fit_or_selection"],
        )
        != ("train", "val", False)
        or Path(fit["source"]).resolve() != raw_data
        or fit["data_protocol"] != "highd_raw_drivers"
        or fit["label_protocol"] != LABEL_PROTOCOL
        or fit["raw_data_identity"] != artifact_identity(raw_data / "manifest.json")
        or fit["response_protocol"] != PROTOCOL
        or fit["response_data_identity"]
        != artifact_identity(RESPONSE_DATA_DIR / "manifest.json")
    ):
        raise ValueError("IDM and MOBIL must share the raw-highD Train/Val calibration")
    return population, fit


def recording_interval(records, metric):
    """Bootstrap whole recordings; frames and repeated styles are not new events."""
    _, index = np.unique([row["recording_id"] for row in records], return_inverse=True)
    values = np.asarray([row[metric] for row in records], float)
    totals, counts = np.bincount(index, weights=values), np.bincount(index)
    draws = np.random.default_rng(20261002).integers(
        len(counts), size=(2000, len(counts))
    )
    means = totals[draws].sum(1) / counts[draws].sum(1)
    return np.quantile(means, [0.025, 0.975]).tolist()


def recording_ratio_interval(records, numerator, denominator):
    """Bootstrap recording totals, preserving actor-frame weighting of errors."""
    _, index = np.unique([row["recording_id"] for row in records], return_inverse=True)
    top = np.bincount(index, weights=[row[numerator] for row in records])
    bottom = np.bincount(index, weights=[row[denominator] for row in records])
    draws = np.random.default_rng(20261002).integers(len(top), size=(2000, len(top)))
    ratios = top[draws].sum(1) / bottom[draws].sum(1)
    return np.quantile(ratios, [0.025, 0.975]).tolist()


def scene_randomness(sequence_ids, *, seed=20261002, frames=149):
    """A physical scene retains its draws across cache reordering and treatment."""
    draws = [
        WorldExogenousState.sample(
            1,
            seed=int(
                np.random.SeedSequence(
                    [seed, *map(int, str(identifier).split("_")[1:])]
                ).generate_state(1)[0]
            ),
            response_steps=frames,
        )
        for identifier in sequence_ids
    ]
    return WorldExogenousState(
        **{
            key: np.concatenate([getattr(draw, key) for draw in draws])
            for key in WORLD_RANDOM_BLOCKS
        }
    )


@torch.no_grad()
def highway_rollout(
    arrays,
    rows,
    plans,
    randomness,
    *,
    controller=None,
    ads_controls=None,
    background_action_delta=None,
    tracking_config=None,
    device="cpu",
):
    """Apply logged ego controls, not logged NPC states, to one closed-loop world."""
    device = torch.device(device)
    logged = np.asarray(arrays["agent_states"][rows], np.float32)
    valid = np.asarray(arrays["agent_valid"][rows], bool)
    dimensions = np.asarray(arrays["vehicle_dimensions"][rows], np.float32)
    baseline_controls = torch.as_tensor(
        ego_controls(logged[:, 24:173, 0], logged[:, 25:174, 0], 0.04),
        device=device,
    )
    world = HighwayEnvClosedLoopWorld(
        device=device,
        controller=controller,
        tracking_config=tracking_config,
    )
    world.reset(
        logged[:, 24],
        valid[:, 24],
        plans,
        np.asarray(arrays["map_polylines"][rows]),
        np.asarray(arrays["map_polyline_valid"][rows]),
        vehicle_dimensions=dimensions,
        exogenous_state=randomness,
        initial_history=logged[:, :25],
        initial_history_valid=valid[:, :25],
    )
    controls = (
        baseline_controls
        if ads_controls is None
        else torch.as_tensor(ads_controls, device=device)
    )
    if controls.shape != baseline_controls.shape:
        raise ValueError(
            "ADS controls must cover the same physical times as baseline controls"
        )
    road_steps = np.zeros(len(rows), dtype=np.int64)
    for index, traffic in enumerate(world.traffic):
        original_step = traffic.road.step

        def counted_step(dt, *, scenario=index, step=original_step):
            road_steps[scenario] += 1
            return step(dt)

        traffic.road.step = counted_step
    trace = {
        key: []
        for key in (
            "states",
            "actions",
            "base_actions",
            "crashed",
            "offroad",
            "pairs",
            "target_y",
            "committed",
        )
    }
    initial_offroad = np.stack(
        [
            traffic.scene_geometry.offroad(
                traffic.states()[:, :2],
                vehicle_width_m=traffic.vehicle_dimensions.width_m,
            )
            & traffic.valid
            for traffic in world.traffic
        ]
    )
    bridge_trace = {}
    initial_states = world.states.detach().cpu().numpy().copy()
    for frame in range(149):
        result = world.advance_response(
            controls[:, frame],
            background_action_delta=(
                None
                if background_action_delta is None
                else background_action_delta[:, frame : frame + 1]
            ),
        )
        for name, value in result.items():
            if name.startswith("takeover_") and isinstance(value, torch.Tensor):
                bridge_trace.setdefault(name, []).append(value.cpu().numpy())
        for name, source in (
            ("states", "agent_states"),
            ("actions", "background_actions"),
            ("base_actions", "base_background_actions"),
            ("crashed", "crashed"),
            ("offroad", "offroad"),
            ("pairs", "collision_pairs"),
        ):
            value = result[source].detach().cpu().numpy()
            trace[name].append(
                value[:, 0] if name in ("actions", "base_actions") else value
            )
        target = None if controller is None else controller._autonomous_lane_target_y
        committed = None if controller is None else controller._autonomous_lane_active
        trace["target_y"].append(
            logged[:, 24, 1:, 1] if target is None else target.cpu().numpy().copy()
        )
        trace["committed"].append(
            np.zeros((len(rows), 6), bool)
            if committed is None
            else committed.cpu().numpy().copy()
        )
    result = {key: np.stack(values, axis=1) for key, values in trace.items()}
    result.update(
        {key: np.stack(values, axis=1) for key, values in bridge_trace.items()}
    )
    result["road_step_calls"] = road_steps
    result["initial_states"] = initial_states
    result["initial_offroad"] = initial_offroad
    result["logged_offroad"] = np.stack(
        [
            traffic.scene_geometry.offroad(
                logged[index, 25:174, :, :2],
                vehicle_width_m=traffic.vehicle_dimensions.width_m,
            )
            for index, traffic in enumerate(world.traffic)
        ]
    )
    result["boundary_clearance_m"] = np.stack(
        [
            traffic.scene_geometry.boundary_clearance(
                result["states"][index, :, :, :2],
                vehicle_width_m=traffic.vehicle_dimensions.width_m,
            )
            for index, traffic in enumerate(world.traffic)
        ]
    )
    result["vehicle_width_m"] = dimensions[..., 1]
    return result
