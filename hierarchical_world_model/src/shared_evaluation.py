"""Shared-split factual evaluation on the actual traffic execution backend."""

from pathlib import Path

import numpy as np
import torch

from .core.utils import load_json
from .data import ego_controls, training_sequence_ids, artifact_identity
from .highway import HighwayEnvClosedLoopWorld
from .randomness import WORLD_RANDOM_BLOCKS, WorldExogenousState
from .stochastic_drivers.idm_population import MODEL_PATH
from .stochastic_drivers.raw_data import DRIVER_DATA_DIR, LABEL_PROTOCOL
from .train import load_checkpoint

ROOT = Path(__file__).resolve().parents[2]
MOBIL_PATH = ROOT / "results/hierarchical_world_model/model_assets/mobil.json"


def load_shared_hiqr(config, experiment, device):
    """Require the corrected HiQR fitted on this shared Train and selected on Val."""
    checkpoint = (
        Path(config["paths"]["output_dir"]) / "checkpoints/stochastic_heads_best.pt"
    )
    model, payload = load_checkpoint(checkpoint, device=device)
    training = payload["training_data"]
    if (
        any(
            training.get(key) != value
            for key, value in training_sequence_ids(experiment).items()
        )
        or training.get("diffusion")
        != artifact_identity(config["paths"]["diffusion_checkpoint"])
        or training["test_used_for_fit_or_selection"]
        or training["history_selection"] != "last_observed_frame_index"
        or training.get("history_controls") != "valid_observed_transitions"
    ):
        raise ValueError(
            "HiQR training identity or Diffusion changed; run Train/Val training before Test"
        )
    return model, training


def load_shared_components(config, experiment, device):
    """Load shared HiQR, IDM and MOBIL without requiring a learned selector."""
    model, training = load_shared_hiqr(config, experiment, device)
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
    ):
        raise ValueError("IDM and MOBIL must share the raw-highD Train/Val calibration")
    return model, training, population, fit


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
    model, arrays, rows, plans, randomness, *, controller=None, recovery=False
):
    """Apply logged ego controls, not logged NPC states, to one closed-loop world."""
    device = next(model.parameters()).device
    logged = np.asarray(arrays["agent_states"][rows], np.float32)
    valid = np.asarray(arrays["agent_valid"][rows], bool)
    dimensions = np.asarray(arrays["vehicle_dimensions"][rows], np.float32)
    world = HighwayEnvClosedLoopWorld(
        model,
        device=device,
        controller=controller,
        spatial_progress_reference=True,
        reference_recovery_enabled=recovery,
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
    controls = torch.as_tensor(
        ego_controls(logged[:, 24:173, 0], logged[:, 25:174, 0], 0.04),
        device=device,
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
            "retired",
            "mode",
            "gate",
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
    for frame in range(149):
        result = world.advance_response(controls[:, frame])
        for name, source in (
            ("states", "agent_states"),
            ("actions", "background_actions"),
            ("base_actions", "base_background_actions"),
            ("crashed", "crashed"),
            ("offroad", "offroad"),
            ("pairs", "collision_pairs"),
            ("retired", "reference_retired"),
        ):
            value = result[source].detach().cpu().numpy()
            trace[name].append(
                value[:, 0] if name in ("actions", "base_actions") else value
            )
        modes, gates = (
            result["controller_response_mode"],
            result["controller_response_gate"],
        )
        trace["mode"].append(
            np.full((len(rows), 6), -1, np.int8)
            if modes is None
            else modes.cpu().numpy()
        )
        trace["gate"].append(
            np.zeros((len(rows), 6), np.float32)
            if gates is None
            else gates.cpu().numpy()
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
    result["road_step_calls"] = road_steps
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


def factual_records(rows, logged, valid, trace):
    """Keep every row, including collisions, rather than filtering on outcome."""
    active = valid[:, 24, 1:]
    observed = active[:, None] & valid[:, 25:174, 1:]
    distance = np.linalg.norm(
        trace["states"][:, :, 1:, :2] - logged[:, 25:174, 1:, :2], axis=-1
    )
    final_mask = observed[:, -1]
    corrected = (np.abs(trace["actions"] - trace["base_actions"]) > 1e-4).any(-1)
    changed_target = np.abs(np.diff(trace["target_y"], axis=1)) > 0.5
    changed_target &= trace["committed"][:, 1:] & trace["committed"][:, :-1]
    counts = observed.sum(axis=(1, 2))
    final_counts = final_mask.sum(axis=1)
    retirement = trace["retired"] & active[:, None]
    offroad = trace["offroad"] & valid[:, 24:25]
    new_offroad = offroad & ~trace["initial_offroad"][:, None]
    logged_offroad = trace["logged_offroad"] & valid[:, 25:174] & valid[:, 24:25]
    introduced_offroad = offroad & ~logged_offroad & valid[:, 25:174]
    overhang = np.maximum(-trace["boundary_clearance_m"], 0)
    center_outside = (
        trace["boundary_clearance_m"] < -0.5 * trace["vehicle_width_m"][:, None]
    ) & valid[:, 24:25]
    offroad_metrics = {
        "offroad": offroad,
        "new_offroad": new_offroad,
        "logged_offroad": logged_offroad,
        "introduced_offroad": introduced_offroad,
    }
    jerk = np.diff(trace["actions"][..., 0], axis=1) / 0.04
    pre_impact = ~(trace["crashed"][:, 1:, 1:] | trace["crashed"][:, :-1, 1:])
    jerk_valid = observed[:, 1:] & observed[:, :-1] & pre_impact
    records = []
    for index, row in enumerate(rows):
        mask = observed[index]
        boundary = {}
        for prefix, actors in (
            ("", slice(0, 7)),
            ("ego_", slice(0, 1)),
            ("npc_", slice(1, 7)),
        ):
            values = overhang[index, :, actors]
            boundary[f"{prefix}maximum_boundary_overhang_m"] = float(
                np.where(offroad[index, :, actors], values, 0).max()
            )
            boundary[f"{prefix}maximum_introduced_boundary_overhang_m"] = float(
                np.where(introduced_offroad[index, :, actors], values, 0).max()
            )
            boundary[f"{prefix}center_outside_road"] = bool(
                center_outside[index, :, actors].any()
            )
        records.append(
            {
                "row": int(row),
                "actor_frames": int(counts[index]),
                "distance_sum_m": float(distance[index][mask].sum()),
                "ADE_m": float(distance[index][mask].mean()) if mask.any() else None,
                "final_actors": int(final_counts[index]),
                "final_distance_sum_m": float(
                    distance[index, -1][final_mask[index]].sum()
                ),
                "collision": bool(trace["pairs"][index].any()),
                **{
                    name: bool(value[index].any())
                    for name, value in offroad_metrics.items()
                },
                **{
                    f"{group}_{name}": bool(value[index, :, actors].any())
                    for group, actors in (("ego", slice(0, 1)), ("npc", slice(1, 7)))
                    for name, value in offroad_metrics.items()
                },
                **boundary,
                "finite": bool(np.isfinite(trace["states"][index]).all()),
                "corrected_actor_frames": int((corrected[index] & mask).sum()),
                "retired_actors": int(retirement[index, -1].sum()),
                "retired_actor_frames": int((retirement[index] & mask).sum()),
                "retired_before_last_second": int(
                    retirement[index, :124].any(axis=0).sum()
                ),
                "committed_target_changes": int(changed_target[index].sum()),
                "maximum_preimpact_acceleration_mps2": (
                    float(
                        trace["actions"][index, :, :, 0][
                            mask & ~trace["crashed"][index, :, 1:]
                        ].max()
                    )
                    if (mask & ~trace["crashed"][index, :, 1:]).any()
                    else None
                ),
            }
        )
    return records, distance[observed], np.abs(jerk[jerk_valid])


def save_error_samples(output, errors, jerks):
    """Keep quantile samples and their batch boundaries without separate shards."""
    np.savez_compressed(
        Path(output) / "error_samples.npz",
        errors=np.concatenate(errors),
        jerks=np.concatenate(jerks),
        error_offsets=np.r_[0, np.cumsum([len(value) for value in errors])],
        jerk_offsets=np.r_[0, np.cumsum([len(value) for value in jerks])],
    )


def summarize_records(records, errors, jerks):
    counts = sum(row["actor_frames"] for row in records)
    final_counts = sum(row["final_actors"] for row in records)
    return {
        "sequences": len(records),
        "actor_frames": counts,
        "ADE_m": sum(row["distance_sum_m"] for row in records) / max(counts, 1),
        "FDE_m": sum(row["final_distance_sum_m"] for row in records)
        / max(final_counts, 1),
        "P95_m": float(np.quantile(np.concatenate(errors), 0.95)),
        "preimpact_longitudinal_jerk_P99_mps3": float(
            np.quantile(np.concatenate(jerks), 0.99)
        ),
        **{
            f"{group}_{name}": sum(row[f"{group}_{name}"] for row in records)
            for group in ("ego", "npc")
            for name in (
                "offroad",
                "new_offroad",
                "logged_offroad",
                "introduced_offroad",
            )
        },
        **{
            name: sum(row[name] for row in records)
            for name in (
                "collision",
                "offroad",
                "new_offroad",
                "logged_offroad",
                "introduced_offroad",
                "finite",
                "corrected_actor_frames",
                "retired_actors",
                "retired_actor_frames",
                "retired_before_last_second",
                "committed_target_changes",
            )
        },
        "boundary_severity": {
            group: boundary_severity(records, prefix)
            for group, prefix in (("scene", ""), ("ego", "ego_"), ("npc", "npc_"))
        },
    }


def boundary_severity(records, prefix):
    """Describe the unchanged offroad count, not a new acceptance threshold."""
    peaks = np.asarray(
        [row[f"{prefix}maximum_introduced_boundary_overhang_m"] for row in records]
    )
    affected = peaks[peaks > 0]
    return {
        "introduced_overhang_at_most_0_1m_sequences": int(
            ((peaks > 0) & (peaks <= 0.1)).sum()
        ),
        "introduced_overhang_over_0_1m_sequences": int((peaks > 0.1).sum()),
        "center_outside_road_sequences": sum(
            row[f"{prefix}center_outside_road"] for row in records
        ),
        "introduced_overhang_q50_q95_m": (
            np.quantile(affected, [0.5, 0.95]).tolist() if len(affected) else None
        ),
        "maximum_overhang_m": max(
            row[f"{prefix}maximum_boundary_overhang_m"] for row in records
        ),
    }
