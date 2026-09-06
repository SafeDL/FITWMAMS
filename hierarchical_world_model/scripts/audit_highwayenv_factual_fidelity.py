#!/usr/bin/env python3
"""Measure factual highD reconstruction on the actual HighwayEnv bridge."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffusion.src.data import ANCHOR_INDEX  # noqa: E402
from hierarchical_world_model.src.data import prepare_experiment_data  # noqa: E402
from hierarchical_world_model.src.evaluation import rollout as offline_rollout  # noqa: E402
from hierarchical_world_model.src.planner import complete_missing_background_plans, frozen_diffusion_plans  # noqa: E402
from hierarchical_world_model.src.protocol import load_protocol_config  # noqa: E402
from hierarchical_world_model.src.reaction_training import (  # noqa: E402
    PolicyTrainingConfig, reaction_controller_rollout,
)
from hierarchical_world_model.src.train import load_checkpoint  # noqa: E402
from hierarchical_world_model.scripts.risk_calibration import (  # noqa: E402
    _factual_fidelity,
    _highway_highd_control_replay,
    _kinematic_highd_control_replay,
)
from world_model.src.core.utils import ensure_dir, save_json, select_device  # noqa: E402


DEFAULT = ROOT / "hierarchical_world_model/config/reaction_policy.yaml"


def _metrics(error: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    per_scene = np.divide((error * mask).sum(axis=1), mask.sum(axis=1).clip(1))
    final_error, final_mask = error[:, -1], mask[:, -1]
    return {
        "ADE_m": float(error[mask].mean()), "FDE_m": float(final_error[final_mask].mean()),
        "per_scene_ADE_p50_m": float(np.median(per_scene)), "per_scene_ADE_p90_m": float(np.quantile(per_scene, .9)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit all eligible test factual reconstruction in HighwayEnv.")
    parser.add_argument("--config", type=Path, default=DEFAULT)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    config = load_protocol_config(args.config.resolve())
    base = load_protocol_config(ROOT / config.get("base_config", "hierarchical_world_model/config/release.yaml"))
    device = select_device(config["training"].get("device", "auto"))
    model, _ = load_checkpoint(base["paths"]["evaluation_checkpoint"], device=device)
    experiment, arrays = prepare_experiment_data(base, ROOT), None
    arrays = experiment.bundle.arrays
    candidates = experiment.test_rows
    anchor, present = arrays["agent_states"][candidates, ANCHOR_INDEX], arrays["agent_valid"][candidates, ANCHOR_INDEX]
    eligible = (present[:, 0] & present[:, 2] & (anchor[:, 0, 0] > anchor[:, 2, 0]) &
        (np.abs(anchor[:, 0, 1] - anchor[:, 2, 1]) < 1.8) & (anchor[:, 2, 2] > anchor[:, 0, 2]))
    rows = candidates
    plans = frozen_diffusion_plans(experiment.bundle, rows, checkpoint=base["paths"]["diffusion_checkpoint"],
        output_dir=Path(config["paths"]["output_dir"]) / "cache" / "factual_audit", device=device,
        batch_size=int(base["training"]["validation_batch_size"]), ddim_steps=int(config["training"]["diffusion_ddim_steps"]),
        experiment_scope=base["training"].get("experiment_scope", "full"))
    states, valid = arrays["agent_states"][rows], arrays["agent_valid"][rows]
    plans = complete_missing_background_plans(plans, states, valid)
    errors: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    first_rollout = None
    for start in range(0, len(rows), int(args.batch_size)):
        stop = min(start + int(args.batch_size), len(rows))
        training = {name: value for name, value in config["training"].items()
                    if name in PolicyTrainingConfig.__dataclass_fields__}
        rollout = reaction_controller_rollout(model, states=states[start:stop], valid=valid[start:stop],
            soft_plans=plans[start:stop], maps=arrays["map_polylines"][rows[start:stop]],
            map_valid=arrays["map_polyline_valid"][rows[start:stop]], controller="none",
            device=device, motion_seed=20260902 + start,
            config=PolicyTrainingConfig(**training))
        if first_rollout is None:
            first_rollout = rollout
        errors.append(np.linalg.norm(
            rollout.states[..., :2] - states[start:stop, ANCHOR_INDEX + 1:ANCHOR_INDEX + 150, :, :2],
            axis=-1,
        ))
        masks.append(valid[start:stop, ANCHOR_INDEX + 1:ANCHOR_INDEX + 150])
    # ``highway_controller_rollout.states[:, 0]`` is the state *after* the
    # first command following the anchor has executed.  It therefore aligns
    # with highD ``ANCHOR_INDEX + 1`` rather than the anchor itself.  The old
    # audit compared x[t + 1] with x[t], which creates an artificial error of
    # roughly one 25-Hz travel step (about 1 m at highway speed).
    target_valid = np.concatenate(masks)
    displacement = np.concatenate(errors)
    assert first_rollout is not None
    diagnostic_count = min(32, len(rows), len(first_rollout.states))
    diagnostic_states = states[:diagnostic_count]
    diagnostic_valid = valid[:diagnostic_count]
    factual_target = diagnostic_states[:, ANCHOR_INDEX:ANCHOR_INDEX + 150]
    continuous_valid = diagnostic_valid[:, ANCHOR_INDEX].copy()
    direct_highway = _highway_highd_control_replay(
        diagnostic_states,
        diagnostic_valid,
        np.asarray(arrays["actions_highd"])[rows[:diagnostic_count]],
    )
    direct_offline = _kinematic_highd_control_replay(
        diagnostic_states,
        diagnostic_valid,
        np.asarray(arrays["actions_highd"])[rows[:diagnostic_count]],
    )
    offline = offline_rollout(
        model,
        diagnostic_states,
        diagnostic_valid,
        plans[:diagnostic_count],
        arrays["map_polylines"][rows[:diagnostic_count]],
        arrays["map_polyline_valid"][rows[:diagnostic_count]],
        device=device,
        history_frames=25,
        motion_seed=None,
        controller="none",
        controller_deterministic=True,
        excluded_slots=(),
    )
    collision_free = ~first_rollout.crashed[:diagnostic_count].any(axis=(1, 2))
    closed_loop_error = np.linalg.norm(
        first_rollout.states[:diagnostic_count, :, :, :2] - offline.states[..., :2], axis=-1
    )
    closed_loop_mask = diagnostic_valid[:, ANCHOR_INDEX + 1:ANCHOR_INDEX + 150] & collision_free[:, None, None]
    result = {
        "schema": "highwayenv_factual_reconstruction_audit_v2", "backend": "HighwayEnvClosedLoopWorld",
        "time_contract": {
            "rollout_state_0": "post_step_state_at_anchor_plus_1",
            "target_slice": "ANCHOR_INDEX+1:ANCHOR_INDEX+150",
        },
        "test_sequences": int(len(candidates)), "evaluated_sequences": int(len(rows)),
        "eligible_same_rear_sequences": int(eligible.sum()),
        "controller": "frozen_hiqr", "ego_control": "logged highD ego replay", "horizon_s": 149 * .04,
        "all_background_slots": _metrics(displacement[:, :, 1:], target_valid[:, :, 1:]),
        "same_rear_slot": _metrics(displacement[:, :, 2], target_valid[:, :, 2]),
        "per_slot_ADE_m": {str(slot): float(displacement[:, :, slot][target_valid[:, :, slot]].mean())
            for slot in range(1, 7) if target_valid[:, :, slot].any()},
        "bridge_decomposition": {
            "diagnostic_sequences": int(diagnostic_count),
            "same_logged_controls_highway_vs_offline": _factual_fidelity(direct_highway, direct_offline, continuous_valid),
            "logged_controls_highway_vs_highd": _factual_fidelity(direct_highway, factual_target, continuous_valid),
            "logged_controls_offline_vs_highd": _factual_fidelity(direct_offline, factual_target, continuous_valid),
            "closed_loop_highway_vs_offline": _metrics(closed_loop_error, closed_loop_mask),
        },
        "comparison_note": "This is a HighwayEnv bridge metric and is not interchangeable with the model-internal rollout metric used by historical training summaries.",
    }
    output = ensure_dir(Path(config["paths"]["output_dir"]) / "evaluation")
    save_json(result, output / "highwayenv_factual_reconstruction_test.json")
    print(output / "highwayenv_factual_reconstruction_test.json")


if __name__ == "__main__":
    main()
