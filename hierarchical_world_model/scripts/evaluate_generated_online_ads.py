#!/usr/bin/env python3
"""Execute the existing Flow bank through Diffusion and the published NPC world."""

from dataclasses import fields
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffusion.src.data import states_from_smooth_positions
from hierarchical_world_model.src.composition import DEFAULT_WORLD_SELECTION
from hierarchical_world_model.src.core.utils import load_json, load_yaml, save_json
from hierarchical_world_model.src.execution import (
    _rollout_sample,
    trajectory_event_risk,
)
from idm_ams.src.world_subset_runner import _build_evaluator
from normalizing_flow.src.metrics import physical_validity_flags
from normalizing_flow.src.scenario import (
    ScenarioBatch,
    inverse_constraint,
    project_constraint,
)

BANK = ROOT / "results/flow/samples/generated_samples.npz"
OUTPUT = ROOT / "results/hierarchical_world_model/generated_world_validation"
MOTION_SEED = 1701
DT = 0.04


def plan_diagnostics(sample):
    """Differentiate the actual composed plan; no feasibility projection here."""
    active = sample.initial_valid[:, 1:]
    states = states_from_smooth_positions(
        sample.initial_states[:, 1:], sample.soft_plan
    )
    acceleration = states[..., 4:6]
    jerk = np.diff(acceleration, axis=1) / DT
    mask = np.broadcast_to(active[:, None], acceleration.shape[:-1])
    jerk_mask = mask[:, 1:]
    fractions = {
        "negative_vx_fraction": (states[..., 2] < 0) & mask,
        "ax_outside_minus8_4_fraction": (
            (acceleration[..., 0] < -8) | (acceleration[..., 0] > 4)
        )
        & mask,
        "abs_ay_above4_fraction": (np.abs(acceleration[..., 1]) > 4) & mask,
        "abs_jx_above12_fraction": (np.abs(jerk[..., 0]) > 12) & jerk_mask,
        "abs_jy_above8_fraction": (np.abs(jerk[..., 1]) > 8) & jerk_mask,
    }
    result = {"plan_finite": np.isfinite(states).all(axis=(1, 2, 3))}
    for name, flags in fractions.items():
        denominator = (
            jerk_mask.sum((1, 2)) if name.startswith("abs_j") else mask.sum((1, 2))
        )
        result[name] = flags.sum((1, 2)) / denominator.clip(min=1)
    time = np.arange(1, 150, dtype=np.float32) * DT
    future = np.broadcast_to(
        sample.initial_states[:, None], (len(active), 149, 7, 6)
    ).copy()
    future[:, :, 0, :2] += time[None, :, None] * sample.initial_states[:, None, 0, 2:4]
    future[:, :, 0, 4:6] = 0
    future[:, :, 1:] = states
    result["plan_risk_constant_velocity_ego"] = trajectory_event_risk(
        np.concatenate((sample.initial_states[:, None], future), axis=1),
        sample.initial_valid,
        vehicle_dimensions=sample.vehicle_dimensions,
    )
    return result


@torch.no_grad()
def main():
    torch.set_num_threads(4)
    config_path = ROOT / "idm_ams/configs/world_subset_idm.yaml"
    config = load_yaml(config_path)
    evaluator, artifacts = _build_evaluator(config, config_path.parent)
    sampler = evaluator.sampler
    source = load_json(BANK.parent.parent / "evaluation_summary.json")
    if (ROOT / source["generated_samples"]).resolve() != BANK.resolve():
        raise ValueError("Flow evidence does not refer to the frozen sample bank")
    with np.load(BANK) as saved:
        arrays = {field.name: saved[field.name] for field in fields(ScenarioBatch)}
    raw, _ = inverse_constraint(
        arrays["constraint_normalized_reference"],
        arrays["slot_mask"],
        sampler.flow_schema,
        arrays["c0"],
    )
    arrays["trajectory_constraint"] = project_constraint(
        raw, arrays["slot_mask"], arrays["c0"]
    )
    invalid, reasons, flags = physical_validity_flags(
        arrays["c0"], arrays["slot_mask"], arrays["vehicle_dimensions"]
    )
    total = len(arrays["c0"])
    # Only diffusion/driver/response draws are used; these are not the bank's Flow latents.
    randomness = sampler.sample_world_exogenous(
        total, seed=MOTION_SEED, response_steps=149
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "running",
        "scenes_requested": total,
        "scenes_completed": 0,
        "components": artifacts,
        "world_selection": str(DEFAULT_WORLD_SELECTION),
        "flow_samples": str(BANK),
        "flow_sampling_seed": source["sample_seed"],
        "motion_seed": MOTION_SEED,
        "horizon_s": 5.96,
        "DDIM_steps": sampler.ddim_steps,
        "initial_invalid_scenes": int(invalid.sum()),
        "initial_overlap_scenes": int(flags["overlap"].sum()),
        "initial_invalid_reasons": reasons,
        "flow_evaluation": str(BANK.parent.parent / "evaluation_summary.json"),
        "protocol": "Current EVT-tail Flow sample bank through 68-D residual Diffusion and the frozen world model; ADS application, not trajectory reconstruction evidence.",
        "generated_road": sampler.generated_road,
        "longitudinal_knot_projection": "reachable interval displacements under initial speed and acceleration bounds; generated K only, never observed K_GT",
        "risk_scope": "Plan risk uses constant initial ego velocity; executed risk uses reactive IDM ego. Both use the natural EVT risk definition and sampled dimensions.",
        "limitations": "Generated conditions have no unique trajectory ground truth. Distribution deviations remain reported. This bank is not a Monte Carlo failure-probability estimate; saved Flow base latents are unavailable, so only its concrete conditions are reused.",
    }
    save_json(report, OUTPUT / "report.json")
    records, trajectories, references = [], [], []
    for begin in range(0, total, evaluator.batch_size):
        selected = slice(begin, begin + evaluator.batch_size)
        scenario = ScenarioBatch(
            **{key: value[selected] for key, value in arrays.items()}
        )
        sample = sampler._compose(
            scenario,
            scenario_seed=report["flow_sampling_seed"],
            motion_seed=MOTION_SEED,
            exogenous_state=randomness.select(selected),
        )
        diagnostic = plan_diagnostics(sample)
        rollout = _rollout_sample(
            sampler,
            sample,
            evaluator.policy,
            reaction_controller="auto",
            steps=149,
            evt_model=evaluator.evt_model,
        )
        contacts = rollout.collision_pairs.any(axis=(2, 3))
        npc_valid = sample.initial_valid[:, None, 1:]
        distance = np.linalg.norm(
            rollout.states[:, 1:, 1:, :2] - sample.soft_plan, axis=-1
        )
        for index in range(len(sample.initial_states)):
            hits = np.flatnonzero(contacts[index])
            first_contact = int(hits[0]) if len(hits) else 149
            # Motion summaries exclude collision aftermath but keep the contact event itself.
            precontact = rollout.background_actions[
                index, : first_contact + 1, sample.initial_valid[index, 1:]
            ]
            lateral_acceleration = rollout.states[index, 1 : first_contact + 2, 1:, 5][
                :, sample.initial_valid[index, 1:]
            ]
            records.append(
                {
                    "scene_index": begin + index,
                    "initial_overlap": bool(flags["overlap"][begin + index]),
                    **{key: value[index].item() for key, value in diagnostic.items()},
                    "execution_finite": bool(rollout.numerical_valid[index]),
                    "collision": bool(contacts[index].any()),
                    "first_contact_s": (first_contact + 1) * DT if len(hits) else None,
                    "offroad": bool(
                        (
                            rollout.offroad[index] & sample.initial_valid[index, None]
                        ).any()
                    ),
                    "mean_departure_from_soft_plan_m": float(
                        (distance[index] * npc_valid[index]).sum()
                        / max(int(npc_valid[index].sum()) * 149, 1)
                    ),
                    "precontact_max_abs_acceleration_mps2": float(
                        np.abs(precontact[..., 0]).max(initial=0)
                    ),
                    "precontact_max_abs_yaw_rate_rps": float(
                        np.abs(precontact[..., 1]).max(initial=0)
                    ),
                    "precontact_max_abs_lateral_acceleration_mps2": float(
                        np.abs(lateral_acceleration).max(initial=0)
                    ),
                    "executed_event_risk": float(rollout.event_risk[index]),
                    "executed_evt_score": float(rollout.evt_score[index]),
                }
            )
        trajectories.append(rollout.states)
        references.append(sample.soft_plan)
        report["scenes_completed"] = len(records)
        pd.DataFrame(records).to_csv(OUTPUT / "records.csv", index=False)
        save_json(report, OUTPUT / "report.json")
        print(f"Generated {len(records)}/{total}", flush=True)
    frame = pd.DataFrame(records)
    report.update(
        status="generated_joint_execution_complete",
        finite_plan_scenes=int(frame.plan_finite.sum()),
        finite_execution_scenes=int(frame.execution_finite.sum()),
        collision_scenes=int(frame.collision.sum()),
        offroad_scenes=int(frame.offroad.sum()),
        natural_evt_tail_threshold=evaluator.evt_model.u,
        planned_tail_scenes=int(
            (frame.plan_risk_constant_velocity_ego > evaluator.evt_model.u).sum()
        ),
        executed_tail_scenes=int(
            (frame.executed_event_risk > evaluator.evt_model.u).sum()
        ),
        mean_plan_violation_fractions={
            key: float(frame[key].mean())
            for key in diagnostic
            if key.endswith("_fraction")
        },
        risk_quantiles={
            key: np.quantile(frame[key], [0.1, 0.5, 0.9]).tolist()
            for key in ("plan_risk_constant_velocity_ego", "executed_event_risk")
        },
    )
    np.savez_compressed(
        OUTPUT / "trajectories.npz",
        initial_valid=np.concatenate(
            (np.ones((total, 1), bool), arrays["slot_mask"]), axis=1
        ),
        vehicle_dimensions=arrays["vehicle_dimensions"],
        c0=arrays["c0"],
        trajectory_constraint=arrays["trajectory_constraint"],
        states=np.concatenate(trajectories),
        soft_plan=np.concatenate(references),
    )
    save_json(report, OUTPUT / "report.json")
    print(
        {
            key: report[key]
            for key in (
                "status",
                "scenes_completed",
                "collision_scenes",
                "planned_tail_scenes",
                "executed_tail_scenes",
            )
        }
    )


if __name__ == "__main__":
    main()
