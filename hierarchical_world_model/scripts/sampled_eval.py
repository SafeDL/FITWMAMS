#!/usr/bin/env python3
"""Formal sampled Flow -> Diffusion -> HiQR -> HighwayEnv evaluation."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffusion.src.data import ANCHOR_INDEX  # noqa: E402
from hierarchical_world_model.src.composition import HierarchicalWorldSampler  # noqa: E402
from hierarchical_world_model.src.data import prepare_experiment_data  # noqa: E402
from hierarchical_world_model.src.protocol import (  # noqa: E402
    RANDOMNESS_NAMESPACE, SAMPLED_END_TO_END, canonical_hash, load_protocol_config,
    logical_path, long_horizon_constraint, release_provenance,
)
from hierarchical_world_model.src.execution import (  # noqa: E402
    hold_current_ego_action, rollout_world,
)
from hierarchical_world_model.src.randomness import WorldExogenousState  # noqa: E402
from tools.evt import load_evt_model  # noqa: E402
from tools.idm_ego import load_idm_ego_config  # noqa: E402
from world_model.src.core.utils import ensure_dir, load_json, save_json, select_device  # noqa: E402
from world_model.src.core.evaluation_scope import (  # noqa: E402
    evaluation_scope_contract,
    require_scoped_evt_model,
    scoped_canonical_trajectory,
)

CONFIG = ROOT / "hierarchical_world_model/config/release.yaml"


class _IDMPolicy:
    def __init__(self, config: dict[str, float]) -> None:
        self.highway_env_idm_config = config


def _digest(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def _collision(states: np.ndarray, valid: np.ndarray) -> np.ndarray:
    ego, background = states[:, :, :1], states[:, :, 1:]
    present = valid[:, None, 1:]
    return ((np.abs(ego[..., 0] - background[..., 0]) < 4.8) &
            (np.abs(ego[..., 1] - background[..., 1]) < 1.8) & present)


def _risk_summary(result, evt, threshold: float) -> dict[str, object]:
    score = np.asarray(result.evt_score, np.float64)
    risk = np.asarray(result.event_risk, np.float64)
    collision = _collision(result.states, result.initial_valid).any(axis=(1, 2))
    ego, background = result.states[:, :, :1], result.states[:, :, 1:]
    active = result.initial_valid[:, None, 1:]
    longitudinal_gap = np.abs(background[..., 0] - ego[..., 0]) - 4.8
    closing = np.maximum(ego[..., 2] - background[..., 2], 0.0)
    ttc = longitudinal_gap / np.maximum(closing, 1.0e-3)
    minimum_gap = np.where(active, longitudinal_gap, np.inf).min(axis=(1, 2))
    minimum_ttc = np.where(active, ttc, np.inf).min(axis=(1, 2))
    # Worlds with no active background slot have no ego-relative gap/TTC
    # observation.  Exclude them from aggregates and emit a finite sentinel
    # rather than allowing inactive-slot padding to produce ``inf`` metrics.
    gap_observed = np.isfinite(minimum_gap)
    ttc_observed = np.isfinite(minimum_ttc)
    gap_mean = float(np.nanmean(np.where(gap_observed, minimum_gap, np.nan))) if gap_observed.any() else 0.0
    ttc_p05 = float(np.nanquantile(np.where(ttc_observed, minimum_ttc, np.nan), 0.05)) if ttc_observed.any() else 0.0
    return {
        "failure": (score >= threshold), "risk": risk, "score": score,
        "summary": {
            "worlds": int(len(risk)), "finite_state_rate": float(result.numerical_valid.mean()),
            "collision_fraction": float(collision.mean()), "failure_probability": float((score >= threshold).mean()),
            "event_risk_mean": float(risk.mean()), "event_risk_p95": float(np.quantile(risk, .95)),
            "evt_score_mean": float(score.mean()), "evt_score_p95": float(np.quantile(score, .95)),
            "minimum_gap_m_mean": gap_mean, "minimum_ttc_s_p05": ttc_p05,
        },
    }


def _distribution(generated: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    return {"KS": float(ks_2samp(generated, reference).statistic),
            "wasserstein": float(wasserstein_distance(generated, reference))}


def _background_nonpaired(generated: np.ndarray, reference: np.ndarray, generated_valid: np.ndarray, reference_valid: np.ndarray) -> dict[str, object]:
    """Pure background kinematics; intentionally no future-ego metrics."""
    def fields(position: np.ndarray, valid: np.ndarray) -> dict[str, np.ndarray]:
        velocity = np.diff(position, axis=1, prepend=position[:, :1]) / .04
        acceleration = np.diff(velocity, axis=1, prepend=velocity[:, :1]) / .04
        jerk = np.diff(acceleration, axis=1, prepend=acceleration[:, :1]) / .04
        mask = np.broadcast_to(valid[:, None, :], position.shape[:3])
        pair = np.linalg.norm(position[:, :, :, None] - position[:, :, None, :], axis=-1)
        pair_mask = np.broadcast_to(
            valid[:, None, :, None] & valid[:, None, None, :], pair.shape
        )
        diagonal = np.broadcast_to(
            np.eye(position.shape[2], dtype=bool)[None, None], pair.shape
        )
        return {"vx": velocity[..., 0][mask], "vy": velocity[..., 1][mask],
                "ax": acceleration[..., 0][mask], "ay": acceleration[..., 1][mask],
                "jx": jerk[..., 0][mask], "jy": jerk[..., 1][mask],
                "endpoint_x": position[:, -1, :, 0][valid], "endpoint_y": position[:, -1, :, 1][valid],
                "pairwise_bg_distance": pair[pair_mask & ~diagonal]}
    left, right = fields(generated, generated_valid), fields(reference, reference_valid)
    return {name: _distribution(left[name], right[name]) for name in left if len(left[name]) and len(right[name])}


def _k_adherence(
    sample, *, flow_schema: dict[str, object], dt_s: float
) -> dict[str, object]:
    schema = sample.scenario  # retain all physical K values sampled from Flow
    # The sampler's macro reference is the Hermite trajectory through K.  The
    # actual Diffusion realization must be compared at the declared knots.
    frames, times = long_horizon_constraint(flow_schema)
    expected_frames = tuple(SAMPLED_END_TO_END["knot_frames"])
    expected_times = tuple(SAMPLED_END_TO_END["knot_times_s"])
    if frames != expected_frames or not np.allclose(times, expected_times, rtol=0.0, atol=1.0e-6):
        raise ValueError(
            "formal K-adherence requires Flow knot contract "
            f"frames={expected_frames}, times={expected_times}; got {frames}, {times}"
        )
    active = np.asarray(schema.slot_mask, bool)
    plan = np.asarray(sample.soft_plan, np.float32)
    macro = np.asarray(sample.state_knot_reference, np.float32)
    constraints = np.asarray(schema.trajectory_constraint, np.float32)
    constraint_valid = np.asarray(schema.trajectory_constraint_valid, bool)
    initial = np.asarray(sample.initial_states[:, 1:], np.float32)
    if plan.ndim != 4 or plan.shape[-2:] != (active.shape[-1], 2):
        raise ValueError("Diffusion plan must have shape [N,149,6,2]")
    if macro.shape != plan.shape or constraints.shape[:2] != active.shape:
        raise ValueError("sampled K and Diffusion plan shapes are inconsistent")
    if constraints.shape[-1] != 12:
        raise ValueError("trajectory_constraint must encode 3 knots × 4 state fields")
    if constraint_valid.shape != constraints.shape:
        raise ValueError("trajectory_constraint_valid must match trajectory_constraint")
    valid_by_knot = constraint_valid.reshape(constraint_valid.shape[0], constraint_valid.shape[1], 3, 4)
    entries: dict[str, object] = {}
    for knot, (number, time) in enumerate(zip(frames, times)):
        index = number - 1
        if index < 0 or index >= plan.shape[1]:
            raise ValueError("Flow knot index exceeds sampled soft-plan horizon")
        if not np.isclose(number * dt_s, time):
            raise ValueError("Flow knot frame/time contract is inconsistent")
        # Per-slot K validity is the slot mask plus finite all four values at
        # this knot; inactive zero padding never enters the denominator.
        offset = knot * 4
        knot_valid = active & valid_by_knot[:, :, knot, :].all(axis=-1)
        target_position = macro[:, index]
        generated_velocity = (plan[:, index] - (initial[:, :, :2] if index == 0 else plan[:, index - 1])) / dt_s
        target_velocity = initial[:, :, 2:4] + constraints[:, :, offset + 2:offset + 4]
        position_error = np.linalg.norm(plan[:, index] - target_position, axis=-1)
        velocity_error = np.linalg.norm(generated_velocity - target_velocity, axis=-1)
        entries[str(time)] = {
            "frame": number, "zero_based_plan_index": index,
            "valid_slots": int(knot_valid.sum()),
            "position_error_m": float(position_error[knot_valid].mean()) if knot_valid.any() else None,
            "velocity_error_mps": float(velocity_error[knot_valid].mean()) if knot_valid.any() else None,
        }
    return {"knot_frames": list(frames), "knot_times_s": list(times), "per_knot": entries}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the formal sampled end-to-end audit.")
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--release-tag", required=True)
    parser.add_argument("--release-session", type=Path)
    parser.add_argument("--worlds", type=int, default=SAMPLED_END_TO_END["worlds"])
    parser.add_argument("--steps", type=int, default=SAMPLED_END_TO_END["response_steps"])
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--idm-config", type=Path, default=ROOT / "tools/idm_ego.yaml")
    args = parser.parse_args()
    if args.worlds != SAMPLED_END_TO_END["worlds"] or args.steps != SAMPLED_END_TO_END["response_steps"]:
        raise ValueError(
            f"formal sampled evaluation requires exactly "
            f"{SAMPLED_END_TO_END['worlds']} worlds and "
            f"{SAMPLED_END_TO_END['response_steps']} steps"
        )
    config = load_protocol_config(args.config)
    provenance = (
        load_json(args.release_session)
        if args.release_session is not None
        else release_provenance(release_tag=args.release_tag, require_clean=True)
    )
    if provenance.get("release_tag") != args.release_tag or not provenance.get("worktree_clean_at_start"):
        raise RuntimeError("release session does not certify this clean release tag")
    if release_provenance().get("code_commit") != provenance.get("code_commit"):
        raise RuntimeError("release session commit no longer matches HEAD")
    device = select_device(config["training"].get("device", "auto"))
    output = ensure_dir(config["paths"]["output_dir"])
    sampler = HierarchicalWorldSampler(
        flow_checkpoint=config["paths"]["flow_checkpoint"], flow_output_dir=config["paths"]["flow_output_dir"],
        diffusion_checkpoint=config["paths"]["diffusion_checkpoint"], diffusion_contract=config["paths"]["diffusion_contract"],
        response_checkpoint=config["paths"]["evaluation_checkpoint"], repo_root=ROOT, device=device,
    )
    exogenous = sampler.sample_world_exogenous(args.worlds, seed=args.seed, response_steps=args.steps)
    exogenous_path = output / "final" / "sampled_end_to_end_exogenous.npz"
    exogenous.save(exogenous_path)
    require_scoped_evt_model(config["paths"]["evt_model"])
    evt = load_evt_model(config["paths"]["evt_model"])
    threshold = float(evt.score(evt.return_level(100)))
    policies = {"hold_current": hold_current_ego_action, "idm": _IDMPolicy(load_idm_ego_config(args.idm_config))}
    values = {name: [] for name in policies}
    sampled_parts = {"soft_plan": [], "state_knot_reference": [], "initial_states": [], "slot_mask": [], "trajectory_constraint": [], "trajectory_constraint_valid": []}
    for start in range(0, args.worlds, args.batch_size):
        subset = exogenous.select(slice(start, min(start + args.batch_size, args.worlds)))
        sample = sampler.compose_exogenous(subset)
        sampled_parts["soft_plan"].append(sample.soft_plan)
        sampled_parts["state_knot_reference"].append(sample.state_knot_reference)
        sampled_parts["initial_states"].append(sample.initial_states)
        sampled_parts["slot_mask"].append(sample.scenario.slot_mask)
        sampled_parts["trajectory_constraint"].append(sample.scenario.trajectory_constraint)
        sampled_parts["trajectory_constraint_valid"].append(sample.scenario.trajectory_constraint_valid)
        for name, policy in policies.items():
            values[name].append(rollout_world(sampler, subset, policy, steps=args.steps, evt_model=evt))
    # Concatenate only report-level vectors; trajectories remain in the NPZ-replayable world input.
    reports = {}
    paired = {}
    for name, chunks in values.items():
        class Result: pass
        merged = Result()
        for field in ("states", "initial_valid", "event_risk", "evt_score", "numerical_valid"):
            setattr(merged, field, np.concatenate([getattr(value, field) for value in chunks]))
        reports[name] = _risk_summary(merged, evt, threshold)
    hold_failed, idm_failed = reports["hold_current"]["failure"], reports["idm"]["failure"]
    paired = {"both_safe": int((~hold_failed & ~idm_failed).sum()), "idm_only_failure": int((~hold_failed & idm_failed).sum()),
              "hold_only_failure": int((hold_failed & ~idm_failed).sum()), "both_failure": int((hold_failed & idm_failed).sum())}
    hold_risk = np.asarray(reports["hold_current"]["risk"], np.float64)
    idm_risk = np.asarray(reports["idm"]["risk"], np.float64)
    paired_worlds = {
        "R_hold": hold_risk.tolist(),
        "R_IDM": idm_risk.tolist(),
        "Delta_R_IDM_minus_hold": (idm_risk - hold_risk).tolist(),
    }
    sample = SimpleNamespace(
        soft_plan=np.concatenate(sampled_parts["soft_plan"]),
        state_knot_reference=np.concatenate(sampled_parts["state_knot_reference"]),
        initial_states=np.concatenate(sampled_parts["initial_states"]),
        scenario=SimpleNamespace(slot_mask=np.concatenate(sampled_parts["slot_mask"]), trajectory_constraint=np.concatenate(sampled_parts["trajectory_constraint"]), trajectory_constraint_valid=np.concatenate(sampled_parts["trajectory_constraint_valid"])),
    )
    experiment = prepare_experiment_data(config, ROOT)
    rows = experiment.test_rows[:args.worlds]
    target_all = np.asarray(experiment.bundle.arrays["agent_states"])[rows]
    target_valid_all = np.asarray(experiment.bundle.arrays["agent_valid"])[rows]
    target_all, target_valid_all = scoped_canonical_trajectory(
        target_all, target_valid_all
    )
    target_states = target_all[:, ANCHOR_INDEX + 1:174, 1:]
    target = target_states[..., :2]
    target_valid = target_valid_all[:, ANCHOR_INDEX + 1:174, 1:].all(axis=1)
    generated = np.asarray(sample.soft_plan)
    nonpaired = {"background_kinematic_distribution": _background_nonpaired(generated, target, sample.scenario.slot_mask, target_valid),
                 "k_adherence": _k_adherence(sample, flow_schema=sampler.flow_schema, dt_s=.04)}
    report = {"schema": SAMPLED_END_TO_END["schema"], "evaluation_scope": evaluation_scope_contract(), "worlds": args.worlds, "response_steps": args.steps,
              "rng_schema": RANDOMNESS_NAMESPACE["world"], "exogenous": {"path": logical_path(exogenous_path), "digest": _digest(exogenous.agent_response_innovations)},
              "threshold": {"evt_return_period_segments": 100, "evt_score": threshold},
              "ADS_conditioned_sampled_world_risk": {name: item["summary"] for name, item in reports.items()},
              "paired_failure_table": paired,
              "paired_world_risk": paired_worlds,
              "sampled_K_to_diffusion_nonpaired_fidelity": nonpaired,
              "provenance": {**provenance, "config_sha256": canonical_hash(config), "checkpoint": logical_path(config["paths"]["evaluation_checkpoint"])} }
    save_json(report, output / "sampled_end_to_end.json")


if __name__ == "__main__":
    main()
