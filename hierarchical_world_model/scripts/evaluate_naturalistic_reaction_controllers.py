#!/usr/bin/env python3
"""Four-arm HighwayEnv CRN evaluation for the naturalistic reaction study."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffusion.src.data import ANCHOR_INDEX  # noqa: E402
from hierarchical_world_model.src.data import prepare_experiment_data  # noqa: E402
from hierarchical_world_model.src.human_prior import HumanActionPrior  # noqa: E402
from hierarchical_world_model.src.influence_graph import dynamic_candidate_scene_mask  # noqa: E402
from hierarchical_world_model.src.planner import complete_missing_background_plans, frozen_diffusion_plans  # noqa: E402
from hierarchical_world_model.src.protocol import load_protocol_config  # noqa: E402
from hierarchical_world_model.src.reaction_controller import IDMResidualReactionController, RLResidualReactionController  # noqa: E402
from hierarchical_world_model.src.reaction_ppo import HighwayControllerRollout, highway_controller_rollout  # noqa: E402
from hierarchical_world_model.src.rule_models import RuleModelBundle  # noqa: E402
from hierarchical_world_model.src.train import load_checkpoint  # noqa: E402
from world_model.src.core.utils import ensure_dir, save_json, select_device  # noqa: E402


DEFAULT = ROOT / "hierarchical_world_model/config/reaction_naturalistic.yaml"


def _load_prior(path: Path, device: torch.device) -> HumanActionPrior:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("schema") != "longitudinal_gail_human_prior_v4":
        raise ValueError("evaluation requires HumanActionPriorV4")
    result = HumanActionPrior().to(device); result.load_state_dict(payload["state_dict"]); result.eval()
    return result


def _load_controller(
    name: str,
    config: dict,
    device: torch.device,
    paths: dict[str, Path],
    human_prior_path: Path | None = None,
):
    if name == "A0_none":
        return "none"
    payload = torch.load(paths[name], map_location=device, weights_only=False)
    expected = "reaction_residual_ppo_dynamic_v3" if name == "A3_rl_residual_gail" else "reaction_residual_ppo_dynamic_v2"
    if payload.get("schema") != expected:
        raise ValueError(f"{name} is a legacy fixed-scope checkpoint; retraining is required")
    if name == "A1_rl_residual":
        result = RLResidualReactionController().to(device)
    else:
        rule = RuleModelBundle.load(config["paths"]["rule_model"])
        # A2 receives the frozen prior solely to compute the same conditional
        # KL diagnostic as A3; it has no naturalness reward and its action
        # formula is unchanged.  Its older checkpoint therefore legitimately
        # lacks the read-only prior keys.
        prior = _load_prior(
            Path(config["paths"]["human_prior"])
            if human_prior_path is None else human_prior_path,
            device,
        )
        result = IDMResidualReactionController(rule, prior, apply_human_projection=False).to(device)
    result.load_state_dict(payload["state_dict"], strict=(name != "A2_rl_residual_idm")); result.eval()
    return result


def _w1(left: np.ndarray, right: np.ndarray) -> float:
    a, b = np.sort(left.reshape(-1)), np.sort(right.reshape(-1))
    count = min(len(a), len(b))
    if not count:
        return 0.0
    return float(np.abs(np.interp(np.linspace(0, len(a) - 1, count), np.arange(len(a)), a) - np.interp(np.linspace(0, len(b) - 1, count), np.arange(len(b)), b)).mean())


def _batched_rollout(*, batch_size: int, motion_seed: int, **kwargs) -> HighwayControllerRollout:
    states = kwargs.pop("states")
    valid = kwargs.pop("valid")
    plans = kwargs.pop("soft_plans")
    maps = kwargs.pop("maps")
    map_valid = kwargs.pop("map_valid")
    outputs = []
    for start in range(0, len(states), int(batch_size)):
        stop = min(start + int(batch_size), len(states))
        outputs.append(highway_controller_rollout(
            states=states[start:stop], valid=valid[start:stop],
            soft_plans=plans[start:stop], maps=maps[start:stop],
            map_valid=map_valid[start:stop],
            motion_seed=int(motion_seed) + start, **kwargs,
        ))
    return HighwayControllerRollout(
        states=np.concatenate([item.states for item in outputs]),
        background_actions=np.concatenate([item.background_actions for item in outputs]),
        base_background_actions=np.concatenate([item.base_background_actions for item in outputs]),
        ego_actions=np.concatenate([item.ego_actions for item in outputs]),
        controller_diagnostics={
            name: np.concatenate([item.controller_diagnostics[name] for item in outputs])
            for name in outputs[0].controller_diagnostics
        },
        collision=np.concatenate([item.collision for item in outputs]),
        crashed=np.concatenate([item.crashed for item in outputs]),
    )


def _summary(base, rollout, *, onset: int) -> dict[str, float]:
    # The causal correction is the action actually submitted to HighwayEnv
    # minus the *same-tick* frozen HiQR action.  Comparing to a no-controller
    # trajectory is not a control-effect metric: once a useful brake changes
    # the realized state, HiQR quite legitimately proposes different future
    # base actions.  Keep A0 comparisons for factual equality and outcomes,
    # but measure controller dose at its execution hook.
    del base
    delta = rollout.background_actions[..., 0] - rollout.base_background_actions[..., 0]
    response = delta[:, onset + 1:onset + 11]
    active = rollout.controller_diagnostics["active"].astype(bool)
    response_active = active[:, onset + 1:onset + 11]
    response_values = response[response_active]
    full_target = delta
    first = []
    for value, enabled in zip(full_target, active):
        found = np.argwhere(enabled[onset + 1:] & (value[onset + 1:] < -.1))
        if len(found):
            first.append(float(found[:, 0].min()))
    jerk = np.abs(np.diff(rollout.background_actions[..., 0], axis=1)) / .04
    rule = rollout.controller_diagnostics.get("rule_action_ax", np.zeros_like(delta))
    kl = rollout.controller_diagnostics.get("natural_kl", np.zeros_like(delta))
    unrelated = delta[~active]
    # A command-window result is insufficient: audit every post-command tick
    # in which the already-realized same-rear state remains inside the 4 s
    # release horizon.
    # An action at tick t observes the HighwayEnv state committed at t-1.
    # Align the audit to that causal action boundary; using the state *after*
    # the action would falsely label a newly formed risk as an acceleration
    # rebound that the controller could already have observed.
    ttc = rollout.controller_diagnostics["influence_predicted_ttc_s"]
    post_command = np.arange(delta.shape[1])[None, :, None] >= onset + 1
    unresolved = post_command & active & (ttc < 4.0)
    unresolved_count = int(unresolved.sum())
    final_target = rollout.background_actions[..., 0]
    role = rollout.controller_diagnostics["influence_role"].astype(np.int64)
    controlled_crash = rollout.crashed[:, :, 1:] & active
    role_counts = {str(index): int((active & (role == index)).sum()) for index in (1, 2, 3, 4)}
    return {
        "target_mean_ax_delta_mps2": float(response_values.mean()) if len(response_values) else 0.0,
        "target_brake_rate": float((response_values < -.1).mean()) if len(response_values) else 0.0,
        "target_response_dose_mps2": float((-response_values).clip(0.).mean()) if len(response_values) else 0.0,
        "first_effective_response_delay_frames": float(np.mean(first)) if first else -1.0,
        "effective_response_sequence_rate": float(len(first) / len(full_target)),
        "target_active_rate": float(active.mean()),
        # ``np.diff`` produces jerk on transitions t-1 -> t, so align the
        # authority mask to ticks 1..T-1.  (Using the unsliced mask produces a
        # shape error on the first learned arm with non-empty authority.)
        "target_jerk_p95_mps3": float(np.quantile(jerk[active[:, 1:]], .95)) if active[:, 1:].any() else 0.0,
        "unrelated_max_abs_correction_mps2": float(np.abs(unrelated).max()) if len(unrelated) else 0.0,
        "target_rule_action_mean_mps2": float(rule[active].mean()) if active.any() else 0.0,
        "target_natural_kl_mean": float(kl[active].mean()) if active.any() else 0.0,
        # Split causal rear-follower crashes from an ego collision with its
        # own front vehicle, which a same-rear longitudinal controller cannot
        # prevent and must not be credited for.
        "collision_sequence_rate": float((rollout.crashed[:, :, 0] | controlled_crash.any(2)).any(1).mean()),
        "target_rear_collision_sequence_rate": float(controlled_crash.any(axis=(1, 2)).mean()),
        "ego_collision_sequence_rate": float(rollout.crashed[:, :, 0].any(1).mean()),
        "minimum_predicted_gap_m": float(np.min(rollout.controller_diagnostics["influence_predicted_min_gap_m"][active])) if active.any() else float("inf"),
        "post_command_unresolved_risk_frames": float(unresolved_count),
        "post_command_unresolved_inactive_rate": 0.0,
        "post_command_positive_acceleration_rebound_rate": float((final_target[unresolved] > .05).mean()) if unresolved_count else 0.0,
        "post_command_brake_rate_while_unresolved": float((final_target[unresolved] < -.1).mean()) if unresolved_count else 0.0,
        "role_active_frame_counts": role_counts,
    }


def _bootstrap_difference(left: np.ndarray, right: np.ndarray, *, seed: int = 7) -> dict[str, float]:
    rng, n = np.random.default_rng(seed), len(left)
    samples = np.asarray([(left[rng.integers(n, size=n)] - right[rng.integers(n, size=n)]).mean() for _ in range(500)])
    return {"mean": float((left - right).mean()), "ci95_low": float(np.quantile(samples, .025)), "ci95_high": float(np.quantile(samples, .975))}


def _append_counterfactual_telemetry(
    collection: dict[str, list[np.ndarray]], rollout, *, arm: str, dose: float,
    duration: int, onset: int, max_rows: int | None = None,
) -> None:
    """Persist per-tick evidence rather than only aggregate success metrics.

    The values are taken at the action hook in the actual HighwayEnv rollout.
    This makes it possible to inspect the causal correction separately from
    future HiQR base-action changes caused by the changed realised state.
    """
    # Aggregate metrics always use every rollout state.  This table only
    # supports plots; retaining all arm × dose × duration × scene Unicode
    # labels can consume several GB and prevent a full 1,014-scene report
    # from being written.  Select evenly from the fixed CRN ordering.
    if max_rows is not None and len(rollout.states) > int(max_rows):
        indices = np.linspace(0, len(rollout.states) - 1, int(max_rows), dtype=np.int64)
        states = rollout.states[indices]
        actions = rollout.background_actions[indices]
        base_actions = rollout.base_background_actions[indices]
        diagnostics = {name: value[indices] for name, value in rollout.controller_diagnostics.items()}
    else:
        states = rollout.states
        actions = rollout.background_actions
        base_actions = rollout.base_background_actions
        diagnostics = rollout.controller_diagnostics
    parent = diagnostics["influence_parent"].astype(np.int64)
    parent_safe = np.maximum(parent, 0)
    batch = np.arange(len(states))[:, None, None]
    time = np.arange(states.shape[1])[None, :, None]
    parent_state = states[batch, time, parent_safe]
    child_state = states[:, :, 1:]
    gap = parent_state[..., 0] - child_state[..., 0] - 4.8
    closing = child_state[..., 2] - parent_state[..., 2]
    ttc = diagnostics["influence_predicted_ttc_s"]
    final = actions[..., 0]
    base = base_actions[..., 0]
    jerk = np.concatenate((np.zeros((len(final), 1, 6), np.float32), np.abs(np.diff(final, axis=1)) / .04), axis=1)
    # Store the observed response and recovery interval, not the pre-event
    # frames.  A0 is retained as an exactly matched no-controller reference.
    step = np.arange(final.shape[1])[None, :, None]
    window = (step >= int(onset) + 1) & (step < int(onset) + int(duration) + 26)
    influenced = diagnostics["influence_authority"] > 0.
    valid = influenced & np.isfinite(final) & window
    fields = {
        "arm_code": np.full(final.shape, {"A0_none": 0, "A1_rl_residual": 1,
            "A2_rl_residual_idm": 2, "A3_rl_residual_gail": 3}[arm], np.int8),
        "dose_mps2": np.full(final.shape, float(dose), np.float32),
        "duration_frames": np.full(final.shape, int(duration), np.int16),
        "step": np.broadcast_to(step, final.shape).astype(np.int16),
        "slot": np.broadcast_to(np.arange(6)[None, None], final.shape).astype(np.int8),
        "final_ax_mps2": final,
        "base_ax_mps2": base,
        "correction_ax_mps2": final - base,
        "jerk_mps3": jerk,
        "gap_m": gap,
        "closing_mps": closing,
        "ttc_s": np.clip(ttc, 0., 10.),
        "active": diagnostics["active"].astype(np.int8),
        "authority": diagnostics["influence_authority"],
        "role": diagnostics["influence_role"].astype(np.int8),
        "parent": parent.astype(np.int8),
        "direct": diagnostics["influence_direct"].astype(np.int8),
        "secondary": diagnostics["influence_secondary"].astype(np.int8),
        "alpha": diagnostics["alpha"],
        "residual_ax_mps2": diagnostics["delta_ax"],
        "desired_ax_mps2": diagnostics["desired_action_ax"],
        "rule_ax_mps2": diagnostics["rule_action_ax"],
        "natural_kl": diagnostics["natural_kl"],
        "phase": diagnostics["phase"],
    }
    for name, value in fields.items():
        collection.setdefault(name, []).append(np.asarray(value)[valid])


def _human_reference_telemetry(states: np.ndarray, valid: np.ndarray) -> dict[str, np.ndarray]:
    """Held-out highD same-rear action/jerk samples with causal context."""
    # Central differences are deliberately avoided: actions use the same
    # 25-Hz forward velocity difference as the controller/evaluation path.
    current = states[:, 25:173]
    previous, following = states[:, 24:172], states[:, 26:174]
    current_valid = valid[:, 25:173]
    present = current_valid[:, :, 0] & current_valid[:, :, 2] & valid[:, 26:174, 2] & valid[:, 24:172, 2]
    ego, rear = current[:, :, 0], current[:, :, 2]
    gap = ego[..., 0] - rear[..., 0] - 4.8
    closing = rear[..., 2] - ego[..., 2]
    same_lane = np.abs(ego[..., 1] - rear[..., 1]) < 1.8
    mask = present & same_lane & (gap > .1)
    action = np.clip((following[:, :, 2, 2] - rear[..., 2]) / .04, -8., 4.)
    previous_action = np.clip((rear[..., 2] - previous[:, :, 2, 2]) / .04, -8., 4.)
    ttc = np.where(closing > 1.e-4, gap / np.maximum(closing, 1.e-4), 10.)
    return {
        "final_ax_mps2": action[mask].astype(np.float32),
        "jerk_mps3": (np.abs(action - previous_action) / .04)[mask].astype(np.float32),
        "gap_m": gap[mask].astype(np.float32),
        "closing_mps": closing[mask].astype(np.float32),
        "ttc_s": np.clip(ttc, 0., 10.)[mask].astype(np.float32),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Four-arm A0/A1/A2/A3 naturalistic reaction evaluation.")
    parser.add_argument("--config", type=Path, default=DEFAULT)
    parser.add_argument("--limit", type=int, default=None,
        help="Optional diagnostic cap; omitted means the complete dynamic-eligible split.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--a1-checkpoint", type=Path, default=None,
        help="Pure PPO checkpoint; defaults to the A1 artifact produced by this experiment.")
    parser.add_argument("--a2-checkpoint", type=Path, default=None)
    parser.add_argument("--a3-checkpoint", type=Path, default=None)
    parser.add_argument("--human-prior", type=Path, default=None,
        help="candidate GAIL V4 prior used by A3")
    parser.add_argument("--arms", default="A0,A1,A2,A3",
        help="comma-separated arm subset; fast A3 validation uses A2,A3")
    parser.add_argument("--doses", default="2,4,6,8",
        help="comma-separated positive brake magnitudes in m/s^2")
    parser.add_argument("--durations", default="5,15,25",
        help="comma-separated intervention durations in 25 Hz frames")
    parser.add_argument("--telemetry-max-rows", type=int, default=64,
        help="Per-condition deterministic plot sample; aggregate metrics always use all rows.")
    parser.add_argument("--factual-only", action="store_true",
        help="Run only the complete-split factual arm audit; skip counterfactual rollouts.")
    parser.add_argument("--counterfactual-only", action="store_true",
        help="Run only the complete dynamic-eligible counterfactual conditions.")
    parser.add_argument("--output-dir", type=Path, default=None,
        help="candidate evaluation directory; formal release output is untouched")
    args = parser.parse_args()
    requested_arms = {item.strip() for item in args.arms.split(",") if item.strip()}
    if not requested_arms <= {"A0", "A1", "A2", "A3"} or not requested_arms:
        raise ValueError("--arms must be a nonempty subset of A0,A1,A2,A3")
    doses = tuple(float(item) for item in args.doses.split(",") if item.strip())
    durations = tuple(int(item) for item in args.durations.split(",") if item.strip())
    config = load_protocol_config(args.config.resolve())
    if args.human_prior is not None:
        config["paths"]["human_prior"] = str(args.human_prior)
    base_config = load_protocol_config(ROOT / config.get("base_config", "hierarchical_world_model/config/release.yaml"))
    device = select_device(config["training"].get("device", "auto"))
    model, _ = load_checkpoint(base_config["paths"]["evaluation_checkpoint"], device=device)
    experiment = prepare_experiment_data(base_config, ROOT)
    source_rows = experiment.validation_rows if args.split == "validation" else experiment.test_rows
    arrays = experiment.bundle.arrays
    eligible = dynamic_candidate_scene_mask(
        arrays["agent_states"], arrays["agent_valid"], rows=source_rows,
        radius_m=float(config["training"].get("influence_radius_m", 50.0)),
        prediction_horizon_s=float(config["training"].get("influence_prediction_horizon_s", 1.5)),
    )
    rows = source_rows[eligible]
    if args.limit is not None:
        rows = rows[:int(args.limit)]
    if not len(rows):
        raise RuntimeError("no dynamic candidate test rows")
    # Factual reconstruction is a property of the complete split, whereas
    # counterfactual interventions are evaluated only where the causal graph
    # registers at least one potentially affected NPC.  Keeping these scopes
    # separate prevents a dynamically selected subset from being mistaken for
    # the full factual benchmark.
    factual_rows = source_rows if args.limit is None else source_rows[:int(args.limit)]
    root = Path(config["paths"]["output_dir"]) if args.output_dir is None else args.output_dir
    paths = {
        "A1_rl_residual": args.a1_checkpoint or root / "controllers/rl_residual/reaction_ppo.pt",
        "A2_rl_residual_idm": args.a2_checkpoint or root / "controllers/rl_residual_idm/reaction_ppo.pt",
        "A3_rl_residual_gail": args.a3_checkpoint or root / "controllers/rl_residual_gail/reaction_ppo.pt",
    }
    arm_names = {
        "A0": "A0_none", "A1": "A1_rl_residual",
        "A2": "A2_rl_residual_idm", "A3": "A3_rl_residual_gail",
    }
    controllers = {}
    for arm in ("A0", "A1", "A2", "A3"):
        if arm not in requested_arms:
            continue
        name = arm_names[arm]
        controllers[name] = "none" if arm == "A0" else _load_controller(
            name, config, device, paths,
            human_prior_path=args.human_prior,
        )
    plans = frozen_diffusion_plans(experiment.bundle, rows, checkpoint=base_config["paths"]["diffusion_checkpoint"],
        output_dir=root / "cache" / "test", device=device, batch_size=int(base_config["training"]["validation_batch_size"]),
        ddim_steps=int(config["training"]["diffusion_ddim_steps"]), experiment_scope=base_config["training"].get("experiment_scope", "full"))
    states, valid = arrays["agent_states"][rows], arrays["agent_valid"][rows]
    plans = complete_missing_background_plans(plans, states, valid)
    if args.counterfactual_only:
        factual_rows, factual_states, factual_valid, factual_plans = rows, states, valid, plans
    else:
        factual_plans = frozen_diffusion_plans(
            experiment.bundle, factual_rows,
            checkpoint=base_config["paths"]["diffusion_checkpoint"],
            output_dir=root / "cache" / args.split,
            device=device, batch_size=int(base_config["training"]["validation_batch_size"]),
            ddim_steps=int(config["training"]["diffusion_ddim_steps"]),
            experiment_scope=base_config["training"].get("experiment_scope", "full"),
        )
        factual_states, factual_valid = arrays["agent_states"][factual_rows], arrays["agent_valid"][factual_rows]
        factual_plans = complete_missing_background_plans(factual_plans, factual_states, factual_valid)
    authority = {
        "reaction_min_frames": int(config["training"]["reaction_min_frames"]),
        "reaction_max_frames": int(config["training"]["reaction_max_frames"]),
        "reaction_recovery_frames": int(config["training"]["reaction_recovery_frames"]),
        "reaction_safety_ttc_s": float(config["training"]["safety_ttc_s"]),
        "reaction_release_ttc_s": float(config["training"].get("reaction_release_ttc_s", 4.0)),
        "influence_radius_m": float(config["training"].get("influence_radius_m", 50.0)),
        "influence_secondary_radius_m": float(config["training"].get("influence_secondary_radius_m", 35.0)),
        "influence_prediction_horizon_s": float(config["training"].get("influence_prediction_horizon_s", 1.5)),
        "influence_stable_release_frames": int(config["training"].get("influence_stable_release_frames", 13)),
    }
    factual, report = {}, {
        "schema": "naturalistic_selected_arm_highwayenv_v3", "backend": "HighwayEnvClosedLoopWorld",
        "common_random_numbers": True, "controllers": list(controllers), "split": args.split,
        "full_split_sequences": int(len(source_rows)),
        # ``--counterfactual-only`` skips factual rollouts but must not
        # relabel the complete-split factual population as the dynamic
        # subset.  Keep the population size explicit so a combined report
        # can safely merge this file with the independently generated full
        # factual audit.
        "factual_rows": int(len(source_rows)),
        "counterfactual_rows": int(len(rows)),
        "rows": int(len(rows)), "eligible_rows_in_split": int(eligible.sum()),
        "evaluation_limit": None if args.limit is None else int(args.limit),
        "status": ("complete_dynamic_counterfactual_only" if args.counterfactual_only else
                    ("complete_dynamic_eligible_split" if args.limit is None else "diagnostic_subset")),
        "interventions": {},
    }
    if not args.counterfactual_only:
        for name, controller in controllers.items():
            print(f"[factual] {name} rows={len(factual_rows)}", flush=True)
            factual[name] = _batched_rollout(
                batch_size=args.batch_size, model=model, states=factual_states, valid=factual_valid,
                soft_plans=factual_plans, maps=arrays["map_polylines"][factual_rows],
                map_valid=arrays["map_polyline_valid"][factual_rows], controller=controller,
                device=device, motion_seed=20260902, **authority,
            )
    if not args.counterfactual_only:
        base_rollout = factual.get("A0_none")
        report["natural_exact_base"] = ({name: bool(np.array_equal(item.background_actions, base_rollout.background_actions)) for name, item in factual.items()}
                                        if base_rollout is not None else {})
        factual_target = factual_states[:, ANCHOR_INDEX + 1:ANCHOR_INDEX + 150, :, :2]
        factual_valid = factual_valid[:, ANCHOR_INDEX + 1:ANCHOR_INDEX + 150]
        report["factual_reconstruction"] = {}
        for name, item in factual.items():
            displacement = np.linalg.norm(item.states[..., :2] - factual_target, axis=-1)
            mask = factual_valid[:, :, 1:]
            report["factual_reconstruction"][name] = {
                "background_ADE_m": float(displacement[:, :, 1:][mask].mean()),
                "background_FDE_m": float(displacement[:, -1, 1:][factual_valid[:, -1, 1:]].mean()),
            }
    else:
        base_rollout = None
        report["natural_exact_base"] = {}
        report["factual_reconstruction"] = {}
    telemetry: dict[str, list[np.ndarray]] = {}
    if args.factual_only:
        report["interventions"] = {}
        report["status"] = "complete_full_split_factual_only"
        output = ensure_dir(root / "evaluation")
        save_json(report, output / f"four_arm_comparison_{args.split}.json")
        save_json({"schema": "naturalistic_reaction_telemetry_v1", "backend": "HighwayEnvClosedLoopWorld",
                   "split": args.split, "counterfactual_samples": 0,
                   "counterfactual_telemetry_rows_per_condition": 0,
                   "arm_codes": {"0": "A0_none", "1": "A1_rl_residual", "2": "A2_rl_residual_idm", "3": "A3_rl_residual_gail"},
                   "human_reference_samples": 0, "counterfactual_window": "not run",
                   "human_reference": "not run"}, output / f"telemetry_manifest_{args.split}.json")
        return
    for duration in durations:
        key, per_arm = f"duration_{duration}_frames", {}
        for dose in doses:
            outcomes = {}
            for name, controller in controllers.items():
                print(f"[counterfactual] duration={duration} dose={dose:g} arm={name} rows={len(rows)}", flush=True)
                item = _batched_rollout(
                    batch_size=args.batch_size, model=model, states=states, valid=valid,
                    soft_plans=plans, maps=arrays["map_polylines"][rows],
                    map_valid=arrays["map_polyline_valid"][rows], controller=controller,
                    device=device, motion_seed=20260902, intervention="brake", dose=dose,
                    intervention_duration_frames=duration, **authority,
                )
                outcomes[name] = _summary(base_rollout, item, onset=25)
                _append_counterfactual_telemetry(telemetry, item, arm=name, dose=dose, duration=duration,
                    onset=25, max_rows=args.telemetry_max_rows)
            per_arm[str(int(dose))] = outcomes
        report["interventions"][key] = per_arm
    # Validation-style bootstrap evidence for the isolated IDM contribution.
    report["idm_minus_pure_ppo_bootstrap"] = {}
    if {"A1_rl_residual", "A2_rl_residual_idm"} <= set(controllers):
        for duration in durations:
            pure = _batched_rollout(batch_size=args.batch_size, model=model, states=states, valid=valid,
                soft_plans=plans, maps=arrays["map_polylines"][rows], map_valid=arrays["map_polyline_valid"][rows],
                controller=controllers["A1_rl_residual"], device=device, motion_seed=20260902,
                intervention="brake", dose=6., intervention_duration_frames=duration, **authority)
            idm = _batched_rollout(batch_size=args.batch_size, model=model, states=states, valid=valid,
                soft_plans=plans, maps=arrays["map_polylines"][rows], map_valid=arrays["map_polyline_valid"][rows],
                controller=controllers["A2_rl_residual_idm"], device=device, motion_seed=20260902,
                intervention="brake", dose=6., intervention_duration_frames=duration, **authority)
            report["idm_minus_pure_ppo_bootstrap"][str(duration)] = _bootstrap_difference(
                idm.background_actions[:, :, 1, 0] - idm.base_background_actions[:, :, 1, 0],
                pure.background_actions[:, :, 1, 0] - pure.base_background_actions[:, :, 1, 0])
    output = ensure_dir(root / "evaluation")
    save_json(report, output / f"four_arm_comparison_{args.split}.json")
    np.savez_compressed(output / f"counterfactual_telemetry_{args.split}.npz", **{name: np.concatenate(value) for name, value in telemetry.items()})
    np.savez_compressed(output / f"human_reference_telemetry_{args.split}.npz", **_human_reference_telemetry(states, valid))
    save_json({
        "schema": "naturalistic_reaction_telemetry_v1", "backend": "HighwayEnvClosedLoopWorld",
        "split": args.split, "counterfactual_samples": int(sum(len(item) for item in telemetry["arm_code"])),
        "counterfactual_telemetry_rows_per_condition": int(args.telemetry_max_rows),
        "arm_codes": {"0": "A0_none", "1": "A1_rl_residual", "2": "A2_rl_residual_idm", "3": "A3_rl_residual_gail"},
        "human_reference_samples": int(len(_human_reference_telemetry(states, valid)["final_ax_mps2"])),
        "counterfactual_window": "post-observation intervention plus 1.0 s recovery",
        "human_reference": "held-out highD same-lane same-rear 25 Hz action and jerk samples",
    }, output / f"telemetry_manifest_{args.split}.json")


if __name__ == "__main__":
    main()
