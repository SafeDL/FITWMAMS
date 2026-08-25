#!/usr/bin/env python3
"""Audit the complete Flow -> Diffusion -> HiQR -> ADS -> EVT interface."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hierarchical_traffic_world_model.src.composition import HierarchicalWorldSampler  # noqa: E402
from hierarchical_traffic_world_model.src.world_execution import (  # noqa: E402
    hold_current_ego_action,
    rollout_world,
)
from hierarchical_traffic_world_model.src.world_randomness import WorldExogenousState  # noqa: E402
from hierarchical_traffic_world_model.src.train import load_checkpoint  # noqa: E402
from tools.evt import load_evt_model  # noqa: E402
from world_model.src.core.utils import ensure_dir, load_yaml, save_json, select_device  # noqa: E402

CONFIG = ROOT / "hierarchical_traffic_world_model/configs/highd_stochastic_causal_hiqr_full.yaml"


def _digest(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def _brake_policy(observation: dict[str, torch.Tensor | int]) -> torch.Tensor:
    action = hold_current_ego_action(observation).clone()
    action[:, 0] = (action[:, 0] - 2.0).clamp_min(-8.0)
    return action


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit the world/ADS/EVT interface for an external AMS runner; "
            "does not run subset simulation."
        )
    )
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--worlds", type=int, default=2)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260823)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = load_yaml(config_path)
    device = select_device(config["training"].get("device", "auto"))
    output = ensure_dir(Path(config["paths"]["output_dir"]) / "final")
    sampler = HierarchicalWorldSampler(
        flow_checkpoint=config["paths"]["flow_checkpoint"],
        flow_output_dir=config["paths"]["flow_output_dir"],
        diffusion_checkpoint=config["paths"]["diffusion_checkpoint"],
        diffusion_contract=config["paths"]["diffusion_contract"],
        response_checkpoint=config["paths"]["evaluation_checkpoint"],
        repo_root=ROOT,
        device=device,
    )
    exogenous = sampler.sample_world_exogenous(
        int(args.worlds),
        seed=int(args.seed),
        response_steps=int(args.steps),
    )
    serialized = output / "ams_readiness_world_exogenous.npz"
    exogenous.save(serialized)
    reloaded = WorldExogenousState.load(serialized)
    evt = load_evt_model(config["paths"]["evt_model"])
    snapshot_sample = sampler.compose_exogenous(exogenous)
    snapshot_world = sampler.create_world(snapshot_sample)
    snapshot_action = hold_current_ego_action(snapshot_world.observe())[:, None]
    snapshot_world.advance_response(snapshot_action)
    snapshot = snapshot_world.snapshot()
    expected_next = snapshot_world.advance_response(
        hold_current_ego_action(snapshot_world.observe())[:, None]
    )["agent_state_frames"]
    snapshot_world.restore(snapshot)
    actual_next = snapshot_world.advance_response(
        hold_current_ego_action(snapshot_world.observe())[:, None]
    )["agent_state_frames"]
    first = rollout_world(
        sampler,
        exogenous,
        hold_current_ego_action,
        steps=args.steps,
        evt_model=evt,
    )
    replay = rollout_world(
        sampler,
        reloaded,
        hold_current_ego_action,
        steps=args.steps,
        evt_model=evt,
    )
    branch = rollout_world(
        sampler,
        exogenous,
        _brake_policy,
        steps=args.steps,
        evt_model=evt,
    )
    response_mutated = exogenous.pcn_mutate(
        "agent_response_innovations",
        beta=0.35,
        seed=args.seed + 1,
    )
    randomized = rollout_world(
        sampler,
        response_mutated,
        hold_current_ego_action,
        steps=args.steps,
        evt_model=evt,
    )
    checkpoint_model, payload = load_checkpoint(
        config["paths"]["evaluation_checkpoint"],
        device=device,
    )
    configured = config["model"]
    config_match = checkpoint_model.cfg.to_dict() == type(checkpoint_model.cfg)(
        **configured
    ).to_dict()
    acceleration = first.background_actions[..., 0]
    yaw_rate = first.background_actions[..., 1]
    risk_probe = evt.score(np.asarray([0.0, evt.u, evt.return_level(100)], np.float64))
    report = {
        "scope": (
            "small-sample AMS readiness audit; it is an interface contract, not a "
            "probability estimate"
        ),
        "worlds": int(args.worlds),
        "steps": int(args.steps),
        "seed": int(args.seed),
        "formal_checkpoint_config_match": bool(config_match),
        "checkpoint_epoch": int(payload["epoch"]),
        "world_serialization_exact": bool(
            all(
                np.array_equal(exogenous.as_dict()[key], reloaded.as_dict()[key])
                for key in exogenous.as_dict()
            )
        ),
        "same_world_same_ads_exact": bool(
            np.array_equal(first.states, replay.states)
            and np.array_equal(first.evt_score, replay.evt_score)
        ),
        "snapshot_restore_exact": bool(torch.equal(expected_next, actual_next)),
        "common_random_number_branch_reuses_exogenous_digest": _digest(exogenous.diffusion_noise),
        "branch_changes_ego_trajectory": bool(
            not np.array_equal(first.states[:, :, 0], branch.states[:, :, 0])
        ),
        "finite_state_rate": float(np.isfinite(first.states).all(axis=(1, 2, 3)).mean()),
        "finite_evt_score_rate": float(np.isfinite(first.evt_score).mean()),
        "background_acceleration_bounds": [float(acceleration.min()), float(acceleration.max())],
        "background_yaw_rate_bounds": [float(yaw_rate.min()), float(yaw_rate.max())],
        "maximum_speed_mps": float(np.linalg.norm(first.states[..., 2:4], axis=-1).max()),
        "event_risk": first.event_risk.tolist(), "evt_score": first.evt_score.tolist(),
        "evt_score_monotone_on_calibration_probe": bool(np.all(np.diff(risk_probe) >= 0.0)),
        "response_risk_variance_under_pcn_mutation": float(
            np.var(np.concatenate((first.event_risk, randomized.event_risk)))
        ),
        "ads_caused_collision_is_retained_as_valid": True,
    }
    report["all_passed"] = bool(
        report["formal_checkpoint_config_match"]
        and report["world_serialization_exact"]
        and report["same_world_same_ads_exact"]
        and report["snapshot_restore_exact"]
        and report["branch_changes_ego_trajectory"]
        and report["finite_state_rate"] == 1.0 and report["finite_evt_score_rate"] == 1.0
        and report["evt_score_monotone_on_calibration_probe"]
    )
    save_json(report, output / "ams_readiness.json")
    if not report["all_passed"]:
        raise RuntimeError("AMS readiness audit failed; inspect ams_readiness.json")


if __name__ == "__main__":
    main()
