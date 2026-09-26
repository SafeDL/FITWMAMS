#!/usr/bin/env python3
"""TrafficBots posterior reconstruction and paired independent-ADS resimulation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from npc_behavior_benchmark.data.causal_cache import load_causal_cache  # noqa: E402
from npc_behavior_benchmark.data.manifest import load_benchmark_config  # noqa: E402
from npc_behavior_benchmark.scripts.evaluate_trafficbots import _batch  # noqa: E402
from external_model_baselines.models.trafficbots.config import load_config as load_tb_config  # noqa: E402
from external_model_baselines.models.trafficbots.evaluation import load_checkpoint  # noqa: E402
from external_model_baselines.models.trafficbots.rollout import (  # noqa: E402
    TrafficBotsHighDRollout,
    logged_ego_controls,
)
from traffic_components.src.core.highd_metrics import factual_metrics  # noqa: E402
from traffic_components.src.core.utils import file_sha256  # noqa: E402


DEFAULT_CONFIG = ROOT / "npc_behavior_benchmark/configs/benchmark_v1.yaml"
DEFAULT_TB_CONFIG = ROOT / "external_model_baselines/models/trafficbots/config/highd.yaml"
DEFAULT_CHECKPOINT = ROOT / "results/baselines/trafficbots_highd/checkpoints/best.ckpt"
DEFAULT_OUTPUT = ROOT / "results/baselines/trafficbots_highd/posterior_resimulation_run.json"


def _same_lane_rear(initial: np.ndarray, active: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ego, background = initial[:, 0], initial[:, 1:]
    dx = background[..., 0] - ego[:, None, 0]
    eligible = active & (dx < -4.8) & (
        np.abs(background[..., 1] - ego[:, None, 1]) < 1.8
    )
    return np.where(eligible, dx, -np.inf).argmax(1), eligible.any(1)


def _collision_rate(states: np.ndarray, active: np.ndarray) -> float:
    valid = np.concatenate((np.ones((len(active), 1), bool), active), axis=1)
    dx = np.abs(states[..., :, None, 0] - states[..., None, :, 0])
    dy = np.abs(states[..., :, None, 1] - states[..., None, :, 1])
    pairs = valid[:, None, :, None] & valid[:, None, None, :]
    upper = np.triu(np.ones((7, 7), bool), 1)[None, None]
    collision = (dx < 4.8) & (dy < 1.8) & pairs & upper
    return float(collision.any((1, 2, 3)).mean())


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> dict:
    benchmark, _ = load_benchmark_config(args.config)
    tb_config = load_tb_config(args.trafficbots_config)
    arrays, metadata, _ = load_causal_cache(args.config)
    split_value = 1 if args.split == "validation" else 2
    rows = np.flatnonzero(metadata["split_index"] == split_value)
    if args.limit:
        rows = rows[: args.limit]
    device = torch.device(
        args.device
        if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    module = load_checkpoint(tb_config, args.checkpoint).to(device).eval()
    runner = TrafficBotsHighDRollout(
        module, common_backend=True
    )
    nominal_states: list[np.ndarray] = []
    treated_states: list[np.ndarray] = []
    target_states: list[np.ndarray] = []
    initial_states: list[np.ndarray] = []
    active_agents: list[np.ndarray] = []
    nominal_actions: list[np.ndarray] = []
    treated_actions: list[np.ndarray] = []
    action_rewrites: list[np.ndarray] = []
    for begin in range(0, len(rows), args.batch_size):
        take = rows[begin : begin + args.batch_size]
        states = np.asarray(arrays["agent_states"][take, 24:174], np.float32)
        valid = np.asarray(arrays["agent_valid"][take, 24:174], bool)
        batch = _batch(
            states,
            valid,
            np.asarray(arrays["map_polylines"][take]),
            np.asarray(arrays["map_polyline_valid"][take]),
        )
        # Full scene data is consumed exactly once by the posterior encoder and
        # GT destination selector.  The resulting episode descriptors are then
        # frozen and shared by both online branches.
        nominal = runner.run(batch, deterministic=True, oracle=True)
        moved = module._move(batch, device)
        controls = logged_ego_controls(
            moved["canonical/states"], moved["canonical/valid"]
        )
        controls = controls.clone()
        controls[:, 25:50, 0] = (
            controls[:, 25:50, 0] - float(args.brake_dose)
        ).clamp_min(-8.0)
        treated = runner.run(
            batch,
            deterministic=True,
            oracle=True,
            ego_controls=controls,
            latent_sample=nominal.latent_sample,
            destination_sample=nominal.destination_sample,
        )
        nominal_states.append(nominal.states.cpu().numpy())
        treated_states.append(treated.states.cpu().numpy())
        nominal_actions.append(nominal.background_actions.cpu().numpy())
        treated_actions.append(treated.background_actions.cpu().numpy())
        requested = nominal.reference_actions.cpu().numpy()
        applied = nominal.background_actions.cpu().numpy()
        action_rewrites.append(np.any(np.abs(requested - applied) > 1.0e-6, -1))
        target_states.append(states[:, 1:])
        initial_states.append(states[:, 0])
        active_agents.append(valid[:, 0, 1:])
    nominal_state = np.concatenate(nominal_states)
    treated_state = np.concatenate(treated_states)
    nominal_action = np.concatenate(nominal_actions)
    treated_action = np.concatenate(treated_actions)
    target = np.concatenate(target_states)
    initial = np.concatenate(initial_states)
    active = np.concatenate(active_agents)
    rewrite = np.concatenate(action_rewrites)
    rear, selected = _same_lane_rear(initial, active)
    selected_rows = np.flatnonzero(selected)
    delta = (
        treated_action[selected_rows, :, rear[selected], 0]
        - nominal_action[selected_rows, :, rear[selected], 0]
    )
    response = delta[:, 25:100]
    reacted = response < -0.05
    latency = np.where(
        reacted.any(1), reacted.argmax(1) * 0.04, np.nan
    )
    report = {
        "schema": "trafficbots_posterior_resimulation_v1",
        "protocol_status": "scene_conditioned_posterior_not_prefix_only_prior",
        "benchmark_id": benchmark["benchmark"]["id"],
        "split": args.split,
        "scenarios": int(len(rows)),
        "background_scope": "all_six",
        "device": str(device),
        "seed": int(args.seed),
        "information_contract": {
            "posterior_latent_inference": "complete reference scene at initialization only",
            "destination": "reference-scene destination frozen at initialization",
            "online_policy_state": "simulated current pose and motion",
            "treated_branch_logged_future_state_overwrite": False,
            "nominal_and_treated_share_latent_and_destination": True,
            "independent_ads": True,
        },
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "factual_reconstruction": factual_metrics(nominal_state, target, active),
        "paired_braking": {
            "dose_mps2": float(args.brake_dose),
            "intervention_frames": [25, 49],
            "eligible_same_lane_rear_scenes": int(selected.sum()),
            "response_rate": float(reacted.any(1).mean()) if len(response) else None,
            "braking_direction_rate": (
                float((response.mean(1) < -0.05).mean()) if len(response) else None
            ),
            "median_latency_s": (
                None
                if not len(latency) or np.isnan(latency).all()
                else float(np.nanmedian(latency))
            ),
            "mean_acceleration_effect_mps2": (
                float(response.mean()) if len(response) else None
            ),
            "median_peak_braking_effect_mps2": (
                float(np.median(response.min(1))) if len(response) else None
            ),
            "all_npc_mean_absolute_action_change_mps2": float(
                np.abs(treated_action[..., 0] - nominal_action[..., 0])[
                    np.broadcast_to(active[:, None], nominal_action[..., 0].shape)
                ].mean()
            ),
        },
        "execution_audit": {
            "finite": bool(
                np.isfinite(nominal_state).all() and np.isfinite(treated_state).all()
            ),
            "nominal_action_rewrite_rate": float(rewrite.mean()),
            "nominal_collision_scene_rate": _collision_rate(nominal_state, active),
            "treated_collision_scene_rate": _collision_rate(treated_state, active),
            "minimum_treated_acceleration_mps2": float(
                treated_action[..., 0][
                    np.broadcast_to(active[:, None], treated_action[..., 0].shape)
                ].min()
            ),
            "maximum_treated_acceleration_mps2": float(
                treated_action[..., 0][
                    np.broadcast_to(active[:, None], treated_action[..., 0].shape)
                ].max()
            ),
        },
        "limitations": [
            (
                "Posterior reconstruction is a scene-conditioned resimulation "
                "diagnostic, not a prior-only deployment score."
            ),
            (
                "The current report uses one deterministic posterior mode and "
                "does not assess posterior calibration."
            ),
            (
                "No logged future state is injected online, but the initialization "
                "latent and destination intentionally encode the reference scene."
            ),
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--trafficbots-config", type=Path, default=DEFAULT_TB_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--brake-dose", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    print(json.dumps(evaluate(args), indent=2))


if __name__ == "__main__":
    main()
