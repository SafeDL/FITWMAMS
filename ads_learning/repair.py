#!/usr/bin/env python3
"""Continue one frozen-base ADS policy on a controlled event mixture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import get_linear_fn

from .env import ADSHighwayEnv
from .event_sampler import build_event_mixture
from .events import HighDEventStore
from .provenance import artifact_record
from .runtime import FrozenWorldFactory
from .verify_protocol import verify_protocol_lock


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-policy", type=Path, required=True)
    parser.add_argument("--normal-events", type=Path, required=True)
    parser.add_argument("--new-events", type=Path, required=True)
    parser.add_argument("--history-events", type=Path, required=True)
    parser.add_argument("--sequence-cache", type=Path, required=True)
    parser.add_argument("--world-checkpoint", type=Path, required=True)
    parser.add_argument("--response-gate", type=Path)
    parser.add_argument("--behavior-randomization", choices=("W0", "W1", "W2"), default="W2")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--mixture-size", type=int, default=1000)
    parser.add_argument("--ratios", type=float, nargs=3, default=(0.5, 0.3, 0.2))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--replay-buffer", type=Path)
    parser.add_argument("--verbose", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--dqn-exploration-epsilon", type=float, default=0.05)
    parser.add_argument("--protocol-lock", type=Path)
    args = parser.parse_args()
    protocol = (
        None
        if args.protocol_lock is None
        else verify_protocol_lock(
            args.protocol_lock, Path(__file__).resolve().parents[1]
        )
    )
    if protocol is not None and not protocol["verified"]:
        raise RuntimeError(f"protocol lock mismatch: {protocol['mismatches'][:3]}")
    if not 0.0 <= args.dqn_exploration_epsilon <= 1.0:
        parser.error("--dqn-exploration-epsilon must lie in [0,1]")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    factory = FrozenWorldFactory(
        args.world_checkpoint,
        device=args.device,
        response_gate_checkpoint=args.response_gate,
        behavior_randomization=args.behavior_randomization,
    )

    def store(path: Path):
        return HighDEventStore(
            path, args.sequence_cache, world_seed=args.seed,
            scene_dim=factory.model.cfg.scene_latent_dim,
            agent_dim=factory.model.cfg.agent_latent_dim,
        )

    normal, new, history = store(args.normal_events), store(args.new_events), store(args.history_events)
    mixture = build_event_mixture(
        normal, new, history, ratios=tuple(args.ratios),
        size=args.mixture_size, seed=args.seed,
    )
    env = Monitor(ADSHighwayEnv(mixture.events, factory, seed=args.seed))
    model = DQN.load(args.base_policy, env=env, device=args.device)
    model.set_random_seed(args.seed)
    model.verbose = args.verbose
    if args.replay_buffer:
        model.load_replay_buffer(args.replay_buffer)
    epsilon = float(args.dqn_exploration_epsilon)
    model.exploration_initial_eps = epsilon
    model.exploration_final_eps = epsilon
    model.exploration_fraction = 1.0
    model.exploration_schedule = get_linear_fn(epsilon, epsilon, 1.0)
    started = time.perf_counter()
    model.learn(total_timesteps=args.steps, reset_num_timesteps=False)
    elapsed = time.perf_counter() - started
    policy_path = args.output_dir / "dqn_repaired"
    model.save(policy_path)
    model.save_replay_buffer(args.output_dir / "replay_buffer.pkl")
    manifest = {
        "base_policy": str(args.base_policy.resolve()),
        "algorithm": "dqn",
        "steps": args.steps,
        "final_model_timesteps": int(model.num_timesteps),
        "elapsed_s": elapsed,
        "decisions_per_s": float(args.steps / elapsed),
        "seed": args.seed,
        "mixture_size": args.mixture_size,
        "ratios": list(args.ratios),
        "category_counts": {
            name: mixture.categories.count(name)
            for name in ("normal", "new_failure", "history")
        },
        "world_checkpoint": str(args.world_checkpoint.resolve()),
        "response_gate": None if args.response_gate is None else str(args.response_gate.resolve()),
        "behavior_randomization": args.behavior_randomization,
        "verbose": args.verbose,
        "dqn_exploration_epsilon": args.dqn_exploration_epsilon,
        "input_artifacts": {
            "base_policy": artifact_record(args.base_policy),
            "normal_events": artifact_record(args.normal_events),
            "new_events": artifact_record(args.new_events),
            "history_events": artifact_record(args.history_events),
            "world_checkpoint": artifact_record(args.world_checkpoint),
            "response_gate": artifact_record(args.response_gate),
            "replay_buffer": artifact_record(args.replay_buffer),
        },
        "protocol": protocol,
    }
    (args.output_dir / "repair.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
