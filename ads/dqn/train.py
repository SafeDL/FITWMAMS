#!/usr/bin/env python3
"""Train or continue a DQN ADS policy in the traffic world."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import gymnasium
import stable_baselines3
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import get_linear_fn

from .env import ADSHighwayEnv
from .events import HighDEventStore
from .provenance import artifact_record
from .runtime import DEFAULT_WORLD_CONFIG, FrozenWorldFactory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--sequence-cache", type=Path, required=True)
    parser.add_argument("--world-config", type=Path, default=DEFAULT_WORLD_CONFIG)
    parser.add_argument(
        "--behavior-randomization", choices=("W0", "W1", "W2"), default="W2"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--continue-from", type=Path)
    parser.add_argument("--replay-buffer", type=Path)
    parser.add_argument("--learning-starts", type=int)
    parser.add_argument("--exploration-fraction", type=float, default=0.3)
    parser.add_argument("--exploration-final-epsilon", type=float, default=0.05)
    parser.add_argument("--continue-exploration-epsilon", type=float, default=0.05)
    parser.add_argument("--checkpoint-frequency", type=int, default=0)
    parser.add_argument("--verbose", type=int, choices=(0, 1, 2), default=0)
    args = parser.parse_args()
    if not 0.0 < args.exploration_fraction <= 1.0:
        parser.error("--exploration-fraction must lie in (0,1]")
    if not 0.0 <= args.exploration_final_epsilon <= 1.0:
        parser.error("--exploration-final-epsilon must lie in [0,1]")
    if not 0.0 <= args.continue_exploration_epsilon <= 1.0:
        parser.error("--continue-exploration-epsilon must lie in [0,1]")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    factory = FrozenWorldFactory(
        args.world_config,
        device=args.device,
        behavior_randomization=args.behavior_randomization,
    )
    events = HighDEventStore(
        args.events,
        args.sequence_cache,
        world_seed=args.seed,
    )
    env = Monitor(ADSHighwayEnv(events, factory, seed=args.seed))
    if args.continue_from:
        model = DQN.load(args.continue_from, env=env, device=args.device)
        if args.replay_buffer:
            model.load_replay_buffer(args.replay_buffer)
        # Keep epsilon fixed when SB3 recomputes the continuation schedule.
        epsilon = float(args.continue_exploration_epsilon)
        model.exploration_initial_eps = epsilon
        model.exploration_final_eps = epsilon
        model.exploration_fraction = 1.0
        model.exploration_schedule = get_linear_fn(epsilon, epsilon, 1.0)
    else:
        learning_starts = (
            args.learning_starts
            if args.learning_starts is not None
            else min(5_000, max(args.steps // 4, 1))
        )
        model = DQN(
            "MlpPolicy",
            env,
            policy_kwargs={"net_arch": [128, 128]},
            learning_rate=1e-4,
            batch_size=128,
            buffer_size=100_000,
            learning_starts=learning_starts,
            gamma=0.99,
            exploration_fraction=args.exploration_fraction,
            exploration_final_eps=args.exploration_final_epsilon,
            seed=args.seed,
            device=args.device,
            verbose=args.verbose,
        )
    callback = (
        None
        if args.checkpoint_frequency < 1
        else CheckpointCallback(
            save_freq=args.checkpoint_frequency,
            save_path=str(args.output_dir / "checkpoints"),
            name_prefix="dqn_ads",
            save_replay_buffer=True,
        )
    )
    started = time.perf_counter()
    model.learn(
        total_timesteps=args.steps,
        reset_num_timesteps=not bool(args.continue_from),
        callback=callback,
    )
    elapsed = time.perf_counter() - started
    checkpoint = args.output_dir / "dqn_ads"
    model.save(checkpoint)
    model.save_replay_buffer(args.output_dir / "replay_buffer.pkl")
    metadata = {
        "algorithm": "dqn",
        "training_steps": args.steps,
        "final_model_timesteps": int(model.num_timesteps),
        "elapsed_s": elapsed,
        "decisions_per_s": float(args.steps / elapsed),
        "seed": args.seed,
        "events": str(args.events.resolve()),
        "world_config": str(args.world_config.resolve()),
        "behavior_randomization": args.behavior_randomization,
        "continue_from": (
            None if args.continue_from is None else str(args.continue_from.resolve())
        ),
        "replay_buffer": (
            None if args.replay_buffer is None else str(args.replay_buffer.resolve())
        ),
        "learning_starts": int(model.learning_starts),
        "exploration_fraction": float(model.exploration_fraction),
        "exploration_final_epsilon": float(model.exploration_final_eps),
        "continue_exploration_epsilon": (
            args.continue_exploration_epsilon if args.continue_from else None
        ),
        "checkpoint_frequency": args.checkpoint_frequency,
        "verbose": args.verbose,
        "observation_shape": list(env.observation_space.shape),
        "action_count": int(env.action_space.n),
        "versions": {
            "torch": torch.__version__,
            "stable_baselines3": stable_baselines3.__version__,
            "gymnasium": gymnasium.__version__,
        },
        "input_artifacts": {
            "events": artifact_record(args.events),
            "world_config": artifact_record(args.world_config),
            "continue_from": artifact_record(args.continue_from),
            "replay_buffer": artifact_record(args.replay_buffer),
        },
    }
    (args.output_dir / "training.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
