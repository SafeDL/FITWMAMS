"""Continue one frozen-base ADS policy on a controlled event mixture."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import get_linear_fn

from ads.common.environment import ADSHighwayEnv
from ads.common.instances import artifact_record
from ads.common.world import (
    DEFAULT_WORLD_CONFIG,
    FrozenWorldFactory,
    HighDEventStore,
    WorldEvent,
)


@dataclass(frozen=True)
class EventMixture:
    events: tuple[WorldEvent, ...]
    categories: tuple[str, ...]


def build_event_mixture(
    normal: Sequence[WorldEvent],
    new_failures: Sequence[WorldEvent],
    history: Sequence[WorldEvent],
    *,
    ratios: tuple[float, float, float] = (0.5, 0.3, 0.2),
    size: int = 1000,
    seed: int = 0,
    forbidden_source_ids: set[str] | None = None,
) -> EventMixture:
    groups = (normal, new_failures, history)
    names = ("normal", "new_failure", "history")
    weights = np.asarray(ratios, np.float64)
    if len(weights) != 3 or np.any(weights < 0.0) or not np.isclose(weights.sum(), 1.0):
        raise ValueError(
            "event mixture ratios must be three non-negative values summing to one"
        )
    if int(size) < 1:
        raise ValueError("mixture size must be positive")
    for name, group, weight in zip(names, groups, weights):
        if weight > 0.0 and not len(group):
            raise ValueError(f"{name} event group is empty but has non-zero weight")
    forbidden = forbidden_source_ids or set()
    leaking = {
        event.source_event_id
        for group in groups
        for event in group
        if event.source_event_id in forbidden or event.split == "test"
    }
    if leaking:
        raise ValueError(
            f"repair mixture contains held-out source events: {sorted(leaking)[:5]}"
        )
    counts = np.floor(weights * int(size)).astype(int)
    counts[np.argmax(weights)] += int(size) - int(counts.sum())
    rng = np.random.default_rng(int(seed))
    events: list[WorldEvent] = []
    categories: list[str] = []
    for name, group, count in zip(names, groups, counts):
        if count == 0:
            continue
        indices = rng.integers(0, len(group), size=int(count))
        events.extend(group[int(index)] for index in indices)
        categories.extend((name,) * int(count))
    order = rng.permutation(len(events))
    return EventMixture(
        tuple(events[int(index)] for index in order),
        tuple(categories[int(index)] for index in order),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-policy", type=Path, required=True)
    parser.add_argument("--normal-events", type=Path, required=True)
    parser.add_argument("--new-events", type=Path, required=True)
    parser.add_argument("--history-events", type=Path, required=True)
    parser.add_argument("--sequence-cache", type=Path, required=True)
    parser.add_argument("--world-config", type=Path, default=DEFAULT_WORLD_CONFIG)
    parser.add_argument(
        "--behavior-randomization", choices=("W0", "W1", "W2"), default="W2"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--mixture-size", type=int, default=1000)
    parser.add_argument("--ratios", type=float, nargs=3, default=(0.5, 0.3, 0.2))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--replay-buffer", type=Path)
    parser.add_argument("--verbose", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--dqn-exploration-epsilon", type=float, default=0.05)
    args = parser.parse_args()
    if not 0.0 <= args.dqn_exploration_epsilon <= 1.0:
        parser.error("--dqn-exploration-epsilon must lie in [0,1]")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    factory = FrozenWorldFactory(
        args.world_config,
        device=args.device,
        behavior_randomization=args.behavior_randomization,
    )

    def store(path: Path):
        return HighDEventStore(
            path,
            args.sequence_cache,
            world_seed=args.seed,
        )

    normal, new, history = (
        store(args.normal_events),
        store(args.new_events),
        store(args.history_events),
    )
    mixture = build_event_mixture(
        normal,
        new,
        history,
        ratios=tuple(args.ratios),
        size=args.mixture_size,
        seed=args.seed,
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
        "world_config": str(args.world_config.resolve()),
        "behavior_randomization": args.behavior_randomization,
        "verbose": args.verbose,
        "dqn_exploration_epsilon": args.dqn_exploration_epsilon,
        "input_artifacts": {
            "base_policy": artifact_record(args.base_policy),
            "normal_events": artifact_record(args.normal_events),
            "new_events": artifact_record(args.new_events),
            "history_events": artifact_record(args.history_events),
            "world_config": artifact_record(args.world_config),
            "replay_buffer": artifact_record(args.replay_buffer),
        },
    }
    (args.output_dir / "repair.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
