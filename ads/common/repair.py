"""Shared reset curriculum; updates remain native to each learning algorithm."""

import time
from collections import Counter, defaultdict

import numpy as np

from ads.common.environment import PolicyEnvironment


class RepairEnvironment(PolicyEnvironment):
    def __init__(self, algorithm, banks, factory, *, round_index, seed=0, ledger=None):
        self.banks = {name: list(items) for name, items in banks.items()}
        self.category_counts = Counter()
        weights = np.asarray((0.5, 0.5, 0.0) if round_index == 0 else (0.5, 0.3, 0.2))
        self.names = ("normal", "new_failure", "history")
        if not self.banks.get("normal"):
            raise ValueError("repair requires independent, unsearched prior instances")
        self.domain_banks = {}
        all_instances = {}
        for index, name in enumerate(self.names):
            groups = defaultdict(list)
            for instance, domain in self.banks.get(name, []):
                groups[domain].append(instance)
                all_instances[instance.instance_hash] = instance
            self.domain_banks[name] = groups
            if not groups:
                weights[0] += weights[index]
                weights[index] = 0
        self.weights = weights / weights.sum()
        super().__init__(
            algorithm, list(all_instances.values()), factory, seed=seed, ledger=ledger
        )

    def _select_event(self, options):
        if options:
            return super()._select_event(options)
        name = str(self._rng.choice(self.names, p=self.weights))
        groups = self.domain_banks[name]
        if name == "normal":
            instances = self.banks[name]
            instance = instances[int(self._rng.integers(len(instances)))][0]
        else:
            domains = sorted(groups)
            group = groups[domains[int(self._rng.integers(len(domains)))]]
            instance = group[int(self._rng.integers(len(group)))]
        self.category_counts[name] += 1
        return instance.event


def train_discrete(
    algorithm,
    env,
    *,
    base,
    steps,
    seed,
    device,
    replay=None,
    progress=None,
):
    from stable_baselines3 import DQN, PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.utils import get_linear_fn

    wrapper = Monitor(env)
    cls = DQN if algorithm == "dqn" else PPO
    model = cls.load(base, env=wrapper, device=device)
    model.set_random_seed(seed)
    if algorithm == "dqn":
        if replay is not None:
            model.load_replay_buffer(replay)
        else:
            # SB3 compares learning_starts with cumulative timesteps on continuation.
            model.learning_starts = int(model.num_timesteps) + min(
                1024, max(1, steps // 4)
            )
        epsilon = 0.05
        model.exploration_initial_eps = model.exploration_final_eps = epsilon
        model.exploration_schedule = get_linear_fn(epsilon, epsilon, 1.0)
    started = time.perf_counter()
    old_steps = model.num_timesteps

    class ProgressCallback(BaseCallback):
        def _on_step(self):
            done = self.num_timesteps - old_steps
            if progress and (done % 1024 == 0 or done == steps):
                progress(done, steps)
            return True

    model.learn(
        total_timesteps=steps,
        reset_num_timesteps=False,
        callback=ProgressCallback(),
    )
    return model, {
        "elapsed_s": time.perf_counter() - started,
        "requested_steps": steps,
        "actual_steps": int(model.num_timesteps - old_steps),
        "category_resets": dict(env.category_counts),
    }
