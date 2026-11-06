"""Shared reset curriculum; updates remain native to each learning algorithm."""

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
