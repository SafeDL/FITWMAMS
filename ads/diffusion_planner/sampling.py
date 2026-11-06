"""Exhaust unseen learning entries before ancestor-balanced review."""

from collections import Counter, defaultdict
from copy import deepcopy

import numpy as np


class BalancedSampler:
    def __init__(
        self,
        samples,
        seed,
        *,
        state=None,
        protected_roots=(),
        key_windows=False,
        critical_visits=2,
        prioritize_regressions=False,
    ):
        self.samples, self.key_windows = samples, key_windows
        self.critical_visits = critical_visits
        self.prioritize_regressions = prioritize_regressions
        self.rng = np.random.default_rng(seed)
        metadata = samples.metadata if hasattr(samples, "metadata") else samples
        self.keys, self.key_indices, self.groups = (
            [],
            set(),
            defaultdict(lambda: defaultdict(list)),
        )
        protected = set(protected_roots)
        self.pools = (
            {"critical": defaultdict(list), "other": defaultdict(list)}
            if key_windows
            else {"all": defaultdict(list)}
        )
        for index, row in enumerate(metadata):
            if (
                row.get("purpose") in ("development", "final_test")
                or row["scene_id"] in protected
            ):
                raise ValueError(
                    "Protected validation ancestor or evaluation material cannot enter learning"
                )
            root, instance = row["scene_id"], row.get("instance_id", row["scene_id"])
            self.keys.append((root, instance, str(row.get("frame_id", index))))
            self.groups[root][instance].append(index)
            critical = bool(row.get("critical_window", False))
            if critical:
                self.key_indices.add(index)
            self.pools[("critical" if critical else "other") if key_windows else "all"][
                root
            ].append(index)
        self.pool_instances = {}
        for pool, groups in self.pools.items():
            self.pool_instances[pool] = {}
            for root, indices in groups.items():
                instances = defaultdict(list)
                for index in indices:
                    instances[self.keys[index][1]].append(index)
                self.pool_instances[pool][root] = dict(instances)
        self.visits, self.queues, self.root_queues = Counter(), {}, {}
        self.coverage_complete = dict.fromkeys(self.pools, False)
        self.review_roots, self.review_instances, self.review_frames = {}, {}, {}
        self.coverage_exposures = 0
        self.priority = {}
        for pool, roots in self.pools.items():
            self.priority[pool] = {
                root: [i for i in indices if metadata[i].get("regression", False)]
                for root, indices in roots.items()
                if any(metadata[i].get("regression", False) for i in indices)
            }
        self.priority_turns, self.priority_exposures = Counter(), 0
        self.external_index = None
        if state:
            if state["keys"] != [list(key) for key in self.keys[: len(state["keys"])]]:
                raise ValueError("Sampler resume material order changed")
            self.rng.bit_generator.state = state["rng"]
            self.visits = Counter({int(k): v for k, v in state["visits"].items()})
            self.coverage_complete = dict(state["coverage_complete"])
            self.queues = deepcopy(state["queues"])
            self.root_queues = deepcopy(state["root_queues"])
            self.review_roots = deepcopy(state["review_roots"])
            self.review_instances = deepcopy(state["review_instances"])
            self.review_frames = deepcopy(state["review_frames"])
            self.coverage_exposures = state["coverage_exposures"]
            self.priority_turns = Counter(state.get("priority_turns", {}))
            self.priority_exposures = state.get("priority_exposures", 0)
            if key_windows and critical_visits > state.get("critical_visits", 2):
                self.coverage_complete["critical"] = False
                self.queues["critical"] = {}
                self.root_queues["critical"] = []
            for pool, groups in self.pools.items():
                for root, indices in groups.items():
                    fresh = [i for i in indices if i >= len(state["keys"])]
                    if fresh:
                        self.coverage_complete[pool] = False
                        self.queues.setdefault(pool, {}).setdefault(root, []).extend(
                            map(int, self.rng.permutation(fresh))
                        )
                        self.root_queues[pool] = []

    def _draw_pool(self, pool):
        groups = self.pools[pool]
        if not groups:
            pool = next(name for name, rows in self.pools.items() if rows)
            groups = self.pools[pool]
        queues = self.queues.setdefault(pool, {})
        if not self.coverage_complete[pool] and not self.root_queues.get(pool):
            pending = [root for root, indices in queues.items() if indices]
            if not pending:
                required = self.critical_visits if pool == "critical" else 1
                queues.update(
                    {
                        root: list(
                            map(
                                int,
                                self.rng.permutation(
                                    [i for i in indices if self.visits[i] < required]
                                ),
                            )
                        )
                        for root, indices in groups.items()
                    }
                )
                pending = [root for root, indices in queues.items() if indices]
            self.root_queues[pool] = list(self.rng.permutation(pending))
            self.coverage_complete[pool] = not bool(pending)
        if not self.coverage_complete[pool]:
            root = self.root_queues[pool].pop()
            index = queues[root].pop()
            self.coverage_exposures += 1
        else:
            if self.prioritize_regressions and self.priority[pool]:
                self.priority_turns[pool] += 1
                if self.priority_turns[pool] % 2:
                    key = "regression:" + pool
                    groups = self.priority[pool]
                    if not self.review_roots.get(key):
                        self.review_roots[key] = list(
                            self.rng.permutation(sorted(groups))
                        )
                    root = self.review_roots[key].pop()
                    frames = groups[root]
                    least = min(self.visits[i] for i in frames)
                    pending = [i for i in frames if self.visits[i] == least]
                    index = int(self.rng.choice(pending))
                    self.priority_exposures += 1
                    self.visits[index] += 1
                    return self.samples[index]
            if not self.review_roots.get(pool):
                self.review_roots[pool] = list(self.rng.permutation(sorted(groups)))
            root = self.review_roots[pool].pop()
            group_key = pool + ":" + root
            instances = self.pool_instances[pool][root]
            if not self.review_instances.get(group_key):
                self.review_instances[group_key] = list(
                    self.rng.permutation(sorted(instances))
                )
            instance = self.review_instances[group_key].pop()
            frame_key = group_key + ":" + instance
            if not self.review_frames.get(frame_key):
                self.review_frames[frame_key] = list(
                    map(
                        int,
                        self.rng.permutation(instances[instance]),
                    )
                )
            index = self.review_frames[frame_key].pop()
        self.visits[index] += 1
        if (
            self.key_windows
            and not self.coverage_complete[pool]
            and not self.root_queues.get(pool)
            and not any(queues.values())
        ):
            required = self.critical_visits if pool == "critical" else 1
            self.coverage_complete[pool] = all(
                self.visits[i] >= required
                for indices in groups.values()
                for i in indices
            )
        return self.samples[index]

    def observe(self, rows):
        if self.external_index is None:
            self.external_index = {key: i for i, key in enumerate(self.keys)}
        for row in rows:
            key = (row["scene_id"], row["instance_id"], str(row["frame_id"]))
            index = self.external_index[key]
            pool = (
                ("critical" if index in self.key_indices else "other")
                if self.key_windows
                else "all"
            )
            required = self.critical_visits if pool == "critical" else 1
            if self.visits[index] < required:
                self.coverage_exposures += 1
            self.visits[index] += 1
            root = key[0]
            queue = self.queues.get(pool, {}).get(root, [])
            if self.visits[index] >= required and index in queue:
                queue.remove(index)
            if not queue and root in self.root_queues.get(pool, []):
                self.root_queues[pool].remove(root)

    def draw(self, count, *, pool=None, prescribed=()):
        if not self.keys:
            raise ValueError("Cannot draw from empty learning material")
        if prescribed:
            self.observe(prescribed)
            if self.key_windows:
                quotas = {"critical": (count + 1) // 2, "other": count // 2}
                for row in prescribed:
                    quotas["critical" if row["critical_window"] else "other"] -= 1
                return list(prescribed) + [
                    self._draw_pool(name)
                    for name, n in quotas.items()
                    for _ in range(n)
                ]
            return list(prescribed) + [
                self._draw_pool("all") for _ in range(count - len(prescribed))
            ]
        if pool is not None:
            rows = []
            for _ in range(count):
                selected = next(
                    (
                        name
                        for name in (pool, *self.pools)
                        if self.pools[name] and not self.coverage_complete[name]
                    ),
                    pool,
                )
                rows.append(self._draw_pool(selected))
            return rows
        if self.key_windows:
            return [
                self._draw_pool("critical" if i < (count + 1) // 2 else "other")
                for i in range(count)
            ]
        return [self._draw_pool("all") for _ in range(count)]

    def state_dict(self):
        return {
            "keys": [list(key) for key in self.keys],
            "rng": deepcopy(self.rng.bit_generator.state),
            "visits": dict(self.visits),
            "coverage_complete": dict(self.coverage_complete),
            "queues": deepcopy(self.queues),
            "root_queues": deepcopy(self.root_queues),
            "review_roots": deepcopy(self.review_roots),
            "review_instances": deepcopy(self.review_instances),
            "review_frames": deepcopy(self.review_frames),
            "coverage_exposures": self.coverage_exposures,
            "critical_visits": self.critical_visits,
            "priority_turns": dict(self.priority_turns),
            "priority_exposures": self.priority_exposures,
        }

    def report(self):
        roots, instances, frames = Counter(), Counter(), Counter()
        for index, count in self.visits.items():
            root, instance, frame = self.keys[index]
            roots[root] += count
            instances[instance] += count
            frames[instance + ":" + frame] += count
        return {
            "exposures": sum(self.visits.values()),
            "regression_review_exposures": self.priority_exposures,
            "coverage_exposures": self.coverage_exposures,
            "ancestor_balanced_exposures": sum(self.visits.values())
            - self.coverage_exposures,
            "available_entries": len(self.keys),
            "visited_entries": len(self.visits),
            "unvisited_entries": len(self.keys) - len(self.visits),
            "completed_entry_passes": min(
                (self.visits[i] for i in range(len(self.keys))), default=0
            ),
            "available_roots": len(self.groups),
            "available_instances": len({k[1] for k in self.keys}),
            "visited_roots": len(roots),
            "visited_instances": len(instances),
            "visited_frames": len(frames),
            "root_visits": dict(roots),
            "instance_visits": dict(instances),
            "frame_visits": dict(frames),
            "available_critical_frames": len(self.key_indices),
            "visited_critical_frames": len(self.key_indices & self.visits.keys()),
            "critical_exposures": sum(self.visits[i] for i in self.key_indices),
            "minimum_critical_visits": min(
                (self.visits[i] for i in self.key_indices), default=0
            ),
            "instances_without_key_exposure": sorted(
                {self.keys[i][1] for i in self.key_indices}
                - {self.keys[i][1] for i in self.key_indices if self.visits[i]}
            ),
        }


def phase_budget(
    native,
    keep,
    corrections,
    *,
    minimum=250,
    exposure=None,
    critical_visits=2,
):
    """Exact entry deficits determine updates; empty key windows transfer quota."""
    metadata = corrections.metadata if hasattr(corrections, "metadata") else corrections
    nk = sum(bool(row.get("critical_window", False)) for row in metadata)
    no = len(corrections) - nk
    critical_quota, other_quota = (
        (16, 16) if nk and no else ((32, 0) if nk else (0, 32))
    )
    exposure = exposure or {}

    def deficit(name, required, count):
        state = exposure.get(name, {})
        visits = Counter({int(k): v for k, v in state.get("visits", {}).items()})
        return sum(max(0, required - visits[i]) for i in range(count))

    correction_visits = Counter(
        {int(k): v for k, v in exposure.get("correction", {}).get("visits", {}).items()}
    )
    key_deficit = sum(
        max(0, critical_visits - correction_visits[i])
        for i, row in enumerate(metadata)
        if row.get("critical_window", False)
    )
    other_deficit = sum(
        max(0, 1 - correction_visits[i])
        for i, row in enumerate(metadata)
        if not row.get("critical_window", False)
    )
    counts = {
        "native": len(native),
        "keep": len(keep),
        "critical": nk,
        "other_correction": no,
    }
    keep_updates = int(np.ceil(deficit("keep", 1, len(keep)) / 32))
    updates = max(
        minimum,
        int(np.ceil(deficit("native", 1, len(native)) / 64)),
        keep_updates,
        int(np.ceil(key_deficit / critical_quota)) if critical_quota else 0,
        int(np.ceil(other_deficit / other_quota)) if other_quota else 0,
    )
    return {
        "updates": updates,
        "entries": counts,
        "critical_quota": critical_quota,
        "other_quota": other_quota,
        "critical_visit_deficit": key_deficit,
        "other_visit_deficit": other_deficit,
    }
