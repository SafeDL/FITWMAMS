"""Bounded failure representatives, executed neighbours and repair feedback."""

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from ads.common.instances import GeneratedInstance


def diverse_representatives(results, *, capacity=16, root_limit=4):
    remaining = sorted(
        {r.instance.instance_hash: r for r in results}.values(),
        key=lambda r: (-r.score, r.instance.instance_hash),
    )
    selected, counts = [], Counter()
    while remaining and len(selected) < capacity:
        eligible = [
            r for r in remaining if counts[r.instance.root_sample_id] < root_limit
        ]
        if not eligible:
            break
        if not selected:
            chosen = eligible[0]
        else:
            descriptions = np.asarray([r.record["descriptor"] for r in selected])
            chosen = max(
                eligible,
                key=lambda r: (
                    float(
                        np.square(descriptions - r.record["descriptor"]).sum(-1).min()
                    ),
                    r.score,
                    r.instance.instance_hash,
                ),
            )
        selected.append(chosen)
        counts[chosen.instance.root_sample_id] += 1
        remaining = [
            r
            for r in remaining
            if r.instance.instance_hash != chosen.instance.instance_hash
        ]
    return selected


class FailureArchive:
    def __init__(self, *, capacity=16, root_limit=4):
        self.capacity, self.root_limit = capacity, root_limit
        self.domains = {}
        self.neighbours = {}
        self.weights = {}
        self.retained = {}

    @property
    def failures(self):
        return [r for group in self.domains.values() for r in group]

    def add(self, results, known):
        groups = defaultdict(list)
        for result in results:
            if result.record.get("execution_kind") == "branch_check":
                continue
            if result.failure:
                groups[result.domain].append(result)
            parent = known.get(result.instance.parent_id)
            if parent is None or result.instance.mutation_block in (None, "joint"):
                continue
            if (
                not np.array_equal(
                    parent.instance.event.valid, result.instance.event.valid
                )
                or parent.failure == result.failure
            ):
                continue
            failure, success = (result, parent) if result.failure else (parent, result)
            self.neighbours.setdefault(failure.instance.instance_hash, success)
        for domain, group in groups.items():
            self.domains[domain] = diverse_representatives(
                self.domains.get(domain, []) + group,
                capacity=self.capacity,
                root_limit=self.root_limit,
            )
        ids = {r.instance.instance_hash for r in self.failures}
        self.neighbours = {k: v for k, v in self.neighbours.items() if k in ids}

    def endpoints(self, limit=None):
        entries = list(
            {
                r.instance.instance_hash: r
                for r in list(self.retained.values())
                + self.failures
                + list(self.neighbours.values())
            }.values()
        )
        if limit is None:
            return entries
        selected, seen = [], set()
        groups = {key: list(value) for key, value in sorted(self.domains.items())}
        while any(groups.values()) and len(selected) < limit:
            for group in groups.values():
                if not group:
                    continue
                failure = group.pop(0)
                for result in (
                    failure,
                    self.neighbours.get(failure.instance.instance_hash),
                ):
                    if result is not None and result.instance.instance_hash not in seen:
                        selected.append(result)
                        seen.add(result.instance.instance_hash)
                        if len(selected) == limit:
                            return selected
        retained_groups = defaultdict(list)
        for result in self.retained.values():
            if result.instance.instance_hash not in seen:
                retained_groups[result.domain].append(result)
        while any(retained_groups.values()) and len(selected) < limit:
            for group in retained_groups.values():
                if group:
                    selected.append(group.pop(0))
                    if len(selected) == limit:
                        break
        return selected

    def retain_previous(self, previous, reevaluated):
        """Keep old endpoints without calling their old labels current failures."""
        current = {r.instance.instance_hash: r for r in reevaluated}
        self.retained = {
            r.instance.instance_hash: current.get(r.instance.instance_hash, r)
            for r in previous.endpoints()
        }

    def feedback(self, reevaluated):
        current = {r.instance.instance_hash: r for r in reevaluated}
        summary = {}
        for domain, failures in self.domains.items():
            old = [r for r in failures if r.instance.instance_hash in current]
            successes = {
                self.neighbours[
                    r.instance.instance_hash
                ].instance.instance_hash: self.neighbours[r.instance.instance_hash]
                for r in old
                if r.instance.instance_hash in self.neighbours
            }
            unresolved = (
                sum(current[r.instance.instance_hash].failure for r in old) / len(old)
                if old
                else 0
            )
            tested_successes = {k: v for k, v in successes.items() if k in current}
            regressed = (
                sum(current[k].failure for k in tested_successes)
                / len(tested_successes)
                if tested_successes
                else 0
            )
            self.weights[domain] = 1 + unresolved + 2 * regressed
            summary[domain] = {
                "old_failures": len(old),
                "success_neighbours": len(tested_successes),
                "unresolved_fraction": unresolved,
                "regression_fraction": regressed,
                "weight": self.weights[domain],
            }
        return summary

    def repair_representatives(self, limit=32):
        groups = {
            k: sorted(v, key=lambda r: (-r.score, r.instance.instance_hash))
            for k, v in self.domains.items()
        }
        chosen, roots = [], set()
        while len(chosen) < limit and any(groups.values()):
            for domain in sorted(groups):
                while groups[domain]:
                    candidate = groups[domain].pop(0)
                    root = candidate.instance.root_sample_id
                    if root not in roots:
                        roots.add(root)
                        chosen.append(candidate)
                        break
                if len(chosen) == limit:
                    break
        return chosen

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        entries = {}
        for result in self.endpoints():
            identity = result.instance.instance_hash
            path = directory / (identity + ".npz")
            if path.exists():
                if GeneratedInstance.load(path).instance_hash != identity:
                    raise ValueError("archive content changed")
            else:
                result.instance.save(path)
            entries[identity] = result.record
        payload = {
            "capacity": self.capacity,
            "root_limit": self.root_limit,
            "weights": self.weights,
            "retained": sorted(self.retained),
            "domains": {
                d: [r.instance.instance_hash for r in rs]
                for d, rs in self.domains.items()
            },
            "neighbours": {
                k: v.instance.instance_hash for k, v in self.neighbours.items()
            },
            "entries": entries,
        }
        (directory / "archive.json").write_text(
            json.dumps(payload, indent=2, allow_nan=False)
        )

    @classmethod
    def load(cls, directory):
        from ads.common.evaluation import EpisodeResult

        directory = Path(directory)
        payload = json.loads((directory / "archive.json").read_text())
        archive = cls(capacity=payload["capacity"], root_limit=payload["root_limit"])
        records = {
            key: EpisodeResult(
                GeneratedInstance.load(directory / (key + ".npz")),
                value,
                np.empty((0, 7, 6)),
                np.empty((0, 2)),
                [],
            )
            for key, value in payload["entries"].items()
        }
        archive.domains = {
            d: [records[key] for key in keys] for d, keys in payload["domains"].items()
        }
        archive.neighbours = {
            key: records[value] for key, value in payload["neighbours"].items()
        }
        archive.weights = payload["weights"]
        archive.retained = {key: records[key] for key in payload["retained"]}
        return archive
