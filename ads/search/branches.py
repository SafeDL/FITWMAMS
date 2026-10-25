"""NSuS hill–valley partitions in the frozen generator's base-randomness space."""

from dataclasses import dataclass

import networkx as nx
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score
from sklearn.svm import LinearSVC

from hierarchical_world_model.src.randomness import WORLD_RANDOM_BLOCKS


def coordinates(instance):
    state = instance.event.exogenous_state
    values = []
    for name in WORLD_RANDOM_BLOCKS:
        array = getattr(state, name)
        if name == "policy_calibration_innovations":
            array = array[:, :1]
        values.append(array.reshape(-1).astype(np.float64))
    return np.concatenate(values)


def midpoint(left, right):
    a = left.event.exogenous_state.as_dict()
    b = right.event.exogenous_state.as_dict()
    return left.event.exogenous_state.replace_arrays(
        {
            name: ((a[name].astype(np.float64) + b[name]) / 2).astype(a[name].dtype)
            for name in WORLD_RANDOM_BLOCKS
        }
    )


def representatives(population, limit, *, seed=20260926):
    # Do not use a global risk filter to pick the graph used to protect weak
    # branches. As in the NSuS implementation, balance ancestry/chains then sample.
    unique = {r.instance.instance_hash: r for r in population}
    remaining = [unique[key] for key in sorted(unique)]
    rng = np.random.default_rng(seed)
    priorities = {r.instance.instance_hash: float(rng.random()) for r in remaining}
    selected, roots, chains = [], set(), set()
    while remaining and len(selected) < limit:
        candidate = max(
            remaining,
            key=lambda r: (
                r.instance.root_sample_id not in roots,
                r.record.get("chain_id", r.instance.instance_hash) not in chains,
                priorities[r.instance.instance_hash],
            ),
        )
        selected.append(candidate)
        roots.add(candidate.instance.root_sample_id)
        chains.add(candidate.record.get("chain_id", candidate.instance.instance_hash))
        remaining = [
            r
            for r in remaining
            if r.instance.instance_hash != candidate.instance.instance_hash
        ]
    return selected


@dataclass
class Partition:
    mean: np.ndarray
    scale: np.ndarray
    components: np.ndarray
    coefficient: np.ndarray
    intercept: np.ndarray
    classes: np.ndarray

    def predict(self, instances):
        x = (np.stack([coordinates(i) for i in instances]) - self.mean) / self.scale
        scores = (x @ self.components.T) @ self.coefficient.T + self.intercept
        index = (
            (scores[:, 0] > 0).astype(int)
            if len(self.classes) == 2
            else scores.argmax(axis=1)
        )
        return self.classes[index]

    def as_dict(self):
        return {
            name: getattr(self, name).tolist() for name in self.__dataclass_fields__
        }

    @classmethod
    def from_dict(cls, value):
        return cls(**{name: np.asarray(array) for name, array in value.items()})


class HillValleyPartitioner:
    def __init__(self, *, graph_size=12, seed=20260926):
        self.graph_size, self.seed = graph_size, seed

    def split(self, population, check):
        samples = representatives(population, self.graph_size, seed=self.seed)
        if len(samples) < 4:
            return None, None
        graph = nx.Graph()
        graph.add_nodes_from(range(len(samples)))
        evidence = []
        for i, left in enumerate(samples):
            for j in range(i + 1, len(samples)):
                right = samples[j]
                result = check(left, right)
                if result is None:
                    # Missing edges are unknown, never evidence of separation.
                    return None, {
                        "status": "incomplete",
                        "checked_pairs": evidence,
                        "representatives": [r.instance.instance_hash for r in samples],
                    }
                connected = result.level_score >= min(
                    left.level_score, right.level_score
                )
                evidence.append(
                    {
                        "left": left.instance.instance_hash,
                        "right": right.instance.instance_hash,
                        "check": result.instance.instance_hash,
                        "connected": bool(connected),
                    }
                )
                if connected:
                    graph.add_edge(i, j)
        labelings, single, candidates = {}, 0, set()
        for repeat in range(100):
            groups = sorted(
                (
                    sorted(g)
                    for g in nx.community.asyn_lpa_communities(
                        graph, seed=self.seed + repeat
                    )
                ),
                key=lambda g: g[0],
            )
            single += len(groups) == 1
            if len(groups) < 2:
                continue
            if any(len(group) < 2 for group in groups):
                candidates.update(
                    samples[group[0]].instance.instance_hash
                    for group in groups
                    if len(group) == 1
                )
                continue
            labels = np.empty(len(samples), dtype=int)
            for label, group in enumerate(groups):
                labels[group] = label
            labelings[tuple(labels)] = labels
        report = {
            "status": "one_branch",
            "checked_pairs": evidence,
            "representatives": [r.instance.instance_hash for r in samples],
            "candidate_singletons": sorted(candidates),
        }
        if single > 50 or not labelings:
            return None, report
        all_x = np.stack([coordinates(r.instance) for r in population])
        mean, scale = all_x.mean(axis=0), np.maximum(all_x.std(axis=0), 1e-6)
        centered = (all_x - mean) / scale
        pca = PCA(
            n_components=min(len(all_x) - 1, all_x.shape[1]), svd_solver="full"
        ).fit(centered)
        sample_x = (np.stack([coordinates(r.instance) for r in samples]) - mean) / scale
        transformed = pca.transform(sample_x)
        best = None
        for key, labels in sorted(labelings.items()):
            classifier = LinearSVC(
                C=1, penalty="l1", dual=False, random_state=self.seed, max_iter=10000
            ).fit(transformed, labels)
            quality = balanced_accuracy_score(labels, classifier.predict(transformed))
            rank = (quality, -len(classifier.classes_))
            if best is None or rank > best[0]:
                best = (rank, classifier)
        classifier = best[1]
        # L1 gives exactly zero coefficients: omit only those unused PCA axes.
        # This leaves the classifier unchanged and keeps branch artifacts compact.
        used = np.any(classifier.coef_ != 0, axis=0)
        partition = Partition(
            mean,
            scale,
            pca.components_[used],
            classifier.coef_[:, used],
            classifier.intercept_,
            classifier.classes_,
        )
        labels = partition.predict([r.instance for r in population])
        groups = [
            {
                r.instance.instance_hash
                for r, actual in zip(population, labels)
                if actual == label
            }
            for label in set(labels)
        ]
        if len(groups) < 2 or any(len(group) < 2 for group in groups):
            report.update(
                status="unconfirmed_partition", balanced_accuracy=float(best[0][0])
            )
            return None, report
        report.update(
            status="split",
            balanced_accuracy=float(best[0][0]),
            active_pca_coordinates=int(used.sum()),
        )
        return partition, report
