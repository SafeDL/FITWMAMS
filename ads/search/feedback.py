"""Frozen behavior-weighted targets and prior-reversible traffic mutations."""

import hashlib
from collections import Counter
from dataclasses import dataclass

import numpy as np

from hierarchical_world_model.src.randomness import WORLD_RANDOM_BLOCKS


def description(result):
    values = np.asarray(result.record.get("descriptor", [])[:4], float)
    if (
        values.shape != (4,)
        or not np.isfinite(values).all()
        or result.record.get("critical_slot", -1) < 1
    ):
        return None
    return values


def trajectory_hash(result):
    digest = hashlib.sha256()
    for value in (result.states, result.controls):
        value = np.ascontiguousarray(value)
        digest.update(str((value.shape, value.dtype.str)).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()






def separation(result, others):
    value = description(result)
    values = [description(other) for other in others]
    values = [other for other in values if other is not None]
    return (
        float(np.linalg.norm(np.asarray(values) - value, axis=1).min())
        if value is not None and values
        else float("inf")
    )


def coverage_radius(initial, elite_fraction):
    ranked = sorted(initial, key=lambda r: (-r.score, r.instance.instance_hash))
    ranked = [r for r in ranked if description(r) is not None]
    ranked = ranked[: max(1, int(np.ceil(len(initial) * elite_fraction)))]
    unique = {}
    for result in ranked:
        unique.setdefault(tuple(description(result)), result)
    representatives = list(unique.values())
    distances = []
    for result in representatives:
        others = [
            r
            for r in representatives
            if r.instance.root_sample_id != result.instance.root_sample_id
        ]
        if others:
            distances.append(separation(result, others))
    return (float(np.median(distances)) if distances else 0.0), [
        r.instance.instance_hash for r in representatives
    ]


@dataclass(frozen=True)
class CoverageTarget:
    threshold: float
    radius: float
    representatives: np.ndarray
    counts: dict
    version: int
    delta: float = 0.05

    def cell(self, result):
        value = description(result)
        if value is None:
            return -2
        if not len(self.representatives):
            return -1
        distance = np.linalg.norm(self.representatives - value, axis=1)
        index = int(distance.argmin())
        return index if distance[index] <= self.radius else -1

    def weight(self, result):
        cell = self.cell(result)
        if cell == -2:
            return self.delta
        n = self.counts.get(cell, 0) if cell >= 0 else 0
        return float(self.delta + np.log1p(1 / (n + 1)))

    def as_dict(self):
        return {
            "threshold": self.threshold,
            "radius": self.radius,
            "representatives": self.representatives.tolist(),
            "counts": self.counts,
            "version": self.version,
            "delta": self.delta,
        }


class BehaviorCatalog:
    def __init__(self, radius, delta=0.05):
        self.radius, self.delta = radius, delta
        self.representatives, self.version = [], 0

    def update(self, records):
        for result in records:
            value = description(result)
            if value is not None and (
                not self.representatives
                or np.linalg.norm(
                    np.asarray(self.representatives) - value, axis=1
                ).min()
                > self.radius
            ):
                self.representatives.append(value.copy())
        self.version += 1

    def target(self, records, threshold):
        target = CoverageTarget(
            threshold,
            self.radius,
            np.asarray(self.representatives).reshape(-1, 4).copy(),
            {},
            self.version,
            self.delta,
        )
        seen = set()
        for result in records:
            identity = result.record.get("trajectory_hash") or trajectory_hash(result)
            if result.level_score < threshold or identity in seen:
                continue
            seen.add(identity)
            cell = target.cell(result)
            if cell >= 0:
                target.counts[cell] = target.counts.get(cell, 0) + 1
        return target

    def gains(self, history, executed, target):
        seen = {
            r.record.get("trajectory_hash") or trajectory_hash(r)
            for r in history
            if r.level_score >= target.threshold
        }
        counts = Counter(target.counts)
        gains, new_cells = [], {}
        for result in executed:
            identity = result.record.get("trajectory_hash") or trajectory_hash(result)
            gain = 0.0
            if result.level_score >= target.threshold and identity not in seen:
                cell = target.cell(result)
                n = counts[cell] if cell >= 0 else 0
                gain = (
                    target.delta if cell == -2 else target.delta + np.log1p(1 / (n + 1))
                )
                # Uncovered points are distinct provisional cells until the batch commits.
                # Resolve nearby new points in execution order using temporary representatives.
                if cell == -1:
                    value = description(result)
                    matches = [
                        (key, point)
                        for key, point in new_cells.items()
                        if np.linalg.norm(point - value) <= target.radius
                    ]
                    if matches:
                        cell = min(
                            matches,
                            key=lambda pair: (np.linalg.norm(pair[1] - value), pair[0]),
                        )[0]
                        gain = target.delta + np.log1p(1 / (counts[cell] + 1))
                    else:
                        cell = -3 - len(new_cells)
                        new_cells[cell] = value
                counts[cell] += 1
                seen.add(identity)
            gains.append(float(gain))
        return gains


def risk_gain(parent, child):
    return (
        0.0
        if parent.failure
        else float(
            max(0.0, child.level_score - parent.level_score) / (1 - parent.level_score)
        )
    )


def static_masks(omega, record):
    masks = {name: np.ones_like(value, bool) for name, value in omega.as_dict().items()}
    for name, coordinates in record.get("static_input_components", {}).items():
        masks[name][...] = False
        masks[name][..., coordinates] = True
    masks["policy_calibration_innovations"][:, 1:] = False
    return masks


def dependencies(result):
    """Indices refer to base random coordinates, never interpolated trajectories."""
    omega = result.instance.event.exogenous_state
    masks = {
        name: np.zeros_like(value, bool) for name, value in omega.as_dict().items()
    }
    for name in WORLD_RANDOM_BLOCKS[:4]:
        masks[name][...] = True
    for name, indices in result.record.get("input_dependencies", {}).items():
        masks[name].reshape(-1)[indices] = True
    return masks


@dataclass
class ProposalConfiguration:
    operators: list
    joint_masks: dict
    utilities: dict
    joint_fraction: float
    uniform: bool = False

    def probabilities(self, result):
        if not self.operators:
            return np.ones(1)
        dep = dependencies(result)
        sizes = Counter(name for name, _ in self.operators)
        scores = np.array(
            [
                (0.05 + (0 if self.uniform else self.utilities[name]))
                / sizes[name]
                * (
                    1
                    if self.uniform
                    else 0.05 + np.count_nonzero(mask & dep[name]) / mask.sum()
                )
                for name, mask in self.operators
            ]
        )
        return np.r_[
            self.joint_fraction, (1 - self.joint_fraction) * scores / scores.sum()
        ]

    def as_dict(self):
        return {
            "operators": [
                {"group": name, "indices": np.flatnonzero(mask).tolist()}
                for name, mask in self.operators
            ],
            "utilities": self.utilities,
            "joint_fraction": self.joint_fraction,
            "uniform": self.uniform,
        }


def proposal_configuration(population, utilities, *, uniform=False, joint_fraction=0.2):
    omega = population[0].instance.event.exogenous_state
    joint = static_masks(omega, population[0].record)
    operators = [(name, joint[name].copy()) for name in WORLD_RANDOM_BLOCKS[:4]]
    deps = [dependencies(result) for result in population]
    union = {
        name: np.logical_or.reduce([d[name] for d in deps])
        for name in WORLD_RANDOM_BLOCKS
    }
    for name in WORLD_RANDOM_BLOCKS[4:]:
        for actor in range(6):
            mask = np.zeros_like(joint[name])
            if name in ("driver_latent", "maneuver_latent"):
                mask[:, actor] = joint[name][:, actor]
                candidates = [mask]
            elif name == "policy_calibration_innovations":
                mask[:, 0, actor] = joint[name][:, 0, actor]
                candidates = [mask]
            else:
                mask[:, :, actor] = joint[name][:, :, actor] & union[name][:, :, actor]
                recent = np.zeros_like(mask)
                for result, dep in zip(population, deps):
                    stop = min(
                        omega.response_steps,
                        int(result.record.get("critical_frame", 0)),
                    )
                    recent[:, max(0, stop - 25) : stop, actor] |= dep[name][
                        :, max(0, stop - 25) : stop, actor
                    ]
                candidates = [mask & recent, mask & ~recent]
            operators.extend((name, mask) for mask in candidates if mask.any())
    return ProposalConfiguration(
        operators, joint, dict(utilities), joint_fraction, uniform
    )


def propose(parent, rng, config, proposal, *, joint_only=False):
    omega = parent.instance.event.exogenous_state
    probabilities = proposal.probabilities(parent)
    operator = 0 if joint_only else int(rng.choice(len(probabilities), p=probabilities))
    mutations = (
        list(proposal.joint_masks.items())
        if operator == 0
        else [proposal.operators[operator - 1]]
    )
    for block, mask in mutations:
        omega = omega.mutate(
            "scenario" if block == "scenario_uniform" else block,
            beta=config.beta,
            seed=int(rng.integers(2**63)),
            **({} if block == "scenario_uniform" else {"coordinate_mask": mask}),
        )
    return omega, operator, float(probabilities[operator])


def systematic_resample(weights, count, rng):
    weights = np.asarray(weights, float)
    if not np.isfinite(weights).all() or weights.sum() <= 0:
        raise ValueError("No supported particles in the next risk target")
    cumulative = np.cumsum(weights / weights.sum())
    cumulative[-1] = 1.0
    return np.searchsorted(cumulative, (rng.random() + np.arange(count)) / count)


def transition_acceptance(parent, child, target, proposal, operator, forward):
    if child.level_score < target.threshold:
        return 0.0
    reverse = proposal.probabilities(child)[operator]
    return float(
        min(1, target.weight(child) * reverse / (target.weight(parent) * forward))
    )
