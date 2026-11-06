"""Traffic-feedback subset search with conditional chains and retained starts."""

import time
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from ads.search.archive import FailureArchive, diverse_representatives
from ads.search.feedback import (
    BehaviorCatalog,
    CoverageTarget,
    coverage_radius,
    description,
    proposal_configuration,
    propose,
    risk_gain,
    separation,
    systematic_resample,
    trajectory_hash,
    transition_acceptance,
)
from hierarchical_world_model.src.randomness import WORLD_RANDOM_BLOCKS


@dataclass(frozen=True)
class SearchConfig:
    budget: int = 5000
    batch_size: int = 500
    generation_batch_size: int = 32
    initial_samples: int = 1000
    replay_limit: int = 256
    elite_fraction: float = 0.1
    beta: float = 0.2
    joint_fraction: float = 0.2
    subset_max_levels: int = 8
    terminal_proposals: int = 250
    terminal_batch_size: int = 50
    archive_capacity: int = 16
    archive_root_limit: int = 4
    active_paths: int = 2
    stall_batches: int = 2
    fixed_budget: bool = True
    coverage_delta: float = 0.05

    def __post_init__(self):
        if not 1 <= self.budget <= 5000 or not 1 <= self.batch_size <= 500:
            raise ValueError("Search is capped at 5000 executions, 500 per batch")
        if (
            min(
                self.initial_samples,
                self.generation_batch_size,
                self.subset_max_levels,
                self.terminal_batch_size,
                self.active_paths,
                self.stall_batches,
            )
            < 1
        ):
            raise ValueError("Search sizes and depths must be positive")
        if self.terminal_proposals < 0 or self.replay_limit < 0:
            raise ValueError("Search quotas cannot be negative")
        if (
            not 0 < self.elite_fraction < 1
            or not 0 < self.beta <= 1
            or not 0 <= self.joint_fraction <= 1
        ):
            raise ValueError("Invalid proposal scale or fraction")


DEFAULT_SEARCH_CONFIG = SearchConfig()


def zero_utilities():
    return {name: 0.0 for name in WORLD_RANDOM_BLOCKS}


@dataclass
class Branch:
    identity: str
    population: list
    depth: int = 0
    threshold: float = 0.0
    allocated: int = 0
    terminal_used: int = 0
    stalled_batches: int = 0
    service: float = 0.0
    weight: float = 1.0
    restored: bool = False
    risk_utilities: dict = field(default_factory=zero_utilities)
    terminal_utilities: dict = field(default_factory=zero_utilities)
    target: CoverageTarget | None = None


class SearchEngine:
    def __init__(
        self,
        source,
        evaluate,
        *,
        config=DEFAULT_SEARCH_CONFIG,
        seed=20260926,
        progress=None,
        on_evaluation=None,
        evaluation_batch_size=1,
        generation_executor=None,
    ):
        self.source, self.evaluate, self.config = source, evaluate, config
        self.seed, self.progress, self.on_evaluation = seed, progress, on_evaluation
        self.evaluation_batch_size = evaluation_batch_size
        self.generation_executor = generation_executor
        self.rng = np.random.default_rng(seed)
        self.records, self.known, self.nodes = [], {}, {}
        self.costs = Counter()
        self.batch_index = 0
        self.archive = FailureArchive(
            capacity=config.archive_capacity, root_limit=config.archive_root_limit
        )
        self.retained, self.restored_ids = {}, set()
        self.coverage = None
        self.radius_samples = []

    def _evaluate(self, instance, *, kind, branch):
        return self._record(self.evaluate(instance), kind=kind, branch=branch)

    def _record(self, result, *, kind, branch):
        result.record.update(
            evaluation_index=len(self.records) + 1,
            search_batch=self.batch_index,
            execution_kind=kind,
            branch_id=branch,
            search_score=result.level_score,
            independent_sampling_root=kind
            in ("initial_prior", "prior", "restart_prior"),
        )
        result.record["trajectory_hash"] = trajectory_hash(result)
        self.records.append(result)
        self.known[result.instance.instance_hash] = result
        self.costs[kind] += 1
        if self.on_evaluation:
            self.on_evaluation(result)
        if self.progress and (
            len(self.records) % 32 == 0 or len(self.records) == self.config.budget
        ):
            self.progress(len(self.records), self.config.budget)
        return result

    def _evaluate_many(self, instances, *, kind, branch):
        instances = instances[: self.config.budget - len(self.records)]
        evaluate_many = getattr(self.evaluate, "many", None)
        results = (
            evaluate_many(instances)
            if evaluate_many is not None
            else [self.evaluate(instance) for instance in instances]
        )
        return [self._record(result, kind=kind, branch=branch) for result in results]

    def _fresh(self, count, *, kind, branch):
        results = []
        count = min(count, self.config.budget - len(self.records))
        for start in range(0, count, self.config.generation_batch_size):
            instances = self.source.sample(
                min(self.config.generation_batch_size, count - start)
            )
            results.extend(self._evaluate_many(instances, kind=kind, branch=branch))
        return results

    def _node(self, branch):
        return self.nodes.setdefault(
            branch.identity,
            {
                "depth": branch.depth,
                "threshold": branch.threshold,
                "status": "active",
                "seed_states": [r.instance.instance_hash for r in branch.population],
                "restored": branch.restored,
                "levels": [],
            },
        )

    def _threshold(self, branch):
        return max(
            branch.threshold,
            float(
                np.quantile(
                    [r.level_score for r in branch.population],
                    1 - self.config.elite_fraction,
                )
            ),
        )

    def _remember(self, candidates, branch, advancing=()):
        if branch.depth >= self.config.subset_max_levels:
            return
        for result in candidates:
            identity = result.instance.instance_hash
            if (
                identity in self.restored_ids
                or description(result) is None
                or result.level_score < branch.threshold
                or separation(result, advancing) <= self.coverage.radius
            ):
                continue
            self.retained.setdefault(
                identity,
                {
                    "result": result,
                    "population": tuple(branch.population),
                    "target": branch.target,
                    "threshold": branch.threshold,
                    "depth": branch.depth,
                    "entered": self.batch_index,
                    "origin": branch.identity,
                    "terminal_used": branch.terminal_used,
                    "stalled_batches": branch.stalled_batches,
                    "risk_utilities": dict(branch.risk_utilities),
                    "terminal_utilities": dict(branch.terminal_utilities),
                },
            )
        representatives = diverse_representatives(
            [entry["result"] for entry in self.retained.values()],
            capacity=self.config.archive_capacity,
            root_limit=self.config.archive_root_limit,
        )
        self.retained = {
            r.instance.instance_hash: self.retained[r.instance.instance_hash]
            for r in representatives
        }

    def _restore(self, active, serial):
        selected = min(
            self.retained,
            key=lambda key: (
                self.retained[key]["entered"],
                -separation(
                    self.retained[key]["result"],
                    [r for b in active for r in b.population],
                ),
                key,
            ),
        )
        entry = self.retained.pop(selected)
        self.restored_ids.add(selected)
        branch = Branch(
            f"restored_{serial}",
            list(entry["population"]),
            depth=entry["depth"],
            threshold=entry["threshold"],
            target=entry["target"],
            restored=True,
            terminal_used=entry["terminal_used"],
            stalled_batches=entry["stalled_batches"],
            service=min((b.service for b in active), default=0.0),
            risk_utilities=entry["risk_utilities"],
            terminal_utilities=entry["terminal_utilities"],
        )
        self._node(branch).update(
            origin=entry["origin"],
            retained_at_batch=entry["entered"],
            retained_anchor=selected,
            restored_snapshot_particles=len(branch.population),
        )
        return branch

    def _advance(self, branch, method):
        node = self._node(branch)
        guided = method == "guided"
        previous_threshold = branch.threshold
        threshold = previous_threshold if branch.restored else self._threshold(branch)
        if guided and threshold > previous_threshold:
            self._remember(
                [r for r in branch.population if r.level_score < threshold],
                branch,
                [r for r in branch.population if r.level_score >= threshold],
            )
        terminal = threshold >= 1.0
        if not terminal and branch.depth >= self.config.subset_max_levels:
            node["status"] = "depth_limit"
            return False
        if terminal and branch.terminal_used >= self.config.terminal_proposals:
            node["status"] = "terminal_quota"
            return False
        intended = min(
            self.config.batch_size,
            self.config.terminal_batch_size if terminal else self.config.batch_size,
            self.config.terminal_proposals - branch.terminal_used
            if terminal
            else self.config.batch_size,
        )
        count = min(intended, self.config.budget - len(self.records))
        if not count:
            return True
        target = self.coverage.target(self.records, threshold)
        denominator = np.array(
            [
                branch.target.weight(r) if branch.target else 1.0
                for r in branch.population
            ]
        )
        numerator = np.array(
            [target.weight(r) if guided else 1.0 for r in branch.population]
        )
        weights = (
            numerator
            / denominator
            * np.array([r.level_score >= threshold for r in branch.population])
        )
        chain_count = min(
            len(branch.population),
            max(1, int(np.ceil(count * self.config.elite_fraction))),
        )
        indices = systematic_resample(weights, chain_count, self.rng)
        seeds = [branch.population[i] for i in indices]
        utilities = branch.terminal_utilities if terminal else branch.risk_utilities
        proposal = proposal_configuration(
            branch.population,
            utilities,
            uniform=method != "guided",
            joint_fraction=self.config.joint_fraction,
        )
        chains, states, jobs_executed = list(seeds), [], []
        reward_groups = {name: [] for name in WORLD_RANDOM_BLOCKS}
        weight_before, start = branch.weight, len(self.records)
        level = {
            "depth": branch.depth,
            "threshold": threshold,
            "previous_threshold": previous_threshold,
            "restored_first_batch": branch.restored,
            "weight_at_start": weight_before,
            "proposal": proposal.as_dict(),
            "target": target.as_dict() if guided else None,
            "old_target_version": branch.target.version if branch.target else None,
            "incremental_weights": weights.tolist(),
            "resampled_indices": indices.tolist(),
            "seed_states": [r.instance.instance_hash for r in seeds],
        }
        index = 0
        while index < count:
            width = min(
                self.evaluation_batch_size,
                len(chains) - index % len(chains),
                count - index,
            )
            jobs, generation = [], []
            for offset in range(width):
                chain_index = (index + offset) % len(chains)
                parent = chains[chain_index]
                # Independent job streams keep proposal/acceptance draws invariant to wave width.
                rng = np.random.default_rng(int(self.rng.integers(2**63)))
                omega, operator, forward = propose(
                    parent, rng, self.config, proposal, joint_only=method == "subset"
                )
                block = (
                    "joint" if operator == 0 else proposal.operators[operator - 1][0]
                )
                jobs.append(
                    (chain_index, parent, operator, forward, rng.random(), block)
                )
                generation.append((parent.instance, omega, block))
            instances = (
                self.source.mutations(generation, self.generation_executor)
                if self.generation_executor is not None
                else [self.source.mutation(*job) for job in generation]
            )
            children = self._evaluate_many(
                instances,
                kind="terminal_proposal" if terminal else "conditional_proposal",
                branch=branch.identity,
            )
            for (chain_index, parent, operator, forward, uniform, block), child in zip(
                jobs, children
            ):
                probability = (
                    transition_acceptance(
                        parent, child, target, proposal, operator, forward
                    )
                    if guided
                    else float(child.level_score >= threshold)
                )
                accepted = uniform < probability
                child.record.update(
                    chain_id=f"{branch.identity}:{len(node['levels'])}:{chain_index}",
                    parent_hash=parent.instance.instance_hash,
                    proposal_block=block,
                    proposal_operator=operator,
                    proposal_forward=forward,
                    proposal_reverse=float(proposal.probabilities(child)[operator]),
                    acceptance_probability=probability,
                    acceptance_uniform=uniform,
                    target_weight_parent=target.weight(parent),
                    target_weight_child=target.weight(child),
                    target_version=target.version,
                    accepted=bool(accepted),
                    risk_gain=risk_gain(parent, child),
                    threshold=threshold,
                    path_depth=branch.depth,
                    proposal_configuration=f"{branch.identity}:{len(node['levels'])}",
                    unchanged_execution=trajectory_hash(child)
                    == trajectory_hash(parent),
                )
                if accepted:
                    chains[chain_index] = child
                states.append(chains[chain_index])
                jobs_executed.append((parent, child, block, accepted))
            index += width
        executed = self.records[start:]
        gains = self.coverage.gains(self.records[:start], executed, target)
        for (parent, child, block, accepted), gain in zip(jobs_executed, gains):
            reward = (
                gain / (self.config.coverage_delta + np.log(2))
                if terminal
                else int(accepted) * risk_gain(parent, child)
            )
            child.record.update(discovery_gain=gain, proposal_reward=reward)
            if block in reward_groups:
                reward_groups[block].append(reward)
        for name, values in reward_groups.items():
            if values:
                utilities[name] = 0.8 * utilities[name] + 0.2 * float(np.mean(values))
        new_count = sum(gain > 0 for gain in gains)
        branch.population, branch.target = states, target if guided else None
        branch.threshold, branch.restored = threshold, False
        branch.allocated += count
        branch.terminal_used += count if terminal else 0
        branch.depth += int(threshold > previous_threshold)
        next_threshold = self._threshold(branch)
        progressed = threshold > previous_threshold or next_threshold > threshold
        branch.stalled_batches = (
            0 if new_count or progressed else branch.stalled_batches + 1
        )
        mean_gain = float(np.mean([r.record["proposal_reward"] for r in executed]))
        branch.service += count / weight_before
        branch.weight = 1 + mean_gain
        self.coverage.update(executed)
        level.update(
            depth_after=branch.depth,
            proposals=count,
            intended_proposals=intended,
            incomplete=count < intended,
            particle_states=[r.instance.instance_hash for r in states],
            next_threshold=next_threshold,
            new_qualified_traces=new_count,
            group_rewards={
                name: {"executions": len(v), "mean": float(np.mean(v)) if v else None}
                for name, v in reward_groups.items()
            },
            utilities=dict(utilities),
            weight_after=branch.weight,
            virtual_service=branch.service,
        )
        node["levels"].append(level)
        node.update(
            depth=branch.depth,
            threshold=threshold,
            allocated=branch.allocated,
            terminal_proposals=branch.terminal_used,
            virtual_service=branch.service,
        )
        self.batch_index += 1
        status = None
        if terminal and branch.terminal_used >= self.config.terminal_proposals:
            status = "terminal_quota"
        elif (
            not terminal
            and branch.depth >= self.config.subset_max_levels
            and next_threshold < 1
        ):
            status = "depth_limit"
        elif (
            not terminal
            and branch.stalled_batches >= self.config.stall_batches
            and not progressed
        ):
            status = "no_threshold_or_discovery_progress"
        if status:
            node["status"] = status
            return False
        return True

    def run(self, method="guided", *, archive=None, initial=None):
        if method not in ("monte_carlo", "subset", "guided"):
            raise ValueError("Unknown search method")
        started, generation_before = (
            time.perf_counter(),
            self.source.generation_elapsed_s,
        )
        anchors = []
        if archive is not None:
            for old in archive.endpoints(
                limit=min(self.config.replay_limit, self.config.budget - 1)
            ):
                result = self._evaluate(
                    old.instance, kind="archive_replay", branch=old.domain
                )
                result.record.update(
                    old_safety_failure=old.failure, old_branch_id=old.domain
                )
                anchors.append(result)
        initial = (
            self._fresh(
                self.config.initial_samples, kind="initial_prior", branch="root"
            )
            if initial is None
            else [
                self._record(result, kind="initial_prior", branch="root")
                for result in initial
            ]
        )
        radius, self.radius_samples = coverage_radius(
            initial, self.config.elite_fraction
        )
        self.coverage = BehaviorCatalog(radius, self.config.coverage_delta)
        self.coverage.update(initial)
        active = (
            [Branch("root", initial)] if initial and method != "monte_carlo" else []
        )
        for branch in active:
            self._node(branch)
        restarts, serial = 0, 0
        guided = method == "guided"
        while len(self.records) < self.config.budget:
            if method == "monte_carlo":
                self._fresh(self.config.batch_size, kind="prior", branch="root")
                self.batch_index += 1
                continue
            while guided and len(active) < self.config.active_paths and self.retained:
                active.append(self._restore(active, serial))
                serial += 1
            if not active:
                if not self.config.fixed_budget:
                    break
                population = self._fresh(
                    self.config.batch_size,
                    kind="restart_prior",
                    branch=f"restart_{restarts}",
                )
                branch = Branch(f"restart_{restarts}", population)
                active.append(branch)
                self._node(branch)
                restarts += 1
                continue
            branch = min(active, key=lambda b: (b.service, b.identity))
            if not self._advance(branch, method):
                active.remove(branch)
        self.archive.add(self.records, self.known)
        feedback = archive.feedback(anchors) if archive else {}
        if archive:
            self.archive.retain_previous(archive, anchors)
        for branch in active:
            self._node(branch).update(
                unfinished=True,
                particle_states=[r.instance.instance_hash for r in branch.population],
            )
        result = SearchResult(
            self.records,
            self.archive,
            feedback,
            self.nodes,
            dict(self.costs),
            "execution_budget"
            if len(self.records) == self.config.budget
            else "scheduled_tasks_complete",
            self.config.budget,
            time.perf_counter() - started,
            self.source.generation_elapsed_s - generation_before,
            [
                {
                    "instance_hash": r.instance.instance_hash,
                    "old_branch_id": r.record["old_branch_id"],
                    "new_branch_id": r.domain,
                    "old_failure": r.record["old_safety_failure"],
                    "new_failure": r.failure,
                }
                for r in anchors
            ],
            {
                "active_branches": [b.identity for b in active],
                "retained_starts": list(self.retained),
            },
            "fixed_budget" if self.config.fixed_budget else "early_stop",
            restarts,
        )
        result.coverage = {
            "radius": radius,
            "calibration_instances": self.radius_samples,
            "representatives": [
                value.tolist() for value in self.coverage.representatives
            ],
            "restored_starts": sorted(self.restored_ids),
        }
        return result


@dataclass
class SearchResult:
    records: list
    archive: FailureArchive
    feedback: dict
    branches: dict
    costs: dict
    stop_reason: str
    budget: int
    elapsed_s: float
    generation_elapsed_s: float
    transitions: list
    pending: dict
    budget_mode: str
    restarts: int

    def summary(self):
        old_to_new, new_to_old = {}, {}
        for item in self.transitions:
            old_to_new.setdefault(item["old_branch_id"], set()).add(
                item["new_branch_id"]
            )
            new_to_old.setdefault(item["new_branch_id"], set()).add(
                item["old_branch_id"]
            )
        relationships = {
            "old_to_new": {key: sorted(value) for key, value in old_to_new.items()},
            "new_to_old": {key: sorted(value) for key, value in new_to_old.items()},
            "split_old_branches": sorted(
                key for key, value in old_to_new.items() if len(value) > 1
            ),
            "merged_new_branches": sorted(
                key for key, value in new_to_old.items() if len(value) > 1
            ),
        }
        regular = [
            r
            for r in self.records
            if r.record["execution_kind"] not in ("branch_check", "archive_replay")
        ]
        failures = {r.instance.instance_hash: r for r in regular if r.failure}
        return {
            "evaluations": len(self.records),
            "execution_budget": self.budget,
            "budget_mode": self.budget_mode,
            "search_restarts": self.restarts,
            "unused_budget": self.budget - len(self.records),
            "stop_reason": self.stop_reason,
            "execution_costs": self.costs,
            "unique_safety_failures": len(failures),
            "unique_collisions": len(
                {
                    r.instance.instance_hash
                    for r in regular
                    if r.record.get("collision", False)
                }
            ),
            "failure_domains": sorted({r.domain for r in failures.values()}),
            "failure_domain_interpretation": "exploration_paths_not_failure_domains",
            "collision_coverage": getattr(self, "coverage", {}),
            "independent_failure_roots": len(
                {r.instance.root_sample_id for r in failures.values()}
            ),
            "hit_fraction": sum(r.failure for r in regular) / max(len(regular), 1),
            "hit_fraction_is_prior_probability": False,
            "elapsed_s": self.elapsed_s,
            "generation_elapsed_s": self.generation_elapsed_s,
            "execution_elapsed_s": sum(
                float(r.record.get("elapsed_s", 0)) for r in self.records
            ),
            "total_search_elapsed_s": self.elapsed_s,
            "feedback": self.feedback,
            "branch_transitions": self.transitions,
            "branch_relationships": relationships,
            "pending_tasks": self.pending,
            "retained_archive_endpoints": len(self.archive.retained),
            "branch_tree": self.branches,
        }
