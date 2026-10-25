"""Budgeted branching subset search; discovery samples are not IID risk estimates."""

import time
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from ads.search.archive import FailureArchive, diverse_representatives
from ads.search.branches import HillValleyPartitioner, midpoint
from hierarchical_world_model.src.randomness import WORLD_RANDOM_BLOCKS


@dataclass(frozen=True)
class SearchConfig:
    budget: int = 5000
    batch_size: int = 250
    generation_batch_size: int = 32
    initial_samples: int = 500
    fresh_samples: int = 500
    branch_check_budget: int = 500
    graph_size: int = 12
    replay_limit: int = 256
    elite_fraction: float = 0.1
    beta: float = 0.2
    joint_fraction: float = 0.2
    subset_max_levels: int = 8
    terminal_proposals: int = 250
    archive_capacity: int = 16
    archive_root_limit: int = 4
    fixed_budget: bool = False

    def __post_init__(self):
        if self.budget > 5000 or self.branch_check_budget > 500:
            raise ValueError(
                "risk-stratified repair caps are 5,000 executions and 500 branch checks"
            )
        if (
            min(
                self.budget,
                self.batch_size,
                self.generation_batch_size,
                self.initial_samples,
                self.subset_max_levels,
            )
            < 1
        ):
            raise ValueError("search budgets and population sizes must be positive")
        if (
            min(
                self.fresh_samples,
                self.branch_check_budget,
                self.replay_limit,
                self.terminal_proposals,
            )
            < 0
        ):
            raise ValueError("search quotas cannot be negative")
        if (
            not 0 < self.elite_fraction < 1
            or not 0 < self.beta <= 1
            or not 0 <= self.joint_fraction <= 1
        ):
            raise ValueError("invalid conditional probability or proposal scale")


DEFAULT_SEARCH_CONFIG = SearchConfig()


@dataclass
class Branch:
    identity: str
    population: list
    depth: int = 0
    threshold: float = 0.0
    path: tuple = ()
    allocated: int = 0
    terminal_used: int = 0
    sensitivity: dict = field(
        default_factory=lambda: {b: 1.0 for b in WORLD_RANDOM_BLOCKS}
    )

    def contains(self, instance):
        return all(
            int(partition.predict([instance])[0]) == label
            for partition, label in self.path
        )


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
    ):
        self.source, self.evaluate, self.config = source, evaluate, config
        self.seed, self.progress = seed, progress
        self.on_evaluation = on_evaluation
        self.rng = np.random.default_rng(seed)
        self.records, self.known, self.nodes = [], {}, {}
        self.costs = Counter()
        self.batch_index = 0
        self.archive = FailureArchive(
            capacity=config.archive_capacity, root_limit=config.archive_root_limit
        )
        self.partitioner = HillValleyPartitioner(
            graph_size=config.graph_size, seed=seed
        )
        self.check_cache = {}
        self.partitions, self.weights = {}, {}
        self.anchors = []
        self.main_limit = config.budget

    def _evaluate(self, instances, *, kind, branch="root", parents=None):
        results = []
        for index, instance in enumerate(instances):
            if len(self.records) >= self.config.budget:
                break
            result = self.evaluate(instance)
            result.record.update(
                evaluation_index=len(self.records) + 1,
                search_batch=self.batch_index,
                execution_kind=kind,
                branch_id=branch,
                search_score=result.level_score,
            )
            if parents is not None:
                result.record["diagnostic_endpoints"] = parents[index]
                result.record["independent_sampling_root"] = False
            else:
                result.record["independent_sampling_root"] = kind in (
                    "initial_prior",
                    "fresh_prior",
                    "prior",
                    "restart_prior",
                )
            self.records.append(result)
            self.known[instance.instance_hash] = result
            self.costs[kind] += 1
            results.append(result)
            if self.on_evaluation is not None:
                self.on_evaluation(result)
            if self.progress and (
                len(self.records) % 32 == 0 or len(self.records) == self.config.budget
            ):
                self.progress(len(self.records), self.config.budget)
        return results

    def _fresh(self, n, *, kind, branch):
        results = []
        n = min(n, max(0, self.main_limit - len(self.records)))
        for start in range(0, n, self.config.generation_batch_size):
            instances = self.source.sample(
                min(self.config.generation_batch_size, n - start)
            )
            results.extend(self._evaluate(instances, kind=kind, branch=branch))
        return results

    def _check(self, left, right):
        pair = tuple(
            sorted((left.instance.instance_hash, right.instance.instance_hash))
        )
        if pair in self.check_cache:
            return self.check_cache[pair]
        if self.config.fixed_budget and len(self.records) >= self.config.budget:
            return None
        omega = midpoint(left.instance, right.instance)
        instance = self.source.mutation(left.instance, omega, "branch_check")
        if instance.instance_hash in self.known:
            result = self.known[instance.instance_hash]
        else:
            if (
                self.costs["branch_check"] >= self.config.branch_check_budget
                or len(self.records) >= self.config.budget
            ):
                return None
            result = self._evaluate(
                [instance], kind="branch_check", parents=[list(pair)]
            )[0]
        self.check_cache[pair] = result
        return result

    def _proposal_configuration(self, branch):
        actors = np.zeros(6, bool)
        starts, stops = [], []
        for result in branch.population:
            slot = int(result.record.get("critical_slot", -1))
            if slot > 0:
                actors[slot - 1] = True
            else:
                actors |= result.instance.event.valid[1:]
            executed = int(result.record.get("physics_steps", 0))
            stop = min(
                int(result.record.get("critical_frame", executed)),
                executed,
                result.instance.event.exogenous_state.response_steps,
            )
            stops.append(max(0, stop))
            starts.append(max(0, stop - 25))
        probabilities = np.asarray(
            [0.05 + branch.sensitivity[b] for b in WORLD_RANDOM_BLOCKS], float
        )
        if not actors.any():
            probabilities[4:] = 0
        if not max(stops, default=0):
            probabilities[-2:] = 0
        probabilities /= probabilities.sum()
        return (
            probabilities,
            actors.copy(),
            (min(starts, default=0), max(stops, default=0)),
        )

    def _mutation(self, parent, *, guided, proposal):
        omega = parent.instance.event.exogenous_state
        if not guided or self.rng.random() < self.config.joint_fraction:
            for block in self.rng.permutation(WORLD_RANDOM_BLOCKS):
                scope = (
                    {"time_range": (0, 1)}
                    if block == "policy_calibration_innovations"
                    else {}
                )
                omega = omega.mutate(
                    "scenario" if block == "scenario_uniform" else str(block),
                    beta=self.config.beta,
                    seed=int(self.rng.integers(2**63)),
                    **scope,
                )
            return omega, "joint"
        probabilities, actors, time_range = proposal
        block = str(self.rng.choice(WORLD_RANDOM_BLOCKS, p=probabilities))
        kwargs = {}
        if block in WORLD_RANDOM_BLOCKS[4:]:
            kwargs["actor_mask"] = actors
        if block == "policy_response_innovations":
            kwargs["time_range"] = time_range
        elif block == "policy_calibration_innovations":
            kwargs["time_range"] = (0, 1)
        return omega.mutate(
            "scenario" if block == "scenario_uniform" else block,
            beta=self.config.beta,
            seed=int(self.rng.integers(2**63)),
            **kwargs,
        ), block

    def _weight(self, branch):
        if branch.identity in self.weights:
            return self.weights[branch.identity]
        anchors = [r for r in self.anchors if branch.contains(r.instance)]
        failures = [r for r in anchors if r.record["old_safety_failure"]]
        successes = [r for r in anchors if not r.record["old_safety_failure"]]
        unresolved = np.mean([r.failure for r in failures]) if failures else 0.0
        regression = np.mean([r.failure for r in successes]) if successes else 0.0
        self.weights[branch.identity] = float(1 + unresolved + 2 * regression)
        return self.weights[branch.identity]

    def _node(self, branch):
        self.nodes.setdefault(
            branch.identity,
            {
                "depth": branch.depth,
                "children": [],
                "status": "active",
                "threshold": branch.threshold,
                "seed_states": [r.instance.instance_hash for r in branch.population],
            },
        )
        return self.nodes[branch.identity]

    def _advance(self, branch, *, guided):
        node = self._node(branch)
        if guided:
            partition, evidence = self.partitioner.split(branch.population, self._check)
            if evidence:
                node.setdefault("partition_checks", []).append(evidence)
            if partition is not None:
                self.partitions[branch.identity] = partition
                labels = partition.predict([r.instance for r in branch.population])
                children = []
                for label in sorted(set(labels.tolist())):
                    identity = f"{branch.identity}.{label}"
                    members = [
                        r
                        for r, actual in zip(branch.population, labels)
                        if actual == label
                    ]
                    child = Branch(
                        identity,
                        members,
                        branch.depth,
                        branch.threshold,
                        branch.path + ((partition, label),),
                        sensitivity=dict(branch.sensitivity),
                    )
                    for result in members:
                        result.record["branch_id"] = identity
                    self._node(child)
                    children.append(child)
                node.update(
                    status="split",
                    children=[b.identity for b in children],
                    partition=partition.as_dict(),
                )
                return children
        scores = np.asarray([r.level_score for r in branch.population])
        threshold = max(
            branch.threshold, float(np.quantile(scores, 1 - self.config.elite_fraction))
        )
        terminal = threshold >= 1
        if terminal:
            representatives = diverse_representatives(
                [r for r in branch.population if r.failure],
                capacity=self.config.archive_capacity,
                root_limit=self.config.archive_root_limit,
            )
            if len(representatives) >= self.config.archive_capacity:
                node["status"] = "representative_target"
                return []
            if branch.terminal_used >= self.config.terminal_proposals:
                node["status"] = "terminal_quota"
                return []
        elif branch.depth >= self.config.subset_max_levels:
            node["status"] = "depth_limit"
            return []
        seeds = [r for r in branch.population if r.level_score >= threshold]
        proposal = self._proposal_configuration(branch)
        level = {
            "depth": branch.depth,
            "threshold": threshold,
            "group_probabilities": dict(zip(WORLD_RANDOM_BLOCKS, proposal[0].tolist())),
            "actor_mask": proposal[1].tolist(),
            "response_time_range": list(proposal[2]),
            "seed_states": [r.instance.instance_hash for r in seeds],
        }
        count = min(self.config.batch_size, self.main_limit - len(self.records))
        if terminal:
            count = min(count, self.config.terminal_proposals - branch.terminal_used)
        if count <= 0:
            return [branch]
        chains = list(seeds)
        states, changes = [], {b: [] for b in WORLD_RANDOM_BLOCKS}
        for index in range(count):
            if len(self.records) >= self.main_limit:
                break
            chain = index % len(chains)
            parent = chains[chain]
            omega, block = self._mutation(parent, guided=guided, proposal=proposal)
            instance = self.source.mutation(parent.instance, omega, block)
            child = self._evaluate(
                [instance],
                kind="terminal_proposal" if terminal else "conditional_proposal",
                branch=branch.identity,
            )[0]
            child.record["chain_id"] = f"{branch.identity}:{branch.depth}:{chain}"
            accepted = child.level_score >= threshold and branch.contains(
                child.instance
            )
            child.record["accepted"] = bool(accepted)
            if accepted:
                chains[chain] = child
            states.append(chains[chain])
            if block in changes:
                changes[block].append(
                    min(abs(child.score - parent.score) / (1 + abs(parent.score)), 10.0)
                )
        for block, values in changes.items():
            if values:
                branch.sensitivity[block] = 0.8 * branch.sensitivity[
                    block
                ] + 0.2 * float(np.mean(values))
        used = len(states)
        # A truncated batch does not silently discard tied, unvisited seed chains.
        if len(chains) > used:
            states.extend(chains[used:])
        branch.population, branch.threshold = states, threshold
        level.update(
            proposals=used, particle_states=[r.instance.instance_hash for r in states]
        )
        node.setdefault("levels", []).append(level)
        branch.allocated += used
        branch.terminal_used += used if terminal else 0
        branch.depth += 0 if terminal else 1
        self.batch_index += 1
        node.update(
            depth=branch.depth,
            threshold=threshold,
            allocated=branch.allocated,
            terminal_proposals=branch.terminal_used,
            feedback_weight=self._weight(branch),
            sensitivity=dict(branch.sensitivity),
        )
        return [branch]

    def _assign(self, result, root="root"):
        identity = root
        while self.nodes.get(identity, {}).get("status") == "split":
            label = int(self.partitions[identity].predict([result.instance])[0])
            identity = f"{identity}.{label}"
        return identity

    def run(self, method="guided", *, archive=None):
        if method not in ("monte_carlo", "subset", "guided"):
            raise ValueError("unknown search method")
        started = time.perf_counter()
        generation_before = self.source.generation_elapsed_s
        feedback = {}
        if archive is not None:
            endpoints = archive.endpoints(
                limit=min(self.config.replay_limit, max(0, self.config.budget - 1))
            )
            for old in endpoints:
                result = self._evaluate(
                    [old.instance], kind="archive_replay", branch=old.domain
                )[0]
                result.record.update(
                    old_safety_failure=old.failure, old_branch_id=old.domain
                )
                self.anchors.append(result)
            feedback = archive.feedback(self.anchors)
        # Early-stop runs reserve baseline diagnostics. Fixed-budget comparisons
        # spend the full allowance on each method's actual search procedure.
        reserve = (
            min(
                self.config.branch_check_budget,
                max(
                    0,
                    self.config.budget
                    - len(self.records)
                    - self.config.initial_samples,
                ),
            )
            if method != "guided" and not self.config.fixed_budget
            else 0
        )
        self.main_limit = self.config.budget - reserve
        initial = self._fresh(
            self.config.initial_samples, kind="initial_prior", branch="root"
        )
        active = [Branch("root", initial)] if initial else []
        if active:
            self._node(active[0])
        fresh_remaining = self.config.fresh_samples
        refresh_count = 0
        restart_count = 0
        while len(self.records) < self.main_limit:
            due = self.batch_index >= (2 if refresh_count == 0 else 4)
            if fresh_remaining and (due or not active):
                n = min(
                    self.config.batch_size,
                    fresh_remaining,
                    self.main_limit - len(self.records),
                )
                identity = f"refresh_{refresh_count}"
                population = self._fresh(n, kind="fresh_prior", branch=identity)
                if population:
                    branch = Branch(identity, population)
                    self._node(branch)
                    active.append(branch)
                fresh_remaining -= len(population)
                refresh_count += 1
                continue
            if method == "monte_carlo":
                population = self._fresh(
                    self.config.batch_size, kind="prior", branch="root"
                )
                self.batch_index += 1
                if not population:
                    break
                continue
            if not active:
                if not self.config.fixed_budget:
                    break
                identity = f"restart_{restart_count}"
                population = self._fresh(
                    self.config.initial_samples, kind="restart_prior", branch=identity
                )
                branch = Branch(identity, population)
                self._node(branch)
                active.append(branch)
                restart_count += 1
                continue
            branch = min(
                active,
                key=lambda b: (
                    b.allocated / (self._weight(b) if method == "guided" else 1),
                    b.identity,
                ),
            )
            active.remove(branch)
            active.extend(self._advance(branch, guided=method == "guided"))
        exploration_limit = self.main_limit
        if method != "guided":
            self.main_limit = self.config.budget
            population = [
                r
                for r in self.records
                if r.record["execution_kind"] != "archive_replay"
            ]
            partition, evidence = (
                self.partitioner.split(population, self._check)
                if population
                else (None, None)
            )
            self.nodes["posthoc"] = {
                "status": "partitioned" if partition is not None else "one_branch",
                "evidence": evidence,
            }
            labels = (
                partition.predict([r.instance for r in population])
                if partition is not None
                else np.zeros(len(population), int)
            )
            for result, label in zip(population, labels):
                result.record["branch_id"] = f"posthoc.{label}"
        else:
            for result in self.records:
                if result.record["execution_kind"] == "branch_check":
                    continue
                # Old labels and rejected proposals may lie outside their
                # recorded parent branch. Reclassify through the complete new
                # partition path from that exploration cohort's broad root.
                origin = (
                    "root"
                    if result.record["execution_kind"] == "archive_replay"
                    else result.record["branch_id"].split(".")[0]
                )
                result.record["branch_id"] = self._assign(result, origin)
        regular = [
            r for r in self.records if r.record["execution_kind"] != "branch_check"
        ]
        self.archive.add(regular, self.known)
        if archive is not None:
            self.archive.retain_previous(archive, self.anchors)
        transitions = [
            {
                "instance_hash": r.instance.instance_hash,
                "old_branch_id": r.record["old_branch_id"],
                "new_branch_id": r.domain,
                "old_failure": r.record["old_safety_failure"],
                "new_failure": r.failure,
            }
            for r in self.anchors
        ]
        stopped = (
            "execution_budget"
            if len(self.records) >= self.config.budget
            else "exploration_limit_with_reserved_diagnostics"
            if active and len(self.records) >= exploration_limit
            else "scheduled_tasks_complete"
        )
        for branch in active:
            self._node(branch).update(
                unfinished=True,
                particle_states=[r.instance.instance_hash for r in branch.population],
            )
        pending = {
            "initial_prior": max(
                0, self.config.initial_samples - self.costs["initial_prior"]
            ),
            "fresh_prior": fresh_remaining,
            "active_branches": [b.identity for b in active],
        }
        return SearchResult(
            self.records,
            self.archive,
            feedback,
            self.nodes,
            dict(self.costs),
            stopped,
            self.config.budget,
            time.perf_counter() - started,
            self.source.generation_elapsed_s - generation_before,
            transitions,
            pending,
            "fixed_budget" if self.config.fixed_budget else "early_stop",
            restart_count,
        )


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
            "failure_domain_interpretation": "empirical_search_branches_not_all_connected_components",
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
