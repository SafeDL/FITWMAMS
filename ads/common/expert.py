"""Bounded closed-loop expert; only validated successful executions yield labels."""

import time
from dataclasses import dataclass, replace

import numpy as np

from ads.common.environment import PolicyEnvironment
from ads.common.safety import initial_safety


@dataclass(frozen=True)
class Candidate:
    lane_id: int
    speed: float
    longitudinal_start: float = 0.0
    lateral_start: float = 0.0
    duration: float = 2.4


def coarse_candidates(event):
    geometry = event.geometry
    ego = event.initial_states[0]
    lane = geometry.nearest_lane(ego[:2])
    lanes = [lane] + [
        i
        for i in (geometry.adjacent_lane(lane, -1), geometry.adjacent_lane(lane, 1))
        if i is not None
    ]
    speed = float(np.linalg.norm(ego[2:4]))
    candidates = [
        Candidate(i, float(np.clip(speed + delta, 0, 45)), start, start)
        for i in lanes
        for delta in (-12, -6, 0, 4)
        for start in (0, 0.4, 0.8, 1.2)
    ]
    candidates += [
        Candidate(i, float(np.clip(speed + delta, 0, 45)), duration=duration)
        for i in lanes
        for delta in (-4, 0)
        for duration in (1.6, 3.2)
    ]
    candidates += [Candidate(lane, 0, start, start) for start in (0, 0.4, 0.8, 1.2)]
    return list(dict.fromkeys(candidates))[:64]


def select_teacher_instances(failures, neighbours, limit=32):
    """Budget includes neighbour supervision; failed representatives have unique roots."""
    selected, roots = [], set()
    for instance, domain in failures:
        if instance.root_sample_id in roots:
            continue
        roots.add(instance.root_sample_id)
        selected.append((instance, domain))
        if len(selected) >= limit:
            break
        neighbour = neighbours.get(instance.instance_hash)
        if neighbour is not None:
            selected.append(neighbour)
        if len(selected) >= limit:
            break
    return selected


def refine_candidates(candidates):
    refined = []
    for c in candidates[:8]:
        for key, delta in (
            ("speed", 2),
            ("speed", -2),
            ("longitudinal_start", 0.2),
            ("longitudinal_start", -0.2),
            ("lateral_start", 0.2),
            ("lateral_start", -0.2),
            ("duration", 0.4),
            ("duration", -0.4),
        ):
            value = getattr(c, key) + delta
            value = (
                float(np.clip(value, 0, 45))
                if key == "speed"
                else max(0.4 if key == "duration" else 0, value)
            )
            refined.append(replace(c, **{key: value}))
    return list(dict.fromkeys(refined))[:64]


class CandidateController:
    def __init__(self, event, candidate):
        self.candidate, self.geometry = candidate, event.geometry
        self.start_y = float(event.initial_states[0, 1])
        self.initial_speed = float(np.linalg.norm(event.initial_states[0, 2:4]))
        self.target_lane_id = candidate.lane_id
        self.target_speed_mps = candidate.speed
        self.time = 0.0

    @property
    def target_y(self):
        return self.geometry.center_y(self.target_lane_id)

    def lateral_reference(self, time):
        u = np.clip(
            (np.asarray(time) - self.candidate.lateral_start) / self.candidate.duration,
            0,
            1,
        )
        fraction = 10 * u**3 - 15 * u**4 + 6 * u**5
        derivative = (30 * u**2 - 60 * u**3 + 30 * u**4) / self.candidate.duration
        return self.start_y + fraction * (self.target_y - self.start_y), derivative * (
            self.target_y - self.start_y
        )

    def update_target(self, action):
        return True

    def control(self, state):
        speed = float(np.linalg.norm(state[2:4]))
        desired_speed = (
            self.initial_speed
            if self.time < self.candidate.longitudinal_start
            else self.candidate.speed
        )
        y, vy = self.lateral_reference(self.time)
        target_vy = vy + 1.5 * (y - state[1])
        heading = np.arctan2(state[3], state[2])
        target_heading = np.arcsin(np.clip(target_vy / max(speed, 0.5), -0.9, 0.9))
        error = np.arctan2(
            np.sin(target_heading - heading), np.cos(target_heading - heading)
        )
        self.time += 0.04
        return np.asarray(
            (
                np.clip(1.5 * (desired_speed - speed), -8, 4),
                np.clip(5 * error, -0.6, 0.6),
            ),
            np.float32,
        )


class TeacherEnvironment(PolicyEnvironment):
    def __init__(self, instance, factory, candidate):
        self.candidate = candidate
        super().__init__("plant2", [instance], factory)

    def _make_controller(self, event):
        return CandidateController(event, self.candidate)

    def _observation(self):
        return np.zeros(1, np.float32)


@dataclass
class TeacherExecution:
    candidate: Candidate
    success: bool
    reward: float
    progress: float
    states: np.ndarray
    controls: np.ndarray


def execute_candidate(instance, factory, candidate):
    env = TeacherEnvironment(instance, factory, candidate)
    env.reset()
    collision, offroad, _ = initial_safety(instance.event)
    terminated, truncated = collision or offroad, False
    info, reward_sum = {"stalled": False}, 0.0
    while not (terminated or truncated):
        _, reward, terminated, truncated, info = env.step(0)
        reward_sum += reward
    states = np.asarray(env.physics_states)
    if not np.isfinite(states).all():
        raise FloatingPointError("expert produced non-finite world states")
    return TeacherExecution(
        candidate,
        not terminated and not info["stalled"],
        reward_sum,
        float(states[-1, 0, 0] - states[0, 0, 0]),
        states,
        np.asarray(env.physics_controls).reshape(-1, 2),
    )


def expert_rank(execution, event):
    if not execution.success:
        return False, len(execution.states), 0.0, execution.reward, execution.progress
    speed = float(np.linalg.norm(event.initial_states[0, 2:4]))
    lane = event.geometry.nearest_lane(event.initial_states[0, :2])
    lane_change = abs(
        event.geometry.center_y(execution.candidate.lane_id)
        - event.geometry.center_y(lane)
    ) / float(event.geometry.lane_widths.min())
    intent_change = lane_change + abs(execution.candidate.speed - speed) / max(speed, 2)
    normalized = execution.controls / np.asarray((8.0, 0.6))
    smoothness = float(np.square(normalized).mean()) if len(normalized) else 0.0
    if len(normalized) > 1:
        smoothness += float(np.square(np.diff(normalized, axis=0)).mean())
    return True, -intent_change, -smoothness, execution.reward, execution.progress


def successful_expert(instance, factory, *, budget=128):
    if not 1 <= budget <= 128:
        raise ValueError("expert candidate budget must be in [1,128]")
    started = time.perf_counter()
    executions, tried = [], set()
    coarse = coarse_candidates(instance.event)[: min(64, budget)]
    for candidate in coarse:
        executions.append(execute_candidate(instance, factory, candidate))
        tried.add(candidate)
    coarse_count = len(executions)
    rank = lambda execution: expert_rank(execution, instance.event)
    leaders = sorted(executions, key=rank, reverse=True)[:8]
    for candidate in refine_candidates([e.candidate for e in leaders]):
        if len(executions) == budget:
            break
        if candidate not in tried:
            executions.append(execute_candidate(instance, factory, candidate))
            tried.add(candidate)
    success = [e for e in executions if e.success]
    return (max(success, key=rank) if success else None), {
        "instance_hash": instance.instance_hash,
        "candidates": len(executions),
        "candidate_budget": budget,
        "coarse_candidates": coarse_count,
        "refined_candidates": len(executions) - coarse_count,
        "successful_candidates": len(success),
        "qualified_demonstration": bool(success),
        "selection_rule": "minimum_intent_change_then_smoothness_v1",
        "elapsed_s": time.perf_counter() - started,
    }
