"""Check whether a successful expert trace is executable by the ADS tracker."""

import time

import numpy as np

from ads.common.environment import PolicyEnvironment
from ads.common.instances import file_hash
from ads.common.observations import to_local
from ads.nuplan.features import rear_offset
from hierarchical_world_model.src.execution import trajectory_event_risk


class LabelTracker:
    """Diagnostic ego reference; never used as a tested ADS or as a learned policy."""

    def __init__(self, instance, states):
        self.states = np.asarray(states)
        self.offset = rear_offset(instance.event.vehicle_dimensions)
        self.angle = np.unwrap(np.arctan2(self.states[:, 0, 3], self.states[:, 0, 2]))

    def predict(self, observation, deterministic=True):
        scene = observation["scene"]
        frame = round(scene.elapsed_s / 0.04)
        query = frame + np.arange(1, 81) * 2.5
        observed_time = np.arange(len(self.states))
        center = np.stack(
            [
                np.interp(query, observed_time, self.states[:, 0, axis])
                for axis in (0, 1)
            ],
            -1,
        )
        beyond = np.maximum(query - observed_time[-1], 0)
        # Continuation only supplies a tracker lookahead at the episode boundary.
        # It never enters training targets; all unobserved future labels stay masked.
        center += beyond[:, None] * 0.04 * self.states[-1, 0, 2:4]
        angles = np.interp(query, observed_time, self.angle)
        rear = center - self.offset * np.stack((np.cos(angles), np.sin(angles)), -1)
        ego = scene.states[0].copy()
        heading = np.arctan2(ego[3], ego[2])
        ego[:2] -= self.offset * np.array((np.cos(heading), np.sin(heading)))
        local = to_local(rear, ego)
        relative = (angles - heading + np.pi) % (2 * np.pi) - np.pi
        return np.column_stack((local, relative)).astype(np.float32), None


def check(instance, states, factory, ledger, *, snapshot=None, trace_path=None):
    ledger.register(instance, "repair")
    started = time.perf_counter()
    env = PolicyEnvironment(
        "diffusion_planner",
        [instance],
        factory,
        purpose="repair",
        ledger=ledger,
        random_events=False,
    )
    observation, _ = env.reset()
    if snapshot is not None:
        observation = env.restore(snapshot)
    start = int(env.world.reference_index)
    policy = LabelTracker(instance, states)
    terminated, truncated = env.initial_failure, False
    info = {
        "stalled": False,
        "collision": env.initial_info["collision"],
        "offroad": env.initial_info["offroad"],
    }
    while not (terminated or truncated):
        action, _ = policy.predict(observation)
        observation, _, terminated, truncated, info = env.step(action)
    actual = np.asarray(env.physics_states)
    overlap = min(len(actual), len(states))
    error = np.linalg.norm(
        actual[start:overlap, 0, :2] - np.asarray(states)[start:overlap, 0, :2], axis=-1
    )
    report = {
        "helper_sha256": file_hash(__file__),
        "success": not terminated and not info["stalled"],
        "collision": info["collision"],
        "offroad": info["offroad"],
        "stalled": info["stalled"],
        "start_frame": start,
        "physics_steps": len(actual) - 1 - start,
        "mean_tracking_error_m": float(error.mean()),
        "max_tracking_error_m": float(error.max()),
        "elapsed_s": time.perf_counter() - started,
        "training_labels_extrapolated": False,
        "event_risk": float(
            trajectory_event_risk(
                actual[None, start:],
                instance.event.valid[None],
                vehicle_dimensions=instance.event.vehicle_dimensions[None],
                excluded_slots=(),
            )[0]
        ),
    }
    if trace_path is not None:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            trace_path, states=actual, controls=np.asarray(env.physics_controls)
        )
        report.update(
            trajectory_path=str(trace_path.resolve()),
            trajectory_sha256=file_hash(trace_path),
        )
    return report
