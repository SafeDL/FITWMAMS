"""Reuse bounded highway experts and mask labels outside the executed horizon."""

import numpy as np

from ads.common.observations import CausalScene, to_local
from ads.nuplan.features import build_features, rear_offset, sample_history


def demonstration(instance, states, algorithm):
    """Native labels in the present ego rear-axle frame, observed futures only."""
    event = instance.event
    samples = []
    for frame in range(0, len(states) - 25, 5):
        current = states[frame]
        h = float(np.arctan2(current[0, 3], current[0, 2]))
        speed = float(np.linalg.norm(current[0, 2:4]))
        yaw = (-current[0, 4] * np.sin(h) + current[0, 5] * np.cos(h)) / max(speed, 0.5)
        wheelbase = 3.089 / 5.176 * event.vehicle_dimensions[0, 0]
        scene = CausalScene(
            current,
            event.valid,
            event.vehicle_dimensions,
            event.geometry,
            frame * 0.04,
            len(event.soft_reference) * 0.04,
            np.zeros(2),
            yaw,
            0,
            float(np.arctan(yaw * wheelbase / max(speed, 0.5))),
            0,
        )
        history, past_valid = sample_history(
            states,
            frame,
            initial_history=event.initial_history,
            initial_history_valid=event.initial_history_valid,
        )
        past_valid &= event.valid[None]
        feature, slots = build_features(scene, history, past_valid, algorithm=algorithm)
        query = frame + np.arange(1, 81) * 2.5
        within = query <= len(states) - 1
        future = (
            np.stack(
                [
                    np.interp(query, np.arange(len(states)), states[:, slot, axis])
                    for slot in slots
                    for axis in range(6)
                ],
                -1,
            )
            .reshape(80, len(slots), 6)
            .transpose(1, 0, 2)
        )
        heading = np.arctan2(future[..., 3], future[..., 2])
        future[0, :, :2] -= rear_offset(event.vehicle_dimensions) * np.stack(
            (np.cos(heading[0]), np.sin(heading[0])), -1
        )
        ego = current[0].copy()
        ego[:2] -= rear_offset(event.vehicle_dimensions) * np.array(
            (np.cos(h), np.sin(h))
        )
        position = to_local(future[..., :2], ego)
        angle = (heading - h + np.pi) % (2 * np.pi) - np.pi
        target = np.concatenate((position, angle[..., None]), -1).astype(np.float32)
        valid = event.valid[slots, None] & within[None]
        target[~valid] = 0
        sample = {
            "features": feature,
            "target": target,
            "future_valid": valid,
            "observation": {
                "features": feature,
                "scene": scene,
                "slots": slots,
                "history": history,
                "history_valid": past_valid,
            },
        }
        samples.append(sample)
    return samples
