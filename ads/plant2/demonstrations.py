"""PlanT-specific targets from a validated closed-loop expert execution."""

import numpy as np

from ads.common.expert import CandidateController
from ads.common.observations import CausalScene, plant_observation, to_local


def demonstration(instance, execution, cfg):
    states = execution.states
    event = instance.event
    controller = CandidateController(event, execution.candidate)
    samples, clipped = [], 0
    training = cfg.model.training
    for frame in range(0, len(states), 5):
        # Eight 0.25 s future points must all lie inside the successful rollout.
        if frame + 50 > len(states) - 1:
            continue
        current = states[frame]
        scene = CausalScene(
            current,
            event.valid,
            event.vehicle_dimensions,
            event.geometry,
            frame * 0.04,
            len(event.soft_reference) * 0.04,
            np.zeros(2),
            0,
            0,
            0,
            0,
        )
        sample = plant_observation(
            scene,
            range_m=training.range,
            front_factor=training.get("range_factor_front", 1),
        )
        times = frame + np.arange(1, 9) * 6.25
        future_xy = np.stack(
            [
                np.interp(times, np.arange(len(states)), states[:, 0, axis])
                for axis in (0, 1)
            ],
            -1,
        )
        sample["waypoints"] = to_local(future_xy, current[0], carla=True).astype(
            np.float32
        )
        # The expert's reference path is supervision; route_original remains map-only.
        dense_x = current[0, 0] + np.linspace(0, 50, 501)
        reference_times = frame * 0.04 + (dense_x - current[0, 0]) / max(
            float(np.linalg.norm(current[0, 2:4])), 2
        )
        dense_y, _ = controller.lateral_reference(reference_times)
        dense_y += (current[0, 1] - dense_y[0]) * np.exp(-(dense_x - dense_x[0]) / 4)
        dense = np.stack((dense_x, dense_y), -1)
        arc = np.concatenate(
            ([0], np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=-1)))
        )
        path = np.stack(
            [np.interp(np.arange(20), arc, dense[:, axis]) for axis in (0, 1)], -1
        )
        sample["path"] = to_local(path, current[0], carla=True).astype(np.float32)
        observed_distance = np.linalg.norm(
            np.diff(states[frame:, 0, :2], axis=0), axis=-1
        ).sum()
        sample["path_valid"] = np.arange(20) <= observed_distance
        future = frame + 6.25
        forecast_states = np.asarray(
            [
                np.interp(future, np.arange(len(states)), states[:, slot, axis])
                for slot in sample["object_slots"]
                for axis in range(6)
            ],
            np.float32,
        ).reshape(-1, 6)
        xy = to_local(forecast_states[:, :2], current[0], carla=True)
        headings = -(
            np.arctan2(forecast_states[:, 3], forecast_states[:, 2]) - scene.heading
        )
        speeds = np.linalg.norm(forecast_states[:, 2:4], axis=-1) * 3.6
        clipped += int((speeds > 120).sum())
        quant = np.stack(
            (
                (xy[:, 0] + training.range) / (2 * training.range),
                (xy[:, 1] + training.range) / (2 * training.range),
                np.mod(np.rad2deg(headings), 360) / 360,
                speeds / 120,
            ),
            -1,
        )
        quant[:, 0] = np.clip(
            quant[:, 0], 0, (1 + training.get("range_factor_front", 1)) / 2
        )
        quant[:, 1:] = np.clip(quant[:, 1:], 0, 1)
        pre = cfg.model.pre_training
        scale = np.asarray(
            (
                2**pre.precision_pos - 1,
                2**pre.precision_pos - 1,
                2**pre.precision_angle - 1,
                2**pre.precision_speed - 1,
            )
        )
        sample["forecast"] = np.rint(quant * scale).astype(np.int64)
        samples.append(sample)
    return samples, clipped
