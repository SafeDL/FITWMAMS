"""Train-only duration and shape calibration for the shared lane executor."""

import numpy as np
import torch

from .lane_trajectory import SHAPE_BASIS, path_coefficients

SHAPE_FIT_TIME_SCALE_S = 2.0


def training_parameters(events, observations):
    features, parameters, acceleration, jerk = [], [], [], []
    for event in events.itertuples():
        data = observations["states"][
            observations["offsets"][event.trajectory_index] : observations["offsets"][
                event.trajectory_index + 1
            ]
        ][25 : 25 + int(round(event.duration_s / 0.04))]
        x = np.r_[0, np.cumsum(data[:-1, 1] * 0.04)]
        target = event.target_y_m - event.source_y_m
        state = torch.tensor([0, data[0, 0], data[0, 1], data[0, 2], 0, data[0, 3]])
        distance = state.new_tensor(float(x[-1]))
        base = path_coefficients(
            state, state.new_tensor(target), distance, state.new_zeros(2)
        ).numpy()
        u = x / x[-1]
        basis = np.asarray(SHAPE_BASIS, float) * (target - data[0, 0])
        vx = data[:, 1].astype(float)
        ax = np.gradient(vx, 0.04)
        polynomial = np.polynomial.polynomial
        position = np.column_stack([polynomial.polyval(u, p) for p in basis])
        slope = np.column_stack(
            [polynomial.polyval(u, polynomial.polyder(p)) for p in basis]
        ) / float(distance)
        curvature = (
            np.column_stack(
                [polynomial.polyval(u, polynomial.polyder(p, 2)) for p in basis]
            )
            / float(distance) ** 2
        )
        base_slope = polynomial.polyval(u, polynomial.polyder(base)) / float(distance)
        base_curvature = (
            polynomial.polyval(u, polynomial.polyder(base, 2)) / float(distance) ** 2
        )
        tau = SHAPE_FIT_TIME_SCALE_S
        # Scale velocity and acceleration errors to the same units as position.
        design = np.vstack(
            (
                position,
                tau * slope * vx[:, None],
                tau**2 * (curvature * vx[:, None] ** 2 + slope * ax[:, None]),
            )
        )
        error = np.r_[
            data[:, 0] - polynomial.polyval(u, base),
            tau * (data[:, 2] - base_slope * vx),
            tau**2 * (data[:, 3] - base_curvature * vx**2 - base_slope * ax),
        ]
        shape = np.linalg.lstsq(design, error, rcond=None)[0]
        direction = np.sign(target - data[0, 0])
        features.append(
            [
                data[0, 1],
                np.log(event.length_m),
                np.log(event.width_m),
                abs(target - data[0, 0]),
                direction * data[0, 2],
                direction * data[0, 3],
            ]
        )
        parameters.append([np.log(x[-1] / data[0, 1]), *shape])
        acceleration.append(np.abs(data[:, 3]).max())
        jerk.append(np.quantile(np.abs(np.diff(data[:, 3])) / 0.04, 0.99))
    return (
        np.asarray(features),
        np.asarray(parameters),
        np.asarray(acceleration),
        np.asarray(jerk),
    )
