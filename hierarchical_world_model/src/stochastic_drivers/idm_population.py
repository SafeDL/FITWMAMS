"""Empirical driver variation for the five-parameter IDM, not a GP posterior."""

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import numpy as np
import torch
from scipy.optimize import least_squares

PARAMETER_NAMES = ("v0_mps", "s0_m", "T_s", "a_mps2", "b_mps2")
IDM_EQUATION = "s_star = s0 + max(0, v*T + v*(v-v_leader)/(2*sqrt(a*b)))"
LOWER = np.asarray([8.0, 0.3, 0.25, 0.2, 0.2])
UPPER = np.asarray([60.0, 12.0, 4.0, 6.0, 8.0])
MODEL_PATH = (
    Path(__file__).resolve().parents[3]
    / "results/hierarchical_world_model/model_assets/idm_population.npz"
)


def idm_acceleration(gap, speed, leader_speed, theta):
    v0, s0, headway, acceleration, braking = np.moveaxis(np.asarray(theta), -1, 0)
    dynamic = speed * headway + speed * (speed - leader_speed) / (
        2.0 * np.sqrt(acceleration * braking)
    )
    desired = s0 + np.maximum(dynamic, 0.0)
    return acceleration * (
        1.0 - (speed / v0) ** 4 - (desired / np.maximum(gap, 1.0e-3)) ** 2
    )


def torch_idm(gap, speed, leader_speed, theta):
    """The same IDM equation in calibration, longitudinal response, and MOBIL."""
    v0, s0, headway, acceleration, braking = theta.unbind(dim=-1)
    speed = speed.clamp_min(0.0)
    dynamic = speed * headway + speed * (speed - leader_speed) / (
        2.0 * torch.sqrt((acceleration * braking).clamp_min(1.0e-8))
    )
    desired = s0 + dynamic.clamp_min(0.0)
    return acceleration * (
        1.0
        - (speed / v0.clamp_min(1.0e-4)).pow(4)
        - (desired / gap.clamp_min(1.0e-3)).pow(2)
    )


def fit_parameters(data, *, prior=None, regularization=0.0):
    observations = {
        name: np.asarray(data[name], np.float64)
        for name in ("gap", "speed", "leader_speed", "acceleration")
    }
    initial = np.log([33.0, 2.0, 1.3, 1.2, 2.0]) if prior is None else prior
    normalizer = np.sqrt(len(observations["gap"]))

    def residual(log_theta):
        errors = (
            idm_acceleration(
                observations["gap"],
                observations["speed"],
                observations["leader_speed"],
                np.exp(log_theta),
            )
            - observations["acceleration"]
        )
        if prior is not None:
            errors = np.r_[
                errors, np.sqrt(regularization * len(errors)) * (log_theta - prior)
            ]
        return errors / normalizer

    def jacobian(log_theta):
        v0, s0, headway, acceleration, braking = np.exp(log_theta)
        speed, gap = observations["speed"], observations["gap"]
        closing = speed - observations["leader_speed"]
        time_gap = speed * headway
        approach_gap = speed * closing / (2.0 * np.sqrt(acceleration * braking))
        active = time_gap + approach_gap > 0.0
        desired = s0 + np.maximum(time_gap + approach_gap, 0.0)
        free = (speed / v0) ** 4
        predicted = acceleration * (1.0 - free - (desired / gap) ** 2)
        sensitivity = -2.0 * acceleration * desired / gap**2
        jac = np.column_stack(
            (
                4.0 * acceleration * free,
                sensitivity * s0,
                sensitivity * time_gap * active,
                predicted - 0.5 * sensitivity * approach_gap * active,
                -0.5 * sensitivity * approach_gap * active,
            )
        )
        if prior is not None:
            jac = np.vstack((jac, np.eye(5) * np.sqrt(regularization * len(gap))))
        return jac / normalizer

    fit = least_squares(
        residual,
        initial,
        jac=jacobian,
        bounds=(np.log(LOWER), np.log(UPPER)),
        loss="soft_l1",
        f_scale=0.3 / normalizer,
        max_nfev=2000,
        x_scale="jac",
        ftol=1.0e-6,
        xtol=1.0e-6,
        gtol=1.0e-8 if prior is None else 1.0e-3,
    )
    if not fit.success:
        raise RuntimeError(
            f"IDM fit did not converge: {fit.message}; "
            f"parameters={np.exp(fit.x)}, optimality={fit.optimality}"
        )
    return fit.x


_FIT_DATA = None
_FIT_PRIOR = None
_FIT_REGULARIZATION = None


def _fit_driver(indices):
    observations = {
        name: _FIT_DATA[name][indices]
        for name in ("gap", "speed", "leader_speed", "acceleration")
    }
    return fit_parameters(
        observations, prior=_FIT_PRIOR, regularization=_FIT_REGULARIZATION
    )


def fit_population(data, *, regularization, global_parameters, workers=1):
    """Shrink individual fits toward Train's global fit to limit identifiability errors."""
    keys = np.column_stack((data["recording"], data["vehicle"]))
    drivers, index, counts = np.unique(
        keys, axis=0, return_inverse=True, return_counts=True
    )
    order = np.argsort(index, kind="stable")
    boundaries = np.r_[0, np.cumsum(counts)]
    eligible = np.flatnonzero(counts >= 30)
    selections = [order[boundaries[i] : boundaries[i + 1]] for i in eligible]
    global _FIT_DATA, _FIT_PRIOR, _FIT_REGULARIZATION
    _FIT_DATA = {
        name: np.asarray(data[name])
        for name in ("gap", "speed", "leader_speed", "acceleration")
    }
    _FIT_PRIOR, _FIT_REGULARIZATION = global_parameters, regularization
    if workers > 1:
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=get_context("fork")
        ) as pool:
            fits = []
            for result in pool.map(_fit_driver, selections, chunksize=64):
                fits.append(result)
                if len(fits) % 5000 == 0:
                    print(f"IDM drivers {len(fits):,}/{len(eligible):,}", flush=True)
    else:
        fits = [_fit_driver(selected) for selected in selections]
    parameters = np.asarray(fits)
    covariance = np.cov(parameters, rowvar=False) + np.eye(5) * 1.0e-6
    error = data["acceleration"] - idm_acceleration(
        data["gap"],
        data["speed"],
        data["leader_speed"],
        np.exp(parameters.mean(axis=0)),
    )
    return {
        "log_mean": parameters.mean(axis=0),
        "log_cov": covariance,
        "residual_std": np.asarray(max(float(np.std(error)), 0.05)),
        "fit_driver_count": np.asarray(len(parameters)),
        "regularization": np.asarray(regularization),
        "driver_keys": drivers[eligible],
        "driver_log_parameters": parameters,
    }


def sample_theta(population, latent):
    """Preserve the joint parameter covariance and one style per physical vehicle."""
    eigenvalues, eigenvectors = np.linalg.eigh(population["log_cov"])
    transform = eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 0.0)))
    return np.clip(
        np.exp(population["log_mean"] + np.asarray(latent) @ transform.T), LOWER, UPPER
    ).astype(np.float32)


def population_driver_parameters(population, data, *, seed=20261002, samples=16):
    drivers, index = np.unique(
        np.column_stack((data["recording"], data["vehicle"])),
        axis=0,
        return_inverse=True,
    )
    latent = np.random.default_rng(seed).standard_normal((samples, len(drivers), 5))
    return sample_theta(population, latent), index


def population_predictions(population, data, *, seed=20261002, samples=16):
    theta, index = population_driver_parameters(
        population, data, seed=seed, samples=samples
    )
    return np.asarray(
        [
            idm_acceleration(
                data["gap"], data["speed"], data["leader_speed"], values[index]
            )
            for values in theta
        ]
    )
