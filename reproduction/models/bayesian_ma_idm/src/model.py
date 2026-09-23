"""Empirical-Bayes hierarchical B-IDM / MA-IDM calibration at 25 Hz.

The project environment intentionally has no PyMC.  This module therefore uses
MAP GP likelihoods plus a Laplace/driver-bootstrap population posterior.  Its
artifact format is joint log-normal draws, so replacing the fitting backend with
the paper's NUTS model does not change simulation or evaluation semantics.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np

PARAMETER_NAMES = ("v0_mps", "s0_m", "T_s", "alpha_mps2", "beta_mps2", "sigma_gp_mps2", "ell_s")
LOWER = np.asarray([8.0, 0.3, 0.25, 0.2, 0.2, 0.02, 0.08])
UPPER = np.asarray([60.0, 12.0, 4.0, 6.0, 8.0, 5.0, 8.0])
# Eq. (19) has a separate velocity observation-error term.  In acceleration
# residual units its native-25-Hz equivalent is fixed here, rather than being
# incorrectly absorbed by the persistent GP (which otherwise drives ell to a
# one-frame boundary).  This is a deliberately conservative 0.004 m speed error.
ACCELERATION_OBSERVATION_NOISE = 0.10


def _nearest_pd(matrix: np.ndarray) -> np.ndarray:
    matrix = (matrix + matrix.T) / 2.0
    vals, vectors = np.linalg.eigh(matrix)
    floor = max(1.0e-6, float(np.max(vals)) * 1.0e-5)
    return (vectors * np.maximum(vals, floor)) @ vectors.T


def sample_driver_joint(posterior: dict[str, np.ndarray], rng: np.random.Generator) -> np.ndarray:
    """Draw all seven parameters together, preserving posterior dependence.

    The standard artifact stores a population log-normal approximation.  A
    PyMC artifact may additionally retain individual ``theta`` draws; this is
    the sampling organisation used by the authors' ring simulator (one MCMC
    draw and one fitted driver), and is preferable for that particular
    retrospective Fig. 10 reproduction.
    """
    if "driver_parameter_draws" in posterior:
        sample = int(rng.integers(len(posterior["driver_parameter_draws"])))
        driver = int(rng.integers(posterior["driver_parameter_draws"].shape[1]))
        theta = posterior["driver_parameter_draws"][sample, driver]
        sigma = posterior["driver_sigma_draws"][sample]
        ell = posterior["driver_ell_draws"][sample]
        # PyMC uses positive transforms already.  Do not impose the empirical-
        # Bayes optimizer bounds here: doing so would alter a retained MCMC
        # draw and defeat the source-simulator sampling scheme.
        iid = (float(posterior["iid_action_sigma_draws"][sample % len(posterior["iid_action_sigma_draws"])])
               if "iid_action_sigma_draws" in posterior else ACCELERATION_OBSERVATION_NOISE)
        return np.r_[theta, sigma, ell, iid]
    draw = int(rng.integers(len(posterior["population_log_mean"])))
    value = rng.multivariate_normal(posterior["population_log_mean"][draw],
                                   posterior["population_log_cov"][draw])
    parameters = np.clip(np.exp(value), LOWER, UPPER)
    if "iid_action_sigma_draws" in posterior:
        iid = np.asarray(posterior["iid_action_sigma_draws"], float)
        parameters = np.r_[parameters, iid[int(rng.integers(len(iid)))]]
    return parameters


def load_posterior(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}
