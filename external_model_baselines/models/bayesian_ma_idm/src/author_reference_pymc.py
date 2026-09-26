"""PyMC implementation matching ``PGM_highD/MA_IDM_hierarchical.ipynb``.

It is intentionally separate from the native-25-Hz migration backend: this
uses the author's fixed 20 cached pairs, 5 Hz states, 20-point/4-s windows and
10-point hop, so its posterior and paper figures are directly comparable.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from .author_reference import load_author_reference

DT, WINDOW, HOP = .2, 20, 10
REFERENCE = np.asarray([33., 2., 1.6, 1.5, 1.67])


def _windows(path: str | Path) -> tuple[dict[str, np.ndarray], list[dict[str, np.ndarray]]]:
    _, pairs = load_author_reference(path)
    values = {name: [] for name in ("speed", "gap", "closing", "next_speed", "driver")}
    for driver, pair in enumerate(pairs):
        # Exact formalize_array(..., slice_len=20, skip=10) count in source.
        count = int(np.floor((len(pair["speed"] if "speed" in pair else pair["follower_v"]) - WINDOW + 1) / HOP))
        for index in range(count):
            begin = index * HOP; end = begin + WINDOW
            values["speed"].append(pair["follower_v"][begin:end])
            values["gap"].append(pair["gap"][begin:end])
            values["closing"].append(pair["follower_v"][begin:end] - pair["leader_v"][begin:end])
            values["next_speed"].append(pair["next_speed"][begin:end])
            values["driver"].append(np.full(WINDOW, driver, dtype=np.int64))
    return {name: np.asarray(value, dtype=np.float64 if name != "driver" else np.int64) for name, value in values.items()}, pairs


def fit_author_reference(dataset_path: str | Path, output_dir: str | Path, *, tune: int = 5000, draws: int = 2500,
                         chains: int = 2, cores: int = 1, target_accept: float = .90, seed: int = 16) -> Path:
    import pymc as pm
    import pytensor.tensor as pt
    arrays, pairs = _windows(dataset_path)
    n_drivers, times = len(pairs), np.arange(WINDOW, dtype=np.float64)
    coords = {"veh_id": np.arange(n_drivers), "obs_id": np.arange(len(arrays["speed"])), "time_stamp": np.arange(WINDOW), "parameter": np.arange(5)}
    with pm.Model(coords=coords):
        driver = pm.Data("id_idx", arrays["driver"], dims=("obs_id", "time_stamp"))
        speed, gap, closing = (pm.Data(name, arrays[name], dims=("obs_id", "time_stamp")) for name in ("speed", "gap", "closing"))
        chol, _, _ = pm.LKJCholeskyCov("chol", n=5, eta=2., sd_dist=pm.Exponential.dist(100., shape=5))
        values_raw = pm.Normal("vals_raw", 0., 1., dims=("veh_id", "parameter"))
        log_mu = pm.Normal("log_mu", 0., 1., dims="parameter")
        theta = pm.Deterministic("theta", pt.exp(log_mu + values_raw @ chol.T) * REFERENCE, dims=("veh_id", "parameter"))
        ell_frames = pm.Normal("ell_frames", mu=5., sigma=5.)
        s2_f = pm.Exponential("s2_f", lam=3.e4)
        s2_a = pm.Exponential("s2_a", lam=1.e5)
        distance = times[:, None] - times[None, :]
        covariance = DT ** 2 * (s2_f * pt.exp(-.5 * (distance / ell_frames) ** 2) + s2_a * pt.eye(WINDOW))
        selected = theta[driver]
        # ``id_idx`` is [window,time] exactly as in the notebook; PyMC6 keeps
        # that pair of axes, so select the final parameter axis explicitly.
        desired_gap = selected[:, :, 1] + speed * selected[:, :, 2] + speed * closing / (2. * pt.sqrt(selected[:, :, 3] * selected[:, :, 4]))
        acceleration = selected[:, :, 3] * (1. - (speed / selected[:, :, 0]) ** 4 - (desired_gap / gap) ** 2)
        pm.MvNormal("obs", mu=speed + DT * acceleration, cov=covariance, observed=arrays["next_speed"], dims=("obs_id", "time_stamp"))
        trace = pm.sample(draws=draws, tune=tune, chains=chains, cores=cores, target_accept=target_accept,
                          random_seed=seed, init="jitter+adapt_diag", return_inferencedata=True, progressbar=True)
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    trace_path = output / "author_reference_trace.nc"; trace.to_netcdf(trace_path)
    theta_draws = np.asarray(trace.posterior["theta"]).reshape(-1, n_drivers, 5)
    sigma_draws = np.sqrt(np.asarray(trace.posterior["s2_f"]).reshape(-1))
    ell_draws = np.abs(np.asarray(trace.posterior["ell_frames"]).reshape(-1)) * DT
    # Retain source-style draws as the primary simulation representation.  The
    # population moments are supplied only for common artifact compatibility.
    means, covariances = [], []
    for sample, sigma, ell in zip(theta_draws, sigma_draws, ell_draws):
        logs = np.log(sample)
        mean = np.r_[np.mean(logs, axis=0), np.log([sigma, ell])]
        cov = np.zeros((7, 7)); cov[:5, :5] = np.cov(logs, rowvar=False) + np.eye(5) * .0025; cov[5, 5] = cov[6, 6] = .0025
        means.append(mean); covariances.append(cov)
    posterior_path = output / "author_reference_ma_idm_posterior.npz"
    np.savez_compressed(posterior_path, parameter_names=np.asarray(("v0_mps", "s0_m", "T_s", "alpha_mps2", "beta_mps2", "sigma_gp_mps2", "ell_s")),
                        population_log_mean=np.asarray(means), population_log_cov=np.asarray(covariances),
                        driver_map=np.column_stack((np.mean(theta_draws, axis=0), np.full(n_drivers, np.mean(sigma_draws)), np.full(n_drivers, np.mean(ell_draws)))),
                        driver_parameter_draws=theta_draws, driver_sigma_draws=sigma_draws, driver_ell_draws=ell_draws,
                        driver_train_indices=np.arange(n_drivers), model=np.asarray("ma_idm"), fit_backend=np.asarray("pymc_author_notebook_exact_5hz"),
                        dt_s=np.asarray(DT), calibration_window_s=np.asarray(4.))
    import arviz as az
    diagnostic = az.summary(trace, var_names=["log_mu", "s2_f", "s2_a", "ell_frames"], kind="diagnostics")
    report = {"reference": "Zhang & Sun public MA_IDM_hierarchical.ipynb", "author_dataset_protocol": "fixed Config.py 20 pairs",
              "fps": 5, "dt_s": DT, "window_points": WINDOW, "window_s": 4., "hop_points": HOP, "hop_s": 2.,
              "tune": tune, "draws_per_chain": draws, "chains": chains, "cores": cores, "target_accept": target_accept,
              "drivers": n_drivers, "windows": int(len(arrays["speed"])),
              "max_rhat": float(np.nanmax(diagnostic["r_hat"])), "min_ess_bulk": float(np.nanmin(diagnostic["ess_bulk"])),
              "divergences": int(np.asarray(trace.sample_stats["diverging"]).sum()),
              "ell_s_mean": float(np.mean(ell_draws)), "source_ring_iid_noise_mps2": .1}
    (output / "author_reference_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return posterior_path


def fit_author_b_reference(dataset_path: str | Path, output_dir: str | Path, *, tune: int = 5000, draws: int = 2500,
                           chains: int = 2, cores: int = 1, target_accept: float = .90, seed: int = 37) -> Path:
    """Reproduce the public B-IDM notebook, including its ``step=2`` choice.

    The notebook subsamples the concatenated 5 Hz arrays by two but retains
    ``Config.dt=.2`` in its likelihood.  That is a source-code characteristic,
    not a corrected 2.5 Hz physical model, and is reported as such.
    """
    import pymc as pm
    import pytensor.tensor as pt
    _, pairs = load_author_reference(dataset_path)
    values = {name: [] for name in ("speed", "gap", "closing", "next_speed", "driver")}
    for driver, pair in enumerate(pairs):
        values["speed"].append(pair["follower_v"]); values["gap"].append(pair["gap"])
        values["closing"].append(pair["follower_v"] - pair["leader_v"]); values["next_speed"].append(pair["next_speed"])
        values["driver"].append(np.full(len(pair["gap"]), driver, dtype=np.int64))
    arrays = {name: np.concatenate(value)[::2] for name, value in values.items()}
    n_drivers = len(pairs)
    coords = {"veh_id": np.arange(n_drivers), "obs_id": np.arange(len(arrays["speed"])), "parameter": np.arange(5)}
    with pm.Model(coords=coords):
        driver = pm.Data("id_idx", arrays["driver"].astype(np.int64), dims="obs_id")
        speed, gap, closing = (pm.Data(name, arrays[name].astype(float), dims="obs_id") for name in ("speed", "gap", "closing"))
        chol, _, _ = pm.LKJCholeskyCov("chol", n=5, eta=2., sd_dist=pm.Exponential.dist(100., shape=5))
        values_raw = pm.Normal("vals_raw", 0., 1., dims=("veh_id", "parameter"))
        log_mu = pm.Normal("log_mu", 0., 1., dims="parameter")
        theta = pm.Deterministic("theta", pt.exp(log_mu + values_raw @ chol.T) * REFERENCE, dims=("veh_id", "parameter"))
        sigma = pm.Exponential("s_a", lam=1.e4)
        selected = theta[driver]
        desired_gap = selected[:, 1] + speed * selected[:, 2] + speed * closing / (2. * pt.sqrt(selected[:, 3] * selected[:, 4]))
        acceleration = selected[:, 3] * (1. - (speed / selected[:, 0]) ** 4 - (desired_gap / gap) ** 2)
        pm.Normal("obs", mu=speed + DT * acceleration, sigma=sigma * DT, observed=arrays["next_speed"], dims="obs_id")
        trace = pm.sample(draws=draws, tune=tune, chains=chains, cores=cores, target_accept=target_accept,
                          random_seed=seed, init="jitter+adapt_diag", return_inferencedata=True, progressbar=True)
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    trace_path = output / "author_reference_b_trace.nc"; trace.to_netcdf(trace_path)
    theta_draws = np.asarray(trace.posterior["theta"]).reshape(-1, n_drivers, 5)
    sigma_draws = np.asarray(trace.posterior["s_a"]).reshape(-1)
    means, covariances = [], []
    for sample, sigma_value in zip(theta_draws, sigma_draws):
        logs = np.log(sample); means.append(np.r_[np.mean(logs, axis=0), np.log([sigma_value, 1.])])
        covariance = np.zeros((7, 7)); covariance[:5, :5] = np.cov(logs, rowvar=False) + np.eye(5) * .0025; covariance[5, 5] = covariance[6, 6] = .0025
        covariances.append(covariance)
    posterior_path = output / "author_reference_b_idm_posterior.npz"
    np.savez_compressed(posterior_path, parameter_names=np.asarray(("v0_mps", "s0_m", "T_s", "alpha_mps2", "beta_mps2", "sigma_gp_mps2", "ell_s")),
                        population_log_mean=np.asarray(means), population_log_cov=np.asarray(covariances),
                        driver_map=np.column_stack((np.mean(theta_draws, axis=0), np.full(n_drivers, np.mean(sigma_draws)), np.ones(n_drivers))),
                        driver_parameter_draws=theta_draws, driver_sigma_draws=sigma_draws, driver_ell_draws=np.ones_like(sigma_draws),
                        driver_train_indices=np.arange(n_drivers), model=np.asarray("b_idm"), fit_backend=np.asarray("pymc_author_b_notebook_step2_dt02"),
                        dt_s=np.asarray(DT), calibration_window_s=np.asarray(0.))
    import arviz as az
    diagnostic = az.summary(trace, var_names=["log_mu", "s_a"], kind="diagnostics")
    report = {"reference": "Zhang & Sun public Bayesian_IDM_hierarchical.ipynb", "author_dataset_protocol": "fixed Config.py 20 pairs",
              "source_step": 2, "source_input_fps": 5, "effective_input_fps": 2.5, "source_likelihood_dt_s": DT,
              "warning": "faithful donor-notebook reproduction: effective input spacing and likelihood dt differ",
              "tune": tune, "draws_per_chain": draws, "chains": chains, "drivers": n_drivers, "observations": int(len(arrays["speed"])),
              "max_rhat": float(np.nanmax(diagnostic["r_hat"])), "min_ess_bulk": float(np.nanmin(diagnostic["ess_bulk"])),
              "divergences": int(np.asarray(trace.sample_stats["diverging"]).sum()), "sigma_a_mean": float(np.mean(sigma_draws))}
    (output / "author_reference_b_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return posterior_path
