"""Joint dynamic-regression calibration and posterior approximation."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

from .data import load_pairs
from .model import ar_spectral_radius, idm

REFERENCE_THETA = np.asarray([33.3, 2., 1.6, 1.5, 1.67])
LOWER = np.asarray([8., .2, .3, .1, .2])
UPPER = np.asarray([70., 12., 4., 5., 10.])


def decision_arrays(pair: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Paper's 5 Hz inputs and completed 0.2 s actions from native highD."""
    ix = np.asarray(pair["decision"], int)
    v, lv, x, lx = (np.asarray(pair[key], float) for key in ("follower_v", "leader_v", "follower_x", "leader_x"))
    return {"speed": v[ix], "closing": v[ix] - lv[ix], "gap": lx[ix] - x[ix] - float(pair["length_sum"]),
            "action": (v[ix + 5] - v[ix]) / .2}


def residual(pair: dict[str, np.ndarray], theta: np.ndarray) -> np.ndarray:
    values = decision_arrays(pair)
    return values["action"] - idm(values["gap"], values["speed"], values["closing"], theta)


def innovations(error: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """Eq. (10) rearranged to white innovations; no sequence boundaries cross."""
    p = len(rho)
    if not p:
        return np.asarray(error, float)
    if len(error) <= p:
        return np.empty(0)
    history = np.stack([error[p - lag - 1:len(error) - lag - 1] for lag in range(p)], axis=1)
    return error[p:] - history @ rho


def _fit_theta(pair: dict[str, np.ndarray], rho: np.ndarray, prior_mean: np.ndarray, prior_precision: np.ndarray,
               initial: np.ndarray) -> np.ndarray:
    def objective(log_theta: np.ndarray) -> float:
        error = residual(pair, np.exp(log_theta))
        eta = innovations(error, rho)
        if len(eta) < 5 or not np.all(np.isfinite(eta)):
            return 1.e30
        delta = log_theta - prior_mean
        # Gaussian innovation likelihood with variance profiled out.  The
        # prior is the MAP analogue of the paper's hierarchical log-normal.
        return .5 * len(eta) * np.log(np.mean(eta ** 2) + 1.e-10) + .5 * float(delta @ prior_precision @ delta)
    result = minimize(objective, np.log(np.clip(initial, LOWER, UPPER)), method="L-BFGS-B",
                      bounds=list(zip(np.log(LOWER), np.log(UPPER))), options={"maxiter": 180, "ftol": 1.e-10})
    if not np.isfinite(result.fun):
        raise RuntimeError("dynamic IDM MAP optimisation failed")
    return np.exp(result.x)


def _fit_rho(errors: list[np.ndarray], order: int, ridge: float = 25.) -> tuple[np.ndarray, np.ndarray]:
    if not order:
        return np.empty(0), np.empty((0, 0))
    X, y = [], []
    for error in errors:
        if len(error) <= order:
            continue
        X.append(np.stack([error[order - lag - 1:len(error) - lag - 1] for lag in range(order)], axis=1))
        y.append(error[order:])
    design, target = np.concatenate(X), np.concatenate(y)
    gram = design.T @ design + ridge * np.eye(order)  # N(0, .2^2) shrinkage, scaled conservatively.
    rho = np.linalg.solve(gram, design.T @ target)
    eta = target - design @ rho
    sigma2 = max(float(np.mean(eta ** 2)), 1.e-8)
    return rho, sigma2 * np.linalg.inv(gram)


def _nearest_pd(matrix: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh((matrix + matrix.T) / 2.)
    return (vectors * np.maximum(values, .0025)) @ vectors.T


def fit_dynamic_map(dataset_path: str | Path, output_path: str | Path, *, ar_order: int = 5,
                    posterior_draws: int = 1000, iterations: int = 4, seed: int = 37,
                    training_recordings: list[int] | None = None) -> Path:
    """Fast, reproducible MAP/Laplace implementation of the paper model.

    This is a deployment-friendly inference backend.  ``fit_pymc`` below is
    supplied for the exact NUTS likelihood; both use exactly the same residual
    equation and paper data protocol.
    """
    pairs = load_pairs(dataset_path)
    if training_recordings is not None:
        selected = {int(value) for value in training_recordings}
        pairs = [pair for pair in pairs if int(pair["recording_id"]) in selected]
    if len(pairs) < 3:
        raise RuntimeError("dynamic fit needs at least three training trajectories")
    log_reference = np.log(REFERENCE_THETA)
    prior_precision = np.diag(1. / np.asarray([.1] * 5))
    theta = np.tile(REFERENCE_THETA, (len(pairs), 1))
    rho = np.zeros(ar_order)
    rho_cov = np.eye(ar_order) * .01 if ar_order else np.empty((0, 0))
    for _ in range(iterations):
        theta = np.asarray([_fit_theta(pair, rho, log_reference, prior_precision, value) for pair, value in zip(pairs, theta)])
        logs = np.log(theta)
        population = np.mean(logs, axis=0)
        cov = _nearest_pd(np.cov(logs, rowvar=False) + np.eye(5) * .02)
        # Weak partial pooling keeps a poorly identified driver parameter from
        # reaching its numerical bounds.
        prior_precision = np.linalg.inv(cov + np.eye(5) * .10)
        theta = np.asarray([_fit_theta(pair, rho, population, prior_precision, value) for pair, value in zip(pairs, theta)])
        rho, rho_cov = _fit_rho([residual(pair, value) for pair, value in zip(pairs, theta)], ar_order)
    errors = [residual(pair, value) for pair, value in zip(pairs, theta)]
    eta = np.concatenate([innovations(value, rho) for value in errors])
    sigma = float(np.sqrt(np.mean(eta ** 2)))
    logs = np.log(theta); population = np.mean(logs, axis=0)
    rng = np.random.default_rng(seed)
    # A group bootstrap + small local Laplace component preserves posterior
    # parameter covariance without presenting deterministic MAP as NUTS draws.
    theta_draws = np.empty((posterior_draws, len(pairs), 5))
    rho_draws = np.empty((posterior_draws, ar_order))
    for draw in range(posterior_draws):
        sampled = logs[rng.integers(0, len(logs), len(logs))]
        # Each paper trajectory is evaluated with draws from *that driver's*
        # posterior, not by reassigning a population driver at every future.
        # The bootstrap moves the shared hierarchical centre modestly while a
        # local Laplace perturbation retains individual uncertainty.
        group_shift = .15 * (np.mean(sampled, axis=0) - population)
        local_cov = _nearest_pd(np.cov(sampled, rowvar=False) * .02 + np.eye(5) * .0004)
        theta_draws[draw] = np.exp(logs + group_shift + rng.multivariate_normal(np.zeros(5), local_cov, size=len(pairs)))
        if ar_order:
            candidate = rng.multivariate_normal(rho, _nearest_pd(rho_cov + np.eye(ar_order) * 1.e-6))
            # Do not silently discard a non-stationary draw: retain it and
            # report the rate below, as requested by the design protocol.
            rho_draws[draw] = candidate
    sigma_draws = np.exp(rng.normal(np.log(max(sigma, 1.e-5)), .06, size=posterior_draws))
    output_path = Path(output_path); output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, theta_draws=theta_draws, rho_draws=rho_draws, sigma_draws=sigma_draws,
                        theta_map=theta, rho_map=rho, sigma_map=np.asarray(sigma), pair_no=np.asarray([p["pair_no"] for p in pairs]),
                        parameter_names=np.asarray(("v0_mps", "s0_m", "T_s", "alpha_mps2", "beta_mps2")), ar_order=np.asarray(ar_order),
                        fit_backend=np.asarray("joint_dynamic_regression_empirical_bayes_map_laplace"), decision_dt_s=np.asarray(.2), plant_dt_s=np.asarray(.04))
    radii = np.asarray([ar_spectral_radius(value) for value in rho_draws]) if ar_order else np.zeros(posterior_draws)
    report = {"model": "Dynamic IDM / AR residual", "fit_backend": "joint_dynamic_regression_empirical_bayes_map_laplace",
              "not_claimed": "This artifact is not a replacement for the optional NUTS posterior; it is a calibrated Laplace approximation.",
              "paper_equations": {"state": "a_t = IDM_t + e_t", "residual": "e_t = sum rho_k e_{t-k} + eta_t", "likelihood": "Gaussian white innovations after joint IDM/AR fitting"},
              "pairs": len(pairs), "training_recordings": ("all" if training_recordings is None else sorted(selected)),
              "ar_order": ar_order, "native_fps": 25, "decision_fps": 5, "iterations": iterations,
              "theta_map_mean": np.mean(theta, axis=0).tolist(), "rho_map": rho.tolist(), "sigma_eta_map_mps2": sigma,
              "posterior_stationarity": {"draws": posterior_draws, "nonstationary_count": int(np.sum(radii >= 1.)), "nonstationary_fraction": float(np.mean(radii >= 1.)), "spectral_radius_q": np.quantile(radii, [.05, .5, .95]).tolist()},
              "paper_table_1_ar5": {"theta": [27.099, 2.843, 1.235, .813, 3.422], "rho": [.874, .580, -.105, -.315, -.071], "sigma_eta": .016}}
    output_path.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    stationarity_path = output_path.with_name(f"{output_path.stem}_stationarity.json")
    stationarity_path.write_text(json.dumps({"model": "dynamic_ar_idm", "ar_order": ar_order,
        "rho_map": rho.tolist(), "rho_map_spectral_radius": ar_spectral_radius(rho), **report["posterior_stationarity"],
        "policy": "paper_faithful: all posterior draws retained and non-stationary rate disclosed; do not silently discard samples"}, indent=2), encoding="utf-8")
    return output_path


def _scalar_diagnostic(summary, column: str, reduction, default: float = float("nan")) -> float:
    """JSON-safe extraction from an ArviZ diagnostic summary."""
    if column not in summary or not len(summary):
        return default
    values = np.asarray(summary[column], float)
    values = values[np.isfinite(values)]
    return float(reduction(values)) if len(values) else default


def fit_pymc(dataset_path: str | Path, output_path: str | Path, *, ar_order: int = 5, tune: int = 3000,
             draws: int = 1000, chains: int = 4, cores: int = 1, seed: int = 37,
             target_accept: float = .90, sampler_init: str = "jitter+adapt_diag_grad") -> Path:
    """Run the paper's hierarchical Dynamic-IDM speed likelihood with NUTS.

    This uses the public highD author's model conventions: parameters are
    log-normal *multipliers* of the recommended IDM vector, innovation scale
    has the ``Exponential(2e6)`` prior, and speed observation noise is fixed
    to 0.001667 m/s.  The indexing below is Eq. (19) written per trajectory,
    which additionally prevents AR history crossing driver boundaries.
    """
    import pymc as pm
    import pytensor.tensor as pt
    pairs = load_pairs(dataset_path)
    values = [decision_arrays(pair) for pair in pairs]
    source_base = np.asarray([33., 2., 1.6, 1.5, 1.67])
    with pm.Model(coords={"driver": np.arange(len(pairs)), "parameter": np.arange(5), "lag": np.arange(ar_order)}):
        chol, _, _ = pm.LKJCholeskyCov("chol", n=5, eta=2., sd_dist=pm.Exponential.dist(100., shape=5))
        log_mu = pm.Normal("log_mu", mu=0., sigma=1., dims="parameter")
        raw = pm.Normal("theta_raw", 0., 1., dims=("driver", "parameter"))
        theta = pm.Deterministic("theta", source_base * pt.exp(log_mu + raw @ chol.T), dims=("driver", "parameter"))
        sigma_eta = pm.Exponential("sigma_eta", lam=2.e6)
        rho = pm.Normal("rho", 0., .2, dims="lag") if ar_order else None
        sigma_v = .001667  # public author's fixed highD speed observation s.d.
        for driver, value in enumerate(values):
            speed, gap, closing, observed = (np.asarray(value[key], float) for key in ("speed", "gap", "closing", "action"))
            selected = theta[driver]
            desired = selected[1] + speed * selected[2] + speed * closing / (2. * pt.sqrt(selected[3] * selected[4]))
            mean_a = selected[3] * (1. - (speed / selected[0]) ** 4 - (desired / gap) ** 2)
            # Eq. 12 / author notebook: each historic term is v_j-v_{j-1}-IDM_{j-1}dt.
            mean_v = speed + .2 * mean_a
            if ar_order:
                for lag in range(ar_order):
                    correction = speed[ar_order - lag:len(speed) - lag] - speed[ar_order - lag - 1:len(speed) - lag - 1] - .2 * mean_a[ar_order - lag - 1:len(speed) - lag - 1]
                    mean_v = pt.set_subtensor(mean_v[ar_order:], mean_v[ar_order:] + rho[lag] * correction)
            pm.Normal(f"speed_obs_{driver}", mu=mean_v[ar_order:], sigma=pt.sqrt((sigma_eta * .2) ** 2 + sigma_v ** 2), observed=speed[ar_order:] + .2 * observed[ar_order:])
        # The Exponential(2e6) prior has a mode at zero whereas its highD
        # posterior is approximately 1e-2 m/s².  Start in that plausible
        # region to avoid an artificial near-zero step size during adaptation;
        # this changes only initialisation, never the target posterior.
        initvals = {"sigma_eta": .016}
        if ar_order:
            initvals["rho"] = np.zeros(ar_order)
        trace = pm.sample(draws=draws, tune=tune, chains=chains, cores=cores, random_seed=seed,
                          init=sampler_init, initvals=initvals, target_accept=target_accept, return_inferencedata=True)
    import arviz as az
    output_path = Path(output_path); output_path.parent.mkdir(parents=True, exist_ok=True)
    trace.to_netcdf(output_path.with_suffix(".nc"))
    flat_theta = np.asarray(trace.posterior["theta"]).reshape(-1, len(pairs), 5)
    flat_rho = (np.asarray(trace.posterior["rho"]).reshape(-1, ar_order)
                if ar_order else np.empty((flat_theta.shape[0], 0)))
    flat_sigma = np.asarray(trace.posterior["sigma_eta"]).reshape(-1)
    np.savez_compressed(output_path, theta_draws=flat_theta, rho_draws=flat_rho, sigma_draws=flat_sigma,
                        theta_map=np.mean(flat_theta, axis=0), rho_map=np.mean(flat_rho, axis=0), sigma_map=np.asarray(np.mean(flat_sigma)),
                        pair_no=np.asarray([p["pair_no"] for p in pairs]), parameter_names=np.asarray(("v0_mps", "s0_m", "T_s", "alpha_mps2", "beta_mps2")), ar_order=np.asarray(ar_order), fit_backend=np.asarray("pymc_nuts_author_speed_likelihood"), decision_dt_s=np.asarray(.2), plant_dt_s=np.asarray(.04))
    diagnostic_names = ["log_mu", "sigma_eta"] + (["rho"] if ar_order else [])
    summary = az.summary(trace, var_names=diagnostic_names, kind="diagnostics")
    divergences = int(np.asarray(trace.sample_stats["diverging"]).sum())
    # ArviZ 1.3 returns a DataTree for ``az.bfmi``.  Calculate the standard
    # BFMI directly from the energy draws so this report is version-stable.
    energy = np.asarray(trace.sample_stats["energy"], float)
    bfmi = np.mean(np.diff(energy, axis=-1) ** 2, axis=-1) / np.var(energy, axis=-1)
    tree_depth = np.asarray(trace.sample_stats["tree_depth"], int)
    max_tree_depth = int(np.max(tree_depth))
    max_rhat = _scalar_diagnostic(summary, "r_hat", np.max)
    min_ess = _scalar_diagnostic(summary, "ess_bulk", np.min)
    finite_rhat = np.isfinite(max_rhat)
    # Thresholds are declared rather than silently treating a completed run as
    # converged.  Draw count is retained because R-hat alone is insufficient.
    audit = {"criteria": {"max_rhat_lt": 1.01, "min_ess_bulk_gt": 400, "divergences_eq": 0, "min_bfmi_gt": .30},
             "passed": bool(finite_rhat and max_rhat < 1.01 and min_ess > 400 and divergences == 0 and np.nanmin(bfmi) > .30)}
    report = {"fit_backend": "pymc_nuts_paper_highd_speed_likelihood",
              "implementation": "paper Eq. (19), per-driver boundary-safe indexing; public-author prior and observation-noise conventions",
              "tune": tune, "draws_per_chain": draws, "chains": chains, "cores": cores, "target_accept": target_accept, "sampler_init": sampler_init,
              "posterior_draws_total": int(flat_theta.shape[0]), "max_rhat": max_rhat, "min_ess_bulk": min_ess,
              "min_bfmi": float(np.nanmin(bfmi)), "divergences": divergences,
              "max_tree_depth_observed": max_tree_depth, "tree_depth_limit": 10,
              "tree_depth_limit_hits": int(np.sum(tree_depth >= 10)),
              "stationarity": {"rho_draw_nonstationary_fraction": float(np.mean([ar_spectral_radius(x) >= 1. for x in flat_rho])) if ar_order else 0., "rho_mean_spectral_radius": ar_spectral_radius(np.mean(flat_rho, axis=0)) if ar_order else 0.},
              "convergence_audit": audit}
    output_path.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output_path
