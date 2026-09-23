"""Small, explicit Bayesian Stage-B fit for the documented highD adaptation.

The paper's likelihood scale and complete HDP-HSMM implementation are not
available.  This keeps that uncertainty visible: labels are supplied offline,
the likelihood is on 5 Hz held-action acceleration, and every retained draw is
joint across the three regimes of one style.
"""
from __future__ import annotations

from pathlib import Path
import json
import numpy as np


REFERENCE = np.asarray((33.3, 2.0, 1.6, 1.5, 1.67), dtype=float)


def stratified_subsample(observation: np.ndarray, action: np.ndarray, labels: np.ndarray, *,
                         maximum_per_regime: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Take a reproducible, label-stratified subset; return source indices too."""
    rng = np.random.default_rng(seed)
    selected = []
    for regime in range(3):
        candidates = np.flatnonzero(labels == regime)
        if not len(candidates):
            raise ValueError(f"offline regime {regime} has no observations")
        selected.append(rng.choice(candidates, size=min(len(candidates), maximum_per_regime), replace=False))
    index = np.concatenate(selected)
    return observation[index], action[index], labels[index], index


def fit_style_nuts(observation: np.ndarray, action: np.ndarray, labels: np.ndarray, output_dir: str | Path, *,
                   style_id: int, maximum_per_regime: int = 400, tune: int = 300, draws: int = 300,
                   chains: int = 2, target_accept: float = .95, seed: int = 20260915) -> Path:
    """Fit a three-regime partially pooled IID B-IDM and retain joint draws."""
    import arviz as az
    import pymc as pm
    import pytensor.tensor as pt

    obs, observed_action, regime, source_index = stratified_subsample(
        np.asarray(observation, float), np.asarray(action, float), np.asarray(labels, int),
        maximum_per_regime=maximum_per_regime, seed=seed + style_id,
    )
    coords = {"regime": np.arange(3), "parameter": np.arange(5), "observation": np.arange(len(obs))}
    with pm.Model(coords=coords):
        regime_idx = pm.Data("regime_idx", regime, dims="observation")
        gap = pm.Data("gap_m", obs[:, 0], dims="observation")
        speed = pm.Data("speed_mps", obs[:, 1], dims="observation")
        closing = pm.Data("closing_mps", obs[:, 2], dims="observation")
        # A dimension-wise hierarchical prior is deliberately used instead of
        # injecting the paper's unspecified-normalization priors into SI data.
        log_population = pm.Normal("log_population", mu=np.log(REFERENCE), sigma=.75, dims="parameter")
        regime_log_sd = pm.HalfNormal("regime_log_sd", sigma=.65, dims="parameter")
        regime_z = pm.Normal("regime_z", 0., 1., dims=("regime", "parameter"))
        theta = pm.Deterministic("theta", pt.exp(log_population + regime_z * regime_log_sd), dims=("regime", "parameter"))
        sigma = pm.HalfNormal("sigma_mps2", sigma=2., dims="regime")
        selected = theta[regime_idx]
        desired_gap = selected[:, 1] + speed * selected[:, 2] + speed * closing / (2. * pt.sqrt(selected[:, 3] * selected[:, 4]))
        acceleration = selected[:, 3] * (1. - (speed / selected[:, 0]) ** 4 - (desired_gap / pt.maximum(gap, .1)) ** 2)
        pm.Normal("action", mu=acceleration, sigma=sigma[regime_idx], observed=observed_action, dims="observation")
        trace = pm.sample(draws=draws, tune=tune, chains=chains, cores=1, target_accept=target_accept,
                          random_seed=seed + style_id, init="jitter+adapt_diag", progressbar=True,
                          return_inferencedata=True)
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    trace.to_netcdf(output / f"style_{style_id}_nuts_trace.nc")
    theta_draws = np.asarray(trace.posterior["theta"]).reshape(-1, 3, 5)
    sigma_draws = np.asarray(trace.posterior["sigma_mps2"]).reshape(-1, 3)
    np.savez_compressed(output / f"style_{style_id}_nuts_posterior.npz", theta_draws=theta_draws,
                        sigma_draws=sigma_draws, source_index=source_index, source_regime=regime,
                        decision_dt_s=np.asarray(.2), reference_theta=REFERENCE)
    # ArviZ/xarray treats a tuple as one composite key; pass a list instead.
    diagnostic = az.summary(trace, var_names=["theta", "sigma_mps2"], kind="diagnostics")
    report = {
        "style_id": style_id,
        "model": "hierarchical_regime_b_idm",
        "calibration": "small_budget_NUTS_documented_adaptation_not_paper_full_NUTS",
        "label_source": "Stage-A offline labels; retrospective/oracle diagnostic only",
        "likelihood": "action at 5Hz held decision grid: Normal(IDM acceleration, regime sigma)",
        "prior": "log population Normal(log(reference_SI), .75); independent regime offsets with HalfNormal(.65) scale",
        "observations_per_regime": [int(np.sum(regime == state)) for state in range(3)],
        "source_observations_per_regime": [int(np.sum(labels == state)) for state in range(3)],
        "tune": tune, "draws_per_chain": draws, "chains": chains, "target_accept": target_accept,
        "max_rhat": float(np.nanmax(diagnostic["r_hat"])),
        "min_ess_bulk": float(np.nanmin(diagnostic["ess_bulk"])),
        "divergences": int(np.asarray(trace.sample_stats["diverging"]).sum()),
        "posterior_theta_median": np.median(theta_draws, axis=0).tolist(),
        "posterior_sigma_mps2_median": np.median(sigma_draws, axis=0).tolist(),
    }
    path = output / f"style_{style_id}_nuts_report.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


def fit_style_pooled_nuts(
    observation: np.ndarray,
    action: np.ndarray,
    labels: np.ndarray,
    output_dir: str | Path,
    *,
    style_id: int,
    maximum_per_regime: int = 400,
    tune: int = 300,
    draws: int = 300,
    chains: int = 2,
    target_accept: float = 0.95,
    seed: int = 20260915,
) -> Path:
    """Fit the actual pooled B-IDM control on the identical style observations."""
    import arviz as az
    import pymc as pm
    import pytensor.tensor as pt

    obs, observed_action, _, source_index = stratified_subsample(
        np.asarray(observation, float),
        np.asarray(action, float),
        np.asarray(labels, int),
        maximum_per_regime=maximum_per_regime,
        seed=seed + style_id,
    )
    coords = {"parameter": np.arange(5), "observation": np.arange(len(obs))}
    with pm.Model(coords=coords):
        gap = pm.Data("gap_m", obs[:, 0], dims="observation")
        speed = pm.Data("speed_mps", obs[:, 1], dims="observation")
        closing = pm.Data("closing_mps", obs[:, 2], dims="observation")
        log_theta = pm.Normal(
            "log_theta", mu=np.log(REFERENCE), sigma=0.75, dims="parameter"
        )
        theta = pm.Deterministic("theta", pt.exp(log_theta), dims="parameter")
        sigma = pm.HalfNormal("sigma_mps2", sigma=2.0)
        desired_gap = (
            theta[1]
            + speed * theta[2]
            + speed * closing / (2.0 * pt.sqrt(theta[3] * theta[4]))
        )
        acceleration = theta[3] * (
            1.0 - (speed / theta[0]) ** 4 - (desired_gap / pt.maximum(gap, 0.1)) ** 2
        )
        pm.Normal(
            "action",
            mu=acceleration,
            sigma=sigma,
            observed=observed_action,
            dims="observation",
        )
        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=1,
            target_accept=target_accept,
            random_seed=seed + 100 + style_id,
            init="jitter+adapt_diag",
            progressbar=True,
            return_inferencedata=True,
        )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    trace.to_netcdf(output / f"style_{style_id}_pooled_nuts_trace.nc")
    theta_draws = np.asarray(trace.posterior["theta"]).reshape(-1, 5)
    sigma_draws = np.asarray(trace.posterior["sigma_mps2"]).reshape(-1)
    np.savez_compressed(
        output / f"style_{style_id}_pooled_nuts_posterior.npz",
        theta_draws=theta_draws,
        sigma_draws=sigma_draws,
        source_index=source_index,
        decision_dt_s=np.asarray(0.2),
        reference_theta=REFERENCE,
    )
    diagnostic = az.summary(
        trace, var_names=["theta", "sigma_mps2"], kind="diagnostics"
    )
    report = {
        "style_id": style_id,
        "model": "pooled_b_idm",
        "calibration": "small_budget_NUTS_documented_adaptation_not_paper_full_NUTS",
        "likelihood": "same stratified 5 Hz action observations as hierarchical control",
        "observations": int(len(obs)),
        "tune": tune,
        "draws_per_chain": draws,
        "chains": chains,
        "target_accept": target_accept,
        "max_rhat": float(np.nanmax(diagnostic["r_hat"])),
        "min_ess_bulk": float(np.nanmin(diagnostic["ess_bulk"])),
        "divergences": int(np.asarray(trace.sample_stats["diverging"]).sum()),
        "posterior_theta_median": np.median(theta_draws, axis=0).tolist(),
        "posterior_sigma_mps2_median": float(np.median(sigma_draws)),
    }
    path = output / f"style_{style_id}_pooled_nuts_report.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path
