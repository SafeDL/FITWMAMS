"""Chain-separated posterior-predictive audit for non-mixing NUTS fits."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .data import load_pairs
from .evaluate import evaluate_paper_cohort


METRICS = ("a_rmse", "v_rmse", "s_rmse", "a_crps", "v_crps", "s_crps")


def _export_chain(nc_path: str | Path, dataset_path: str | Path, output_path: Path, chain: int,
                  max_draws: int) -> Path:
    """Export one chain without ever pooling it with another chain."""
    import arviz as az

    idata = az.from_netcdf(nc_path)
    posterior = idata.posterior
    chains = int(posterior.sizes["chain"])
    if chain < 0 or chain >= chains:
        raise IndexError(f"chain {chain} outside [0, {chains})")
    available = int(posterior.sizes["draw"])
    indices = np.linspace(0, available - 1, min(max_draws, available), dtype=int)
    theta = np.asarray(posterior["theta"].isel(chain=chain, draw=indices), float)
    sigma = np.asarray(posterior["sigma_eta"].isel(chain=chain, draw=indices), float)
    if "rho" in posterior:
        rho = np.asarray(posterior["rho"].isel(chain=chain, draw=indices), float)
    else:
        rho = np.empty((len(indices), 0))
    pairs = load_pairs(dataset_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        theta_draws=theta,
        rho_draws=rho,
        sigma_draws=sigma,
        theta_map=np.mean(theta, axis=0),
        rho_map=np.mean(rho, axis=0) if rho.shape[1] else np.empty(0),
        sigma_map=np.asarray(np.mean(sigma)),
        pair_no=np.asarray([pair["pair_no"] for pair in pairs]),
        parameter_names=np.asarray(("v0_mps", "s0_m", "T_s", "alpha_mps2", "beta_mps2")),
        ar_order=np.asarray(rho.shape[1]),
        fit_backend=np.asarray("pymc_nuts_single_chain_failed_global_convergence_audit"),
        source_chain=np.asarray(chain),
        source_draws=np.asarray(available),
        exported_draws=np.asarray(len(indices)),
        decision_dt_s=np.asarray(.2),
        plant_dt_s=np.asarray(.04),
    )
    return output_path


def audit_chainwise_predictions(dataset_path: str | Path, ar0_nc: str | Path, ar5_nc: str | Path,
                                output_path: str | Path, *, futures: int = 64,
                                max_horizon_s: int = 5, chain_draws: int = 512,
                                seed: int = 20260916) -> Path:
    """Compare AR(0)/AR(5) predictions without pooling non-converged chains.

    The output is a sensitivity diagnostic, not a substitute for a converged
    posterior.  The strongest available qualitative result is reported as the
    number of all 4x4 cross-model chain comparisons favouring AR(5).
    """
    import arviz as az

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ar0_idata, ar5_idata = az.from_netcdf(ar0_nc), az.from_netcdf(ar5_nc)
    n0 = int(ar0_idata.posterior.sizes["chain"])
    n5 = int(ar5_idata.posterior.sizes["chain"])
    rows: dict[str, list[dict]] = {"ar0": [], "ar5": []}
    for label, nc_path, count in (("ar0", ar0_nc, n0), ("ar5", ar5_nc, n5)):
        for chain in range(count):
            chain_dir = output_path.parent / "chainwise" / label / f"chain_{chain}"
            posterior_path = _export_chain(nc_path, dataset_path, chain_dir / "posterior.npz", chain, chain_draws)
            metrics_path = evaluate_paper_cohort(
                dataset_path, posterior_path, chain_dir / "metrics.json", futures=futures,
                max_horizon_s=max_horizon_s, seed=seed + chain, rho_mode="posterior",
            )
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            horizon = metrics["horizons"][str(max_horizon_s)]
            rows[label].append({"chain": chain, **{name: horizon[name]["mean"] for name in METRICS}})

    comparison = {}
    all_better = True
    for name in METRICS:
        ar0_values = np.asarray([row[name] for row in rows["ar0"]])
        ar5_values = np.asarray([row[name] for row in rows["ar5"]])
        pairwise = ar5_values[:, None] < ar0_values[None, :]
        comparison[name] = {
            "ar0_chain_range": [float(np.min(ar0_values)), float(np.max(ar0_values))],
            "ar5_chain_range": [float(np.min(ar5_values)), float(np.max(ar5_values))],
            "all_pairwise_better_count": int(np.sum(pairwise)),
            "all_pairwise_comparisons": int(pairwise.size),
            "ar5_mean_relative_change_pct": float(100. * (np.mean(ar5_values) / np.mean(ar0_values) - 1.)),
        }
        all_better &= bool(np.all(pairwise))
    report = {
        "audit": "chain-separated NUTS posterior-predictive sensitivity",
        "dataset": str(dataset_path),
        "ar0_source": str(ar0_nc),
        "ar5_source": str(ar5_nc),
        "chains": {"ar0": n0, "ar5": n5},
        "draws_exported_per_chain": chain_draws,
        "futures_per_origin": futures,
        "horizon_s": max_horizon_s,
        "pooled_chains": False,
        "global_convergence": "failed for both source fits; these results are sensitivity evidence only",
        "chain_metrics": rows,
        "comparison": comparison,
        "all_six_metrics_favour_ar5_for_every_cross_chain_pair": all_better,
    }
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output_path
