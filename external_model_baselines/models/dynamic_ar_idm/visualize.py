"""Paper-oriented diagnostic figures: covariance, posterior rollout, and Fig. 9 curve."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from scipy.linalg import solve_discrete_lyapunov
import matplotlib.pyplot as plt

from .data import load_pairs
from .fit import residual


def ar_covariance(rho: np.ndarray, sigma: float, length: int = 50) -> np.ndarray:
    """Stationary AR covariance from the companion state-space model."""
    rho = np.asarray(rho, float); p = len(rho)
    if not p:
        return np.eye(length) * sigma ** 2
    companion = np.zeros((p, p)); companion[0] = rho
    if p > 1:
        companion[1:, :-1] = np.eye(p - 1)
    q = np.zeros((p, p)); q[0, 0] = sigma ** 2
    try:
        stationary = solve_discrete_lyapunov(companion, q)
    except Exception:
        stationary = np.eye(p) * sigma ** 2
    h = np.empty(length)
    power = np.eye(p)
    for lag in range(length):
        h[lag] = (power @ stationary)[0, 0]
        power = power @ companion
    return np.fromfunction(lambda i, j: h[np.abs(i - j).astype(int)], (length, length))


def make_figures(dataset_path: str | Path, posterior_path: str | Path, metrics_path: str | Path, output_dir: str | Path) -> list[Path]:
    """Generate counterparts to paper Figs. 5, 7, and 9 from retained artifacts."""
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    pairs = load_pairs(dataset_path)
    with np.load(posterior_path, allow_pickle=False) as raw:
        posterior = {key: raw[key] for key in raw.files}
    theta, rho, sigma = posterior["theta_map"], posterior["rho_map"], float(posterior["sigma_map"])
    all_residuals = [residual(pair, theta[index]) for index, pair in enumerate(pairs)]
    size = min(50, min(map(len, all_residuals)))
    empirical = np.mean([np.outer(value[:size] - np.mean(value), value[:size] - np.mean(value)) for value in all_residuals], axis=0)
    covariance = ar_covariance(rho, sigma, size)
    se = covariance[0, 0] * np.exp(-.5 * ((np.arange(size)[:, None] - np.arange(size)[None, :]) * .2 / 1.44) ** 2)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    for axis, value, title in zip(axes, (empirical, se, covariance), ("empirical residual covariance", "SE kernel (ell=1.44 s)", "Dynamic IDM AR covariance")):
        vmax = max(float(np.max(np.abs(value))), 1.e-6)
        image = axis.imshow(value, cmap="coolwarm", vmin=-vmax, vmax=vmax, origin="lower")
        axis.set(title=title, xlabel="5 Hz lag", ylabel="5 Hz lag")
        fig.colorbar(image, ax=axis, shrink=.78, label="covariance (m/s²)²")
    figure5 = output_dir / "figure5_residual_covariance.png"; fig.savefig(figure5, dpi=180); plt.close(fig)

    rollout_file = Path(metrics_path).with_name("representative_rollout.npz")
    figure7 = output_dir / "figure7_representative_posterior_rollout.png"
    if rollout_file.exists():
        with np.load(rollout_file, allow_pickle=False) as value:
            rollout = {key: value[key] for key in value.files}
        t = np.arange(1, len(rollout["true_velocity"]) + 1) * .04
        gap = rollout["leader_x"] - rollout["true_position"] - float(rollout["length_sum"])
        sample_gap = rollout["leader_x"][None, :] - rollout["position"] - float(rollout["length_sum"])
        fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True, constrained_layout=True)
        specs = ((rollout["acceleration"][:, 4::5], rollout["true_action"], "acceleration (m/s²)", t[4::5]),
                 (rollout["velocity"], rollout["true_velocity"], "speed (m/s)", t),
                 (sample_gap, gap, "gap (m)", t))
        for axis, (sample, truth, label, tx) in zip(axes, specs):
            axis.fill_between(tx, np.quantile(sample, .05, axis=0), np.quantile(sample, .95, axis=0), color="tab:red", alpha=.25, label="90% posterior")
            axis.plot(tx, np.mean(sample, axis=0), color="tab:red", lw=1.3, label="posterior mean")
            axis.plot(tx, truth, color="black", lw=1.2, label="highD")
            axis.set_ylabel(label); axis.grid(alpha=.2)
        axes[0].legend(loc="best", ncol=3, fontsize=8); axes[-1].set_xlabel("forecast time (s)")
        fig.savefig(figure7, dpi=180); plt.close(fig)

    report = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
    horizon = np.asarray(sorted(map(int, report["horizons"])))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)
    for name, label in (("a_rmse", "acceleration"), ("v_rmse", "speed"), ("s_rmse", "gap")):
        axes[0].plot(horizon, [report["horizons"][str(h)][name]["mean"] for h in horizon], marker="o", label=label)
    for name, label in (("a_crps", "acceleration"), ("v_crps", "speed"), ("s_crps", "gap")):
        axes[1].plot(horizon, [report["horizons"][str(h)][name]["mean"] for h in horizon], marker="o", label=label)
    axes[0].set(title="RMSE versus rollout horizon", xlabel="horizon (s)", ylabel="RMSE")
    axes[1].set(title="CRPS versus rollout horizon", xlabel="horizon (s)", ylabel="CRPS")
    for axis in axes:
        axis.grid(alpha=.2); axis.legend(fontsize=8)
    figure9 = output_dir / "figure9_horizon_metrics.png"; fig.savefig(figure9, dpi=180); plt.close(fig)
    return [figure5, figure7, figure9]
