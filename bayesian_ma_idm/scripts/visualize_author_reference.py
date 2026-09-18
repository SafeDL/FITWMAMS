#!/usr/bin/env python3
"""Paper-organised figures from the fixed author 20-pair reference cohort."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import arviz as az
import matplotlib.pyplot as plt
import numpy as np

from bayesian_ma_idm.src.author_reference import load_author_reference
from bayesian_ma_idm.src.reference_kernels import GPHistory, idm

DT = .2


def save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def fig5(pairs: list[dict], output: Path) -> None:
    fig, axes = plt.subplots(4, 5, figsize=(14, 8.5), sharex=False)
    for axis, pair in zip(axes.flat, pairs):
        time = np.arange(len(pair["gap"])) * DT
        axis.plot(time, pair["leader_x"], color="#d62728", lw=.75, label="leader")
        axis.plot(time, pair["follower_x"], color="#1f77b4" if pair["vehicle_class"] == "Car" else "#2ca02c", lw=.75, label="follower")
        axis.set_title(f"{pair['vehicle_class']} #{int(pair['follower_id'])} (pair {int(pair['pair_no'])})", fontsize=7)
        axis.set(xlabel="time [s]", ylabel="longitudinal x [m]")
        axis.tick_params(labelsize=6)
    axes[0, 0].legend(fontsize=6)
    fig.suptitle("Zhang--Sun fixed 20-pair highD cohort at 5 Hz (Fig. 5 protocol)")
    save(fig, output / "fig5_author_reference_cohort.png")


def fig6_7(trace_path: Path, pairs: list[dict], output: Path) -> None:
    trace = az.from_netcdf(trace_path)
    theta = np.asarray(trace.posterior["theta"]).reshape(-1, len(pairs), 5)
    labels = (r"$v_0$ [m/s]", r"$s_0$ [m]", r"$T$ [s]", r"$\alpha$ [m/s²]", r"$\beta$ [m/s²]")
    for driver_id, filename in ((273, "car"), (211, "truck")):
        index = next(i for i, pair in enumerate(pairs) if int(pair["follower_id"]) == driver_id)
        draws = theta[:, index]
        fig, axes = plt.subplots(2, 3, figsize=(10.5, 6))
        for axis, values, label in zip(axes.flat[:5], draws.T, labels):
            axis.hist(values, bins=32, density=True, color="#1f77b4" if filename == "car" else "#2ca02c", alpha=.7)
            axis.set(xlabel=label, ylabel="density")
        corr = np.corrcoef(np.log(draws), rowvar=False)
        image = axes.flat[5].imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
        axes.flat[5].set(xticks=range(5), yticks=range(5), xticklabels=[r"$v_0$", r"$s_0$", r"$T$", r"$\alpha$", r"$\beta$"],
                         yticklabels=[r"$v_0$", r"$s_0$", r"$T$", r"$\alpha$", r"$\beta$"], title="log-parameter correlation")
        fig.colorbar(image, ax=axes.flat[5], fraction=.046)
        fig.suptitle(f"Author-reference NUTS posterior: driver #{driver_id} (Fig. 6/7 analogue)")
        save(fig, output / f"fig6_7_author_reference_{filename}.png")


def simulate_exogenous(pair: dict, theta_draws: np.ndarray, sigma_draws: np.ndarray, ell_draws: np.ndarray,
                       iid_sigma: float, driver_index: int, draws: int = 64, seed: int = 1116) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Source-style 5 Hz simulation with the measured leader left exogenous."""
    rng = np.random.default_rng(seed)
    horizon = len(pair["gap"]) - 1
    states = [np.empty((draws, horizon)) for _ in range(3)]
    # Section VI-B fixes IDM parameters to posterior means for its selected
    # driver; GP hyperparameters are likewise fixed to posterior means.
    theta = np.mean(theta_draws[:, driver_index], axis=0)
    sigma, ell = float(np.mean(sigma_draws)), float(np.mean(ell_draws))
    for draw in range(draws):
        x, speed = float(pair["follower_x"][0]), float(pair["follower_v"][0])
        history = GPHistory([], [], sigma, ell, observation_noise=1.e-8, memory_seconds=5.)
        for step in range(horizon):
            leader_speed = float(pair["leader_v"][step])
            gap = max(float(pair["leader_x"][step]) - x - float(pair["follower_length"]), 1.e-3)
            # Use the same posterior white-noise term as the Table-II
            # source-protocol simulation.  A fixed .1 m/s² belongs only to
            # the separate public ring-script comparison, not this Fig. 8/9.
            residual = history.step(step * DT, rng.standard_normal()) + iid_sigma * rng.standard_normal()
            acceleration = max(float(idm(gap, speed, speed - leader_speed, theta) + residual), -10.)
            next_speed = max(speed + acceleration * DT, 0.)
            x += .5 * (speed + next_speed) * DT
            states[0][draw, step], states[1][draw, step], states[2][draw, step] = x, next_speed, acceleration
            speed = next_speed
    return tuple(states)


def fig8_9(trace_path: Path, pairs: list[dict], output: Path) -> None:
    trace = az.from_netcdf(trace_path)
    theta = np.asarray(trace.posterior["theta"]).reshape(-1, len(pairs), 5)
    sigma = np.sqrt(np.asarray(trace.posterior["s2_f"]).reshape(-1))
    ell = np.abs(np.asarray(trace.posterior["ell_frames"]).reshape(-1)) * DT
    iid_sigma = float(np.sqrt(np.asarray(trace.posterior["s2_a"]).mean()))
    driver_index = next(i for i, pair in enumerate(pairs) if int(pair["follower_id"]) == 211)
    pair = pairs[driver_index]
    position, speed, acceleration = simulate_exogenous(pair, theta, sigma, ell, iid_sigma, driver_index)
    time = np.arange(position.shape[1]) * DT
    truth = (pair["follower_x"][1:], pair["follower_v"][1:], np.diff(pair["follower_v"]) / DT)
    fig, axes = plt.subplots(3, 1, figsize=(10, 7.5), sharex=True)
    for axis, sample, observed, label in zip(axes, (position, speed, acceleration), truth, ("position [m]", "speed [m/s]", "acceleration [m/s²]")):
        low, mid, high = np.quantile(sample, [.05, .5, .95], axis=0)
        axis.fill_between(time, low, high, color="#d62728", alpha=.2, label="90% stochastic envelope")
        axis.plot(time, mid, color="#d62728", lw=1., label="MA-IDM median")
        axis.plot(time, observed, color="black", lw=.9, label="highD")
        axis.set_ylabel(label)
    axes[0].legend(fontsize=7, ncol=3); axes[-1].set_xlabel("time [s]")
    fig.suptitle("Fixed-driver exogenous-leader MA-IDM simulation: truck #211 (Fig. 8 analogue)")
    save(fig, output / "fig8_author_reference_truck211.png")
    relative = pair["leader_x"][1:] - position - float(pair["follower_length"])
    observed_gap = pair["gap"][1:]
    fig, axis = plt.subplots(figsize=(10, 4.5))
    low, mid, high = np.quantile(relative, [.05, .5, .95], axis=0)
    axis.fill_between(time, low, high, color="#d62728", alpha=.2, label="MA-IDM 90% interval")
    axis.plot(time, mid, color="#d62728", lw=1., label="MA-IDM median")
    axis.plot(time, observed_gap, color="black", lw=.9, label="highD relative space")
    axis.set(xlabel="time [s]", ylabel="relative space [m]", title="Posterior time--space uncertainty, truck #211 (Fig. 9 analogue)")
    axis.legend(fontsize=8)
    save(fig, output / "fig9_author_reference_truck211.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(ROOT / "bayesian_ma_idm/evidence/paper_reference/zhang_sun_author_20pairs_5hz.npz"))
    parser.add_argument("--trace", default=str(ROOT / "bayesian_ma_idm/evidence/paper_reference/nuts/author_reference_trace.nc"))
    parser.add_argument("--output-dir", default=str(ROOT / "bayesian_ma_idm/evidence/paper_reference/figures"))
    args = parser.parse_args(); output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    _, pairs = load_author_reference(args.dataset)
    fig5(pairs, output); fig6_7(Path(args.trace), pairs, output); fig8_9(Path(args.trace), pairs, output)
    print(output)


if __name__ == "__main__":
    main()
