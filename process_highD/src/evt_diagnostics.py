"""EVT diagnostic plots for preprocessed highD fixed-window risks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import genpareto

from tools.evt import fit_gpd_excess
from tools.plot_style import (
    GENERATED_COLOR,
    PAPER_FIGURE_DPI,
    PAPER_PANEL_LABELSIZE,
    PAPER_PANEL_RC,
    PAPER_SIX_PANEL_FIGSIZE,
    PAPER_SIX_PANEL_LAYOUT,
    REAL_COLOR,
    REFERENCE_COLOR,
    get_pyplot,
    style_axes,
)


def _math_name(name: str) -> str:
    parts = str(name).split("_", 1)
    if len(parts) == 2 and parts[0] and parts[1]:
        return rf"{parts[0]}_{{\mathrm{{{parts[1]}}}}}"
    return str(name)


def _gpd_conditional_ppf(
    probability: np.ndarray, *, xi: float, beta: float
) -> np.ndarray:
    p = np.clip(np.asarray(probability, dtype=float), 1.0e-9, 1.0 - 1.0e-9)
    beta = max(float(beta), 1.0e-12)
    if abs(float(xi)) < 1.0e-10:
        return -beta * np.log1p(-p)
    return beta * (np.power(1.0 - p, -float(xi)) - 1.0) / float(xi)


def _empirical_survival(values: np.ndarray, query: np.ndarray) -> np.ndarray:
    sorted_values = np.sort(np.asarray(values, dtype=float))
    right = np.searchsorted(sorted_values, np.asarray(query, dtype=float), side="right")
    return (sorted_values.size - right) / float(sorted_values.size)


def _bootstrap_gpd_survival_band(
    survival_x: np.ndarray,
    *,
    u: float,
    xi: float,
    beta: float,
    exceedance_rate: float,
    n_excess: int,
    samples: int = 400,
    random_seed: int = 20240613,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(random_seed))
    curves: list[np.ndarray] = []
    for _ in range(int(samples)):
        uniforms = rng.uniform(1.0e-9, 1.0 - 1.0e-9, size=int(n_excess))
        boot_excess = _gpd_conditional_ppf(uniforms, xi=xi, beta=beta)
        try:
            xi_hat, beta_hat = fit_gpd_excess(boot_excess)
        except ValueError:
            continue
        tail = genpareto.sf(
            np.maximum(np.asarray(survival_x, dtype=float) - float(u), 0.0),
            c=float(xi_hat),
            loc=0.0,
            scale=max(float(beta_hat), 1.0e-12),
        )
        curve = float(exceedance_rate) * tail
        if np.all(np.isfinite(curve)):
            curves.append(curve)
    if not curves:
        nan = np.full_like(survival_x, np.nan, dtype=float)
        return nan, nan
    arr = np.asarray(curves, dtype=float)
    return np.nanpercentile(arr, 2.5, axis=0), np.nanpercentile(arr, 97.5, axis=0)


def _bootstrap_threshold_stability_band(
    values: np.ndarray,
    candidate_u: np.ndarray,
    *,
    samples: int = 120,
    max_points: int = 80,
    random_seed: int = 20240614,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if candidate_u.size == 0:
        empty = np.asarray([], dtype=float)
        return empty, empty, empty, empty, empty
    if candidate_u.size <= int(max_points):
        grid_u = candidate_u
    else:
        idx = np.unique(
            np.round(np.linspace(0, candidate_u.size - 1, int(max_points))).astype(int)
        )
        grid_u = candidate_u[idx]

    rng = np.random.default_rng(int(random_seed))
    xi_low: list[float] = []
    xi_high: list[float] = []
    scale_low: list[float] = []
    scale_high: list[float] = []
    for threshold in grid_u:
        excess = np.asarray(values[values > threshold] - threshold, dtype=float)
        excess = excess[np.isfinite(excess) & (excess > 0.0)]
        if excess.size < 20:
            xi_low.append(float("nan"))
            xi_high.append(float("nan"))
            scale_low.append(float("nan"))
            scale_high.append(float("nan"))
            continue
        boot_xi: list[float] = []
        boot_scale: list[float] = []
        for _ in range(int(samples)):
            boot_excess = rng.choice(excess, size=excess.size, replace=True)
            try:
                xi_hat, beta_hat = fit_gpd_excess(boot_excess)
            except ValueError:
                continue
            boot_xi.append(float(xi_hat))
            boot_scale.append(float(beta_hat - xi_hat * float(threshold)))
        if boot_xi:
            xi_low.append(float(np.nanpercentile(boot_xi, 2.5)))
            xi_high.append(float(np.nanpercentile(boot_xi, 97.5)))
            scale_low.append(float(np.nanpercentile(boot_scale, 2.5)))
            scale_high.append(float(np.nanpercentile(boot_scale, 97.5)))
        else:
            xi_low.append(float("nan"))
            xi_high.append(float("nan"))
            scale_low.append(float("nan"))
            scale_high.append(float("nan"))
    return (
        np.asarray(grid_u, dtype=float),
        np.asarray(xi_low, dtype=float),
        np.asarray(xi_high, dtype=float),
        np.asarray(scale_low, dtype=float),
        np.asarray(scale_high, dtype=float),
    )


def write_gpd_diagnostic_panel(
    figure_dir: Path,
    *,
    model: Any,
    values: np.ndarray,
    risk_variable: str = "R_tau",
    output_filename: str = "gpd_diagnostics.png",
    output_key: str = "gpd_diagnostics",
    max_plot_value: float | None = None,
    stability_legend_loc: str = "lower left",
    stability_threshold_ymax: float = 1.0,
) -> dict[str, str]:
    """Write a six-panel POT/GPD diagnostic plot for peak EVT calibration."""
    path = figure_dir / output_filename
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    candidates = list(getattr(model, "threshold_candidates", []) or [])
    u = float(model.u)
    xi = float(model.xi)
    beta = float(model.beta)
    lam = float(model.exceedance_rate)
    if not (values.size and candidates and all(np.isfinite([u, xi, beta, lam]))):
        return {}

    tail_values = np.sort(values[values > u])
    excess = tail_values - u
    if excess.size == 0:
        return {}
    plot_max = float(max_plot_value) if max_plot_value is not None else float("nan")
    if np.isfinite(plot_max) and plot_max <= u:
        plot_max = float("nan")

    cand = sorted(candidates, key=lambda row: float(row["u"]))
    cand_u = np.asarray([float(row["u"]) for row in cand], dtype=float)
    cand_xi = np.asarray([float(row["xi"]) for row in cand], dtype=float)
    cand_mod_scale = np.asarray(
        [float(row["modified_scale"]) for row in cand], dtype=float
    )
    mean_excess = np.asarray(
        [np.mean(values[values > threshold] - threshold) for threshold in cand_u],
        dtype=float,
    )

    full_empirical_cdf = np.arange(1, excess.size + 1, dtype=float) / (
        excess.size + 1.0
    )
    full_gpd_quantiles = genpareto.ppf(
        full_empirical_cdf,
        c=xi,
        loc=0.0,
        scale=max(beta, 1.0e-12),
    )
    full_gpd_cdf = genpareto.cdf(
        excess,
        c=xi,
        loc=0.0,
        scale=max(beta, 1.0e-12),
    )
    if np.isfinite(plot_max):
        qq_mask = tail_values <= plot_max
        empirical_cdf = full_empirical_cdf[qq_mask]
        gpd_quantiles = full_gpd_quantiles[qq_mask]
        gpd_cdf = full_gpd_cdf[qq_mask]
        plot_excess = excess[qq_mask]
        if plot_excess.size == 0:
            empirical_cdf = full_empirical_cdf
            gpd_quantiles = full_gpd_quantiles
            gpd_cdf = full_gpd_cdf
            plot_excess = excess
    else:
        empirical_cdf = full_empirical_cdf
        gpd_quantiles = full_gpd_quantiles
        gpd_cdf = full_gpd_cdf
        plot_excess = excess

    figure_dir.mkdir(parents=True, exist_ok=True)
    plt = get_pyplot()
    risk_math = _math_name(risk_variable)
    with plt.rc_context(PAPER_PANEL_RC):
        fig, axes = plt.subplots(2, 3, figsize=PAPER_SIX_PANEL_FIGSIZE)
        axes = axes.ravel()

        x_limit = max(
            float(np.quantile(values, 0.999)), u * 1.08, float(np.max(tail_values))
        )
        x_limit = min(x_limit, float(np.max(values)))
        if np.isfinite(plot_max):
            x_limit = min(x_limit, plot_max)
        clipped = values[values <= x_limit]
        _counts, bins, _patches = axes[0].hist(
            clipped,
            bins=64,
            color=REAL_COLOR,
            alpha=0.76,
            label="Peak distribution",
        )
        xs = np.linspace(u, x_limit, 360)
        bin_width = float(np.mean(np.diff(bins)))
        tail_count = int(np.sum(values > u))
        density = genpareto.pdf(
            xs - u,
            c=xi,
            loc=0.0,
            scale=max(beta, 1.0e-12),
        )
        axes[0].plot(
            xs,
            np.maximum(tail_count * bin_width * density, 1.0e-12),
            color=GENERATED_COLOR,
            linewidth=1.8,
            label="GPD fitted tail",
        )
        axes[0].axvline(
            u,
            color=REFERENCE_COLOR,
            linestyle="--",
            linewidth=1.2,
            label=r"selected $u_e$",
        )
        axes[0].set_yscale("log")
        axes[0].set_xlabel(rf"Original risk ${risk_math}$")
        axes[0].set_ylabel("Count")
        axes[0].set_title("Peak distribution")
        axes[0].legend(frameon=False, loc="upper right")

        survival_x = np.sort(values[values >= u])
        if np.isfinite(plot_max):
            survival_x = survival_x[survival_x <= plot_max]
        if survival_x.size == 0:
            survival_x = np.asarray([u], dtype=float)
        survival_grid = np.linspace(u, float(np.max(survival_x)), 360)
        empirical = np.maximum(
            _empirical_survival(values, survival_x), 1.0 / values.size
        )
        fitted_survival = np.maximum(
            np.asarray(model.survival(survival_grid), dtype=float), 1.0e-12
        )
        survival_low, survival_high = _bootstrap_gpd_survival_band(
            survival_grid,
            u=u,
            xi=xi,
            beta=beta,
            exceedance_rate=lam,
            n_excess=excess.size,
        )
        if np.any(np.isfinite(survival_low)) and np.any(np.isfinite(survival_high)):
            axes[1].fill_between(
                survival_grid,
                np.maximum(survival_low, 1.0e-12),
                np.maximum(survival_high, 1.0e-12),
                color=GENERATED_COLOR,
                alpha=0.20,
                linewidth=0.0,
                label="95% bootstrap CI",
            )
        axes[1].scatter(
            survival_x,
            empirical,
            color=REAL_COLOR,
            s=10,
            alpha=0.48,
            label="Empirical survival",
        )
        axes[1].plot(
            survival_grid,
            fitted_survival,
            color=GENERATED_COLOR,
            linewidth=1.8,
            label="GPD fitted survival",
        )
        axes[1].axvline(
            u,
            color=REFERENCE_COLOR,
            linestyle="--",
            linewidth=1.2,
            label=r"POT threshold $u_e$",
        )
        axes[1].set_yscale("log")
        axes[1].set_xlabel(rf"Original risk ${risk_math}$")
        axes[1].set_ylabel(rf"$\Pr({risk_math}>y)$")
        axes[1].set_title("Tail survival")
        handles, labels = axes[1].get_legend_handles_labels()
        order = [
            "Empirical survival",
            "GPD fitted survival",
            "95% bootstrap CI",
            r"POT threshold $u_e$",
        ]
        ordered = [
            (handles[labels.index(label)], label) for label in order if label in labels
        ]
        axes[1].legend(
            [item[0] for item in ordered],
            [item[1] for item in ordered],
            frameon=False,
            loc="upper right",
        )

        cand_plot_mask = np.ones_like(cand_u, dtype=bool)
        if np.isfinite(plot_max):
            cand_plot_mask = cand_u <= plot_max
            if not np.any(cand_plot_mask):
                cand_plot_mask = np.ones_like(cand_u, dtype=bool)
        cand_u_plot = cand_u[cand_plot_mask]
        cand_xi_plot = cand_xi[cand_plot_mask]
        cand_mod_scale_plot = cand_mod_scale[cand_plot_mask]
        mean_excess_plot = mean_excess[cand_plot_mask]

        axes[2].plot(cand_u_plot, mean_excess_plot, color=REAL_COLOR, linewidth=1.5)
        axes[2].axvline(
            u,
            color=REFERENCE_COLOR,
            linestyle="--",
            linewidth=1.2,
            label=r"selected $u_e$",
        )
        axes[2].axvspan(
            u, float(np.max(cand_u_plot)), color=GENERATED_COLOR, alpha=0.08
        )
        axes[2].set_xlabel(r"Threshold $u$")
        axes[2].set_ylabel(rf"$E[{risk_math}-u\mid {risk_math}>u]$")
        axes[2].set_title("Mean residual life")
        axes[2].legend(frameon=False, loc="upper right")

        band_u, xi_low, xi_high, scale_low, scale_high = (
            _bootstrap_threshold_stability_band(values, cand_u_plot)
        )
        ax_scale = axes[3].twinx()
        if band_u.size:
            axes[3].fill_between(
                band_u, xi_low, xi_high, color=REAL_COLOR, alpha=0.17, linewidth=0.0
            )
            ax_scale.fill_between(
                band_u,
                scale_low,
                scale_high,
                color=GENERATED_COLOR,
                alpha=0.17,
                linewidth=0.0,
            )
        line_xi = axes[3].plot(
            cand_u_plot, cand_xi_plot, color=REAL_COLOR, linewidth=1.45, label=r"$\xi$"
        )
        line_scale = ax_scale.plot(
            cand_u_plot,
            cand_mod_scale_plot,
            color=GENERATED_COLOR,
            linewidth=1.45,
            label=r"$\tilde{\sigma}$",
        )
        threshold_line = axes[3].axvline(
            u,
            ymin=0.0,
            ymax=float(np.clip(stability_threshold_ymax, 0.0, 1.0)),
            color=REFERENCE_COLOR,
            linestyle="--",
            linewidth=1.2,
            label=r"selected $u_e$",
        )
        axes[3].axhline(xi, color=REAL_COLOR, linestyle=":", linewidth=0.9, alpha=0.55)
        axes[3].set_xlabel(r"Threshold $u$")
        axes[3].set_ylabel(r"Shape $\xi$", color=REAL_COLOR)
        ax_scale.set_ylabel(r"Modified scale $\tilde{\sigma}$", color=GENERATED_COLOR)
        axes[3].tick_params(axis="y", colors=REAL_COLOR)
        ax_scale.tick_params(axis="y", colors=GENERATED_COLOR)
        axes[3].spines["left"].set_color(REAL_COLOR)
        ax_scale.spines["right"].set_color(GENERATED_COLOR)
        axes[3].set_title("Threshold stability")
        legend_handles = [line_xi[0], line_scale[0], threshold_line]
        axes[3].legend(
            legend_handles,
            [handle.get_label() for handle in legend_handles],
            frameon=False,
            loc=stability_legend_loc,
        )

        max_q = float(max(np.max(plot_excess), np.max(gpd_quantiles)))
        if np.isfinite(plot_max):
            max_q = min(max_q, max(float(plot_max - u), 1.0e-9))
        axes[4].scatter(gpd_quantiles, plot_excess, color=REAL_COLOR, s=10, alpha=0.45)
        axes[4].plot(
            [0.0, max_q],
            [0.0, max_q],
            color=REFERENCE_COLOR,
            linestyle="--",
            linewidth=1.2,
        )
        axes[4].set_xlim(0.0, max_q * 1.02)
        axes[4].set_ylim(0.0, max_q * 1.02)
        axes[4].set_xlabel("GPD quantile")
        axes[4].set_ylabel("Empirical quantile")
        axes[4].set_title("QQ plot")

        axes[5].scatter(empirical_cdf, gpd_cdf, color=GENERATED_COLOR, s=10, alpha=0.45)
        axes[5].plot(
            [0.0, 1.0], [0.0, 1.0], color=REFERENCE_COLOR, linestyle="--", linewidth=1.2
        )
        axes[5].set_xlim(0.0, 1.0)
        axes[5].set_ylim(0.0, 1.0)
        axes[5].set_xlabel("Empirical CDF")
        axes[5].set_ylabel("GPD CDF")
        axes[5].set_title("PP plot")

        for label, ax in zip(("a", "b", "c", "d", "e", "f"), axes):
            style_axes(ax)
            ax.text(
                -0.105,
                1.065,
                label,
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=PAPER_PANEL_LABELSIZE,
                fontweight="bold",
                clip_on=False,
            )
        style_axes(ax_scale, grid=False)
        ax_scale.spines["right"].set_visible(True)
        ax_scale.spines["right"].set_color(GENERATED_COLOR)
        fig.tight_layout(**PAPER_SIX_PANEL_LAYOUT)
        fig.savefig(path, dpi=PAPER_FIGURE_DPI)
        plt.close(fig)
    outputs = {output_key: str(path)}
    outputs.update(
        _write_gpd_diagnostic_figures(
            figure_dir,
            model=model,
            values=values,
            risk_variable=risk_variable,
            survival_band=(survival_grid, survival_low, survival_high),
            stability_band=(band_u, xi_low, xi_high, scale_low, scale_high),
        )
    )
    return outputs


def _write_gpd_diagnostic_figures(
    figure_dir: Path,
    *,
    model: Any,
    values: np.ndarray,
    risk_variable: str = "R_SEI",
    survival_band: tuple[np.ndarray, np.ndarray, np.ndarray],
    stability_band: tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ],
) -> dict[str, str]:
    """Write focused standalone versions of the six GPD diagnostics."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    u = float(model.u)
    xi = float(model.xi)
    beta = float(model.beta)
    tail = np.sort(values[values > u])
    excess = tail - u
    candidates = sorted(model.threshold_candidates, key=lambda row: float(row["u"]))
    candidate_u = np.asarray([row["u"] for row in candidates], dtype=float)
    candidate_xi = np.asarray([row["xi"] for row in candidates], dtype=float)
    modified_scale = np.asarray(
        [row["modified_scale"] for row in candidates], dtype=float
    )
    mean_excess = np.asarray(
        [np.mean(values[values > threshold] - threshold) for threshold in candidate_u]
    )
    empirical_cdf = np.arange(1, excess.size + 1, dtype=float) / (excess.size + 1.0)
    gpd_quantiles = genpareto.ppf(empirical_cdf, c=xi, loc=0.0, scale=beta)
    gpd_cdf = genpareto.cdf(excess, c=xi, loc=0.0, scale=beta)
    risk_math = _math_name(risk_variable)
    figure_dir.mkdir(parents=True, exist_ok=True)
    plt = get_pyplot()
    outputs: dict[str, str] = {}

    with plt.rc_context(PAPER_PANEL_RC):
        fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
        _counts, bins, _patches = ax.hist(
            values,
            bins=64,
            color=REAL_COLOR,
            alpha=0.76,
            label="Peak distribution",
        )
        xs = np.linspace(u, float(np.max(values)), 360)
        bin_width = float(np.mean(np.diff(bins)))
        density = genpareto.pdf(xs - u, c=xi, loc=0.0, scale=beta)
        ax.plot(
            xs,
            np.maximum(excess.size * bin_width * density, 1.0e-12),
            color=GENERATED_COLOR,
            linewidth=1.8,
            label="GPD fitted tail",
        )
        ax.axvline(u, color=REFERENCE_COLOR, linestyle="--", label=r"selected $u_e$")
        ax.set_yscale("log")
        ax.set_xlabel(rf"Original risk ${risk_math}$")
        ax.set_ylabel("Count")
        ax.set_title("Peak distribution")
        style_axes(ax)
        ax.legend(frameon=False)
        path = figure_dir / "risk_distribution.png"
        fig.savefig(path, dpi=PAPER_FIGURE_DPI)
        plt.close(fig)
        outputs["risk_distribution"] = str(path)

        fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
        empirical_survival = np.maximum(
            _empirical_survival(values, tail), 1.0 / len(values)
        )
        fitted_survival = np.maximum(np.asarray(model.survival(tail), float), 1.0e-12)
        survival_x, survival_low, survival_high = survival_band
        ax.fill_between(
            survival_x,
            np.maximum(survival_low, 1.0e-12),
            np.maximum(survival_high, 1.0e-12),
            color=GENERATED_COLOR,
            alpha=0.20,
            linewidth=0.0,
            label="95% bootstrap CI",
        )
        ax.scatter(
            tail,
            empirical_survival,
            color=REAL_COLOR,
            s=10,
            alpha=0.48,
            label="Empirical survival",
        )
        ax.plot(
            tail,
            fitted_survival,
            color=GENERATED_COLOR,
            linewidth=1.8,
            label="GPD fitted survival",
        )
        ax.axvline(
            u, color=REFERENCE_COLOR, linestyle="--", label=r"POT threshold $u_e$"
        )
        ax.set_yscale("log")
        ax.set_xlabel(rf"Original risk ${risk_math}$")
        ax.set_ylabel(rf"$\Pr({risk_math}>y)$")
        ax.set_title("Tail survival")
        style_axes(ax)
        ax.legend(frameon=False)
        path = figure_dir / "tail_survival.png"
        fig.savefig(path, dpi=PAPER_FIGURE_DPI)
        plt.close(fig)
        outputs["tail_survival"] = str(path)

        fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
        ax.plot(candidate_u, mean_excess, color=REAL_COLOR)
        ax.axvline(u, color=REFERENCE_COLOR, linestyle="--", label=r"selected $u_e$")
        ax.axvspan(u, float(np.max(candidate_u)), color=GENERATED_COLOR, alpha=0.08)
        ax.set_xlabel(r"Threshold $u$")
        ax.set_ylabel(rf"$E[{risk_math}-u\mid {risk_math}>u]$")
        ax.set_title("Mean residual life")
        style_axes(ax)
        ax.legend(frameon=False)
        path = figure_dir / "mean_residual_life.png"
        fig.savefig(path, dpi=PAPER_FIGURE_DPI)
        plt.close(fig)
        outputs["mean_residual_life"] = str(path)

        fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
        scale_ax = ax.twinx()
        band_u, xi_low, xi_high, scale_low, scale_high = stability_band
        ax.fill_between(
            band_u,
            xi_low,
            xi_high,
            color=REAL_COLOR,
            alpha=0.17,
            linewidth=0.0,
        )
        scale_ax.fill_between(
            band_u,
            scale_low,
            scale_high,
            color=GENERATED_COLOR,
            alpha=0.17,
            linewidth=0.0,
        )
        shape_line = ax.plot(
            candidate_u, candidate_xi, color=REAL_COLOR, label=r"$\xi$"
        )
        scale_line = scale_ax.plot(
            candidate_u,
            modified_scale,
            color=GENERATED_COLOR,
            label=r"$\tilde{\sigma}$",
        )
        threshold_line = ax.axvline(
            u,
            color=REFERENCE_COLOR,
            linestyle="--",
            label=r"selected $u_e$",
        )
        ax.set_xlabel(r"Threshold $u$")
        ax.set_ylabel(r"Shape $\xi$", color=REAL_COLOR)
        scale_ax.set_ylabel(r"Modified scale $\tilde{\sigma}$", color=GENERATED_COLOR)
        ax.tick_params(axis="y", colors=REAL_COLOR)
        scale_ax.tick_params(axis="y", colors=GENERATED_COLOR)
        ax.set_title("Threshold stability")
        style_axes(ax)
        style_axes(scale_ax, grid=False)
        scale_ax.spines["right"].set_visible(True)
        scale_ax.spines["right"].set_color(GENERATED_COLOR)
        handles = [shape_line[0], scale_line[0], threshold_line]
        ax.legend(handles, [handle.get_label() for handle in handles], frameon=False)
        path = figure_dir / "threshold_stability.png"
        fig.savefig(path, dpi=PAPER_FIGURE_DPI)
        plt.close(fig)
        outputs["threshold_stability"] = str(path)

        max_q = float(max(np.max(excess), np.max(gpd_quantiles)))
        fig, ax = plt.subplots(figsize=(6.2, 5.2), constrained_layout=True)
        ax.scatter(gpd_quantiles, excess, color=REAL_COLOR, s=10, alpha=0.45)
        ax.plot([0.0, max_q], [0.0, max_q], color=REFERENCE_COLOR, linestyle="--")
        ax.set_xlim(0.0, max_q * 1.02)
        ax.set_ylim(0.0, max_q * 1.02)
        ax.set_xlabel("GPD quantile")
        ax.set_ylabel("Empirical quantile")
        ax.set_title("QQ plot")
        style_axes(ax)
        path = figure_dir / "gpd_qq.png"
        fig.savefig(path, dpi=PAPER_FIGURE_DPI)
        plt.close(fig)
        outputs["gpd_qq"] = str(path)

        fig, ax = plt.subplots(figsize=(6.2, 5.2), constrained_layout=True)
        ax.scatter(empirical_cdf, gpd_cdf, color=GENERATED_COLOR, s=10, alpha=0.45)
        ax.plot([0.0, 1.0], [0.0, 1.0], color=REFERENCE_COLOR, linestyle="--")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Empirical CDF")
        ax.set_ylabel("GPD CDF")
        ax.set_title("PP plot")
        style_axes(ax)
        path = figure_dir / "gpd_pp.png"
        fig.savefig(path, dpi=PAPER_FIGURE_DPI)
        plt.close(fig)
        outputs["gpd_pp"] = str(path)

    return outputs
