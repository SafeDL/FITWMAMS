"""EVT diagnostic plots shared by highD following and cut-in fitting."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import genpareto

from tools.evt import RETURN_PERIODS, empirical_survival, fit_gpd_excess
from tools.plot_style import (
    CRITICAL_COLOR,
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


def _mean_excess_rows(
    values: np.ndarray,
    candidates: list[dict[str, float]],
) -> tuple[np.ndarray, np.ndarray]:
    u_values = np.asarray([row["u"] for row in candidates], dtype=np.float64)
    means = []
    for u_value in u_values:
        excess = values[values > u_value] - u_value
        means.append(float(np.mean(excess)) if excess.size else float("nan"))
    return u_values, np.asarray(means, dtype=np.float64)


def _math_name(name: str) -> str:
    parts = str(name).split("_", 1)
    if len(parts) == 2 and parts[0] and parts[1]:
        return rf"{parts[0]}_{{\mathrm{{{parts[1]}}}}}"
    return str(name)


def _gpd_conditional_ppf(probability: np.ndarray, *, xi: float, beta: float) -> np.ndarray:
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
        idx = np.unique(np.round(np.linspace(0, candidate_u.size - 1, int(max_points))).astype(int))
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
            boot_scale.append(float(beta_hat + xi_hat * float(threshold)))
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
    risk_variable: str = "Y_long",
    output_filename: str = "peak_evt_gpd_diagnostic_panel.png",
    output_key: str = "peak_gpd_diagnostic_panel",
    force: bool = True,
    max_plot_value: float | None = None,
    stability_legend_loc: str = "lower left",
    stability_threshold_ymax: float = 1.0,
) -> dict[str, str]:
    """Write a six-panel POT/GPD diagnostic plot for peak EVT calibration."""
    path = figure_dir / output_filename
    if path.exists() and not force:
        return {output_key: str(path)}

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
    cand_mod_scale = np.asarray([float(row["modified_scale"]) for row in cand], dtype=float)
    mean_excess = np.asarray(
        [np.mean(values[values > threshold] - threshold) for threshold in cand_u],
        dtype=float,
    )

    full_empirical_cdf = np.arange(1, excess.size + 1, dtype=float) / (excess.size + 1.0)
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

        x_limit = max(float(np.quantile(values, 0.999)), u * 1.08, float(np.max(tail_values)))
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
        axes[0].axvline(u, color=REFERENCE_COLOR, linestyle="--", linewidth=1.2, label=r"selected $u_e$")
        axes[0].set_yscale("log")
        axes[0].set_xlabel(fr"Original risk ${risk_math}$")
        axes[0].set_ylabel("Count")
        axes[0].set_title("Peak distribution")
        axes[0].legend(frameon=False, loc="upper right")

        survival_x = np.sort(values[values >= u])
        if np.isfinite(plot_max):
            survival_x = survival_x[survival_x <= plot_max]
        if survival_x.size == 0:
            survival_x = np.asarray([u], dtype=float)
        survival_grid = np.linspace(u, float(np.max(survival_x)), 360)
        empirical = np.maximum(_empirical_survival(values, survival_x), 1.0 / values.size)
        fitted_survival = np.maximum(np.asarray(model.survival(survival_grid), dtype=float), 1.0e-12)
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
        axes[1].axvline(u, color=REFERENCE_COLOR, linestyle="--", linewidth=1.2, label=r"POT threshold $u_e$")
        axes[1].set_yscale("log")
        axes[1].set_xlabel(fr"Original risk ${risk_math}$")
        axes[1].set_ylabel(fr"$\Pr({risk_math}>y)$")
        axes[1].set_title("Tail survival")
        handles, labels = axes[1].get_legend_handles_labels()
        order = ["Empirical survival", "GPD fitted survival", "95% bootstrap CI", r"POT threshold $u_e$"]
        ordered = [(handles[labels.index(label)], label) for label in order if label in labels]
        axes[1].legend([item[0] for item in ordered], [item[1] for item in ordered], frameon=False, loc="upper right")

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
        axes[2].axvline(u, color=REFERENCE_COLOR, linestyle="--", linewidth=1.2, label=r"selected $u_e$")
        axes[2].axvspan(u, float(np.max(cand_u_plot)), color=GENERATED_COLOR, alpha=0.08)
        axes[2].set_xlabel(r"Threshold $u$")
        axes[2].set_ylabel(fr"$E[{risk_math}-u\mid {risk_math}>u]$")
        axes[2].set_title("Mean residual life")
        axes[2].legend(frameon=False, loc="upper right")

        band_u, xi_low, xi_high, scale_low, scale_high = _bootstrap_threshold_stability_band(values, cand_u_plot)
        ax_scale = axes[3].twinx()
        if band_u.size:
            axes[3].fill_between(band_u, xi_low, xi_high, color=REAL_COLOR, alpha=0.17, linewidth=0.0)
            ax_scale.fill_between(band_u, scale_low, scale_high, color=GENERATED_COLOR, alpha=0.17, linewidth=0.0)
        line_xi = axes[3].plot(cand_u_plot, cand_xi_plot, color=REAL_COLOR, linewidth=1.45, label=r"$\xi$")
        line_scale = ax_scale.plot(cand_u_plot, cand_mod_scale_plot, color=GENERATED_COLOR, linewidth=1.45, label=r"$\tilde{\sigma}$")
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
        axes[4].plot([0.0, max_q], [0.0, max_q], color=REFERENCE_COLOR, linestyle="--", linewidth=1.2)
        axes[4].set_xlim(0.0, max_q * 1.02)
        axes[4].set_ylim(0.0, max_q * 1.02)
        axes[4].set_xlabel("GPD quantile")
        axes[4].set_ylabel("Empirical quantile")
        axes[4].set_title("QQ plot")

        axes[5].scatter(empirical_cdf, gpd_cdf, color=GENERATED_COLOR, s=10, alpha=0.45)
        axes[5].plot([0.0, 1.0], [0.0, 1.0], color=REFERENCE_COLOR, linestyle="--", linewidth=1.2)
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
    return {output_key: str(path)}


def write_evt_diagnostic_plots(
    figure_dir: Path,
    *,
    model: Any,
    values: np.ndarray,
    collision_critical_level: float,
    risk_variable: str = "Y_long",
    histogram_filename: str = "peak_evt_y_long_histogram.png",
    histogram_key: str = "peak_y_long_histogram",
) -> dict[str, str]:
    """Write standard EVT diagnostic plots without exposure-dependent plots."""
    plt = get_pyplot()

    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {}

    figure_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    risk_label = str(risk_variable)
    risk_math = _math_name(risk_label)
    risk_peak_math = rf"{risk_math},\mathrm{{peak}}"
    x_limit = float(np.quantile(values, 0.999))
    x_limit = max(x_limit, float(model.u) * 1.05, 1.0)
    fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    clipped_values = values[values <= x_limit]
    _, bins, _ = ax.hist(
        clipped_values,
        bins=70,
        color=REAL_COLOR,
        alpha=0.62,
        label="Empirical peaks",
    )
    ax.axvline(float(model.u), color=REFERENCE_COLOR, linestyle="--", label=r"POT threshold $u_e$")
    tail_count = int(np.sum(values > float(model.u)))
    if tail_count > 0 and x_limit > float(model.u):
        bin_width = float(np.mean(np.diff(bins)))
        xs = np.linspace(float(model.u), x_limit, 320)
        tail_pdf = genpareto.pdf(
            xs - float(model.u),
            c=float(model.xi),
            scale=float(model.beta),
        )
        expected_tail_bin_count = tail_count * bin_width * tail_pdf
        ax.plot(
            xs,
            np.maximum(expected_tail_bin_count, 1.0e-12),
            color=GENERATED_COLOR,
            linewidth=2.2,
            label="GPD fit",
        )
    for period in RETURN_PERIODS:
        key = f"z{period}"
        if key in model.return_levels:
            z_value = float(model.return_levels[key])
            if z_value <= x_limit:
                ax.axvline(z_value, linestyle=":", label=rf"$z_{{{period}}}$")
    if float(collision_critical_level) <= x_limit:
        ax.axvline(
            float(collision_critical_level),
            color=CRITICAL_COLOR,
            linestyle="-.",
            label=r"$x_e^\star$",
        )
    ax.set_yscale("log")
    ax.set_xlim(left=0.0, right=x_limit)
    ax.set_xlabel(fr"Peak original risk ${risk_math}$")
    ax.set_ylabel("Count (log)")
    ax.set_title(fr"Peak ${risk_math}$ distribution")
    style_axes(ax)
    ax.legend(frameon=False)
    path = figure_dir / histogram_filename
    fig.savefig(path, dpi=PAPER_FIGURE_DPI)
    plt.close(fig)
    paths[histogram_key] = str(path)

    sorted_values = np.sort(values[values >= float(model.u)])
    if sorted_values.size:
        empirical = empirical_survival(values, sorted_values)
        model_survival = model.survival(sorted_values)
        fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), constrained_layout=True)
        focus_max = float(
            max(
                float(model.return_levels.get("z100", model.return_level(100))) * 1.35,
                np.quantile(values, 0.995),
            )
        )
        panels = (
            (axes[0], focus_max, "return-level region"),
            (axes[1], float(np.max(values)), "full tail"),
        )
        for ax, right_limit, title in panels:
            mask = sorted_values <= right_limit
            ax.scatter(
                sorted_values[mask],
                empirical[mask],
                color=REAL_COLOR,
                label="Empirical",
                s=7,
                alpha=0.38,
            )
            ax.plot(
                sorted_values[mask],
                model_survival[mask],
                label="GPD",
                linewidth=2.0,
                color=GENERATED_COLOR,
            )
            ax.axvline(float(model.u), color=REFERENCE_COLOR, linestyle="--", label=r"$u_e$")
            for period in RETURN_PERIODS:
                key = f"z{period}"
                if key not in model.return_levels:
                    continue
                z_value = float(model.return_levels[key])
                if z_value <= right_limit:
                    survival = float(model.survival(z_value))
                    ax.scatter([z_value], [survival], s=32)
                    ax.annotate(key, (z_value, survival))
            ax.set_yscale("log")
            ax.set_xlabel(fr"Peak ${risk_math}$")
            ax.set_title(title)
            style_axes(ax)
        axes[0].set_ylabel(fr"$\Pr({risk_peak_math} > y)$")
        axes[1].legend(frameon=False)
        path = figure_dir / "peak_evt_survival_fit.png"
        fig.savefig(path, dpi=PAPER_FIGURE_DPI)
        plt.close(fig)
        paths["peak_survival_fit"] = str(path)

    candidates = model.threshold_candidates
    if candidates:
        u = np.asarray([row["u"] for row in candidates], dtype=np.float64)
        xi = np.asarray([row["xi"] for row in candidates], dtype=np.float64)
        modified_scale = np.asarray(
            [row["modified_scale"] for row in candidates],
            dtype=np.float64,
        )
        exceedance_rate = np.asarray(
            [row["exceedance_rate"] for row in candidates],
            dtype=np.float64,
        )
        fig, axes = plt.subplots(3, 1, figsize=(8.0, 7.2), sharex=True)
        for ax in axes:
            ax.axvline(float(model.u), color=REFERENCE_COLOR, linestyle="--", linewidth=1.3)
            ax.axvspan(
                float(model.u),
                float(np.max(u)),
                color=GENERATED_COLOR,
                alpha=0.08,
                label=fr"${risk_math}>u_e$",
            )
            style_axes(ax)
        axes[0].plot(u, xi, linewidth=1.4)
        axes[0].scatter([float(model.u)], [float(model.xi)], color=REFERENCE_COLOR, s=28)
        axes[0].set_ylabel(r"$\xi$")
        axes[0].set_title("Threshold stability")
        axes[1].plot(u, modified_scale, linewidth=1.4, color=REAL_COLOR)
        axes[1].set_ylabel(r"$\tilde{\sigma}$")
        axes[2].plot(u, exceedance_rate, linewidth=1.4, color=GENERATED_COLOR)
        axes[2].set_xlabel(r"Threshold $u$")
        axes[2].set_ylabel(fr"$\Pr({risk_math}>u)$")
        fig.tight_layout()
        path = figure_dir / "peak_evt_threshold_stability.png"
        fig.savefig(path, dpi=PAPER_FIGURE_DPI)
        plt.close(fig)
        paths["peak_threshold_stability"] = str(path)

        mean_u, mean_excess = _mean_excess_rows(values, candidates)
        fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
        ax.plot(mean_u, mean_excess, linewidth=1.5)
        ax.axvline(float(model.u), color=REFERENCE_COLOR, linestyle="--", label=r"$u_e$")
        ax.axvspan(float(model.u), float(np.max(mean_u)), color=GENERATED_COLOR, alpha=0.08)
        ax.set_xlabel(r"Threshold $u$")
        ax.set_ylabel(fr"$E[{risk_math}-u \mid {risk_math}>u]$")
        ax.set_title("Mean residual life")
        style_axes(ax)
        ax.legend(frameon=False)
        path = figure_dir / "peak_evt_mean_excess.png"
        fig.savefig(path, dpi=PAPER_FIGURE_DPI)
        plt.close(fig)
        paths["peak_mean_excess"] = str(path)

    excess = np.sort(values[values > float(model.u)] - float(model.u))
    if excess.size:
        empirical_cdf = np.arange(1, excess.size + 1, dtype=np.float64)
        empirical_cdf = empirical_cdf / (excess.size + 1.0)
        gpd_cdf = genpareto.cdf(
            excess,
            c=float(model.xi),
            loc=0.0,
            scale=max(float(model.beta), 1.0e-12),
        )
        gpd_quantiles = genpareto.ppf(
            empirical_cdf,
            c=float(model.xi),
            loc=0.0,
            scale=max(float(model.beta), 1.0e-12),
        )
        fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.6), constrained_layout=True)
        max_q = float(max(np.max(excess), np.max(gpd_quantiles)))
        axes[0].scatter(gpd_quantiles, excess, s=8, alpha=0.40)
        axes[0].plot([0.0, max_q], [0.0, max_q], color=REFERENCE_COLOR, linestyle="--")
        axes[0].set_xlabel("GPD quantile")
        axes[0].set_ylabel("Empirical quantile")
        axes[0].set_title("QQ plot")
        style_axes(axes[0])
        axes[1].scatter(empirical_cdf, gpd_cdf, s=8, alpha=0.40, color=GENERATED_COLOR)
        axes[1].plot([0.0, 1.0], [0.0, 1.0], color=REFERENCE_COLOR, linestyle="--")
        axes[1].set_xlabel("Empirical CDF")
        axes[1].set_ylabel("GPD CDF")
        axes[1].set_title("PP plot")
        style_axes(axes[1])
        path = figure_dir / "peak_evt_tail_fit_diagnostics.png"
        fig.savefig(path, dpi=PAPER_FIGURE_DPI)
        plt.close(fig)
        paths["peak_tail_fit_diagnostics"] = str(path)

    return paths
