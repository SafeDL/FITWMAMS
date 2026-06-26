"""Build cut-in paper experiment artifacts from existing results.

This is a read-only post-processing script for existing experiment outputs.
It does not retrain models, refit EVT models, or rerun subset simulation.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

_REPO_ROOT_FOR_IMPORTS = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_FOR_IMPORTS))

import numpy as np

from process_highD.src.cutin_tail_generation import (
    _dataset_indices_for_tail_rows as _cutin_dataset_indices_for_tail_rows,
)
from tools.evt import GPDTailModel, fit_gpd_excess, return_level_for_tail_exposure
from tools.plot_style import (
    build_manifest,
    CRITICAL_COLOR,
    fget,
    GENERATED_COLOR,
    gpd_survival,
    PAPER_ANNOTATION_FONTSIZE,
    PAPER_FIGURE_DPI,
    PAPER_NOTE_BBOX,
    PAPER_PANEL_LABELSIZE,
    PAPER_PANEL_RC,
    PAPER_SINGLE_PANEL_FIGSIZE,
    PAPER_SIX_PANEL_FIGSIZE,
    PAPER_SIX_PANEL_LAYOUT,
    PAPER_SUBSET_HISTOGRAM_FIGSIZE,
    REAL_COLOR,
    read_json,
    record,
    REFERENCE_COLOR,
    rel_path,
    SAMPLED_COLOR,
    save_figure as save_figure_to,
    get_pyplot,
    descriptive_condition_label_for,
    label_for,
    style_axes,
    write_experiment_readme,
    write_json,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "paper_experiments" / "cutin"
FIGURES = OUT
LOGS = OUT / "logs"

SOURCE_PATHS = {
    "event_scores": RESULTS / "highd_events" / "cutin_event_scores.csv",
    "event_cache_summary": RESULTS / "highd_events" / "cutin_event_cache_summary.json",
    "subset_summary": ROOT
    / "IDM_subset"
    / "results"
    / "cutin"
    / "latent_subset_summary.json",
    "subset_level_stats": ROOT
    / "IDM_subset"
    / "results"
    / "cutin"
    / "latent_subset_level_stats.csv",
    "subset_samples": ROOT
    / "IDM_subset"
    / "results"
    / "cutin"
    / "latent_subset_samples.npz",
    "monte_carlo_summary": ROOT
    / "IDM_subset"
    / "results"
    / "monte_carlo_cutin"
    / "latent_monte_carlo_summary.json",
    "cutin_diffusion_dataset": RESULTS / "diffusion_natural" / "cutin" / "dataset.npz",
    "evt_model": RESULTS / "highd_cutin_tail" / "evt" / "cutin_peak_evt_model.json",
    "evt_summary": RESULTS / "highd_cutin_tail" / "evt" / "cutin_peak_evt_summary.json",
    "exposure_summary": RESULTS / "highd_cutin_tail" / "exposure" / "highd_cutin_exposure_summary.json",
    "tail_condition_distribution": RESULTS
    / "highd_cutin_tail"
    / "contexts"
    / "scenario_condition_distribution.npz",
    "tail_contexts": RESULTS / "highd_cutin_tail" / "contexts" / "tail_contexts.npz",
    "tail_generated_scenarios": RESULTS
    / "highd_cutin_tail"
    / "generated"
    / "diffusion_generated_scenarios.npz",
    "tail_generated_summary": RESULTS
    / "highd_cutin_tail"
    / "generated"
    / "diffusion_generated_scenarios_summary.json",
    "tail_distribution_similarity_summary": RESULTS
    / "highd_cutin_tail"
    / "generated"
    / "figures"
    / "distribution_similarity_summary.json",
}


def rel(path: Path) -> str:
    return rel_path(path, ROOT)


def save_figure(fig: Any, path: Path, *, force: bool, dpi: int = PAPER_FIGURE_DPI) -> list[str]:
    return save_figure_to(fig, path, ROOT, force=force, dpi=dpi)


def _artifact_status(outputs: list[str]) -> str:
    if outputs and all("/logs/" in item for item in outputs):
        return "skipped"
    return "generated"


def _gpd_conditional_cdf(excess: np.ndarray, *, xi: float, beta: float) -> np.ndarray:
    z = np.maximum(np.asarray(excess, dtype=float), 0.0)
    beta = max(float(beta), 1.0e-12)
    if abs(float(xi)) < 1.0e-10:
        return 1.0 - np.exp(-z / beta)
    return 1.0 - np.power(np.maximum(1.0 + float(xi) * z / beta, 1.0e-300), -1.0 / float(xi))


def _gpd_conditional_pdf(excess: np.ndarray, *, xi: float, beta: float) -> np.ndarray:
    z = np.maximum(np.asarray(excess, dtype=float), 0.0)
    beta = max(float(beta), 1.0e-12)
    if abs(float(xi)) < 1.0e-10:
        return np.exp(-z / beta) / beta
    base = np.maximum(1.0 + float(xi) * z / beta, 1.0e-300)
    return np.power(base, -1.0 / float(xi) - 1.0) / beta


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
        curve = gpd_survival(
            survival_x,
            u=u,
            xi=xi_hat,
            beta=beta_hat,
            exceedance_rate=exceedance_rate,
        )
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


def _return_level_curve_for_distance(
    distances_km: np.ndarray,
    *,
    tail_peak_rate_per_km: float,
    u: float,
    xi: float,
    beta: float,
) -> np.ndarray:
    expected_tail_exceedances = np.asarray(distances_km, dtype=float) * float(tail_peak_rate_per_km)
    return np.asarray(
        [
            return_level_for_tail_exposure(
                expected_tail_exceedances=float(expected),
                u=float(u),
                xi=float(xi),
                beta=float(beta),
            )
            for expected in expected_tail_exceedances
        ],
        dtype=float,
    )


def _evt_scores_for_finite_values(model: GPDTailModel, values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    scores = np.full_like(arr, np.nan, dtype=float)
    finite = np.isfinite(arr)
    if np.any(finite):
        scores[finite] = np.asarray(model.score(arr[finite]), dtype=float)
    return scores


def _bootstrap_return_level_distance_band(
    values: np.ndarray,
    distances_km: np.ndarray,
    *,
    total_exposure_km: float,
    chosen_tail_count: int,
    samples: int = 300,
    random_seed: int = 20240615,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < max(int(chosen_tail_count) + 1, 10) or int(samples) <= 0:
        nan = np.full_like(distances_km, np.nan, dtype=float)
        return nan, nan

    rng = np.random.default_rng(int(random_seed))
    curves: list[np.ndarray] = []
    for _ in range(int(samples)):
        boot = np.sort(rng.choice(values, size=values.size, replace=True))
        k = min(max(5, int(chosen_tail_count)), boot.size - 1)
        u_hat = float(boot[boot.size - k - 1])
        excess = boot[boot > u_hat] - u_hat
        excess = excess[np.isfinite(excess) & (excess > 0.0)]
        if excess.size < 5:
            continue
        try:
            xi_hat, beta_hat = fit_gpd_excess(excess)
        except ValueError:
            continue
        rate_hat = float(excess.size / max(float(total_exposure_km), 1.0e-12))
        curve = _return_level_curve_for_distance(
            distances_km,
            tail_peak_rate_per_km=rate_hat,
            u=u_hat,
            xi=xi_hat,
            beta=beta_hat,
        )
        if np.any(np.isfinite(curve)):
            curves.append(curve)

    if not curves:
        nan = np.full_like(distances_km, np.nan, dtype=float)
        return nan, nan
    arr = np.asarray(curves, dtype=float)
    return np.nanpercentile(arr, 2.5, axis=0), np.nanpercentile(arr, 97.5, axis=0)


def _write_cutin_safety_threshold_inverse_calibration(
    evt_model: dict[str, Any],
    exposure: dict[str, Any],
    *,
    force: bool,
) -> list[str]:
    model = GPDTailModel.from_dict(evt_model)
    values = np.asarray(evt_model.get("calibration_values", []), dtype=float)
    values = values[np.isfinite(values)]
    human = exposure.get("human_calibrated_safety_threshold", {}) or {}
    u = fget(evt_model, "u")
    xi = fget(evt_model, "xi")
    beta = fget(evt_model, "beta")
    total_exposure_km = fget(exposure, "total_exposure_km", fget(exposure, "all_vehicle_km"))
    tail_rate_km = fget(
        exposure,
        "tail_peak_rate_per_km",
        fget(exposure, "tail_peak_rate_per_all_vehicle_km"),
    )
    target_km = fget(human, "target_return_period_km")
    target_level = fget(human, "target_level", fget(exposure, "collision_critical_level"))
    required = [
        u,
        xi,
        beta,
        total_exposure_km,
        tail_rate_km,
        target_km,
        target_level,
    ]
    if not (values.size and all(np.isfinite(np.asarray(required, dtype=float)))):
        skip = {"status": "skipped", "reason": "missing inputs for cut-in safety threshold inverse calibration"}
        path = LOGS / "exp2_skipped_cutin_safety_threshold_inverse_calibration.json"
        write_json(path, skip, force=force)
        return [rel(path)]

    u = float(u)
    xi = float(xi)
    beta = float(beta)
    total_exposure_km = float(total_exposure_km)
    tail_rate_km = float(tail_rate_km)
    target_km = float(target_km)
    target_level = float(target_level)

    tail_values = np.sort(values[values > u])
    if tail_values.size == 0 or tail_rate_km <= 0.0 or total_exposure_km <= 0.0:
        skip = {"status": "skipped", "reason": "missing positive tail exposure for inverse calibration"}
        path = LOGS / "exp2_skipped_cutin_safety_threshold_inverse_calibration.json"
        write_json(path, skip, force=force)
        return [rel(path)]

    ranks = np.arange(tail_values.size, 0, -1, dtype=float)
    empirical_return_km = total_exposure_km / ranks
    plot_min = max(10.0, float(np.nanmin(empirical_return_km)) * 0.85)
    plot_max = max(1.0e6, target_km * 20.0, float(np.nanmax(empirical_return_km)) * 1.7)
    distances_km = np.logspace(np.log10(plot_min), np.log10(plot_max), 420)
    return_levels = _return_level_curve_for_distance(
        distances_km,
        tail_peak_rate_per_km=tail_rate_km,
        u=u,
        xi=xi,
        beta=beta,
    )
    tail_scores = _evt_scores_for_finite_values(model, tail_values)
    return_level_scores = _evt_scores_for_finite_values(model, return_levels)
    target_score = float(model.score(target_level))
    level_low, level_high = _bootstrap_return_level_distance_band(
        values,
        distances_km,
        total_exposure_km=total_exposure_km,
        chosen_tail_count=tail_values.size,
    )
    level_low_scores = _evt_scores_for_finite_values(model, level_low)
    level_high_scores = _evt_scores_for_finite_values(model, level_high)

    plt = get_pyplot()
    with plt.rc_context(PAPER_PANEL_RC):
        fig, ax = plt.subplots(figsize=PAPER_SINGLE_PANEL_FIGSIZE)

        ax.scatter(
            empirical_return_km,
            tail_scores,
            facecolors="none",
            edgecolors=REAL_COLOR,
            linewidths=0.75,
            s=17,
            alpha=0.56,
            label=r"Empirical EVT scores",
            zorder=3,
        )
        ax.plot(
            distances_km,
            return_level_scores,
            color=GENERATED_COLOR,
            linewidth=1.9,
            label=r"GPD inverse $\gamma_{\mathrm{ci}}^\star$",
        )
        band_mask = np.isfinite(level_low_scores) & np.isfinite(level_high_scores)
        if np.any(band_mask):
            ax.fill_between(
                distances_km[band_mask],
                level_low_scores[band_mask],
                level_high_scores[band_mask],
                color=GENERATED_COLOR,
                alpha=0.18,
                linewidth=0.0,
                label="95% bootstrap CI",
            )
        ax.axvline(
            target_km,
            color=CRITICAL_COLOR,
            linestyle="--",
            linewidth=1.25,
            label=r"Selected $L^\star$",
        )
        ax.axhline(
            target_score,
            color=REFERENCE_COLOR,
            linestyle="-.",
            linewidth=1.35,
            label=r"Inferred $\gamma_{\mathrm{ci}}^\star$",
        )
        ax.scatter(
            [target_km],
            [target_score],
            color=CRITICAL_COLOR,
            edgecolors="white",
            linewidths=0.5,
            s=42,
            zorder=5,
        )
        ax.set_xscale("log")
        ax.set_xlabel(r"Target return mileage $L^\star$ (all-vehicle km)")
        ax.set_ylabel(r"EVT risk threshold $\gamma_{\mathrm{ci}}^\star$")
        ax.legend(frameon=False, loc="upper left")
        note_lines = [
            rf"$L^\star={target_km:,.0f}$ km",
            rf"$\gamma_{{\mathrm{{ci}}}}^\star={target_score:.3f}$",
        ]
        ax.text(
            0.985,
            0.045,
            "\n".join(note_lines),
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=PAPER_ANNOTATION_FONTSIZE,
            color=REFERENCE_COLOR,
            bbox=PAPER_NOTE_BBOX,
        )
        style_axes(ax)
        fig.tight_layout()
        outputs = save_figure(
            fig,
            FIGURES / "cutin_safety_threshold_inverse_calibration.png",
            force=force,
            dpi=PAPER_FIGURE_DPI,
        )
        plt.close(fig)
    return outputs


def _write_cutin_gpd_diagnostic_panel(
    evt_model: dict[str, Any],
    evt_summary: dict[str, Any],
    *,
    force: bool,
) -> list[str]:
    values = np.asarray(evt_model.get("calibration_values", []), dtype=float)
    values = values[np.isfinite(values)]
    candidates = evt_model.get("threshold_candidates", []) or []
    u = fget(evt_model, "u")
    xi = fget(evt_model, "xi")
    beta = fget(evt_model, "beta")
    lam = fget(evt_model, "exceedance_rate")
    if not (values.size and candidates and all(np.isfinite([u, xi, beta, lam]))):
        skip = {"status": "skipped", "reason": "missing EVT calibration values or threshold candidates"}
        path = LOGS / "exp2_skipped_cutin_gpd_diagnostic_panel.json"
        write_json(path, skip, force=force)
        return [rel(path)]

    u = float(u)
    xi = float(xi)
    beta = float(beta)
    lam = float(lam)
    tail_values = np.sort(values[values > u])
    excess = tail_values - u
    if excess.size == 0:
        skip = {"status": "skipped", "reason": "no exceedances above POT threshold"}
        path = LOGS / "exp2_skipped_cutin_gpd_diagnostic_panel.json"
        write_json(path, skip, force=force)
        return [rel(path)]

    cand = sorted(candidates, key=lambda row: float(row["u"]))
    cand_u = np.asarray([float(row["u"]) for row in cand], dtype=float)
    cand_xi = np.asarray([float(row["xi"]) for row in cand], dtype=float)
    cand_mod_scale = np.asarray([float(row["modified_scale"]) for row in cand], dtype=float)
    mean_excess = np.asarray(
        [np.mean(values[values > threshold] - threshold) for threshold in cand_u],
        dtype=float,
    )

    empirical_cdf = np.arange(1, excess.size + 1, dtype=float) / (excess.size + 1.0)
    gpd_quantiles = _gpd_conditional_ppf(empirical_cdf, xi=xi, beta=beta)
    gpd_cdf = _gpd_conditional_cdf(excess, xi=xi, beta=beta)

    plt = get_pyplot()
    with plt.rc_context(PAPER_PANEL_RC):
        fig, axes = plt.subplots(2, 3, figsize=PAPER_SIX_PANEL_FIGSIZE)
        axes = axes.ravel()

        panel_labels = ("a", "b", "c", "d", "e", "f")

        x_limit = max(float(np.quantile(values, 0.999)), u * 1.08, float(np.max(tail_values)))
        x_limit = min(x_limit, float(np.max(values)))
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
        density = _gpd_conditional_pdf(xs - u, xi=xi, beta=beta)
        axes[0].plot(
            xs,
            np.maximum(tail_count * bin_width * density, 1.0e-12),
            color=GENERATED_COLOR,
            linewidth=1.8,
            label="GPD fitted tail",
        )
        axes[0].axvline(u, color=REFERENCE_COLOR, linestyle="--", linewidth=1.2, label=r"selected $u_e$")
        axes[0].set_yscale("log")
        axes[0].set_xlabel(r"Original risk $Y_{\mathrm{cutin}}$")
        axes[0].set_ylabel("Count")
        axes[0].set_title("Peak distribution")
        axes[0].legend(frameon=False, loc="upper right")

        survival_x = np.sort(values[values >= u])
        survival_grid = np.linspace(u, float(np.max(survival_x)), 360)
        empirical_survival = np.maximum(_empirical_survival(values, survival_x), 1.0 / values.size)
        model_survival = np.maximum(
            gpd_survival(survival_grid, u=u, xi=xi, beta=beta, exceedance_rate=lam),
            1.0e-12,
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
            empirical_survival,
            color=REAL_COLOR,
            s=10,
            alpha=0.48,
            label="Empirical survival",
        )
        axes[1].plot(
            survival_grid,
            model_survival,
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
        axes[1].set_xlabel(r"Original risk $Y_{\mathrm{cutin}}$")
        axes[1].set_ylabel(r"$\Pr(Y_{\mathrm{cutin}}>y)$")
        axes[1].set_title("Tail survival")
        handles, labels = axes[1].get_legend_handles_labels()
        legend_order = [
            "Empirical survival",
            "GPD fitted survival",
            "95% bootstrap CI",
            r"POT threshold $u_e$",
        ]
        ordered = [
            (handles[labels.index(label)], label)
            for label in legend_order
            if label in labels
        ]
        axes[1].legend(
            [item[0] for item in ordered],
            [item[1] for item in ordered],
            frameon=False,
            loc="upper right",
        )

        axes[2].plot(cand_u, mean_excess, color=REAL_COLOR, linewidth=1.5)
        axes[2].axvline(u, color=REFERENCE_COLOR, linestyle="--", linewidth=1.2, label=r"selected $u_e$")
        axes[2].axvspan(u, float(np.max(cand_u)), color=GENERATED_COLOR, alpha=0.08)
        axes[2].set_xlabel(r"Threshold $u$")
        axes[2].set_ylabel(r"$E[Y_{\mathrm{cutin}}-u\mid Y_{\mathrm{cutin}}>u]$")
        axes[2].set_title("Mean residual life")
        axes[2].legend(frameon=False, loc="upper right")

        band_u, xi_low, xi_high, scale_low, scale_high = _bootstrap_threshold_stability_band(values, cand_u)
        ax_scale = axes[3].twinx()
        if band_u.size:
            axes[3].fill_between(
                band_u,
                xi_low,
                xi_high,
                color=REAL_COLOR,
                alpha=0.17,
                linewidth=0.0,
            )
            ax_scale.fill_between(
                band_u,
                scale_low,
                scale_high,
                color=GENERATED_COLOR,
                alpha=0.17,
                linewidth=0.0,
            )
        line_xi = axes[3].plot(cand_u, cand_xi, color=REAL_COLOR, linewidth=1.45, label=r"$\xi$")
        line_scale = ax_scale.plot(
            cand_u,
            cand_mod_scale,
            color=GENERATED_COLOR,
            linewidth=1.45,
            label=r"$\tilde{\sigma}$",
        )
        threshold_line = axes[3].axvline(
            u,
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
            loc="lower left",
        )

        max_q = float(max(np.max(excess), np.max(gpd_quantiles)))
        axes[4].scatter(gpd_quantiles, excess, color=REAL_COLOR, s=10, alpha=0.45)
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

        for label, ax in zip(panel_labels, axes):
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
        outputs = save_figure(
            fig,
            FIGURES / "cutin_gpd_diagnostic_panel.png",
            force=force,
            dpi=PAPER_FIGURE_DPI,
        )
        plt.close(fig)
    return outputs


def _condition_matrix_with_log_gap(
    values: np.ndarray,
    keys: list[str],
) -> tuple[np.ndarray, list[str]]:
    arr = np.asarray(values, dtype=np.float64).copy()
    out_keys = [str(key) for key in keys]
    if "initial_gap" in out_keys:
        idx = out_keys.index("initial_gap")
        arr[:, idx] = np.log(np.maximum(arr[:, idx], 1.0e-6))
        out_keys[idx] = "log_initial_gap"
    return arr, out_keys


def _matched_real_tail_arrays(
    *,
    event_ids: np.ndarray,
    tail_context_path: Path,
    dataset_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tail_context = np.load(tail_context_path, allow_pickle=True)
    data = np.load(dataset_path, allow_pickle=True)

    tail_by_event: dict[str, int] = {}
    synthetic_context = (
        tail_context["synthetic_context"].astype(np.int8)
        if "synthetic_context" in tail_context.files
        else np.zeros(len(tail_context["event_id"]), dtype=np.int8)
    )
    for idx, event_id in enumerate(tail_context["event_id"].tolist()):
        if int(synthetic_context[idx]) == 0:
            tail_by_event.setdefault(str(event_id), int(idx))

    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for event_id_raw in event_ids:
        event_id = str(event_id_raw)
        tail_idx = tail_by_event.get(event_id)
        if tail_idx is None:
            missing.append(event_id)
            continue
        row: dict[str, Any] = {"event_id": event_id}
        if "anchor_frame" in tail_context.files:
            row["anchor_frame"] = int(tail_context["anchor_frame"][tail_idx])
        rows.append(row)
    if not rows:
        raise RuntimeError("no EVT-tail cut-in event ids matched the tail context cache")
    if missing:
        raise RuntimeError(
            "some EVT-tail cut-in event ids were missing from the tail context cache: "
            f"{missing[:10]} (total={len(missing)})"
        )
    idx_arr = _cutin_dataset_indices_for_tail_rows(data, rows)
    initial = data["initial_states"][idx_arr].astype(np.float64)
    target = data["future_states"][idx_arr, :, 1, :].astype(np.float64)
    actions = data["actions"][idx_arr].astype(np.float64)
    return initial, target, actions


def _tail_lateral_displacement_from_start(
    initial_states: np.ndarray,
    target_trajectory: np.ndarray,
) -> np.ndarray:
    initial = np.asarray(initial_states, dtype=np.float64)
    target = np.asarray(target_trajectory, dtype=np.float64)
    return target[:, :, 1] - initial[:, 1, 1:2]


def _lane_change_direction(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    direction = np.sign(arr[:, -1])
    direction = np.where(direction == 0.0, 1.0, direction)
    return direction


def _valid_lateral_displacement_rows(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    valid = np.all(np.isfinite(arr), axis=1)
    valid &= np.nanmax(np.abs(arr), axis=1) <= 6.0
    if np.count_nonzero(valid) < max(5, int(arr.shape[0] * 0.5)):
        valid = np.all(np.isfinite(arr), axis=1)
    return valid


def _downsample_indices(size: int, max_count: int, seed: int) -> np.ndarray:
    if int(size) <= int(max_count):
        return np.arange(int(size), dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    return np.sort(rng.choice(int(size), size=int(max_count), replace=False))


def _hist_density_panel(
    ax: Any,
    arrays: list[np.ndarray],
    labels: list[str],
    colors: list[str],
    *,
    xlabel: str,
    title: str,
    bins: int = 28,
) -> None:
    clean = [np.asarray(values, dtype=float).reshape(-1) for values in arrays]
    clean = [values[np.isfinite(values)] for values in clean]
    pooled = np.concatenate([values for values in clean if values.size])
    lo, hi = np.nanpercentile(pooled, [0.5, 99.5])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(pooled)), float(np.nanmax(pooled))
    if hi <= lo:
        hi = lo + 1.0
    edges = np.linspace(float(lo), float(hi), int(bins) + 1)
    for values, label, color in zip(clean, labels, colors):
        values = values[(values >= edges[0]) & (values <= edges[-1])]
        ax.hist(
            values,
            bins=edges,
            density=True,
            color=color,
            alpha=0.34,
            label=label,
            linewidth=0.0,
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.set_title(title)
    if len(clean) >= 2:
        distances = _distribution_distances(clean[0], clean[1])
        ax.text(
            0.985,
            0.955,
            rf"$W_1$={distances['wasserstein']:.3g}"
            + "\n"
            + rf"$D_{{\mathrm{{KS}}}}$={distances['ks']:.3g}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=PAPER_ANNOTATION_FONTSIZE,
            bbox=PAPER_NOTE_BBOX,
        )


def _distribution_distances(real: np.ndarray, generated: np.ndarray) -> dict[str, float]:
    from scipy.stats import ks_2samp, wasserstein_distance

    a = np.asarray(real, dtype=np.float64).reshape(-1)
    b = np.asarray(generated, dtype=np.float64).reshape(-1)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return {"wasserstein": float("nan"), "ks": float("nan")}
    return {
        "wasserstein": float(wasserstein_distance(a, b)),
        "ks": float(ks_2samp(a, b).statistic),
    }


def _write_cutin_tail_diffusion_generalization_panel(*, force: bool) -> list[str]:
    required = [
        SOURCE_PATHS["tail_condition_distribution"],
        SOURCE_PATHS["tail_contexts"],
        SOURCE_PATHS["tail_generated_scenarios"],
        SOURCE_PATHS["cutin_diffusion_dataset"],
    ]
    missing = [rel(path) for path in required if not path.exists()]
    if missing:
        skip = {
            "status": "skipped",
            "reason": "missing inputs for cut-in tail diffusion generalization panel",
            "missing": missing,
        }
        path = LOGS / "exp3_skipped_cutin_tail_diffusion_generalization_panel.json"
        write_json(path, skip, force=force)
        return [rel(path)]

    context_data = np.load(SOURCE_PATHS["tail_condition_distribution"], allow_pickle=True)
    generated = np.load(SOURCE_PATHS["tail_generated_scenarios"], allow_pickle=True)

    real_keys = [str(item) for item in context_data["condition_keys"].tolist()]
    gen_keys = [str(item) for item in generated["condition_keys"].tolist()]
    if real_keys != gen_keys:
        skip = {
            "status": "skipped",
            "reason": "tail context and generated scenario condition keys differ",
            "tail_keys": real_keys,
            "generated_keys": gen_keys,
        }
        path = LOGS / "exp3_skipped_cutin_tail_diffusion_generalization_panel.json"
        write_json(path, skip, force=force)
        return [rel(path)]

    real_conditions, feature_keys = _condition_matrix_with_log_gap(
        context_data["scenario_conditions"],
        real_keys,
    )
    realized_conditions, _ = _condition_matrix_with_log_gap(
        generated["realized_scenario_conditions"],
        gen_keys,
    )
    real_initial, real_target, real_actions = _matched_real_tail_arrays(
        event_ids=context_data["event_id"],
        tail_context_path=SOURCE_PATHS["tail_contexts"],
        dataset_path=SOURCE_PATHS["cutin_diffusion_dataset"],
    )
    gen_initial = generated["initial_states"].astype(np.float64)
    gen_actions = generated["actions"].astype(np.float64)
    gen_target = generated["target_trajectory"].astype(np.float64)
    real_y = _tail_lateral_displacement_from_start(real_initial, real_target)
    gen_y = _tail_lateral_displacement_from_start(gen_initial, gen_target)
    horizon = min(real_y.shape[1], gen_y.shape[1])
    real_y = real_y[:, :horizon]
    gen_y = gen_y[:, :horizon]
    t = np.arange(horizon, dtype=float) * 0.04
    real_actions = real_actions[:, :horizon]
    gen_actions = gen_actions[:, :horizon]

    valid_cols = (
        np.all(np.isfinite(real_conditions), axis=0)
        & np.all(np.isfinite(realized_conditions), axis=0)
    )
    real_mu = np.nanmean(real_conditions[:, valid_cols], axis=0)
    real_std = np.nanstd(real_conditions[:, valid_cols], axis=0)
    scale_cols = real_std > 1.0e-8
    if np.count_nonzero(scale_cols) < 2:
        skip = {"status": "skipped", "reason": "not enough variable condition dimensions for PCA"}
        path = LOGS / "exp3_skipped_cutin_tail_diffusion_generalization_panel.json"
        write_json(path, skip, force=force)
        return [rel(path)]
    real_mu = real_mu[scale_cols]
    real_std = real_std[scale_cols]

    def project(values: np.ndarray, components: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        z = (values[:, valid_cols][:, scale_cols] - real_mu) / real_std
        if components is None:
            _, _s, vh = np.linalg.svd(z, full_matrices=False)
            components = vh[:2].T
        return z @ components, components

    real_proj, pca_components = project(real_conditions)
    realized_proj, _ = project(realized_conditions, pca_components)

    comparison_labels = ["EVT tail", "Diffusion"]
    condition_colors = [REAL_COLOR, GENERATED_COLOR]
    hist_specs = [
        ("log_initial_gap", "Real vs generated: initial gap"),
        ("initial_delta_vx", "Real vs generated: relative speed"),
    ]
    action_hist_specs = [
        (
            real_actions[:, :, 0],
            gen_actions[:, :, 0],
            "Real vs generated: longitudinal accel.",
            "Target longitudinal acceleration\n"
            r"$a_{x,\mathrm{tar}}^{t}$ (m/s$^2$)",
        ),
        (
            real_actions[:, :, 1],
            gen_actions[:, :, 1],
            "Real vs generated: lateral accel.",
            "Target lateral acceleration\n"
            r"$a_{y,\mathrm{tar}}^{t}$ (m/s$^2$)",
        ),
    ]

    plt = get_pyplot()
    with plt.rc_context(PAPER_PANEL_RC):
        fig, axes = plt.subplots(2, 3, figsize=PAPER_SIX_PANEL_FIGSIZE)
        axes = axes.ravel()

        real_idx = _downsample_indices(real_proj.shape[0], 500, 20240616)
        realized_idx = _downsample_indices(realized_proj.shape[0], 1400, 20240618)
        axes[0].scatter(
            realized_proj[realized_idx, 0],
            realized_proj[realized_idx, 1],
            s=7,
            color=GENERATED_COLOR,
            alpha=0.24,
            label="Diffusion",
            rasterized=True,
        )
        axes[0].scatter(
            real_proj[real_idx, 0],
            real_proj[real_idx, 1],
            s=12,
            color=REAL_COLOR,
            alpha=0.70,
            label="EVT tail",
            rasterized=True,
        )
        axes[0].set_xlabel(r"Scenario-condition PC1 of $o_{\mathrm{ci}}$")
        axes[0].set_ylabel(r"Scenario-condition PC2 of $o_{\mathrm{ci}}$")
        axes[0].set_title("Tail context similarity")
        axes[0].legend(frameon=False, loc="upper left")

        for ax, (feature, title) in zip(axes[1:3], hist_specs):
            idx = feature_keys.index(feature)
            _hist_density_panel(
                ax,
                [
                    real_conditions[:, idx],
                    realized_conditions[:, idx],
                ],
                comparison_labels,
                condition_colors,
                xlabel=descriptive_condition_label_for(feature),
                title=title,
            )
            ax.legend(frameon=False, loc="upper left")

        for ax, (real_action, gen_action, title, xlabel) in zip(
            (axes[3], axes[5]),
            action_hist_specs,
        ):
            _hist_density_panel(
                ax,
                [
                    real_action,
                    gen_action,
                ],
                comparison_labels,
                condition_colors,
                xlabel=xlabel,
                title=title,
            )
            ax.legend(frameon=False, loc="upper left")

        for values, color, label, seed in [
            (real_y[_valid_lateral_displacement_rows(real_y)], REAL_COLOR, "EVT tail", 20240619),
            (gen_y[_valid_lateral_displacement_rows(gen_y)], GENERATED_COLOR, "Diffusion", 20240620),
        ]:
            direction = _lane_change_direction(values)
            for direction_sign, direction_label, linestyle, seed_offset in (
                (1.0, r"$+\Delta y$", "-", 0),
                (-1.0, r"$-\Delta y$", "--", 97),
            ):
                group = values[direction == direction_sign]
                if group.size == 0:
                    continue
                idx = _downsample_indices(group.shape[0], 60, seed + seed_offset)
                axes[4].plot(
                    t,
                    group[idx].T,
                    color=color,
                    alpha=0.035,
                    linewidth=0.62,
                    linestyle=linestyle,
                    rasterized=True,
                )
                median = np.nanmedian(group, axis=0)
                lower = np.nanquantile(group, 0.25, axis=0)
                upper = np.nanquantile(group, 0.75, axis=0)
                axes[4].fill_between(
                    t,
                    lower,
                    upper,
                    color=color,
                    alpha=0.10,
                    linewidth=0.0,
                )
                axes[4].plot(
                    t,
                    median,
                    color=color,
                    linewidth=1.8,
                    linestyle=linestyle,
                    label=f"{label} {direction_label}",
                )
        axes[4].axhline(0.0, color=REFERENCE_COLOR, linestyle=":", linewidth=1.0, alpha=0.70)
        axes[4].set_xlabel(r"$t$ from anchor (s)")
        axes[4].set_ylabel("Signed lateral displacement (m)")
        axes[4].set_title("Direction-resolved cut-in trajectories")
        axes[4].legend(frameon=False, loc="lower right")
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
        fig.tight_layout(**PAPER_SIX_PANEL_LAYOUT)
        outputs = save_figure(
            fig,
            FIGURES / "cutin_tail_diffusion_generalization_panel.png",
            force=force,
            dpi=PAPER_FIGURE_DPI,
        )
        plt.close(fig)
    return outputs


def tail_diffusion_generalization_panel(
    manifest: dict[str, Any],
    *,
    force: bool,
) -> None:
    outputs = _write_cutin_tail_diffusion_generalization_panel(force=force)
    sources = [
        rel(path)
        for path in (
            SOURCE_PATHS["tail_condition_distribution"],
            SOURCE_PATHS["tail_contexts"],
            SOURCE_PATHS["tail_generated_scenarios"],
            SOURCE_PATHS["cutin_diffusion_dataset"],
        )
        if path.exists()
    ]
    record(
        manifest,
        "cutin_tail_diffusion_generalization_panel",
        status=_artifact_status(outputs),
        outputs=outputs,
        sources=sources,
        notes="tail diffusion generalization panel is rebuilt from existing EVT-tail contexts and generated scenarios",
    )


def evt_diagnostic_panels(manifest: dict[str, Any], data: dict[str, Any], *, force: bool) -> None:
    outputs: list[str] = []
    outputs.extend(
        _write_cutin_safety_threshold_inverse_calibration(
            data["evt_model"],
            data["exposure"],
            force=force,
        )
    )
    outputs.extend(
        _write_cutin_gpd_diagnostic_panel(
            data["evt_model"],
            data["evt_summary"],
            force=force,
        )
    )
    record(
        manifest,
        "cutin_evt_diagnostic_panels",
        status="generated",
        outputs=outputs,
    )


def _write_cutin_subset_level_score_histograms(*, force: bool) -> list[str]:
    path = SOURCE_PATHS["subset_samples"]
    if not path.exists():
        skip = {
            "status": "skipped",
            "reason": "missing cut-in subset sample file",
            "missing": rel(path),
        }
        log_path = LOGS / "subset_level_score_histograms_skipped.json"
        write_json(log_path, skip, force=force)
        return [rel(log_path)]

    samples = np.load(path, allow_pickle=True)
    scores = np.asarray(samples["scores"], dtype=np.float64)
    if scores.ndim != 2 or scores.shape[0] < 1:
        skip = {
            "status": "skipped",
            "reason": f"expected 2-D level x sample scores, got shape={scores.shape}",
        }
        log_path = LOGS / "subset_level_score_histograms_skipped.json"
        write_json(log_path, skip, force=force)
        return [rel(log_path)]

    failure_threshold = float(np.asarray(samples["failure_threshold"]).reshape(()))
    finite_by_level = [row[np.isfinite(row)] for row in scores]
    pooled = np.concatenate([row for row in finite_by_level if row.size])
    if pooled.size == 0:
        skip = {"status": "skipped", "reason": "subset scores contain no finite values"}
        log_path = LOGS / "subset_level_score_histograms_skipped.json"
        write_json(log_path, skip, force=force)
        return [rel(log_path)]

    x_min = min(0.0, float(np.nanmin(pooled)))
    x_hi_candidates = [
        float(np.nanmax(pooled)),
        failure_threshold * 1.08 if np.isfinite(failure_threshold) else float("nan"),
    ]
    x_max = max(value for value in x_hi_candidates if np.isfinite(value))
    if x_max <= x_min:
        x_max = x_min + 1.0
    edges = np.linspace(x_min, x_max, 34)

    plt = get_pyplot()
    with plt.rc_context(PAPER_PANEL_RC):
        fig, ax = plt.subplots(figsize=PAPER_SUBSET_HISTOGRAM_FIGSIZE)
        colors = [REAL_COLOR, GENERATED_COLOR, SAMPLED_COLOR, "#B279A2"]
        level_indices = list(range(scores.shape[0]))
        if len(level_indices) > len(colors):
            level_indices = [0, len(level_indices) // 2, len(level_indices) - 1]
        for order, level_idx in enumerate(level_indices):
            values = finite_by_level[level_idx]
            if values.size == 0:
                continue
            if level_idx == 0:
                label = "Level 0"
            elif level_idx == scores.shape[0] - 1:
                label = f"Final level ({level_idx})"
            else:
                label = f"Level {level_idx}"
            ax.hist(
                values,
                bins=edges,
                density=True,
                color=colors[order % len(colors)],
                alpha=0.34,
                label=label,
                linewidth=0.0,
            )
        y_top = ax.get_ylim()[1]
        if np.isfinite(failure_threshold):
            ax.axvspan(
                failure_threshold,
                x_max,
                color=CRITICAL_COLOR,
                alpha=0.055,
                linewidth=0.0,
                label=r"$R_{\mathrm{ci}}(X)\geq\gamma_{\mathrm{ci}}^\star$",
            )
            ax.axvline(
                failure_threshold,
                color=CRITICAL_COLOR,
                linestyle="--",
                linewidth=1.15,
                label=r"$\gamma_{\mathrm{ci}}^\star$",
            )
            ax.set_ylim(0.0, y_top)
        ax.set_xlim(x_min, x_max)
        ax.set_xlabel(r"EVT risk score $R_{\mathrm{ci}}(X)$")
        ax.set_ylabel("Density")
        ax.legend(frameon=False, loc="upper left")
        style_axes(ax)
        fig.tight_layout()
        outputs = save_figure(
            fig,
            FIGURES / "cutin_subset_level_score_histograms.png",
            force=force,
            dpi=PAPER_FIGURE_DPI,
        )
        plt.close(fig)
    return outputs


def subset_level_score_histograms(manifest: dict[str, Any], *, force: bool) -> None:
    outputs = _write_cutin_subset_level_score_histograms(force=force)
    record(
        manifest,
        "cutin_subset_level_score_histograms",
        status=_artifact_status(outputs),
        outputs=outputs,
        sources=[rel(SOURCE_PATHS["subset_samples"])],
        notes="level-wise risk-score histograms are built from stored cut-in subset samples",
    )


def write_readme(manifest: dict[str, Any], *, force: bool) -> None:
    write_experiment_readme(
        OUT / "CUTIN_EXPERIMENT_README.md",
        manifest,
        title="Cut-in Paper Experiments",
        description="This directory contains post-processed cut-in paper artifacts built from existing results only.",
        no_rerun_note="No cut-in diffusion training, EVT fitting, or subset simulation rerun was performed.",
        interpretation_notes=[
            "All paper figures use the shared TREAD paper style: 300 dpi export, Times-compatible serif text, and STIX/LaTeX-style math rendering.",
            "Main exposure denominator is `all_vehicle_km`.",
            "ADS intensity is `conditional exceedance probability x highD tail peak exposure rate`.",
            (
                "The probabilities are conditional on the highD cutin tail scenario-condition distribution, "
                "not unconditional road crash rates."
            ),
        ],
        force=force,
    )


def build(force: bool) -> dict[str, Any]:
    FIGURES.mkdir(parents=True, exist_ok=True)
    data = {
        "subset": read_json(SOURCE_PATHS["subset_summary"]),
        "evt_model": read_json(SOURCE_PATHS["evt_model"]),
        "evt_summary": read_json(SOURCE_PATHS["evt_summary"]),
        "exposure": read_json(SOURCE_PATHS["exposure_summary"]),
    }
    manifest = build_manifest(
        "cut_in",
        "results/build_cutin_paper_experiments.py",
        ROOT,
        SOURCE_PATHS,
    )
    evt_diagnostic_panels(manifest, data, force=force)
    tail_diffusion_generalization_panel(manifest, force=force)
    subset_level_score_histograms(manifest, force=force)
    write_json(OUT / "cutin_experiment_manifest.json", manifest, force=True)
    write_readme(manifest, force=force)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="overwrite existing paper experiment artifacts")
    args = parser.parse_args()
    manifest = build(force=args.force)
    print(f"Wrote {rel(OUT / 'cutin_experiment_manifest.json')}")
    print(f"Experiments: {len(manifest['experiments'])}")


if __name__ == "__main__":
    main()
