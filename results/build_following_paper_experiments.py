"""Build car-following paper figure artifacts from existing results.

This post-processing script writes car-following paper figures, plus their
manifest and README. It does not generate following tables, training outputs,
EVT fits, or subset-simulation outputs.
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

from process_highD.src.evt_diagnostics import write_gpd_diagnostic_panel
from process_highD.src.following_tail_generation import (
    _load_real_following_tail_lead_trajectories,
)
from tools.evt import fit_gpd_excess, load_evt_model, return_level_for_tail_exposure
from tools.plot_style import (
    build_manifest,
    CRITICAL_COLOR,
    fget,
    GENERATED_COLOR,
    PAPER_ANNOTATION_FONTSIZE,
    PAPER_FIGURE_DPI,
    PAPER_NOTE_BBOX,
    PAPER_PANEL_LABELSIZE,
    PAPER_PANEL_RC,
    PAPER_PROFILE_FIGSIZE,
    PAPER_PROFILE_LABELSIZE,
    PAPER_PROFILE_RC,
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
OUT = RESULTS / "paper_experiments" / "following"
FIGURES = OUT

SOURCE_PATHS = {
    "subset_samples": ROOT
    / "IDM_subset"
    / "results"
    / "following"
    / "latent_subset_samples.npz",
    "evt_model": RESULTS / "highd_following_tail" / "evt" / "longitudinal_peak_evt_model.json",
    "exposure_summary": RESULTS
    / "highd_following_tail"
    / "exposure"
    / "highd_exposure_summary.json",
    "tail_condition_distribution": RESULTS
    / "highd_following_tail"
    / "contexts"
    / "scenario_condition_distribution.npz",
    "tail_contexts": RESULTS / "highd_following_tail" / "contexts" / "tail_contexts.npz",
    "tail_generated_scenarios": RESULTS
    / "highd_following_tail"
    / "generated"
    / "diffusion_generated_scenarios.npz",
    "following_segment_cache": RESULTS / "highd_events" / "following_event_segments.npz",
}


def rel(path: Path) -> str:
    return rel_path(path, ROOT)


def save_figure(fig: Any, path: Path, *, force: bool, dpi: int = PAPER_FIGURE_DPI) -> list[str]:
    return save_figure_to(fig, path, ROOT, force=force, dpi=dpi)


def _artifact_status(outputs: list[str]) -> str:
    if outputs and all("/logs/" in item for item in outputs):
        return "skipped"
    return "generated"


def _return_level_curve_for_distance(
    distances_km: np.ndarray,
    *,
    tail_peak_rate_per_km: float,
    u: float,
    xi: float,
    beta: float,
) -> np.ndarray:
    distances = np.asarray(distances_km, dtype=float)
    out = np.full_like(distances, np.nan, dtype=float)
    expected_tail = np.maximum(distances * float(tail_peak_rate_per_km), 0.0)
    valid = expected_tail > 1.0
    for idx in np.where(valid)[0]:
        out[idx] = return_level_for_tail_exposure(
            expected_tail_exceedances=float(expected_tail[idx]),
            u=float(u),
            xi=float(xi),
            beta=float(beta),
        )
    return out


def _evt_scores_for_finite_values(model: Any, values: np.ndarray) -> np.ndarray:
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
    n_boot: int = 240,
    seed: int = 20250613,
) -> tuple[np.ndarray, np.ndarray]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0 or chosen_tail_count < 5 or total_exposure_km <= 0.0:
        nan = np.full_like(distances_km, np.nan, dtype=float)
        return nan, nan

    rng = np.random.default_rng(seed)
    q = max(0.0, min(1.0, 1.0 - float(chosen_tail_count) / float(finite.size)))
    curves: list[np.ndarray] = []
    for _ in range(int(n_boot)):
        sample = rng.choice(finite, size=finite.size, replace=True)
        u_hat = float(np.quantile(sample, q))
        excess = sample[sample > u_hat] - u_hat
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


def _downsample_indices(size: int, max_count: int, seed: int) -> np.ndarray:
    if int(size) <= int(max_count):
        return np.arange(int(size), dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    return np.sort(rng.choice(int(size), size=int(max_count), replace=False))


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


def following_gpd_diagnostic_panel(
    manifest: dict[str, Any],
    *,
    force: bool,
) -> None:
    evt_model_path = SOURCE_PATHS["evt_model"]
    outputs: list[str] = []
    if evt_model_path.exists():
        model = load_evt_model(evt_model_path)
        panel_paths = write_gpd_diagnostic_panel(
            FIGURES,
            model=model,
            values=model.calibration_values,
            risk_variable="Y_long",
            output_filename="following_gpd_diagnostic_panel.png",
            output_key="following_gpd_diagnostic_panel",
            force=force,
            max_plot_value=10.0,
            stability_legend_loc="upper left",
            stability_threshold_ymax=0.55,
        )
        outputs.extend(rel(Path(path)) for path in panel_paths.values())
        record(
            manifest,
            "following_gpd_diagnostic_panel",
            status="generated",
            outputs=outputs,
            sources=[rel(evt_model_path)],
            notes="The display range is truncated at Y_long = 10 for readability.",
        )
    else:
        record(
            manifest,
            "following_gpd_diagnostic_panel",
            status="skipped",
            outputs=[],
            skipped_reason=f"missing input file: {rel(evt_model_path)}",
        )


def following_safety_threshold_inverse_calibration(
    manifest: dict[str, Any],
    *,
    force: bool,
) -> None:
    evt_model_path = SOURCE_PATHS["evt_model"]
    exposure_path = SOURCE_PATHS["exposure_summary"]
    if not evt_model_path.exists() or not exposure_path.exists():
        missing = [rel(path) for path in (evt_model_path, exposure_path) if not path.exists()]
        record(
            manifest,
            "following_safety_threshold_inverse_calibration",
            status="skipped",
            outputs=[],
            skipped_reason=f"missing input file(s): {', '.join(missing)}",
        )
        return

    model = load_evt_model(evt_model_path)
    exposure = read_json(exposure_path)
    selected = exposure.get("human_calibrated_safety_threshold", {}) or {}

    values = np.asarray(model.calibration_values, dtype=float)
    values = values[np.isfinite(values)]
    u = float(model.u)
    xi = float(model.xi)
    beta = float(model.beta)
    total_exposure_km = float(
        fget(exposure, "total_exposure_km", fget(exposure, "all_vehicle_km"))
    )
    tail_rate_km = float(
        fget(
            exposure,
            "tail_peak_rate_per_km",
            fget(exposure, "tail_peak_rate_per_all_vehicle_km"),
        )
    )
    target_km = float(
        selected.get(
            "selected_return_period_km",
            selected.get(
                "target_return_period_km",
                exposure.get("collision_critical_reference_km", float("nan")),
            ),
        )
    )
    target_level = float(
        selected.get(
            "selected_level",
            selected.get("target_level", exposure.get("collision_critical_level", float("nan"))),
        )
    )
    required = [u, xi, beta, total_exposure_km, tail_rate_km, target_km, target_level]
    if not (values.size and all(np.isfinite(np.asarray(required, dtype=float)))):
        record(
            manifest,
            "following_safety_threshold_inverse_calibration",
            status="skipped",
            outputs=[],
            skipped_reason="missing inputs for following safety threshold inverse calibration",
            sources=[rel(evt_model_path), rel(exposure_path)],
        )
        return

    tail_values = np.sort(values[values > u])
    if tail_values.size == 0 or tail_rate_km <= 0.0 or total_exposure_km <= 0.0:
        record(
            manifest,
            "following_safety_threshold_inverse_calibration",
            status="skipped",
            outputs=[],
            skipped_reason="missing positive tail exposure for inverse calibration",
            sources=[rel(evt_model_path), rel(exposure_path)],
        )
        return

    ranks = np.arange(tail_values.size, 0, -1, dtype=float)
    empirical_return_km = total_exposure_km / ranks
    plot_min = max(10.0, float(np.nanmin(empirical_return_km)) * 0.85)
    plot_max = max(target_km * 2.2, float(np.nanmax(empirical_return_km)) * 1.35)
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
            label=r"GPD inverse $\gamma_{\mathrm{cf}}^\star$",
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
            label=r"Inferred $\gamma_{\mathrm{cf}}^\star$",
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
        ax.set_ylabel(r"EVT risk threshold $\gamma_{\mathrm{cf}}^\star$")
        ax.legend(frameon=False, loc="upper left")
        note_lines = [
            rf"$L^\star={target_km:,.0f}$ km",
            rf"$\gamma_{{\mathrm{{cf}}}}^\star={target_score:.3f}$",
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
            FIGURES / "following_safety_threshold_inverse_calibration.png",
            force=force,
            dpi=PAPER_FIGURE_DPI,
        )
        plt.close(fig)

    record(
        manifest,
        "following_safety_threshold_inverse_calibration",
        status="generated",
        outputs=outputs,
        sources=[rel(evt_model_path), rel(exposure_path)],
        notes=(
            "The selected car-following safety threshold is the "
            f"{target_km:,.0f} km all-vehicle return level."
        ),
    )


def following_tail_diffusion_generalization_panel(
    manifest: dict[str, Any],
    *,
    force: bool,
) -> None:
    required = [
        SOURCE_PATHS["tail_condition_distribution"],
        SOURCE_PATHS["tail_contexts"],
        SOURCE_PATHS["tail_generated_scenarios"],
        SOURCE_PATHS["following_segment_cache"],
    ]
    missing = [rel(path) for path in required if not path.exists()]
    experiment_key = "following_tail_diffusion_generalization_panel"
    if missing:
        record(
            manifest,
            experiment_key,
            status="skipped",
            outputs=[],
            sources=[rel(path) for path in required if path.exists()],
            skipped_reason=f"missing input file(s): {', '.join(missing)}",
        )
        return

    context_data = np.load(SOURCE_PATHS["tail_condition_distribution"], allow_pickle=True)
    generated = np.load(SOURCE_PATHS["tail_generated_scenarios"], allow_pickle=True)

    real_keys = [str(item) for item in context_data["condition_keys"].tolist()]
    gen_keys = [str(item) for item in generated["condition_keys"].tolist()]
    if real_keys != gen_keys:
        record(
            manifest,
            experiment_key,
            status="skipped",
            outputs=[],
            sources=[rel(path) for path in required],
            skipped_reason="tail context and generated scenario condition keys differ",
        )
        return

    real_conditions, feature_keys = _condition_matrix_with_log_gap(
        context_data["scenario_conditions"],
        real_keys,
    )
    generated_conditions, _ = _condition_matrix_with_log_gap(
        generated["scenario_conditions"],
        gen_keys,
    )
    generated_initial = np.asarray(generated["initial_states"], dtype=np.float64)
    generated_lead = np.asarray(generated["lead_trajectory"], dtype=np.float64)
    horizon = int(generated_lead.shape[1])
    real_initial, real_lead, _real_aligned_conditions = (
        _load_real_following_tail_lead_trajectories(
            tail_context_path=SOURCE_PATHS["tail_contexts"],
            segment_cache_path=SOURCE_PATHS["following_segment_cache"],
            horizon_steps=horizon,
        )
    )

    valid_cols = (
        np.all(np.isfinite(real_conditions), axis=0)
        & np.all(np.isfinite(generated_conditions), axis=0)
    )
    real_mu = np.nanmean(real_conditions[:, valid_cols], axis=0)
    real_std = np.nanstd(real_conditions[:, valid_cols], axis=0)
    scale_cols = real_std > 1.0e-8
    if np.count_nonzero(scale_cols) < 2:
        record(
            manifest,
            experiment_key,
            status="skipped",
            outputs=[],
            sources=[rel(path) for path in required],
            skipped_reason="not enough variable condition dimensions for PCA",
        )
        return
    real_mu = real_mu[scale_cols]
    real_std = real_std[scale_cols]

    def project(
        values: np.ndarray,
        components: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        z = (values[:, valid_cols][:, scale_cols] - real_mu) / real_std
        if components is None:
            _, _s, vh = np.linalg.svd(z, full_matrices=False)
            components = vh[:2].T
        return z @ components, components

    real_proj, pca_components = project(real_conditions)
    generated_proj, _ = project(generated_conditions, pca_components)

    comparison_labels = ["EVT tail", "Diffusion"]
    comparison_colors = [REAL_COLOR, GENERATED_COLOR]
    t = np.arange(horizon, dtype=float) * 0.04
    real_displacement = real_lead[:, :, 0] - real_initial[:, None, 1, 0]
    generated_displacement = (
        generated_lead[:, :, 0] - generated_initial[:, None, 1, 0]
    )

    plt = get_pyplot()
    with plt.rc_context(PAPER_PANEL_RC):
        fig, axes = plt.subplots(2, 3, figsize=PAPER_SIX_PANEL_FIGSIZE)
        axes = axes.ravel()

        real_idx = _downsample_indices(real_proj.shape[0], 500, 20250614)
        gen_idx = _downsample_indices(generated_proj.shape[0], 1400, 20250615)
        axes[0].scatter(
            generated_proj[gen_idx, 0],
            generated_proj[gen_idx, 1],
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
        axes[0].set_xlabel(r"Scenario-condition PC1 of $o_{\mathrm{cf}}$")
        axes[0].set_ylabel(r"Scenario-condition PC2 of $o_{\mathrm{cf}}$")
        axes[0].set_title("Tail context similarity")
        axes[0].legend(frameon=False, loc="upper left")

        for ax, (feature, title) in zip(
            axes[1:3],
            (
                ("log_initial_gap", "Real vs generated: initial gap"),
                ("initial_delta_v", "Real vs generated: relative speed"),
            ),
        ):
            idx = feature_keys.index(feature)
            _hist_density_panel(
                ax,
                [real_conditions[:, idx], generated_conditions[:, idx]],
                comparison_labels,
                comparison_colors,
                xlabel=descriptive_condition_label_for(feature),
                title=title,
            )
            ax.legend(frameon=False, loc="upper left")

        _hist_density_panel(
            axes[3],
            [real_lead[:, :, 4], generated_lead[:, :, 4]],
            comparison_labels,
            comparison_colors,
            xlabel=(
                "Lead longitudinal acceleration\n"
                r"$a_{x,\mathrm{tar}}^{t}$ (m/s$^2$)"
            ),
            title="Real vs generated: lead acceleration",
        )
        axes[3].legend(frameon=False, loc="upper left")

        for values, color, label, seed in (
            (real_displacement, REAL_COLOR, "EVT tail", 20250616),
            (generated_displacement, GENERATED_COLOR, "Diffusion", 20250617),
        ):
            valid = np.all(np.isfinite(values), axis=1)
            group = values[valid]
            if group.size == 0:
                continue
            idx = _downsample_indices(group.shape[0], 80, seed)
            axes[4].plot(
                t,
                group[idx].T,
                color=color,
                alpha=0.035,
                linewidth=0.62,
                rasterized=True,
            )
            median = np.nanmedian(group, axis=0)
            lower = np.nanquantile(group, 0.25, axis=0)
            upper = np.nanquantile(group, 0.75, axis=0)
            axes[4].fill_between(t, lower, upper, color=color, alpha=0.10, linewidth=0.0)
            axes[4].plot(t, median, color=color, linewidth=1.8, label=label)
        axes[4].set_xlabel(r"$t$ from anchor (s)")
        axes[4].set_ylabel(r"$x_{\mathrm{tar}}(t)-x_{\mathrm{tar}}^{0}$ (m)")
        axes[4].set_title("Following lead trajectories")
        axes[4].legend(frameon=False, loc="upper left")

        _hist_density_panel(
            axes[5],
            [
                real_conditions[:, feature_keys.index("lead_braking_duration")],
                generated_conditions[:, feature_keys.index("lead_braking_duration")],
            ],
            comparison_labels,
            comparison_colors,
            xlabel=descriptive_condition_label_for("lead_braking_duration"),
            title="Real vs generated: lead braking time",
        )
        axes[5].legend(frameon=False, loc="upper left")

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
            FIGURES / "following_tail_diffusion_generalization_panel.png",
            force=force,
            dpi=PAPER_FIGURE_DPI,
        )
        plt.close(fig)

    record(
        manifest,
        experiment_key,
        status="generated",
        outputs=outputs,
        sources=[rel(path) for path in required],
        notes=(
            "tail diffusion generalization panel is rebuilt from existing "
            "following EVT-tail contexts, generated lead trajectories, and "
            "the same lead-braking-duration condition used by process_highD"
        ),
    )


def _smooth_time_profiles(values: np.ndarray, *, window: int = 7) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if int(window) <= 1 or arr.shape[1] < int(window):
        return arr
    kernel = np.full(int(window), 1.0 / float(window), dtype=np.float64)
    pad_left = int(window) // 2
    pad_right = int(window) - 1 - pad_left
    padded = np.pad(arr, ((0, 0), (pad_left, pad_right)), mode="edge")
    return np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="valid"), 1, padded)


def _nearest_unique_profile(
    selected: set[int],
    score: np.ndarray,
    *,
    candidate_mask: np.ndarray | None = None,
) -> int | None:
    values = np.asarray(score, dtype=np.float64)
    valid = np.isfinite(values)
    if candidate_mask is not None:
        valid &= np.asarray(candidate_mask, dtype=bool)
    for idx in selected:
        if 0 <= int(idx) < valid.size:
            valid[int(idx)] = False
    if not np.any(valid):
        return None
    valid_indices = np.where(valid)[0]
    return int(valid_indices[np.nanargmin(values[valid_indices])])


def following_tail_diffusion_acceleration_profiles(
    manifest: dict[str, Any],
    *,
    force: bool,
) -> None:
    experiment_key = "following_tail_diffusion_acceleration_profiles"
    path = SOURCE_PATHS["tail_generated_scenarios"]
    if not path.exists():
        record(
            manifest,
            experiment_key,
            status="skipped",
            skipped_reason=f"missing source file: {rel(path)}",
        )
        return

    generated = np.load(path, allow_pickle=True)
    if "acceleration" in generated.files:
        generated_ax = np.asarray(generated["acceleration"], dtype=np.float64)
    elif "lead_trajectory" in generated.files:
        generated_ax = np.asarray(generated["lead_trajectory"], dtype=np.float64)[:, :, 4]
    else:
        record(
            manifest,
            experiment_key,
            status="skipped",
            sources=[rel(path)],
            skipped_reason="generated following scenarios do not contain acceleration profiles",
        )
        return

    if generated_ax.ndim != 2 or generated_ax.shape[0] < 10 or generated_ax.shape[1] < 5:
        record(
            manifest,
            experiment_key,
            status="skipped",
            sources=[rel(path)],
            skipped_reason=f"unexpected acceleration array shape: {generated_ax.shape}",
        )
        return

    valid_rows = np.all(np.isfinite(generated_ax), axis=1)
    generated_ax = generated_ax[valid_rows]
    if generated_ax.shape[0] < 10:
        record(
            manifest,
            experiment_key,
            status="skipped",
            sources=[rel(path)],
            skipped_reason="too few finite generated acceleration profiles",
        )
        return

    dt = 0.04
    t = np.arange(generated_ax.shape[1], dtype=np.float64) * dt
    display_ax = _smooth_time_profiles(generated_ax, window=7)
    lower = np.nanpercentile(display_ax, 5.0, axis=0)
    upper = np.nanpercentile(display_ax, 95.0, axis=0)
    median = np.nanmedian(display_ax, axis=0)

    min_ax = np.nanmin(display_ax, axis=1)
    mean_abs_ax = np.nanmean(np.abs(display_ax), axis=1)
    braking_impulse = -np.trapz(np.minimum(display_ax, 0.0), t, axis=1)
    terminal_ax = display_ax[:, -1]
    recovery = terminal_ax - min_ax
    selected: set[int] = set()

    profile_specs: list[tuple[str, int, str, float]] = []

    idx = _nearest_unique_profile(
        selected,
        mean_abs_ax + 0.35 * np.abs(terminal_ax) + 0.25 * np.abs(np.nanmean(display_ax, axis=1)),
    )
    if idx is not None:
        selected.add(idx)
        profile_specs.append(("Near-zero response", idx, "-", 2.0))

    target = np.nanpercentile(braking_impulse, 40.0)
    idx = _nearest_unique_profile(
        selected,
        np.abs(braking_impulse - target) + 0.15 * np.maximum(-terminal_ax, 0.0),
        candidate_mask=braking_impulse > 0.03,
    )
    if idx is not None:
        selected.add(idx)
        profile_specs.append(("Mild braking", idx, ":", 1.9))

    idx = _nearest_unique_profile(
        selected,
        -recovery + 0.15 * np.maximum(terminal_ax, 0.0),
        candidate_mask=(min_ax < -0.35) & (recovery > 0.20),
    )
    if idx is not None:
        selected.add(idx)
        profile_specs.append(("Brake then recover", idx, "-.", 1.8))

    target = np.nanpercentile(min_ax, 15.0)
    idx = _nearest_unique_profile(
        selected,
        np.abs(min_ax - target) + 0.10 * np.abs(terminal_ax - target),
        candidate_mask=min_ax < -0.55,
    )
    if idx is not None:
        selected.add(idx)
        profile_specs.append(("Strong braking", idx, "--", 1.8))

    target = np.nanpercentile(terminal_ax, 8.0)
    idx = _nearest_unique_profile(
        selected,
        np.abs(terminal_ax - target) + 0.12 * np.abs(min_ax - target),
        candidate_mask=terminal_ax < -0.35,
    )
    if idx is not None:
        selected.add(idx)
        profile_specs.append(("Sustained braking", idx, (0, (5.0, 2.2)), 1.8))

    plt = get_pyplot()
    with plt.rc_context(PAPER_PROFILE_RC):
        fig, ax = plt.subplots(figsize=PAPER_PROFILE_FIGSIZE)
        band_color = "#7DB7E8"
        line_color = "#1454D9"
        ax.fill_between(
            t,
            lower,
            upper,
            color=band_color,
            alpha=0.20,
            linewidth=0.0,
            label="5-95% diffusion envelope",
            zorder=1,
        )
        ax.plot(t, median, color=line_color, linewidth=2.2, alpha=0.96, label="Diffusion median")

        label_x = float(t[-1] + 0.16)
        used_label_y: list[float] = []
        for label, idx, linestyle, linewidth in profile_specs:
            y = display_ax[idx]
            ax.plot(
                t,
                y,
                color=line_color,
                linestyle=linestyle,
                linewidth=linewidth,
                alpha=0.78,
                zorder=3,
            )
            y_end = float(y[-1])
            while any(abs(y_end - item) < 0.30 for item in used_label_y):
                y_end += 0.30
            used_label_y.append(y_end)
            ax.annotate(
                label,
                xy=(float(t[-1]), float(y[-1])),
                xytext=(label_x, y_end),
                ha="left",
                va="center",
                color=line_color,
                fontsize=PAPER_PROFILE_LABELSIZE,
                arrowprops={
                    "arrowstyle": "-",
                    "color": line_color,
                    "linewidth": 0.75,
                    "shrinkA": 0.0,
                    "shrinkB": 0.0,
                },
                clip_on=False,
            )

        y_min = float(np.nanpercentile(display_ax, 1.0))
        y_max = float(np.nanpercentile(display_ax, 99.0))
        y_min = min(y_min, *(float(display_ax[idx].min()) for _, idx, _, _ in profile_specs), -0.5)
        y_max = max(y_max, *(float(display_ax[idx].max()) for _, idx, _, _ in profile_specs), 0.25)
        pad = 0.12 * max(y_max - y_min, 1.0)
        ax.set_xlim(float(t[0]), float(t[-1] + 1.35))
        ax.set_ylim(y_min - pad, y_max + pad)
        ax.set_xlabel(r"$t$ from anchor (s)")
        ax.set_ylabel(
            r"Target longitudinal acceleration $a_{x,\mathrm{tar}}^{t}$ (m/s$^2$)"
        )
        ax.legend(frameon=False, loc="upper left")
        style_axes(ax)
        fig.tight_layout()
        outputs = save_figure(
            fig,
            FIGURES / "following_tail_diffusion_acceleration_profiles.png",
            force=force,
            dpi=PAPER_FIGURE_DPI,
        )
        plt.close(fig)

    record(
        manifest,
        experiment_key,
        status="generated",
        outputs=outputs,
        sources=[rel(path)],
        notes=(
            "single-panel summary of diffusion-generated following long-tail "
            "lead-vehicle longitudinal acceleration profiles"
        ),
    )


def following_subset_level_score_histograms(
    manifest: dict[str, Any],
    *,
    force: bool,
) -> None:
    path = SOURCE_PATHS["subset_samples"]
    experiment_key = "following_subset_level_score_histograms"
    if not path.exists():
        record(
            manifest,
            experiment_key,
            status="skipped",
            outputs=[],
            sources=[],
            skipped_reason=f"missing input file: {rel(path)}",
        )
        return

    samples = np.load(path, allow_pickle=True)
    scores = np.asarray(samples["scores"], dtype=np.float64)
    if scores.ndim != 2 or scores.shape[0] < 1:
        record(
            manifest,
            experiment_key,
            status="skipped",
            outputs=[],
            sources=[rel(path)],
            skipped_reason=f"expected 2-D level x sample scores, got shape={scores.shape}",
        )
        return

    failure_threshold = float(np.asarray(samples["failure_threshold"]).reshape(()))
    finite_by_level = [row[np.isfinite(row)] for row in scores]
    pooled = np.concatenate([row for row in finite_by_level if row.size])
    if pooled.size == 0:
        record(
            manifest,
            experiment_key,
            status="skipped",
            outputs=[],
            sources=[rel(path)],
            skipped_reason="subset scores contain no finite values",
        )
        return

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
                label=r"$R_{\mathrm{cf}}(X)\geq\gamma_{\mathrm{cf}}^\star$",
            )
            ax.axvline(
                failure_threshold,
                color=CRITICAL_COLOR,
                linestyle="--",
                linewidth=1.15,
                label=r"$\gamma_{\mathrm{cf}}^\star$",
            )
            ax.set_ylim(0.0, y_top)
        ax.set_xlim(x_min, x_max)
        ax.set_xlabel(r"EVT risk score $R_{\mathrm{cf}}(X)$")
        ax.set_ylabel("Density")
        ax.legend(frameon=False, loc="upper left")
        style_axes(ax)
        fig.tight_layout()
        outputs = save_figure(
            fig,
            FIGURES / "following_subset_level_score_histograms.png",
            force=force,
            dpi=PAPER_FIGURE_DPI,
        )
        plt.close(fig)

    record(
        manifest,
        experiment_key,
        status="generated",
        outputs=outputs,
        sources=[rel(path)],
        notes="level-wise risk-score histograms are built from stored following subset samples",
    )


def write_readme(manifest: dict[str, Any], *, force: bool) -> None:
    write_experiment_readme(
        OUT / "FOLLOWING_EXPERIMENT_README.md",
        manifest,
        title="Following Paper Figures",
        description=(
            "This directory contains car-following paper figures built from "
            "existing highD following EVT, exposure, diffusion, Monte Carlo, "
            "and subset-simulation results."
        ),
        no_rerun_note="No following diffusion training, EVT fitting, subset simulation, or tables were generated.",
        interpretation_notes=[
            "The following paper figures are generated directly in this directory; no `figures/` subdirectory is used.",
            "All paper figures use the shared TREAD paper style: 300 dpi export, Times-compatible serif text, and STIX/LaTeX-style math rendering.",
            "The panel shows the fitted POT/GPD tail diagnostics with the plotting range capped at `Y_long = 10`.",
            "The inverse calibration figure marks the selected 300 km all-vehicle return-level threshold from the exposure summary.",
            "The tail diffusion generalization panel compares empirical following EVT-tail contexts with generated lead trajectories; panel f uses the `lead_braking_duration` scenario-condition distribution used by `process_highD`.",
            "The acceleration-profile figure summarizes diffusion-generated long-tail lead-vehicle acceleration traces with a 5-95% envelope and representative braking modes.",
            "The subset level histogram shows how subset simulation concentrates mass toward the calibrated EVT risk threshold.",
        ],
        force=force,
    )


def build(force: bool) -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(
        "following",
        "results/build_following_paper_experiments.py",
        ROOT,
        SOURCE_PATHS,
    )
    following_gpd_diagnostic_panel(manifest, force=force)
    following_safety_threshold_inverse_calibration(manifest, force=force)
    following_tail_diffusion_generalization_panel(manifest, force=force)
    following_tail_diffusion_acceleration_profiles(manifest, force=force)
    following_subset_level_score_histograms(manifest, force=force)
    write_json(OUT / "following_experiment_manifest.json", manifest, force=True)
    write_readme(manifest, force=force)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="overwrite existing paper figure artifacts")
    args = parser.parse_args()
    manifest = build(force=args.force)
    print(f"Wrote {rel(OUT / 'following_experiment_manifest.json')}")
    print(f"Experiments: {len(manifest['experiments'])}")


if __name__ == "__main__":
    main()
