"""Human-calibrated EVT safety thresholds from all-vehicle-km return levels."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from tools.evt import (
    fit_gpd_excess,
    gpd_conditional_survival,
    return_level_for_tail_exposure,
)
from tools.highd_exposure import KM_PER_MILE
from tools.plot_style import (
    CRITICAL_COLOR,
    GENERATED_COLOR,
    REAL_COLOR,
    REFERENCE_COLOR,
    get_pyplot,
    style_axes,
)


def is_human_threshold_enabled(peak_cfg: dict[str, Any]) -> bool:
    cfg = dict(peak_cfg.get("human_safety_threshold", {}))
    return bool(cfg.get("enabled", False))


def target_return_period_km(peak_cfg: dict[str, Any]) -> float:
    cfg = dict(peak_cfg.get("human_safety_threshold", {}))
    if "target_return_period_km" in cfg:
        target = float(cfg["target_return_period_km"])
    else:
        target = float(cfg.get("target_return_period_miles", 1000.0)) * KM_PER_MILE
    if target <= 0.0:
        raise ValueError("human_safety_threshold.target_return_period_km must be positive")
    return target


def km_return_level_threshold(
    *,
    model: Any,
    tail_peak_rate_per_km: float,
    tail_peak_rate_per_hour: float,
    target_return_km: float,
) -> dict[str, float | str]:
    """Return x* such that highD expects one tail event per target km."""
    rate_km = max(float(tail_peak_rate_per_km), 0.0)
    rate_hour = max(float(tail_peak_rate_per_hour), 0.0)
    target_km = float(target_return_km)
    if target_km <= 0.0:
        raise ValueError("target_return_km must be positive")

    expected_tail_exceedances = rate_km * target_km
    level = return_level_for_tail_exposure(
        expected_tail_exceedances=expected_tail_exceedances,
        u=float(model.u),
        xi=float(model.xi),
        beta=float(model.beta),
    )
    conditional_probability = gpd_conditional_survival(
        level,
        u=float(model.u),
        xi=float(model.xi),
        beta=float(model.beta),
    )
    intensity_km = rate_km * conditional_probability
    intensity_mile = intensity_km * KM_PER_MILE
    intensity_hour = rate_hour * conditional_probability
    target_intensity_km = 1.0 / target_km
    return {
        "method": "highd_all_vehicle_km_return_level_pot_gpd",
        "target_return_period_km": target_km,
        "target_return_period_miles": target_km / KM_PER_MILE,
        "target_intensity_per_km": target_intensity_km,
        "target_intensity_per_mile": target_intensity_km * KM_PER_MILE,
        "expected_tail_exceedances_at_target_km": expected_tail_exceedances,
        "expected_tail_exceedances_at_target_mileage": expected_tail_exceedances,
        "target_tail_conditional_probability": (
            1.0 / expected_tail_exceedances
            if expected_tail_exceedances > 0.0
            else float("inf")
        ),
        "target_level": float(level),
        "tail_conditional_probability_at_target_level": float(
            conditional_probability
        ),
        "probability_per_independent_peak_at_target_level": float(
            model.survival(level)
        ),
        "highd_safety_critical_intensity_per_mile": float(intensity_mile),
        "highd_safety_critical_return_period_miles": (
            float(1.0 / intensity_mile)
            if intensity_mile > 0.0
            else float("inf")
        ),
        "highd_safety_critical_intensity_per_km": float(intensity_km),
        "highd_safety_critical_return_period_km": (
            float(1.0 / intensity_km)
            if intensity_km > 0.0
            else float("inf")
        ),
        "highd_safety_critical_intensity_per_hour": float(intensity_hour),
        "highd_safety_critical_return_period_hours": (
            float(1.0 / intensity_hour)
            if intensity_hour > 0.0
            else float("inf")
        ),
    }


def _return_level_curve(
    distances_km: np.ndarray,
    *,
    tail_peak_rate_per_km: float,
    u: float,
    xi: float,
    beta: float,
) -> np.ndarray:
    expected = np.asarray(distances_km, dtype=np.float64) * float(
        tail_peak_rate_per_km
    )
    return np.asarray(
        [
            return_level_for_tail_exposure(
                expected_tail_exceedances=float(value),
                u=float(u),
                xi=float(xi),
                beta=float(beta),
            )
            for value in expected
        ],
        dtype=np.float64,
    )


def _bootstrap_distance_curve(
    values: np.ndarray,
    distances_km: np.ndarray,
    *,
    total_exposure_km: float,
    chosen_k: int,
    samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if samples <= 0 or values.size < max(chosen_k + 1, 10):
        nan = np.full_like(distances_km, np.nan, dtype=np.float64)
        return nan, nan

    rng = np.random.default_rng(int(seed))
    curves: list[np.ndarray] = []
    for _ in range(int(samples)):
        boot = np.sort(rng.choice(values, size=values.size, replace=True))
        k = min(max(5, int(chosen_k)), values.size - 1)
        u = float(boot[boot.size - k - 1])
        excess = boot[boot > u] - u
        if excess.size < 5:
            continue
        try:
            xi, beta = fit_gpd_excess(excess)
        except ValueError:
            continue
        rate_per_km = float(excess.size / max(total_exposure_km, 1.0e-12))
        curves.append(
            _return_level_curve(
                distances_km,
                tail_peak_rate_per_km=rate_per_km,
                u=u,
                xi=xi,
                beta=beta,
            )
        )
    if not curves:
        nan = np.full_like(distances_km, np.nan, dtype=np.float64)
        return nan, nan
    stack = np.stack(curves, axis=0)
    return (
        np.nanquantile(stack, 0.05, axis=0),
        np.nanquantile(stack, 0.95, axis=0),
    )


def write_km_threshold_plots(
    figure_dir: Path,
    *,
    values: np.ndarray,
    model: Any,
    total_exposure_km: float,
    tail_peak_rate_per_km: float,
    target_return_km: float,
    target_level: float,
    risk_variable: str,
    bootstrap_samples: int,
    distance_min_km: float,
    distance_max_km: float,
    random_seed: int,
    filename_prefix: str,
) -> dict[str, str]:
    """Write auditable all-vehicle-km return-level threshold diagnostics."""
    plt = get_pyplot()
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {}

    figure_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    rate_km = max(float(tail_peak_rate_per_km), 0.0)
    target_km = float(target_return_km)
    level = float(target_level)
    num_tail = int(np.sum(values > float(model.u)))

    tail_values = np.sort(values[values > float(model.u)])
    if tail_values.size > 0:
        ranks = np.arange(tail_values.size, 0, -1, dtype=np.float64)
        empirical_return_km = float(total_exposure_km) / ranks
    else:
        empirical_return_km = np.array([], dtype=np.float64)

    plot_min = max(float(distance_min_km), 1.0e-6)
    if tail_values.size > 0:
        plot_min = min(plot_min, float(np.nanmin(empirical_return_km)))
    plot_max = max(float(distance_max_km), target_km * 3.0, plot_min * 10.0)
    distances_km = np.logspace(np.log10(plot_min), np.log10(plot_max), 360)
    levels = _return_level_curve(
        distances_km,
        tail_peak_rate_per_km=rate_km,
        u=float(model.u),
        xi=float(model.xi),
        beta=float(model.beta),
    )
    lower, upper = _bootstrap_distance_curve(
        values,
        distances_km,
        total_exposure_km=float(total_exposure_km),
        chosen_k=num_tail,
        samples=int(bootstrap_samples),
        seed=int(random_seed),
    )

    fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    if tail_values.size > 0:
        ax.scatter(
            empirical_return_km,
            tail_values,
            facecolors="none",
            edgecolors=REFERENCE_COLOR,
            linewidths=0.8,
            s=18,
            alpha=0.55,
            zorder=3,
            label="Empirical tail peaks",
        )
    ax.plot(
        distances_km,
        levels,
        color=GENERATED_COLOR,
        linewidth=2.2,
        label="GPD return level",
    )
    band_mask = np.isfinite(lower) & np.isfinite(upper)
    if np.any(band_mask):
        ax.fill_between(
            distances_km[band_mask],
            lower[band_mask],
            upper[band_mask],
            color=CRITICAL_COLOR,
            alpha=0.16,
            linewidth=0.0,
            label="90% bootstrap band",
        )
    ax.axvline(target_km, color=CRITICAL_COLOR, linestyle="--", linewidth=1.5)
    ax.axhline(level, color=CRITICAL_COLOR, linestyle="--", linewidth=1.5)
    ax.scatter(
        [target_km],
        [level],
        color=CRITICAL_COLOR,
        s=48,
        zorder=5,
        label=r"Human-calibrated $x^\star$",
    )
    ax.set_xscale("log")
    ax.set_xlabel("Human return period (all-vehicle km)")
    ax.set_ylabel(f"Risk return level {risk_variable}")
    ax.set_title("Human-calibrated EVT threshold")
    style_axes(ax)
    ax.legend(frameon=False)
    path = figure_dir / f"{filename_prefix}_human_return_level_threshold.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    paths["human_return_level_threshold"] = str(path)

    y_max = max(
        float(np.nanquantile(values, 0.999)),
        level * 1.2,
        float(model.u) * 1.2,
        1.0,
    )
    y_grid = np.linspace(float(model.u), y_max, 360)
    survival = np.asarray(model.survival(y_grid), dtype=np.float64)
    conditional = np.asarray(
        [
            gpd_conditional_survival(
                float(y),
                u=float(model.u),
                xi=float(model.xi),
                beta=float(model.beta),
            )
            for y in y_grid
        ],
        dtype=np.float64,
    )
    intensity_km = rate_km * conditional
    target_probability = float(model.survival(level))
    target_intensity = 1.0 / target_km

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
    axes[0].plot(y_grid, survival, color=GENERATED_COLOR, linewidth=2.0)
    axes[0].axhline(target_probability, color=CRITICAL_COLOR, linestyle="--")
    axes[0].axvline(level, color=CRITICAL_COLOR, linestyle="--")
    axes[0].axvline(float(model.u), color=REFERENCE_COLOR, linestyle=":")
    axes[0].set_yscale("log")
    axes[0].set_xlabel(f"Risk value {risk_variable}")
    axes[0].set_ylabel(r"$P_H(Y > y)$ per independent peak")
    axes[0].set_title("Survival threshold")
    style_axes(axes[0])

    axes[1].plot(y_grid, intensity_km, color=GENERATED_COLOR, linewidth=2.0)
    axes[1].axhline(target_intensity, color=CRITICAL_COLOR, linestyle="--")
    axes[1].axvline(level, color=CRITICAL_COLOR, linestyle="--")
    axes[1].axvline(float(model.u), color=REFERENCE_COLOR, linestyle=":")
    axes[1].set_yscale("log")
    axes[1].set_xlabel(f"Risk value {risk_variable}")
    axes[1].set_ylabel("Human risk intensity (per all-vehicle km)")
    axes[1].set_title("Risk intensity threshold")
    style_axes(axes[1])
    path = figure_dir / f"{filename_prefix}_survival_intensity_threshold.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    paths["survival_intensity_threshold"] = str(path)

    candidates = list(getattr(model, "threshold_candidates", []) or [])
    if candidates:
        u_values: list[float] = []
        threshold_values: list[float] = []
        for row in candidates:
            k = float(row.get("k", float("nan")))
            if not np.isfinite(k) or k <= 0.0:
                continue
            candidate_rate = k / max(float(total_exposure_km), 1.0e-12)
            u_values.append(float(row["u"]))
            threshold_values.append(
                return_level_for_tail_exposure(
                    expected_tail_exceedances=candidate_rate * target_km,
                    u=float(row["u"]),
                    xi=float(row["xi"]),
                    beta=float(row["beta"]),
                )
            )
        if u_values:
            fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
            ax.plot(u_values, threshold_values, color=GENERATED_COLOR, linewidth=1.7)
            ax.axvline(float(model.u), color=REFERENCE_COLOR, linestyle="--", label=r"$u_e$")
            ax.axhline(level, color=CRITICAL_COLOR, linestyle="--", label=r"$x_e^\star$")
            ax.scatter([float(model.u)], [level], color=CRITICAL_COLOR, s=38, zorder=5)
            ax.set_xlabel(r"POT threshold $u_e$")
            ax.set_ylabel(r"Return-level threshold $x_e^\star(u_e)$")
            ax.set_title("Threshold sensitivity")
            style_axes(ax)
            ax.legend(frameon=False)
            path = figure_dir / f"{filename_prefix}_threshold_sensitivity.png"
            fig.savefig(path, dpi=170)
            plt.close(fig)
            paths["threshold_sensitivity"] = str(path)

    return paths
