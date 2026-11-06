"""Raw-data calibration and compact conditional corrections for NPC drivers."""

from copy import deepcopy
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp, softmax

from ..stochastic_drivers.driver_policy import LONGITUDINAL_FEATURE_NAMES
from ..stochastic_drivers.idm_population import (
    idm_acceleration,
    population_driver_parameters,
    population_predictions,
)
from .metrics import (
    FEATURE_EDGES,
    hellinger,
)
from .residual import (
    driver_residuals,
    fit_autoregression,
    residual_correlations,
)

ROOT = Path(__file__).resolve().parents[3]
MOBIL_PATH = ROOT / "results/hierarchical_world_model/model_assets/mobil.json"
SEED = 20261012


def compact_mobil(mobil):
    """Publish executable parameters, leaving fitting diagnostics in calibration reports."""
    result = deepcopy(mobil)
    if "lane_style" in result:
        result["lane_style"] = {
            key: result["lane_style"][key] for key in ("mean", "std")
        }
    if "correction" in result:
        correction = result["correction"]
        for name in ("diagnostics", "ridge"):
            correction.pop(name, None)
        parameters = correction.get("longitudinal")
        if (
            parameters is not None
            and parameters["mean_strength"] == parameters["noise_strength"] == 0
        ):
            del correction["longitudinal"]
    return result


def compact_population(population):
    return {
        key: np.asarray(population[key])
        for key in (
            "log_mean",
            "log_cov",
            "residual_std",
            "fit_driver_count",
            "regularization",
        )
    }


def lane_inputs(data, population, parameters=None):
    """Resolve exact IDM incentives using the same current physical neighbors."""
    theta = np.exp(population["log_mean"]) if parameters is None else parameters["own"]
    target_theta = theta if parameters is None else parameters["target_rear"]
    rear_theta = theta if parameters is None else parameters["current_rear"]
    own_pair_theta = theta if theta.ndim == 1 else theta[:, None, :]
    v, length = data["own_speed"], data["own_length"]
    current = idm_acceleration(
        data["current_front_gap"], v, data["current_front_speed"], theta
    )
    after = idm_acceleration(
        data["target_front_gap"], v[:, None], data["target_front_speed"], own_pair_theta
    )
    rear_after = idm_acceleration(
        data["target_rear_gap"], data["target_rear_speed"], v[:, None], target_theta
    )
    rear_before = idm_acceleration(
        data["target_rear_gap"] + length[:, None] + data["target_front_gap"],
        data["target_rear_speed"],
        data["target_front_speed"],
        target_theta,
    )
    old_before = idm_acceleration(
        data["current_rear_gap"], data["current_rear_speed"], v, rear_theta
    )
    old_after = idm_acceleration(
        data["current_rear_gap"] + length + data["current_front_gap"],
        data["current_rear_speed"],
        data["current_front_speed"],
        rear_theta,
    )
    old_delta = np.where(data["current_rear_gap"] < 1e5, old_after - old_before, 0)
    own = after - current[:, None]
    social = (
        np.where(data["target_rear_gap"] < 1e5, rear_after - rear_before, 0)
        + old_delta[:, None]
    )
    safe = data["available"] & ((data["target_rear_gap"] >= 1e5) | (rear_after >= -4))
    return own.astype(np.float32), social.astype(np.float32), safe


def exclusive_choices(side):
    lower, higher = side.T
    both = lower * higher
    return np.column_stack(
        (
            (1 - lower) * (1 - higher),
            lower * (1 - higher) + both * (lower >= higher),
            higher * (1 - lower) + both * (higher > lower),
        )
    )


def lane_probabilities(data, population, mobil, inputs=None, style_offset=None):
    own, social, safe = lane_inputs(data, population) if inputs is None else inputs
    score = (own + mobil["politeness"] * social).clip(-12, 12)
    correction = mobil.get("correction", {}).get("lane")
    side = expit(mobil["bias"] + mobil["slope"] * score)
    choice = exclusive_choices(np.where(safe, side, 0))
    if correction:
        x = (data["features"] - correction["feature_mean"]) / correction["feature_std"]
        shift = correction["coef"][0] + x @ np.asarray(correction["coef"][1:])
        intent = exclusive_choices(np.where(safe, side, 0))
        logits = np.log(intent.clip(1e-12))
        logits[:, 1:] += shift
        logits[:, 1:] += correction.get("logit_bias", 0.0)
        logits[:, 1:] += np.array([-1, 1]) * correction.get("direction_bias", 0.0)
        logits[:, 1:] = np.where(safe, logits[:, 1:], -np.inf)
        choice = (1 - correction["strength"]) * choice + correction[
            "strength"
        ] * softmax(logits, axis=1)
    if style_offset is not None:
        logits = np.log(choice.clip(1e-12))
        logits[:, 1:] += np.asarray(style_offset)[..., None]
        logits[:, 1:] = np.where(safe, logits[:, 1:], -np.inf)
        choice = softmax(logits, axis=1)
    return choice


def sampled_lane_probabilities(data, population, mobil, samples=16, seed=SEED):
    """Marginalize persistent focal and follower styles, keyed by physical identity."""
    from ..stochastic_drivers.idm_population import sample_theta

    recording = data["recording"].astype(np.int64)
    # Native highD vehicle ids are below 100,000; this encodes a pair, not a hash.
    identities = {
        "own": recording * 100000 + data["vehicle"],
        "current_rear": recording * 100000 + data["current_rear_vehicle"],
        "target_rear": recording[:, None] * 100000 + data["target_rear_vehicle"],
    }
    all_codes = np.unique(
        np.concatenate([code.ravel() for code in identities.values()])
    )
    indices = {
        name: np.searchsorted(all_codes, value) for name, value in identities.items()
    }
    latent = np.random.default_rng([seed, 22]).standard_normal(
        (samples, len(all_codes), 5)
    )
    parameters = sample_theta(population, latent)
    style = mobil.get("lane_style", {"mean": 0.0, "std": 0.0})
    style_latent = np.random.default_rng([seed, 23]).standard_normal(
        (samples, len(all_codes))
    )
    probability = np.zeros((len(recording), 3), np.float64)
    for sample, realization in enumerate(parameters):
        local = {name: realization[index] for name, index in indices.items()}
        probability += lane_probabilities(
            data,
            population,
            mobil,
            lane_inputs(data, population, local),
            style_offset=(
                style["mean"] + style["std"] * style_latent[sample, indices["own"]]
                if "lane_style" in mobil
                else None
            ),
        )
    return probability / samples


def longitudinal_design(data, mean=None, std=None):
    raw = np.column_stack(
        (
            data["speed"],
            np.log1p(data["gap"].clip(0, 120)),
            data["speed"] - data["leader_speed"],
            data["free"],
            data["leader_acceleration"],
            data["leader_history_available"],
            data["changing"],
            data["progress"],
            data["lateral_speed"],
            *data["entry_features"].T,
        )
    )
    mean = raw.mean(0) if mean is None else np.asarray(mean)
    std = raw.std(0).clip(1e-3) if std is None else np.asarray(std)
    return np.column_stack((np.ones(len(raw)), (raw - mean) / std)), mean, std


def fit_longitudinal_correction(data, population, ridge):
    """Estimate within-driver residuals before fitting conditional dispersion."""
    # The published population stays fixed; driver centering below separates
    # persistent individual offsets from the conditional residual variance.
    theta = np.exp(population["log_mean"])
    residual = data["acceleration"] - idm_acceleration(
        data["gap"],
        data["speed"],
        data["leader_speed"],
        theta,
    ).clip(-8, 4)
    x, mean, std = longitudinal_design(data)
    penalty = np.eye(x.shape[1]) * ridge * len(x)
    penalty[0, 0] = 0
    coefficient = np.linalg.solve(x.T @ x + penalty, x.T @ residual)
    centered, driver_means = driver_residuals(data, residual - x @ coefficient)
    variance_coefficient = np.linalg.solve(
        x.T @ x + penalty, x.T @ np.log(np.square(centered).clip(1e-4))
    )
    # Log-squared Gaussian observations have E[log chi-square(1)] = -1.27036.
    variance_coefficient[0] += 1.270362845
    sigma = np.exp(0.5 * (x @ variance_coefficient).clip(-16, 4))
    autoregression, correlations = fit_autoregression(data, centered / sigma)
    return {
        "feature_names": list(LONGITUDINAL_FEATURE_NAMES),
        "feature_mean": mean.tolist(),
        "feature_std": std.tolist(),
        "mean_coef": coefficient.tolist(),
        "variance_coef": variance_coefficient.tolist(),
        "autoregression": autoregression,
        "mean_strength": 1.0,
        "noise_strength": 1.0,
        "diagnostics": {
            "raw_residual_std_mps2": float(residual.std()),
            "driver_mean_std_mps2": float(driver_means.std()),
            "within_driver_std_mps2": float(centered.std()),
            "raw_correlations": residual_correlations(data, residual),
            "standardized_within_driver_correlations": correlations,
        },
    }


def fit_correction(data, lane, population, mobil, ridge):
    longitudinal = fit_longitudinal_correction(data, population, ridge)
    features = lane["features"]
    lane_mean, lane_std = (
        features.reshape(-1, 8).mean(0),
        features.reshape(-1, 8).std(0).clip(1e-3),
    )
    z = (features - lane_mean) / lane_std
    z = np.concatenate((np.ones((*z.shape[:2], 1)), z), -1).astype(np.float64)
    own, social, safe = lane_inputs(lane, population)
    intended = np.where(
        safe,
        expit(
            mobil["bias"]
            + mobil["slope"] * (own + mobil["politeness"] * social).clip(-12, 12)
        ),
        0,
    )
    base = np.log(exclusive_choices(intended).clip(1e-12)).astype(np.float64)
    if "lane_style" in mobil:
        base[:, 1:] += mobil["lane_style"]["mean"]
        base -= logsumexp(base, axis=1, keepdims=True)
    labels = lane["label"].astype(int)

    def objective(beta):
        logits = base.copy()
        logits[:, 1:] += z @ np.r_[0.0, beta]
        logits[:, 1:] = np.where(safe, logits[:, 1:], -np.inf)
        probability = softmax(logits, axis=1)
        loss = -np.log(probability[np.arange(len(labels)), labels].clip(1e-12)).mean()
        error = probability[:, 1:] - np.eye(3)[labels][:, 1:]
        possible = (labels == 0) | safe[np.arange(len(labels)), (labels - 1).clip(0)]
        possible &= probability[np.arange(len(labels)), labels] > 1e-12
        error *= safe & possible[:, None]
        gradient = np.einsum("ns,nsk->k", error, z) / len(labels)
        return (
            float(loss + 0.5 * ridge * np.dot(beta, beta)),
            gradient[1:] + ridge * beta,
        )

    result = minimize(
        objective,
        np.zeros(8),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 100, "gtol": 1e-6, "ftol": 1e-10},
    )
    if not result.success:
        raise RuntimeError(f"Conditional lane correction: {result.message}")
    return {
        "longitudinal": longitudinal,
        "lane": {
            "feature_mean": lane_mean.tolist(),
            "feature_std": lane_std.tolist(),
            "coef": [0.0, *result.x.tolist()],
            "strength": 1.0,
            "logit_bias": 0.0,
            "direction_bias": 0.0,
        },
        "ridge": float(ridge),
        "response_blend": 1.0,
        "trajectory": {"log_duration_scale": 0.0, "shape_shift": [0.0, 0.0]},
        "diagnostics": longitudinal["diagnostics"],
    }


def conditional_nll(data, population, mobil, samples=16):
    """Val likelihood marginalizes the same persistent driver distribution."""
    parameters = mobil["correction"]["longitudinal"]
    styles, index = population_driver_parameters(
        population, data, samples=samples, seed=SEED
    )
    log_probability = []
    x, _, _ = longitudinal_design(
        data, parameters["feature_mean"], parameters["feature_std"]
    )
    shift = x @ np.asarray(parameters["mean_coef"])
    sigma = np.exp(0.5 * (x @ np.asarray(parameters["variance_coef"])).clip(-16, 4))
    for theta in styles:
        theta = theta[index]
        mean = idm_acceleration(
            data["gap"], data["speed"], data["leader_speed"], theta
        ).clip(-8, 4)
        mean += shift
        log_probability.append(
            -0.5 * np.square((data["acceleration"] - mean) / sigma)
            - np.log(sigma * np.sqrt(2 * np.pi))
        )
    return float(-(logsumexp(log_probability, axis=0) - np.log(samples)).mean())


def conditional_predictions(data, population, mobil, samples=16, seed=SEED):
    values = population_predictions(population, data, samples=samples, seed=seed).clip(
        -8, 4
    )
    parameters = mobil.get("correction", {}).get("longitudinal")
    if parameters:
        x, _, _ = longitudinal_design(
            data,
            parameters["feature_mean"],
            parameters["feature_std"],
        )
        mean_constant = x @ np.asarray(parameters["mean_coef"])
        variance_constant = x @ np.asarray(parameters["variance_coef"])
        # Conditional marginals use stationary innovations; time correlation is checked in rollouts.
        innovations = (
            np.random.default_rng([seed, 21])
            .standard_normal(values.shape)
            .astype(np.float32)
        )
        urgent = data["gap"] < np.maximum(2, 0.1 * data["speed"])
        sigma = np.exp(0.5 * variance_constant.clip(-16, 4))
        for sample in range(samples):
            corrected = (
                values[sample]
                + parameters["mean_strength"] * mean_constant
                + parameters["noise_strength"] * sigma * innovations[sample]
            )
            braking = urgent | (
                (data["speed"] > data["leader_speed"]) & (values[sample] < 0)
            )
            corrected[braking] = np.minimum(corrected[braking], values[sample, braking])
            values[sample] = corrected.clip(-8, 4)
    return values.astype(np.float32)


def group_histograms(values, recording, keep, bins, recordings):
    selected = np.flatnonzero(keep)
    _, rec = np.unique(recording, return_inverse=True)
    result = np.zeros((len(recordings), len(bins) - 1), float)
    for sample in np.atleast_2d(values):
        cell = np.searchsorted(bins, sample[selected], side="right") - 1
        flat = rec[selected] * (len(bins) - 1) + cell.clip(0, len(bins) - 2)
        result += np.bincount(flat, minlength=result.size).reshape(result.shape)
    return result / len(np.atleast_2d(values))


def conditional_metrics(data, lane, population, mobil, *, samples=16, bootstrap=2000):
    values = conditional_predictions(data, population, mobil, samples)
    recordings = np.unique(data["recording"])
    draws = np.random.default_rng(SEED).integers(
        len(recordings), size=(bootstrap, len(recordings))
    )
    free = data["free"]
    closing = data["speed"] - data["leader_speed"]
    groups = {
        "free_driving": free,
        "steady_following": ~free & (abs(closing) <= 1),
        "approaching": ~free & (closing > 1),
        "receding": ~free & (closing < -1),
        "close_following": ~free & (data["gap"] < 20),
        "lateral_motion": data["changing"].astype(bool),
        "lane_keeping": ~data["changing"].astype(bool),
    }
    histograms, rows = {}, []
    for group, keep in groups.items():
        real = group_histograms(
            data["acceleration"],
            data["recording"],
            keep,
            FEATURE_EDGES["acceleration"],
            recordings,
        )
        model = group_histograms(
            values, data["recording"], keep, FEATURE_EDGES["acceleration"], recordings
        )
        boots = hellinger(real[draws].sum(1), model[draws].sum(1))
        histograms[group + "_real"], histograms[group + "_model"] = real, model
        error = values[:, keep].mean(0) - data["acceleration"][keep]
        histograms[group + "_squared_error_recording_sums"] = np.bincount(
            np.searchsorted(recordings, data["recording"][keep]),
            weights=np.square(error),
            minlength=len(recordings),
        )
        rows.append(
            {
                "group": group,
                "observations": int(keep.sum()),
                "Hellinger": float(hellinger(real.sum(0), model.sum(0))),
                "SE": float(boots.std(ddof=1)),
                "CI95": np.quantile(boots, [0.025, 0.975]).tolist(),
                "mean_RMSE_mps2": float(np.sqrt(np.square(error).mean())),
                "real_quantiles": np.quantile(
                    data["acceleration"][keep], [0.05, 0.5, 0.95]
                ).tolist(),
                "model_quantiles": np.quantile(
                    values[:, keep], [0.05, 0.5, 0.95]
                ).tolist(),
            }
        )
    probability = sampled_lane_probabilities(lane, population, mobil, samples)
    error = np.square(probability - np.eye(3)[lane["label"]]).sum(1)
    lane_recordings, rec = np.unique(lane["recording"], return_inverse=True)
    sums, counts = np.bincount(rec, weights=error), np.bincount(rec)
    histograms["lane_Brier_recording_sums"] = sums
    histograms["lane_Brier_recording_counts"] = counts
    histograms["lane_recordings"] = lane_recordings
    boot_indices = np.random.default_rng(SEED).integers(
        len(counts), size=(bootstrap, len(counts))
    )
    boots = sums[boot_indices].sum(1) / counts[boot_indices].sum(1)
    rear_bins = FEATURE_EDGES["gap"]
    real_acceptance, model_acceptance, acceptance_intervals = [], [], []
    for variable in ("target_rear_gap", "target_front_gap"):
        real = np.zeros(len(rear_bins) - 1)
        model = real.copy()
        for side in range(2):
            real += np.histogram(
                lane[variable][:, side],
                rear_bins,
                weights=(lane["label"] == side + 1).astype(float),
            )[0]
            model += np.histogram(
                lane[variable][:, side], rear_bins, weights=probability[:, side + 1]
            )[0]
        histograms[variable + "_real"], histograms[variable + "_model"] = real, model
        real_acceptance.append(float(hellinger(real, model)))
        # Recording bootstrap of weighted acceptance histograms retains zero-start recordings.
        r, m = [], []
        for recording_id in lane_recordings:
            keep = lane["recording"] == recording_id
            r.append(
                sum(
                    np.histogram(
                        lane[variable][keep, side],
                        rear_bins,
                        weights=(lane["label"][keep] == side + 1).astype(float),
                    )[0]
                    for side in range(2)
                )
            )
            m.append(
                sum(
                    np.histogram(
                        lane[variable][keep, side],
                        rear_bins,
                        weights=probability[keep, side + 1],
                    )[0]
                    for side in range(2)
                )
            )
        real_recording, model_recording = np.asarray(r), np.asarray(m)
        histograms[variable + "_real_recording"] = real_recording
        histograms[variable + "_model_recording"] = model_recording
        distribution = hellinger(
            real_recording[boot_indices].sum(1), model_recording[boot_indices].sum(1)
        )
        model_acceptance.append(float(distribution.std(ddof=1)))
        acceptance_intervals.append(np.quantile(distribution, [0.025, 0.975]).tolist())
    lane_result = {
        "opportunities": len(error),
        "Brier": float(error.mean()),
        "parameter_sampling": f"{samples} persistent physical-driver draws",
        "Brier_SE": float(boots.std(ddof=1)),
        "Brier_CI95": np.quantile(boots, [0.025, 0.975]).tolist(),
        "observed_choice_frequency": np.bincount(lane["label"], minlength=3)
        .astype(float)
        .tolist(),
        "predicted_choice_frequency": probability.sum(0).tolist(),
        "acceptance_Hellinger": real_acceptance,
        "acceptance_SE": model_acceptance,
        "acceptance_CI95": acceptance_intervals,
    }
    histograms["recordings"] = recordings
    histograms["lane_probability"] = probability.astype(np.float32)
    histograms["acceptance_edges"] = rear_bins
    for name, source_values in (
        ("real", data["acceleration"]),
        ("model", values.mean(0)),
    ):
        trend = []
        for lower, upper in zip(
            (-np.inf, -5, -3, -1, 1, 3, 5), (-5, -3, -1, 1, 3, 5, np.inf)
        ):
            keep = ~free & (closing >= lower) & (closing < upper)
            _, cluster = np.unique(data["recording"][keep], return_inverse=True)
            sums = np.bincount(cluster, weights=source_values[keep])
            counts = np.bincount(cluster)
            draw = np.random.default_rng(SEED).integers(
                len(counts), size=(bootstrap, len(counts))
            )
            means = sums[draw].sum(1) / counts[draw].sum(1)
            trend.append(
                [
                    keep.sum(),
                    float(source_values[keep].mean()),
                    *np.quantile(means, [0.025, 0.975]),
                ]
            )
        histograms[f"closing_response_{name}"] = np.asarray(trend)
    for name, lane_values in (
        ("real", np.eye(3)[lane["label"]][:, 1:]),
        ("model", probability[:, 1:]),
    ):
        grouped = []
        for lower, upper in zip((0, 15, 30, 60), (15, 30, 60, np.inf)):
            keep = (
                (lane["target_rear_gap"] >= lower)
                & (lane["target_rear_gap"] < upper)
                & lane["available"]
            )
            ids = np.broadcast_to(lane["recording"][:, None], keep.shape)[keep]
            _, cluster = np.unique(ids, return_inverse=True)
            observations = lane_values[keep]
            sums = np.bincount(cluster, weights=observations)
            counts = np.bincount(cluster)
            draw = np.random.default_rng(SEED).integers(
                len(counts), size=(bootstrap, len(counts))
            )
            rates = sums[draw].sum(1) / counts[draw].sum(1)
            grouped.append(
                [
                    keep.sum(),
                    float(observations.mean()),
                    *np.quantile(rates, [0.025, 0.975]),
                ]
            )
        histograms[f"lane_probability_{name}"] = np.asarray(grouped)
    return {"acceleration": rows, "lane": lane_result}, histograms
