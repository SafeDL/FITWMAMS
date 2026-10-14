"""Calibration outputs retain the same semantics as the executed driver policy."""

from copy import deepcopy

import numpy as np
import pytest
import torch

from hierarchical_world_model.src.npc_experiments.calibration import (
    conditional_predictions,
    lane_probabilities,
    longitudinal_design,
    compact_mobil,
)
from hierarchical_world_model.src.npc_experiments import calibration
from hierarchical_world_model.src.npc_experiments.residual import (
    autoregression_parameters,
)
from hierarchical_world_model.src.stochastic_drivers.driver_policy import (
    corrected_acceleration,
    exclusive_lane_probabilities,
    longitudinal_features,
)
from hierarchical_world_model.src.stochastic_drivers.idm_population import (
    idm_acceleration,
    torch_idm,
)


def _population():
    return {
        "log_mean": np.log([33, 2, 1.2, 1.0, 2.0]),
        "log_cov": np.zeros((5, 5)),
    }


@pytest.mark.parametrize("with_correction", [False, True])
def test_lane_calibration_probabilities_match_executed_competing_choices(
    with_correction,
):
    rng = np.random.default_rng(17)
    count = 200
    own = rng.uniform(-30, 30, (count, 2)).astype(np.float32)
    social = rng.uniform(-15, 15, (count, 2)).astype(np.float32)
    safe = rng.random((count, 2)) > 0.3
    safe[:2] = [[False, False], [True, True]]
    own[1], social[1] = [2, 2], [0, 0]  # Double-choice ties always go to lower-y.
    features = rng.normal(size=(count, 2, 8)).astype(np.float32)
    mobil = {"politeness": 0.3, "bias": -3, "slope": 0.2}
    if with_correction:
        mobil["correction"] = {
            "lane": {
                "feature_mean": rng.normal(size=8).tolist(),
                "feature_std": rng.uniform(0.2, 2, 8).tolist(),
                "coef": rng.normal(scale=0.15, size=9).tolist(),
                "strength": 0.75,
            }
        }
    numpy_probability = lane_probabilities(
        {"features": features, "available": safe},
        _population(),
        mobil,
        inputs=(own, social, safe),
    )
    executed = exclusive_lane_probabilities(
        torch.from_numpy(own + mobil["politeness"] * social),
        torch.from_numpy(safe),
        mobil,
        features=torch.from_numpy(features),
        correction=mobil.get("correction"),
    ).numpy()
    np.testing.assert_allclose(numpy_probability, executed, rtol=2e-6, atol=2e-7)
    np.testing.assert_allclose(numpy_probability.sum(-1), 1, atol=1e-7)
    assert np.all(numpy_probability[:, 1:][~safe] == 0)
    assert numpy_probability[1, 1] > numpy_probability[1, 2]


def _following():
    gap = np.array([-1, 0, 1, 5, 20, 1e6], np.float32)
    data = {
        "gap": gap,
        "speed": np.array([15, 15, 15, 15, 25, 35], np.float32),
        "leader_speed": np.array([20, 20, 20, 20, 28, 35], np.float32),
        "free": gap >= 1e5,
        "recording": np.array([1, 1, 1, 2, 2, 2]),
        "vehicle": np.arange(6),
        "changing": np.array([0, 1, 1, 0, 1, 0], np.float32),
        "progress": np.linspace(0, 1, 6, dtype=np.float32),
        "lateral_speed": np.array([0, 0.2, 0.4, 0, 0.6, 0.1], np.float32),
        "past_acceleration": np.array([0, 0.2, -0.1, 0, 0.4, 0], np.float32),
        "leader_acceleration": np.array([0, 0.5, -0.3, 0, 0.2, 0], np.float32),
        "history_available": np.array([False, True, True, False, True, False]),
        "leader_history_available": np.array([False, True, True, False, True, False]),
    }
    for name in ("gap", "speed", "leader_speed"):
        data["past_" + name] = data[name].copy()
    past = idm_acceleration(
        data["past_gap"],
        data["past_speed"],
        data["past_leader_speed"],
        np.exp(_population()["log_mean"]),
    ).clip(-8, 4)
    data["past_residual"] = np.where(
        data["history_available"], data["past_acceleration"] - past, 0
    )
    return data


def test_longitudinal_normalization_matches_runtime_physical_features():
    data = _following()
    design, mean, std = longitudinal_design(data)
    torch_features = longitudinal_features(
        *[torch.from_numpy(data[key]) for key in ("gap", "speed", "leader_speed")],
        history_features=torch.from_numpy(
            np.column_stack(
                [
                    data[key]
                    for key in (
                        "leader_acceleration",
                        "leader_history_available",
                    )
                ]
            )
        ),
        maneuver_features=torch.from_numpy(
            np.column_stack(
                [data[key] for key in ("changing", "progress", "lateral_speed")]
            )
        ),
    ).numpy()
    np.testing.assert_allclose(design[:, 1:], (torch_features - mean) / std, atol=2e-6)
    np.testing.assert_allclose(design[:, 1:].mean(0), 0, atol=1e-6)
    np.testing.assert_allclose(design[:, 1:].std(0), 1, atol=1e-6)
    reused, reused_mean, reused_std = longitudinal_design(data, mean, std)
    np.testing.assert_array_equal(design, reused)
    np.testing.assert_array_equal(mean, reused_mean)
    np.testing.assert_array_equal(std, reused_std)


def test_longitudinal_design_excludes_own_realized_action_history():
    data = _following()
    first, _, _ = longitudinal_design(data)
    data["past_acceleration"][:] = 1e4
    data["past_residual"][:] = -1e4
    data["history_available"][:] = ~data["history_available"]
    second, _, _ = longitudinal_design(data)
    np.testing.assert_array_equal(first, second)


def test_publication_removes_inactive_residuals_without_changing_actions():
    data, population = _following(), _population()
    _, mean, std = longitudinal_design(data)
    model = {
        "correction": {
            "longitudinal": {
                "feature_mean": mean.tolist(),
                "feature_std": std.tolist(),
                "mean_coef": [0.0] * 10,
                "variance_coef": [0.0] * 10,
                "mean_strength": 0.0,
                "noise_strength": 0.0,
            },
            "diagnostics": {"fit": 1},
            "ridge": 0.1,
            "response_blend": 1.0,
        }
    }
    deployed = compact_mobil(model)
    assert "longitudinal" not in deployed["correction"]
    assert "longitudinal" in model["correction"]
    assert deployed["correction"] == {"response_blend": 1.0}
    np.testing.assert_array_equal(
        conditional_predictions(data, population, model),
        conditional_predictions(data, population, deployed),
    )


def test_finite_idm_actions_agree_for_zero_and_negative_gaps():
    data = _following()
    theta = np.exp(_population()["log_mean"])
    numpy_action = idm_acceleration(
        data["gap"].astype(np.float64),
        data["speed"].astype(np.float64),
        data["leader_speed"].astype(np.float64),
        theta,
    )
    executed = torch_idm(
        *[
            torch.from_numpy(data[key]).double()
            for key in ("gap", "speed", "leader_speed")
        ],
        torch.from_numpy(theta),
    ).numpy()
    assert np.isfinite(numpy_action).all() and np.isfinite(executed).all()
    np.testing.assert_allclose(numpy_action, executed, rtol=1e-7, atol=2e-7)


def test_conditional_acceleration_distribution_matches_stationary_runtime_correction():
    data, population = _following(), _population()
    _, mean, std = longitudinal_design(data)
    correction = {
        "longitudinal": {
            "feature_mean": mean.tolist(),
            "feature_std": std.tolist(),
            "mean_coef": [0.2, 0.05, -0.1, 0.03, 0.02, -0.02, 0, 0, 0, 0],
            "variance_coef": [-2, 0.02, -0.1, 0.03, 0.05, 0, 0, 0, 0, 0],
            "autoregression": autoregression_parameters([1.8, -0.85]),
            "mean_strength": 0.5,
            "noise_strength": 0.5,
        }
    }
    samples, seed = 16, 11
    calibrated = conditional_predictions(
        data, population, {"correction": correction}, samples, seed
    )
    innovations = (
        np.random.default_rng([seed, 21])
        .standard_normal((samples, len(data["gap"])))
        .astype(np.float32)
    )
    executed, state = corrected_acceleration(
        *[
            torch.from_numpy(data[key])[None].expand(samples, -1)
            for key in ("gap", "speed", "leader_speed")
        ],
        torch.tensor(np.exp(population["log_mean"]), dtype=torch.float32).expand(
            samples, len(data["gap"]), -1
        ),
        correction=correction,
        innovation=torch.from_numpy(innovations),
        history_features=torch.from_numpy(
            np.column_stack(
                [
                    data[key]
                    for key in (
                        "leader_acceleration",
                        "leader_history_available",
                    )
                ]
            )
        )[None].expand(samples, -1, -1),
        maneuver_features=torch.from_numpy(
            np.column_stack(
                [data[key] for key in ("changing", "progress", "lateral_speed")]
            )
        )[None].expand(samples, -1, -1),
    )
    np.testing.assert_allclose(calibrated, executed.numpy(), rtol=3e-6, atol=3e-6)
    np.testing.assert_array_equal(state[..., 0].numpy(), innovations)
    assert np.isfinite(calibrated).all()
    assert calibrated.min() >= -8 and calibrated.max() <= 4


def test_conditional_lane_gradient_matches_finite_difference_with_impossible_labels(
    monkeypatch,
):
    from types import SimpleNamespace

    population, data = _population(), _following()
    data["acceleration"] = np.linspace(-0.2, 0.2, 6)
    data["frame"] = np.arange(6) * 5
    population["driver_keys"] = np.column_stack((data["recording"], data["vehicle"]))
    population["driver_log_parameters"] = np.tile(population["log_mean"], (6, 1))
    lane = {
        "label": np.array([1, 1, 0]),
        "features": np.random.default_rng(3).normal(size=(3, 2, 8)),
    }
    safe = np.array([[False, False], [True, False], [True, True]])
    lane["available"] = safe
    own = np.array([[0, 0], [2, 1], [1, -1]], dtype=np.float32)
    monkeypatch.setattr(
        calibration, "lane_inputs", lambda *args: (own, np.zeros_like(own), safe)
    )
    evaluated = []

    def minimize(objective, initial, **kwargs):
        point = np.linspace(-0.2, 0.3, 8)
        loss, gradient = objective(point)
        numeric = np.zeros(8)
        for coordinate in range(8):
            delta = np.zeros(8)
            delta[coordinate] = 1e-5
            numeric[coordinate] = (
                objective(point + delta)[0] - objective(point - delta)[0]
            ) / 2e-5
        np.testing.assert_allclose(gradient, numeric, rtol=2e-6, atol=2e-7)
        assert np.isfinite(loss) and np.isfinite(gradient).all()
        evaluated.append((loss, gradient))
        return SimpleNamespace(success=True, x=point)

    monkeypatch.setattr(calibration, "minimize", minimize)
    mobil = {"politeness": 0.3, "bias": -3, "slope": 0.1}
    calibration.fit_correction(data, lane, population, mobil, 0.1)
    lane["label"][
        0
    ] = 2  # Both impossible choices contribute the same constant clipped loss.
    calibration.fit_correction(data, lane, population, mobil, 0.1)
    assert evaluated[0][0] == evaluated[1][0]
    np.testing.assert_array_equal(evaluated[0][1], evaluated[1][1])


def test_lane_marginalization_reuses_physical_driver_traits_across_neighbor_roles(
    monkeypatch,
):
    population = _population()
    population["log_cov"] = np.eye(5) * 0.03
    data = {
        "recording": np.array([4, 4, 4]),
        "vehicle": np.array([10, 10, 11]),
        "current_rear_vehicle": np.array([11, 11, 10]),
        "target_rear_vehicle": np.array([[12, 13], [12, 13], [10, 12]]),
        "features": np.zeros((3, 2, 8)),
    }
    sampled_traits = []

    def inputs(data, population, parameters):
        codes = data["vehicle"]
        focal_index = {
            int(actor): int(np.flatnonzero(codes == actor)[0])
            for actor in np.unique(codes)
        }
        for position, rear in enumerate(data["current_rear_vehicle"]):
            if int(rear) in focal_index:
                np.testing.assert_array_equal(
                    parameters["current_rear"][position],
                    parameters["own"][focal_index[int(rear)]],
                )
        for position, actor in enumerate(codes):
            np.testing.assert_array_equal(
                parameters["own"][position], parameters["own"][focal_index[int(actor)]]
            )
        sampled_traits.append(parameters["own"].copy())
        own = np.column_stack(
            (parameters["own"][:, 0] - 33, 5 * (parameters["own"][:, 3] - 1))
        )
        return own, np.zeros_like(own), np.ones_like(own, bool)

    monkeypatch.setattr(calibration, "lane_inputs", inputs)
    mobil = {"politeness": 0.3, "bias": -3, "slope": 0.2}
    probability = calibration.sampled_lane_probabilities(
        data, population, mobil, samples=16
    )
    traits = np.stack(sampled_traits)
    assert np.std(traits[:, 0, 0]) > 0.1
    np.testing.assert_array_equal(probability[0], probability[1])
    sampled_traits.clear()
    order = np.array([2, 1, 0])
    reordered = calibration.sampled_lane_probabilities(
        {key: value[order] for key, value in data.items()},
        population,
        mobil,
        samples=16,
    )
    np.testing.assert_allclose(probability[order], reordered, atol=1e-12)


def test_acceptance_intervals_can_be_recomputed_from_saved_recording_histograms(
    monkeypatch, tmp_path
):
    from hierarchical_world_model.src.npc_experiments.metrics import hellinger

    closing = np.tile([-6, -4, -2, 0, 2, 4, 6, 0], 2)
    free = np.tile([False] * 7 + [True], 2)
    data = {
        "speed": np.full(16, 20.0),
        "leader_speed": 20.0 - closing,
        "gap": np.where(free, 1e6, 15),
        "free": free,
        "acceleration": -0.05 * closing,
        "recording": np.repeat([1, 2], 8),
        "changing": np.tile([False, True], 8),
    }
    rear_gap = np.tile([10, 20, 40, 80], 2)
    lane = {
        "recording": np.repeat([1, 2], 4),
        "label": np.array([1, 0, 2, 0, 0, 1, 0, 2]),
        "available": np.ones((8, 2), bool),
        "target_rear_gap": np.column_stack((rear_gap, rear_gap + 3)),
        "target_front_gap": np.column_stack((rear_gap + 5, rear_gap + 10)),
    }
    probability = np.tile(
        [[0.8, 0.15, 0.05], [0.7, 0.1, 0.2], [0.6, 0.3, 0.1], [0.9, 0.05, 0.05]],
        (2, 1),
    )
    monkeypatch.setattr(
        calibration,
        "conditional_predictions",
        lambda *args: np.stack(
            (data["acceleration"] + 0.05, data["acceleration"] - 0.02)
        ),
    )
    monkeypatch.setattr(
        calibration, "sampled_lane_probabilities", lambda *args: probability
    )
    samples = 64
    report, numeric = calibration.conditional_metrics(
        data, lane, _population(), {}, bootstrap=samples
    )
    path = tmp_path / "statistics.npz"
    np.savez_compressed(path, **numeric)
    with np.load(path) as saved:
        for row in report["acceleration"]:
            count = saved[row["group"] + "_real"].sum()
            squared_error = saved[row["group"] + "_squared_error_recording_sums"].sum()
            np.testing.assert_allclose(
                np.sqrt(squared_error / count), row["mean_RMSE_mps2"]
            )
        recordings = saved["lane_recordings"]
        draws = np.random.default_rng(calibration.SEED).integers(
            len(recordings), size=(samples, len(recordings))
        )
        for index, variable in enumerate(("target_rear_gap", "target_front_gap")):
            real = saved[variable + "_real_recording"]
            model = saved[variable + "_model_recording"]
            np.testing.assert_allclose(real.sum(0), saved[variable + "_real"])
            np.testing.assert_allclose(model.sum(0), saved[variable + "_model"])
            values = hellinger(real[draws].sum(1), model[draws].sum(1))
            assert values.std(ddof=1) > 0
            np.testing.assert_allclose(
                values.std(ddof=1), report["lane"]["acceptance_SE"][index]
            )
            np.testing.assert_allclose(
                np.quantile(values, [0.025, 0.975]),
                report["lane"]["acceptance_CI95"][index],
            )
