from copy import deepcopy

import numpy as np

from hierarchical_world_model.src.npc_experiments.search import (
    MAIN_COMPONENTS,
    continuous_subset,
    exposure_summary,
    load_candidate,
    objective,
    paired_closed_acceptable,
    paired_objective_se,
    refinement_points,
    save_candidate,
    shifted_population,
)


def closed():
    metrics = {name: 0.2 for name in MAIN_COMPONENTS}
    return {
        "components": {**metrics, "acceleration": 0.4},
        "bootstrap": {
            **{name: [0.2] * 20 for name in MAIN_COMPONENTS},
            "acceleration": [0.4] * 20,
            "contact_exposure_ratio": [0.01] * 20,
            "boundary_exposure_ratio": [0.02] * 20,
            "road_collision_run_fraction": [0.0] * 20,
        },
        "exposure": {
            "contact_exposure_ratio": 0.01,
            "boundary_exposure_ratio": 0.02,
            "road_collision_run_fraction": 0.0,
        },
    }


def test_exposure_uses_matched_real_vehicle_time():
    real = np.array([100.0, 10, 1, 0, 0, 100])
    first = exposure_summary(np.array([200.0, 10, 1, 4, 2, 100]), real)
    second = exposure_summary(np.array([100.0, 10, 1, 4, 2, 100]), real)
    assert first["contact_exposure_ratio"] == second["contact_exposure_ratio"] == 0.04


def test_val_guards_cover_each_main_distribution_and_keep_failures():
    base = closed()
    assert paired_closed_acceptable(base, base, [{"status": "complete"}])[0]
    changed = deepcopy(base)
    changed["components"]["acceleration"] = 1
    assert paired_closed_acceptable(changed, base, [{"status": "complete"}])[0]
    for name in MAIN_COMPONENTS:
        changed = deepcopy(base)
        changed["components"][name] += 0.01
        assert not paired_closed_acceptable(changed, base, [{"status": "complete"}])[0]
    assert not paired_closed_acceptable(base, base, [{"status": "failed"}])[0]


def test_objective_uses_all_four_physical_targets_for_both_search_arms():
    local = {
        "acceleration": [{"Hellinger": 0.2}],
        "lane": {"Brier": 0.01, "acceptance_Hellinger": [0.2, 0.3]},
    }
    weights = {
        **{name: 1 for name in MAIN_COMPONENTS},
        "acceleration": 0.1,
        "local": 0.1,
        "change": 0.01,
    }
    score = objective(
        closed(), local, None, np.zeros(16), np.tile([-1, 1], (16, 1)), weights
    )
    expected = 0.8 + 0.04 + 0.1 * np.mean([0.2, 0.005, 0.25])
    assert abs(score - expected) < 1e-12


def test_selection_uncertainty_uses_paired_differences():
    first, second = {"closed": closed()}, {"closed": closed()}
    for name in (*MAIN_COMPONENTS, "acceleration"):
        first["closed"]["bootstrap"][name] = np.linspace(0.1, 0.9, 20)
        second["closed"]["bootstrap"][name] = first["closed"]["bootstrap"][name] - 0.02
    weights = {name: 1 for name in (*MAIN_COMPONENTS, "acceleration")}
    assert paired_objective_se(first, second, weights) < 1e-15


def test_population_shift_preserves_correlation_and_candidate_roundtrip(tmp_path):
    covariance = np.eye(5) + 0.2
    population = {
        "log_mean": np.zeros(5),
        "log_cov": covariance,
        "residual_std": np.array(0.3),
        "fit_driver_count": np.array(10),
        "regularization": np.array(0.1),
    }
    vector = np.r_[np.full(5, 0.1), 1.2, 0, 1]
    result = shifted_population(population, vector)
    np.testing.assert_allclose(result["log_cov"], covariance * 1.44)
    save_candidate(tmp_path, "population", result, {"bias": -4}, "population", vector)
    restored, _ = load_candidate(tmp_path, "population")
    np.testing.assert_array_equal(restored["log_mean"], result["log_mean"])
    assert not np.array_equal(restored["log_mean"], population["log_mean"])


def test_refinement_contains_inactive_longitudinal_anchor():
    bounds = np.tile([-1.0, 1.0], (16, 1))
    points = refinement_points(np.zeros(16), bounds, 0, 8)
    assert points.shape == (8, 16)
    assert (points >= -1).all() and (points <= 1).all()
    assert (points[0, 8:10] == 0).all()


def test_refinement_count_preserves_the_default_sobol_sequence():
    from scipy.stats import qmc

    from hierarchical_world_model.src.npc_experiments.search import SEED

    center = np.zeros(8)
    bounds = np.tile([-1.0, 1.0], (8, 1))
    expected = qmc.Sobol(8, scramble=True, seed=SEED + 10).random_base2(3) - 0.5
    expected[0] = center
    np.testing.assert_array_equal(refinement_points(center, bounds, 0, 8), expected)
    assert refinement_points(center, bounds, 0, 4).shape == (4, 8)


def test_temporal_learning_subset_keeps_contiguous_physical_histories():
    data = {
        "recording": np.repeat([1, 2], 400),
        "vehicle": np.tile(np.repeat(np.arange(10), 40), 2),
        "frame": np.tile(np.arange(40) * 5, 20),
        "free": np.zeros(800),
        "changing": np.zeros(800),
    }
    selected = continuous_subset(data, 0.1)
    assert len(selected["frame"]) == 80
    for recording in np.unique(selected["recording"]):
        frames = selected["frame"][selected["recording"] == recording]
        np.testing.assert_array_equal(np.diff(frames), np.full(39, 5))
