import json

import numpy as np

from active_inference_driver.config import ActiveInferenceConfig
from active_inference_driver.model import ActiveInferenceDriver
from bayesian_ma_idm.src.driver import BayesianIDMDriver
from dynamic_ar_idm.model import DynamicIDMDriver
from multi_regime_bidm.src.driver import MultiRegimeIDMDriver, load_hsmm_models

from driver_reproduction.scorecard import ROOT


def _json(relative_path: str) -> dict:
    return json.loads((ROOT / relative_path).read_text(encoding="utf-8"))


def test_retained_evidence_keeps_successes_and_failures_explicit() -> None:
    ma = _json(
        "bayesian_ma_idm/evidence/paper_reference/nuts/author_reference_report.json"
    )
    assert ma["divergences"] == 0 and ma["max_rhat"] < 1.01

    dynamic = _json(
        "dynamic_ar_idm/artifacts/full_posterior/"
        "dynamic_ar5_full_posterior_stationarity.json"
    )
    assert dynamic["rho_map_spectral_radius"] < 1.0
    assert dynamic["nonstationary_fraction"] > 0.0

    multi_posterior = _json(
        "multi_regime_bidm/evidence/posterior/posterior_diagnostics.json"
    )
    multi_acceptance = _json(
        "multi_regime_bidm/evidence/heldout/recording_heldout_acceptance_audit.json"
    )
    assert multi_posterior["sampler_diagnostics_passed"]
    assert not multi_acceptance["accepted_calibrated_stochastic_driver"]

    active = _json("active_inference_driver/artifacts/full_highd/validation_status.json")
    assert active["native_paper_human_metric"]["status"] == "passed"
    assert active["official_pomdp_external_25hz_wrapper"]["status"] == "passed"
    assert active["longitudinal_adapter_25hz_bridge"]["status"] == "failed"

    matched = _json(
        "results/driver_reproduction/matched_highd/matched_metrics.json"
    )
    assert matched["cohort_pairs"] == 218
    assert matched["training_pairs"] == 182
    assert matched["test_pairs"] == 36
    assert matched["all_eligible_test_pairs_evaluated"]
    assert set(matched["models"]) == {
        "b_idm", "ma_idm", "dynamic_ar5", "pooled_b_idm", "multi_regime"
    }


def test_deployment_posteriors_use_the_complete_source_cohort() -> None:
    for model in ("b_idm", "ma_idm"):
        report = _json(
            f"bayesian_ma_idm/evidence/deployment/{model}_all_251.json"
        )
        assert report["pairs"] == 251
        assert report["training_recordings"] == "all qualifying recordings"
        assert report["training_fraction_per_pair"] == 1.0


def test_retained_posteriors_instantiate_stateful_online_drivers() -> None:
    observation = {"gap_m": 25.0, "speed_mps": 18.0, "leader_speed_mps": 17.5}

    ma = BayesianIDMDriver.from_population_posterior(
        ROOT / "bayesian_ma_idm/evidence/population/ma_idm_holdout_25.npz",
        model="ma_idm",
        seed=5,
    )
    assert np.isfinite(ma.decision(**observation).requested_acceleration_mps2)

    dynamic = DynamicIDMDriver.from_posterior(
        ROOT / "dynamic_ar_idm/artifacts/full_posterior/dynamic_ar5_full_posterior.npz",
        seed=5,
    )
    assert np.isfinite(dynamic.decision(**observation))

    models = load_hsmm_models(
        ROOT / "multi_regime_bidm/evidence/segmentation/stage_a_finite_hsmm_models.npz"
    )
    multi = MultiRegimeIDMDriver.from_posterior(
        models[1],
        ROOT / "multi_regime_bidm/evidence/posterior/style_1_nuts_posterior.npz",
        seed=5,
    )
    multi.reset(np.asarray([[25.0, 18.0, 0.5]]), seed=5)
    assert np.isfinite(multi.decision(**observation).requested_acceleration_mps2)

    active = ActiveInferenceDriver(
        ActiveInferenceConfig().reduced(plans=4, iterations=1, particles=4)
    )
    active.reset(
        gap_m=25.0,
        ego_speed_mps=18.0,
        leader_speed_mps=17.5,
        seed=5,
    )
    assert np.isfinite(
        active.decide(gap_m=25.0, ego_speed_mps=18.0, leader_speed_mps=17.5).action
    )
