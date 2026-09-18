"""Matched recording-held-out comparison on one shared highD cohort."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from bayesian_ma_idm.src.data import load_ragged_pairs
from bayesian_ma_idm.src.evaluation import _prefix_history, crps_ensemble
from bayesian_ma_idm.src.model import load_posterior, sample_driver_joint
from bayesian_ma_idm.src.reference_kernels import idm as bayesian_idm
from dynamic_ar_idm.model import DynamicIDMState, idm as dynamic_idm
from multi_regime_bidm.src.evaluation import _initial_filter, _models, _rollout as multi_rollout, _style_from_prefix


ROOT = Path(__file__).resolve().parents[1]


def _label(path: str | Path) -> str:
    return str(Path(path).resolve().relative_to(ROOT))


def _bayesian_rollout(
    pair: dict[str, np.ndarray], parameters: np.ndarray, *, model: str,
    prefix: int, horizon: int, process_z: np.ndarray, iid_z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta, sigma, ell = parameters[:5], float(parameters[5]), float(parameters[6])
    iid_sigma = float(parameters[7]) if len(parameters) > 7 else .1
    history = None
    if model == "ma_idm":
        history = _prefix_history(pair, theta, prefix, 5., iid_sigma)
        history.sigma, history.lengthscale, history.memory_seconds = sigma, ell, 5.
    x, speed = float(pair["follower_x"][prefix]), float(pair["follower_v"][prefix])
    position, velocity, acceleration = [], [], []
    action = 0.
    for step in range(horizon):
        absolute = prefix + step
        if step % 5 == 0:
            decision = step // 5
            noise = (
                history.step(absolute * .04, process_z[decision]) + iid_sigma * iid_z[decision]
                if model == "ma_idm" else sigma * process_z[decision]
            )
            leader_speed = float(pair["leader_v"][absolute])
            gap = float(pair["leader_x"][absolute]) - x - float(pair["length_sum"])
            action = float(bayesian_idm(gap, speed, speed - leader_speed, theta) + noise)
        x += speed * .04 + .5 * action * .04**2
        speed = max(0., speed + action * .04)
        position.append(x); velocity.append(speed); acceleration.append(action)
    return np.asarray(position), np.asarray(velocity), np.asarray(acceleration)


def _dynamic_history(pair: dict[str, np.ndarray], theta: np.ndarray, prefix: int, order: int) -> np.ndarray:
    indices = np.arange(0, prefix - 4, 5, dtype=int)
    speed = pair["follower_v"]
    observed = (speed[indices + 5] - speed[indices]) / .2
    residual = observed - dynamic_idm(
        pair["gap"][indices], speed[indices],
        speed[indices] - pair["leader_v"][indices], theta,
    )
    return np.pad(residual[-order:][::-1], (0, max(0, order - len(residual))))[:order]


def _dynamic_rollout(
    pair: dict[str, np.ndarray], theta: np.ndarray, rho: np.ndarray, sigma: float, *,
    prefix: int, horizon: int, process_z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    state = DynamicIDMState(theta, rho, sigma, _dynamic_history(pair, theta, prefix, len(rho)))
    x, speed = float(pair["follower_x"][prefix]), float(pair["follower_v"][prefix])
    position, velocity, acceleration = [], [], []
    for step in range(horizon):
        absolute = prefix + step
        leader_speed = float(pair["leader_v"][absolute])
        gap = float(pair["leader_x"][absolute]) - x - float(pair["length_sum"])
        if step % 5 == 0:
            state.decision(gap, speed, speed - leader_speed, process_z[step // 5])
        action = state.held_acceleration
        x += speed * .04 + .5 * action * .04**2
        speed = max(0., speed + action * .04)
        position.append(x); velocity.append(speed); acceleration.append(action)
    return np.asarray(position), np.asarray(velocity), np.asarray(acceleration)


def evaluate_matched(
    dataset: str | Path,
    b_posterior: str | Path,
    ma_posterior: str | Path,
    dynamic_posterior: str | Path,
    stage_a: str | Path,
    multi_posterior_dir: str | Path,
    output_dir: str | Path,
    *,
    test_recordings: tuple[int, ...] = (26, 36),
    prefix_s: float = 5., horizon_s: float = 3., futures: int = 64,
    seed: int = 20260918,
) -> Path:
    """Evaluate every locally eligible test event under one causal protocol."""
    _, all_pairs = load_ragged_pairs(dataset)
    prefix, horizon = round(prefix_s / .04), round(horizon_s / .04)
    pairs = [pair for pair in all_pairs if int(pair["recording_id"]) in test_recordings
             and len(pair["follower_v"]) >= prefix + horizon + 1]
    if not pairs:
        raise RuntimeError("matched evaluation selected no test pairs")
    b, ma = load_posterior(b_posterior), load_posterior(ma_posterior)
    with np.load(dynamic_posterior, allow_pickle=False) as raw:
        dynamic = {key: raw[key] for key in raw.files}
    stage_a = Path(stage_a)
    labels = np.load(stage_a / "stage_a_offline_labels.npz", allow_pickle=False)
    feature, train_style = labels["style_feature_raw"], labels["style_id"]
    centers = np.asarray([np.mean(feature[train_style == value], axis=0) for value in range(3)])
    feature_mean, feature_scale = np.mean(feature, axis=0), np.maximum(np.std(feature, axis=0), 1.e-8)
    hsmm = _models(stage_a)
    posterior_dir = Path(multi_posterior_dir)
    multi = {}
    pooled = {}
    for style in range(3):
        with np.load(posterior_dir / f"style_{style}_nuts_posterior.npz", allow_pickle=False) as raw:
            multi[style] = {key: raw[key] for key in raw.files}
        with np.load(posterior_dir / f"style_{style}_pooled_nuts_posterior.npz", allow_pickle=False) as raw:
            pooled[style] = {key: raw[key] for key in raw.files}
    names = ("b_idm", "ma_idm", "dynamic_ar5", "pooled_b_idm", "multi_regime")
    shape = (len(pairs), futures, horizon)
    samples = {name: [np.empty(shape) for _ in range(3)] for name in names}
    actual = [np.empty((len(pairs), horizon)) for _ in range(3)]
    styles = np.empty(len(pairs), dtype=int)
    rng = np.random.default_rng(seed)
    decisions = int(np.ceil(horizon / 5))
    for row, pair in enumerate(pairs):
        style = _style_from_prefix(pair, centers, feature_mean, feature_scale, prefix)
        styles[row] = style
        initial_filter = _initial_filter(hsmm[style], pair, prefix)
        actual[0][row] = pair["follower_x"][prefix + 1:prefix + horizon + 1]
        actual[1][row] = pair["follower_v"][prefix + 1:prefix + horizon + 1]
        actual[2][row] = np.diff(pair["follower_v"][prefix:prefix + horizon + 1]) / .04
        for future in range(futures):
            process_z, iid_z, regime_u = rng.standard_normal(decisions), rng.standard_normal(decisions), rng.random(decisions)
            parameter_rng = np.random.default_rng(rng.integers(0, 2**63 - 1))
            generated = {
                "b_idm": _bayesian_rollout(pair, sample_driver_joint(b, parameter_rng), model="b_idm",
                                            prefix=prefix, horizon=horizon, process_z=process_z, iid_z=iid_z),
                "ma_idm": _bayesian_rollout(pair, sample_driver_joint(ma, parameter_rng), model="ma_idm",
                                             prefix=prefix, horizon=horizon, process_z=process_z, iid_z=iid_z),
            }
            dynamic_draw = int(parameter_rng.integers(len(dynamic["sigma_draws"])))
            dynamic_driver = int(parameter_rng.integers(dynamic["theta_draws"].shape[1]))
            generated["dynamic_ar5"] = _dynamic_rollout(
                pair, dynamic["theta_draws"][dynamic_draw, dynamic_driver], dynamic["rho_map"],
                float(dynamic["sigma_draws"][dynamic_draw]), prefix=prefix, horizon=horizon, process_z=process_z,
            )
            multi_draw = int(parameter_rng.integers(len(multi[style]["theta_draws"])))
            pooled_draw = int(parameter_rng.integers(len(pooled[style]["theta_draws"])))
            generated["multi_regime"] = multi_rollout(
                pair, theta=multi[style]["theta_draws"][multi_draw], sigma=multi[style]["sigma_draws"][multi_draw],
                state=initial_filter, model=hsmm[style], prefix_frames=prefix, horizon_frames=horizon,
                innovations=process_z, regime_uniforms=regime_u, filtered=True,
            )
            generated["pooled_b_idm"] = multi_rollout(
                pair, theta=pooled[style]["theta_draws"][pooled_draw], sigma=pooled[style]["sigma_draws"][pooled_draw],
                state=initial_filter, model=hsmm[style], prefix_frames=prefix, horizon_frames=horizon,
                innovations=process_z, regime_uniforms=regime_u, filtered=False,
            )
            for name, values in generated.items():
                for dimension in range(3):
                    samples[name][dimension][row, future] = values[dimension]
    def metrics(values: list[np.ndarray]) -> dict[str, float]:
        low, high = np.quantile(values[0], (.05, .95), axis=1)
        return {
            "position_rmse_m": float(np.sqrt(np.mean((np.mean(values[0], axis=1) - actual[0])**2))),
            "speed_rmse_mps": float(np.sqrt(np.mean((np.mean(values[1], axis=1) - actual[1])**2))),
            "acceleration_crps_mps2": crps_ensemble(values[2], actual[2]),
            "position_90_coverage": float(np.mean((actual[0] >= low) & (actual[0] <= high))),
            "mean_position_interval_width_m": float(np.mean(high - low)),
        }
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    report = {
        "benchmark_id": "highd_shared_218_train25_test26_36_25hz_v1",
        "dataset": _label(dataset),
        "cohort_pairs": len(all_pairs),
        "training_recordings": [25], "training_pairs": len(all_pairs) - len(pairs),
        "test_recordings": list(test_recordings),
        "test_pairs": len(pairs), "prefix_s": prefix_s, "horizon_s": horizon_s, "futures": futures,
        "native_fps": 25, "decision_fps": 5, "common_process_innovations": True,
        "all_eligible_test_pairs_evaluated": True,
        "parameter_policy": "fresh episode-level joint posterior draw; unseen Dynamic driver uses a train-driver population draw",
        "comparison_scope": "matched highD adaptation benchmark, not a replacement for each paper-native reproduction",
        "training_data_policy": {
            "b_idm_ma_idm_dynamic_ar5": "all completed 5 Hz actions from all 182 recording-25 trajectories",
            "pooled_b_idm_multi_regime": "Stage A uses all 182 trajectories; Stage B uses the same retained regime-stratified NUTS observations for both models",
        },
        "comparison_limit": "test data, clocks, origins, futures and metrics are identical; paper-specific inference procedures and Stage-B observation budgets remain model-specific",
        "posterior_artifacts": {
            "b_idm": _label(b_posterior),
            "ma_idm": _label(ma_posterior),
            "dynamic_ar5": _label(dynamic_posterior),
            "multi_regime_stage_a": _label(stage_a / "stage_a_finite_hsmm_models.npz"),
            "multi_regime_stage_b": {
                str(style): _label(posterior_dir / f"style_{style}_nuts_posterior.npz")
                for style in range(3)
            },
        },
        "models": {name: metrics(samples[name]) for name in names},
    }
    path = output / "matched_metrics.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    arrays = {"actual_position": actual[0], "actual_speed": actual[1], "actual_acceleration": actual[2],
              "style_from_prefix": styles}
    for name, values in samples.items():
        for label, value in zip(("position", "speed", "acceleration"), values):
            arrays[f"{name}_{label}"] = value
    np.savez_compressed(output / "matched_predictions.npz", **arrays)
    return path
