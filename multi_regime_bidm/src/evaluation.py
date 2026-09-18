"""Recording-held-out causal evaluation for the documented multi-regime B-IDM."""
from __future__ import annotations

from pathlib import Path
import json
import numpy as np

from bayesian_ma_idm.src.evaluation import crps_ensemble
from bayesian_ma_idm.src.reference_kernels import idm
from .hsmm import FiniteHSMM
from .online_filter import filter_step, initialize_filter, regime_posterior
from .style import decision_grid, style_features


def _models(path: Path) -> list[FiniteHSMM]:
    data = np.load(path / "stage_a_finite_hsmm_models.npz", allow_pickle=False)
    return [FiniteHSMM(data["means"][style], data["variances"][style], data["transition"][style], data["initial"][style],
                       data["duration_lambda"][style], int(data["duration_max"]), data["observation_mean"][style],
                       data["observation_scale"][style]) for style in range(3)]


def _style_from_prefix(pair: dict[str, np.ndarray], centers: np.ndarray, mean: np.ndarray, scale: np.ndarray,
                       prefix_frames: int) -> int:
    prefix = {key: np.asarray(value[:prefix_frames + 1]) for key, value in pair.items() if isinstance(value, np.ndarray)}
    feature = style_features(prefix)
    return int(np.argmin(np.sum(((feature - centers) / scale) ** 2, axis=1)))


def _initial_filter(model: FiniteHSMM, pair: dict[str, np.ndarray], prefix_frames: int):
    observations, _ = decision_grid(pair)
    # The rollout begins at ``prefix_frames``.  Its frame-start observation is
    # already available and may update the regime belief before that action;
    # include it without reading any later frame.
    count = prefix_frames // 5 + 1
    state = initialize_filter(model, observations[0])
    for observation in observations[1:count]:
        state = filter_step(model, state, observation)
    return state


def _rollout(pair: dict[str, np.ndarray], *, theta: np.ndarray, sigma: np.ndarray, state, model: FiniteHSMM,
             prefix_frames: int, horizon_frames: int, innovations: np.ndarray,
             regime_uniforms: np.ndarray, filtered: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dt = .04; x = float(pair["follower_x"][prefix_frames]); v = float(pair["follower_v"][prefix_frames])
    position, speed, acceleration = [], [], []
    static_theta = np.asarray(theta, float)
    static_sigma = float(np.asarray(sigma)) if not filtered else 0.0
    for step in range(horizon_frames):
        absolute = prefix_frames + step
        if step % 5 == 0:
            decision = step // 5
            cumulative = np.cumsum(regime_posterior(state))
            regime = (int(min(np.searchsorted(cumulative, regime_uniforms[decision], side="right"), 2))
                      if filtered else 0)
            current_theta = theta[regime] if filtered else static_theta
            current_sigma = float(sigma[regime]) if filtered else static_sigma
            noise = current_sigma * innovations[decision]
        leader_x, leader_v = float(pair["leader_x"][absolute]), float(pair["leader_v"][absolute])
        gap = max(.1, leader_x - x - float(pair["length_sum"]))
        a = float(idm(gap, v, v - leader_v, current_theta) + noise)
        x += v * dt + .5 * a * dt * dt; v = max(0., v + a * dt)
        position.append(x); speed.append(v); acceleration.append(a)
        if filtered and (step + 1) % 5 == 0 and step + 1 < horizon_frames:
            next_absolute = absolute + 1
            next_leader_x = float(pair["leader_x"][next_absolute])
            next_leader_v = float(pair["leader_v"][next_absolute])
            next_gap = max(.1, next_leader_x - x - float(pair["length_sum"]))
            state = filter_step(model, state, np.asarray((next_gap, v, v - next_leader_v)))
    return np.asarray(position), np.asarray(speed), np.asarray(acceleration)


def evaluate(dataset_path: str | Path, stage_a: str | Path, posterior_dir: str | Path, output_dir: str | Path, *,
             test_recordings: tuple[int, ...] = (26, 36), prefix_s: float = 5., horizon_s: float = 3.,
             futures: int = 64, seed: int = 20260915) -> Path:
    from bayesian_ma_idm.src.data import load_ragged_pairs
    _, pairs = load_ragged_pairs(dataset_path); stage_a, posterior_dir, output = Path(stage_a), Path(posterior_dir), Path(output_dir)
    labels = np.load(stage_a / "stage_a_offline_labels.npz", allow_pickle=False)
    feature = labels["style_feature_raw"]; style = labels["style_id"]
    centers = np.asarray([np.mean(feature[style == value], axis=0) for value in range(3)])
    mean, scale = np.mean(feature, axis=0), np.maximum(np.std(feature, axis=0), 1.e-8)
    models = _models(stage_a); prefix, horizon = int(round(prefix_s / .04)), int(round(horizon_s / .04))
    eligible = [pair for pair in pairs if int(pair["recording_id"]) in test_recordings and len(pair["follower_v"]) >= prefix + horizon + 1]
    if not eligible:
        raise RuntimeError("no held-out pairs satisfy prefix plus horizon")
    shape = (len(eligible), futures, horizon); filtered = [np.empty(shape) for _ in range(3)]; pooled = [np.empty(shape) for _ in range(3)]
    actual = [np.empty((len(eligible), horizon)) for _ in range(3)]; selected_style = []
    rng = np.random.default_rng(seed)
    for row, pair in enumerate(eligible):
        style_id = _style_from_prefix(pair, centers, mean, scale, prefix); selected_style.append(style_id)
        initial = _initial_filter(models[style_id], pair, prefix)
        posterior = np.load(posterior_dir / f"style_{style_id}_nuts_posterior.npz", allow_pickle=False)
        theta_draws, sigma_draws = posterior["theta_draws"], posterior["sigma_draws"]
        pooled_posterior = np.load(
            posterior_dir / f"style_{style_id}_pooled_nuts_posterior.npz",
            allow_pickle=False,
        )
        pooled_theta_draws = pooled_posterior["theta_draws"]
        pooled_sigma_draws = pooled_posterior["sigma_draws"]
        actual[0][row] = pair["follower_x"][prefix + 1:prefix + horizon + 1]
        actual[1][row] = pair["follower_v"][prefix + 1:prefix + horizon + 1]
        actual[2][row] = np.diff(pair["follower_v"][prefix:prefix + horizon + 1]) / .04
        for future in range(futures):
            draw = int(rng.integers(len(theta_draws)))
            pooled_draw = int(rng.integers(len(pooled_theta_draws)))
            decisions = int(np.ceil(horizon / 5))
            innovations = rng.standard_normal(decisions)
            regime_uniforms = rng.random(decisions)
            specifications = (
                (filtered, True, theta_draws[draw], sigma_draws[draw]),
                (pooled, False, pooled_theta_draws[pooled_draw], pooled_sigma_draws[pooled_draw]),
            )
            for destination, active, theta, sigma in specifications:
                values = _rollout(
                    pair,
                    theta=theta,
                    sigma=sigma,
                    state=initial,
                    model=models[style_id],
                    prefix_frames=prefix,
                    horizon_frames=horizon,
                    innovations=innovations,
                    regime_uniforms=regime_uniforms,
                    filtered=active,
                )
                for dimension in range(3): destination[dimension][row, future] = values[dimension]
    output.mkdir(parents=True, exist_ok=True)
    artifact = output / "recording_heldout_evaluation.npz"
    np.savez_compressed(artifact, filtered_position=filtered[0], filtered_speed=filtered[1], filtered_acceleration=filtered[2],
                        pooled_position=pooled[0], pooled_speed=pooled[1], pooled_acceleration=pooled[2], actual_position=actual[0],
                        actual_speed=actual[1], actual_acceleration=actual[2], style_from_prefix=np.asarray(selected_style),
                        test_recordings=np.asarray(test_recordings), prefix_s=np.asarray(prefix_s), horizon_s=np.asarray(horizon_s), dt_s=np.asarray(.04))
    def metrics(samples: list[np.ndarray]) -> dict[str, float]:
        low, high = np.quantile(samples[0], (.05, .95), axis=1)
        return {"position_rmse_m": float(np.sqrt(np.mean((np.mean(samples[0], axis=1) - actual[0]) ** 2))),
                "speed_rmse_mps": float(np.sqrt(np.mean((np.mean(samples[1], axis=1) - actual[1]) ** 2))),
                "acceleration_crps_mps2": crps_ensemble(samples[2], actual[2]),
                "position_90_coverage": float(np.mean((actual[0] >= low) & (actual[0] <= high))),
                "mean_position_interval_width_m": float(np.mean(high - low))}
    report = {"kind": "recording-held-out causal comparison", "calibration": "Stage-B small-budget NUTS documented adaptation",
              "test_recordings": list(test_recordings), "test_pairs": len(eligible), "prefix_style": "K-means nearest center using completed 5s prefix only",
              "regime": "explicit-duration filter using observed prefix; future filter updates use simulated causal state only",
              "pooled_baseline": "separately fitted style-conditional pooled B-IDM on the identical Stage-B observations and NUTS budget",
              "common_random_numbers": True,
              "filtered_hierarchical": metrics(filtered), "pooled_b_idm": metrics(pooled),
              "warning": "only 36 held-out events are available locally; compare directions, not population-significance claims"}
    (output / "recording_heldout_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return artifact
