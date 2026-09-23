"""Closed-loop 25 Hz stochastic evaluation and paper Table-2 metrics."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from .data import load_pairs
from .fit import decision_arrays
from .model import DynamicIDMState, idm


def crps_ensemble(samples: np.ndarray, truth: np.ndarray) -> float:
    """Empirical CRPS, averaged over a [ensemble, time] array."""
    samples, truth = np.asarray(samples, float), np.asarray(truth, float)
    first = np.mean(np.abs(samples - truth[None, :]), axis=0)
    ordered = np.sort(samples, axis=0); n = len(samples)
    weights = (2 * np.arange(1, n + 1) - n - 1)[:, None]
    second = np.sum(weights * ordered, axis=0) / (n * n)
    return float(np.mean(first - second))


def _history(pair: dict[str, np.ndarray], theta: np.ndarray, anchor_decision: int, order: int) -> np.ndarray:
    """Observed completed residuals before a prediction origin, newest first."""
    values = decision_arrays(pair)
    error = values["action"] - idm(values["gap"], values["speed"], values["closing"], theta)
    completed = error[max(0, anchor_decision - order):anchor_decision]
    return np.pad(completed[::-1], (0, max(0, order - len(completed))))[:order]


def rollout_25hz(pair: dict[str, np.ndarray], theta: np.ndarray, rho: np.ndarray, sigma: float, *,
                  anchor_decision: int, horizon_s: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run a 5 Hz Dynamic IDM decision process through a 25 Hz ballistic plant."""
    decision = np.asarray(pair["decision"], int)
    start = int(decision[anchor_decision]); frames = int(round(horizon_s / .04))
    if start + frames >= len(pair["leader_x"]):
        raise ValueError("prediction horizon exceeds observed leader trajectory")
    state = DynamicIDMState(np.asarray(theta, float), np.asarray(rho, float), float(sigma), _history(pair, theta, anchor_decision, len(rho)))
    x, v = float(pair["follower_x"][start]), float(pair["follower_v"][start])
    pos, speed, action, error = [], [], [], []
    for frame in range(frames):
        absolute = start + frame
        leader_x, leader_v = float(pair["leader_x"][absolute]), float(pair["leader_v"][absolute])
        gap = leader_x - x - float(pair["length_sum"])
        if frame % 5 == 0:
            state.decision(gap, v, v - leader_v, rng.standard_normal())
        # Native plant only constrains non-negative speed.  Crucially, any
        # clamp does not feed back into `residual_history` (generative mode).
        a = state.held_acceleration
        x += v * .04 + .5 * a * .04 ** 2
        v = max(0., v + a * .04)
        pos.append(x); speed.append(v); action.append(a); error.append(state.residual_history[0] if len(rho) else 0.)
    return np.asarray(pos), np.asarray(speed), np.asarray(action), np.asarray(error)


def _interval(values: np.ndarray, rng: np.random.Generator, n: int = 2000) -> list[float]:
    if len(values) < 2:
        return [float(values[0]), float(values[0])]
    means = np.asarray([np.mean(values[rng.integers(0, len(values), len(values))]) for _ in range(n)])
    return [float(np.quantile(means, .025)), float(np.quantile(means, .975))]


def evaluate_paper_cohort(dataset_path: str | Path, posterior_path: str | Path, output_path: str | Path, *,
                          futures: int = 128, max_horizon_s: int = 10, anchor_stride_s: float = 5., seed: int = 20260916,
                          rho_mode: str = "posterior") -> Path:
    """Paper-aligned 1--10 s posterior rollouts using 25 Hz integration."""
    pairs = load_pairs(dataset_path)
    with np.load(posterior_path, allow_pickle=False) as raw:
        posterior = {key: raw[key] for key in raw.files}
    theta_draws, rho_draws, sigma_draws = posterior["theta_draws"], posterior["rho_draws"], posterior["sigma_draws"]
    if rho_mode not in {"posterior", "map"}:
        raise ValueError("rho_mode must be 'posterior' or 'map'")
    max_frames, stride = max_horizon_s * 25, max(1, int(round(anchor_stride_s * 5)))
    rng = np.random.default_rng(seed)
    metrics: dict[str, list[dict[str, float]]] = {str(h): [] for h in range(1, max_horizon_s + 1)}
    representative = None
    for driver, pair in enumerate(pairs):
        decisions = pair["decision"]
        # Keep five seconds of observed decisions before every forecast.
        last = len(decisions) - int(np.ceil(max_horizon_s / .2))
        anchors = range(25, max(25, last), stride)
        per_driver = {str(h): {"a_rmse": [], "v_rmse": [], "s_rmse": [], "a_crps": [], "v_crps": [], "s_crps": []} for h in range(1, max_horizon_s + 1)}
        for anchor in anchors:
            samples = []
            for _ in range(futures):
                draw = int(rng.integers(0, len(sigma_draws)))
                rho = rho_draws[draw] if rho_mode == "posterior" else posterior["rho_map"]
                samples.append(rollout_25hz(pair, theta_draws[draw, driver], rho, sigma_draws[draw], anchor_decision=anchor, horizon_s=max_horizon_s, rng=rng))
            position, velocity, acceleration, _ = (np.asarray([sample[index] for sample in samples]) for index in range(4))
            start = int(decisions[anchor])
            true_position = pair["follower_x"][start + 1:start + max_frames + 1]
            true_velocity = pair["follower_v"][start + 1:start + max_frames + 1]
            # Paper actions are 5 Hz transitions.  Select the held action at
            # each native decision and infer the matching observed transition.
            true_action = (pair["follower_v"][start + 5:start + max_frames + 1:5] - pair["follower_v"][start:start + max_frames:5]) / .2
            predicted_action = acceleration[:, 4::5]
            for horizon in range(1, max_horizon_s + 1):
                n, a_n, key = horizon * 25, horizon * 5, str(horizon)
                vmean = np.mean(velocity[:, :n], axis=0)
                amean = np.mean(predicted_action[:, :a_n], axis=0)
                gap_truth = pair["leader_x"][start + 1:start + n + 1] - true_position[:n] - float(pair["length_sum"])
                gap_sample = pair["leader_x"][start + 1:start + n + 1][None, :] - position[:, :n] - float(pair["length_sum"])
                per_driver[key]["a_rmse"].append(float(np.sqrt(np.mean((amean - true_action[:a_n]) ** 2))))
                per_driver[key]["v_rmse"].append(float(np.sqrt(np.mean((vmean - true_velocity[:n]) ** 2))))
                per_driver[key]["s_rmse"].append(float(np.sqrt(np.mean((np.mean(gap_sample, axis=0) - gap_truth) ** 2))))
                per_driver[key]["a_crps"].append(crps_ensemble(predicted_action[:, :a_n], true_action[:a_n]))
                per_driver[key]["v_crps"].append(crps_ensemble(velocity[:, :n], true_velocity[:n]))
                per_driver[key]["s_crps"].append(crps_ensemble(gap_sample, gap_truth))
            if representative is None:
                representative = {"driver": driver, "pair_no": int(pair["pair_no"]), "anchor": anchor, "start": start,
                                  "position": position, "velocity": velocity, "acceleration": acceleration,
                                  "true_position": true_position, "true_velocity": true_velocity, "true_action": true_action,
                                  "leader_x": pair["leader_x"][start + 1:start + max_frames + 1], "length_sum": float(pair["length_sum"])}
        for key, value in per_driver.items():
            if value["a_rmse"]:
                metrics[key].append({name: float(np.mean(samples)) for name, samples in value.items()})
    fit_backend = str(posterior["fit_backend"].item())
    if rho_mode == "posterior":
        rho_note = ("all rho Laplace draws retained, including non-stationary draws"
                    if "laplace" in fit_backend else
                    "rho sampled only from the supplied posterior artifact; no stationarity filtering")
    else:
        rho_note = "stable MAP rho held fixed; deployment diagnostic, not a draw from the original posterior"
    report = {"model": "dynamic_ar_idm", "fit_backend": fit_backend, "dataset": str(dataset_path),
              "native_fps": 25, "decision_fps": 5, "decision_action_hold_frames": 5, "futures": futures, "anchor_stride_s": anchor_stride_s,
              "pairs": len(pairs), "anchors": int(sum(len(range(25, max(25, len(pair["decision"]) - int(np.ceil(max_horizon_s / .2))), stride)) for pair in pairs)),
              "rho_mode": rho_mode, "evaluation": "in-sample driver-specific posterior rolling origins; comparison to Table 2 is directional, not OOF",
              "rho_mode_note": rho_note, "horizons": {},
              "paper_table_2_ar5_real_units": {"a_rmse": .166, "v_rmse": .265, "s_rmse": .429, "a_crps": .095, "v_crps": .149, "s_crps": .217}}
    bootstrap = np.random.default_rng(seed + 1)
    for key, values in metrics.items():
        report["horizons"][key] = {name: {"mean": float(np.mean([row[name] for row in values])), "driver_bootstrap_95": _interval(np.asarray([row[name] for row in values]), bootstrap)} for name in values[0]}
    output_path = Path(output_path); output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if representative is not None:
        np.savez_compressed(output_path.with_name("representative_rollout.npz"), **representative)
    return output_path
