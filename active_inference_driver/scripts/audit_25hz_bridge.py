"""Compare native rear-end braking timings with the 25 Hz longitudinal adapter.

This is a timing-contract audit, not a replacement for the paper's human
metric.  Native response times were extracted by the released author script;
adapter onset is intentionally a transparent action-onset proxy because the
adapter has no steering action dimension.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from active_inference_driver.config import ActiveInferenceConfig
from active_inference_driver.model import ActiveInferenceDriver


CONDITIONS = ((1.5, 10.0, 11), (2.0, 10.0, 15), (2.5, 10.0, 19),
              (3.0, 10.0, 23), (3.5, 10.0, 27))


def _rollout(time_gap: float, speed: float, seed: int, config: ActiveInferenceConfig) -> dict[str, np.ndarray]:
    """Run the published jerked-leader setup through a 25 Hz external plant."""
    driver = ActiveInferenceDriver(config)
    driver.reset(gap_m=time_gap * speed, ego_speed_mps=speed, leader_speed_mps=speed, seed=seed)
    ego_speed, leader_speed = speed, speed
    ego_x, leader_x, leader_acceleration = 0.0, time_gap * speed + 4.2, 0.0
    held_action = config.coast_acceleration_mps2
    actions, ego_speeds, leader_speeds = [], [], []
    for tick in range(300):  # 12 s, matching the native T=60 at 0.2 s.
        time_s = tick * config.native_dt_s
        gap = leader_x - ego_x - 4.2
        if tick % config.native_ticks_per_decision == 0:
            held_action = driver.decide(gap_m=gap, ego_speed_mps=ego_speed,
                                        leader_speed_mps=leader_speed).action
        # The released rear-end process updates target acceleration at its
        # 0.2 s native clock, not at every external 25 Hz plant tick.  Hold
        # that jerked acceleration over the following five plant ticks.
        if tick % config.native_ticks_per_decision == 0 and time_s >= .6:
            leader_acceleration = max(-8.0, leader_acceleration - 2.0)
        ego_x += ego_speed * config.native_dt_s + .5 * held_action * config.native_dt_s ** 2
        leader_x += leader_speed * config.native_dt_s + .5 * leader_acceleration * config.native_dt_s ** 2
        ego_speed = max(0.0, ego_speed + held_action * config.native_dt_s)
        leader_speed = max(0.0, leader_speed + leader_acceleration * config.native_dt_s)
        actions.append(held_action); ego_speeds.append(ego_speed); leader_speeds.append(leader_speed)
    actions = np.asarray(actions)
    after_brake = np.flatnonzero((np.arange(len(actions)) * config.native_dt_s >= .6) & (actions < -1.0))
    onset = None if not len(after_brake) else float(after_brake[0] * config.native_dt_s - .6)
    return {"action": actions, "ego_speed": np.asarray(ego_speeds),
            "leader_speed": np.asarray(leader_speeds), "onset_s": onset}


def _experiment_row(extracted: np.ndarray, experiment: int) -> int:
    rows = np.flatnonzero(extracted[6, :, 0].astype(int) == experiment)
    if len(rows) != 1:
        raise RuntimeError(f"expected one extracted row for experiment {experiment}, got {len(rows)}")
    return int(rows[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=32)
    args = parser.parse_args()
    extracted = np.load(args.native_results / "Results_rear_end" / "Extracted_data.npy")
    config = ActiveInferenceConfig(seed=0)
    rows, representative = [], None
    for time_gap, speed, experiment in CONDITIONS:
        native_rt = extracted[3, _experiment_row(extracted, experiment)]
        native_mean = float(np.nanmean(native_rt))
        samples = [_rollout(time_gap, speed, seed, config) for seed in range(args.replicates)]
        onset = np.array([sample["onset_s"] for sample in samples], dtype=float)
        adapter_mean = float(np.nanmean(onset)) if np.isfinite(onset).any() else None
        delta = None if adapter_mean is None else adapter_mean - native_mean
        rows.append({"time_gap_s": time_gap, "speed_mps": speed, "native_experiment": experiment,
                     "native_response_mean_s": native_mean, "native_response_n": int(np.isfinite(native_rt).sum()),
                     "adapter_onset_proxy_mean_s": adapter_mean,
                     "adapter_onset_proxy_n": int(np.isfinite(onset).sum()),
                     "adapter_minus_native_s": delta,
                     "within_one_native_tick": delta is not None and abs(delta) <= .2})
        if representative is None:
            representative = samples
    assert representative is not None
    report = {
        "contract": {"plant_hz": 25, "decision_dt_s": .2, "plant_ticks_per_decision": 5,
                     "internal_horizon_s": 6.0, "no_steering_adaptation": True},
        "native_metric": "released piecewise-linear extractor", 
        "adapter_metric": "first 25 Hz action below -1 m/s^2 after leader braking; timing proxy only",
        "adapter_replicates": args.replicates,
        "rows": rows,
        "passes_all_one_tick": all(row["within_one_native_tick"] for row in rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    time = np.arange(300) * .04
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True, layout="constrained")
    actions = np.asarray([sample["action"] for sample in representative])
    speeds = np.asarray([sample["ego_speed"] for sample in representative])
    axes[0].fill_between(time, np.quantile(actions, .1, axis=0), np.quantile(actions, .9, axis=0),
                         color="tab:blue", alpha=.25, label="adapter 10–90%")
    axes[0].plot(time, actions.mean(axis=0), color="tab:blue", label="adapter mean action")
    axes[0].axvline(.6, color="crimson", ls="--", label="leader brake")
    axes[0].set_ylabel("acceleration (m/s²)"); axes[0].legend(fontsize=8)
    axes[1].fill_between(time, np.quantile(speeds, .1, axis=0), np.quantile(speeds, .9, axis=0),
                         color="tab:blue", alpha=.25, label="adapter ego 10–90%")
    axes[1].plot(time, representative[0]["leader_speed"], color="tab:orange", label="external leader")
    axes[1].set_ylabel("speed (m/s)"); axes[1].set_xlabel("time (s)"); axes[1].legend(fontsize=8)
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, dpi=160); plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
