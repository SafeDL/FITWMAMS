"""Run the released POMDP against a 25 Hz external bicycle plant.

This is a native-condition bridge test, not a second human-data fit.  The
official POMDP receives only the current external state at 5 Hz; every
returned request is held for exactly five 25 Hz ticks.  The 25 Hz plant uses
the released Bicycle class at runtime and discards its own source environment
observations, so scheduled-source future state never becomes an agent input.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np


DT = .04
TICKS = 5
EXPERIMENTS = (7, 11, 15, 19, 23)  # 10 m/s, 1.42--3.42 s initial THW


def onset(acceleration: np.ndarray, leader_acceleration: np.ndarray) -> float | None:
    event = np.flatnonzero(leader_acceleration < -1.)
    response = np.flatnonzero(acceleration < -1.)
    if not len(event) or not len(response):
        return None
    after = response[response >= event[0]]
    return None if not len(after) else float((after[0] - event[0]) * DT)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--following-dir", type=Path, required=True)
    parser.add_argument("--native-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    package_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(package_root))
    from official_wrapper import OfficialPOMDP25Hz

    # Loaded only from the pinned external checkout; no official source is
    # copied into this repository.
    sys.path.insert(0, str(args.source_dir))
    import torch
    from src.common.bicycle import Bicycle

    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("this released Bicycle implementation requires CUDA")
    device = torch.device("cuda", 0)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    representative = None
    for experiment in EXPERIMENTS:
        path = args.native_results / "Results_rear_end" / f"Exp_{experiment}" / f"Exp_{experiment}.pkl"
        with path.open("rb") as handle:
            native = pickle.load(handle)
        eta0 = native["eta"][0, 0]
        batch = native["eta"].shape[0]
        driver = OfficialPOMDP25Hz(source_dir=args.source_dir, following_dir=args.following_dir,
                                   device=args.device, batch_size=batch)
        driver.reset(gap_m=float(eta0[5] - eta0[0] - 4.2), ego_speed_mps=float(eta0[4]),
                     target_speed_mps=float(eta0[9]), seed=0)
        bicycle = Bicycle(lf=2.1, lr=2.1, d=1.72, a_max=8., w_max=1.22, dt=DT).to(device)
        ego = torch.as_tensor(np.repeat(eta0[None, :5], batch, axis=0), device=device)
        target = torch.as_tensor(np.repeat(eta0[None, 5:10], batch, axis=0), device=device)
        target_acceleration = torch.zeros(batch, device=device)
        countdown = torch.full((batch,), .6, device=device)
        acceleration: list[np.ndarray] = []
        steering: list[np.ndarray] = []
        leader_acceleration: list[np.ndarray] = []
        lateral: list[np.ndarray] = []
        for _ in range(60):  # same 12 s physical horizon as native rear-end run
            ego_np, target_np = ego.detach().cpu().numpy(), target.detach().cpu().numpy()
            observation = np.column_stack((ego_np, target_np, target_acceleration.detach().cpu().numpy(),
                                           np.zeros(batch, dtype=np.float32)))
            decision = driver.step(observation)
            requested = torch.as_tensor(np.column_stack((decision.acceleration_mps2, decision.steering_rate_rps)),
                                        dtype=torch.float32, device=device)
            # The released rear-end target updates its scheduled jerk at the
            # native boundary.  Preserve that behavior, then zero-order hold
            # the resulting target control across five 25 Hz plant ticks.
            countdown = torch.maximum(countdown - .2, torch.zeros_like(countdown))
            target_acceleration = torch.where(countdown <= 0.,
                torch.clamp(target_acceleration - 10. * .2, min=-6., max=8.), target_acceleration)
            target_request = torch.stack((target_acceleration, torch.zeros_like(target_acceleration)), dim=-1)
            for _ in range(TICKS):
                with torch.no_grad():
                    ego = bicycle.forward(ego, requested, dt=DT)
                    target = bicycle.forward(target, target_request, dt=DT)
                acceleration.append(decision.acceleration_mps2.copy())
                steering.append(decision.steering_rate_rps.copy())
                leader_acceleration.append(target_acceleration.detach().cpu().numpy().copy())
                lateral.append(ego[:, 1].detach().cpu().numpy().copy())
        acc = np.asarray(acceleration).T
        steer = np.asarray(steering).T
        lead_acc = np.asarray(leader_acceleration).T
        lat = np.asarray(lateral).T
        native_acc = np.repeat(native["a_cont"][0, :, :, 0].T, TICKS, axis=1)
        native_lead_acc = np.repeat(native["eta"][:, :, 10], TICKS, axis=1)
        wrapped_onsets = [onset(acc[row], lead_acc[row]) for row in range(batch)]
        native_onsets = [onset(native_acc[row], native_lead_acc[row]) for row in range(batch)]
        comparable = [(wrapped, original) for wrapped, original in zip(wrapped_onsets, native_onsets)
                      if wrapped is not None and original is not None]
        response_delta = [wrapped - original for wrapped, original in comparable]
        record = {
            "experiment": experiment, "initial_speed_mps": float(eta0[4]),
            "initial_thw_s": float(eta0[5] / eta0[9]), "replicates": batch,
            "wrapped_response_s_median": float(np.median([x for x in wrapped_onsets if x is not None])),
            "native_response_s_median": float(np.median([x for x in native_onsets if x is not None])),
            "response_delta_s_median": float(np.median(response_delta)),
            "response_delta_s_max_abs": float(np.max(np.abs(response_delta))),
            "max_lateral_displacement_m": float(np.max(np.abs(lat))),
            "action_hold_exact_five_ticks": bool(np.array_equal(acc[:, 0::TICKS], acc[:, 1::TICKS])
                                                 and np.array_equal(acc[:, 0::TICKS], acc[:, 2::TICKS])
                                                 and np.array_equal(acc[:, 0::TICKS], acc[:, 3::TICKS])
                                                 and np.array_equal(acc[:, 0::TICKS], acc[:, 4::TICKS])),
        }
        records.append(record)
        if representative is None:
            representative = (acc[0], native_acc[0], lead_acc[0])

    assert representative is not None
    time = np.arange(representative[0].shape[0]) * DT
    fig, axis = plt.subplots(figsize=(9, 3.4), layout="constrained")
    axis.plot(time, representative[2], color="tab:orange", label="external leader acceleration")
    axis.plot(time, representative[0], color="tab:blue", label="official POMDP / 25 Hz hold")
    axis.plot(time, representative[1], color="black", ls="--", label="native 0.2 s execution")
    axis.set(xlabel="time (s)", ylabel="acceleration (m/s²)", title="Native-condition 25 Hz bridge")
    axis.legend(fontsize=8)
    fig.savefig(args.output_dir / "official_25hz_native_bridge.png", dpi=160)
    plt.close(fig)
    response_errors = [record["response_delta_s_max_abs"] for record in records]
    report = {
        "model": "official_pomdp_external_25hz_wrapper",
        "source_code_copied": False, "native_dt_s": .2, "plant_dt_s": DT,
        "plant_ticks_per_decision": TICKS, "native_horizon_s": 6.,
        "test_conditions": records,
        "response_time_gate_s": .2,
        "maximum_condition_response_time_difference_s": float(max(response_errors)),
        # Float32/decimal conversion can express an exact .2 s boundary as
        # 0.20000000000000018; this tolerance is numerical only, not an
        # additional behavioral allowance.
        "gate_numerical_tolerance_s": 1e-8,
        "passes_all_conditions_one_native_tick": bool(max(response_errors) <= .2 + 1e-8),
        "future_action_firewall": "The POMDP receives only the current external state at each 0.2 s boundary; target trajectory is used only by the 25 Hz external plant.",
        "lateral_note": "The external bridge executes both official acceleration and steering-rate requests; its source-style straight-road target is a bridge plant, not highD geometry.",
    }
    (args.output_dir / "official_25hz_bridge.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
