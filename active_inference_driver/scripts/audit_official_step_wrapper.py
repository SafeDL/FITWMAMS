"""Audit one stepwise official-POMDP instance against a native saved run.

The test feeds the released observation sequence back into the wrapper.  It
therefore isolates the wrapper boundary from any 25 Hz plant approximation:
matching first-control actions, particle beliefs, and particle weights proves
that each external call is the released POMDP's native 0.2 s computation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import sys

import numpy as np


def _max_abs(actual: list[np.ndarray], expected: np.ndarray) -> float:
    observed = np.stack(actual, axis=1)
    if observed.shape != expected.shape:
        raise RuntimeError(f"shape mismatch: wrapper {observed.shape}, native {expected.shape}")
    return float(np.max(np.abs(observed - expected)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--following-dir", type=Path, required=True)
    parser.add_argument("--native-results", type=Path, required=True)
    parser.add_argument("--experiment", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--timesteps", type=int,
                        help="diagnostic prefix length; omit for the full native trajectory")
    args = parser.parse_args()

    package_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(package_root))
    from official_wrapper import NATIVE_DT_S, NATIVE_TICKS_PER_DECISION, PLANT_DT_S, OfficialPOMDP25Hz

    path = args.native_results / "Results_rear_end" / f"Exp_{args.experiment}" / f"Exp_{args.experiment}.pkl"
    with path.open("rb") as handle:
        native = pickle.load(handle)
    eta0 = native["eta"][0, 0]
    batch_size = int(native["o"].shape[0])
    wrapper = OfficialPOMDP25Hz(source_dir=args.source_dir, following_dir=args.following_dir,
                                device=args.device, batch_size=batch_size)
    wrapper.reset(gap_m=float(eta0[5] - eta0[0] - 4.2), ego_speed_mps=float(eta0[4]),
                  target_speed_mps=float(eta0[9]), seed=0)
    incoming_looming = []

    acceleration: list[np.ndarray] = []
    steering: list[np.ndarray] = []
    beliefs: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    timesteps = args.timesteps or int(native["o"].shape[1])
    if not 1 <= timesteps <= native["o"].shape[1]:
        raise ValueError("--timesteps must be within the native trajectory")
    # ``Filter.KDE_update`` mutates its observation tensor in-place into
    # looming coordinates.  The released runner saves that post-update array
    # as ``o``.  Its physical input was the first 12 components of ``eta``;
    # replay those unmodified current-only observations instead.
    physical_observations = native["eta"][:, :timesteps, :12]
    for timestep in range(timesteps):
        with wrapper._torch.no_grad():
            incoming = wrapper._torch.as_tensor(physical_observations[:, timestep, :],
                device=wrapper._agent.planner.device).unsqueeze(-2)
            incoming_looming.append(bool(wrapper._agent.encoder.decoder.test_looming_viability(incoming).all()))
        decision = wrapper.step(physical_observations[:, timestep, :])
        acceleration.append(decision.acceleration_mps2)
        steering.append(decision.steering_rate_rps)
        beliefs.append(decision.belief)
        weights.append(decision.weights)

    controls = np.stack((np.stack(acceleration, axis=1), np.stack(steering, axis=1)), axis=-1)
    expected_controls = native["a_cont"][0, :timesteps].transpose(1, 0, 2)
    expected_beliefs = native["b"][:, :timesteps]
    expected_weights = native["w"][:, :timesteps]
    errors = {
        "executed_continuous_control": float(np.max(np.abs(controls - expected_controls))),
        "belief": _max_abs(beliefs, expected_beliefs),
        "weights": _max_abs(weights, expected_weights),
    }
    control_error_by_timestep = np.max(np.abs(controls - expected_controls), axis=(0, 2))
    belief_array = np.stack(beliefs, axis=1)
    belief_error_by_timestep = np.max(np.abs(belief_array - expected_beliefs), axis=(0, 2, 3))
    tolerance = 1e-5
    report = {
        "official_step_wrapper": True,
        "native_experiment": args.experiment,
        "native_pomdp_dt_s": NATIVE_DT_S,
        "external_plant_dt_s": PLANT_DT_S,
        "plant_ticks_held_per_native_decision": NATIVE_TICKS_PER_DECISION,
        "batch_size": batch_size,
        "timesteps": timesteps,
        "tolerance": tolerance,
        "max_abs_by_signal": errors,
        "first_control_divergence_timestep": int(np.flatnonzero(control_error_by_timestep > tolerance)[0])
            if np.any(control_error_by_timestep > tolerance) else None,
        "first_belief_divergence_timestep": int(np.flatnonzero(belief_error_by_timestep > tolerance)[0])
            if np.any(belief_error_by_timestep > tolerance) else None,
        "control_max_abs_first_five_timesteps": control_error_by_timestep[:5].tolist(),
        "belief_max_abs_first_five_timesteps": belief_error_by_timestep[:5].tolist(),
        "first_replicate_control_first_five_timesteps": {
            "wrapper": controls[0, :5].tolist(), "native": expected_controls[0, :5].tolist(),
        },
        "first_replicate_particle0_first_two_timesteps": {
            "wrapper": belief_array[0, :2, 0].tolist(),
            "native": expected_beliefs[0, :2, 0].tolist(),
        },
        "incoming_observation_looming_viable_first_five": incoming_looming[:5],
        "passes_native_step_equivalence": max(errors.values()) <= tolerance,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
