"""Execute the unmodified official v1.0.0 rear-end simulator.

This runner is deliberately a thin orchestration layer.  It imports and calls
the released simulator rather than re-implementing the agent; all generated
``Results_rear_end`` pickles are therefore native backend outputs.  The source
repository must be the v1.0.0 commit recorded in the paper plan.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
import types


OFFICIAL_COMMIT = "56655de845644c45f01ab2898544e55316a3279b"
GRID = tuple((time_gap, speed) for time_gap in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5)
             for speed in (25.0, 20.0, 15.0, 10.0))
SMOKE_GRID = ((1.0, 10.0), (1.5, 15.0), (2.5, 20.0), (3.5, 25.0))


def model_parameters(variant: str = "full") -> dict[str, object]:
    """Literal full-model values from official ``simulation_rear_end.py``."""
    parameters: dict[str, object] = {
        "a_sd_model": 3.0, "Loom_perc": True, "d_phi_thres": 0.00215,
        "Loom_change_obs": -1, "perc_noise_factor": 0.01,
        "noise_pred_fac": 0.2, "num_plan": 100, "a_sd_plan": 5.0,
        "sample_steering_rate": True, "use_pedals": True, "H": 30,
        "plan_ignore_w": True, "plan_smooth_delta": True,
        "pref_v_sd": 0.5, "pref_a_sd": 0.1, "pref_w_sd": 0.02,
        "lane_cost": -15000, "lane_change_cost": -1000, "coll_cost": -10000,
        "road_pref": 0, "Loom_reward": "V7", "weigh_particles": 0.001,
        "full_violation_factor": 0.01, "unpunished_heading": 85,
        "collision_cost_adjusted": True, "N_norm": 32, "H_norm": 20,
        "alpha": 1.0, "EA_mode": "Surprise", "EA_fac": -5.95,
        "EA_init": False,
    }
    if variant == "no_evidence":
        parameters["EA_mode"] = "None"
    elif variant == "no_pedal":
        parameters["use_pedals"] = False
    elif variant != "full":
        raise ValueError(f"unknown variant: {variant}")
    return parameters


def initial_state() -> dict[str, float]:
    return {
        "lane_width": 3.65, "d": 1.72, "lf": 2.1, "lr": 2.1,
        "a_max": 8.0, "w_max": 1.22, "x_ego": 0.0, "y_ego": 0.0,
        "theta_ego": 0.0, "delta_ego": 0.0, "v_ego": 0.0, "x_tar": 0.0,
        "y_tar": 0.0, "theta_tar": 0.0, "delta_tar": 0.0, "v_tar": 0.0,
    }


def _prepare(source_dir: Path, output_dir: Path, following_dir: Path, *, batch_size: int,
             suite: str, variant: str) -> None:
    import subprocess

    commit = subprocess.check_output(["git", "-C", str(source_dir), "rev-parse", "HEAD"], text=True).strip()
    if commit != OFFICIAL_COMMIT:
        raise RuntimeError(f"official source must be {OFFICIAL_COMMIT}, found {commit}")
    output_dir.mkdir(parents=True, exist_ok=True)
    link = output_dir / "Results_following"
    if not following_dir.is_dir():
        raise RuntimeError(f"missing released Results_following dependency: {following_dir}")
    if not link.exists() and not link.is_symlink():
        link.symlink_to(following_dir, target_is_directory=True)
    metadata = {
        "official_commit": OFFICIAL_COMMIT, "official_source": str(source_dir),
        "results_following_source": str(following_dir),
        "native_dt_s": 0.2, "horizon_steps": 30, "particles": 75,
        "cem_plans": 100, "cem_iterations": 10, "replicates": batch_size,
        "suite": suite, "variant": variant,
        "grid": [{"time_gap_s": gap, "speed_mps": speed} for gap, speed in GRID],
    }
    (output_dir / "run_contract.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def run(source_dir: Path, output_dir: Path, following_dir: Path, suite: str, batch_size: int,
        variant: str, condition: tuple[float, float] | None = None) -> None:
    _prepare(source_dir, output_dir, following_dir, batch_size=batch_size, suite=suite,
             variant=variant)
    sys.path.insert(0, str(source_dir))
    os.chdir(output_dir)
    # The released simulation script imports its plotting/analysis scripts at
    # module import time.  Those scripts expect a completed result directory;
    # placeholders let us call the released ``set_config``/``simulate`` API
    # without altering its source.  Analysis is invoked after the grid exists.
    sys.modules.setdefault("Analysis_rear_end", types.ModuleType("Analysis_rear_end"))
    sys.modules.setdefault("visualization_rear_end", types.ModuleType("visualization_rear_end"))
    import torch
    from simulation_rear_end import find_parameters, set_config, simulate
    from src.utils.saving import save_results

    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    grid = (condition,) if condition is not None else (GRID if suite == "full" else SMOKE_GRID)
    manifest_path = output_dir / f"{suite}_progress.json"
    completed: list[dict[str, object]] = []
    for ordinal, (time_gap, speed) in enumerate(grid, start=1):
        params = model_parameters(variant)
        state = initial_state()
        state["v_ego"] = speed
        state["v_tar"] = speed
        state["x_tar"] = speed * time_gap + state["lf"] + state["lr"]
        thw = state["x_tar"] / speed
        state["x_tar"] = speed * thw
        v_diff, target_minimum_acceleration = find_parameters(speed, params["EA_fac"],
            params["noise_pred_fac"], params["H"], params["d_phi_thres"], thw)
        params["v_diff"] = v_diff
        params["a_tar_min_intensity"] = -target_minimum_acceleration / state["a_max"]
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)
            torch.cuda.empty_cache()
        config_snapshot, config = set_config(state, params, -6.0)
        config["T"] = 60
        config["rollout_batch_size"] = batch_size
        config_snapshot["T"] = 60
        started = time.monotonic()
        data = simulate(config, device)
        save_results(data, config_snapshot, "rear_end")
        completed.append({"ordinal": ordinal, "time_gap_s": time_gap, "speed_mps": speed,
                          "batch_size": batch_size, "elapsed_s": time.monotonic() - started})
        manifest_path.write_text(json.dumps({"suite": suite, "completed": completed}, indent=2), encoding="utf-8")
        print(json.dumps(completed[-1]), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--following-dir", type=Path, required=True,
                        help="released Results_following directory required by the official script")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--suite", choices=("smoke", "full"), required=True)
    parser.add_argument("--variant", choices=("full", "no_evidence", "no_pedal"), default="full")
    parser.add_argument("--time-gap", type=float,
                        help="run one declared initial time-gap condition")
    parser.add_argument("--speed", type=float,
                        help="run one declared initial-speed condition")
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()
    default_batch = 32 if args.suite == "full" else 8
    if (args.time_gap is None) != (args.speed is None):
        parser.error("--time-gap and --speed must be specified together")
    condition = None if args.time_gap is None else (args.time_gap, args.speed)
    run(args.source_dir.resolve(), args.output_dir.resolve(), args.following_dir.resolve(),
        args.suite, args.batch_size or default_batch, args.variant, condition)


if __name__ == "__main__":
    main()
