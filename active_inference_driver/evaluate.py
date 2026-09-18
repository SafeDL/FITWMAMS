"""25 Hz closed-loop highD evaluation and paper-style collision-response plots."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from .config import ActiveInferenceConfig
from .data import DEFAULT_DATASET, leader_brake_events, load_highd_pairs
from .model import ActiveInferenceDriver


def _observed_acceleration(speed: np.ndarray) -> np.ndarray:
    return np.diff(speed, prepend=speed[0]) / .04


def _response_time(acceleration: np.ndarray, event_frame: int) -> float | None:
    """Adapter-only onset proxy: first physical brake sample after the event."""
    after = acceleration[event_frame:]
    hits = np.flatnonzero(after < -1.0)
    if not len(hits):
        return None
    return float(hits[0] * .04)


def rollout_pair(pair: dict[str, np.ndarray], start: int, frames: int, config: ActiveInferenceConfig,
                 *, seed: int, no_evidence: bool = False, no_pedal: bool = False) -> dict[str, np.ndarray]:
    """Closed-loop ego rollout; leader is observed only at the present 25 Hz frame."""
    cfg = replace(config, evidence_drift=0.0) if no_evidence else config
    if no_pedal:
        cfg = replace(cfg, enforce_pedal_constraint=False)
    driver = ActiveInferenceDriver(cfg)
    leader_acc = _observed_acceleration(pair["leader_v"])
    gap = float(pair["gap"][start])
    driver.reset(gap_m=gap, ego_speed_mps=float(pair["follower_v"][start]),
                 leader_speed_mps=float(pair["leader_v"][start]),
                 leader_acceleration_mps2=float(leader_acc[start]), seed=seed)
    x = float(pair["follower_x"][start]); speed = float(pair["follower_v"][start])
    position = np.empty(frames); velocity = np.empty(frames); acceleration = np.empty(frames)
    evidence = np.empty(frames); surprise = np.empty(frames); replan = np.zeros(frames, bool)
    pedal = np.empty(frames); held_action = cfg.coast_acceleration_mps2
    for local in range(frames):
        absolute = start + local
        current_gap = float(pair["leader_x"][absolute] - x - pair["length_sum"])
        if local % cfg.native_ticks_per_decision == 0:
            trace = driver.decide(gap_m=current_gap, ego_speed_mps=speed,
                                  leader_speed_mps=float(pair["leader_v"][absolute]))
            held_action = trace.action
            evidence[local:local + cfg.native_ticks_per_decision] = trace.evidence_before
            surprise[local:local + cfg.native_ticks_per_decision] = trace.surprise
            replan[local] = trace.replanned
        x += speed * cfg.native_dt_s + .5 * held_action * cfg.native_dt_s ** 2
        speed = max(0., speed + held_action * cfg.native_dt_s)
        position[local], velocity[local], acceleration[local] = x, speed, held_action
        pedal[local] = np.sign(held_action - cfg.coast_acceleration_mps2)
    truth_position = pair["follower_x"][start + 1:start + frames + 1]
    truth_speed = pair["follower_v"][start + 1:start + frames + 1]
    truth_gap = pair["leader_x"][start + 1:start + frames + 1] - truth_position - pair["length_sum"]
    sim_gap = pair["leader_x"][start + 1:start + frames + 1] - position - pair["length_sum"]
    return {"position": position, "speed": velocity, "acceleration": acceleration, "gap": sim_gap,
            "evidence": evidence, "surprise": surprise, "replan": replan, "pedal": pedal,
            "truth_speed": truth_speed, "truth_acceleration": _observed_acceleration(pair["follower_v"])[start + 1:start + frames + 1],
            "truth_gap": truth_gap, "leader_speed": pair["leader_v"][start + 1:start + frames + 1]}


def _plot_representative(result: dict[str, np.ndarray], event_local: int, output: Path) -> None:
    time = np.arange(len(result["speed"])) * .04
    fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True, layout="constrained")
    axes[0].plot(time, result["truth_speed"], color="black", label="highD driver")
    axes[0].plot(time, result["leader_speed"], color="tab:orange", label="highD leader")
    axes[0].plot(time, result["speed"], color="tab:blue", label="active-inference rollout")
    axes[0].set_ylabel("speed (m/s)"); axes[0].legend(ncol=3, fontsize=8)
    axes[1].plot(time, result["truth_acceleration"], color="black", alpha=.7)
    axes[1].plot(time, result["acceleration"], color="tab:blue")
    axes[1].axhline(-1., color="gray", lw=.8); axes[1].set_ylabel("acceleration (m/s²)")
    axes[2].plot(time, result["truth_gap"], color="black", label="observed")
    axes[2].plot(time, result["gap"], color="tab:blue", label="rollout")
    axes[2].set_ylabel("gap (m)"); axes[2].legend(fontsize=8)
    axes[3].plot(time, result["evidence"], label="evidence E")
    axes[3].plot(time, result["surprise"] * 1e-5, label="surprise × 1e−5", alpha=.7)
    axes[3].scatter(time[result["replan"]], np.ones(result["replan"].sum()), marker="|", s=120, color="crimson", label="full replan")
    axes[3].set_ylabel("evidence"); axes[3].set_xlabel("time since window start (s)"); axes[3].legend(ncol=3, fontsize=8)
    for axis in axes:
        axis.axvline(event_local * .04, color="crimson", ls="--", lw=.8)
    fig.savefig(output, dpi=160); plt.close(fig)


def _plot_metrics(rows: list[dict[str, float]], output: Path) -> None:
    names = ["full", "no evidence", "no pedal"]
    values = [[np.mean([row[key] for row in rows if row["variant"] == name]) for key in ("speed_rmse", "gap_rmse", "min_gap")] for name in names]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), layout="constrained")
    for axis, column, title in zip(axes, range(3), ("speed RMSE (m/s)", "gap RMSE (m)", "minimum gap (m)")):
        axis.bar(names, [value[column] for value in values], color=["tab:blue", "tab:orange", "tab:green"])
        axis.set_title(title); axis.tick_params(axis="x", rotation=20)
    fig.savefig(output, dpi=160); plt.close(fig)


def _plot_ensemble(samples: list[dict[str, np.ndarray]], output: Path) -> None:
    """Show stochastic trajectories rather than hiding the driver's random stream."""
    time = np.arange(len(samples[0]["speed"])) * .04
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True, layout="constrained")
    for axis, key, truth_key, label in ((axes[0], "speed", "truth_speed", "speed (m/s)"),
                                        (axes[1], "gap", "truth_gap", "gap (m)")):
        values = np.asarray([sample[key] for sample in samples])
        axis.fill_between(time, np.quantile(values, .1, axis=0), np.quantile(values, .9, axis=0),
                          color="tab:blue", alpha=.25, label="10–90% stochastic rollout")
        axis.plot(time, np.mean(values, axis=0), color="tab:blue", label="ensemble mean")
        axis.plot(time, samples[0][truth_key], color="black", label="highD driver")
        axis.set_ylabel(label); axis.legend(fontsize=8)
    axes[1].set_xlabel("time since window start (s)")
    fig.savefig(output, dpi=160); plt.close(fig)


def evaluate_highd(dataset: str | Path = DEFAULT_DATASET, output_dir: str | Path = "active_inference_driver/artifacts",
                   *, max_events: int = 8, seed: int = 20260916) -> Path:
    """Run a bounded, reproducible highD event suite and generate evaluation figures."""
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    config = ActiveInferenceConfig(seed=seed)
    selected: list[tuple[dict[str, np.ndarray], int]] = []
    for pair in load_highd_pairs(dataset):
        for event in leader_brake_events(pair):
            selected.append((pair, event))
            if len(selected) == max_events:
                break
        if len(selected) == max_events:
            break
    if not selected:
        raise RuntimeError("no highD leader-brake events found")
    rows: list[dict[str, float]] = []
    representative = None
    for index, (pair, event) in enumerate(selected):
        start = event - 50; frames = 175
        for variant, no_evidence, no_pedal in (("full", False, False), ("no evidence", True, False), ("no pedal", False, True)):
            result = rollout_pair(pair, start, frames, config, seed=seed + index, no_evidence=no_evidence, no_pedal=no_pedal)
            rows.append({"variant": variant, "recording": int(pair["recording_id"]), "pair": int(pair["pair_no"]),
                         "event_frame": int(event), "speed_rmse": float(np.sqrt(np.mean((result["speed"] - result["truth_speed"]) ** 2))),
                         "gap_rmse": float(np.sqrt(np.mean((result["gap"] - result["truth_gap"]) ** 2))),
                         "min_gap": float(np.min(result["gap"])), "collision": bool(np.min(result["gap"]) < .05),
                         "model_response_s": _response_time(result["acceleration"], 50),
                         "human_response_s": _response_time(result["truth_acceleration"], 50),
                         "full_replans": int(np.sum(result["replan"]))})
            if representative is None and variant == "full":
                representative = result
    assert representative is not None
    _plot_representative(representative, 50, output / "representative_highd_response.png")
    _plot_metrics(rows, output / "ablation_metrics.png")
    ensemble = [rollout_pair(selected[0][0], selected[0][1] - 50, 175, config, seed=seed + 1000 + draw)
                for draw in range(12)]
    _plot_ensemble(ensemble, output / "stochastic_response_band.png")
    full = [row for row in rows if row["variant"] == "full"]
    report = {"model": "active_inference_longitudinal_adapted", "dataset": str(dataset),
              "native_fps": 25, "decision_fps": 5, "decision_action_hold_frames": 5,
              "no_steering_adaptation": True, "events": len(selected), "event_rows": rows,
              "summary": {key: float(np.mean([row[key] for row in full])) for key in ("speed_rmse", "gap_rmse", "min_gap", "full_replans")},
              "representative_stochastic_ensemble": {"rollouts": len(ensemble), "seed_range": [seed + 1000, seed + 1011],
                  "collision_probability": float(np.mean([np.min(sample["gap"]) < .05 for sample in ensemble]))},
              "future_action_firewall": "planner receives current gap/speed observations and internally sampled leader particles only; highD leader future is used solely by the external 25 Hz evaluation plant.",
              "native_validation_status": "see validation_status.json; this highD run must not overwrite the independent native-paper audit",
              "planning_budget": {"particles": config.particles, "horizon_steps": config.horizon, "cem_plans": config.cem_plans, "cem_iterations": config.cem_iterations}}
    path = output / "highd_evaluation.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    # Keep highD-only contracts independently addressable.  Native-paper and
    # bridge status are published by publish_validation_status.py and must not
    # be overwritten when this external-domain evaluation is re-run.
    evidence_files = {
        "planning_budget.json": report["planning_budget"],
        "future_action_firewall.json": {"native_fps": 25, "decision_fps": 5,
            "decision_action_hold_frames": 5, "contract": report["future_action_firewall"]},
    }
    for name, value in evidence_files.items():
        (output / name).write_text(json.dumps(value, indent=2), encoding="utf-8")
    return path
