#!/usr/bin/env python3
"""Full shared Val comparison or frozen Test analysis of the same world chain."""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from hierarchical_world_model.src.core.utils import save_json, select_device
from hierarchical_world_model.src.data import (
    prepare_experiment_data,
    artifact_identity,
    event_metadata,
)
from hierarchical_world_model.src.npc_experiments.lane_execution import (
    recording_interval,
)
from hierarchical_world_model.src.planner import frozen_diffusion_plans
from hierarchical_world_model.src.config import load_protocol_config
from hierarchical_world_model.src.composition import load_selected_components
from hierarchical_world_model.src.shared_evaluation import (
    MOBIL_PATH,
    factual_records,
    highway_rollout,
    load_shared_components,
    recording_ratio_interval,
    scene_randomness,
    save_error_samples,
    summarize_records,
)
from hierarchical_world_model.src.stochastic_drivers.idm_population import (
    IDM_EQUATION,
    MODEL_PATH,
    sample_theta,
)
from hierarchical_world_model.src.stochastic_drivers.online import (
    OnlineMAIDMController,
    calibrated_mobil_parameters,
)
from hierarchical_world_model.src.stochastic_drivers.lane_trajectory import (
    MODEL_PATH as LANE_TRAJECTORY_PATH,
)
from tools.plot_style import GENERATED_COLOR, REAL_COLOR, get_pyplot, style_axes

OUTPUT = ROOT / "results/hierarchical_world_model/factual_validation"
BATCH_SIZE = 64
ARMS = ("rules", "rules_recovery")


def make_controller(randomness, fit, population):
    theta = sample_theta(population, randomness.driver_latent)
    return OnlineMAIDMController(
        theta,
        maneuver_latent=randomness.maneuver_latent,
        **calibrated_mobil_parameters(fit["mobil"]),
    )


def plot_validation(records, split, output_dir):
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plt = get_pyplot()
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
    statistics = {}
    for index, (arm, rows) in enumerate(records.items()):
        statistics[arm] = {}
        label = (
            "World model" if arm == "rules_recovery" else "Without reference recovery"
        )
        color = GENERATED_COLOR if arm == "rules_recovery" else REAL_COLOR
        for axis, metric, top, bottom in (
            (axes[0], "ADE_m", "distance_sum_m", "actor_frames"),
            (axes[1], "FDE_m", "final_distance_sum_m", "final_actors"),
        ):
            value = sum(row[top] for row in rows) / sum(row[bottom] for row in rows)
            interval = recording_ratio_interval(rows, top, bottom)
            errors = np.sort([row[top] / row[bottom] for row in rows if row[bottom]])
            axis.plot(
                errors,
                np.arange(1, len(errors) + 1) / len(errors),
                color=color,
                label=label,
            )
            axis.axvline(value, color=color, linestyle="--", linewidth=1)
            axis.axvspan(*interval, color=color, alpha=0.12)
            axis.text(
                0.97,
                0.16 + 0.13 * index,
                f"Weighted {metric[:3]}: {value * 100:.2f} cm\n95% CI: [{interval[0] * 100:.2f}, {interval[1] * 100:.2f}] cm",
                transform=axis.transAxes,
                ha="right",
                color=color,
                fontsize=9,
            )
            statistics[arm][metric] = {
                "value": value,
                "recording_bootstrap_95CI": interval,
            }
        for metric in (
            "collision",
            "introduced_offroad",
            "npc_introduced_offroad",
            "ego_introduced_offroad",
        ):
            statistics[arm][metric] = {
                "rate": float(np.mean([row[metric] for row in rows])),
                "recording_bootstrap_95CI": recording_interval(rows, metric),
            }
    for index, (axis, metric) in enumerate(zip(axes, ("ADE", "FDE"))):
        style_axes(axis)
        axis.set(xlabel=f"Scene {metric} [m]", ylabel="Empirical CDF", ylim=(0, 1.02))
        axis.set_xscale("symlog", linthresh=0.01)
        axis.set_xlim(left=0)
        axis.set_title(f"({chr(97 + index)}) {metric}", loc="left")
        axis.legend(frameon=False, fontsize=9)
    recordings = len({row["recording_id"] for row in next(iter(records.values()))})
    figure.suptitle(
        f"Shared {split.title()}: {len(next(iter(records.values()))):,} sequences, {recordings} recordings | actual-size HighwayEnv"
    )
    figure.text(
        0.5,
        0.012,
        "Observed M, C0, K_GT, D_GT; integrated NPC actions, not position playback.\n"
        "ECDFs describe scenes; dashed means weight actor-frames (ADE) or final actors (FDE).\n"
        "Shading: 95% recording-bootstrap CI (2,000 resamples); conditional reconstruction, not C0-only forecasting.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.18, 1, 0.92))
    figure.savefig(figures / "trajectory_accuracy.png", dpi=300)
    plt.close(figure)
    return statistics


def main(split="test"):
    torch.set_num_threads(4)
    config = load_protocol_config(ROOT / "hierarchical_world_model/config.yaml")
    experiment = prepare_experiment_data(config, ROOT)
    model, _, population, fit = load_shared_components(
        config, experiment, select_device("auto")
    )
    device = next(model.parameters()).device
    checkpoint = (
        Path(config["paths"]["output_dir"]) / "checkpoints/stochastic_heads_best.pt"
    )
    arms = ARMS
    if split == "test":
        selection, selected_population = load_selected_components(checkpoint)
        if selection["mobil"] != fit["mobil"]:
            raise ValueError("Test must use the frozen public MOBIL parameters")
        population = selected_population
        arms = (selection["selected_arm"],)
    rows = experiment.validation_rows if split == "val" else experiment.test_rows
    output = OUTPUT if split == "val" else OUTPUT.with_name("factual_test")
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "preparing_plans",
        "split": split,
        "sequences_requested": len(rows),
        "fit_split": "train",
        "selection_split": "val",
        "source": config["paths"]["sequence_cache_dir"],
        "response_checkpoint": str(checkpoint),
        "response_checkpoint_mtime_ns": checkpoint.stat().st_mtime_ns,
        "driver_parameter_source": str(MODEL_PATH),
        "driver_population": {
            name: population[name].tolist() for name in ("log_mean", "log_cov")
        },
        "lane_trajectory_model": str(LANE_TRAJECTORY_PATH),
        "lane_trajectory_model_mtime_ns": LANE_TRAJECTORY_PATH.stat().st_mtime_ns,
        "mobil_source": str(MOBIL_PATH),
        "mobil_parameters": fit["mobil"],
        "diffusion_checkpoint": config["paths"]["diffusion_checkpoint"],
        "diffusion": artifact_identity(config["paths"]["diffusion_checkpoint"]),
        "evaluation_role": "Test K_GT conditional reconstruction; generated ADS worlds are a separate application",
        "randomness_identity": "physical sequence_id, not cache row",
        "DDIM_steps": 20,
        "test_used_for_selection": False,
        "public_weights_changed": False,
        "historical_test_used_for_lateral_development": True,
        "test_is_independent_final_evaluation": False,
        "backend": "HighwayEnv with matching unicycle dynamics and actual dimensions",
        "offroad_comparison": "retain scene-wide raw/initial-relative/per-frame logged-GT comparisons; separately report NPC and logged-ego scopes, which overlap and are not summed; scene-wide acceptance gates unchanged",
        "IDM_blend": "state-triggered bounded IDM response; clearance guard retained; no learned selector",
        "reference_recovery": "retire physically unreachable longitudinal timetable; latch until reset; realised x/vx and full original spatial lateral path",
        "idm_equation": IDM_EQUATION,
        "arms": list(arms),
    }
    save_json(report, output / "report.json")
    plans = frozen_diffusion_plans(
        experiment.bundle,
        rows,
        checkpoint=config["paths"]["diffusion_checkpoint"],
        output_dir=ROOT / "results/hierarchical_world_model/diffusion_plans" / split,
        device=device,
        batch_size=256,
        ddim_steps=20,
    )
    records = {arm: [] for arm in arms}
    errors = {arm: [] for arm in arms}
    jerks = {arm: [] for arm in arms}
    started = time.monotonic()
    arrays = experiment.bundle.arrays
    metadata = event_metadata(
        Path(config["paths"]["sequence_cache_dir"]) / "sequence_cache", arrays
    )
    for begin in range(0, len(rows), BATCH_SIZE):
        chosen = rows[begin : begin + BATCH_SIZE]
        randomness = scene_randomness(np.asarray(arrays["sequence_id"][chosen]))
        for arm in arms:
            controller = make_controller(randomness, fit, population)
            trace = highway_rollout(
                model,
                arrays,
                chosen,
                plans[begin : begin + BATCH_SIZE],
                randomness,
                controller=controller,
                recovery=arm == "rules_recovery",
            )
            row_records, distances, jerk = factual_records(
                chosen,
                np.asarray(arrays["agent_states"][chosen]),
                np.asarray(arrays["agent_valid"][chosen]),
                trace,
            )
            for record, row in zip(row_records, chosen):
                record["recording_id"] = int(metadata.iloc[row].recording_id)
                record["sequence_id"] = str(arrays["sequence_id"][row])
            records[arm].extend(row_records)
            errors[arm].append(distances)
            jerks[arm].append(jerk)
        count = len(records[arms[0]])
        print(
            f"{split.title()} {count}/{len(rows)}; elapsed {time.monotonic() - started:.1f}s",
            flush=True,
        )
        if begin % (5 * BATCH_SIZE) == 0 or count == len(rows):
            report.update(
                status="running",
                sequences_completed=count,
                elapsed_s=time.monotonic() - started,
            )
            report["factual"] = {
                arm: summarize_records(records[arm], errors[arm], jerks[arm])
                for arm in arms
            }
            save_json(report, output / "report.json")
            for arm in arms:
                pd.DataFrame(records[arm]).to_csv(
                    output
                    / (
                        "scene_metrics.csv" if split == "test" else f"{arm}_records.csv"
                    ),
                    index=False,
                )
    if split == "val":
        base = report["factual"]["rules"]
        report["factual_gates"] = {
            arm: {
                "ADE_noninferiority": value["ADE_m"] <= base["ADE_m"] * 1.05,
                "FDE_noninferiority": value["FDE_m"] <= base["FDE_m"] * 1.05,
                "P95_noninferiority": value["P95_m"] <= base["P95_m"] * 1.05,
                "no_more_collisions": value["collision"] <= base["collision"],
                "no_more_new_offroad": value["new_offroad"] <= base["new_offroad"],
                "no_more_introduced_offroad": value["introduced_offroad"]
                <= base["introduced_offroad"],
                "finite": value["finite"] == len(rows),
                "no_committed_target_changes": value["committed_target_changes"] == 0,
            }
            for arm, value in report["factual"].items()
        }
    report["status"] = (
        "factual_complete_requires_public_chain_selection"
        if split == "val"
        else "frozen_test_factual_complete_not_world_acceptance"
    )
    report["recording_bootstrap_statistics"] = plot_validation(records, split, output)
    if split == "test":
        save_error_samples(output, errors[arms[0]], jerks[arms[0]])
    save_json(report, output / "report.json")
    print(json.dumps(report["factual"], indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    main(parser.parse_args().split)
