#!/usr/bin/env python3
"""Frozen full-Test natural replay, with three paired NPC realizations."""

from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
import gzip
import json
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from hierarchical_world_model.src import bridge_evaluation as experiment
from hierarchical_world_model.src.bridge_evaluation import (
    AuditedTakeover,
    REASONS,
    records_from_trace,
)
from hierarchical_world_model.src.core.utils import (
    load_json,
    load_yaml,
    save_json,
    file_sha256,
)
from hierarchical_world_model.src.shared_evaluation import (
    highway_rollout,
    scene_randomness,
)
from hierarchical_world_model.src.stochastic_drivers.idm_population import sample_theta
from hierarchical_world_model.src.stochastic_drivers.online import (
    calibrated_mobil_parameters,
)
from hierarchical_world_model.src.takeover import TakeoverConfig
from hierarchical_world_model.src.tracking import TrackingConfig

OUTPUT = ROOT / "results/hierarchical_world_model/bridge"
BATCH = 128
WORKERS = 16
SEEDS = experiment.SEEDS


def evaluate_batch(offset):
    config, arrays, rows, plans, population, fit = experiment._DATA
    selected = np.arange(offset, min(offset + BATCH, len(rows)))
    chosen = rows[selected]
    draws = scene_randomness(arrays["sequence_id"][chosen], seed=SEEDS[0])
    baseline = highway_rollout(
        arrays,
        chosen,
        plans[selected],
        draws,
        tracking_config=TrackingConfig(**config["tracking"]),
    )
    records = records_from_trace(baseline, arrays, chosen, seed=SEEDS[0], arm="tracker")
    trace_directory = OUTPUT / ".cache" / "natural" / "admitted_traces"
    trace_directory.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        draws = scene_randomness(arrays["sequence_id"][chosen], seed=seed)
        controller = AuditedTakeover(
            sample_theta(population, draws.driver_latent),
            takeover_config=TakeoverConfig(**config["takeover"]),
            maneuver_latent=draws.maneuver_latent,
            **calibrated_mobil_parameters(fit["mobil"]),
        )
        trace = highway_rollout(
            arrays,
            chosen,
            plans[selected],
            draws,
            controller=controller,
            tracking_config=TrackingConfig(**config["tracking"]),
        )
        response = records_from_trace(
            trace,
            arrays,
            chosen,
            seed=seed,
            arm="takeover",
            controller=controller,
            baseline=baseline,
        )
        records.extend(response)
        for k, record in enumerate(response):
            if record["takeovers"] or record["any_collision"]:
                np.savez_compressed(
                    trace_directory / f"{record['row']}_{seed}.npz",
                    **{name: value[k] for name, value in trace.items()},
                    tracker_states=baseline["states"][k],
                    tracker_actions=baseline["actions"][k],
                    logged_states=np.asarray(
                        arrays["agent_states"][record["row"], 25:174]
                    ),
                    logged_valid=np.asarray(
                        arrays["agent_valid"][record["row"], 25:174]
                    ),
                    plan=plans[selected[k]],
                )
    experiment.save_compact_json(
        dict(records=records), OUTPUT / ".cache" / "natural" / f"{offset}.json"
    )
    return offset, len(chosen), sum(r["takeovers"] for r in records)


def aggregate(records):
    actor_frames = sum(r["actor_frames"] for r in records)
    scene_errors = [r["ADE_m"] for r in records if r["ADE_m"] is not None]
    NPC_runs = sum(r["valid_NPCs"] > 0 for r in records)
    admitted_runs = sum(r["takeovers"] > 0 for r in records)
    return dict(
        runs=len(records),
        unique_scenes=len({r["row"] for r in records}),
        vehicle_episodes=sum(r["valid_NPCs"] for r in records),
        admitted_scenes=len({r["row"] for r in records if r["takeovers"]}),
        admitted_runs=admitted_runs,
        scenes_with_NPCs=len({r["row"] for r in records if r["valid_NPCs"]}),
        no_NPC_runs=len(records) - NPC_runs,
        scene_takeover_rate_with_NPCs=admitted_runs / NPC_runs if NPC_runs else None,
        scene_takeover_rate=float(np.mean([r["takeovers"] > 0 for r in records])),
        vehicle_takeovers=sum(r["takeovers"] for r in records),
        vehicle_takeover_rate=sum(r["takeovers"] for r in records)
        / sum(r["valid_NPCs"] for r in records),
        reasons={
            name: sum(e["reason"] == code for r in records for e in r["events"])
            for code, name in REASONS.items()
        },
        reason_scene_counts={
            name: len(
                {
                    r["row"]
                    for r in records
                    if any(e["reason"] == code for e in r["events"])
                }
            )
            for code, name in REASONS.items()
        },
        ADE_actor_frame_weighted_m=sum(r["position_error_sum_m"] for r in records)
        / actor_frames,
        ADE_defined_runs=len(scene_errors),
        ADE_scene_mean_m=float(np.mean(scene_errors)) if scene_errors else None,
        ADE_scene_quantiles_m=(
            np.quantile(scene_errors, [0.5, 0.95, 0.99, 1]).tolist()
            if scene_errors
            else None
        ),
        actor_frames=actor_frames,
        ego_collision_runs=sum(r["ego_collision"] for r in records),
        any_collision_runs=sum(r["any_collision"] for r in records),
        NPC_offroad_runs=sum(r["NPC_offroad"] for r in records),
        repeated_takeovers=sum(r["repeated_takeovers"] for r in records),
        ownership_releases=sum(r["ownership_releases"] for r in records),
    )


def summarize(records):
    tracker = {r["row"]: r for r in records if r["arm"] == "tracker"}
    response = [r for r in records if r["arm"] == "takeover"]
    assert len(tracker) == 10151 and len(response) == 10151 * len(SEEDS)
    assert len({(r["row"], r["seed"]) for r in response}) == len(response)
    groups = {"tracker": aggregate(list(tracker.values()))}
    groups.update(
        {
            f"takeover_{seed}": aggregate([r for r in response if r["seed"] == seed])
            for seed in SEEDS
        }
    )
    groups["takeover_all_seeds"] = aggregate(response)
    admitted = {r["row"] for r in response if r["takeovers"]}
    if admitted:
        groups["admitted_subset_tracker"] = aggregate(
            [tracker[row] for row in sorted(admitted)]
        )
        groups["admitted_subset_takeover"] = aggregate(
            [r for r in response if r["row"] in admitted]
        )
    pairs = []
    for r in response:
        base = tracker[r["row"]]
        pairs.append(
            dict(
                row=r["row"],
                seed=r["seed"],
                recording_id=r["recording"],
                admitted=int(r["takeovers"] > 0),
                actor_frames=r["actor_frames"],
                delta_error_sum_m=r["position_error_sum_m"]
                - base["position_error_sum_m"],
                introduced_any_collision=int(
                    r["any_collision"] and not base["any_collision"]
                ),
                resolved_any_collision=int(
                    base["any_collision"] and not r["any_collision"]
                ),
                introduced_ego_collision=int(
                    r["ego_collision"] and not base["ego_collision"]
                ),
                resolved_ego_collision=int(
                    base["ego_collision"] and not r["ego_collision"]
                ),
                introduced_NPC_offroad=int(
                    r["NPC_offroad"] and not base["NPC_offroad"]
                ),
            )
        )
    recording_ids = sorted({r["recording"] for r in response})
    by_recording = []
    for recording in recording_ids:
        base = [r for r in tracker.values() if r["recording"] == recording]
        new = [r for r in response if r["recording"] == recording]
        by_recording.append(
            dict(
                recording_id=recording, tracker=aggregate(base), takeover=aggregate(new)
            )
        )
    from hierarchical_world_model.src.shared_evaluation import (
        recording_interval,
        recording_ratio_interval,
    )

    report = dict(
        complete=True,
        split="test",
        unique_scenes=len(tracker),
        recordings=recording_ids,
        behavior_seeds=list(SEEDS),
        executed_rollouts=len(records),
        physical_steps=len(records) * 149,
        evaluation="Natural no-intervention replay with logged ego controls; not generated scenarios or a new ADS evaluation",
        test_role="Frozen diagnostic evaluation; these Test recordings have been inspected in earlier development, not a new blind test",
        parameters_frozen=True,
        parameters_selected_on_test=False,
        execution="Every Test scene executed once with tracking and three times with the current bridge; no old conflict-only screening",
        summary=groups,
        by_recording=by_recording,
        paired_outcomes={
            key: sum(r[key] for r in pairs)
            for key in pairs[0]
            if key.startswith(("introduced_", "resolved_"))
        },
        paired_ADE_delta_m=sum(r["delta_error_sum_m"] for r in pairs)
        / sum(r["actor_frames"] for r in pairs),
        recording_bootstrap_95_percent=dict(
            takeover_scene_rate=recording_interval(pairs, "admitted"),
            paired_ADE_delta_m=recording_ratio_interval(
                pairs, "delta_error_sum_m", "actor_frames"
            ),
            note="2000 whole-recording resamples; only nine recordings, repeated seeds are not independent scenes",
        ),
        unadmitted_runs=sum(r["takeovers"] == 0 for r in response),
        unadmitted_exact_tracker_runs=sum(
            r["takeovers"] == 0
            and r["exact_tracker_states"]
            and r["exact_tracker_actions"]
            for r in response
        ),
        adequacy="Low admission rate and absence of collisions do not establish that each handover is necessary or behaviorally reasonable",
        intervention_and_generated_ADS_evaluation_included=False,
    )
    review = dict(
        records=[
            dict(
                r,
                tracker_ADE_m=tracker[r["row"]]["ADE_m"],
                tracker_any_collision=tracker[r["row"]]["any_collision"],
            )
            for r in response
            if r["takeovers"] or r["any_collision"]
        ],
        reason_counts=dict(
            Counter(REASONS[e["reason"]] for r in response for e in r["events"])
        ),
        interpretation="Observed conditions and plan trends support mechanism inspection, not causal necessity labels. No new ADS was used.",
    )
    save_json(review, OUTPUT / "test_admissions.json")
    save_json(report, OUTPUT / "test_report.json")
    return report


def main():
    config = load_yaml(ROOT / "hierarchical_world_model/config.yaml")
    experiment.initialize_worker(config, "test")
    _, arrays, rows, plans, _, _ = experiment._DATA
    sources = [
        Path(__file__),
        ROOT / "hierarchical_world_model/src/bridge_evaluation.py",
    ]
    sources.extend(sorted((ROOT / "hierarchical_world_model/src").rglob("*.py")))
    sources.extend(
        [
            ROOT / "hierarchical_world_model/config.yaml",
            OUTPUT / "selection.json",
            ROOT / config["takeover"]["error_trajectories_path"],
            ROOT
            / "results/hierarchical_world_model/diffusion_plans/test/positions.npz",
        ]
    )
    sources.extend(
        ROOT / "results/hierarchical_world_model/model_assets" / name
        for name in ("mobil.json", "idm_population.npz", "lane_trajectory.json")
    )
    hashes = {str(p.relative_to(ROOT)): file_sha256(p) for p in sources}
    protocol = dict(
        split="test",
        scenes=len(rows),
        seeds=list(SEEDS),
        steps=149,
        dt_s=0.04,
        parameters=config["takeover"],
        tracking=config["tracking"],
        source_sha256=hashes,
        evaluation="Full natural replay; logged ego controls; pure tracker and current bridge; no artificial intervention",
        test_used_to_fit_or_select_parameters=False,
    )
    path = OUTPUT / "evaluation_protocol.json"
    if path.exists() and load_json(path) != protocol:
        raise ValueError(
            "Test inputs changed; preserve previous results before starting a new frozen evaluation"
        )
    save_json(protocol, path)
    offsets = list(range(0, len(rows), BATCH))
    directory = OUTPUT / ".cache" / "natural"
    directory.mkdir(parents=True, exist_ok=True)
    jobs = [offset for offset in offsets if not (directory / f"{offset}.json").exists()]
    with ProcessPoolExecutor(
        max_workers=WORKERS,
        mp_context=get_context("spawn"),
        initializer=experiment.initialize_worker,
        initargs=(config, "test"),
    ) as pool:
        futures = [pool.submit(evaluate_batch, offset) for offset in jobs]
        for number, future in enumerate(as_completed(futures), 1):
            offset, count, admissions = future.result()
            print(
                f"Test {number}/{len(jobs)} batch={offset} scenes={count} admissions={admissions}",
                flush=True,
            )
    assert all(
        file_sha256(ROOT / name) == digest for name, digest in hashes.items()
    ), "Frozen inputs changed during evaluation"
    records = [
        r
        for offset in offsets
        for r in load_json(directory / f"{offset}.json")["records"]
    ]
    with gzip.open(OUTPUT / "natural_records.json.gz", "wt") as stream:
        json.dump(records, stream, separators=(",", ":"), allow_nan=False)
    traces = {}
    for path in sorted((directory / "admitted_traces").glob("*.npz")):
        with np.load(path) as values:
            traces.update(
                {f"{path.stem}/{name}": values[name] for name in values.files}
            )
    np.savez_compressed(OUTPUT / "natural_traces.npz", **traces)
    report = summarize(records)
    shutil.rmtree(directory)
    print(
        "Complete",
        report["unique_scenes"],
        report["summary"]["takeover_all_seeds"],
        flush=True,
    )


if __name__ == "__main__":
    main()
