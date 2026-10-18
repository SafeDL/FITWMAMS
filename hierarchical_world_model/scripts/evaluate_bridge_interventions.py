#!/usr/bin/env python3
"""Paired scripted Test interventions under the frozen bridge parameters."""

from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from hierarchical_world_model.src import bridge_evaluation as experiment
from hierarchical_world_model.src.core.utils import (
    load_yaml,
    save_json,
    file_sha256,
)
from hierarchical_world_model.src.data import ego_controls
from hierarchical_world_model.src.scene_geometry import torch_map_lane_geometry
from hierarchical_world_model.src.shared_evaluation import (
    highway_rollout,
    scene_randomness,
)
from hierarchical_world_model.src.stochastic_drivers.driver_policy import (
    physical_leader_observation,
)
from hierarchical_world_model.src.tracking import TrackingConfig

OUTPUT = ROOT / "results/hierarchical_world_model/bridge"
SEEDS = experiment.SEEDS
ONSET = 25


def select_cases():
    _, arrays, rows, plans, _, _ = experiment._DATA
    states = torch.tensor(np.asarray(arrays["agent_states"][rows, 24 + ONSET]))
    valid = torch.tensor(np.asarray(arrays["agent_valid"][rows, 24 + ONSET]))
    dimensions = torch.tensor(np.asarray(arrays["vehicle_dimensions"][rows]))
    gap, speed, _, leader = physical_leader_observation(states, valid, dimensions)
    lanes, present, _ = torch_map_lane_geometry(
        torch.tensor(np.asarray(arrays["map_polylines"][rows])),
        torch.tensor(np.asarray(arrays["map_polyline_valid"][rows])),
    )
    recording = np.array(
        [int(str(s).split("_")[1]) for s in arrays["sequence_id"][rows]]
    )
    specs, allocation = [], []
    for rec in np.unique(recording):
        for family in ("following", "adjacent"):
            candidates = []
            for index in np.flatnonzero(recording == rec):
                s = states[index].numpy()
                centers = lanes[index][present[index]].numpy()
                ego_lane = int(np.argmin(abs(centers - s[0, 1])))
                # Keep the initial mechanism interpretable; selection never reads outcomes.
                log = np.asarray(arrays["agent_states"][rows[index], 24:174])
                if np.ptp(log[:, 0, 1]) > 0.5:
                    continue
                for actor in range(1, 7):
                    if (
                        not valid[index, actor]
                        or not np.asarray(
                            arrays["agent_valid"][rows[index], 24:174, actor]
                        ).all()
                    ):
                        continue
                    if abs(float(speed[index, actor] - speed[index, 0])) > 2:
                        continue
                    if np.ptp(plans[index, :, actor - 1, 1]) > 0.5:
                        continue
                    dx = float(
                        s[0, 0]
                        - s[actor, 0]
                        - 0.5 * (dimensions[index, 0, 0] + dimensions[index, actor, 0])
                    )
                    headway = dx / max(float(speed[index, actor]), 1)
                    if not 0.5 <= headway <= 1.2:
                        continue
                    actor_lane = int(np.argmin(abs(centers - s[actor, 1])))
                    if family == "following":
                        if leader[index, actor] != 0:
                            continue
                        neighbors = [
                            j
                            for j in range(len(centers))
                            if j != ego_lane and abs(centers[j] - centers[ego_lane]) < 5
                        ]
                        if not neighbors:
                            continue
                        target_y = float(centers[neighbors[0]])
                    else:
                        if (
                            actor_lane == ego_lane
                            or not 2.5
                            < abs(centers[actor_lane] - centers[ego_lane])
                            < 5
                        ):
                            continue
                        target_y = float(centers[actor_lane])
                    candidates.append(
                        (
                            abs(headway - 0.7),
                            int(rows[index]),
                            actor,
                            index,
                            target_y,
                            float(centers[ego_lane]),
                            headway,
                        )
                    )
            candidates.sort()
            allocation.append(
                dict(recording=int(rec), family=family, eligible=len(candidates))
            )
            if not candidates:
                continue
            _, row, actor, index, target_y, source_y, headway = candidates[0]
            kinds = (
                ("null", "braking", "acceleration", "corridor_release")
                if family == "following"
                else ("null", "cut_in")
            )
            for kind in kinds:
                specs.append(
                    dict(
                        case=len(specs),
                        row=row,
                        index=int(index),
                        recording=int(rec),
                        sequence_id=str(arrays["sequence_id"][row]),
                        family=family,
                        kind=kind,
                        actor=actor - 1,
                        onset_frame=ONSET,
                        onset_s=1.0,
                        target_y=target_y,
                        source_y=source_y,
                        initial_headway_s=headway,
                        brake_dose={0: 0, 1: 4, 2: 8}[int(rec) % 3],
                    )
                )
    # An existing natural free-flow case illustrates removal of the old trigger.
    index = int(np.flatnonzero(rows == 24319)[0])
    specs.append(
        dict(
            case=len(specs),
            row=24319,
            index=index,
            recording=21,
            sequence_id=str(arrays["sequence_id"][24319]),
            family="free_flow",
            kind="free_flow",
            actor=1,
            onset_frame=ONSET,
            onset_s=1.0,
            target_y=0.0,
            source_y=0.0,
            brake_dose=0,
        )
    )
    save_json(
        dict(
            cases=specs,
            allocation=allocation,
            selection="One speed-matched close follower and one adjacent rear vehicle per Test recording, closest to 0.7 s headway; chosen before outcomes. Road/plan eligibility counts retained.",
            intended_scope="Illustrative mechanism checks, not population performance estimates or parameter selection",
            protocols=dict(
                braking="Existing held-out doses -1.5/-3/-6 m/s2 for 0.8/1.6/2.4 s",
                acceleration="+4 m/s2 for 2 s, capped by the existing physical action limit",
                lateral="2 s quintic displacement to an adjacent mapped lane; all controls recovered once, then executed open-loop",
            ),
        ),
        OUTPUT / "intervention_cases.json",
    )
    return specs


def controls_for_case(spec, arrays):
    states = np.asarray(arrays["agent_states"][spec["row"], 24:174, 0]).copy()
    baseline = ego_controls(states[:-1], states[1:], 0.04)
    kind, start = spec["kind"], spec["onset_frame"]
    if kind == "braking":
        dose = spec["brake_dose"]
        duration = (20, 40, 60)[dose // 3]
        baseline[start : start + duration, 0] += (-1.5, -3.0, -6.0)[dose % 3]
    elif kind == "acceleration":
        baseline[start : start + 50, 0] += 4
    elif kind in ("cut_in", "corridor_release"):
        phase = np.clip((np.arange(150) * 0.04 - start * 0.04) / 2.0, 0, 1)
        derivative = (30 * phase**2 - 60 * phase**3 + 30 * phase**4) / 2.0
        states[:, 3] += (spec["target_y"] - spec["source_y"]) * derivative
        baseline = ego_controls(states[:-1], states[1:], 0.04)
    baseline[:, 0] = np.clip(baseline[:, 0], -8, 4)
    baseline[:, 1] = np.clip(baseline[:, 1], -0.6, 0.6)
    return baseline


def intervention_regression(config, specs):
    _, arrays, _, plans, population, fit = experiment._DATA
    selected = np.array([s["index"] for s in specs])
    rows = np.array([s["row"] for s in specs])
    controls = np.stack([controls_for_case(s, arrays) for s in specs])
    traces = {}
    results = []
    baseline = None
    for arm, seed in [("tracker", SEEDS[0])] + [("takeover", seed) for seed in SEEDS]:
        draws = scene_randomness(arrays["sequence_id"][rows], seed=seed)
        policy = (
            None
            if arm == "tracker"
            else experiment.make_controller(config, draws, population, fit)
        )
        trace = highway_rollout(
            arrays,
            rows,
            plans[selected],
            draws,
            controller=policy,
            ads_controls=controls,
            tracking_config=TrackingConfig(**config["tracking"]),
        )
        records = experiment.records_from_trace(
            trace,
            arrays,
            rows,
            arm=arm,
            seed=seed,
            controller=policy,
            baseline=baseline,
        )
        if arm == "tracker":
            baseline = trace
        for k, (spec, record) in enumerate(zip(specs, records)):
            record.update(
                case=spec["case"],
                stimulus=spec["kind"],
                family=spec["family"],
                target_actor=spec["actor"],
            )
            records[k] = record
            values = dict(
                **{name: value[k] for name, value in trace.items()},
                ego_controls=controls[k],
                plan=plans[selected[k]],
                dimensions=np.asarray(arrays["vehicle_dimensions"][spec["row"]]),
                valid=np.asarray(arrays["agent_valid"][spec["row"], 24]),
                maps=np.asarray(arrays["map_polylines"][spec["row"]]),
                map_valid=np.asarray(arrays["map_polyline_valid"][spec["row"]]),
            )
            traces.update(
                {
                    f"{spec['case']}_{arm}_{seed}/{name}": value
                    for name, value in values.items()
                }
            )
        results.extend(records)
        print(
            "Interventions",
            arm,
            seed,
            "runs",
            len(records),
            "admissions",
            sum(r["takeovers"] for r in records),
            flush=True,
        )
    base = {r["case"]: r for r in results if r["arm"] == "tracker"}
    summaries = {}
    for kind in sorted({s["kind"] for s in specs}):
        new = [r for r in results if r["arm"] == "takeover" and r["stimulus"] == kind]
        original = [
            r for r in results if r["arm"] == "tracker" and r["stimulus"] == kind
        ]
        summaries[kind] = dict(
            cases=len(original),
            bridge_runs=len(new),
            admitted_runs=sum(r["takeovers"] > 0 for r in new),
            vehicle_takeovers=sum(r["takeovers"] for r in new),
            reasons={
                name: sum(e["reason"] == code for r in new for e in r["events"])
                for code, name in experiment.REASONS.items()
            },
            tracker_collision_cases=sum(r["any_collision"] for r in original),
            bridge_collision_runs=sum(r["any_collision"] for r in new),
            introduced_collision_pairs=sum(
                r["any_collision"] and not base[r["case"]]["any_collision"] for r in new
            ),
            resolved_collision_pairs=sum(
                not r["any_collision"] and base[r["case"]]["any_collision"] for r in new
            ),
        )
    result = dict(
        complete=True,
        parameters=config["takeover"],
        cases=len(specs),
        runs=len(results),
        seeds=list(SEEDS),
        summary=summaries,
        records=results,
        artificially_intervened_ego=True,
        real_ADS_policy=False,
        parameters_fitted_on_these_cases=False,
        interpretation="Scripted Test mechanisms; all selected outcomes retained. Collisions are outcomes, not the takeover objective. No response-success rate or necessity label inferred from collision counts.",
    )
    np.savez_compressed(OUTPUT / "intervention_traces.npz", **traces)
    save_json(result, OUTPUT / "interventions.json")
    return result


def main():
    config = load_yaml(ROOT / "hierarchical_world_model/config.yaml")
    experiment.initialize_worker(config, "test")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    sources = list((ROOT / "hierarchical_world_model/src").rglob("*.py"))
    sources += [
        Path(__file__),
        ROOT / "hierarchical_world_model/config.yaml",
        ROOT / config["takeover"]["error_trajectories_path"],
    ]
    signature = {str(path.relative_to(ROOT)): file_sha256(path) for path in sources}
    save_json(
        dict(
            source_sha256=signature,
            parameters=config["takeover"],
            parameter_selection="Frozen natural Train/Val parameters; no selection on intervention outcomes",
            test_role="Previously diagnosed Test recordings",
        ),
        OUTPUT / "intervention_protocol.json",
    )
    specs = select_cases()
    intervention_regression(config, specs)
    assert all(file_sha256(ROOT / name) == digest for name, digest in signature.items())
    experiment.latency_report(config)


if __name__ == "__main__":
    main()
