#!/usr/bin/env python3
"""Scene-level ADS acceleration dose ordering for the online NPC world."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffusion.src.data import ANCHOR_INDEX  # noqa: E402
from hierarchical_world_model.scripts.evaluate_online_ads import DOSES, EFFECT, _screen  # noqa: E402
from hierarchical_world_model.scripts.evaluate_online_factual import _theta_for_rows  # noqa: E402
from hierarchical_world_model.src.ads_interventions import AbsoluteAccelerationPolicy  # noqa: E402
from hierarchical_world_model.src.data import prepare_experiment_data  # noqa: E402
from hierarchical_world_model.src.evaluation import _logged_ego_actions, rollout  # noqa: E402
from hierarchical_world_model.src.planner import frozen_diffusion_plans  # noqa: E402
from hierarchical_world_model.src.protocol import load_protocol_config  # noqa: E402
from hierarchical_world_model.src.provenance import online_runtime_sha256  # noqa: E402
from hierarchical_world_model.src.stochastic_drivers.online import (  # noqa: E402
    DEFAULT_MA_IDM_POSTERIOR, OnlineMAIDMController,
)
from hierarchical_world_model.src.train import load_checkpoint  # noqa: E402
from traffic_components.src.core.utils import file_sha256, save_json, select_device  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "results/hierarchical_world_model/evaluation/dose_order_test.json",
    )
    args = parser.parse_args()
    controller_hash = file_sha256(
        ROOT / "hierarchical_world_model/src/stochastic_drivers/online.py"
    )
    config = load_protocol_config(ROOT / "hierarchical_world_model/config/world_model.yaml")
    experiment = prepare_experiment_data(config, ROOT)
    test_rows = np.asarray(experiment.test_rows, np.int64)
    arrays = experiment.bundle.arrays
    all_states = np.asarray(arrays["agent_states"][test_rows], np.float32)
    all_valid = np.asarray(arrays["agent_valid"][test_rows], bool)
    selected, receiver = _screen(all_states, all_valid, "same")
    rows = test_rows[selected]
    states, valid = all_states[selected], all_valid[selected]
    receiver = receiver[selected]
    maps = np.asarray(arrays["map_polylines"][rows], np.float32)
    map_valid = np.asarray(arrays["map_polyline_valid"][rows], bool)
    device = select_device("cuda")
    full_plans = frozen_diffusion_plans(
        experiment.bundle, test_rows, checkpoint=config["paths"]["diffusion_checkpoint"],
        output_dir=ROOT / "results/hierarchical_world_model/evaluation/plans_full",
        device=device, batch_size=64, ddim_steps=20, experiment_scope="full",
    )
    plans = full_plans[selected]
    theta = _theta_for_rows(test_rows, DEFAULT_MA_IDM_POSTERIOR)[selected]
    model, _ = load_checkpoint(Path(config["paths"]["evaluation_checkpoint"]), device=device)
    model.eval()
    effects = np.empty((len(rows), len(DOSES)), np.float32)
    autonomous_lane = np.zeros((len(rows), len(DOSES)), bool)
    for begin in range(0, len(rows), 64):
        end = min(begin + 64, len(rows))
        part = slice(begin, end)
        controls = _logged_ego_actions(states[part], valid[part])

        def execute(policy=None):
            return rollout(
                model, states[part], valid[part], plans[part], maps[part], map_valid[part],
                device=device, history_frames=25, motion_seed=None,
                ads_policy=policy,
                controller=OnlineMAIDMController(theta[part]).to(device),
                record_policy_trace=False,
            )

        # Order is deliberate: no logged-ADS world is precomputed or passed
        # to an intervention. It is used only to measure the completed runs.
        branches = [execute(AbsoluteAccelerationPolicy(controls, dose)) for dose in DOSES]
        reference = execute()
        local = np.arange(end - begin)
        target = receiver[part]
        baseline = reference.background_actions[local, :, target, 0]
        for column, branch in enumerate(branches):
            action = branch.background_actions[local, :, target, 0]
            effects[part, column] = (action[:, EFFECT] - baseline[:, EFFECT]).mean(axis=1)
            diagnostics = branch.controller_diagnostics or {}
            lane_active = np.asarray(
                diagnostics.get(
                    "autonomous_lane_active",
                    np.zeros((end - begin, 149, 6), bool),
                ), bool,
            )
            autonomous_lane[part, column] = lane_active[local, :, target].any(axis=1)
        print(f"completed {end}/{len(rows)}", flush=True)
    strict = np.diff(effects, axis=1) >= -1.0e-5
    tolerance = np.diff(effects, axis=1) >= -0.05
    adjacent_change = np.diff(effects, axis=1)
    inversion = np.maximum(-adjacent_change, 0.0)
    anchor = states[:, ANCHOR_INDEX]
    gap = anchor[:, 0, 0] - anchor[np.arange(len(rows)), receiver + 1, 0] - 4.8
    nearby = gap < 45.0
    no_autonomous_lane = ~autonomous_lane.any(axis=1)
    report = {
        "schema": "online_ma_idm_dose_order_test_v1",
        "full_test_rows_screened": len(test_rows),
        "same_lane_rear_scenes": len(rows),
        "within_45m_scenes": int(nearby.sum()),
        "doses_mps2": list(DOSES),
        "effect_window_frames": [EFFECT.start, EFFECT.stop],
        "strict_all_adjacent_pairs_monotonic_rate": float(strict.all(axis=1).mean()),
        "within_45m_strict_monotonic_rate": float(strict[nearby].all(axis=1).mean()),
        "tolerance_0p05_all_pairs_monotonic_rate": float(tolerance.all(axis=1).mean()),
        "autonomous_lane_scenes_by_dose": autonomous_lane.sum(axis=0).tolist(),
        "autonomous_lane_any_dose_scenes": int((~no_autonomous_lane).sum()),
        "no_autonomous_lane_scenes": int(no_autonomous_lane.sum()),
        "no_autonomous_lane_strict_monotonic_rate": float(
            strict[no_autonomous_lane].all(axis=1).mean()
        ),
        "no_autonomous_lane_tolerance_0p05_monotonic_rate": float(
            tolerance[no_autonomous_lane].all(axis=1).mean()
        ),
        "autonomous_lane_strict_monotonic_rate": float(
            strict[~no_autonomous_lane].all(axis=1).mean()
        ),
        "autonomous_lane_tolerance_0p05_monotonic_rate": float(
            tolerance[~no_autonomous_lane].all(axis=1).mean()
        ),
        "adjacent_dose_pairs": [
            {
                "lower_mps2": float(DOSES[column]),
                "upper_mps2": float(DOSES[column + 1]),
                "strict_inversion_scenes": int((~strict[:, column]).sum()),
                "inversion_over_0p05_scenes": int((~tolerance[:, column]).sum()),
                "maximum_inversion_mps2": float(inversion[:, column].max()),
                "inversion_p95_mps2": float(np.quantile(inversion[:, column], 0.95)),
            }
            for column in range(len(DOSES) - 1)
        ],
        "worst_inversion_scenes": [
            {
                "test_row": int(rows[index]),
                "anchor_gap_m": float(gap[index]),
                "worst_inversion_mps2": float(inversion[index].max()),
                "effects_mps2": effects[index].tolist(),
            }
            for index in np.argsort(-inversion.max(axis=1), kind="stable")[:20]
            if inversion[index].max() > 0.05
        ],
        "mean_effect_mps2_by_dose": effects.mean(axis=0).tolist(),
        "online_controller_sha256": controller_hash,
        "online_runtime_sha256": online_runtime_sha256(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(report, args.output)
    print(report)


if __name__ == "__main__":
    main()
