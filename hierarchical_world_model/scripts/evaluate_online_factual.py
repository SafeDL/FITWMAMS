#!/usr/bin/env python3
"""Evaluate factual reconstruction and NPC response on the full highD Test split.

The logged ADS action drives one responsive rollout per row. NPC control uses
realized traffic and the current HiQR action throughout each rollout.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffusion.src.data import ANCHOR_INDEX  # noqa: E402
from external_model_baselines.models.bayesian_ma_idm.src.model import (  # noqa: E402
    load_posterior,
    sample_driver_joint,
)
from hierarchical_world_model.src.data import prepare_experiment_data  # noqa: E402
from hierarchical_world_model.src.collision_attribution import collision_events  # noqa: E402
from hierarchical_world_model.src.evaluation import rollout  # noqa: E402
from hierarchical_world_model.src.planner import frozen_diffusion_plans  # noqa: E402
from hierarchical_world_model.src.protocol import load_protocol_config  # noqa: E402
from hierarchical_world_model.src.provenance import online_runtime_sha256  # noqa: E402
from hierarchical_world_model.src.stochastic_drivers.online import (  # noqa: E402
    DEFAULT_MA_IDM_POSTERIOR,
    OnlineMAIDMController,
    nearest_leader_observation,
    planned_adjacent_lane_intent,
)
from hierarchical_world_model.src.train import load_checkpoint  # noqa: E402
from traffic_components.src.core.utils import file_sha256, save_json, select_device  # noqa: E402


def _theta_for_rows(rows: np.ndarray, posterior_path: Path) -> np.ndarray:
    posterior = load_posterior(posterior_path)
    result = np.empty((len(rows), 6, 5), np.float32)
    for index, row in enumerate(rows):
        for slot in range(6):
            digest = hashlib.sha256(f"online_factual_v1:{int(row)}:{slot}".encode()).digest()
            rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
            result[index, slot] = sample_driver_joint(posterior, rng)[:5]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "hierarchical_world_model/config/world_model.yaml")
    parser.add_argument("--posterior", type=Path, default=DEFAULT_MA_IDM_POSTERIOR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--hiqr-only", action="store_true", help="single-world factual ablation")
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "results/hierarchical_world_model/evaluation/factual_test.json",
    )
    args = parser.parse_args()
    controller_hash = file_sha256(
        ROOT / "hierarchical_world_model/src/stochastic_drivers/online.py"
    )
    config = load_protocol_config(args.config.resolve())
    experiment = prepare_experiment_data(config, ROOT)
    full_test_rows = np.asarray(experiment.test_rows, np.int64)
    rows = full_test_rows if args.max_rows is None else full_test_rows[:args.max_rows]
    if len(rows) < 1:
        raise ValueError("no test rows selected")
    device = select_device(args.device)
    arrays = experiment.bundle.arrays
    states = np.asarray(arrays["agent_states"][rows], np.float32)
    valid = np.asarray(arrays["agent_valid"][rows], bool)
    observed_speed = np.linalg.norm(states[:, ANCHOR_INDEX:174, :, 2:4], axis=-1)
    observed_speed = observed_speed[valid[:, ANCHOR_INDEX:174]]
    maps = np.asarray(arrays["map_polylines"][rows], np.float32)
    map_valid = np.asarray(arrays["map_polyline_valid"][rows], bool)
    plans = frozen_diffusion_plans(
        experiment.bundle, rows,
        checkpoint=config["paths"]["diffusion_checkpoint"],
        output_dir=args.output.resolve().parent / ("plans_full" if args.max_rows is None else f"plans_{len(rows)}"),
        device=device, batch_size=args.batch_size, ddim_steps=20,
        experiment_scope="full" if args.max_rows is None else f"prefix_{len(rows)}",
    )
    theta = _theta_for_rows(rows, args.posterior.resolve())
    model, _ = load_checkpoint(Path(config["paths"]["evaluation_checkpoint"]), device=device)
    model.eval()
    sum_distance = 0.0
    count_distance = 0
    sum_final = 0.0
    count_final = 0
    correction_count = 0
    active_count = 0
    corrected_scenes = 0
    ads_leader_correction_frames = 0
    npc_leader_correction_frames = 0
    npc_leader_correction_scenes = 0
    factual_raw_overlap_scenes = 0
    factual_npc_npc_overlap_scenes = 0
    lane_repair_count = 0
    lane_repair_scenes = 0
    spatial_path_count = 0
    spatial_path_scenes = 0
    autonomous_lane_frames = 0
    autonomous_lane_scenes = 0
    lane_control_examples: list[dict[str, int]] = []
    planned_lane_changes = 0
    completed_planned_lane_changes = 0
    planned_lane_terminal_errors: list[np.ndarray] = []
    finite_scenes = 0
    scene_ade_records: list[dict[str, float | int]] = []
    for start in range(0, len(rows), args.batch_size):
        stop = min(start + args.batch_size, len(rows))
        result = rollout(
            model, states[start:stop], valid[start:stop], plans[start:stop],
            maps[start:stop], map_valid[start:stop],
            device=device, history_frames=25, motion_seed=None,
            controller=(
                None if args.hiqr_only
                else OnlineMAIDMController(theta[start:stop]).to(device)
            ),
            record_policy_trace=False,
        )
        target = states[start:stop, ANCHOR_INDEX + 1:174]
        active = valid[start:stop, ANCHOR_INDEX, 1:]
        distance = np.linalg.norm(
            result.states[..., 1:, :2] - target[..., 1:, :2], axis=-1
        )
        mask = np.broadcast_to(active[:, None], distance.shape)
        scene_denominator = mask.sum(axis=(1, 2))
        scene_ade = np.divide(
            (distance * mask).sum(axis=(1, 2)), scene_denominator,
            out=np.zeros(stop - start, np.float64), where=scene_denominator > 0,
        )
        scene_ade_records.extend(
            {"test_row": int(row), "ADE_m": float(ade)}
            for row, ade in zip(rows[start:stop], scene_ade, strict=True)
        )
        sum_distance += float(distance[mask].sum())
        count_distance += int(mask.sum())
        sum_final += float(distance[:, -1][active].sum())
        count_final += int(active.sum())
        diagnostics = result.controller_diagnostics or {}
        correction = np.abs(
            diagnostics.get("delta_ax", np.zeros((stop - start, 149, 6), np.float32))
        ) > 1.0e-4
        correction &= active[:, None]
        correction_count += int(correction.sum())
        active_count += int(mask.sum())
        corrected_scenes += int(correction.any(axis=(1, 2)).sum())
        observed = np.concatenate(
            (states[start:stop, ANCHOR_INDEX:ANCHOR_INDEX + 1], result.states[:, :-1]),
            axis=1,
        )
        repeated_valid = np.repeat(
            valid[start:stop, ANCHOR_INDEX:ANCHOR_INDEX + 1], 149, axis=1
        )
        leader_index = nearest_leader_observation(
            torch.from_numpy(observed.reshape(-1, 7, 6)),
            torch.from_numpy(repeated_valid.reshape(-1, 7)),
            prediction_horizon_s=2.0,
        )[3].reshape(stop - start, 149, 6).numpy()
        ads_leader_correction_frames += int((correction & (leader_index == 0)).sum())
        npc_leader_correction = correction & (leader_index > 0)
        npc_leader_correction_frames += int(npc_leader_correction.sum())
        npc_leader_correction_scenes += int(
            npc_leader_correction.any(axis=(1, 2)).sum()
        )
        geometry = collision_events(
            np.concatenate(
                (states[start:stop, ANCHOR_INDEX:ANCHOR_INDEX + 1], result.states),
                axis=1,
            ),
            active,
        )
        factual_raw_overlap_scenes += int(geometry["raw"].sum())
        factual_npc_npc_overlap_scenes += int(geometry["npc_npc"].sum())
        autonomous_lane = np.asarray(
            diagnostics.get("autonomous_lane_active", np.zeros_like(correction)), bool
        )
        autonomous_lane &= active[:, None]
        autonomous_lane_frames += int(autonomous_lane.sum())
        autonomous_lane_scenes += int(autonomous_lane.any(axis=(1, 2)).sum())
        lane_repair = np.asarray(diagnostics.get("policy_active", np.zeros_like(correction)), bool)
        lane_repair &= ~autonomous_lane
        lane_repair &= active[:, None]
        lane_repair_count += int(lane_repair.sum())
        lane_repair_scenes += int(lane_repair.any(axis=(1, 2)).sum())
        spatial_path = np.asarray(
            diagnostics.get("spatial_path_active", np.zeros_like(correction)), bool
        )
        spatial_path &= active[:, None]
        spatial_path_count += int(spatial_path.sum())
        spatial_path_scenes += int(spatial_path.any(axis=(1, 2)).sum())
        for local_scene, npc_slot in np.argwhere(lane_repair.any(axis=1)):
            if len(lane_control_examples) >= 20:
                break
            lane_control_examples.append({
                "test_row": int(rows[start + local_scene]),
                "npc_slot": int(npc_slot + 1),
                "active_frames": int(lane_repair[local_scene, :, npc_slot].sum()),
            })
        origin_y = states[start:stop, ANCHOR_INDEX, 1:, 1]
        terminal_y = plans[start:stop, -1, :, 1]
        lane_intent, _, target_lane_y_tensor = planned_adjacent_lane_intent(
            torch.from_numpy(origin_y), torch.from_numpy(terminal_y),
            torch.from_numpy(maps[start:stop]), torch.from_numpy(map_valid[start:stop]),
        )
        planned_change = active & lane_intent.numpy()
        target_lane_y = target_lane_y_tensor.numpy()
        final_npc = result.states[:, -1, 1:]
        terminal_error = np.abs(final_npc[..., 1] - target_lane_y)
        final_heading = np.arctan2(final_npc[..., 3], final_npc[..., 2])
        planned_lane_changes += int(planned_change.sum())
        completed_planned_lane_changes += int((
            planned_change & (terminal_error < 0.75) & (np.abs(final_heading) < 0.05)
        ).sum())
        planned_lane_terminal_errors.append(terminal_error[planned_change])
        finite_scenes += int(np.isfinite(result.states).all(axis=(1, 2, 3)).sum())
        if stop == len(rows) or stop % 1024 == 0:
            print(f"completed {stop}/{len(rows)} test scenes", flush=True)
    report = {
        "schema": "online_ma_idm_factual_test_v1",
        "test_split_size": int(len(full_test_rows)),
        "evaluated_rows": int(len(rows)),
        "full_test": args.max_rows is None,
        "protocol": (
            "logged ADS + frozen Diffusion plan + HiQR-only; one ablation rollout per scene"
            if args.hiqr_only else
            "logged ADS + frozen Diffusion plan + HiQR + online MA-IDM; one responsive rollout per scene"
        ),
        "controller_mode": "hiqr_only_ablation" if args.hiqr_only else "online_ma_idm",
        "legacy_handcrafted_adapter_applied": False,
        "posterior": str(args.posterior.resolve()),
        "online_controller_sha256": controller_hash,
        "online_runtime_sha256": online_runtime_sha256(),
        "evaluated_observed_max_speed_mps": float(observed_speed.max()),
        "evaluated_observed_speed_samples_above_50_mps": int((observed_speed > 50.0).sum()),
        "ADE_m": sum_distance / count_distance,
        "scene_ADE_p95_m": float(np.quantile([item["ADE_m"] for item in scene_ade_records], 0.95)),
        "scene_ADE_p99_m": float(np.quantile([item["ADE_m"] for item in scene_ade_records], 0.99)),
        "worst_scene_ADE": sorted(
            scene_ade_records, key=lambda item: item["ADE_m"], reverse=True
        )[:10],
        "FDE_m": sum_final / count_final,
        "npc_frame_correction_rate": correction_count / active_count,
        "scene_with_correction_rate": corrected_scenes / len(rows),
        "ads_leader_correction_frames": ads_leader_correction_frames,
        "npc_leader_correction_frames": npc_leader_correction_frames,
        "npc_leader_correction_scenes": npc_leader_correction_scenes,
        "factual_raw_overlap_scenes": factual_raw_overlap_scenes,
        "factual_npc_npc_overlap_scenes": factual_npc_npc_overlap_scenes,
        "npc_lane_repair_frame_rate": lane_repair_count / active_count,
        "scene_with_lane_repair_rate": lane_repair_scenes / len(rows),
        "npc_spatial_path_frame_rate": spatial_path_count / active_count,
        "npc_autonomous_lane_frames": autonomous_lane_frames,
        "npc_autonomous_lane_scenes": autonomous_lane_scenes,
        "scene_with_spatial_path_rate": spatial_path_scenes / len(rows),
        "npc_lane_semantic_control_examples": lane_control_examples,
        "npc_planned_lane_change_count": planned_lane_changes,
        "npc_planned_lane_completion_rate": (
            completed_planned_lane_changes / planned_lane_changes
            if planned_lane_changes else None
        ),
        "npc_planned_lane_terminal_error_p95_m": (
            float(np.quantile(np.concatenate(planned_lane_terminal_errors), 0.95))
            if planned_lane_changes else None
        ),
        "npc_planned_lane_completion_criteria": "target map lane centre error <0.75 m and heading <0.05 rad",
        "finite_scene_rate": finite_scenes / len(rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(report, args.output)
    print(report)


if __name__ == "__main__":
    main()
