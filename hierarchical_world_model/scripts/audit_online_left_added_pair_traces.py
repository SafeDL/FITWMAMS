#!/usr/bin/env python3
"""Regress three previously added ADS cut-in overlaps against the HiQR base."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffusion.src.data import ANCHOR_INDEX  # noqa: E402
from hierarchical_world_model.scripts.evaluate_online_factual import _theta_for_rows  # noqa: E402
from hierarchical_world_model.src.ads_interventions import SemanticLaneChangePolicy  # noqa: E402
from hierarchical_world_model.src.collision_attribution import collision_events  # noqa: E402
from hierarchical_world_model.src.data import prepare_experiment_data  # noqa: E402
from hierarchical_world_model.src.evaluation import _logged_ego_actions, rollout  # noqa: E402
from hierarchical_world_model.src.planner import frozen_diffusion_plans  # noqa: E402
from hierarchical_world_model.src.protocol import load_protocol_config  # noqa: E402
from hierarchical_world_model.src.provenance import online_runtime_sha256  # noqa: E402
from hierarchical_world_model.src.reaction_controller import ReactionControllerOutput  # noqa: E402
from hierarchical_world_model.src.stochastic_drivers.online import (  # noqa: E402
    DEFAULT_MA_IDM_POSTERIOR, OnlineMAIDMController, nearest_leader_observation,
)
from hierarchical_world_model.src.train import load_checkpoint  # noqa: E402
from traffic_components.src.core.utils import file_sha256, save_json, select_device  # noqa: E402


DIRECTORY = ROOT / "results/hierarchical_world_model/evaluation"
HISTORICAL_ADDED_ROWS = (58883, 634, 76098, 19594, 73169)
HISTORICAL_ADDED_SLOTS = (4, 4, 4, 4, 4)


class BaseOnly(torch.nn.Module):
    """The HiQR base with both the old adapter and online MA-IDM disabled."""

    def forward(self, context, *, deterministic=False):
        del deterministic
        zeros = torch.zeros_like(context.base_actions[:, 0, :, 0])
        inactive = torch.zeros_like(zeros, dtype=torch.bool)
        return ReactionControllerOutput(
            actions=context.base_actions, alpha=zeros, delta_ax=zeros,
            active=inactive,
        )


class TrackedOnline(OnlineMAIDMController):
    def __init__(self, theta):
        super().__init__(theta)
        self.pass_gate = []
        self.leader_index = []

    def _passing_cutin_clearance(self, context):
        result = super()._passing_cutin_clearance(context)
        self.pass_gate.append(result.detach().cpu().numpy())
        _, _, _, leader_index = nearest_leader_observation(
            context.current, context.current_valid,
            lane_half_width_m=self.lane_half_width_m,
            vehicle_length_m=self.vehicle_length_m,
            prediction_horizon_s=self.prediction_horizon_s,
        )
        self.leader_index.append(leader_index.detach().cpu().numpy())
        return result


def _first_overlap(trace: np.ndarray, npc_slot: int) -> int | None:
    dx = np.abs(trace[:, 0, 0] - trace[:, npc_slot, 0])
    dy = np.abs(trace[:, 0, 1] - trace[:, npc_slot, 1])
    hits = np.flatnonzero((dx < 4.8) & (dy < 1.8))
    return None if len(hits) == 0 else int(hits[0])


def main() -> None:
    rows = np.asarray(HISTORICAL_ADDED_ROWS, np.int64)
    slots = np.asarray(HISTORICAL_ADDED_SLOTS, np.int64)
    config = load_protocol_config(ROOT / "hierarchical_world_model/config/world_model.yaml")
    experiment = prepare_experiment_data(config, ROOT)
    test_rows = np.asarray(experiment.test_rows, np.int64)
    position = {int(row): index for index, row in enumerate(test_rows)}
    selected = np.asarray([position[int(row)] for row in rows], np.int64)
    arrays = experiment.bundle.arrays
    states = np.asarray(arrays["agent_states"][rows], np.float32)
    valid = np.asarray(arrays["agent_valid"][rows], bool)
    maps = np.asarray(arrays["map_polylines"][rows], np.float32)
    map_valid = np.asarray(arrays["map_polyline_valid"][rows], bool)
    device = select_device("cuda")
    plans = frozen_diffusion_plans(
        experiment.bundle, test_rows,
        checkpoint=config["paths"]["diffusion_checkpoint"],
        output_dir=DIRECTORY / "plans_full", device=device,
        batch_size=64, ddim_steps=20, experiment_scope="full",
    )[selected]
    model, _ = load_checkpoint(Path(config["paths"]["evaluation_checkpoint"]), device=device)
    model.eval()
    theta = _theta_for_rows(rows, DEFAULT_MA_IDM_POSTERIOR)
    controls = _logged_ego_actions(states, valid)

    def execute(controller):
        policy = SemanticLaneChangePolicy(
            controls, states[:, ANCHOR_INDEX],
            map_polylines=maps, map_polyline_valid=map_valid,
            direction="left",
        )
        return rollout(
            model, states, valid, plans, maps, map_valid,
            device=device, history_frames=25, motion_seed=None,
            ads_policy=policy, controller=controller,
            record_policy_trace=False,
        )

    online_controller = TrackedOnline(theta).to(device)
    branches = {
        "online": execute(online_controller),
        "controller_free_hiqr": execute(None),
        "hiqr_base_only": execute(BaseOnly().to(device)),
    }
    gate = np.stack(online_controller.pass_gate, axis=1)
    leader_indices = np.stack(online_controller.leader_index, axis=1)
    records = []
    for index, row in enumerate(rows):
        slot = int(slots[index])
        record = {
            "test_row": int(row), "npc_slot": slot, "branches": {},
            "mapped_lane_centers_y_m": [
                float(value) for value in maps[index, map_valid[index].any(axis=-1), 0, 1]
            ],
        }
        for name, branch in branches.items():
            trace = np.concatenate(
                (states[index:index + 1, ANCHOR_INDEX], branch.states[index]), axis=0
            )
            events = collision_events(trace[None], valid[index:index + 1, ANCHOR_INDEX, 1:])
            record["branches"][name] = {
                "first_ads_npc_overlap_frame": _first_overlap(trace, slot),
                "raw_overlap": bool(events["raw"][0]),
                "npc_npc_overlap": bool(events["npc_npc"][0]),
                "ads_minus_npc_x_at_75_m": float(trace[75, 0, 0] - trace[75, slot, 0]),
                "npc_x_at_75_m": float(trace[75, slot, 0]),
                "npc_ax_mean_frames_31_70_mps2": float(
                    branch.background_actions[index, 30:70, slot - 1, 0].mean()
                ),
            }
        online = branches["online"]
        base = branches["hiqr_base_only"]
        online_trace = np.concatenate(
            (states[index:index + 1, ANCHOR_INDEX], online.states[index]), axis=0
        )
        base_trace = np.concatenate(
            (states[index:index + 1, ANCHOR_INDEX], base.states[index]), axis=0
        )
        first = record["branches"]["online"]["first_ads_npc_overlap_frame"]
        frames = sorted(set((25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90))
                        | ({first} if first is not None else set()))
        record["online_pass_gate_frames"] = int(gate[index, :, slot - 1].sum())
        record["online_pass_gate_frame_indices"] = (
            np.flatnonzero(gate[index, :, slot - 1]) + 1
        ).tolist()
        record["selected_leader_slots_frames_25_45"] = leader_indices[
            index, 24:45, slot - 1
        ].tolist()
        record["trace"] = [
            {
                "frame": frame,
                "ads_x_m": float(online_trace[frame, 0, 0]),
                "ads_y_m": float(online_trace[frame, 0, 1]),
                "ads_vx_mps": float(online_trace[frame, 0, 2]),
                "ads_vy_mps": float(online_trace[frame, 0, 3]),
                "estimated_entry_time_s": float(min(
                    2.0,
                    max(0.0, online_trace[frame, slot, 1]
                        - online_trace[frame, 0, 1] - 1.8)
                    / max(1.0, online_trace[frame, 0, 3]),
                )),
                "online_npc_x_m": float(online_trace[frame, slot, 0]),
                "online_npc_y_m": float(online_trace[frame, slot, 1]),
                "online_npc_vx_mps": float(online_trace[frame, slot, 2]),
                "base_npc_x_m": float(base_trace[frame, slot, 0]),
                "base_npc_y_m": float(base_trace[frame, slot, 1]),
                "online_npc_ax_mps2": float(online.background_actions[index, frame - 1, slot - 1, 0]),
                "hiqr_base_ax_at_online_state_mps2": float(
                    online.base_background_actions[index, frame - 1, slot - 1, 0]
                ),
                "hiqr_base_only_ax_mps2": float(base.background_actions[index, frame - 1, slot - 1, 0]),
                "online_pass_gate": bool(gate[index, frame - 1, slot - 1]),
                "selected_leader_slot": int(leader_indices[index, frame - 1, slot - 1]),
            }
            for frame in frames if frame is not None and frame < 150
        ]
        records.append(record)
    report = {
        "schema": "online_left_cutin_regressions_v3",
        "historical_added_pairs": len(rows),
        "online_controller_sha256": file_sha256(
            ROOT / "hierarchical_world_model/src/stochastic_drivers/online.py"
        ),
        "online_runtime_sha256": online_runtime_sha256(),
        "records": records,
    }
    if any(
        item["branches"]["online"]["raw_overlap"]
        and not item["branches"]["controller_free_hiqr"]["raw_overlap"]
        for item in records
    ):
        raise AssertionError("online response introduced an ADS cut-in overlap")
    if any(abs(
        item["branches"]["controller_free_hiqr"]["npc_x_at_75_m"]
        - item["branches"]["hiqr_base_only"]["npc_x_at_75_m"]
    ) > 0.001 for item in records):
        raise AssertionError("controller-free world is not the HiQR base")
    output = DIRECTORY / "left_added_pair_trace_audit.json"
    save_json(report, output)
    print({"output": str(output), "summary": [
        {"row": item["test_row"], "branches": item["branches"],
         "pass_gate_frames": item["online_pass_gate_frames"]} for item in records
    ]})


if __name__ == "__main__":
    main()
