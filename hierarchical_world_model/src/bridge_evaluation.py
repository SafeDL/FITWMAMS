"""Shared data, event records and timing for current bridge experiments."""

import json
import time
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from .core.utils import save_json
from .data import ego_controls, prepare_experiment_data
from .highway import HighwayEnvClosedLoopWorld
from .motion import observed_acceleration
from .shared_evaluation import load_shared_components, scene_randomness
from .stochastic_drivers.driver_policy import (
    anticipated_entry_features,
    physical_leader_observation,
)
from .stochastic_drivers.idm_population import sample_theta
from .stochastic_drivers.online import calibrated_mobil_parameters
from .takeover import TakeoverConfig, TakeoverController
from .tracking import TrackingConfig

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "results/hierarchical_world_model/bridge"
SEEDS = (20261015, 20261016, 20261017)
DT = 0.04
_DATA = None
REASONS = {
    1: "initial_conflict",
    2: "runtime_conflict",
    3: "expiry",
    4: "following",
    5: "release",
}


def save_compact_json(value, path):
    """Atomic bulk records; human-readable summaries still use save_json."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".pending")
    pending.write_text(json.dumps(value, separators=(",", ":"), allow_nan=False))
    pending.replace(path)


def load_plans(data, split):
    rows = getattr(
        data,
        {"train": "train_rows", "val": "validation_rows", "test": "test_rows"}[split],
    )
    with np.load(
        ROOT
        / "results/hierarchical_world_model/diffusion_plans"
        / split
        / "positions.npz"
    ) as d:
        if not np.array_equal(
            d["sequence_id"].astype(str),
            np.asarray(data.bundle.arrays["sequence_id"])[rows].astype(str),
        ):
            raise ValueError("Cached Diffusion plans do not match scene identity/order")
        return rows, d["positions"]


def initialize_worker(config, split):
    threadpool_limits(limits=1)
    global _DATA
    torch.set_num_threads(1)
    data = prepare_experiment_data(config, ROOT)
    rows, plans = load_plans(data, split)
    population, fit = load_shared_components()
    _DATA = (
        config,
        data.bundle.arrays,
        rows,
        plans,
        population,
        fit,
    )


def trace_records(trace, arrays, rows, *, arm, kind, seed, doses, starts):
    result = []
    log = np.asarray(arrays["agent_states"][rows, 25:174])
    for k, row in enumerate(rows):
        initial = np.asarray(arrays["agent_valid"][row, 24, 1:])
        valid = initial[None] & np.asarray(arrays["agent_valid"][row, 25:174, 1:])
        error = np.linalg.norm(
            trace["states"][k, :, 1:, :2] - log[k, :, 1:, :2], axis=-1
        )
        entered = trace.get("takeover_started", np.zeros((len(rows), 149, 6), bool))[k]
        owned = trace.get("takeover_owned", np.zeros((len(rows), 149, 6), bool))[k]
        reasons = trace.get("takeover_reason", np.zeros((len(rows), 149, 6), int))[k]
        pairs = trace["pairs"][k]
        contact = np.flatnonzero(pairs.any(axis=(-2, -1)) | trace["crashed"][k].any(-1))
        first = int(contact[0]) if len(contact) else -1
        target = -1
        if first >= 0:
            involved = np.flatnonzero(pairs[first, 0, 1:] | pairs[first, 1:, 0])
            if len(involved):
                target = int(involved[0])
        entry_frames = np.where(entered.any(0), entered.argmax(0), -1)
        end = first if first >= 0 else 149
        delta = trace["actions"][k, :end] - trace["base_actions"][k, :end]
        changed = (
            (np.abs(delta[..., 0]) > 0.2) | (np.abs(delta[..., 1]) > 0.01)
        ) & owned[:end]
        delays = [
            (
                int(np.flatnonzero(changed[:, actor])[0] - entry_frames[actor])
                if changed[:, actor].any()
                else -1
            )
            for actor in range(6)
        ]
        events = [
            {
                "actor": int(actor),
                "frame": int(frame),
                "reason": int(reasons[frame, actor]),
                "support": float(trace["takeover_score"][k, frame, actor]),
                "predicted_conflict_s": float(
                    trace["takeover_conflict_time_s"][k, frame, actor]
                ),
                "influencer": int(trace["takeover_influencer"][k, frame, actor]),
            }
            for frame, actor in np.argwhere(entered)
        ]
        result.append(
            {
                "row": int(row),
                "recording": int(str(arrays["sequence_id"][row]).split("_")[1]),
                "arm": arm,
                "stimulus": kind,
                "seed": seed,
                "dose": int(doses[k]),
                "start_frame": int(starts[k]),
                "actor_frames": int(valid.sum()),
                "position_error_sum_m": float(error[valid].sum(dtype=np.float64)),
                "ADE_m": float(error[valid].mean()) if valid.any() else None,
                "any_collision": bool(trace["crashed"][k].any()),
                "ego_collision": bool(trace["crashed"][k, :, 0].any()),
                "NPC_offroad": bool((trace["offroad"][k, :, 1:] & valid).any()),
                "takeovers": int(entered.sum()),
                "initial_takeovers": int((reasons == 1).sum()),
                "interaction_takeovers": int(np.isin(reasons, [2, 4, 5]).sum()),
                "following_takeovers": int((reasons == 4).sum()),
                "release_takeovers": int((reasons == 5).sum()),
                "expiry_takeovers": int((reasons == 3).sum()),
                "repeated_takeovers": int((entered.sum(0) > 1).sum()),
                "ownership_releases": int((owned[:-1] & ~owned[1:]).sum()),
                "first_collision_frame": first,
                "target_actor": target,
                "admission_frames": entry_frames.tolist(),
                "events": events,
                "precontact_actor_frames": int(valid[:end].sum()),
                "precontact_reference_error_sum_m": float(
                    error[:end][valid[:end]].sum(dtype=np.float64)
                ),
                "response_delay_frames": delays,
                "response_duration_frames": changed.sum(0).astype(int).tolist(),
                "response_peak_ax_delta_mps2": (
                    np.abs(delta[..., 0]).max(0).tolist() if end else [0.0] * 6
                ),
                "response_peak_yaw_delta_rps": (
                    np.abs(delta[..., 1]).max(0).tolist() if end else [0.0] * 6
                ),
            }
        )
    return result


class AuditedTakeover(TakeoverController):
    """Record visible interaction at handover without changing policy actions."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.events = []

    def forward(self, context, *, deterministic=False):
        result = super().forward(context, deterministic=deterministic)
        if not result.takeover_started.any():
            return result
        current, valid, dims = (
            context.current,
            context.current_valid,
            context.vehicle_dimensions,
        )
        gap, speed, leader_speed, leader = physical_leader_observation(
            current, valid, dims
        )
        past = context.history[:, -2]
        old_gap, old_speed, _, old_leader = physical_leader_observation(
            past, context.history_valid[:, -2], dims
        )
        acceleration = observed_acceleration(
            current, history=context.history, history_valid=context.history_valid
        )
        entries = anticipated_entry_features(
            current,
            valid,
            dims,
            history=context.history,
            history_valid=context.history_valid,
        )
        plan = context.planned_path_xy[:, context.response_index :]
        for batch, slot in result.takeover_started.nonzero().tolist():
            actor = slot + 1
            influencer = int(result.takeover_influencer[batch, slot])
            # This is the plan's observed local speed trend, not a response label.
            plan_acceleration = None
            if len(plan[batch]) > 1:
                end = min(5, len(plan[batch]) - 1)
                initial_v = (
                    plan[batch, 0, slot, 0] - context.planned_current_xy[batch, slot, 0]
                ) / 0.04
                later_v = (
                    plan[batch, end, slot, 0] - plan[batch, end - 1, slot, 0]
                ) / 0.04
                plan_acceleration = float((later_v - initial_v) / (end * 0.04))
            self.events.append(
                {
                    "batch": batch,
                    "actor": slot,
                    "frame": context.response_index,
                    "reason": int(result.takeover_reason[batch, slot]),
                    "influencer": influencer,
                    "support": float(result.takeover_score[batch, slot]),
                    "interaction_score": float(
                        result.takeover_interaction_score[batch, slot]
                    ),
                    "following_growth_s": float(
                        result.takeover_following_score_s[batch, slot]
                    ),
                    "release_importance_per_s": float(
                        result.takeover_release_score_per_s[batch, slot]
                    ),
                    "predicted_interaction_s": float(
                        result.takeover_interaction_time_s[batch, slot]
                    ),
                    "current_leader": int(leader[batch, actor]),
                    "previous_leader": int(old_leader[batch, actor]),
                    "current_gap_m": (
                        float(gap[batch, actor]) if leader[batch, actor] >= 0 else None
                    ),
                    "previous_gap_m": (
                        float(old_gap[batch, actor])
                        if old_leader[batch, actor] >= 0
                        else None
                    ),
                    "speed_mps": float(speed[batch, actor]),
                    "previous_speed_mps": float(old_speed[batch, actor]),
                    "leader_speed_mps": float(leader_speed[batch, actor]),
                    "influencer_ax_mps2": (
                        float(acceleration[batch, influencer, 0])
                        if influencer >= 0
                        else None
                    ),
                    "plan_initial_acceleration_mps2": plan_acceleration,
                    "plan_maneuver_started": bool(self.plan_started[batch, slot]),
                    "plan_source_y": float(self.plan_source_y[batch, slot]),
                    "plan_target_y": float(self.plan_target_y[batch, slot]),
                    "plan_progress": float(self.plan_progress[batch, slot]),
                    "anticipated_entry": entries[batch, actor].tolist(),
                }
            )
        return result


def records_from_trace(
    trace, arrays, rows, *, seed, arm, controller=None, baseline=None
):
    records = trace_records(
        trace,
        arrays,
        rows,
        arm=arm,
        kind="null",
        seed=seed,
        doses=np.zeros(len(rows), int),
        starts=np.zeros(len(rows), int),
    )
    for k, record in enumerate(records):
        row = record["row"]
        record["valid_NPCs"] = int(np.asarray(arrays["agent_valid"][row, 24, 1:]).sum())
        record["recording_id"] = record["recording"]
        record["numerical_valid"] = bool(np.isfinite(trace["states"][k]).all())
        record["native_road_steps"] = int(trace["road_step_calls"][k])
        if controller is not None:
            record["events"] = [
                {name: value for name, value in event.items() if name != "batch"}
                for event in controller.events
                if event["batch"] == k
            ]
            for event in record["events"]:
                f, slot = event["frame"], event["actor"]
                event["executed_action"] = trace["actions"][k, f, slot].tolist()
                event["tracking_action"] = trace["base_actions"][k, f, slot].tolist()
                event["after_first_contact"] = bool(
                    record["first_collision_frame"] >= 0
                    and f > record["first_collision_frame"]
                )
        if baseline is not None:
            record["exact_tracker_states"] = bool(
                np.array_equal(trace["states"][k], baseline["states"][k])
            )
            record["exact_tracker_actions"] = bool(
                np.array_equal(trace["actions"][k], baseline["actions"][k])
            )
            if record["takeovers"] == 0:
                assert (
                    record["exact_tracker_states"] and record["exact_tracker_actions"]
                ), row
        assert record["native_road_steps"] == 149
        assert record["numerical_valid"]
        assert record["repeated_takeovers"] == 0 and record["ownership_releases"] == 0
    return records


def make_controller(config, draws, population, fit):
    return AuditedTakeover(
        sample_theta(population, draws.driver_latent),
        takeover_config=TakeoverConfig(**config["takeover"]),
        maneuver_latent=draws.maneuver_latent,
        **calibrated_mobil_parameters(fit["mobil"]),
    )


class TimedTakeoverController(TakeoverController):
    def __init__(self, *args, force_count=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.force_count = force_count
        self.admission_ms = []
        self.commitment_ms = 0.0

    def _update_plan_maneuver(self, context):
        start = time.perf_counter()
        super()._update_plan_maneuver(context)
        self.commitment_ms = (time.perf_counter() - start) * 1000

    def _admission(self, context):
        start = time.perf_counter()
        result = super()._admission(context)
        if context.response_index == 0 and self.force_count:
            result[0][:, : self.force_count] |= context.current_valid[
                :, 1 : 1 + self.force_count
            ]
        self.admission_ms.append(
            (time.perf_counter() - start) * 1000 + self.commitment_ms
        )
        return result


def latency_report(config):
    torch.set_num_threads(1)
    threadpool_limits(1)
    data = prepare_experiment_data(config, ROOT)
    rows, plans = load_plans(data, "val")
    population, fit = load_shared_components()
    recording = np.array(
        [int(str(i).split("_")[1]) for i in data.bundle.arrays["sequence_id"][rows]]
    )
    selected = []
    for rec in np.unique(recording):
        choices = np.flatnonzero(recording == rec)
        selected.extend(choices[np.linspace(0, len(choices) - 1, 2).astype(int)])
    counts = np.asarray(data.bundle.arrays["agent_valid"][rows, 24]).sum(1)
    selected = sorted(set(selected + [int(counts.argmax())]))
    report = {}
    for forced in (0, 3, 6):
        durations = []
        bridge = []
        for index in selected:
            row = rows[index]
            log = np.asarray(data.bundle.arrays["agent_states"][row : row + 1])
            valid = np.asarray(data.bundle.arrays["agent_valid"][row : row + 1])
            draws = scene_randomness(
                data.bundle.arrays["sequence_id"][row : row + 1], seed=SEEDS[0]
            )
            controller = TimedTakeoverController(
                sample_theta(population, draws.driver_latent),
                force_count=forced,
                takeover_config=TakeoverConfig(**config["takeover"]),
                maneuver_latent=draws.maneuver_latent,
                **calibrated_mobil_parameters(fit["mobil"]),
            )
            world = HighwayEnvClosedLoopWorld(
                controller=controller,
                tracking_config=TrackingConfig(**config["tracking"]),
            )
            world.reset(
                log[:, 24],
                valid[:, 24],
                plans[index : index + 1],
                data.bundle.arrays["map_polylines"][row : row + 1],
                data.bundle.arrays["map_polyline_valid"][row : row + 1],
                vehicle_dimensions=data.bundle.arrays["vehicle_dimensions"][
                    row : row + 1
                ],
                exogenous_state=draws,
                initial_history=log[:, :25],
                initial_history_valid=valid[:, :25],
            )
            controls = ego_controls(log[:, 24:173, 0], log[:, 25:174, 0], DT)
            for frame in range(world.reference_index, 149):
                before = len(controller.admission_ms)
                start = time.perf_counter()
                world.advance_response(torch.tensor(controls[:, frame]))
                durations.append((time.perf_counter() - start) * 1000)
                bridge.append(
                    controller.admission_ms[-1]
                    if len(controller.admission_ms) > before
                    else 0.0
                )
        report[str(forced)] = {
            "world_ms_quantiles": np.quantile(durations, [0.5, 0.95, 0.99, 1]).tolist(),
            "bridge_ms_quantiles": np.quantile(bridge, [0.5, 0.95, 0.99, 1]).tolist(),
            "steps": len(durations),
            "steps_over_40ms": int((np.asarray(durations) > 40).sum()),
        }
    p99 = max(r["world_ms_quantiles"][2] for r in report.values())
    save_json(
        {
            "cpu": report,
            "P99_ms": p99,
            "passed": p99 < 40,
            "ADS_compute_included": False,
            "physical_step_ms": 40,
            "maximum_active_entities": int(counts[selected].max()),
            "torch_threads": torch.get_num_threads(),
            "bridge_timing": "plan commitment update plus shared prediction and admission scoring",
        },
        OUTPUT / "latency.json",
    )
