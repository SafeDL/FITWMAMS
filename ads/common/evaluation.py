"""Physical-step safety evidence and interaction descriptors, separate from search."""

import time
from dataclasses import dataclass

import numpy as np

from ads.common.environment import PolicyEnvironment
from ads.common.instances import json_hash
from ads.common.randomness import PolicyRandomContext
from ads.common.safety import initial_safety
from hierarchical_world_model.src.execution import trajectory_event_risk
from hierarchical_world_model.src.reaction_controller import ManeuverPhase
from process_highD.src.safety_envelope_risk import (
    SafetyEnvelopeRiskOptions,
    pairwise_safety_envelope_intrusion,
)


@dataclass
class EpisodeResult:
    instance: object
    record: dict
    states: np.ndarray
    controls: np.ndarray
    diagnostics: list[dict]

    @property
    def failure(self):
        return bool(self.record["safety_failure"])

    @property
    def score(self):
        return float(self.record["event_risk"])

    @property
    def level_score(self):
        return 1.0 if self.failure else self.score / (1.0 + self.score)

    @property
    def domain(self):
        return self.record.get("branch_id", "root")


def summarise_trace(
    instance, states, controls, diagnostics, info, *, evt_model, reward, elapsed_s
):
    event = instance.event
    states = np.asarray(states, np.float32)
    if not np.isfinite(states).all():
        raise FloatingPointError(f"non-finite world rollout: {instance.instance_hash}")
    risk = float(
        trajectory_event_risk(
            states[None],
            event.valid[None],
            vehicle_dimensions=event.vehicle_dimensions[None],
            excluded_slots=(),
        )[0]
    )
    score = float(evt_model.score(risk))
    frames = len(states)
    pair_scores = np.full((frames, 6), -np.inf)
    clearance = np.full((frames, 6), np.inf)
    minimum_ttc = 1_000.0
    for slot in np.flatnonzero(event.valid[1:]) + 1:
        ego, other = states[:, 0], states[:, slot]
        pair, _ = pairwise_safety_envelope_intrusion(
            **{
                "ego_" + name: ego[:, i]
                for i, name in enumerate(("x", "y", "vx", "vy", "ax", "ay"))
            },
            **{
                "other_" + name: other[:, i]
                for i, name in enumerate(("x", "y", "vx", "vy", "ax", "ay"))
            },
            ego_length=event.vehicle_dimensions[0, 0],
            ego_width=event.vehicle_dimensions[0, 1],
            other_length=event.vehicle_dimensions[slot, 0],
            other_width=event.vehicle_dimensions[slot, 1],
            valid=np.ones(frames, bool),
            options=SafetyEnvelopeRiskOptions(),
        )
        pair_scores[:, slot - 1] = pair
        dx, dy = (other[:, :2] - ego[:, :2]).T
        longitudinal = (
            np.abs(dx)
            - (event.vehicle_dimensions[0, 0] + event.vehicle_dimensions[slot, 0]) / 2
        )
        lateral = (
            np.abs(dy)
            - (event.vehicle_dimensions[0, 1] + event.vehicle_dimensions[slot, 1]) / 2
        )
        clearance[:, slot - 1] = np.hypot(
            np.maximum(longitudinal, 0), np.maximum(lateral, 0)
        ) + np.minimum(np.maximum(longitudinal, lateral), 0)
        gap = (
            dx
            - (event.vehicle_dimensions[0, 0] + event.vehicle_dimensions[slot, 0]) / 2
        )
        closing = ego[:, 2] - other[:, 2]
        mask = (gap > 0) & (lateral < 0) & (closing > 0.05)
        if mask.any():
            minimum_ttc = min(minimum_ttc, float(np.min(gap[mask] / closing[mask])))
    collision = bool(info.get("collision", False))
    offroad = bool(info.get("offroad", False))
    safety_failure = collision or offroad
    npc_collision = initial_safety(event)[2] or any(
        np.asarray(d.get("collision_pairs", np.zeros((7, 7), bool)))[1:, 1:].any()
        for d in diagnostics
    )
    critical_slot, critical_frame = -1, 0
    descriptor = [0.0] * 5
    stage = "tracking"
    if event.valid[1:].any():
        critical_frame, index = np.unravel_index(
            pair_scores.argmax(), pair_scores.shape
        )
        critical_slot = int(index + 1)
        before = max(0, critical_frame - 5)
        ego, other = states[before, 0], states[before, critical_slot]
        same_lane = event.geometry.nearest_lane(ego[:2]) == event.geometry.nearest_lane(
            other[:2]
        )
        role = ("front" if other[0] >= ego[0] else "rear") if same_lane else "adjacent"
        owned = False
        if diagnostics and before:
            owned = bool(
                np.asarray(
                    diagnostics[min(before - 1, len(diagnostics) - 1)].get(
                        "takeover_owned", np.zeros(6, bool)
                    )
                ).reshape(-1)[critical_slot - 1]
            )
        lateral = abs(float(other[3] - ego[3])) > 0.15
        stage = ("lateral" if lateral else "response") if owned else "tracking"
        domain = role + ("_owned" if owned else "_tracking")
        descriptor = [
            float((other[0] - ego[0]) / 80),
            float((other[1] - ego[1]) / 8),
            float((other[2] - ego[2]) / 20),
            float((other[3] - ego[3]) / 10),
            critical_frame / max(len(event.soft_reference), 1),
        ]
    else:
        domain = "free_road"
    if offroad:
        lane_centers = [
            event.geometry.center_y(int(i)) for i in event.geometry.lane_ids
        ]
        domain = (
            "offroad_left"
            if states[-1, 0, 1] > np.mean(lane_centers)
            else "offroad_right"
        )
    speeds = np.linalg.norm(states[:, 0, 2:4], axis=-1)
    dx = states[:, 1:, 0] - states[:, 0:1, 0]
    dy = np.abs(states[:, 1:, 1] - states[:, 0:1, 1])
    overlap_width = (
        event.vehicle_dimensions[1:, 1] + event.vehicle_dimensions[0, 1]
    ) / 2 + 0.5
    ahead = (dx > 0) & (dy < overlap_width) & event.valid[None, 1:]
    gap = dx - (event.vehicle_dimensions[1:, 0] + event.vehicle_dimensions[0, 0]) / 2
    front_gap = np.where(ahead, gap, np.inf).min(axis=1)
    sustained_low = frames >= 51 and np.mean(speeds[-50:] < 2) >= 0.9
    open_opportunity = float(np.mean(front_gap[-50:] > 30))
    avoidable_stop = bool(
        not safety_failure
        and sustained_low
        and open_opportunity >= 0.8
        and states[-1, 0, 0] - states[-51, 0, 0] < 2
    )
    record = {
        "event_id": instance.instance_hash,
        "instance_hash": instance.instance_hash,
        "root_sample_id": instance.root_sample_id,
        "parent_id": instance.parent_id,
        "purpose": instance.purpose,
        "assets_hash": json_hash(instance.assets),
        "safety_failure": safety_failure,
        "failure": safety_failure,
        "collision": collision,
        "offroad": offroad,
        "npc_collision": bool(npc_collision or info.get("npc_collision", False)),
        "stalled": bool(info.get("stalled", False)),
        "sustained_low_speed": bool(sustained_low),
        "open_front_fraction_last_2s": open_opportunity,
        "avoidable_stop": avoidable_stop,
        "numerical_valid": True,
        "progress_m": float(states[-1, 0, 0] - states[0, 0, 0]),
        "speed_mps": float(np.linalg.norm(states[-1, 0, 2:4])),
        "task_reward": float(reward),
        "evt_score": score,
        "event_risk": risk,
        "min_clearance_m": min(float(clearance.min()), 1_000.0),
        "min_ttc_s": minimum_ttc,
        "interaction_type": domain,
        "branch_id": "root",
        "stage": stage,
        "descriptor": descriptor,
        "critical_slot": critical_slot,
        "critical_frame": int(critical_frame),
        "physics_steps": frames - 1,
        "elapsed_s": elapsed_s,
        "initial_safety_failure": bool(any(initial_safety(event)[:2])),
        "source_kind": instance.source_kind,
        "source_metadata": instance.source_metadata,
        "recording_id": str(instance.source_metadata.get("recording_id", "")),
    }
    return EpisodeResult(
        instance,
        record,
        states,
        np.asarray(controls, np.float32).reshape(-1, 2),
        diagnostics,
    )


def annotate_dependencies(result, controller):
    """Trace random writes through the frozen residual recurrence and ownership.

    Reading a noise row does not imply that row changes persistent state. The
    residual's age gate is replayed at float32 precision, including writes made
    before takeover. This bookkeeping never changes the traffic controller.
    """
    omega = result.instance.event.exogenous_state
    masks = {
        name: np.zeros_like(value, bool) for name, value in omega.as_dict().items()
    }
    slot = int(result.record.get("critical_slot", -1))
    related = {slot}
    for diagnostic in result.diagnostics[: result.record["critical_frame"]]:
        for name in ("takeover_influencer", "controller_realized_leader_index"):
            values = np.asarray(diagnostic.get(name, [])).reshape(-1)
            for actor in tuple(related):
                if 1 <= actor <= len(values) and values[actor - 1] > 0:
                    related.add(int(values[actor - 1]))
    actors = np.array(
        [
            actor in related and result.instance.event.valid[actor]
            for actor in range(1, 7)
        ]
    )
    parameters = (controller.driver_correction or {}).get("longitudinal")
    residual = [set(), set(), set()]
    age = np.float32(0)
    consumed, writes, opportunities, adoptions = [], [], [], []
    missing_maneuver_frames, unknown_admissions = [], []
    onset = controller.mobil_onset_calibration is not None
    for tick, diagnostic in enumerate(
        result.diagnostics[: result.record["critical_frame"]]
    ):
        owned = np.asarray(diagnostic.get("takeover_owned", np.zeros(6)), bool).reshape(
            -1
        )
        if parameters is not None:
            current_noise = ("response", tick, 1)
            if tick == 0:
                residual = [
                    {current_noise},
                    {current_noise, ("calibration", 0, 0)},
                    {current_noise, ("calibration", 0, 0), ("calibration", 0, 1)},
                ]
                writes.append(tick)
            else:
                age = np.float32(age + controller.native_dt_s)
                if age >= parameters["autoregression"]["interval_seconds"] - 1e-7:
                    current, previous = residual[2], residual[0]
                    residual = [current, previous, current | previous | {current_noise}]
                    age = np.float32(
                        max(0, age - parameters["autoregression"]["interval_seconds"])
                    )
                    writes.append(tick)
            for actor in np.flatnonzero(actors & owned):
                for group, time_index, channel in residual[0] | residual[2]:
                    name = (
                        "policy_response_innovations"
                        if group == "response"
                        else "policy_calibration_innovations"
                    )
                    masks[name][0, time_index, actor, channel] = True
                masks["driver_latent"][0, actor] = True
                consumed.append([tick, int(actor), 1])
        if onset and owned.any():
            if not all(
                name in diagnostic
                for name in ("input_maneuver_phase", "input_maneuver_cooldown")
            ):
                missing_maneuver_frames.append(tick)
                continue
            phase = np.asarray(diagnostic["input_maneuver_phase"]).reshape(-1).copy()
            cooldown = np.maximum(
                np.asarray(diagnostic["input_maneuver_cooldown"]).reshape(-1) - 1, 0
            )
            # The frozen NPC policy decrements cooldown before testing onset.
            phase[(phase == ManeuverPhase.REBASE) & (cooldown == 0)] = (
                ManeuverPhase.TRACK
            )
            entered = np.asarray(diagnostic["takeover_started"], bool).reshape(-1)
            eligible = actors & owned
            if "input_maneuver_adopted" in diagnostic:
                adopted = np.asarray(
                    diagnostic["input_maneuver_adopted"], bool
                ).reshape(-1)
                phase[adopted] = ManeuverPhase.EXECUTE
                for actor in np.flatnonzero(eligible & adopted):
                    masks["maneuver_latent"][0, actor, 2:4] = True
                    adoptions.append([tick, int(actor)])
            else:
                unknown_admissions.extend(
                    [tick, int(actor)] for actor in np.flatnonzero(eligible & entered)
                )
                eligible &= ~entered
            for actor in np.flatnonzero(eligible):
                state = result.states[tick, actor + 1]
                lane = result.instance.event.geometry.nearest_lane(state[:2])
                center = result.instance.event.geometry.center_y(lane)
                geometry = result.instance.event.geometry
                width = geometry.lane_widths[geometry.lane_ids == lane][0]
                # Eligible draws count even if the sampled maneuver did not start.
                if (
                    phase[actor] in (ManeuverPhase.TRACK, ManeuverPhase.RESPOND)
                    and cooldown[actor] == 0
                    and abs(state[1] - center) < 0.2 * width
                    and abs(state[3]) < 0.3
                ):
                    masks["policy_response_innovations"][0, tick, actor, 0] = True
                    masks["maneuver_latent"][0, actor] = True
                    opportunities.append([tick, int(actor)])
    maneuver = [2, 3]
    if (
        controller.mobil_onset_calibration is None
        and controller.mobil_parameter_variation
    ):
        maneuver = [0, 1, 2, 3]
    elif onset and controller.lane_style is not None:
        maneuver = [0, 2, 3]
    masks["maneuver_latent"][..., [i for i in range(4) if i not in maneuver]] = False
    result.record.update(
        input_dependencies={
            name: np.flatnonzero(mask).tolist()
            for name, mask in masks.items()
            if name
            not in (
                "scenario_uniform",
                "c0_base_latent",
                "k_base_latent",
                "diffusion_noise",
            )
        },
        static_input_components={"maneuver_latent": maneuver},
        interaction_actors=sorted(a for a in related if a > 0),
        response_consumed_coordinates=consumed,
        response_write_ticks=writes,
        maneuver_opportunities=opportunities,
        maneuver_adoptions=adoptions,
        maneuver_unobserved_frames=missing_maneuver_frames,
        maneuver_unobserved_admissions=unknown_admissions,
    )


class WorldEvaluator:
    def __init__(
        self,
        model,
        algorithm,
        world_factory,
        evt_model,
        *,
        ledger=None,
        purpose="search",
        seed=20260926,
    ):
        self.model, self.algorithm = model, algorithm
        self.world_factory, self.evt_model = world_factory, evt_model
        self.ledger, self.purpose = ledger, purpose
        self.seed = seed

    def __call__(self, instance):
        started = time.perf_counter()
        context = PolicyRandomContext(
            self.algorithm,
            self.seed,
            self.purpose if self.purpose in ("development", "final_test") else "search",
            instance.instance_hash,
        )
        if hasattr(self.model, "reset"):
            self.model.reset(context)
        env = PolicyEnvironment(
            self.algorithm,
            [instance],
            self.world_factory,
            purpose=self.purpose,
            ledger=self.ledger,
            random_events=False,
        )
        obs, info = env.reset()
        collision, offroad, npc_collision = initial_safety(instance.event)
        info.update(
            collision=collision,
            offroad=offroad,
            npc_collision=npc_collision,
            stalled=False,
        )
        total = 0.0
        terminated = collision or offroad
        truncated = False
        decision = 0
        while not (terminated or truncated):
            with context.step(decision, getattr(self.model, "device", "cpu")):
                action, _ = self.model.predict(obs, deterministic=True)
            decision += 1
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
        result = summarise_trace(
            instance,
            env.physics_states,
            env.physics_controls,
            env.physics_diagnostics,
            info,
            evt_model=self.evt_model,
            reward=total,
            elapsed_s=time.perf_counter() - started,
        )
        result.record["policy_stream_id"] = context.identity
        result.record["policy_random_mode"] = context.mode
        controller = getattr(env.world, "controller", None)
        if controller is not None:
            annotate_dependencies(result, controller)
        return result
