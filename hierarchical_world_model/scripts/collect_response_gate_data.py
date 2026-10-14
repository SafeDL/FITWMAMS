#!/usr/bin/env python3
"""Collect causal rule-teacher samples from real highD event rollouts."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from ads_learning.events import HighDEventStore
from ads_learning.runtime import FrozenWorldFactory
from ads_learning.verify_protocol import verify_protocol_lock
from hierarchical_world_model.src.response_gate import (
    LONGITUDINAL_RESPONSE_THRESHOLD_MPS2,
    MANEUVER_RESPONSE_THRESHOLD_RPS,
)


def select_event_indices(frame, *, max_events: int, seed: int) -> list[int]:
    """Select a deterministic, interaction-distinct, recording-balanced subset."""
    if max_events <= 0:
        raise ValueError("max-events must be positive")
    candidates = frame.reset_index().rename(columns={"index": "_event_index"}).copy()
    stable_order = candidates.sort_values("event_id", kind="stable").index.to_numpy()
    priority = np.empty(len(candidates), dtype=np.int64)
    priority[stable_order] = np.random.default_rng(seed).permutation(len(candidates))
    candidates["_score"] = priority
    if "interaction_group" in candidates:
        candidates = (
            candidates.sort_values(["_score", "event_id"], kind="stable")
            .drop_duplicates("interaction_group", keep="first")
        )
    recording_key = (
        "recording_id" if "recording_id" in candidates else "source_event_id"
    )
    candidates = candidates.sort_values(
        [recording_key, "_score", "event_id"], kind="stable"
    )
    candidates["_recording_rank"] = candidates.groupby(
        recording_key, sort=True
    ).cumcount()
    selected = candidates.sort_values(
        ["_recording_rank", "_score", "event_id"], kind="stable"
    ).head(min(max_events, len(candidates)))
    return [int(value) for value in selected["_event_index"]]


def _hold_action(state: np.ndarray) -> np.ndarray:
    speed = max(float(np.linalg.norm(state[2:4])), 0.5)
    heading = float(np.arctan2(state[3], max(abs(state[2]), 1.0e-4)))
    acceleration = float(state[4] * np.cos(heading) + state[5] * np.sin(heading))
    yaw = float((-state[4] * np.sin(heading) + state[5] * np.cos(heading)) / speed)
    return np.asarray((np.clip(acceleration, -8.0, 4.0), np.clip(yaw, -0.6, 0.6)), np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--sequence-cache", type=Path, required=True)
    parser.add_argument("--world-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-events", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--protocol-lock", type=Path)
    args = parser.parse_args()
    protocol = (
        None
        if args.protocol_lock is None
        else verify_protocol_lock(args.protocol_lock, Path(__file__).resolve().parents[2])
    )
    if protocol is not None and not protocol["verified"]:
        raise RuntimeError(f"protocol lock mismatch: {protocol['mismatches'][:3]}")
    factory = FrozenWorldFactory(args.world_checkpoint, device=args.device)
    events = HighDEventStore(
        args.events, args.sequence_cache, world_seed=args.seed,
        scene_dim=factory.model.cfg.scene_latent_dim,
        agent_dim=factory.model.cfg.agent_latent_dim,
    )
    observed_actions = np.load(
        args.sequence_cache / "actions_highd.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    if observed_actions.shape[:2] != (len(events.sequence_ids), 149):
        raise ValueError("actions_highd must have shape [sequences,149,6,2]")
    features, base_actions, rule_actions, teacher_actions, target_authorities, sources = [], [], [], [], [], []
    event_ids, source_event_ids, frame_indices, actor_indices = [], [], [], []
    recording_ids, interaction_groups = [], []
    selected_indices = select_event_indices(
        events.frame, max_events=args.max_events, seed=args.seed
    )
    response_scores: list[tuple[int, float, int]] = []
    for selection_rank, event_index in enumerate(selected_indices):
        candidate = events[event_index]
        ego = candidate.initial_states[0]
        ego_lane = candidate.geometry.nearest_lane(ego[:2])
        rear_gaps = []
        for slot in np.flatnonzero(candidate.valid[1:]):
            npc = candidate.initial_states[slot + 1]
            if (
                candidate.geometry.nearest_lane(npc[:2]) == ego_lane
                and -50.0 < float(npc[0] - ego[0]) < 5.0
            ):
                rear_gaps.append(abs(float(npc[0] - ego[0])))
        # Prefer controlled perturbations with a nearby realised follower;
        # deterministic rank breaks ties while preserving half factual data.
        score = (
            100.0 + len(rear_gaps) * 10.0 - min(rear_gaps)
            if rear_gaps
            else -float(selection_rank)
        )
        response_scores.append((event_index, score, selection_rank))
    controlled_count = len(selected_indices) // 2
    controlled_indices = {
        event_index
        for event_index, _, _ in sorted(
            response_scores, key=lambda value: (-value[1], value[2])
        )[:controlled_count]
    }
    for selection_rank, event_index in enumerate(selected_indices):
        event = events[event_index]
        event_row = events.frame.iloc[event_index]
        base_event_id = str(event_row.get("base_event_id", event.event_id))
        cache_row = events.row_by_id[base_event_id]
        active_actors = np.flatnonzero(event.valid[1:])
        if not len(active_actors):
            continue
        world = factory(event)
        world.reset(
            torch.as_tensor(event.initial_states)[None],
            torch.as_tensor(event.valid)[None],
            torch.as_tensor(event.soft_reference)[None],
            torch.as_tensor(event.map_polylines)[None],
            torch.as_tensor(event.map_polyline_valid)[None],
            exogenous_state=event.exogenous_state,
            vehicle_dimensions=(
                None
                if event.vehicle_dimensions is None
                else torch.as_tensor(event.vehicle_dimensions)[None]
            ),
        )
        controlled = event_index in controlled_indices
        for frame in range(len(event.soft_reference)):
            action = _hold_action(world.observe()["agent_states"][0, 0].detach().cpu().numpy())
            source = "observation"
            if controlled and 25 <= frame < 50:
                action[0] = -4.0
                source = "teacher"
            elif controlled and 75 <= frame < 100:
                action[0] = 2.0
                action[1] = 0.12 if selection_rank % 4 == 1 else -0.12
                source = "mismatch"
            elif controlled and frame >= 25:
                source = "teacher"
            transition = world.advance_response(torch.as_tensor(action, device=world.device)[None])
            current_features = transition["controller_response_gate_features"]
            if current_features is None:
                raise RuntimeError("online controller did not expose response-gate features")
            features.append(current_features[0, active_actors].cpu().numpy())
            base_actions.append(
                transition["base_background_actions"][0, 0, active_actors]
                .cpu()
                .numpy()
            )
            rule_full = (
                transition["background_actions"][0, 0].cpu().numpy()
            )
            if source == "observation":
                target_actions = np.asarray(
                    observed_actions[cache_row, frame], np.float32
                )
                target_authority = np.ones(6, np.float32)
            else:
                target_actions = rule_full.copy()
                # Controlled teacher/mismatch samples deliberately cover
                # partial longitudinal authority. This supervises L2's
                # continuous head instead of circularly making every active
                # rule candidate a gate target of exactly one.
                designed_authority = (
                    0.5
                    if source == "mismatch"
                    else 0.25 + 0.75 * ((frame % 25) / 24.0)
                )
                target_authority = np.full(
                    6, designed_authority, np.float32
                )
                base_full = transition["base_background_actions"][0, 0].cpu().numpy()
                target_actions[:, 0] = base_full[:, 0] + target_authority * (
                    rule_full[:, 0] - base_full[:, 0]
                )
            rule_actions.append(
                rule_full[active_actors]
            )
            teacher_actions.append(target_actions[active_actors])
            target_authorities.append(target_authority[active_actors])
            active_count = len(active_actors)
            sources.extend((source,) * active_count)
            event_ids.extend((event.event_id,) * active_count)
            source_event_ids.extend((event.source_event_id,) * active_count)
            recording_ids.extend(
                (int(event_row.get("recording_id", -1)),) * active_count
            )
            interaction_groups.extend(
                (
                    str(
                        event_row.get(
                            "interaction_group", event.source_event_id
                        )
                    ),
                )
                * active_count
            )
            frame_indices.extend((frame,) * active_count)
            actor_indices.extend(active_actors.tolist())
    if not features:
        raise RuntimeError("selected events contain no valid NPC samples")
    feature_array = np.concatenate(features, axis=0).astype(np.float32)
    base_array = np.concatenate(base_actions, axis=0).astype(np.float32)
    rule_array = np.concatenate(rule_actions, axis=0).astype(np.float32)
    teacher_array = np.concatenate(teacher_actions, axis=0).astype(np.float32)
    authority_array = np.concatenate(target_authorities, axis=0).astype(np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        features=feature_array,
        base_actions=base_array,
        rule_actions=rule_array,
        teacher_actions=teacher_array,
        target_authority=authority_array,
        source=np.asarray(sources),
        event_id=np.asarray(event_ids),
        source_event_id=np.asarray(source_event_ids),
        recording_id=np.asarray(recording_ids, dtype=np.int16),
        interaction_group=np.asarray(interaction_groups),
        frame_index=np.asarray(frame_indices, dtype=np.int16),
        actor_index=np.asarray(actor_indices, dtype=np.int8),
        protocol_id=np.asarray(
            "unlocked" if protocol is None else protocol["protocol_id"]
        ),
        protocol_lock_sha256=np.asarray(
            "unlocked" if protocol is None else protocol["protocol_lock_sha256"]
        ),
    )
    delta = teacher_array - base_array
    modes = np.where(
        np.abs(delta[:, 1]) > MANEUVER_RESPONSE_THRESHOLD_RPS,
        2,
        np.where(
            np.abs(delta[:, 0])
            > LONGITUDINAL_RESPONSE_THRESHOLD_MPS2,
            1,
            0,
        ),
    )
    print({
        "samples": int(len(feature_array)),
        "track": int((modes == 0).sum()),
        "longitudinal": int((modes == 1).sum()),
        "maneuver": int((modes == 2).sum()),
        "events": int(len(selected_indices)),
        "recordings": int(len(set(recording_ids))),
        "interaction_groups": int(len(set(interaction_groups))),
    })


if __name__ == "__main__":
    main()
