"""Overlap diagnostics for completed ADS and NPC response rollouts.

The masks retain individual overlapping vehicle pairs and distinguish direct
ADS pursuit and lane-entry exposure for controller analysis.
"""

from __future__ import annotations

import numpy as np


def collision_events(
    states: np.ndarray,
    active_npcs: np.ndarray,
    *,
    intervention_start_frame: int = 25,
    ads_acceleration_setpoint_mps2: float | None = None,
) -> dict[str, np.ndarray]:
    """Return per-scene overlap masks, excluding only clear ADS pursuit pairs.

    A pair is labelled ADS pursuit when its first overlap follows the
    intervention, the accelerating ADS approaches a slower NPC from behind,
    and neither vehicle has shifted laterally by a lane-change amount. The
    pair is excluded from the adjusted test statistic; all other overlapping
    pairs in that scene remain counted.
    """
    states = np.asarray(states)
    active_npcs = np.asarray(active_npcs, bool)
    if states.ndim != 4 or states.shape[2:] != (7, 6):
        raise ValueError("states must have shape [scene,time,7,6]")
    if active_npcs.shape != (len(states), 6):
        raise ValueError("active_npcs must have shape [scene,6]")
    if not 1 <= intervention_start_frame < states.shape[1]:
        raise ValueError("intervention_start_frame must follow at least one state")

    valid = np.concatenate((np.ones((len(states), 1), bool), active_npcs), axis=1)
    dx = np.abs(states[:, :, :, None, 0] - states[:, :, None, :, 0])
    dy = np.abs(states[:, :, :, None, 1] - states[:, :, None, :, 1])
    upper = np.triu(np.ones((7, 7), bool), 1)
    overlap = (
        (dx < 4.8)
        & (dy < 1.8)
        & valid[:, None, :, None]
        & valid[:, None, None, :]
        & upper[None, None]
    )
    pair_overlap = overlap.any(axis=1)
    pursuit_pairs = np.zeros((len(states), 6), bool)
    ego_npc_overlap = overlap[:, :, 0, 1:]
    first_ego_npc_overlap = ego_npc_overlap.argmax(axis=1)
    row = np.arange(len(states))[:, None]
    npc = np.arange(1, 7)[None, :]
    start_ego_y = states[:, intervention_start_frame - 1, 0, 1]
    start_npc_y = states[:, intervention_start_frame - 1, 1:, 1]
    first_ego_y = states[row, first_ego_npc_overlap, 0, 1]
    first_npc_y = states[row, first_ego_npc_overlap, npc, 1]
    lateral_shift = first_ego_y - start_ego_y[:, None]
    # This is an exposure label, not a fault judgment or exclusion rule.
    # It identifies a pair whose first overlap occurred while the ADS moved
    # into an NPC's previously occupied adjacent lane.
    initial_lateral_offset = start_npc_y - start_ego_y[:, None]
    ads_cutin_pairs = (
        ego_npc_overlap.any(axis=1)
        & (first_ego_npc_overlap >= intervention_start_frame)
        & (np.abs(lateral_shift) > 0.8)
        & (initial_lateral_offset * lateral_shift > 0.0)
        & (np.abs(initial_lateral_offset) > 1.8)
        & (np.abs(initial_lateral_offset) < 5.4)
        & (np.abs(first_npc_y - start_npc_y) < 0.8)
    )
    if ads_acceleration_setpoint_mps2 is not None and ads_acceleration_setpoint_mps2 > 0:
        first = first_ego_npc_overlap
        previous = np.maximum(first - 1, 0)
        prior_ego = states[row, previous, 0]
        prior_npc = states[row, previous, npc]
        anchor_ego_y = states[:, intervention_start_frame - 1, 0, 1]
        anchor_npc_y = states[:, intervention_start_frame - 1, 1:, 1]
        pursuit_pairs = (
            ego_npc_overlap.any(axis=1)
            & (first >= intervention_start_frame)
            & (prior_ego[..., 0] < prior_npc[..., 0])
            & (prior_ego[..., 2] > prior_npc[..., 2] + 0.1)
            & (np.abs(prior_ego[..., 1] - anchor_ego_y[:, None]) < 0.8)
            & (np.abs(prior_npc[..., 1] - anchor_npc_y) < 0.8)
        )

    included_pairs = pair_overlap.copy()
    included_pairs[:, 0, 1:] &= ~pursuit_pairs
    return {
        "raw_pairs": pair_overlap,
        "adjusted_pairs": included_pairs,
        "raw": pair_overlap.any(axis=(1, 2)),
        "adjusted": included_pairs.any(axis=(1, 2)),
        "ads_pursuit": pursuit_pairs.any(axis=1),
        "ads_cutin_exposure": ads_cutin_pairs.any(axis=1),
        "ads_cutin_exposure_pairs": ads_cutin_pairs,
        "ego_npc": pair_overlap[:, 0, 1:].any(axis=1),
        "npc_npc": pair_overlap[:, 1:, 1:].any(axis=(1, 2)),
        "included_ego_npc": included_pairs[:, 0, 1:].any(axis=1),
    }


def paired_collision_summary(
    passive: dict[str, np.ndarray],
    responsive: dict[str, np.ndarray],
    *,
    logged_baseline: dict[str, np.ndarray] | None = None,
) -> dict[str, float | int]:
    """Summarize same-ADS passive/responsive branches using scene counts."""
    count = len(responsive["raw"])
    if len(passive["raw"]) != count:
        raise ValueError("paired collision cohorts differ in size")
    result: dict[str, float | int] = {"scenes": count}
    for label, events in (("passive", passive), ("responsive", responsive)):
        for key in ("raw", "adjusted", "ads_pursuit", "ads_cutin_exposure", "ego_npc", "npc_npc", "included_ego_npc"):
            result[f"{label}_{key}_scene_count"] = int(events[key].sum())
            result[f"{label}_{key}_scene_rate"] = float(events[key].mean())
    for key in ("raw", "adjusted"):
        result[f"{key}_avoided_with_npc_response"] = int(
            (passive[key] & ~responsive[key]).sum()
        )
        result[f"{key}_introduced_with_npc_response"] = int(
            (~passive[key] & responsive[key]).sum()
        )
        passive_pairs = passive[f"{key}_pairs"]
        responsive_pairs = responsive[f"{key}_pairs"]
        result[f"{key}_responsive_overlap_npc_response_added_pair_scene_count"] = int(
            (responsive_pairs & ~passive_pairs).any(axis=(1, 2)).sum()
        )
        if logged_baseline is not None:
            baseline_pairs = logged_baseline[f"{key}_pairs"]
            result[f"{key}_ads_intervention_added_pair_scene_count"] = int(
                (passive_pairs & ~baseline_pairs).any(axis=(1, 2)).sum()
            )
            result[f"{key}_responsive_overlap_preexisting_pair_scene_count"] = int(
                (responsive_pairs & baseline_pairs).any(axis=(1, 2)).sum()
            )
            result[f"{key}_responsive_overlap_ads_added_pair_scene_count"] = int(
                (responsive_pairs & passive_pairs & ~baseline_pairs).any(axis=(1, 2)).sum()
            )
    return result
