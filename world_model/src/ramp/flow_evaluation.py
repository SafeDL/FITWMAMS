"""Formal 8×4 EVT Flow × RAMP-WM composition test."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from world_model.src.core.flow_composition import (
    FLOW_COMPOSITION_SEED,
    HISTORY_FRAMES,
    INNER_WORLD_SAMPLES,
    ROLLOUT_FRAMES,
    decode_flow_starts,
    load_flow_tail_starts,
    repeat_batch,
    tensor,
    translated_ego_replay,
    write_flow_composition_report,
)
from world_model.src.core.utils import select_device

from .train import load_ramp_checkpoint


def _batch(starts: dict[str, np.ndarray], donors: np.ndarray, cache: dict[str, np.ndarray], device) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    count = len(donors)
    initial, present, anchor, anchor_valid = decode_flow_starts(starts)
    states = np.repeat(initial[:, None], HISTORY_FRAMES + ROLLOUT_FRAMES, axis=1)
    valid = np.repeat(present[:, None], HISTORY_FRAMES + ROLLOUT_FRAMES, axis=1)
    ego = np.stack([translated_ego_replay(cache["agent_states"][row], initial[index, 0]) for index, row in enumerate(donors)])
    states[:, HISTORY_FRAMES:, 0] = ego
    valid[:, :, 0] = True
    return {
        "agent_states": tensor(states, device), "agent_valid": tensor(valid, device),
        "ego_index": tensor(np.zeros(count, np.int64), device),
        "map_polylines": tensor(cache["map_polylines"][donors], device),
        "map_polyline_valid": tensor(cache["map_polyline_valid"][donors], device),
        "lane_graph_edges": tensor(cache["lane_graph_edges"][donors], device),
        "actions_highd": tensor(np.zeros((count, ROLLOUT_FRAMES, 6, 2), np.float32), device),
        "behavior_anchor_raw": tensor(anchor, device), "behavior_anchor_valid": tensor(anchor_valid, device),
    }, ego, present[:, 1:]


def evaluate_flow_composition(*, checkpoint: Path, output_dir: Path) -> dict[str, Any]:
    """Write the single canonical stochastic Flow × RAMP result JSON."""
    import torch

    device = select_device("auto")
    starts, cache, donors = load_flow_tail_starts(Path(__file__).resolve().parents[3], device=device)
    model = load_ramp_checkpoint(checkpoint, device=device)
    outputs: list[np.ndarray] = []
    egos: list[np.ndarray] = []
    valids: list[np.ndarray] = []
    reference, reference_ego, reference_valid = [], [], []
    batch_size = 16
    for start in range(0, len(donors), batch_size):
        stop = min(start + batch_size, len(donors))
        part = {name: value[start:stop] for name, value in starts.items()}
        batch, ego, valid = _batch(part, donors[start:stop], cache, device)
        with torch.no_grad():
            rollout = model.rollout_roll_mode(repeat_batch(batch, INNER_WORLD_SAMPLES), seed=FLOW_COMPOSITION_SEED + start, deterministic=False)
        outputs.append(rollout["predicted_states"][:, :, 1:].cpu().numpy())
        egos.append(np.repeat(ego, INNER_WORLD_SAMPLES, axis=0))
        valids.append(np.repeat(valid, INNER_WORLD_SAMPLES, axis=0)[:, None].repeat(ROLLOUT_FRAMES, axis=1))
        reference.append(cache["agent_states"][donors[start:stop], HISTORY_FRAMES:, 1:])
        reference_ego.append(cache["agent_states"][donors[start:stop], HISTORY_FRAMES:, 0])
        reference_valid.append(cache["agent_valid"][donors[start:stop], HISTORY_FRAMES:, 1:])
        print(f"Flow × RAMP starts {stop}/{len(donors)}", flush=True)
    generated, ego, valid = map(np.concatenate, (outputs, egos, valids))
    target, target_ego, target_valid = map(np.concatenate, (reference, reference_ego, reference_valid))
    return write_flow_composition_report(
        checkpoint=checkpoint,
        output_dir=output_dir,
        protocol={
            "name": "held-out EVT Flow × RAMP composition", "outer_flow_samples": 8,
            "inner_world_samples": 4, "horizon_seconds": 5.0,
            "supported_held_out_replays": int(len(np.unique(donors))),
            "flow_initial_conditions": int(len(donors)),
            "generated_world_futures": int(len(donors) * INNER_WORLD_SAMPLES),
            "not_a_paired_reconstruction": True, "seed": FLOW_COMPOSITION_SEED,
        },
        generated=generated,
        ego=ego,
        valid=valid,
        target=target,
        target_ego=target_ego,
        target_valid=target_valid,
    )
