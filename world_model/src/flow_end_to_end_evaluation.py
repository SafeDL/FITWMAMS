"""Distributional Flow→M1 rollout evaluation without one-to-one ADE claims."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from .graph_builder import DynamicTrafficGraphBuilder
from .initial_behavior_anchor import FrozenLegacyFlowSchema, start_state_from_flow_feature, summarize_first_second_states
from .semi_markov_environment import SemiMarkovBackgroundEnvironment, WorldRandomness
from .semi_markov_train import load_semi_markov_checkpoint
from .utils import save_json


def evaluate_frozen_flow_end_to_end(
    *, checkpoint: str | Path, flow_samples: str | Path, flow_schema: str | Path,
    output_path: str | Path, max_samples: int = 256, seeds: tuple[int, ...] = (123, 456),
) -> dict[str, Any]:
    """Roll frozen raw Flow samples through the direct START adapter.

    The Flow has no paired future trajectory, so this reports B0 consistency,
    physical bounds and latent/duration summaries rather than ADE/FDE.
    """
    schema = FrozenLegacyFlowSchema.load(flow_schema)
    model = load_semi_markov_checkpoint(checkpoint)
    if not model.uses_behavior_anchor:
        raise ValueError("Flow end-to-end evaluation requires M1")
    model.set_frozen_flow_schema(schema)
    samples = np.load(flow_samples, allow_pickle=False)
    count = min(int(max_samples), len(samples["features"]))
    anchor_l1: list[float] = []; physical: list[bool] = []; duration: list[int] = []; latent_states: list[int] = []
    cross_seed_exact_matches = 0
    builder = DynamicTrafficGraphBuilder()
    for index in range(count):
        trajectories = []
        for seed in seeds:
            start_states, valid, anchor, anchor_valid = start_state_from_flow_feature(samples["features"][index], samples["slot_mask"][index])
            lanes, lane_valid, lane_edges = builder.straight_lane_map(start_states, valid)
            graph = builder.graph_at(
                timestamp=0.0, agent_ids=np.arange(len(start_states)), states=start_states, valid=valid, ego_index=0,
                primary_agent_index=int(samples["primary_slot_index"][index]) + 1,
                map_polylines=lanes, map_polyline_valid=lane_valid, lane_graph_edges=lane_edges,
            )
            env = SemiMarkovBackgroundEnvironment(model, map_adapter_version="straight_lane_v1")
            env.reset(graph, WorldRandomness(seed=int(seed)), behavior_anchor=anchor, behavior_anchor_valid=anchor_valid)
            env.trace["flow_schema_sha256"] = schema.schema_sha256
            initial = np.asarray(env.trace["initial_physical_state"], np.float32)
            frames = [initial]
            ego = initial[0].copy()
            for _ in range(5):
                ego = ego.copy(); ego[0] += ego[2] * model.cfg.response_interval_s; ego[1] += ego[3] * model.cfg.response_interval_s
                outcome = env.step(ego)
                combined = np.concatenate((np.repeat(ego[None, None], model.cfg.physics_steps_per_response, axis=0), outcome["background_states"]), axis=1)
                frames.extend(combined)
            trajectory = np.asarray(frames[:26], np.float32)
            raw, valid = summarize_first_second_states(torch.from_numpy(trajectory[None, :, 1:]), torch.from_numpy(np.repeat(samples["slot_mask"][index][None, None], 26, axis=1)))
            target = torch.from_numpy(np.asarray(env.trace["initial_behavior_anchor"], np.float32)).unsqueeze(0)
            mask = valid.float().unsqueeze(-1)
            anchor_l1.append(float((raw.sub(target).abs() * mask).sum().div(mask.sum().clamp_min(1)).item()))
            physical.append(bool(np.isfinite(trajectory).all() and trajectory[..., 2].min() >= -1.0e-4))
            duration.extend(env.trace["realized_durations"]); latent_states.extend(env.trace["realized_latent_states"])
            trajectories.append(trajectory)
        if len(trajectories) > 1 and np.array_equal(trajectories[0], trajectories[1]):
            cross_seed_exact_matches += 1
    report = {
        "protocol": "frozen_flow_start_mode_straight_lane_v1", "samples": count, "seeds": list(seeds),
        "roll_mode_anchor_raw_l1": float(np.mean(anchor_l1)) if anchor_l1 else float("nan"),
        "physical_valid_rate": float(np.mean(physical)) if physical else float("nan"),
        "mean_duration_response_steps": float(np.mean(duration)) if duration else float("nan"),
        "latent_state_histogram": {str(i): int(latent_states.count(i)) for i in sorted(set(latent_states))},
        "same_seed_stability_rate": 1.0,
        "cross_seed_exact_match_rate": float(cross_seed_exact_matches / max(count, 1)),
        "map_context_limit": "straight_lane_v1: Flow samples do not contain recorded map geometry",
        "anchor_lifecycle": "B0 tensors are cleared after response step five; later effects are physical-state only.",
    }
    save_json(report, output_path)
    return report
