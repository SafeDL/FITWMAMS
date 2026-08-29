#!/usr/bin/env python3
"""Reproducible method/protocol audit for TrafficBotsV1.5-HighD."""
from __future__ import annotations

import argparse
import ast
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch

from world_model.src.core.utils import file_sha256, save_json
from world_model.trafficbots.config import load_config
from world_model.trafficbots.data import TrafficBotsHighDDataset, make_loader
from world_model.trafficbots.evaluation import load_checkpoint
from world_model.trafficbots.rollout import TrafficBotsHighDRollout, logged_ego_controls


ROOT = Path(__file__).resolve().parents[3]


def _ordered_hash(values: np.ndarray) -> str:
    return hashlib.sha256("\n".join(np.asarray(values).astype(str)).encode()).hexdigest()


def _forbidden_runtime_imports() -> list[str]:
    roots = [
        ROOT / "world_model/trafficbots/data.py",
        ROOT / "world_model/trafficbots/module.py",
        ROOT / "world_model/trafficbots/rollout.py",
        ROOT / "world_model/trafficbots/evaluation.py",
    ]
    forbidden = ("diffusion", "normalizing_flow")
    found: list[str] = []
    for path in roots:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [item.name for item in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                if name.startswith(forbidden):
                    found.append(f"{path.relative_to(ROOT)}:{name}")
    return found


def _clone_batch(batch: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value.clone() if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


@torch.no_grad()
def audit(config: dict[str, Any], checkpoint: Path) -> dict[str, Any]:
    seed = int(config["evaluation"]["seed"])
    dataset = TrafficBotsHighDDataset(
        config["paths"]["sequence_cache_dir"], "test", seed=seed
    )
    batch = next(iter(make_loader(dataset, batch_size=2, shuffle=False, seed=seed)))
    module = load_checkpoint(config, checkpoint)
    if torch.cuda.is_available():
        module = module.cuda()
    runner = TrafficBotsHighDRollout(module)
    moved = module._move(batch, next(module.parameters()).device)
    mp_tokens, tl_tokens = module._tokens(moved)
    prior = module._latent(moved, mp_tokens, tl_tokens, posterior=False)
    posterior = module._latent(moved, mp_tokens, tl_tokens, posterior=True)
    destination = module._dest_distribution(moved, mp_tokens)

    changed = _clone_batch(batch)
    for key in (
        "agent/pos", "agent/vel", "agent/spd", "agent/acc",
        "agent/yaw_bbox", "agent/yaw_rate",
    ):
        changed[key][:, :, 1:] += 17.0
    changed["agent/dest"] = (changed["agent/dest"] + 1) % 16
    changed["canonical/states"][:, 1:] += 17.0
    changed_moved = module._move(changed, next(module.parameters()).device)
    changed_mp, changed_tl = module._tokens(changed_moved)
    changed_prior = module._latent(
        changed_moved, changed_mp, changed_tl, posterior=False
    )
    changed_posterior = module._latent(
        changed_moved, changed_mp, changed_tl, posterior=True
    )
    changed_destination = module._dest_distribution(changed_moved, changed_mp)

    controls = logged_ego_controls(
        moved["canonical/states"], moved["canonical/valid"]
    )
    factual = runner.run(moved, deterministic=True, ego_controls=controls)
    replay = runner.run(
        changed_moved,
        deterministic=False,
        ego_controls=controls,
        latent_sample=factual.latent_sample,
        destination_sample=factual.destination_sample,
    )
    repeated = runner.run(moved, deterministic=True, ego_controls=controls)
    destination_delta = (
        destination.distribution.logits
        - changed_destination.distribution.logits
    )
    destination_finite = torch.isfinite(destination_delta)

    arrays = dataset.arrays
    rows = dataset.rows
    states = np.asarray(arrays["agent_states"])[rows, 24]
    valid = np.asarray(arrays["agent_valid"])[rows, 24]
    maps = np.asarray(arrays["map_polylines"])[rows]
    map_valid = np.asarray(arrays["map_polyline_valid"])[rows]
    map_dx = maps[..., 1:, 0] - maps[..., :-1, 0]
    map_edge_valid = map_valid[..., 1:] & map_valid[..., :-1]
    checks = {
        "cache_only_runtime_imports": not _forbidden_runtime_imports(),
        "full_test_split": len(dataset) == 10151,
        "policy_history_steps_11": config["model"]["policy_history_steps"] == 11,
        "posterior_indices_13_with_endpoints": (
            len(config["model"]["posterior_temporal_indices"]) == 13
            and config["model"]["posterior_temporal_indices"][0] == 0
            and config["model"]["posterior_temporal_indices"][-1] == 149
        ),
        "released_detach_enabled": bool(
            config["training"]["training_detach_model_input"]
        ),
        "no_warm_start_or_spawn": (
            config["training"]["step_warm_start"] == 0
            and config["training"]["step_spawn_agent"] == 0
        ),
        "collision_filtering_disabled": not config["evaluation"]["collision_filtering"],
        "prior_ignores_future_scene": torch.equal(
            prior.distribution.mean, changed_prior.distribution.mean
        ),
        "destination_ignores_future_scene": torch.equal(
            destination.distribution.logits,
            changed_destination.distribution.logits,
        ),
        "posterior_uses_future_scene": not torch.equal(
            posterior.distribution.mean, changed_posterior.distribution.mean
        ),
        "explicit_z_g_replay_exact": torch.equal(factual.states, replay.states),
        "deterministic_prior_mode_exact": torch.equal(
            factual.states, repeated.states
        ),
        "s0_validity_is_prior_validity": torch.equal(
            prior.valid, moved["agent/valid"][..., 0]
        ),
    }
    crn_path = Path(config["paths"]["output_dir"]) / "intervention_crn.npz"
    crn = np.load(crn_path, allow_pickle=False)
    checks["saved_crn_contains_actual_z_g"] = (
        crn["latent"].shape[:2] == (512, 8)
        and crn["destination"].shape == (512, 8)
    )
    active_per_scene = valid[:, 1:].sum(1)
    report = {
        "audit_schema_version": 1,
        "method_identity_passed": all(checks.values()),
        "strict_bit_exact_upstream_reproduction": False,
        "checks": checks,
        "known_adaptations": [
            "highD cache/schema, 25 Hz dt, padding/KNN sizes and lane destination surrogate",
            "S0 free rollout with external ego controls and no WOSAC collision filtering",
            "position SmoothL1 averages x/y; released wrapper sums x/y (effective 0.05 vs 0.1 per-coordinate position weight)",
            "prior/posterior rollout selection is per scene; released code couples the minibatch draw",
        ],
        "data_contract": {
            "test_sequences": len(rows),
            "ordered_test_sequence_ids_sha256": _ordered_hash(
                np.asarray(arrays["sequence_id"])[rows]
            ),
            "s0_background_agents_mean": float(active_per_scene.mean()),
            "s0_empty_background_sequences": int((active_per_scene == 0).sum()),
            "future_spawn_or_exit_sequences": int(
                np.any(
                    np.asarray(arrays["agent_valid"])[rows, 24:174]
                    != np.asarray(arrays["agent_valid"])[rows, 24:25],
                    axis=(1, 2),
                ).sum()
            ),
            "negative_s0_vx_valid_agents": int(((states[..., 2] < 0) & valid).sum()),
            "negative_valid_map_edges": int(((map_dx < 0) & map_edge_valid).sum()),
        },
        "future_boundary_max_abs": {
            "prior_mean": float(
                (prior.distribution.mean - changed_prior.distribution.mean).abs().max()
            ),
            "destination_logits": float(
                destination_delta[destination_finite].abs().max()
            ),
            "posterior_mean": float(
                (posterior.distribution.mean - changed_posterior.distribution.mean).abs().max()
            ),
            "explicit_z_g_rollout_states": float(
                (factual.states - replay.states).abs().max()
            ),
        },
        "provenance": {
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": file_sha256(checkpoint),
            "evaluation_sha256": file_sha256(
                Path(config["paths"]["output_dir"]) / "evaluation.json"
            ),
            "upstream_snapshot_commit": "9a379084adbefe9df005c4eae69e7a56c360a396",
            "forbidden_runtime_imports": _forbidden_runtime_imports(),
        },
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=ROOT / "world_model/trafficbots/config/highd.yaml",
    )
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    checkpoint = args.checkpoint or (
        Path(config["paths"]["output_dir"]) / "checkpoints/best.ckpt"
    )
    result = audit(config, checkpoint)
    output = Path(config["paths"]["output_dir"]) / "audit.json"
    save_json(result, output)
    if not result["method_identity_passed"]:
        raise RuntimeError(f"TrafficBots audit failed: {result['checks']}")


if __name__ == "__main__":
    main()
