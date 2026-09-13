#!/usr/bin/env python3
"""Train the CIH-WM human-response calibrator on its factual executor.

This script never imports or executes HighwayEnv.  It emits unselected checkpoints only; a complete
formal validation report is the sole route to candidate selection/promotion.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from hierarchical_world_model.src.data import prepare_experiment_data
from hierarchical_world_model.src.cih_training import (
    causal_contrast_cache,
    causal_contrast_loss,
    cih_result_root,
    load_cih_config,
    response_optimization_config,
    response_policy_buffer,
    response_policy_update,
    response_supervised_loss,
    response_teacher_cache,
    supported_event_indices,
)
from hierarchical_world_model.src.human_response_prior import HumanResponseQuerySet
from hierarchical_world_model.src.cih_model import build_response_policy
from hierarchical_world_model.src.reaction_controller import CausalInfluenceResponsePolicy
from hierarchical_world_model.src.reaction_evidence import ReactionEventReference, event_identity
from hierarchical_world_model.src.train import load_checkpoint
from hierarchical_world_model.src.planner import (
    complete_endogenous_response_plans,
    frozen_diffusion_plans,
)
from world_model.src.core.utils import file_sha256, select_device
from hierarchical_world_model.src.protocol import canonical_hash


def _arrays(bundle, rows: np.ndarray) -> dict[str, np.ndarray]:
    names = ("agent_states", "agent_valid", "map_polylines", "map_polyline_valid")
    result = {name: np.asarray(bundle.arrays[name])[rows] for name in names}
    result["row_index"] = np.asarray(rows, np.int64)
    return result


def _controller(config: dict, device: torch.device) -> CausalInfluenceResponsePolicy:
    return build_response_policy(config, root=ROOT, device=device)


def _event_key(reference: ReactionEventReference, index: int) -> str:
    event = reference.events
    return event_identity(
        event.recording_id[index], event.leader_id[index],
        event.follower_id[index], event.absolute_onset_frame[index],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--supervised-updates", type=int, default=None)
    parser.add_argument("--policy-updates", type=int, default=None)
    parser.add_argument("--event-groups-per-update", type=int, default=None)
    parser.add_argument("--futures-per-event", type=int, default=None)
    parser.add_argument("--event-limit", type=int, default=None, help="development-only cap; recorded in the manifest")
    parser.add_argument("--causal-contrast-rows", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--reuse-supervised", action="store_true")
    args = parser.parse_args()

    config, world = load_cih_config(ROOT)
    status = str(config["method"].get("protocol_status", ""))
    if status != "cih_world_model":
        raise RuntimeError("trainer requires the CIH-WM protocol")
    device = select_device(world["training"].get("device", "auto"))
    seed = int(config["policy_optimization"]["seed"])
    torch.manual_seed(seed); np.random.seed(seed)
    root = (ROOT / args.output_dir) if args.output_dir is not None else cih_result_root(ROOT, config) / "continuation"
    root.mkdir(parents=True, exist_ok=True)
    if (root / "response_policy.pt").exists():
        raise RuntimeError("training output is immutable; choose a new --output-dir")

    model, checkpoint = load_checkpoint(ROOT / world["paths"]["evaluation_checkpoint"], device=device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    experiment = prepare_experiment_data(world, ROOT)
    reference = ReactionEventReference.load(ROOT / config["paths"]["event_reference"] / "train")
    all_event_rows = np.unique(reference.events.row_index[reference.events.indices(reference.supported_cells)]).astype(np.int64)
    arrays = _arrays(experiment.bundle, all_event_rows)
    event_pool = supported_event_indices(reference, arrays)
    if args.event_limit is not None:
        if args.event_limit <= 0:
            raise ValueError("--event-limit must be positive")
        event_pool = event_pool[:args.event_limit]
    if not len(event_pool):
        raise RuntimeError("formal training has no replayable supported response events")
    # Restrict the expensive frozen-plan cache *after* fixing the replayable
    # event population.  This makes development caps real while the default
    # still covers every supported train event.
    event_rows = np.unique(reference.events.row_index[event_pool]).astype(np.int64)
    arrays = _arrays(experiment.bundle, event_rows)
    event_pool = supported_event_indices(reference, arrays)
    if args.event_limit is not None:
        event_pool = event_pool[:args.event_limit]
    plans = frozen_diffusion_plans(
        experiment.bundle, event_rows, checkpoint=world["paths"]["diffusion_checkpoint"],
        output_dir=root / "frozen_diffusion_plans", device=device,
        batch_size=int(world["training"]["validation_batch_size"]), ddim_steps=20,
        experiment_scope=str(world["training"].get("experiment_scope", "full")),
    )
    plans = complete_endogenous_response_plans(
        plans, arrays["agent_states"], arrays["agent_valid"]
    )
    controller = _controller(config, device)
    influence_config = {
        str(key): value for key, value in config["influence"].items()
    }
    optimization = config["policy_optimization"]
    runtime = response_optimization_config(optimization)
    supervised_updates = int(
        config["supervised"]["updates"]
        if args.supervised_updates is None else args.supervised_updates
    )
    policy_updates = int(
        optimization["maximum_updates"]
        if args.policy_updates is None else args.policy_updates
    )
    groups_per_update = int(
        optimization["event_groups_per_update"]
        if args.event_groups_per_update is None else args.event_groups_per_update
    )
    futures = int(
        optimization["futures_per_event"]
        if args.futures_per_event is None else args.futures_per_event
    )
    manifest = {
        "schema": "cih_world_model_training",
        "protocol": {
            "world_executor": "hierarchical_world_model.src.evaluation.rollout",
            "dynamics": "KinematicTrafficDynamics",
            "diffusion_plans": "frozen_diffusion_plans, same construction as released evaluation",
            "response_scope": "all slots retained for response supervision and ADS evaluation",
            "influence_routing": "direct and attenuated second-level edges from realized states",
            "selection": "none during training; full formal validation is mandatory",
            "highwayenv": "not imported or executed",
        },
        "world_checkpoint": str(ROOT / world["paths"]["evaluation_checkpoint"]),
        "world_checkpoint_sha256": file_sha256(ROOT / world["paths"]["evaluation_checkpoint"]),
        "response_prior_checkpoint_sha256": file_sha256(
            ROOT / config["paths"]["response_prior_checkpoint"]
        ),
        "world_epoch": int(checkpoint["epoch"]),
        "event_rows": int(len(event_rows)),
        "event_pool": int(len(event_pool)),
        "event_limit": args.event_limit,
        "seed": seed,
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "working_tree_diff_hash": hashlib.sha256(
            subprocess.check_output(["git", "diff", "--binary"], cwd=ROOT)
        ).hexdigest(),
        "resolved_config_hash": canonical_hash({"cih": config, "world": world}),
        "dataset_split_hash": hashlib.sha256(np.asarray(event_rows, np.int64).tobytes()).hexdigest(),
        "event_schema_hash": file_sha256(
            ROOT / config["paths"]["event_reference"] / "train" / "manifest.json"
        ),
        "reference_contract_hash": canonical_hash({
            "mode": "conditioned_reconstruction", "future_ego_action_is_model_input": False,
            "excluded_slots": [],
        }),
        "plan_cache_hash": file_sha256(root / "frozen_diffusion_plans" / "frozen_diffusion_test_plans.npz"),
        "action_map_hash": hashlib.sha256(
            inspect.getsource(CausalInfluenceResponsePolicy.map_calibration_tensors).encode()
        ).hexdigest(),
        "observation_timing_hash": canonical_hash({
            "decision": "observe H_t,x_t", "action": "applied [t,t+dt)",
            "next_state": "x_t_plus_1", "dt_s": 0.04,
        }),
        "randomness_spec_hash": file_sha256(ROOT / "hierarchical_world_model/src/randomness.py"),
        "training_schedule": {
            "supervised_updates": supervised_updates,
            "policy_updates": policy_updates,
            "event_groups_per_policy_update": groups_per_update,
            "futures_per_event": futures,
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    cache_path = root / "response_teacher_cache.pt"
    contrast_path = root / "causal_contrast_cache.pt"
    if args.reuse_supervised:
        if not cache_path.is_file() or not contrast_path.is_file() or not (root / "response_supervised.pt").is_file():
            raise RuntimeError("--reuse-supervised requires both teacher caches and response_supervised.pt")
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        contrast = torch.load(contrast_path, map_location="cpu", weights_only=False)
        payload = torch.load(root / "response_supervised.pt", map_location=device, weights_only=False)
        controller.adapter.load_state_dict(payload["calibrator_state_dict"], strict=True)
    else:
        cache = response_teacher_cache(
            model, arrays=arrays, plans=plans, controller=controller, reference=reference,
            event_indices=event_pool, device=device,
            influence_graph_config=influence_config,
        )
        torch.save(cache, cache_path)
        contrast_rows = int(
            config["auxiliary"].get("causal_contrast_rows", 256)
            if args.causal_contrast_rows is None else args.causal_contrast_rows
        )
        def build_contrast_cache():
            return causal_contrast_cache(
                model,
                arrays=arrays,
                plans=plans,
                controller=controller,
                device=device,
                maximum_rows=contrast_rows,
                batch_size=int(world["training"]["validation_batch_size"]),
                influence_graph_config=influence_config,
                rule_delta_threshold_mps2=float(
                    config["auxiliary"].get("rule_delta_threshold_mps2", 0.10)
                ),
            )

        contrast = build_contrast_cache()
        torch.save(contrast, contrast_path)
        manifest["causal_contrast_frames"] = int(len(contrast["nominal_features"]))
        manifest["causal_contrast_secondary_frames"] = int(contrast["secondary"].sum())
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        optimizer = torch.optim.Adam(controller.adapter.parameters(), lr=float(config["supervised"]["learning_rate"]))
        refresh_interval = int(config["auxiliary"].get("causal_refresh_interval", 0))
        for update in range(supervised_updates):
            if refresh_interval > 0 and update > 0 and update % refresh_interval == 0:
                # Refresh on the policy's newly realized state distribution;
                # a single fixed cache allows an off-policy shortcut that does
                # not survive closed-loop evaluation.
                contrast = build_contrast_cache()
                torch.save(contrast, contrast_path)
            indices = torch.randint(len(cache["features"]), (int(config["supervised"]["minibatch_size"]),))
            loss = response_supervised_loss(
                controller, cache, indices,
                prior_weight=float(config["supervised"]["prior_regularization"]),
                non_weight=float(config["supervised"]["non_event_calibration_weight"]),
                event_weight=float(config["supervised"].get("event_weight", 1.0)),
                non_event_weight=float(config["supervised"].get("non_event_weight", 1.0)),
                event_gate_weight=float(config["supervised"].get("event_gate_weight", 1.0)),
                mechanism_weight=float(config["auxiliary"].get("mechanism_weight", 1.0)),
                direction_weight=float(config["auxiliary"].get("direction_weight", 1.0)),
                magnitude_weight=float(config["auxiliary"].get("magnitude_weight", 0.1)),
                direction_margin_mps2=float(config["auxiliary"].get("direction_margin_mps2", 0.05)),
            )
            contrast_indices = torch.randint(
                len(contrast["nominal_features"]),
                (min(int(config["supervised"]["minibatch_size"]), len(contrast["nominal_features"])),),
            )
            contrast_objective, _ = causal_contrast_loss(
                controller,
                contrast,
                contrast_indices,
                direction_margin_mps2=float(config["auxiliary"].get("direction_margin_mps2", 0.05)),
                direction_weight=float(config["auxiliary"].get("causal_direction_weight", 1.0)),
                magnitude_weight=float(config["auxiliary"].get("causal_magnitude_weight", 0.1)),
                monotonic_weight=float(config["auxiliary"].get("causal_monotonic_weight", 1.0)),
            )
            loss = loss + float(config["auxiliary"].get("causal_contrast_weight", 1.0)) * contrast_objective
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        torch.save({"schema": "cih_world_model", "stage": "response_supervised", "calibrator_state_dict": controller.adapter.state_dict()}, root / "response_supervised.pt")

    optimizer = torch.optim.Adam(controller.adapter.parameters(), lr=float(optimization["learning_rate_start"]))
    reference_adapter = copy.deepcopy(controller.adapter).eval()
    query = HumanResponseQuerySet.load(
        ROOT / config["paths"]["human_response_evidence"] / "train"
    )
    lookup = {str(key): index for index, key in enumerate(query.event_key)}
    if futures < 3:
        raise ValueError("policy optimization requires at least three futures for leave-one-out Energy Score rewards")
    coefficient = float(optimization["reference_kl_coefficient"])
    history = []
    rng = np.random.default_rng(seed)
    for update in range(policy_updates):
        fraction = update / max(1, policy_updates - 1)
        optimizer.param_groups[0]["lr"] = float(optimization["learning_rate_start"]) + fraction * (float(optimization["learning_rate_end"]) - float(optimization["learning_rate_start"]))
        # A replayable highD event is not necessarily an event that opens the
        # causal controller in the formal state trajectory.  Do not silently
        # turn such a batch into a no-op PPO update: resample from the fixed
        # training population and fail loudly if none is executable.
        buffer = None
        for _ in range(16):
            groups = rng.choice(event_pool, size=groups_per_update, replace=len(event_pool) < groups_per_update)
            proposal = response_policy_buffer(
                model, arrays=arrays, plans=plans, controller=controller, reference=reference,
                groups=groups, futures=futures, query_response=query.response,
                query_iqr=query.response_iqr, query_support=query.support_label,
                query_lookup=lookup, event_key=_event_key,
                support_weights=config["human_response"]["support_weight"], device=device,
                influence_graph_config=influence_config,
            )
            if bool((proposal["mask"] & proposal["active"]).any()):
                buffer = proposal
                break
        if buffer is None:
            raise RuntimeError("formal PPO batch has no active supported response samples after 16 fixed-population draws")
        def auxiliary_loss() -> torch.Tensor:
            teacher_indices = torch.randint(
                len(cache["features"]),
                (min(int(config["supervised"]["minibatch_size"]), len(cache["features"])),),
            )
            teacher = response_supervised_loss(
                controller, cache, teacher_indices,
                prior_weight=float(config["supervised"]["prior_regularization"]),
                non_weight=float(config["supervised"]["non_event_calibration_weight"]),
                event_weight=float(config["supervised"].get("event_weight", 1.0)),
                non_event_weight=float(config["supervised"].get("non_event_weight", 1.0)),
            )
            contrast_indices = torch.randint(
                len(contrast["nominal_features"]),
                (min(256, len(contrast["nominal_features"])),),
            )
            mechanism, _ = causal_contrast_loss(
                controller, contrast, contrast_indices,
                direction_margin_mps2=float(config["auxiliary"].get("direction_margin_mps2", 0.05)),
                direction_weight=float(config["auxiliary"].get("causal_direction_weight", 1.0)),
                magnitude_weight=float(config["auxiliary"].get("causal_magnitude_weight", 0.1)),
                monotonic_weight=float(config["auxiliary"].get("causal_monotonic_weight", 1.0)),
            )
            return 0.1 * teacher + float(config["auxiliary"].get("causal_contrast_weight", 1.0)) * mechanism

        entry, coefficient = response_policy_update(
            controller, reference_adapter, optimizer, buffer, runtime,
            optimization, coefficient, auxiliary_loss=auxiliary_loss,
        )
        history.append({"update": update + 1, **entry})
    (root / "training_history.json").write_text(json.dumps(history, indent=2) + "\n")
    manifest["actual_optimizer_steps"] = {
        "supervised": supervised_updates,
        "policy_updates_attempted": len(history),
        "policy_updates_rolled_back": int(sum(bool(item.get("rolled_back")) for item in history)),
    }
    manifest["termination_reason"] = "fixed_budget_complete"
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    torch.save({"schema": "cih_world_model", "stage": "response_policy_unselected", "calibrator_state_dict": controller.adapter.state_dict()}, root / "response_policy.pt")
    manifest["calibrator_checkpoint_hash"] = file_sha256(root / "response_policy.pt")
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (root / "selection_status.json").write_text(json.dumps({
        "selected": False,
        "promotion_status": "requires_complete_validation_before_candidate_selection",
        "candidate": "response_policy.pt",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
