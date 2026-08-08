#!/usr/bin/env python3
"""Run B0 and hierarchical-innovation probes for one HiQR-v2 model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from world_model.src.core.cli_config import materialize_config  # noqa: E402
from world_model.src.core.initial_behavior_anchor import (
    FrozenLegacyFlowSchema,
)  # noqa: E402
from world_model.src.core.initial_behavior_anchor import (
    summarize_first_second_states,
)  # noqa: E402
from world_model.src.core.sequential_dataset import (
    sequence_cache_owner_dir,
)  # noqa: E402
from world_model.src.core.utils import (
    ensure_dir,
    save_json,
    select_device,
)  # noqa: E402
from world_model.src.hiqr_v2.data import (  # noqa: E402
    load_hiqr_v2_arrays,
    make_hiqr_v2_loader,
    to_hiqr_v2_batch,
)
from world_model.src.hiqr_v2.train import (  # noqa: E402
    load_hiqr_v2_checkpoint,
)


def _fde_1s(rollout: dict) -> torch.Tensor:
    prediction, target, valid = (
        rollout["predicted_states"][:, 24, 1:, :2],
        rollout["target_states"][:, 24, 1:, :2],
        rollout["target_valid"][:, 24, 1:],
    )
    error = torch.linalg.vector_norm(prediction - target, dim=-1)
    return (error * valid.float()).sum() / valid.float().sum().clamp_min(1.0)


def _original_b0_consistency(model, batch: dict, rollout: dict) -> torch.Tensor:
    """Compare generated START behavior to the unmodified Flow B0 target."""
    anchor = int(model.cfg.anchor_state_index)
    states = torch.cat(
        (
            batch["agent_states"][:, anchor : anchor + 1, 1:],
            rollout["predicted_states"][:, :25, 1:],
        ),
        dim=1,
    )
    valid = batch["agent_valid"][:, anchor : anchor + 26, 1:]
    summary, summary_valid = summarize_first_second_states(states, valid)
    summary_valid &= batch["behavior_anchor_valid"].bool()
    error = (summary - batch["behavior_anchor_raw"]).abs().mean(dim=-1)
    return (
        error * summary_valid.float()
    ).sum() / summary_valid.float().sum().clamp_min(1.0)


@torch.no_grad()
def _b0_probe(model, loader, device: torch.device) -> dict:
    generator = torch.Generator().manual_seed(91)
    rows = {name: {"fde": [], "summary": []} for name in ("correct", "zero", "shuffle")}
    for values in loader:
        batch = to_hiqr_v2_batch(values, loader.field_names, device)
        for name in rows:
            current = dict(batch)
            if name == "zero":
                current["behavior_anchor_raw"] = torch.zeros_like(
                    batch["behavior_anchor_raw"]
                )
            elif name == "shuffle":
                permutation = torch.randperm(len(values[0]), generator=generator)
                current["behavior_anchor_raw"] = batch["behavior_anchor_raw"][
                    permutation.to(device)
                ]
                current["behavior_anchor_valid"] = batch["behavior_anchor_valid"][
                    permutation.to(device)
                ]
            rollout = model.rollout_reconstruction(
                current, response_steps=5, deterministic=True
            )
            rows[name]["fde"].append(float(_fde_1s(rollout).cpu()))
            rows[name]["summary"].append(
                float(_original_b0_consistency(model, batch, rollout).cpu())
            )
    return {
        name: {
            "fde_1s_m": float(np.mean(item["fde"])),
            "original_b0_summary_loss": float(np.mean(item["summary"])),
        }
        for name, item in rows.items()
    }


def _pair_distance(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(
        torch.linalg.vector_norm(left[..., :2] - right[..., :2], dim=-1).mean().cpu()
    )


@torch.no_grad()
def _latent_probe(model, batch: dict, samples: int = 8) -> dict:
    current, valid = (
        batch["agent_states"][:, model.cfg.anchor_state_index],
        batch["agent_valid"][:, model.cfg.anchor_state_index],
    )
    ego = model._ego_mask(batch)
    state = model.initialize_start(
        current,
        valid,
        ego,
        batch["map_polylines"],
        batch["map_polyline_valid"],
        batch["behavior_anchor_raw"],
        batch["behavior_anchor_valid"],
    )
    shape_g = (len(current), model.cfg.scene_latent_dim)
    shape_z = (len(current), current.shape[1], model.cfg.agent_residual_dim)

    def trajectory(
        scene_noise: torch.Tensor, residual_noise: torch.Tensor
    ) -> torch.Tensor:
        return model.plan_step(
            None,
            None,
            current,
            valid,
            ego,
            batch["map_polylines"],
            batch["map_polyline_valid"],
            filter_state=state,
            deterministic=False,
            scene_standard_normal=scene_noise,
            agent_standard_normal=residual_noise,
        )["background_future_states"]

    fixed_scene, changed_scene = [], []
    for seed in range(samples):
        generator = torch.Generator(device=current.device).manual_seed(1_000 + seed)
        zeros_g = torch.zeros(shape_g, device=current.device)
        first_z, second_z = torch.randn(
            shape_z, generator=generator, device=current.device
        ), torch.randn(shape_z, generator=generator, device=current.device)
        fixed_scene.append(
            _pair_distance(trajectory(zeros_g, first_z), trajectory(zeros_g, second_z))
        )
        first_g, second_g = torch.randn(
            shape_g, generator=generator, device=current.device
        ), torch.randn(shape_g, generator=generator, device=current.device)
        changed_scene.append(
            _pair_distance(trajectory(first_g, first_z), trajectory(second_g, second_z))
        )
    return {
        "fixed_scene_residual_distance_m": float(np.mean(fixed_scene)),
        "changed_scene_joint_distance_m": float(np.mean(changed_scene)),
        "samples": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "world_model/scripts/configs/highd_hiqr_v2_world_model.yaml",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-sequences", type=int, default=256)
    args = parser.parse_args()
    config, config_path = materialize_config(
        args.config,
        args.output_dir,
        config_name=args.config.name,
        resolve_path_keys=(
            "sequence_cache_dir",
            "flow_schema",
            "source_dataset_dir",
            "hiqr_sidecar_output_dir",
        ),
    )
    device = select_device(str(config["evaluation"].get("device", "auto")))
    model = load_hiqr_v2_checkpoint(args.checkpoint, device=device)
    schema = FrozenLegacyFlowSchema.load(config["paths"]["flow_schema"])
    arrays, _ = load_hiqr_v2_arrays(
        cache_owner=sequence_cache_owner_dir(config, config_dir=config_path.parent),
        hiqr_sidecar_output_dir=config["paths"]["hiqr_sidecar_output_dir"],
        flow_schema=schema,
        source_dataset_dir=config["paths"]["source_dataset_dir"],
    )
    loader = make_hiqr_v2_loader(
        arrays,
        "val",
        batch_size=int(config["evaluation"].get("batch_size", 48)),
        maximum=args.max_sequences,
        shuffle=False,
        seed=int(config["evaluation"].get("seed", 42)),
        num_workers=0,
    )
    first = to_hiqr_v2_batch(next(iter(loader)), loader.field_names, device)
    report = {
        "checkpoint": str(args.checkpoint),
        "sequences": int(len(loader.dataset)),
        "b0": _b0_probe(model, loader, device),
        "latent": _latent_probe(model, first),
    }
    output = ensure_dir(args.output_dir)
    save_json(report, output / "architecture_probes.json")
    print(report)


if __name__ == "__main__":
    main()
