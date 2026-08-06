#!/usr/bin/env python3
"""Measure whether logged ego transitions are faithfully recovered as controls."""

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
from world_model.src.core.initial_behavior_anchor import FrozenLegacyFlowSchema  # noqa: E402
from world_model.src.core.sequential_dataset import sequence_cache_owner_dir  # noqa: E402
from world_model.src.core.utils import ensure_dir, save_json, select_device  # noqa: E402
from world_model.src.hiqr_v2.config import HiQRV2Config  # noqa: E402
from world_model.src.hiqr_v2.data import (  # noqa: E402
    load_hiqr_v2_arrays,
    make_hiqr_v2_loader,
    to_hiqr_v2_batch,
)
from world_model.src.hiqr_v2.model import HiQRV2WorldModel  # noqa: E402


@torch.no_grad()
def evaluate_logged_ego_controls(model, loader, device: torch.device) -> dict:
    """Replay only recovered ego controls and return per-sequence ADE/FDE."""
    ade_rows, fde_rows = [], []
    anchor, frames = int(model.cfg.anchor_state_index), int(model.cfg.rollout_frames)
    for values in loader:
        batch = to_hiqr_v2_batch(values, loader.field_names, device)
        states, valid = batch["agent_states"], batch["agent_valid"]
        source = states[:, anchor : anchor + frames, 0]
        target = states[:, anchor + 1 : anchor + 1 + frames, 0]
        control_valid = (
            valid[:, anchor : anchor + frames, 0]
            & valid[:, anchor + 1 : anchor + 1 + frames, 0]
        )
        controls = model._logged_ego_controls(source, target, control_valid)
        current = source[:, :1]
        predicted = []
        for frame in range(frames):
            current = model.dynamics.step(
                current,
                controls[:, frame : frame + 1],
                control_valid[:, frame : frame + 1],
                model.cfg.simulation_dt_s,
            )
            predicted.append(current[:, 0])
        generated = torch.stack(predicted, dim=1)
        distance = torch.linalg.vector_norm(generated[..., :2] - target[..., :2], dim=-1)
        weights = control_valid.float()
        ade_rows.append(
            ((distance * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0))
            .cpu()
            .numpy()
        )
        fde_rows.append(
            (distance[:, -1] * control_valid[:, -1].float()).cpu().numpy()
        )
    ade, fde = np.concatenate(ade_rows), np.concatenate(fde_rows)
    return {
        "sequences": int(len(ade)),
        "horizon_seconds": float(frames * model.cfg.simulation_dt_s),
        "ade_m": float(ade.mean()),
        "fde_m": float(fde.mean()),
        "p95_fde_m": float(np.quantile(fde, 0.95)),
        "per_sequence_ade_m": ade,
        "per_sequence_fde_m": fde,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "world_model/scripts/configs/highd_hiqr_v2_world_model.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/highd_world_model/hiqr_v2_world_model/readiness",
    )
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument("--max-fde-m", type=float, default=0.20)
    args = parser.parse_args()
    if args.max_sequences < 0 or args.max_fde_m <= 0.0:
        raise ValueError("max-sequences must be non-negative and max-fde-m positive")
    config, config_path = materialize_config(
        args.config,
        args.output_dir,
        config_name=args.config.name,
        resolve_path_keys=(
            "sequence_cache_dir",
            "flow_schema",
            "source_dataset_dir",
            "v1_sidecar_output_dir",
        ),
    )
    device = select_device(str(config["evaluation"].get("device", "auto")))
    schema = FrozenLegacyFlowSchema.load(config["paths"]["flow_schema"])
    arrays, _ = load_hiqr_v2_arrays(
        cache_owner=sequence_cache_owner_dir(config, config_dir=config_path.parent),
        v1_sidecar_output_dir=config["paths"]["v1_sidecar_output_dir"],
        flow_schema=schema,
        source_dataset_dir=config["paths"]["source_dataset_dir"],
    )
    loader = make_hiqr_v2_loader(
        arrays,
        args.split,
        batch_size=int(config["evaluation"].get("batch_size", 48)),
        maximum=args.max_sequences,
        shuffle=False,
        seed=int(config["evaluation"].get("seed", 42)),
        num_workers=int(config["evaluation"].get("num_workers", 0)),
    )
    result = evaluate_logged_ego_controls(
        HiQRV2WorldModel(HiQRV2Config(**config.get("model", {}))).to(device),
        loader,
        device,
    )
    output = ensure_dir(args.output_dir)
    np.savez_compressed(
        output / "ego_control_recovery_per_sequence.npz",
        ade_m=result.pop("per_sequence_ade_m"),
        fde_m=result.pop("per_sequence_fde_m"),
    )
    report = {
        **result,
        "split": args.split,
        "max_fde_m": float(args.max_fde_m),
        "acceptance": bool(result["fde_m"] < args.max_fde_m),
        "acceptance_rule": "5.96 s logged-ego FDE < max_fde_m",
    }
    save_json(report, output / "ego_control_recovery.json")
    print(report)


if __name__ == "__main__":
    main()
