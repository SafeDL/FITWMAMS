"""Frozen diffusion plan provider for the hierarchical world model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from diffusion.src.data import (
    BackgroundTrajectoryDataset,
    DataBundle,
    decode_residual_trajectory,
    make_loader,
)
from diffusion.src.train import load_checkpoint
from hierarchical_world_model.src.core.utils import ensure_dir, save_json
from hierarchical_world_model.src.core.evaluation_scope import EVALUATION_SCOPE_SCHEMA
from .data import artifact_identity


@torch.no_grad()
def frozen_diffusion_plans(
    bundle: DataBundle,
    rows: np.ndarray,
    *,
    checkpoint: str | Path,
    output_dir: str | Path,
    device: torch.device,
    batch_size: int,
    ddim_steps: int,
    experiment_scope: str = "full",
) -> np.ndarray:
    """Return deterministic frozen-diffusion positions for exact row indices."""
    selected = np.asarray(rows, np.int64)
    output = ensure_dir(output_dir)
    cache = output / "positions.npz"
    manifest_path = output / "metadata.json"
    sequence_ids = np.asarray(bundle.arrays["sequence_id"])[selected].astype(str)
    checkpoint_path = Path(checkpoint).resolve()
    checkpoint_stat = checkpoint_path.stat()
    inputs = {
        "scenario_dataset": artifact_identity(bundle.flow_schema["dataset_npz"]),
        "decoder": artifact_identity(__file__),
    }
    if cache.exists() and manifest_path.exists():
        from hierarchical_world_model.src.core.utils import load_json

        manifest = load_json(manifest_path)
        if (
            manifest.get("checkpoint") == str(checkpoint_path)
            and manifest.get("checkpoint_size_bytes") == checkpoint_stat.st_size
            and manifest.get("checkpoint_mtime_ns") == checkpoint_stat.st_mtime_ns
            and int(manifest.get("ddim_steps", -1)) == int(ddim_steps)
            and manifest.get("experiment_scope") == str(experiment_scope)
            and manifest.get("evaluation_scope") == EVALUATION_SCOPE_SCHEMA
            and manifest.get("inputs") == inputs
        ):
            with np.load(cache) as saved:
                if np.array_equal(saved["sequence_id"], sequence_ids):
                    return np.asarray(saved["positions"], np.float32)
    model, payload = load_checkpoint(Path(checkpoint), device=device)
    model.eval()
    contract = payload["dataset_contract"]
    dataset = BackgroundTrajectoryDataset(
        bundle, selected, contract, evaluation_scope=True
    )
    loader = make_loader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        workers=0,
        seed=0,
    )
    results: list[np.ndarray] = []
    for batch in loader:
        condition = batch["condition"].to(device)
        mask = batch["target_mask"].to(device)
        normalized = model.sample_ddim(
            condition,
            mask,
            inference_steps=int(ddim_steps),
            initial_noise=torch.zeros_like(mask, dtype=condition.dtype),
        )
        active = mask[:, 0].cpu().numpy().reshape(-1, 6, 2)[..., 0]
        results.append(
            decode_residual_trajectory(
                normalized.cpu().numpy(),
                batch["trajectory_reference"].numpy(),
                active,
                contract,
            )
        )
    positions = np.concatenate(results).astype(np.float32)
    np.savez_compressed(cache, sequence_id=sequence_ids, positions=positions)
    save_json(
        {
            "checkpoint": str(checkpoint_path),
            "checkpoint_size_bytes": checkpoint_stat.st_size,
            "checkpoint_mtime_ns": checkpoint_stat.st_mtime_ns,
            "sequences": len(selected),
            "ddim_steps": int(ddim_steps),
            "initial_noise": "zero",
            "experiment_scope": str(experiment_scope),
            "evaluation_scope": EVALUATION_SCOPE_SCHEMA,
            "inputs": inputs,
            "condition_dim": contract["condition_dim"],
            "target_representation": contract["target_representation"],
        },
        manifest_path,
    )
    return positions
