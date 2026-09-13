"""Frozen diffusion plan provider for the hierarchical world model."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

from diffusion.src.data import (
    ANCHOR_INDEX,
    BackgroundTrajectoryDataset,
    DataBundle,
    make_loader,
    smooth_position_residual,
)
from diffusion.src.train import load_checkpoint
from world_model.src.core.utils import ensure_dir, file_sha256, save_json
from world_model.src.core.evaluation_scope import EVALUATION_SCOPE_SCHEMA


def _row_digest(rows: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(rows, np.int64).tobytes()).hexdigest()


def complete_endogenous_response_plans(
    positions: np.ndarray,
    agent_states: np.ndarray,
    agent_valid: np.ndarray,
) -> np.ndarray:
    """Repair an accidentally empty present-agent plan from its anchor state.

    This is a defensive fallback only: all canonical slots, including
    ``same_rear``, are supplied to the diffusion plan provider.  A zero
    trajectory for a present non-origin vehicle would otherwise turn into
    maximum braking in the soft-plan controller.  Logged future states are
    never read and a valid diffusion plan is returned byte-for-byte unchanged.
    """
    result = np.asarray(positions, np.float32).copy()
    states = np.asarray(agent_states, np.float32)
    valid = np.asarray(agent_valid, bool)
    expected = (len(states), 149, 6, 2)
    if result.shape != expected:
        raise ValueError(f"positions must have shape {expected}, got {result.shape}")
    if states.ndim == 4:
        if states.shape[1] <= ANCHOR_INDEX:
            raise ValueError("agent_states do not contain the observed anchor")
        anchor_state = states[:, ANCHOR_INDEX]
        anchor_valid = valid[:, ANCHOR_INDEX]
    elif states.ndim == 3:
        anchor_state = states
        anchor_valid = valid
    else:
        raise ValueError("agent_states must be [batch,time,7,6] or [batch,7,6]")
    if anchor_state.shape[1:] != (7, 6) or anchor_valid.shape != anchor_state.shape[:2]:
        raise ValueError("anchor state/valid shapes must end in [7,6]/[7]")
    anchor = anchor_state[:, 1:, :2]
    present = anchor_valid[:, 1:]
    missing = (
        np.abs(result).max(axis=(1, 3)) <= 1.0e-6
    ) & present & (np.abs(anchor).max(axis=-1) > 1.0e-3)
    time = (np.arange(1, 150, dtype=np.float32) * 0.04)[None, :, None]
    velocity = anchor_state[:, 1:, 2:4]
    acceleration = np.clip(anchor_state[:, 1:, 4:6], -8.0, 4.0)
    extrapolated = (
        anchor[:, None]
        + velocity[:, None] * time[..., None]
        + 0.5 * acceleration[:, None] * np.square(time[..., None])
    )
    for batch, slot in np.argwhere(missing):
        result[batch, :, slot] = extrapolated[batch, :, slot]
    return result


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
    cache = output / "frozen_diffusion_test_plans.npz"
    manifest_path = output / "frozen_diffusion_test_plans.json"
    checkpoint_hash = file_sha256(checkpoint)
    digest = _row_digest(selected)
    if cache.exists() and manifest_path.exists():
        from world_model.src.core.utils import load_json

        manifest = load_json(manifest_path)
        if (
            manifest.get("checkpoint_sha256") == checkpoint_hash
            and manifest.get("row_digest") == digest
            and int(manifest.get("ddim_steps", -1)) == int(ddim_steps)
            and manifest.get("experiment_scope") == str(experiment_scope)
            and manifest.get("evaluation_scope") == EVALUATION_SCOPE_SCHEMA
        ):
            saved = np.load(cache)
            if np.array_equal(saved["row_index"], selected):
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
    residual_mean = np.asarray(contract["position_residual"]["mean"], np.float32)
    residual_std = np.asarray(contract["position_residual"]["std"], np.float32)
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
        residual = normalized.cpu().numpy().reshape(-1, 149, 6, 2)
        residual = smooth_position_residual(residual * residual_std + residual_mean)
        active = mask[:, 0].cpu().numpy().reshape(-1, 6, 2)[..., 0]
        reference = batch["trajectory_reference"].numpy()
        results.append((reference + residual) * active[:, None, :, None])
    positions = np.concatenate(results).astype(np.float32)
    np.savez_compressed(cache, row_index=selected, positions=positions)
    save_json(
        {
            "checkpoint": str(Path(checkpoint).resolve()),
            "checkpoint_sha256": checkpoint_hash,
            "row_digest": digest,
            "sequences": len(selected),
            "ddim_steps": int(ddim_steps),
            "initial_noise": "zero",
            "experiment_scope": str(experiment_scope),
            "evaluation_scope": EVALUATION_SCOPE_SCHEMA,
        },
        manifest_path,
    )
    return positions


@torch.no_grad()
def stochastic_diffusion_plan_samples(
    bundle: DataBundle,
    rows: np.ndarray,
    *,
    checkpoint: str | Path,
    device: torch.device,
    batch_size: int,
    ddim_steps: int,
    motion_seeds: tuple[int, ...],
) -> list[np.ndarray]:
    """Sample long plans while holding every 118-D condition fixed."""
    selected = np.asarray(rows, np.int64)
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
    generators = [
        torch.Generator(device=device).manual_seed(int(seed)) for seed in motion_seeds
    ]
    collected: list[list[np.ndarray]] = [[] for _ in motion_seeds]
    residual_mean = np.asarray(contract["position_residual"]["mean"], np.float32)
    residual_std = np.asarray(contract["position_residual"]["std"], np.float32)
    for batch in loader:
        condition = batch["condition"].to(device)
        mask = batch["target_mask"].to(device)
        active = mask[:, 0].cpu().numpy().reshape(-1, 6, 2)[..., 0]
        reference = batch["trajectory_reference"].numpy()
        for index, generator in enumerate(generators):
            normalized = model.sample_ddim(
                condition,
                mask,
                inference_steps=int(ddim_steps),
                generator=generator,
            )
            residual = normalized.cpu().numpy().reshape(-1, 149, 6, 2)
            residual = smooth_position_residual(residual * residual_std + residual_mean)
            collected[index].append((reference + residual) * active[:, None, :, None])
    return [np.concatenate(values).astype(np.float32) for values in collected]
