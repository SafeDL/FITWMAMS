#!/usr/bin/env python3
"""Freeze the validated staged checkpoint into one auditable final artifact."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hierarchical_world_model.src.config import WorldModelConfig  # noqa: E402
from hierarchical_world_model.src.model import DiffusionGuidedHiQR  # noqa: E402
from hierarchical_world_model.src.protocol import (  # noqa: E402
    canonical_hash, environment_provenance, load_protocol_config, logical_path,
    RANDOMNESS_NAMESPACE, release_provenance, FORMAL_PROTOCOL_VERSION,
)
from hierarchical_world_model.src.train import load_checkpoint, save_checkpoint  # noqa: E402
from world_model.src.core.utils import (  # noqa: E402
    file_sha256,
    load_json,
    save_json,
    select_device,
)

CONFIG = ROOT / "hierarchical_world_model/config/release.yaml"


def _probe(model: DiffusionGuidedHiQR, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(71)
    history = torch.randn((1, 25, 7, 6), generator=generator, device=device)
    history[..., 2] = history[..., 2].abs() + 20.0
    valid = torch.ones((1, 25, 7), dtype=torch.bool, device=device)
    reference = torch.randn((1, 25, 6, 2), generator=generator, device=device)
    maps = torch.zeros((1, 8, 8, 6), device=device)
    maps[..., 0] = torch.linspace(-100.0, 100.0, 8, device=device)[None, None]
    maps[..., 1] = torch.arange(-3, 5, device=device)[None, :, None] * 3.6
    maps[..., 2] = 1.0
    maps[..., 4] = 3.6
    noise_scene = torch.randn(
        (1, model.cfg.scene_latent_dim),
        generator=generator,
        device=device,
    )
    noise_agent = torch.randn(
        (1, 7, model.cfg.agent_latent_dim),
        generator=generator,
        device=device,
    )
    return model(
        history,
        valid,
        history[:, -1],
        valid[:, -1],
        reference,
        history[:, -1, 1:, :2],
        maps,
        torch.ones((1, 8, 8), dtype=torch.bool, device=device),
        scene_standard_normal=noise_scene,
        agent_standard_normal=noise_agent,
    ).actions


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize the validated final checkpoint without retraining; "
            "normally run once."
        )
    )
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--release-tag", required=True)
    parser.add_argument("--release-session", type=Path)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = load_protocol_config(config_path)
    provenance = (
        load_json(args.release_session)
        if args.release_session is not None
        else release_provenance(release_tag=args.release_tag, require_clean=True)
    )
    if provenance.get("release_tag") != args.release_tag or not provenance.get("worktree_clean_at_start"):
        raise RuntimeError("release session does not certify this clean release tag")
    if release_provenance().get("code_commit") != provenance.get("code_commit"):
        raise RuntimeError("release session commit no longer matches HEAD")
    device = select_device(config["training"].get("device", "auto"))
    destination = Path(config["paths"]["evaluation_checkpoint"])
    if destination.name != "final_world_model.pt":
        destination = destination.with_name("final_world_model.pt")
    source = destination.with_name("stochastic_heads_best.pt")
    base = destination.with_name("base_best.pt")
    manifest = destination.with_name("final_model_manifest.json")
    if destination.exists() and not args.force:
        raise FileExistsError(f"{destination} already exists; use --force after review")
    if not base.is_file():
        raise FileNotFoundError(f"missing base stage checkpoint: {base}")
    staged, payload = load_checkpoint(source, device=device)
    staged.eval()
    final = DiffusionGuidedHiQR(WorldModelConfig(**config["model"])).to(device).eval()
    final.load_state_dict(staged.state_dict())
    torch.testing.assert_close(_probe(final, device), _probe(staged, device), rtol=0.0, atol=0.0)
    save_checkpoint(
        destination, final, epoch=int(payload["epoch"]),
        validation_metric=float(payload["validation_metric"]),
        experiment_scope=str(payload["experiment_scope"]),
    )
    save_json(
        {
            "protocol_version": FORMAL_PROTOCOL_VERSION,
            "artifact": logical_path(destination),
            "checkpoint_sha256": file_sha256(destination),
            "source_checkpoint": logical_path(source),
            "source_checkpoint_sha256": file_sha256(source),
            "base_checkpoint": logical_path(base),
            "base_checkpoint_sha256": file_sha256(base),
            "flow_checkpoint": logical_path(config["paths"]["flow_checkpoint"]),
            "flow_checkpoint_sha256": file_sha256(config["paths"]["flow_checkpoint"]),
            "diffusion_checkpoint": logical_path(config["paths"]["diffusion_checkpoint"]),
            "diffusion_checkpoint_sha256": file_sha256(config["paths"]["diffusion_checkpoint"]),
            "model_config": final.cfg.to_dict(),
            "model_config_sha256": canonical_hash(final.cfg.to_dict()),
            "data_split": "recording-level 72,771/13,133/10,151",
            "training_seed": int(config["training"]["seed"]),
            "rng_schema": RANDOMNESS_NAMESPACE,
            "environment": environment_provenance(),
            **provenance,
            "finalization": "bitwise exact staged stochastic checkpoint freeze",
        },
        manifest,
    )


if __name__ == "__main__":
    main()
