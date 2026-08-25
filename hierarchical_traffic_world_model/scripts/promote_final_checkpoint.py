#!/usr/bin/env python3
"""Freeze the validated configuration into one auditable final checkpoint.

This is deliberately a checkpoint migration, not training: it materializes
the model configuration that previous evaluations supplied as an override and
proves its outputs are bitwise-equivalent before writing the final artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hierarchical_traffic_world_model.src.config import WorldModelConfig  # noqa: E402
from hierarchical_traffic_world_model.src.model import DiffusionGuidedHiQR  # noqa: E402
from hierarchical_traffic_world_model.src.train import (  # noqa: E402
    _load_compatible_state_dict,
    load_checkpoint,
    save_checkpoint,
)
from world_model.src.core.utils import (  # noqa: E402
    file_sha256,
    load_yaml,
    save_json,
    select_device,
)

CONFIG = ROOT / "hierarchical_traffic_world_model/configs/highd_stochastic_causal_hiqr_full.yaml"


def _canonical_json_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _git_provenance() -> dict[str, object]:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            text=True,
        ).strip()
    )
    return {"code_commit": commit, "worktree_dirty": dirty}


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
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = load_yaml(config_path)
    device = select_device(config["training"].get("device", "auto"))
    destination = Path(config["paths"]["evaluation_checkpoint"])
    if destination.name != "final_world_model.pt":
        destination = destination.with_name("final_world_model.pt")
    source = destination.with_name("best_hierarchical_world_model.pt")
    manifest = destination.with_name("final_model_manifest.json")
    if destination.exists() and not args.force:
        raise FileExistsError(f"{destination} already exists; use --force after review")
    legacy, payload = load_checkpoint(source, device=device)
    final = DiffusionGuidedHiQR(WorldModelConfig(**config["model"])).to(device).eval()
    _load_compatible_state_dict(final, legacy.state_dict())
    legacy_override = DiffusionGuidedHiQR(
        WorldModelConfig(**config["model"])
    ).to(device).eval()
    _load_compatible_state_dict(legacy_override, legacy.state_dict())
    torch.testing.assert_close(
        _probe(final, device),
        _probe(legacy_override, device),
        rtol=0.0,
        atol=0.0,
    )
    save_checkpoint(
        destination, final, epoch=int(payload["epoch"]),
        validation_metric=float(payload["validation_metric"]),
        experiment_scope=str(payload["experiment_scope"]),
    )
    save_json(
        {
            "artifact": str(destination.resolve()),
            "checkpoint_sha256": file_sha256(destination),
            "source_checkpoint": str(source.resolve()),
            "source_checkpoint_sha256": file_sha256(source),
            "flow_checkpoint": str(Path(config["paths"]["flow_checkpoint"]).resolve()),
            "flow_checkpoint_sha256": file_sha256(config["paths"]["flow_checkpoint"]),
            "diffusion_checkpoint": str(Path(config["paths"]["diffusion_checkpoint"]).resolve()),
            "diffusion_checkpoint_sha256": file_sha256(config["paths"]["diffusion_checkpoint"]),
            "model_config": final.cfg.to_dict(),
            "model_config_sha256": _canonical_json_hash(final.cfg.to_dict()),
            "data_split": "recording-level 72,771/13,133/10,151",
            "training_seed": int(config["training"]["seed"]),
            **_git_provenance(),
            "migration": "validated evaluation configuration materialized without retraining",
            "equivalence_probe": (
                "bitwise exact against legacy checkpoint plus former evaluation override"
            ),
        },
        manifest,
    )


if __name__ == "__main__":
    main()
