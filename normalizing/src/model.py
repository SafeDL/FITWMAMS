"""Normalizing-flow model construction and checkpoint helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from .utils import add_local_nflows_to_path


def build_maf_flow(
    *,
    num_features: int,
    context_features: int,
    model_cfg: dict[str, Any],
    repo_root: str | Path,
):
    add_local_nflows_to_path(repo_root)
    from nflows.distributions.normal import StandardNormal
    from nflows.flows.base import Flow
    from nflows.transforms.autoregressive import (
        MaskedAffineAutoregressiveTransform,
        MaskedPiecewiseRationalQuadraticAutoregressiveTransform,
    )
    from nflows.transforms.base import CompositeTransform
    from nflows.transforms.permutations import ReversePermutation

    layers = []
    num_layers = int(model_cfg.get("num_layers", 6))
    hidden_features = int(model_cfg.get("hidden_features", 128))
    num_blocks = int(model_cfg.get("num_blocks", 2))
    dropout = float(model_cfg.get("dropout_probability", 0.0))
    use_residual = bool(model_cfg.get("use_residual_blocks", True))
    use_batch_norm = bool(model_cfg.get("use_batch_norm", False))
    transform_type = str(model_cfg.get("transform_type", "affine")).lower()
    for _ in range(num_layers):
        if transform_type in {"rq_spline", "rational_quadratic_spline", "spline"}:
            layers.append(
                MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                    features=int(num_features),
                    hidden_features=hidden_features,
                    context_features=int(context_features),
                    num_bins=int(model_cfg.get("num_bins", 8)),
                    tails="linear",
                    tail_bound=float(model_cfg.get("tail_bound", 4.0)),
                    num_blocks=num_blocks,
                    use_residual_blocks=use_residual,
                    random_mask=False,
                    activation=F.relu,
                    dropout_probability=dropout,
                    use_batch_norm=use_batch_norm,
                )
            )
        elif transform_type == "affine":
            layers.append(
                MaskedAffineAutoregressiveTransform(
                    features=int(num_features),
                    hidden_features=hidden_features,
                    context_features=int(context_features),
                    num_blocks=num_blocks,
                    use_residual_blocks=use_residual,
                    random_mask=False,
                    activation=F.relu,
                    dropout_probability=dropout,
                    use_batch_norm=use_batch_norm,
                )
            )
        else:
            raise ValueError(f"Unsupported MAF transform_type={transform_type!r}")
        layers.append(ReversePermutation(features=int(num_features)))
    transform = CompositeTransform(layers)
    distribution = StandardNormal([int(num_features)])
    return Flow(transform=transform, distribution=distribution)


def build_realnvp_flow(
    *,
    num_features: int,
    model_cfg: dict[str, Any],
    repo_root: str | Path,
):
    add_local_nflows_to_path(repo_root)
    from nflows.flows.realnvp import SimpleRealNVP

    return SimpleRealNVP(
        features=int(num_features),
        hidden_features=int(model_cfg.get("hidden_features", 128)),
        num_layers=int(model_cfg.get("num_layers", 6)),
        num_blocks_per_layer=int(model_cfg.get("num_blocks_per_layer", 2)),
        use_volume_preserving=bool(model_cfg.get("use_volume_preserving", False)),
        dropout_probability=float(model_cfg.get("dropout_probability", 0.0)),
        batch_norm_within_layers=bool(model_cfg.get("batch_norm_within_layers", False)),
        batch_norm_between_layers=bool(model_cfg.get("batch_norm_between_layers", False)),
    )


def save_checkpoint(
    path: str | Path,
    *,
    flow,
    model_cfg: dict[str, Any],
    schema: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": flow.state_dict(),
            "model_cfg": dict(model_cfg),
            "schema": schema,
            "metrics": metrics,
        },
        p,
    )


def load_maf_checkpoint(
    path: str | Path,
    *,
    repo_root: str | Path,
    map_location: str | torch.device = "cpu",
):
    payload = torch.load(Path(path), map_location=map_location)
    schema = payload["schema"]
    model_cfg = payload["model_cfg"]
    flow = build_maf_flow(
        num_features=len(schema["feature_names"]),
        context_features=len(schema["context_names"]),
        model_cfg=model_cfg,
        repo_root=repo_root,
    )
    flow.load_state_dict(payload["state_dict"])
    flow.to(map_location)
    flow.eval()
    return flow, payload
