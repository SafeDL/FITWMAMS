"""Normalizing-flow model construction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

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
    transform_type = str(model_cfg.get("transform_type", "rq_spline")).lower()
    if transform_type != "rq_spline":
        raise ValueError(
            f"Unsupported MAF transform_type={transform_type!r}; expected 'rq_spline'"
        )
    for _ in range(num_layers):
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
        layers.append(ReversePermutation(features=int(num_features)))
    transform = CompositeTransform(layers)
    distribution = StandardNormal([int(num_features)])
    return Flow(transform=transform, distribution=distribution)
