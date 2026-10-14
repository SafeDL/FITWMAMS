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
    for _ in range(int(model_cfg["num_layers"])):
        layers.append(
            MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                features=int(num_features),
                hidden_features=int(model_cfg["hidden_features"]),
                context_features=int(context_features),
                num_bins=int(model_cfg["num_bins"]),
                tails="linear",
                tail_bound=float(model_cfg["tail_bound"]),
                num_blocks=int(model_cfg["num_blocks"]),
                use_residual_blocks=True,
                random_mask=False,
                activation=F.relu,
                dropout_probability=float(model_cfg["dropout_probability"]),
                use_batch_norm=False,
            )
        )
        layers.append(ReversePermutation(features=int(num_features)))
    transform = CompositeTransform(layers)
    distribution = StandardNormal([int(num_features)])
    return Flow(transform=transform, distribution=distribution)
