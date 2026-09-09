"""Map canonical rows to prefix-only sampled nominal references."""
from __future__ import annotations

from dataclasses import replace
import hashlib
from typing import Any

import numpy as np

from .composition import HierarchicalWorldSampler, SampledWorldBatch
from .randomness import WorldExogenousState


def prefix_only_sample(
    sampler: HierarchicalWorldSampler, *, c0: np.ndarray, slot_mask: np.ndarray,
    exogenous: WorldExogenousState,
) -> SampledWorldBatch:
    """Compose one world solely from logged C0 plus declared base variables."""
    sample = sampler.compose_constraints_from_base_randomness(
        np.asarray(c0, np.float32), np.asarray(slot_mask, bool),
        k_base_latent=exogenous.k_base_latent,
        diffusion_noise=exogenous.diffusion_noise,
    )
    return replace(sample, exogenous_state=exogenous)


def row_prefix_inputs(bundle: Any, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return only Flow C0 and slot mask for canonical sequence row IDs."""
    rows = np.asarray(rows, np.int64)
    flow_rows = np.asarray(bundle.flow_row_for_sequence, np.int64)[rows]
    return (
        np.asarray(bundle.flow_arrays["features"], np.float32)[flow_rows].copy(),
        np.asarray(bundle.flow_arrays["slot_mask"], bool)[flow_rows].copy(),
    )


def reference_input_digest(c0: np.ndarray, slot_mask: np.ndarray, exogenous: WorldExogenousState) -> str:
    """Stable provenance for a cache key; excludes all highD future arrays."""
    digest = hashlib.sha256()
    for value in (c0, slot_mask, exogenous.k_base_latent, exogenous.diffusion_noise,
                  exogenous.scene_innovations, exogenous.agent_response_innovations,
                  exogenous.policy_response_innovations):
        digest.update(np.ascontiguousarray(value).view(np.uint8))
    return digest.hexdigest()
