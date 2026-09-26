"""Source/config fingerprint for the maintained single-pass traffic runtime.

Frozen model weights and Diffusion plans retain their own checkpoint/cache
manifests; this hash detects changes to the execution code and YAML contract.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNTIME_SOURCES = (
    "hierarchical_world_model/config/world_model.yaml",
    "hierarchical_world_model/src/ads_interventions.py",
    "hierarchical_world_model/src/composition.py",
    "hierarchical_world_model/src/continuation.py",
    "hierarchical_world_model/src/evaluation.py",
    "hierarchical_world_model/src/execution.py",
    "hierarchical_world_model/src/highway.py",
    "hierarchical_world_model/src/model.py",
    "hierarchical_world_model/src/planner.py",
    "hierarchical_world_model/src/reaction_controller.py",
    "hierarchical_world_model/src/stochastic_drivers/online.py",
    "traffic_components/src/core/dynamics.py",
)


def online_runtime_sha256() -> str:
    digest = hashlib.sha256()
    for relative in RUNTIME_SOURCES:
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update((ROOT / relative).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()
