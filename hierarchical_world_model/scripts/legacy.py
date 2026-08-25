#!/usr/bin/env python3
"""Record why pre-release artifacts are ineligible for formal results."""

from __future__ import annotations

import hashlib
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hierarchical_world_model.src.protocol import (  # noqa: E402
    FORMAL_PROTOCOL_VERSION, logical_path, save_json,
)

LEGACY_ROOT = ROOT / "results/legacy/world_model"


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record why pre-release world-model artifacts are not formal results."
    )
    parser.add_argument("--source", type=Path, default=LEGACY_ROOT)
    args = parser.parse_args()
    output = args.source.resolve()
    if not output.exists():
        raise FileNotFoundError(f"legacy artifact directory does not exist: {output}")
    artifacts = [path for path in output.rglob("*") if path.is_file() and path.name != "legacy_manifest.json"]
    save_json({
        "status": "deprecated_not_formal_result",
        "reason": "pre-release artifacts may reference a legacy initial checkpoint and dirty worktree provenance",
        "formal_reader_policy": f"reject unless final manifest protocol_version is {FORMAL_PROTOCOL_VERSION}",
        "artifacts": [{"path": logical_path(path), "sha256": _hash(path)} for path in artifacts],
    }, output / "legacy_manifest.json")


if __name__ == "__main__":
    main()
