"""Small content-addressed provenance records for experiment inputs."""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: str | Path) -> str:
    value = Path(path).resolve()
    digest = hashlib.sha256()
    with value.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_record(path: str | Path | None) -> dict[str, object] | None:
    """Describe an immutable file input by resolved path, size and digest."""
    if path is None:
        return None
    value = Path(path).resolve()
    if not value.is_file():
        raise FileNotFoundError(value)
    return {
        "path": str(value),
        "bytes": int(value.stat().st_size),
        "sha256": sha256_file(value),
    }
