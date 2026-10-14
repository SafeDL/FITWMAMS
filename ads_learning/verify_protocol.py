#!/usr/bin/env python3
"""Fail closed when a paper experiment no longer matches its protocol lock."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import highway_env
import diffusion
import hierarchical_world_model
import normalizing_flow

from .provenance import software_versions


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_protocol_lock(lock: Path, repo_root: Path) -> dict[str, object]:
    root = repo_root.resolve()
    payload = json.loads(lock.read_text(encoding="utf-8"))
    if payload.get("schema") != "fitwmams_paper_protocol_lock_v2":
        raise ValueError("unsupported protocol-lock schema")
    if not isinstance(payload.get("files"), list) or not payload["files"]:
        raise ValueError("protocol lock has no files")
    mismatches = []
    seen: set[str] = set()
    for record in payload["files"]:
        relative = str(record["path"])
        if relative in seen:
            mismatches.append({"path": relative, "error": "duplicate_lock_entry"})
            continue
        seen.add(relative)
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError:
            mismatches.append({"path": relative, "error": "outside_repository"})
            continue
        if not path.is_file():
            mismatches.append({"path": record["path"], "error": "missing"})
            continue
        actual = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
        if actual["bytes"] != record["bytes"] or actual["sha256"] != record["sha256"]:
            mismatches.append(
                {"path": record["path"], "expected": record, "actual": actual}
            )
    current_software = software_versions()
    if payload.get("software") != current_software:
        mismatches.append(
            {
                "path": "<software>",
                "expected": payload.get("software"),
                "actual": current_software,
            }
        )
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if payload.get("gpu") != gpu:
        mismatches.append(
            {"path": "<gpu>", "expected": payload.get("gpu"), "actual": gpu}
        )
    current_imports = {}
    for name, module in {
        "diffusion": diffusion,
        "hierarchical_world_model": hierarchical_world_model,
        "highway_env": highway_env,
        "normalizing_flow": normalizing_flow,
    }.items():
        try:
            current_imports[name] = str(
                Path(module.__file__).resolve().relative_to(root)
            )
        except ValueError:
            current_imports[name] = str(Path(module.__file__).resolve())
    if payload.get("runtime_imports") != current_imports:
        mismatches.append(
            {
                "path": "<runtime_imports>",
                "expected": payload.get("runtime_imports"),
                "actual": current_imports,
            }
        )
    return {
        "protocol_id": payload.get("protocol_id"),
        "protocol_lock": str(lock.resolve()),
        "protocol_lock_sha256": _sha256(lock),
        "checked_files": len(payload["files"]),
        "mismatches": mismatches,
        "verified": not mismatches,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    args = parser.parse_args()
    report = verify_protocol_lock(args.lock, args.repo_root)
    print(json.dumps(report, indent=2))
    if report["mismatches"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
