#!/usr/bin/env python3
"""Write content hashes for the frozen ADS/world experimental protocol."""

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--file", type=Path, action="append", default=[])
    parser.add_argument(
        "--file-list",
        type=Path,
        help="UTF-8 file with one repository-relative input per non-comment line",
    )
    parser.add_argument(
        "--tree",
        type=Path,
        action="append",
        default=[],
        help="Recursively include protocol source/config files under this path",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol-id", default="fitwmams-paper-final")
    args = parser.parse_args()
    if not args.file and args.file_list is None and not args.tree:
        parser.error("at least one --file, --file-list, or --tree is required")
    root = args.repo_root.resolve()
    values = list(args.file)
    if args.file_list is not None:
        values.extend(
            Path(line.strip())
            for line in args.file_list.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    allowed_tree_suffixes = {
        ".py", ".yaml", ".yml", ".json", ".csv", ".md", ".pdf"
    }
    for tree_value in args.tree:
        tree = tree_value if tree_value.is_absolute() else root / tree_value
        if not tree.is_dir():
            raise NotADirectoryError(tree)
        values.extend(
            path
            for path in sorted(tree.rglob("*"))
            if path.is_file()
            and path.suffix.lower() in allowed_tree_suffixes
            and "__pycache__" not in path.parts
        )
    files = []
    seen: set[str] = set()
    for value in values:
        path = value if value.is_absolute() else root / value
        if not path.is_file():
            raise FileNotFoundError(path)
        try:
            relative = str(path.resolve().relative_to(root))
        except ValueError as error:
            raise ValueError(f"protocol input lies outside repository: {path}") from error
        if relative in seen:
            continue
        seen.add(relative)
        files.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    payload = {
        "schema": "fitwmams_paper_protocol_lock_v2",
        "protocol_id": args.protocol_id,
        "files": files,
        "software": software_versions(),
        "gpu": gpu,
        "file_count": len(files),
        "runtime_imports": {
            name: str(Path(module.__file__).resolve().relative_to(root))
            for name, module in {
                "diffusion": diffusion,
                "hierarchical_world_model": hierarchical_world_model,
                "highway_env": highway_env,
                "normalizing_flow": normalizing_flow,
            }.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
