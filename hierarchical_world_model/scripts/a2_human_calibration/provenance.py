#!/usr/bin/env python3
"""Write the immutable Phase-0 environment and frozen-artifact provenance."""

from __future__ import annotations

import json
import subprocess
import sys

import torch

from common import ROOT, load_config, result_root

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from world_model.src.core.utils import file_sha256


def command(*args: str) -> str:
    return subprocess.run(args, cwd=ROOT, text=True, capture_output=True, check=True).stdout.strip()


def main() -> None:
    config, world = load_config()
    output = result_root(config) / "provenance.json"
    if output.exists():
        raise RuntimeError("Phase-0 provenance is immutable and will not be overwritten")
    cuda = torch.version.cuda
    output.write_text(json.dumps({
        "schema": "a2_human_calibration_phase_zero",
        "git_head": command("git", "rev-parse", "HEAD"),
        "dirty_diff": command("git", "diff", "--binary", "HEAD"),
        "git_status": command("git", "status", "--short"),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "conda_environment": command("conda", "env", "export", "-n", "tread"),
        "conda_history": command("conda", "list", "-n", "tread", "--revisions"),
        "frozen_artifact_sha256": {
            "hiq": file_sha256(ROOT / world["paths"]["evaluation_checkpoint"]),
            "legacy_a2": file_sha256(ROOT / config["paths"]["baseline_checkpoint"]),
            "rule_model": file_sha256(ROOT / config["paths"]["rule_model"]),
        },
    }, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
