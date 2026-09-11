#!/usr/bin/env python3
"""Record reproducibility and frozen-baseline facts before A2 training."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys

import torch

from common import ROOT, load_config, result_root


def command(*args: str) -> str:
    return subprocess.run(args, cwd=ROOT, text=True, capture_output=True, check=True).stdout.strip()


def digest(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    config, base = load_config()
    root = result_root(config)
    root.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "world_model": ROOT / base["paths"]["evaluation_checkpoint"],
        "baseline_a2": ROOT / config["paths"]["baseline_checkpoint"],
        "rule_model": ROOT / config["paths"]["rule_model"],
    }
    report = {
        "schema_name": "human_response_a2_phase_zero",
        "frozen_world_model": True,
        "git_head": command("git", "rev-parse", "HEAD"),
        "git_status": command("git", "status", "--short"),
        "artifact_sha256": {name: digest(path) for name, path in artifacts.items()},
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "baseline_status": "response_effect_verified_only; factual_noninferiority_not_claimed",
    }
    (root / "phase_zero.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
