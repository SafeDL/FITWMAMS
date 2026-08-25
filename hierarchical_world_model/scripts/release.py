#!/usr/bin/env python3
"""Run the formal staged release from one clean tagged starting point."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hierarchical_world_model.src.protocol import load_protocol_config, release_provenance  # noqa: E402
from world_model.src.core.utils import ensure_dir, save_json  # noqa: E402

CONFIG = ROOT / "hierarchical_world_model/config/release.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the formal tagged release pipeline.")
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--release-tag", required=True)
    args = parser.parse_args()
    config = load_protocol_config(args.config)
    # This is the only clean-worktree check.  Subsequent commands legitimately
    # create release artifacts and consume this immutable start certificate.
    session = ensure_dir(Path(config["paths"]["output_dir"]) / "final") / "release_session.json"
    save_json(release_provenance(release_tag=args.release_tag, require_clean=True), session)
    commands = [
        ["train.py", "--config", str(args.config), "--stage", "base"],
        ["train.py", "--config", str(args.config), "--stage", "stochastic_heads"],
        ["promote.py", "--config", str(args.config), "--release-tag", args.release_tag, "--release-session", str(session)],
        ["evaluate.py", "--config", str(args.config)],
        ["randomness_eval.py", "--config", str(args.config)],
        ["sampled_eval.py", "--config", str(args.config), "--release-tag", args.release_tag, "--release-session", str(session)],
        ["ams_readiness.py", "--config", str(args.config)],
        ["acceptance.py", "--config", str(args.config)],
    ]
    for command in commands:
        subprocess.run([sys.executable, str(ROOT / "hierarchical_world_model/scripts" / command[0]), *command[1:]], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
