#!/usr/bin/env python3
"""Evaluate, sample, and visualize the highD EVT-tail flow."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from normalizing.src.evaluation import evaluate_tail_flow  # noqa: E402
from normalizing.src.utils import load_yaml, setup_logging  # noqa: E402


CONFIG_PATH = (
    Path(__file__).resolve().parent
    / "configs"
    / "highd_tail_flow_best.yaml"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--output-prefix", default="generated_samples")
    parser.add_argument("--sampling-temperature", type=float, default=None)
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    setup_logging(args.log_level)
    config_path = Path(args.config).resolve()
    config = load_yaml(config_path)
    if args.sampling_temperature is not None:
        eval_cfg = dict(config.get("evaluation", {}))
        eval_cfg["sampling_temperature"] = float(args.sampling_temperature)
        config["evaluation"] = eval_cfg
    evaluate_tail_flow(
        config,
        config_dir=config_path.parent,
        repo_root=ROOT,
        checkpoint_path=Path(args.checkpoint).resolve() if args.checkpoint else None,
        num_samples=args.num_samples,
        output_prefix=str(args.output_prefix),
        run_baselines=not bool(args.skip_baselines),
        generate_figures=not bool(args.skip_figures),
    )


if __name__ == "__main__":
    main()
