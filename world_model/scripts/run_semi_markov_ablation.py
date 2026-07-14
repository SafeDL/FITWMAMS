#!/usr/bin/env python3
"""Run B0/B1/B2/Full Semi-Markov ablations under one data protocol."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from world_model.src.semi_markov_evaluation import evaluate_semi_markov_world_model
from world_model.src.semi_markov_train import train_semi_markov_world_model
from world_model.src.utils import load_yaml, setup_logging

CONFIG = Path(__file__).resolve().parent / "configs" / "highd_semi_markov_relational_10k.yaml"


def ablation_config(config: dict, variant: str, config_dir: Path) -> dict:
    result = {**config, "paths": dict(config["paths"]), "model": dict(config.get("model", {}))}
    output = Path(result["paths"]["output_dir"])
    if not output.is_absolute():
        output = (config_dir / output).resolve()
    cache_owner = Path(result["paths"].get("sequence_cache_dir", config["paths"]["output_dir"]))
    if not cache_owner.is_absolute():
        cache_owner = (config_dir / cache_owner).resolve()
    result["paths"]["output_dir"] = str(output / f"ablation_{variant}")
    result["paths"]["sequence_cache_dir"] = str(cache_owner)
    model = result["model"]
    model["learn_duration"] = True
    model["use_intent_response"] = True
    if variant == "b0":
        model.update({"num_latent_states": 1, "beta_latent": 0.0, "learn_duration": False})
    elif variant == "b1":
        model["learn_duration"] = False
    elif variant == "b2":
        model["use_intent_response"] = False
    elif variant != "full":  # pragma: no cover - argparse owns choices
        raise ValueError(variant)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(CONFIG))
    parser.add_argument("--variant", choices=("b0", "b1", "b2", "full"), required=True)
    parser.add_argument("--stage", choices=("train", "evaluate"), default="train")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    setup_logging(args.log_level)
    config_path = Path(args.config).resolve()
    config = ablation_config(load_yaml(config_path), args.variant, config_path.parent)
    if args.stage == "train":
        train_semi_markov_world_model(
            config, config_dir=config_path.parent, epochs=args.epochs,
            max_sequences=args.max_sequences or None,
        )
    else:
        checkpoint = Path(args.checkpoint).resolve() if args.checkpoint else None
        evaluate_semi_markov_world_model(
            config, config_dir=config_path.parent, checkpoint=checkpoint, max_sequences=args.max_sequences,
        )


if __name__ == "__main__":
    main()
