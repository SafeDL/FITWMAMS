from __future__ import annotations
import argparse
from pathlib import Path
from world_model.trafficbots.config import load_config
from world_model.trafficbots.evaluation import evaluate
def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--config", type=Path); parser.add_argument("--checkpoint", type=Path, required=True); parser.add_argument("--max-sequences", type=int, default=0); args=parser.parse_args()
    evaluate(load_config(args.config) if args.config else load_config(), args.checkpoint, maximum=args.max_sequences)
if __name__ == "__main__": main()
