"""Command-line entry point for the bounded highD evaluation suite."""
from __future__ import annotations

import argparse

from .evaluate import evaluate_highd


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the 25 Hz active-inference driver on highD braking events.")
    parser.add_argument("--dataset", default="dynamic_ar_idm/artifacts/full_data/full_highd_25hz.npz")
    parser.add_argument("--output", default="active_inference_driver/artifacts")
    parser.add_argument("--max-events", type=int, default=8)
    args = parser.parse_args()
    print(evaluate_highd(args.dataset, args.output, max_events=args.max_events))


if __name__ == "__main__":
    main()
