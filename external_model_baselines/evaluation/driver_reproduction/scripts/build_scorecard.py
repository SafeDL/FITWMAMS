"""Write the cross-model scorecard from retained evidence."""

from __future__ import annotations

import argparse
from pathlib import Path

from external_model_baselines.evaluation.driver_reproduction.scorecard import ROOT, write_scorecard


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "results"
        / "driver_reproduction"
        / "matched_highd"
        / "driver_model_scorecard.json",
        help="JSON destination; defaults beside the matched driver evaluation.",
    )
    args = parser.parse_args()
    print(write_scorecard(args.output))


if __name__ == "__main__":
    main()
