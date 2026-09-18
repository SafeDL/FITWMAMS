from __future__ import annotations

import argparse
from pathlib import Path

from counterfactual_response.experiment import Scenario, run_experiment
from counterfactual_response.visualize import render_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run paired stochastic-driver braking probes")
    parser.add_argument("--output", type=Path, default=Path("counterfactual_response/artifacts"))
    parser.add_argument("--futures", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--skip-gif", action="store_true")
    args = parser.parse_args()
    scenario = Scenario()
    report, ensemble = run_experiment(args.output, scenario=scenario, futures=args.futures, seed=args.seed)
    if not args.skip_gif:
        render_outputs(report, ensemble, args.output)
    print(args.output / "response_metrics.json")


if __name__ == "__main__":
    main()
