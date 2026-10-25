"""Execute a registered experiment unit or prepare its persisted search."""

import argparse
import fcntl
import os
from pathlib import Path

from ads.common.diagnostics import capture_failure
from ads.common.reporting import write_json
from ads.common.setup import REPO_ROOT
from ads.risk_stratified_repair import Campaign
from ads.tasks import POLICY_SUFFIX, Task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("baseline", "repair", "freeze", "iid", "diagnostic", "report")
    )
    parser.add_argument("algorithm", nargs="?", choices=tuple(POLICY_SUFFIX))
    parser.add_argument("seed", nargs="?", type=int)
    parser.add_argument(
        "--method", choices=("initial", "monte_carlo", "subset", "guided")
    )
    parser.add_argument("--round", dest="round_index", type=int, choices=(1, 2))
    operation = parser.add_mutually_exclusive_group()
    operation.add_argument("--training-only", action="store_true")
    operation.add_argument("--search-only", action="store_true")
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "ads/configs/risk_stratified_repair.yaml",
    )
    args = parser.parse_args()
    if args.training_only and args.stage != "repair":
        raise ValueError("--training-only requires a completed repair search")
    if args.search_only and args.stage != "repair":
        raise ValueError("--search-only requires a registered repair unit")
    campaign = Campaign(args.config)
    campaign.verify_runtime()
    if args.stage not in ("freeze", "report") and (
        args.algorithm not in campaign.config["algorithms"]
        or args.seed not in campaign.config["seeds"]
    ):
        raise ValueError("worker identity is not in the registered protocol")
    if (
        args.stage in ("baseline", "repair")
        and (campaign.output / "frozen_policies.json").exists()
    ):
        raise ValueError("training workers cannot run after policy freeze")
    if args.stage == "repair" and (args.method is None or args.round_index is None):
        raise ValueError("repair worker requires a registered method and round")
    task = Task(args.stage, args.algorithm, args.seed, args.method, args.round_index)
    directory = task.directory(campaign.output)
    directory.mkdir(parents=True, exist_ok=True)
    if args.search_only and (directory / "training.json").exists():
        raise ValueError("cannot prepare search for an already trained repair unit")
    status = {
        "status": "running",
        "pid": os.getpid(),
        "stage": args.stage,
        "algorithm": args.algorithm,
        "seed": args.seed,
        "method": args.method,
        "round": args.round_index,
    }
    if args.training_only:
        status["training_only"] = True
    if args.search_only:
        status["search_only"] = True
    with (directory / "unit.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write_json(directory / "unit_status.json", status)
        try:
            if args.stage == "baseline":
                policy = campaign.baseline(args.algorithm, args.seed)
            elif args.stage == "repair":
                initial = campaign.checkpoint(
                    campaign.output / args.algorithm / str(args.seed) / "baseline",
                    args.algorithm,
                )
                if args.training_only:
                    policy = campaign.train_saved_search(
                        args.algorithm, args.seed, args.method, args.round_index
                    )
                else:
                    policy = campaign.run_branch(
                        args.algorithm,
                        args.seed,
                        args.method,
                        initial,
                        through_round=args.round_index,
                        search_only=args.search_only,
                    )
            elif args.stage == "freeze":
                from ads.audit import audit_training

                write_json(
                    campaign.output / "training_audit.json",
                    audit_training(campaign.output, campaign.config),
                )
                campaign.run()
                campaign.final_cohorts()
                policy = campaign.output / "frozen_policies.json"
            elif args.stage == "iid":
                campaign.evaluate_one(f"{args.algorithm}/{args.seed}/{args.method}")
                policy = campaign.output / "evaluation"
            elif args.stage == "diagnostic":
                key = f"{args.algorithm}/{args.seed}/{args.method}"
                result = campaign.archive_diagnostics(
                    campaign.verify_freeze(), keys=[key]
                )[key]
                policy = directory / "result.json"
                write_json(policy, result)
            else:
                campaign.evaluate()
                from ads.audit import audit_campaign

                audit_campaign(args.config, deep=True)
                policy = campaign.output / "evaluation/summary.json"
        except Exception as error:
            try:
                diagnostic = capture_failure(error, campaign.output)
            except Exception as diagnostic_error:  # noqa: BLE001 -- preserve the original experiment error
                diagnostic = f"capture failed: {diagnostic_error}"
            write_json(
                directory / "unit_status.json",
                {
                    **status,
                    "status": "failed",
                    "error": str(error),
                    "diagnostic": diagnostic,
                },
            )
            raise
        write_json(
            directory / "unit_status.json",
            {
                **status,
                "status": "search_complete" if args.search_only else "complete",
                "search_summary" if args.search_only else "policy": str(policy),
            },
        )


if __name__ == "__main__":
    main()
