"""Check actual campaign artifacts, budgets, policy ancestry and IID isolation."""

import argparse
import json
import pickle
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch
import yaml

from ads.common.evaluation_sampling import verify_final_roots, verify_sampling_spec
from ads.common.instances import GeneratedInstance, file_hash, json_hash
from ads.common.reporting import write_json
from ads.common.setup import REPO_ROOT
from ads.tasks import (
    NATIVE_PLANNERS,
    POLICY_SUFFIX,
    SUPERVISED_ALGORITHMS,
    experiment_graph,
    training_budget,
)


def checkpoint_progress(path, algorithm):
    if algorithm in ("dqn", "ppo"):
        with ZipFile(path) as archive:
            return json.loads(archive.read("data"))["num_timesteps"]
    path = Path(path)
    if algorithm == "str2":
        optimizer = (
            torch.load(
                path / "ads_optimizer.pth", map_location="cpu", weights_only=False
            )
            if (path / "ads_optimizer.pth").exists()
            else None
        )
    else:
        payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        optimizer = (
            payload.get("ads_optimizer")
            if algorithm in NATIVE_PLANNERS
            else payload.get("optimizer")
        )
    return (
        max((int(state["step"]) for state in optimizer["state"].values()), default=0)
        if optimizer
        else 0
    )


def check_archive_execution_labels(archive, snapshots):
    """Validate historical roles against the policy snapshot that observed them."""
    history = defaultdict(list)
    for snapshot in snapshots:
        for identity, records in snapshot.items():
            history[identity].extend(records)
    for identity, row in archive["entries"].items():
        # Common post-hoc partitions append annotations after the archive is
        # saved. Every original execution field must still match its journal.
        if not any(
            all(
                key in executed and executed[key] == value for key, value in row.items()
            )
            for executed in history.get(identity, [])
        ):
            raise ValueError(
                "archive contains an outcome absent from real execution journals"
            )
    for domain, identities in archive["domains"].items():
        if len(identities) != len(set(identities)):
            raise ValueError("duplicate instances within archive domain")
        for identity in identities:
            entry = archive["entries"][identity]
            if not any(
                row["safety_failure"]
                and row["branch_id"] == domain
                and row["root_sample_id"] == entry["root_sample_id"]
                for row in history[identity]
            ):
                raise ValueError(
                    "archive failure domain label differs from execution history"
                )
    for failure, success in archive["neighbours"].items():
        if not any(
            any(row["safety_failure"] for row in snapshot.get(failure, []))
            and any(not row["safety_failure"] for row in snapshot.get(success, []))
            for snapshot in snapshots
        ):
            raise ValueError(
                "success neighbour has no boundary under one policy snapshot"
            )


def check_dqn_replay(directory, *, baseline_steps, round_index, steps):
    directory = Path(directory)
    with ZipFile(directory / "policy.zip") as archive:
        settings = json.loads(archive.read("data"))
    warmup = min(1024, max(1, steps // 4))
    if settings["learning_starts"] != baseline_steps + warmup:
        raise ValueError("DQN fresh-buffer warmup differs from the cumulative baseline")
    with (directory / "replay_buffer.pkl").open("rb") as stream:
        replay = pickle.load(stream)
    expected = min(settings["buffer_size"], round_index * steps)
    if replay.size() != expected or replay.full != (
        round_index * steps >= settings["buffer_size"]
    ):
        raise ValueError(
            "DQN replay does not contain exactly the new-world repair experience"
        )
    if replay.observations.shape[-1] != 49 or replay.action_space.n != 5:
        raise ValueError(
            "DQN replay changed the registered observation or action interface"
        )
    if (
        not np.isfinite(replay.observations[:expected]).all()
        or not np.isfinite(replay.next_observations[:expected]).all()
    ):
        raise ValueError("DQN replay contains non-finite observations")
    if not np.isin(replay.actions[:expected], np.arange(5)).all():
        raise ValueError("DQN replay contains an invalid semantic action")


def check_records(records, *, assets_hash, cohort=None):
    for row in records:
        if row["assets_hash"] != assets_hash:
            raise ValueError("record uses different frozen world assets")
        if row["event_id"] != row["instance_hash"]:
            raise ValueError("event and concrete instance identity differ")
        if bool(row["safety_failure"]) != bool(row["collision"] or row["offroad"]):
            raise ValueError("safety label differs from ego collision or offroad")
        if (
            not row["numerical_valid"]
            or not np.isfinite(
                [
                    row[key]
                    for key in ("progress_m", "evt_score", "event_risk", "elapsed_s")
                ]
            ).all()
        ):
            raise ValueError("invalid numerical rollout evidence")
        if not 0 <= row["critical_frame"] <= row["physics_steps"] <= 149:
            raise ValueError("physical-step evidence is outside the world horizon")
        if cohort is not None:
            instance = cohort.get(row["instance_hash"])
            if instance is None or row["root_sample_id"] != instance.root_sample_id:
                raise ValueError("evaluation used a different concrete IID instance")
            if row["purpose"] != "final_test":
                raise ValueError("IID record has a development data purpose")
    if cohort is not None and (
        len(records) != len(cohort)
        or {row["instance_hash"] for row in records} != set(cohort)
    ):
        raise ValueError("IID evaluation has missing or duplicate instances")


def audit_training(output, config, *, deep=False):
    """Check executed evidence; an explicitly skipped no-label update is valid."""
    output = Path(output)
    protocol = json.loads((output / "protocol.json").read_text())
    if protocol["settings"] != dict(config, run_kind=config.get("run_kind", "formal")):
        raise ValueError("audit settings differ from protocol")
    graph, metadata, executions = experiment_graph(config), {}, 0
    for task in graph:
        if task.stage not in ("baseline", "repair"):
            continue
        directory = task.directory(output)
        report = json.loads((directory / "training.json").read_text())
        checkpoint = directory / ("policy" + POLICY_SUFFIX[task.algorithm])
        if report["policy_sha256"] != file_hash(checkpoint) or report[
            "protocol_hash"
        ] != json_hash(protocol):
            raise ValueError(
                "policy or protocol differs from executed training evidence"
            )
        field, budget = training_budget(
            config, task.algorithm, baseline=task.stage == "baseline"
        )
        actual = report["learning"][field]
        skipped = report["learning"].get("skipped") == "no_qualified_demonstrations"
        if (
            task.algorithm == "dtpp"
            and task.stage == "repair"
            and not skipped
            and report["learning"].get("dtpp_attention_mask_layout")
            != "batch_then_head"
        ):
            raise ValueError("DTPP repair lacks the corrected attention-mask layout")
        if actual != budget and not (
            task.algorithm in SUPERVISED_ALGORITHMS and actual == 0 and skipped
        ):
            raise ValueError("repair learning budget differs")
        if task.stage == "baseline":
            if report["base_sha256"] != file_hash(report["source_path"]):
                raise ValueError("initial policy source differs")
        elif (
            report["base_sha256"] != metadata[next(iter(graph[task]))]["policy_sha256"]
        ):
            raise ValueError("repair policy ancestry differs")
        manifest_path = directory / "inputs/curriculum.json"
        if report["curriculum_sha256"] != file_hash(manifest_path):
            raise ValueError("ordered training curriculum differs")
        manifest = json.loads(manifest_path.read_text())
        entries = [row for group in manifest["categories"].values() for row in group]
        entries += (
            list(manifest["success_neighbours"].values()) + manifest["teacher_history"]
        )
        for identity in {row["instance_hash"] for row in entries}:
            path = directory / "inputs/instances" / (identity + ".npz")
            if not path.exists():
                raise ValueError("training instance is missing")
            if deep:
                instance = GeneratedInstance.load(path)
                if (
                    instance.instance_hash != identity
                    or instance.purpose == "final_test"
                    or instance.source_kind != "tail_generated"
                ):
                    raise ValueError("training instance source or hash differs")
        for teacher in report.get("teacher", {}).get("reports", []):
            if not 0 <= teacher["candidates"] <= config["teacher"]["candidate_budget"]:
                raise ValueError("expert candidate budget exceeded")
            if teacher.get("samples", 0) and not teacher["qualified_demonstration"]:
                raise ValueError("unsuccessful expert received a positive label")
        if task.stage == "repair":
            rows = [
                json.loads(line)
                for line in (directory / "search.jsonl").read_text().splitlines()
            ]
            summary = json.loads((directory / "search_summary.json").read_text())
            if not 0 < len(rows) <= config["search"]["budget"] or summary[
                "evaluations"
            ] != len(rows):
                raise ValueError("search execution ledger differs from actual budget")
            if [r["evaluation_index"] for r in rows] != list(range(1, len(rows) + 1)):
                raise ValueError("search execution sequence differs")
            counts = Counter(r["execution_kind"] for r in rows)
            if (
                dict(counts) != summary["execution_costs"]
                or counts["branch_check"] > config["search"]["branch_check_budget"]
            ):
                raise ValueError("branch-check or total cost ledger differs")
            if counts["archive_replay"] > config["search"]["replay_limit"]:
                raise ValueError("archive replay limit exceeded")
            if summary["hit_fraction_is_prior_probability"]:
                raise ValueError(
                    "search results were interpreted as prior accident probability"
                )
            check_records(rows, assets_hash=json_hash(protocol["assets"]))
            if any(
                r["execution_kind"] == "branch_check" and r["independent_sampling_root"]
                for r in rows
            ):
                raise ValueError("diagnostic check is an independent sampling ancestor")
            if deep:
                for identity in {r["instance_hash"] for r in rows}:
                    instance = GeneratedInstance.load(
                        directory / "search_instances" / (identity + ".npz")
                    )
                    if (
                        instance.instance_hash != identity
                        or instance.purpose == "final_test"
                    ):
                        raise ValueError("executed search instance differs")
            archive = json.loads((directory / "archive/archive.json").read_text())
            snapshots = []
            for number in range(1, task.round_index + 1):
                journal = directory.parent / f"round_{number}" / "search.jsonl"
                snapshot = defaultdict(list)
                for line in journal.read_text().splitlines():
                    row = json.loads(line)
                    snapshot[row["instance_hash"]].append(row)
                snapshots.append(snapshot)
            check_archive_execution_labels(archive, snapshots)
            if task.round_index > 1:
                previous = json.loads(
                    (
                        directory.parent
                        / f"round_{task.round_index - 1}"
                        / "archive/archive.json"
                    ).read_text()
                )
                if set(archive["retained"]) != set(previous["entries"]):
                    raise ValueError("old archive endpoints were lost between rounds")
            for group in archive["domains"].values():
                if len(group) > archive["capacity"]:
                    raise ValueError("branch archive capacity exceeded")
                roots = Counter(
                    archive["entries"][key]["root_sample_id"] for key in group
                )
                if roots and max(roots.values()) > archive["root_limit"]:
                    raise ValueError("archive ancestor limit exceeded")
            if deep and task.algorithm == "dqn":
                baseline = output / "dqn" / str(task.seed) / "baseline/policy.zip"
                check_dqn_replay(
                    directory,
                    baseline_steps=checkpoint_progress(baseline, "dqn"),
                    round_index=task.round_index,
                    steps=budget,
                )
            executions += len(rows)
        metadata[task] = report
    return {
        "status": "passed",
        "protocol_hash": json_hash(protocol),
        "search_executions": executions,
        "training_units": len(metadata),
        "deep": deep,
    }


def audit_campaign(config_path, *, deep=False):
    from ads.common.cohorts import verify_evaluation_assets
    from ads.common.evaluation_sampling import COHORTS
    from ads.common.reporting import paired_rows

    config = yaml.safe_load(Path(config_path).read_text())
    output = REPO_ROOT / config["output_dir"]
    result = audit_training(output, config, deep=deep)
    protocol = json.loads((output / "protocol.json").read_text())
    frozen = json.loads((output / "frozen_policies.json").read_text())
    if frozen["protocol_hash"] != json_hash(protocol):
        raise ValueError("final policy freeze uses a different protocol")
    revision = output / "implementation_revision.json"
    if frozen.get("implementation_revision_sha256") != (
        file_hash(revision) if revision.exists() else None
    ):
        raise ValueError("implementation correction changed after policy freeze")
    verify_sampling_spec(frozen, protocol)
    verify_evaluation_assets(frozen["evaluation_assets"])
    executions = 0
    for cohort_name in COHORTS:
        instances = [
            GeneratedInstance.load(p)
            for p in sorted((output / "instances" / cohort_name).glob("*.npz"))
        ]
        verify_final_roots(
            instances, protocol, cohort_name, frozen["evaluation_assets"]
        )
        cohort = {i.instance_hash: i for i in instances}
        by_policy = {}
        for key, entry in frozen["policies"].items():
            if file_hash(REPO_ROOT / entry["path"]) != entry["sha256"]:
                raise ValueError("frozen policy changed")
            algorithm, seed, arm = key.split("/")
            payload = json.loads(
                (
                    output
                    / "evaluation"
                    / cohort_name
                    / algorithm
                    / seed
                    / (arm + ".json")
                ).read_text()
            )
            if payload["policy_sha256"] != entry["sha256"]:
                raise ValueError("evaluated policy differs")
            if payload["protocol_hash"] != json_hash(protocol):
                raise ValueError("evaluation protocol differs")
            rows = payload["records"]
            for row in rows:
                check_records(
                    [row], assets_hash=json_hash(cohort[row["instance_hash"]].assets)
                )
            if len(rows) != len(cohort) or {r["instance_hash"] for r in rows} != set(
                cohort
            ):
                raise ValueError("final evaluation instances differ")
            by_policy[key] = rows
            executions += len(rows)
        for algorithm in config["algorithms"]:
            for seed in config["seeds"]:
                for method in config["methods"]:
                    paired_rows(
                        by_policy[f"{algorithm}/{seed}/initial"],
                        by_policy[f"{algorithm}/{seed}/{method}"],
                    )
    with sqlite3.connect(output / "usage.sqlite") as db:
        if db.execute(
            "SELECT identity FROM usage GROUP BY identity HAVING COUNT(DISTINCT group_name)>1"
        ).fetchone():
            raise ValueError("test and development identities overlap")
    result.update(
        final_executions=executions, cohort_instances=config["evaluation"]["instances"]
    )
    write_json(output / "audit.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "ads/configs/risk_stratified_repair.yaml",
    )
    parser.add_argument("--deep", action="store_true")
    args = parser.parse_args()
    print(json.dumps(audit_campaign(args.config, deep=args.deep), indent=2))


if __name__ == "__main__":
    main()
