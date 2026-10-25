"""Report supervision coverage and its policy/expert sources separately."""

import json
from pathlib import Path

from ads.common.instances import file_hash
from ads.tasks import SUPERVISED_ALGORITHMS

ROLES = ("normal_prior", "new_failure", "success_neighbour", "history")
FIELDS = (
    "algorithm",
    "seed",
    "method",
    "round",
    "role",
    "selected_instances",
    "qualified_instances",
    "qualified_fraction",
    "instances_with_labels",
    "available_label_frames",
    "policy_success_instances",
    "expert_instances",
    "expert_qualified_instances",
    "denominator_scope",
)


def coverage_for_training(directory, config, *, seed, method, round_index):
    directory = Path(directory)
    metadata = json.loads((directory / "training.json").read_text())
    if "teacher" not in metadata:
        return []
    curriculum_path = directory / "inputs/curriculum.json"
    if file_hash(curriculum_path) != metadata["curriculum_sha256"]:
        raise ValueError("expert coverage curriculum differs from training")
    curriculum = json.loads(curriculum_path.read_text())
    limit = config["repair"]["plant_normal_teacher_instances"]
    expected = [
        (row["instance_hash"], "normal_prior")
        for row in curriculum["categories"]["normal"][:limit]
    ]
    selected = json.loads((directory / "teacher_selection.json").read_text())
    failures = {row["instance_hash"] for row in curriculum["categories"]["new_failure"]}
    neighbours = {
        row["instance_hash"]
        for row in curriculum.get("success_neighbours", {}).values()
    }
    outcomes = {}
    with (directory / "search.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            identity, failure = row["instance_hash"], bool(row["safety_failure"])
            if identity in outcomes and outcomes[identity] != failure:
                raise ValueError("one expert instance has conflicting search outcomes")
            outcomes[identity] = failure
    for row in selected:
        identity = row["instance_hash"]
        if identity in failures and identity not in neighbours:
            role, failure = "new_failure", True
        elif identity in neighbours and identity not in failures:
            role, failure = "success_neighbour", False
        else:
            raise ValueError("selected expert has no unambiguous curriculum role")
        if outcomes.get(identity) is not failure:
            raise ValueError("expert role differs from its executed search outcome")
        expected.append((identity, role))
    expected.extend(
        (row["instance_hash"], "history")
        for row in curriculum.get("teacher_history", [])
    )
    reports = metadata["teacher"]["reports"]
    if [report["instance_hash"] for report in reports] != [
        identity for identity, _ in expected
    ]:
        raise ValueError("expert reports differ from the ordered training selections")
    counts = {
        role: {
            "selected_instances": 0,
            "qualified_instances": 0,
            "instances_with_labels": 0,
            "available_label_frames": 0,
            "policy_success_instances": 0,
            "expert_instances": 0,
            "expert_qualified_instances": 0,
        }
        for role in ROLES
    }
    for (_, role), report in zip(expected, reports):
        qualified = bool(report["qualified_demonstration"])
        samples = report.get("samples", 0)
        if not isinstance(samples, int) or samples < 0 or (samples and not qualified):
            raise ValueError("unqualified expert supplied positive supervision")
        counts[role]["selected_instances"] += 1
        counts[role]["qualified_instances"] += int(qualified)
        counts[role]["instances_with_labels"] += int(samples > 0)
        counts[role]["available_label_frames"] += samples
        if report.get("label_source") == "policy_success":
            counts[role]["policy_success_instances"] += 1
        else:
            counts[role]["expert_instances"] += 1
            counts[role]["expert_qualified_instances"] += int(qualified)
    return [
        {
            "algorithm": metadata.get("algorithm", "plant2"),
            "seed": seed,
            "method": method,
            "round": round_index,
            "role": role,
            **count,
            "qualified_fraction": (
                count["qualified_instances"] / count["selected_instances"]
                if count["selected_instances"]
                else None
            ),
            # Reused demonstrations remain available to each learner. Their
            # generation cost is counted once in teacher_generation.csv.
            "denominator_scope": "training_consumer_and_role",
        }
        for role, count in counts.items()
    ]


def teacher_coverage(output, config):
    output, rows = Path(output), []
    for algorithm in config["algorithms"]:
        if algorithm not in SUPERVISED_ALGORITHMS:
            continue
        for seed in config["seeds"]:
            base = output / algorithm / str(seed)
            directories = [
                (base / method / f"round_{index}", method, index)
                for method in config["methods"]
                for index in range(1, config["rounds"] + 1)
            ]
            for directory, method, index in directories:
                rows.extend(
                    coverage_for_training(
                        directory, config, seed=seed, method=method, round_index=index
                    )
                )
    return rows
