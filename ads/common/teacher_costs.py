"""Count each required cached expert generation once, regardless of its consumer."""

import json
from pathlib import Path

from ads.tasks import SUPERVISED_ALGORITHMS


def teacher_generation_costs(output, config):
    output = Path(output)
    unique = {}
    for algorithm in config["algorithms"]:
        if algorithm not in SUPERVISED_ALGORITHMS:
            continue
        for seed in config["seeds"]:
            base = output / algorithm / str(seed)
            directories = [
                base / method / f"round_{round_index}"
                for method in config["methods"]
                for round_index in range(1, config["rounds"] + 1)
            ]
            for directory in directories:
                metadata = json.loads((directory / "training.json").read_text())
                if "teacher" not in metadata:
                    continue
                curriculum = json.loads(
                    (directory / "inputs/curriculum.json").read_text()
                )
                limit = config["repair"]["plant_normal_teacher_instances"]
                normal_count = min(limit, len(curriculum["categories"]["normal"]))
                for index, report in enumerate(metadata["teacher"]["reports"]):
                    if report.get("label_source") == "policy_success":
                        continue
                    name = "normal" if index < normal_count else "failures"
                    path = Path(report["cache_path"])
                    actual = json.loads(path.read_text())
                    for field in (
                        "instance_hash",
                        "teacher_config",
                        "candidates",
                        "candidate_budget",
                        "coarse_candidates",
                        "refined_candidates",
                        "qualified_demonstration",
                        "elapsed_s",
                    ):
                        if actual[field] != report[field]:
                            raise ValueError(
                                "expert cost differs from the required cache generation"
                            )
                    if (
                        not 0
                        <= actual["candidates"]
                        <= actual["candidate_budget"]
                        <= 128
                        or actual["coarse_candidates"] > 64
                        or actual["refined_candidates"] > 64
                    ):
                        raise ValueError(
                            "expert cache exceeded the registered candidate budget"
                        )
                    identity = str(path.resolve().relative_to(output.resolve()))
                    if identity not in unique:
                        unique[identity] = {
                            "cache": identity,
                            "instance_hash": actual["instance_hash"],
                            "cache_kind": name,
                            "candidate_executions": actual["candidates"],
                            "generation_s": actual["elapsed_s"],
                            "qualified_demonstration": actual[
                                "qualified_demonstration"
                            ],
                            "training_consumers": 0,
                            "unique_label_frames": 0,
                            "cost_scope": "unique_required_completed_cache_generation",
                        }
                    unique[identity]["training_consumers"] += 1
                    unique[identity]["unique_label_frames"] = max(
                        unique[identity]["unique_label_frames"],
                        report.get("samples", 0),
                    )
    return list(unique.values())
