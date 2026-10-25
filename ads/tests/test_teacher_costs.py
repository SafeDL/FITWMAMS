import pytest

from ads.common.instances import json_hash
from ads.common.reporting import write_json
from ads.common.teacher_costs import teacher_generation_costs


def test_teacher_costs_count_shared_generation_once_and_reject_wrong_cache(
    tmp_path, instance
):
    config = {
        "algorithms": ["plant2"],
        "seeds": [1],
        "methods": ["a", "b"],
        "rounds": 1,
        "repair": {"plant_normal_teacher_instances": 1},
    }
    report = {
        "instance_hash": instance.instance_hash,
        "teacher_config": {"budget": 128},
        "candidates": 124,
        "candidate_budget": 128,
        "coarse_candidates": 64,
        "refined_candidates": 60,
        "qualified_demonstration": True,
        "elapsed_s": 1.5,
    }
    key = json_hash(
        {"instance": instance.instance_hash, "teacher": report["teacher_config"]}
    )
    cache = tmp_path / "demonstrations/normal" / (key + ".json")
    write_json(cache, report)
    write_json(tmp_path / "plant2/1/baseline/training.json", {"baseline": True})
    for name in ["a/round_1", "b/round_1"]:
        directory = tmp_path / "plant2/1" / name
        write_json(
            directory / "training.json",
            {
                "baseline": False,
                "teacher": {
                    "reports": [
                        dict(
                            report, cache_path=str(cache), new_candidates=0, samples=20
                        )
                    ]
                },
            },
        )
        write_json(
            directory / "inputs/curriculum.json",
            {"categories": {"normal": [{"instance_hash": instance.instance_hash}]}},
        )
    rows = teacher_generation_costs(tmp_path, config)
    assert len(rows) == 1
    assert rows[0]["candidate_executions"] == 124
    assert rows[0]["training_consumers"] == 2
    # The generation may have been prefetched: zero new candidates per consumer
    # must never erase its actual, once-only expert cost.
    write_json(cache, dict(report, candidates=125))
    with pytest.raises(ValueError, match="cost differs"):
        teacher_generation_costs(tmp_path, config)
