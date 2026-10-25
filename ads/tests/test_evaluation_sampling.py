import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from ads.common.evaluation_sampling import (
    COHORTS,
    expected_final_roots,
    final_sampling_spec,
    final_source,
    verify_final_roots,
    verify_sampling_spec,
)


def protocol(kind="formal", count=2000):
    return {
        "settings": {
            "run_kind": kind,
            "master_seed": 20260926,
            "evaluation": {"instances": count},
            "output_dir": kind,
        },
        "assets": {"frozen_world": "unchanged"},
    }


def test_final_cohort_namespaces_are_distinct_and_sealed():
    value = protocol()
    assets = {
        "natural_world": {"natural": "flow"},
        "empirical_roots": [f"empirical_{i}" for i in range(2000)],
    }
    frozen = {
        "evaluation_sampling": final_sampling_spec(value, assets),
        "evaluation_assets": assets,
    }
    verify_sampling_spec(frozen, value)
    root_sets = [expected_final_roots(value, cohort, assets) for cohort in COHORTS]
    assert all(len(roots) == 2000 for roots in root_sets)
    assert len(set.union(*root_sets)) == 6000
    assert not root_sets[0].intersection(expected_final_roots(protocol("smoke", 4)))
    changed = deepcopy(frozen)
    changed["evaluation_sampling"]["cohorts"]["natural_generated"]["instances"] -= 1
    with pytest.raises(ValueError, match="sealed freeze"):
        verify_sampling_spec(changed, value)


def test_final_roots_and_source_purpose_are_checked():
    value = protocol(count=5)
    source = final_source(value)
    roots = sorted(expected_final_roots(value))
    assert source.prior_identity(0)[1] in roots
    instances = [
        SimpleNamespace(
            root_sample_id=root, purpose="final_test", source_kind="tail_generated"
        )
        for root in roots
    ]
    verify_final_roots(instances, value)
    instances[-1].root_sample_id = roots[0]
    with pytest.raises(ValueError, match="sampling roots"):
        verify_final_roots(instances, value)


def test_new_protocol_has_new_final_stream():
    before = protocol()
    after = deepcopy(before)
    after["settings"]["output_dir"] = "another_campaign"
    assert not expected_final_roots(before).intersection(expected_final_roots(after))


@pytest.mark.parametrize("interrupt", [False, True])
def test_iid_completed_cases_are_journaled_before_cohort_finishes(
    tmp_path, monkeypatch, interrupt
):
    from ads.risk_stratified_repair import Campaign

    campaign = Campaign.__new__(Campaign)
    campaign.output, campaign.device = tmp_path, "cpu"
    campaign.factory = campaign.evt = campaign.ledger = None
    campaign.protocol_hash = "frozen-protocol"
    (tmp_path / "protocol.json").write_text("{}")
    entry = {"sha256": "frozen-policy", "path": "synthetic-policy"}
    frozen = {
        "policies": {"dqn/20260926/initial": entry},
        "evaluation_assets": {},
    }
    worlds = [SimpleNamespace(instance_hash=name) for name in ("first", "second")]

    def evaluate(instance):
        if interrupt and instance.instance_hash == "second":
            raise RuntimeError("synthetic interruption")
        return SimpleNamespace(record={"instance_hash": instance.instance_hash})

    monkeypatch.setattr(
        "ads.risk_stratified_repair.verify_final_roots", lambda *args: None
    )
    monkeypatch.setattr(
        "ads.risk_stratified_repair.load_policy", lambda *args, **kwargs: object()
    )
    monkeypatch.setattr(
        "ads.risk_stratified_repair.WorldEvaluator", lambda *args, **kwargs: evaluate
    )
    kwargs = {"frozen": frozen, "instances": {"tail_generated": worlds}}
    if interrupt:
        with pytest.raises(RuntimeError, match="synthetic interruption"):
            campaign.evaluate_one("dqn/20260926/initial", **kwargs)
    else:
        campaign.evaluate_one("dqn/20260926/initial", **kwargs)
    directory = tmp_path / "evaluation/tail_generated/dqn/20260926"
    journal = next((directory / "initial.attempts").glob("*.jsonl"))
    rows = [json.loads(line) for line in journal.read_text().splitlines()]
    assert [row["record"]["instance_hash"] for row in rows] == (
        ["first"] if interrupt else ["first", "second"]
    )
    assert all(row["policy_sha256"] == entry["sha256"] for row in rows)
    assert all(row["protocol_hash"] == campaign.protocol_hash for row in rows)
    report = directory / "initial.json"
    assert report.exists() == (not interrupt)
    if not interrupt:
        saved = json.loads(report.read_text())
        assert saved["records"] == [row["record"] for row in rows]
        assert tmp_path / saved["execution_journal"] == journal
