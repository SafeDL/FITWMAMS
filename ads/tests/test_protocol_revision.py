import json
from copy import deepcopy

import pytest

from ads.common.instances import file_hash, json_hash
from ads.risk_stratified_repair import Campaign, registered_protocol


def revision_case():
    before = {"ads/dtpp/policy.py": "old", "ads/search/engine.py": "frozen"}
    after = dict(before, **{"ads/dtpp/policy.py": "corrected"})
    saved = {
        "settings": {"seed": 20260926},
        "assets": {"world": "frozen"},
        "ads_implementation": json_hash(before),
    }
    current = dict(saved, ads_implementation=json_hash(after))
    revision = {
        "base_files": before,
        "revised_files": after,
        "changes": {"ads/dtpp/policy.py": {"before": "old", "after": "corrected"}},
    }
    return saved, current, revision


def test_registered_correction_preserves_original_protocol_and_sampling_identity():
    saved, current, revision = revision_case()
    assert registered_protocol(current, saved, revision) is saved


@pytest.mark.parametrize("field", ["settings", "assets"])
def test_code_revision_never_allows_configuration_or_world_changes(field):
    saved, current, revision = revision_case()
    current = deepcopy(current)
    current[field]["unexpected"] = "change"
    with pytest.raises(ValueError, match="configuration/assets"):
        registered_protocol(current, saved, revision)


def test_unregistered_code_or_incomplete_change_manifest_is_rejected():
    saved, current, revision = revision_case()
    with pytest.raises(ValueError, match="missing"):
        registered_protocol(current, saved, None)
    current["ads_implementation"] = "unregistered"
    with pytest.raises(ValueError, match="registered correction"):
        registered_protocol(current, saved, revision)
    saved, current, revision = revision_case()
    revision["changes"] = {}
    with pytest.raises(ValueError, match="registered correction"):
        registered_protocol(current, saved, revision)


def test_revision_cannot_change_after_the_policy_freeze(tmp_path):
    revision = tmp_path / "implementation_revision.json"
    revision.write_text("original correction")
    frozen = {
        "protocol_hash": "fixed-protocol",
        "implementation_revision_sha256": file_hash(revision),
    }
    (tmp_path / "frozen_policies.json").write_text(json.dumps(frozen))
    revision.write_text("changed correction")
    campaign = Campaign.__new__(Campaign)
    campaign.output, campaign.protocol_hash = tmp_path, "fixed-protocol"
    with pytest.raises(ValueError, match="correction changed after policy freeze"):
        campaign.verify_freeze()
