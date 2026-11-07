"""The manual handoff packages frozen files without calling either simulator."""

import json
import re
from types import SimpleNamespace

import pytest

from ads.common.instances import file_hash
from ads.nuplan.handoff import PLANNERS, prepare_handoff
from ads.nuplan.prepare_data import WORKSPACE
from ads.tasks import POLICY_SUFFIX


def test_selected_policies_match_native_workspace():
    launcher = (WORKSPACE / "run_closed_loop.sh").read_text()
    planner_cases = launcher.split('case "$PLANNER" in', 1)[1].split("\nesac", 1)[0]
    native_algorithms = set(
        re.findall(r"^    ([a-z_0-9]+)\)", planner_cases, re.MULTILINE)
    )
    model_algorithms = {
        path.name
        for path in (WORKSPACE / "FITWMAMS_models/ads").iterdir()
        if path.is_dir()
    }
    assert set(POLICY_SUFFIX) == set(PLANNERS) == native_algorithms == model_algorithms
    assert native_algorithms == {"carl", "diffusion_planner"}


@pytest.mark.parametrize("algorithm", PLANNERS)
def test_manual_pair_has_shared_scenarios_native_weights_and_separate_metrics(
    algorithm, tmp_path, monkeypatch
):
    monkeypatch.setattr("ads.nuplan.handoff.REPO_ROOT", tmp_path)
    sources = {}
    for role in ("initial", "guided"):
        path = tmp_path / (role + ".source")
        path.write_text(role)
        sources[f"{algorithm}/20260926/{role}"] = {
            "path": path.name,
            "sha256": file_hash(path),
        }
    frozen = {"protocol_hash": "frozen-protocol", "policies": sources}
    campaign = SimpleNamespace(
        config={
            "methods": ["guided"],
            "seeds": [20260926],
            "manual_nuplan": {"output_dir": "separate_nuplan"},
        },
        verify_freeze=lambda: frozen,
    )
    exports = []

    def export(algorithm, source, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
        exports.append(destination)

    monkeypatch.setattr("ads.nuplan.handoff.export_policy", export)
    scenarios = tmp_path / "user_scenarios.csv"
    scenarios.write_text("log_name,scenario_token\nlog_a,token_a\n")
    paths = prepare_handoff(campaign, algorithm, "guided", scenarios)
    assert len(paths) == 1 and len(exports) == 2
    record = json.loads(paths[0].read_text())
    root = tmp_path / "separate_nuplan"
    for role, entry in record["policies"].items():
        assert file_hash(root / entry["native_path"]) == entry["native_sha256"]
        assert (root / record["native_metrics"][role]).is_dir()
    assert (
        root / record["common_scenarios"]["path"]
    ).read_bytes() == scenarios.read_bytes()
    assert (
        record["automatic_evaluation"]
        is record["included_in_highd_statistics"]
        is False
    )
    assert not record["used_for_model_selection"]
    assert prepare_handoff(campaign, algorithm, "guided", scenarios) == paths
    assert len(exports) == 2
    scenarios.write_text("changed scenario selection")
    with pytest.raises(ValueError, match="same frozen scenario"):
        prepare_handoff(campaign, algorithm, "guided", scenarios)
    assert len(exports) == 2
