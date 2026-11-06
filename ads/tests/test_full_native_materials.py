"""The full-pool protocol processes reserves without replacing historical reports."""

from concurrent.futures import ThreadPoolExecutor

import pytest

from ads.common.reporting import write_json
from ads.diffusion_planner import prepare_data as module


def candidate(purpose, index):
    return {
        "log": f"{purpose}_{index}",
        "token": str(index),
        "purpose": purpose,
        "session": f"{purpose}_{index}",
        "city": "boston",
        "tags": [],
    }


def test_full_pool_processes_every_reserve_and_resumes(tmp_path, monkeypatch):
    manifest = {
        "scenes": {
            purpose: {
                "selected": [candidate(purpose, 0)],
                "reserve": [candidate(purpose, i) for i in (1, 2, 3)],
            }
            for purpose in module.COUNTS
        }
    }
    root = tmp_path / "sources"
    root.mkdir()
    manifest_path, config = root / "scenes.json", root / "args.json"
    write_json(manifest_path, manifest)
    write_json(config, {})
    full = root / "diffusion_planner_full"
    monkeypatch.setattr(module, "ROOT", root)
    monkeypatch.setattr(module, "FULL_ROOT", full)
    monkeypatch.setattr(module, "FULL_REPORT", full / "materials.json")
    monkeypatch.setattr(module, "MANIFEST", manifest_path)
    monkeypatch.setattr(module, "CONFIG", config)
    monkeypatch.setattr(module, "select_scenes", lambda: manifest)
    monkeypatch.setattr(module, "ProcessPoolExecutor", ThreadPoolExecutor)
    calls = []

    def encode(row, directory):
        calls.append((row["purpose"], row["token"]))
        identity = module.json_hash([row["log"], row["token"]])
        result = {"scene_id": identity, "source": row}
        if row["token"] == "1":
            return {**result, "excluded": "ValueError: No valid native route"}
        path = directory / (identity + ".npz")
        path.write_bytes(b"feature")
        return {**result, "path": str(path), "sha256": module.file_hash(path)}

    monkeypatch.setattr(module, "encode_scene", encode)
    report = module.prepare(workers=2)
    assert len(calls) == 8
    assert all(report["coverage"][p]["candidates"] == 4 for p in module.COUNTS)
    assert all(report["coverage"][p]["scenes"] == 3 for p in module.COUNTS)
    assert len(report["excluded"]) == 2
    assert not (root / "diffusion_planner/materials.json").exists()
    assert module.prepare(workers=2) == report
    assert len(calls) == 8


def test_reserve_session_leakage_is_rejected():
    a, b = candidate("replay", 1), candidate("development", 1)
    b["session"] = a["session"]
    manifest = {
        "scenes": {
            "replay": {"selected": [], "reserve": [a]},
            "development": {"selected": [b], "reserve": []},
        }
    }
    with pytest.raises(ValueError, match="sessions overlap"):
        module.all_candidates(manifest)
