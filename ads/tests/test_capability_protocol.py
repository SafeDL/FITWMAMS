"""Safety admission and early rollback in the single-cycle repair protocol."""

from ads.diffusion_planner import capability


def record(name, collision=False, progress=100.0):
    return {
        "event_id": name,
        "instance_hash": name,
        "root_sample_id": name,
        "assets_hash": "world",
        "policy_stream_id": name + "-stream",
        "collision": collision,
        "offroad": False,
        "stalled": False,
        "safety_failure": collision,
        "progress_m": progress,
    }


def validation(collision=True, natural_collision=False, native_loss=1.0):
    return {
        "records": {
            "tail_generated": [record("tail_a", collision), record("tail_b")],
            "natural_generated": [
                record("natural_a", natural_collision),
                record("natural_b"),
            ],
            "dangerous": [record("danger_a", collision), record("neighbour")],
        },
        "native": {
            "scene_ids": ["native"],
            "materials_sha256": "fixed",
            "overall": {
                "native_loss": native_loss,
                "ade_m": 2.0,
                "fde_m": 3.0,
                "function_drift": 0.1,
            },
        },
    }


def test_safety_improvement_is_not_vetoed_by_native_supervised_error():
    before = validation()
    improved = capability.candidate_decision(
        before, before, validation(collision=False)
    )
    assert improved["eligible"] and improved["continue"]
    assert not capability.candidate_decision(before, before, validation(False, True))[
        "continue"
    ]
    assert capability.candidate_decision(
        before, before, validation(False, native_loss=1.01)
    )["continue"]
    unchanged = capability.candidate_decision(before, before, before)
    assert unchanged["continue"] and not unchanged["eligible"]


def test_failed_first_phase_restores_initial_and_does_not_collect_queries(
    tmp_path, monkeypatch
):
    from ads.diffusion_planner import checkpoints

    monkeypatch.setattr(checkpoints, "MODEL_ROOT", tmp_path / "models")
    initial = tmp_path / "official/model.pth"
    initial.parent.mkdir()
    initial.write_bytes(b"official")
    initial.with_name("args.json").write_text("{}")
    loaded, updates = [], []

    class Model:
        repair_state = None

        def save(self, path):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"candidate")

    def load(algorithm, path, device):
        loaded.append(path)
        return Model()

    def train(model, banks, **kwargs):
        updates.append(kwargs["updates"])
        model.repair_state = {
            "gradient_updates": 250,
            "exposure": {
                "correction": {"minimum_critical_visits": 0, "unvisited_entries": 0}
            },
            "losses": [],
            "samplers": {},
            "training_elapsed_s": 1.0,
        }
        kwargs["checkpoint"](model, 250)
        return {"gradient_updates": 250, "cumulative_updates": 250}

    monkeypatch.setattr(capability, "official_path", lambda name: initial)
    monkeypatch.setattr(capability, "load_policy", load)
    monkeypatch.setattr(capability, "corrections", lambda *args: ([], []))
    monkeypatch.setattr(capability, "train", train)
    monkeypatch.setattr(capability, "validate", lambda *args: {})
    monkeypatch.setattr(
        capability,
        "candidate_decision",
        lambda *args: {"continue": False, "eligible": False},
    )
    monkeypatch.setattr(
        capability,
        "aggregate_queries",
        lambda *args: (_ for _ in ()).throw(
            AssertionError("Rejected phase must not query")
        ),
    )
    settings = {
        "master_seed": 20260926,
        "runtime": {"device": "cpu"},
        "repair": {
            "minimum_phase_updates": 250,
            "micro_batch_size": 8,
            "learning_rate": 1e-5,
            "reference_updates": 250,
            "ema_decay": 0.999,
            "retention_coefficient": 1.0,
        },
    }
    result = capability.repair_method(
        settings, tmp_path, {"protected_roots": []}, {}, "guided", [], [], {}
    )
    assert updates == [250]
    assert loaded[-1] == initial
    assert not result["accepted_update"]
    assert (tmp_path / "repairs/guided/policy.pth").read_bytes() == b"official"
