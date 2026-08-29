from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import yaml

from IDM_subset.scripts.render_subset_playbacks import _diverse_tail_indices, _render_case
from IDM_subset.src.multi_world_runner import (
    build_comparison_report,
    build_explicit_subset_comparison,
    validate_comparison_contract,
)
from IDM_subset.src.world_model_registry import get_world_model, world_model_ids
from world_model.src.core.utils import file_sha256, load_json, save_json


def _config(tmp_path: Path, model_id: str) -> tuple[dict, Path]:
    common = tmp_path / "release.yaml"
    evt = tmp_path / "evt.json"
    idm = tmp_path / "idm.yaml"
    for path, content in (
        (common, "common-prior"),
        (evt, "evt"),
        (idm, "idm"),
    ):
        path.write_text(content, encoding="utf-8")
    paths = {
        "evt_model": str(evt),
        "idm_ego_config": str(idm),
        (
            "world_model_config"
            if model_id == "hierarchical"
            else "common_world_config"
        ): str(common),
    }
    config = {
        "paths": paths,
        "evaluation_scope": {
            "schema": "highd_follower_excluded_v1",
            "excluded_background_slots": ["same_rear"],
        },
        "simulation": {"steps": 149, "execution_backend": "local_highway_env"},
        "failure_event": {"return_period": 100},
        "subset_simulation": {
            "num_samples": 512,
            "p0": 0.1,
            "max_levels": 8,
            "pcn_beta": 0.2,
            "mcmc_steps": 3,
            "output_dir": str(tmp_path / model_id / "subset"),
        },
        "monte_carlo": {
            "num_samples": 2000,
            "output_dir": str(tmp_path / model_id / "monte_carlo"),
        },
    }
    return config, tmp_path / f"{model_id}.yaml"


def test_registry_exposes_both_world_models_and_aliases() -> None:
    assert world_model_ids() == ("hierarchical", "trafficbots")
    assert get_world_model("HiQR").model_id == "hierarchical"
    assert get_world_model("trafficbot").model_id == "trafficbots"
    with pytest.raises(ValueError, match="unknown IDM world model"):
        get_world_model("unknown")


def test_comparison_contract_accepts_model_specific_latent_spaces(tmp_path: Path) -> None:
    hierarchical, hierarchical_path = _config(tmp_path, "hierarchical")
    trafficbots, trafficbots_path = _config(tmp_path, "trafficbots")
    models = {
        "hierarchical": (
            get_world_model("hierarchical"),
            hierarchical,
            hierarchical_path,
        ),
        "trafficbots": (
            get_world_model("trafficbots"),
            trafficbots,
            trafficbots_path,
        ),
    }
    contract = validate_comparison_contract(models)
    assert contract["steps"] == 149
    assert contract["dt_s"] == 0.04
    assert contract["subset_num_samples"] == 512
    assert "model-specific latent priors" in contract["comparison_semantics"]


def test_comparison_contract_rejects_protocol_mismatch(tmp_path: Path) -> None:
    hierarchical, hierarchical_path = _config(tmp_path, "hierarchical")
    trafficbots, trafficbots_path = _config(tmp_path, "trafficbots")
    trafficbots = copy.deepcopy(trafficbots)
    trafficbots["simulation"]["steps"] = 100
    models = {
        "hierarchical": (
            get_world_model("hierarchical"),
            hierarchical,
            hierarchical_path,
        ),
        "trafficbots": (
            get_world_model("trafficbots"),
            trafficbots,
            trafficbots_path,
        ),
    }
    with pytest.raises(ValueError, match="incomparable IDM evaluation configs: steps"):
        validate_comparison_contract(models)


def test_comparison_contract_rejects_shared_output_directory(tmp_path: Path) -> None:
    hierarchical, hierarchical_path = _config(tmp_path, "hierarchical")
    trafficbots, trafficbots_path = _config(tmp_path, "trafficbots")
    trafficbots["subset_simulation"]["output_dir"] = hierarchical[
        "subset_simulation"
    ]["output_dir"]
    models = {
        "hierarchical": (
            get_world_model("hierarchical"),
            hierarchical,
            hierarchical_path,
        ),
        "trafficbots": (
            get_world_model("trafficbots"),
            trafficbots,
            trafficbots_path,
        ),
    }
    with pytest.raises(ValueError, match="must not share result directories"):
        validate_comparison_contract(models)


def test_model_specific_playback_keeps_common_visual_contract(tmp_path: Path) -> None:
    states = np.zeros((3, 7, 6), dtype=np.float32)
    states[:, 1, 0] = np.asarray([10.0, 5.0, 0.0], np.float32)
    valid = np.zeros(7, dtype=bool)
    valid[:2] = True
    path = tmp_path / "trafficbots_case.gif"
    result = _render_case(
        path=path,
        case={
            "case_id": "trafficbots_subset_final_0001",
            "world_exogenous_state": "case.npz",
            "event_risk": 1.0,
            "evt_score": 5.0,
            "collision": True,
        },
        states=states,
        valid=valid,
        frame_stride=1,
        model_name="TrafficBots V1.5-HighD",
        background_label="TrafficBots background",
        background_color="#6a3d9a",
        background_color_name="purple",
    )
    assert path.is_file()
    assert result["first_collision_frame"] == 2
    assert result["playback_frames"] == 3


def test_diverse_tail_selection_covers_collision_and_clearance_mechanisms() -> None:
    score = np.asarray((5.1, 5.2, 5.3, 5.4, 5.5, 5.6), np.float64)
    risk = np.asarray((0.3, 0.8, 0.4, 0.7, 0.5, 0.9), np.float64)
    collision = np.asarray((False, True, False, False, True, False))
    gap = np.asarray((0.1, 0.0, 2.0, 8.0, 0.0, 20.0), np.float64)
    selected = _diverse_tail_indices(
        score, risk, collision, gap, failure_threshold=5.0, count=5
    )
    assert len(selected) == 5
    assert len(np.unique(selected)) == 5
    assert collision[selected].any()
    assert np.max(gap[selected]) >= 8.0


def test_comparison_report_writes_checked_json_csv_and_figure(tmp_path: Path) -> None:
    suite_models = []
    for index, model_id in enumerate(("hierarchical", "trafficbots")):
        config, config_path = _config(tmp_path, model_id)
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        suite_models.append({"id": model_id, "config": str(config_path)})
        provenance = {
            "execution_backend": "local_highway_env",
            "flow_checkpoint_sha256": "shared-flow-checkpoint",
            "evt_model_sha256": file_sha256(config["paths"]["evt_model"]),
            "idm_ego_config_sha256": file_sha256(
                config["paths"]["idm_ego_config"]
            ),
            (
                "world_model_config_sha256"
                if model_id == "hierarchical"
                else "common_world_config_sha256"
            ): file_sha256(
                config["paths"][
                    "world_model_config"
                    if model_id == "hierarchical"
                    else "common_world_config"
                ]
            ),
        }
        common = {
            "world_model_id": model_id,
            "formal": True,
            "probability": 0.01 + index * 0.01,
            "failure_event": {
                "return_period": 100,
                "event_risk_threshold": 0.8,
                "evt_score_threshold": 4.6,
            },
                "evaluation_contract": {
                    "steps": 149,
                    "population_scope": {
                        "schema": "highd_follower_excluded_v1",
                        "excluded_background_slots": ["same_rear"],
                        "training_population_modified": False,
                        "semantics": (
                            "same_rear is absent before model inference and simulation; "
                            "metrics, risk, collision detection and visualization inherit "
                            "the same mask"
                        ),
                    },
                },
            "mutation_kernel": {"pcn_beta": 0.2, "mcmc_steps": 3},
            "uncertainty": {
                "probability_standard_error": 0.001,
                "probability_ci95_lower": 0.008 + index * 0.01,
                "probability_ci95_upper": 0.012 + index * 0.01,
            },
            "provenance": provenance,
        }
        subset_dir = Path(config["subset_simulation"]["output_dir"])
        save_json(
            {
                **common,
                "schema": (
                    "highway_env_idm_subset_simulation"
                    if model_id == "hierarchical"
                    else "trafficbots_highway_env_idm_subset_simulation"
                ),
                "dimensions": {"response_steps": 149},
                "simulation_counts": {"world_evaluations": 2048},
                "num_levels": 2,
                "final_failure_fraction": 0.2,
                "level_statistics": [
                    {
                        "collision_fraction": 0.1,
                        "failure_fraction": 0.02,
                        "num_samples": 512,
                        "level": 0,
                    }
                ],
            },
            subset_dir / "world_subset_summary.json",
        )
        monte_dir = Path(config["monte_carlo"]["output_dir"])
        save_json(
            {
                **common,
                "simulation_counts": {"world_evaluations": 2000},
                "failure_count": 10,
                "collision_fraction": 0.02,
                "numerical_valid_fraction": 1.0,
                "evt_score_summary": {"mean": 2.0, "p95": 3.0, "max": 5.0},
            },
            monte_dir / "world_monte_carlo_summary.json",
        )
    suite = {
        "world_models": suite_models,
        "output": {"comparison_dir": str(tmp_path / "comparison")},
    }
    report_path = build_comparison_report(suite, tmp_path)
    report = load_json(report_path)
    assert report["schema"] == "idm_multi_world_model_comparison_v1"
    assert len(report["records"]) == 4
    assert Path(report["artifacts"]["table_csv"]).is_file()
    assert Path(report["artifacts"]["comparison_figure"]).is_file()
    explicit = build_explicit_subset_comparison(
        tmp_path / "hierarchical/subset/world_subset_summary.json",
        tmp_path / "trafficbots/subset/world_subset_summary.json",
        tmp_path / "explicit_comparison",
    )
    explicit_report = load_json(explicit)
    assert explicit_report["publishable"] is True
    assert explicit_report["effect"]["trafficbots_over_hierarchical"] == 2.0
