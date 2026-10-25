"""Evaluate the selected NPC policy's behavior and autonomous traffic statistics."""

import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from hierarchical_world_model.src.config import load_protocol_config
from hierarchical_world_model.src.core.utils import load_json, save_json
from hierarchical_world_model.src.npc_experiments.animations import (
    render_autonomous_examples,
)
from hierarchical_world_model.src.npc_experiments.calibration import MOBIL_PATH
from hierarchical_world_model.src.npc_experiments.evaluation import (
    closed_evaluation,
    conditional_evaluation,
    stage_models,
    verify_npc_results,
)
from hierarchical_world_model.src.npc_experiments.figures import plot_npc_results
from hierarchical_world_model.src.stochastic_drivers.idm_population import MODEL_PATH
from hierarchical_world_model.src.stochastic_drivers.lane_trajectory import (
    MODEL_PATH as LANE_PATH,
)


def main():
    torch.set_num_threads(1)
    started = time.monotonic()
    config = load_protocol_config(ROOT / "hierarchical_world_model/config.yaml")
    settings = config["npc_response"]
    output = ROOT / settings["output_dir"]
    conditional_evaluation(output, settings)
    closed = closed_evaluation(output, settings)
    plot_npc_results(output)
    verified = verify_npc_results(output, settings)
    if not verified["passed"]:
        raise RuntimeError(
            f"NPC statistics verification failed: {output / 'verification.json'}"
        )
    _, models = stage_models(output)
    candidate, population, mobil = models["reference"]
    published = load_json(MOBIL_PATH)
    with np.load(MODEL_PATH) as current:
        unchanged = all(
            np.array_equal(current[name], population[name])
            for name in ("log_mean", "log_cov")
        )
    unchanged = (
        unchanged
        and {key: value for key, value in mobil.items() if key != "lane_model"}
        == {
            key: value
            for key, value in published["mobil"].items()
            if key != "lane_model"
        }
        and mobil["lane_model"] == load_json(LANE_PATH)
    )
    if not unchanged:
        from hierarchical_world_model.src.npc_experiments.publication import publish_npc

        publish_npc(config, output)
    movies = render_autonomous_examples(output)
    report = {
        "status": "complete",
        "published_model": load_json(MOBIL_PATH)["selected_candidate"],
        "selected_candidate": candidate,
        "fit_split": "train",
        "selection_split": "val",
        "recordings_evaluated": 60,
        "standalone_runs": closed["completed_runs"],
        "actual_road_steps": closed["actual_road_steps"],
        "animation_road_steps": movies["actual_additional_road_steps"],
        "verification": str(output / "verification.json"),
        "invocation_elapsed_seconds": time.monotonic() - started,
        "measured_simulation_compute_seconds": closed["measured_compute_seconds"],
    }
    save_json(report, output / "report.json")
    save_json(
        {
            "status": "current_model_verified",
            "model": load_json(MOBIL_PATH)["selected_candidate"],
            "fit_split": load_json(MOBIL_PATH)["fit_split"],
            "selection_evidence": load_json(MOBIL_PATH)["selection_report"],
            "verification": str(output / "verification.json"),
        },
        output / "publication.json",
    )
    print("Selected NPC policy evaluation complete", report, flush=True)


if __name__ == "__main__":
    main()
