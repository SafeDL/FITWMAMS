"""Complete frozen-Test factual regression, independent of NPC model selection."""

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
import time

import numpy as np
import pandas as pd
import torch

from ..core.utils import (
    load_json,
    save_json,
)
from ..data import (
    artifact_identity,
    prepare_experiment_data,
)
from ..shared_evaluation import (
    load_shared_hiqr,
    scene_randomness,
    highway_rollout,
    factual_records,
    save_error_samples,
    summarize_records,
)
from ..stochastic_drivers.idm_population import sample_theta
from ..stochastic_drivers.online import (
    OnlineMAIDMController,
    calibrated_mobil_parameters,
)
from ..stochastic_drivers.lane_trajectory import MODEL_PATH as LANE_PATH
from ..planner import frozen_diffusion_plans

ROOT = Path(__file__).resolve().parents[3]
_CONTEXT = None


def _initialize(config, population, fit, plans_path):
    global _CONTEXT
    torch.set_num_threads(1)
    experiment = prepare_experiment_data(config, ROOT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_shared_hiqr(config, experiment, device)
    with np.load(plans_path) as saved:
        plans = saved["positions"]
    _CONTEXT = config, experiment, model, population, fit, plans


def _batch(bounds):
    begin, end = bounds
    config, experiment, model, population, fit, plans = _CONTEXT
    rows = np.asarray(experiment.test_rows[begin:end], int)
    arrays = experiment.bundle.arrays
    ids = np.asarray(arrays["sequence_id"])[rows].astype(str)
    randomness = scene_randomness(ids, seed=20261002, frames=149)
    controller = OnlineMAIDMController(
        sample_theta(population, randomness.driver_latent),
        maneuver_latent=randomness.maneuver_latent,
        **calibrated_mobil_parameters(fit["mobil"]),
    )
    started = time.monotonic()
    trace = highway_rollout(
        model,
        arrays,
        rows,
        plans[begin:end],
        randomness,
        controller=controller,
        recovery=True,
    )
    records, errors, jerks = factual_records(
        rows,
        np.asarray(arrays["agent_states"][rows]),
        np.asarray(arrays["agent_valid"][rows]),
        trace,
    )
    for record, identifier in zip(records, ids):
        record.update(
            sequence_id=identifier, recording_id=int(identifier.split("_")[1]), repeat=0
        )
    return (
        begin,
        {
            "status": "complete",
            "rows": rows.tolist(),
            "sequence_ids": ids.tolist(),
            "elapsed_seconds": time.monotonic() - started,
            "road_steps": int(trace["road_step_calls"].sum()),
            "records": records,
        },
        {"errors": errors, "jerks": jerks},
    )


def consolidate_factual_batches(destination):
    """Keep complete-Test evidence, then remove redundant resume shards."""
    destination = Path(destination)
    protocol = load_json(destination / "protocol.json")
    batches = destination / "batches"
    errors, jerks, execution, rows, identifiers = [], [], [], [], []
    starts = range(0, len(protocol["rows"]), protocol["batch_size"])
    for begin in starts:
        result = load_json(batches / f"{begin:06d}.json")
        if result["status"] != "complete":
            raise ValueError(f"Incomplete factual batch: {begin}")
        rows.extend(result["rows"])
        identifiers.extend(result["sequence_ids"])
        execution.append(
            {
                "begin": begin,
                "sequences": len(result["rows"]),
                "road_steps": result["road_steps"],
                "elapsed_seconds": result["elapsed_seconds"],
                "status": result["status"],
            }
        )
        with np.load(batches / f"{begin:06d}.npz") as numeric:
            errors.append(numeric["errors"])
            jerks.append(numeric["jerks"])
    records = pd.read_csv(destination / "scene_metrics.csv")
    if (
        rows != protocol["rows"]
        or identifiers != protocol["sequence_ids"]
        or rows != records.row.tolist()
        or identifiers != records.sequence_id.tolist()
    ):
        raise ValueError("Factual batches do not match the full Test scene manifest")
    save_error_samples(destination, errors, jerks)
    save_json(execution, destination / "execution.json")
    for begin in starts:
        for suffix in (".json", ".npz"):
            (batches / f"{begin:06d}{suffix}").unlink()
    # Failed-run logs, if any, remain available rather than being silently removed.
    if not any(batches.iterdir()):
        batches.rmdir()


def evaluate_factual_regression(config, output):
    from .search import load_candidate

    output = Path(output)
    fit = load_json(output / "policy_calibration/selected_model.json")
    population, _ = load_candidate(
        output / "policy_calibration", fit["selected_candidate"]
    )
    destination = ROOT / "results/hierarchical_world_model/factual_test_candidate"
    experiment = prepare_experiment_data(config, ROOT)
    rows = np.asarray(experiment.test_rows, int)
    ids = np.asarray(experiment.bundle.arrays["sequence_id"])[rows].astype(str)
    settings = config["npc_response"]
    batch_size = settings["world_batch_size"]
    protocol = {
        "rows": rows.tolist(),
        "sequence_ids": ids.tolist(),
        "batch_size": batch_size,
        "mobil": fit["mobil"],
        "driver_population": fit["driver_population"],
        "selected_candidate": fit["selected_candidate"],
        "seed": 20261002,
        "response": artifact_identity(config["paths"]["evaluation_checkpoint"]),
        "diffusion": artifact_identity(config["paths"]["diffusion_checkpoint"]),
        "lane": artifact_identity(LANE_PATH),
        "npc_lane_model": fit["mobil"].get("lane_model"),
        "npc_fit_split": fit["fit_split"],
        "horizon_frames": 149,
        "response_activation": "longitudinal_override_requires_longitudinal_component",
    }
    published = ROOT / "results/hierarchical_world_model/factual_test"
    if (published / "protocol.json").exists() and load_json(
        published / "protocol.json"
    ) == protocol:
        return load_json(published / "report.json")
    destination.mkdir(parents=True, exist_ok=True)
    manifest = destination / "protocol.json"
    if manifest.exists() and load_json(manifest) != protocol:
        raise ValueError(
            f"Factual regression parameters changed; archive {destination.name} before rerunning"
        )
    save_json(protocol, manifest)
    plan_path = (
        ROOT / "results/hierarchical_world_model/diffusion_plans/test/positions.npz"
    )
    frozen_diffusion_plans(
        experiment.bundle,
        rows,
        checkpoint=config["paths"]["diffusion_checkpoint"],
        output_dir=plan_path.parent,
        device=torch.device("cuda"),
        batch_size=512,
        ddim_steps=20,
        experiment_scope="full",
    )
    batches = destination / "batches"
    batches.mkdir(exist_ok=True)
    pending = [
        (begin, min(begin + batch_size, len(rows)))
        for begin in range(0, len(rows), batch_size)
        if not (batches / f"{begin:06d}.json").exists()
    ]
    with ProcessPoolExecutor(
        max_workers=settings["world_workers"],
        mp_context=get_context("spawn"),
        initializer=_initialize,
        initargs=(config, population, fit, plan_path),
    ) as pool:
        futures = {pool.submit(_batch, bounds): bounds for bounds in pending}
        for future in as_completed(futures):
            bounds = futures[future]
            try:
                begin, metadata, numeric = future.result()
            except Exception as error:
                save_json(
                    {
                        "status": "failed",
                        "range": list(bounds),
                        "reason": f"{type(error).__name__}: {error}",
                    },
                    batches / f"{bounds[0]:06d}_failed.json",
                )
                raise
            np.savez_compressed(batches / f"{begin:06d}.npz", **numeric)
            save_json(metadata, batches / f"{begin:06d}.json")
            print("Factual Test batch", begin, "/", len(rows), flush=True)
    records, errors, jerks, metadata = [], [], [], []
    for begin in range(0, len(rows), batch_size):
        result = load_json(batches / f"{begin:06d}.json")
        metadata.append(result)
        records.extend(result["records"])
        with np.load(batches / f"{begin:06d}.npz") as numeric:
            errors.append(numeric["errors"])
            jerks.append(numeric["jerks"])
    frame = pd.DataFrame(records)
    if not np.array_equal(frame.sequence_id.to_numpy(), ids) or not np.array_equal(
        frame.row.to_numpy(), rows
    ):
        raise ValueError(
            "Factual Test scene identities or order differ from the complete Test manifest"
        )
    if not frame.finite.all():
        raise ValueError(
            "Nonfinite Test execution is recorded and cannot be published as complete"
        )
    frame.to_csv(destination / "scene_metrics.csv", index=False)
    metrics = summarize_records(records, errors, jerks)
    from hierarchical_world_model.scripts.evaluate import plot_validation

    intervals = plot_validation({"rules_recovery": records}, "test", destination)
    report = {
        "status": "frozen_test_factual_complete_not_world_acceptance",
        "split": "test",
        "sequences_requested": len(rows),
        "sequences_completed": len(frame),
        "fit_split": "train",
        "selection_split": "val",
        "test_used_for_selection": False,
        "response_checkpoint": protocol["response"]["path"],
        "response_checkpoint_mtime_ns": protocol["response"]["mtime_ns"],
        "diffusion": protocol["diffusion"],
        "diffusion_checkpoint": protocol["diffusion"]["path"],
        "lane_trajectory_model": protocol["lane"]["path"],
        "lane_trajectory_model_mtime_ns": protocol["lane"]["mtime_ns"],
        "mobil_parameters": fit["mobil"],
        "mobil_source": str(output / "policy_calibration/selected_model.json"),
        "driver_parameter_source": fit["driver_parameter_source"],
        "npc_model_source_artifacts": [
            artifact_identity(fit["driver_parameter_source"]),
            artifact_identity(
                output / "policy_calibration" / (fit["selected_candidate"] + ".json")
            ),
        ],
        "driver_population": fit["driver_population"],
        "npc_lane_model": fit["mobil"].get("lane_model"),
        "npc_fit_split": fit["fit_split"],
        "reference_plan_cache": str(plan_path),
        "source": config["paths"]["sequence_cache_dir"],
        "arms": ["rules_recovery"],
        "factual": {"rules_recovery": metrics},
        "recording_bootstrap_statistics": intervals,
        "DDIM_steps": 20,
        "evaluation_role": "Complete Test conditional factual execution; independent publication regression",
        "inputs": "Observed history, physical C0, K_GT through frozen Diffusion, true dimensions and roads",
        "backend": "highway_env.road.road.Road.step",
        "actual_road_steps": sum(item["road_steps"] for item in metadata),
        "measured_compute_seconds": sum(item["elapsed_seconds"] for item in metadata),
        "npc_calibration": {
            "selected_candidate": fit["selected_candidate"],
            "raw_data_identity": fit["raw_data_identity"],
            "selection_report": fit["selection_report"],
        },
        "source_artifacts": [
            protocol[name] for name in ("response", "diffusion", "lane")
        ],
    }
    save_json(report, destination / "report.json")
    checks = {
        "complete_test_identity": np.array_equal(frame.sequence_id.to_numpy(), ids)
        and np.array_equal(frame.row.to_numpy(), rows),
        "complete_road_steps": report["actual_road_steps"] == 149 * len(ids),
        "all_batches_complete": all(item["status"] == "complete" for item in metadata),
        "finite_execution": frame.finite.all(),
        "ADE_matches_records": np.isclose(
            metrics["ADE_m"],
            frame.distance_sum_m.sum() / frame.actor_frames.sum(),
            atol=1e-9,
            rtol=0,
        ),
        "FDE_matches_records": np.isclose(
            metrics["FDE_m"],
            frame.final_distance_sum_m.sum() / frame.final_actors.sum(),
            atol=1e-9,
            rtol=0,
        ),
        "candidate_population_matches": all(
            np.array_equal(population[name], fit["driver_population"][name])
            for name in ("log_mean", "log_cov")
        ),
        "upstream_weights_unchanged": all(
            identity == artifact_identity(config["paths"][name])
            for name, identity in load_json(
                output / "policy_calibration/upstream_sources.json"
            ).items()
        ),
    }
    checks = {name: bool(value) for name, value in checks.items()}
    verification = {"passed": all(checks.values()), "checks": checks}
    save_json(verification, destination / "verification.json")
    if not verification["passed"]:
        raise ValueError(f"Factual regression verification failed: {destination}")
    consolidate_factual_batches(destination)
    return report


def publish_response(config, output):
    """Publish only after frozen standalone evaluation and complete Test execution."""
    import shutil
    from .search import load_candidate
    from .calibration import (
        MOBIL_PATH,
        compact_population,
    )
    from ..stochastic_drivers.idm_population import MODEL_PATH

    output = Path(output)
    if not load_json(output / "verification.json")["passed"]:
        raise ValueError("Standalone NPC statistics have not passed verification")
    pending = ROOT / "results/hierarchical_world_model/factual_test_candidate"
    report = load_json(pending / "report.json")
    if report["status"] != "frozen_test_factual_complete_not_world_acceptance":
        raise ValueError(
            "Complete factual Test regression is required before publication"
        )
    expected = load_json(output / "policy_calibration/upstream_sources.json")
    if any(
        value != artifact_identity(config["paths"][name])
        for name, value in expected.items()
    ):
        raise ValueError(
            "Flow, Diffusion or HiQR weights changed during NPC calibration"
        )
    fit = load_json(output / "policy_calibration/selected_model.json")
    population, _ = load_candidate(
        output / "policy_calibration", fit["selected_candidate"]
    )
    values = compact_population(population)
    with np.load(MODEL_PATH) as current:
        identical = set(current.files) == set(values) and all(
            np.array_equal(current[name], values[name]) for name in values
        )
    if not identical:
        np.savez_compressed(MODEL_PATH, **values)
    fit["status"] = "npc_response_verified"
    fit["driver_parameter_source"] = str(MODEL_PATH)
    save_json(fit, MOBIL_PATH)
    if load_json(LANE_PATH) != fit["mobil"]["lane_model"]:
        save_json(fit["mobil"]["lane_model"], LANE_PATH)
    archive = ROOT / "archive/hierarchical_world_model/npc_population_environment"
    destination = ROOT / "results/hierarchical_world_model/factual_test"
    if destination.exists():
        shutil.move(str(destination), str(archive / "previous_test"))
    shutil.move(str(pending), str(destination))
    from hierarchical_world_model.scripts.publish import main as publish

    publish()
    from hierarchical_world_model.scripts.verify import main as verify

    verify()
    save_json(
        {
            "status": "published",
            "selected_candidate": fit["selected_candidate"],
            "test_report": str(destination / "report.json"),
            "upstream_weights_unchanged": True,
        },
        output / "publication.json",
    )
    return report
