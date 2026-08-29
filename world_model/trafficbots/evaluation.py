"""Fair highD factual, stochastic, oracle, and paired-intervention evaluation."""
from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from torch.utils.data import Subset

from hierarchical_world_model.src.calibration import evaluation_response_calibration
from world_model.src.core.dynamics import KinematicTrafficDynamics
from world_model.src.core.highd_metrics import (
    distribution_metrics, ego_replay_gate, ego_replay_metrics, factual_metrics,
    intervention_dose_response, intervention_metrics, semantic_cutin_agents,
    temporal_factual_metrics,
)
from world_model.src.core.utils import file_sha256, save_json
from world_model.src.core.evaluation_scope import evaluation_scope_contract

from .data import TrafficBotsHighDDataset, make_loader
from .module import HighDTrafficBotsModule
from .rollout import TrafficBotsHighDRollout, logged_ego_controls


def load_checkpoint(config: dict[str, Any], checkpoint: str | Path) -> HighDTrafficBotsModule:
    model = HighDTrafficBotsModule(config)
    payload = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(payload.get("state_dict", payload))
    return model.eval()


def _append(items: list[np.ndarray], value: torch.Tensor) -> None:
    items.append(value.detach().cpu().numpy())


def _factual_strata(generated: np.ndarray, target: np.ndarray, active: np.ndarray, evt_tail: np.ndarray, semantic_cutin: np.ndarray) -> dict[str, dict[str, float]]:
    strata = {"all_natural": np.ones(len(generated), bool), "evt_labelled": np.asarray(evt_tail, bool), "semantic_cutin": np.asarray(semantic_cutin, bool)}
    report: dict[str, dict[str, float]] = {}
    for name, selected in strata.items():
        if not selected.any():
            raise RuntimeError(f"held-out factual stratum is empty: {name}")
        report[name] = factual_metrics(generated[selected], target[selected], active[selected])
    return report


def _rollout_object(items: list[SimpleNamespace]) -> SimpleNamespace:
    return SimpleNamespace(states=np.concatenate([item.states for item in items]), background_actions=np.concatenate([item.background_actions for item in items]))


@torch.no_grad()
def evaluate(config: dict[str, Any], checkpoint: str | Path, *, maximum: int = 0) -> dict[str, Any]:
    """Use the hierarchy's full/fixed stochastic/fixed intervention cohorts."""
    evaluation = config["evaluation"]
    seed = int(evaluation.get("seed", config["experiment"]["seed"]))
    np.random.seed(seed); torch.manual_seed(seed)
    dataset = TrafficBotsHighDDataset(
        config["paths"]["sequence_cache_dir"],
        str(evaluation.get("split", "test")),
        seed=seed,
        maximum=maximum,
        evaluation_scope=True,
    )
    full_test = TrafficBotsHighDDataset(
        config["paths"]["sequence_cache_dir"],
        "test",
        seed=seed,
        evaluation_scope=True,
    )
    batch_size = int(evaluation["batch_size"])
    module = load_checkpoint(config, checkpoint)
    if torch.cuda.is_available(): module = module.cuda()
    runner = TrafficBotsHighDRollout(module)

    deterministic: list[np.ndarray] = []; oracle: list[np.ndarray] = []
    targets: list[np.ndarray] = []; active: list[np.ndarray] = []
    evt_tail: list[np.ndarray] = []; semantic: list[np.ndarray] = []
    for batch in make_loader(dataset, batch_size=batch_size, shuffle=False):
        canonical, valid = batch["canonical/states"], batch["canonical/valid"]
        factual, diagnosed = runner.run(batch, deterministic=True), runner.run(batch, deterministic=True, oracle=True)
        _append(deterministic, factual.states); _append(oracle, diagnosed.states)
        _append(targets, canonical[:, 1:]); _append(active, valid[:, 0, 1:]); _append(evt_tail, batch["is_evt_tail"])
        semantic.append(semantic_cutin_agents(canonical.numpy(), valid.numpy()).any(1))
    generated, target, mask = np.concatenate(deterministic), np.concatenate(targets), np.concatenate(active)
    drift = ego_replay_metrics(generated, target)
    output = Path(config["paths"]["output_dir"]); output.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"evaluation_schema_version": 3, "evaluation_scope": evaluation_scope_contract(), "test_sequences": len(dataset), "ego_replay_drift": drift, "ego_replay_gate_passed": ego_replay_gate(drift, evaluation["ego_replay_gate"])}
    if not report["ego_replay_gate_passed"]:
        save_json(report, output / "ego_replay_drift.json")
        raise RuntimeError("external ego replay drift gate failed; do not publish factual comparison")
    deterministic_report = {"factual": factual_metrics(generated, target, mask), "temporal": temporal_factual_metrics(generated, target, mask), "event_strata": _factual_strata(generated, target, mask, np.concatenate(evt_tail), np.concatenate(semantic))}

    stochastic_count = min(int(evaluation.get("stochasticity_subset_sequences", 1024)), len(dataset))
    stochastic: list[list[SimpleNamespace]] = [[] for _ in range(int(evaluation["stochastic_samples"]))]
    initial: list[np.ndarray] = []; stochastic_target: list[np.ndarray] = []; stochastic_active: list[np.ndarray] = []; highd_actions: list[np.ndarray] = []
    for batch_index, batch in enumerate(make_loader(Subset(dataset, range(stochastic_count)), batch_size=batch_size, shuffle=False)):
        canonical, valid = batch["canonical/states"], batch["canonical/valid"]
        _append(initial, canonical[:, 0]); _append(stochastic_target, canonical[:, 1:]); _append(stochastic_active, valid[:, 0, 1:]); _append(highd_actions, batch["canonical/actions_highd"])
        for sample, values in enumerate(stochastic):
            torch.manual_seed(seed + 10_000 * batch_index + sample)
            rollout = runner.run(batch, deterministic=False)
            values.append(SimpleNamespace(states=rollout.states.cpu().numpy(), background_actions=rollout.background_actions.cpu().numpy()))
    initial_state, stochastic_target_array, stochastic_mask, real_highd = np.concatenate(initial), np.concatenate(stochastic_target), np.concatenate(stochastic_active), np.concatenate(highd_actions)
    source = np.concatenate((initial_state[:, None, 1:], stochastic_target_array[:, :-1, 1:]), 1)
    real_actions = KinematicTrafficDynamics.controls_from_highd_actions(torch.from_numpy(real_highd), torch.from_numpy(source.copy())).numpy()
    stochastic_report = distribution_metrics([_rollout_object(values) for values in stochastic], initial_state, stochastic_target_array, real_actions, real_highd, stochastic_mask)

    intervention_count = min(int(evaluation.get("intervention_subset_sequences", 512)), len(dataset))
    _, natural_calibration = evaluation_response_calibration(
        full_test.arrays,
        full_test.rows,
        minimum_events=30,
        evaluation_scope=True,
    )
    doses = {"brake": (1.5, 2.25, 3.0), "accelerate": (1.0, 1.5, 2.0), "left": (.08, .12, .16)}
    baseline: list[SimpleNamespace] = []; treatments: dict[str, dict[float, list[SimpleNamespace]]] = {kind: {dose: [] for dose in values} for kind, values in doses.items()}
    intervention_initial: list[np.ndarray] = []; intervention_active: list[np.ndarray] = []
    crn_latents: list[np.ndarray] = []; crn_destinations: list[np.ndarray] = []; crn_ids: list[str] = []; crn_seeds: list[int] = []
    for batch_index, batch in enumerate(make_loader(Subset(dataset, range(intervention_count)), batch_size=batch_size, shuffle=False)):
        canonical, valid = batch["canonical/states"], batch["canonical/valid"]
        controls = logged_ego_controls(canonical, valid)
        torch.manual_seed(seed + 1_000_000 + batch_index)
        factual = runner.run(batch, deterministic=False, ego_controls=controls)
        baseline.append(SimpleNamespace(states=factual.states.cpu().numpy(), background_actions=factual.background_actions.cpu().numpy()))
        _append(intervention_initial, batch["canonical/full_states"]); _append(intervention_active, valid[:, 0, 1:])
        crn_latents.append(factual.latent_sample.cpu().numpy()); crn_destinations.append(factual.destination_sample.cpu().numpy())
        crn_ids.extend(str(value) for value in batch["sequence_id"]); crn_seeds.extend([seed + 1_000_000 + batch_index] * len(batch["sequence_id"]))
        for kind, values in doses.items():
            for dose in values:
                modified = controls.clone()
                if kind == "brake": modified[:, 25:50, 0] = (modified[:, 25:50, 0] - dose).clamp_min(-8.0)
                elif kind == "accelerate": modified[:, 25:50, 0] = (modified[:, 25:50, 0] + dose).clamp_max(4.0)
                else: modified[:, 25:50, 1] = (modified[:, 25:50, 1] + dose).clamp_max(.6)
                response = runner.run(batch, deterministic=False, ego_controls=modified, latent_sample=factual.latent_sample, destination_sample=factual.destination_sample)
                treatments[kind][dose].append(SimpleNamespace(states=response.states.cpu().numpy(), background_actions=response.background_actions.cpu().numpy()))
    baseline_rollout = _rollout_object(baseline)
    full_initial, intervention_mask = np.concatenate(intervention_initial), np.concatenate(intervention_active)
    intervention_report: dict[str, Any] = {}
    for kind, dose_values in doses.items():
        rollout_values = {dose: _rollout_object(values) for dose, values in treatments[kind].items()}
        natural = np.asarray(natural_calibration[kind]["effect_samples_mps2"], np.float32) if kind in {"brake", "accelerate"} else None
        intervention_report[kind] = intervention_metrics(baseline_rollout, rollout_values[dose_values[0]], rollout_values[dose_values[-1]], full_initial, intervention_mask, kind, natural)
        intervention_report[kind]["dose_response"] = intervention_dose_response(baseline_rollout, rollout_values, full_initial, intervention_mask, kind, natural_calibration)
    np.savez_compressed(output / "intervention_crn.npz", sequence_id=np.asarray(crn_ids), seed=np.asarray(crn_seeds), latent=np.concatenate(crn_latents), destination=np.concatenate(crn_destinations))
    report.update({"factual_fidelity": {"deterministic_prior_mode": deterministic_report, "TrafficBots_Oracle": factual_metrics(np.concatenate(oracle), target, mask)}, "distribution_stochasticity": stochastic_report, "intervention_effectiveness": intervention_report, "evaluation_protocol": {"stochasticity_subset_sequences": stochastic_count, "intervention_subset_sequences": intervention_count, "natural_response_reference_sequences": len(full_test), "intervention_common_random_numbers": True}, "reproducibility": {"seed": seed, "sequence_ids_sha256": hashlib.sha256("\n".join(sorted(crn_ids)).encode()).hexdigest(), "stochastic_samples": len(stochastic), "checkpoint": str(Path(checkpoint).resolve()), "checkpoint_sha256": file_sha256(checkpoint)}})
    report["deterministic_prior_mode"] = deterministic_report
    report["causal_prior_samples_16"] = stochastic_report
    report["TrafficBots_Oracle"] = report["factual_fidelity"]["TrafficBots_Oracle"]
    save_json(report, output / "evaluation.json")
    return report
