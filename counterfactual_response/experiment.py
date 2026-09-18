"""highD-anchored braking probes for the retained stochastic drivers.

The protocol mirrors the CIH-WM response playback.  A logged highD leader plan
is replayed with an added braking intervention.  The same intervened leader is
then paired with (a) follower actions frozen from the no-intervention rollout
and (b) a closed-loop follower that observes the intervention.  Consequently,
the gap difference between the two branches is caused only by follower response,
not by comparing two different leader trajectories.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_ma_idm.src.driver import BayesianIDMDriver
from bayesian_ma_idm.src.model import load_posterior, sample_driver_joint
from dynamic_ar_idm.model import DynamicIDMDriver
from multi_regime_bidm.src.driver import MultiRegimeIDMDriver
from multi_regime_bidm.src.hsmm import FiniteHSMM
from multi_regime_bidm.src.online_filter import FilterState, initialize_filter


ROOT = Path(__file__).resolve().parents[1]
MODEL_NAMES = ("b_idm", "ma_idm", "dynamic_ar5", "multi_regime")


@dataclass(frozen=True)
class Scenario:
    dt_s: float = 0.04
    decision_dt_s: float = 0.2
    horizon_s: float = 6.0
    brake_start_s: float = 1.0
    brake_duration_s: float = 1.0
    brake_doses_mps2: tuple[float, ...] = (1.5, 3.0, 5.0)
    # Deterministic validation event selected using pre-intervention state only:
    # among supported ego-led events, row 55398 is nearest the shared calibrated
    # speed/gap support of the four retained driver models.  No future response
    # outcome is used in this selection.
    anchor_row: int = 55398
    anchor_index: int = 24
    leader_slot: int = 0
    follower_slot: int = 2
    vehicle_length_m: float = 4.8
    min_acceleration_mps2: float = -10.0
    max_acceleration_mps2: float = 4.0
    response_threshold_mps2: float = 0.15

    @property
    def frames(self) -> int:
        return int(round(self.horizon_s / self.dt_s))

    @property
    def hold_frames(self) -> int:
        return int(round(self.decision_dt_s / self.dt_s))

    @property
    def decisions(self) -> int:
        return int(np.ceil(self.frames / self.hold_frames))


@dataclass(frozen=True)
class HighDAnchor:
    row: int
    recording_id: int
    leader_id: int
    follower_id: int
    leader_x: np.ndarray
    leader_v: np.ndarray
    follower_x: np.ndarray
    follower_v: np.ndarray
    follower_a: np.ndarray
    logged_gap: np.ndarray


def select_common_support_anchor_row(
    evidence: str | Path,
    *, target_speed_mps: float = 15.0, target_gap_m: float = 30.0,
    speed_scale_mps: float = 3.0, gap_scale_m: float = 10.0,
    closing_scale_mps: float = 2.0,
) -> int:
    """Select a supported ego-led event using pre-intervention state only."""
    with np.load(evidence, allow_pickle=False) as events:
        eligible = np.flatnonzero((events["leader_slot"] == 0) & (events["cell"] == 2))
        pre = events["trajectory"][eligible, 0]
        score = (
            np.square((pre[:, 2] - target_speed_mps) / speed_scale_mps)
            + np.square((pre[:, 3] - target_gap_m) / gap_scale_m)
            + np.square(pre[:, 4] / closing_scale_mps)
        )
        return int(events["row_index"][eligible[int(np.argmin(score))]])


def load_highd_anchor(scenario: Scenario) -> HighDAnchor:
    """Load the same immutable validation event used by the CIH-WM playback."""
    cache = ROOT / "results/highd_shared_training_data/highd_sequence_cache/sequence_cache"
    states = np.load(cache / "agent_states.npy", mmap_mode="r", allow_pickle=False)
    valid = np.load(cache / "agent_valid.npy", mmap_mode="r", allow_pickle=False)
    stop = scenario.anchor_index + scenario.frames
    if stop > states.shape[1]:
        raise ValueError("requested response horizon exceeds the canonical highD row")
    pair_valid = np.asarray(
        valid[scenario.anchor_row, scenario.anchor_index:stop, [scenario.leader_slot, scenario.follower_slot]],
        bool,
    )
    if not pair_valid.all():
        raise ValueError("highD anchor leader/follower is not valid for the complete horizon")
    window = np.asarray(states[scenario.anchor_row, scenario.anchor_index:stop], float)
    leader = window[:, scenario.leader_slot]
    follower = window[:, scenario.follower_slot]
    evidence = ROOT / "results/hierarchical_world_model/cih_wm/evidence/reaction_events/validation/reaction_events.npz"
    with np.load(evidence, allow_pickle=False) as events:
        match = np.flatnonzero(events["row_index"] == scenario.anchor_row)
        if len(match) != 1:
            raise ValueError("anchor row must identify exactly one frozen reaction event")
        event = int(match[0])
        if int(events["leader_slot"][event]) != scenario.leader_slot or int(events["follower_slot"][event]) != scenario.follower_slot:
            raise ValueError("configured leader/follower slots disagree with reaction evidence")
        recording_id = int(events["recording_id"][event])
        leader_id = int(events["leader_id"][event])
        follower_id = int(events["follower_id"][event])
    leader_x = leader[:, 0] - follower[0, 0]
    follower_x = follower[:, 0] - follower[0, 0]
    gap = leader_x - follower_x - scenario.vehicle_length_m
    return HighDAnchor(
        row=scenario.anchor_row,
        recording_id=recording_id,
        leader_id=leader_id,
        follower_id=follower_id,
        leader_x=leader_x,
        leader_v=leader[:, 2].copy(),
        follower_x=follower_x,
        follower_v=follower[:, 2].copy(),
        follower_a=follower[:, 4].copy(),
        logged_gap=gap,
    )


def leader_trajectory(
    scenario: Scenario, dose_mps2: float, anchor: HighDAnchor | None = None,
) -> dict[str, np.ndarray]:
    """Replay logged ego controls and optionally add braking during the probe window."""
    anchor = load_highd_anchor(scenario) if anchor is None else anchor
    logged_acceleration = np.empty(scenario.frames)
    logged_acceleration[:-1] = np.diff(anchor.leader_v) / scenario.dt_s
    logged_acceleration[-1] = logged_acceleration[-2]
    x = float(anchor.leader_x[0])
    v = float(anchor.leader_v[0])
    position = np.empty(scenario.frames)
    speed = np.empty(scenario.frames)
    acceleration = np.empty(scenario.frames)
    for frame in range(scenario.frames):
        t = frame * scenario.dt_s
        active = scenario.brake_start_s <= t < scenario.brake_start_s + scenario.brake_duration_s
        a = float(logged_acceleration[frame] - (dose_mps2 if active else 0.0))
        position[frame], speed[frame], acceleration[frame] = x, v, a
        x += v * scenario.dt_s + 0.5 * a * scenario.dt_s**2
        v = max(0.0, v + a * scenario.dt_s)
    return {"x": position, "v": speed, "a": acceleration, "logged_a": logged_acceleration}


def _copy_filter(state: FilterState) -> FilterState:
    return FilterState(np.array(state.mass, copy=True), state.observations_seen, state.low_observation_likelihood)


def _models(path: Path) -> list[FiniteHSMM]:
    data = np.load(path, allow_pickle=False)
    return [
        FiniteHSMM(
            data["means"][style], data["variances"][style], data["transition"][style],
            data["initial"][style], data["duration_lambda"][style], int(data["duration_max"]),
            data["observation_mean"][style], data["observation_scale"][style],
        )
        for style in range(3)
    ]


class Resources:
    def __init__(self, scenario: Scenario, anchor: HighDAnchor):
        self.scenario = scenario
        self.anchor = anchor
        # Counterfactual simulation is a deployment probe, not an OOF score.
        # Use the all-cohort deployment posterior; CV folds remain evaluation-only.
        self.ma_posterior = load_posterior(
            ROOT / "bayesian_ma_idm/evidence/deployment/ma_idm_all_251.npz"
        )
        self.b_posterior = load_posterior(
            ROOT / "bayesian_ma_idm/evidence/deployment/b_idm_all_251.npz"
        )
        with np.load(ROOT / "dynamic_ar_idm/artifacts/full_posterior/dynamic_ar5_full_posterior.npz", allow_pickle=False) as data:
            self.dynamic = {key: data[key] for key in data.files}
        segmentation = ROOT / "multi_regime_bidm/evidence/segmentation"
        self.hsmm = _models(segmentation / "stage_a_finite_hsmm_models.npz")
        with np.load(segmentation / "stage_a_offline_labels.npz", allow_pickle=False) as labels:
            style = np.asarray(labels["style_id"], int)
        initial_gap = float(anchor.logged_gap[0])
        initial_speed = float(anchor.follower_v[0])
        initial_closing = float(anchor.follower_v[0] - anchor.leader_v[0])
        # The canonical response row has no completed five-second driver prefix.
        # Use the modal train-only style rather than fabricating max-acceleration
        # features from future frames or repeating the initial observation.
        self.style_id = int(np.argmax(np.bincount(style, minlength=3)))
        with np.load(ROOT / f"multi_regime_bidm/evidence/posterior/style_{self.style_id}_nuts_posterior.npz", allow_pickle=False) as data:
            self.multi = {key: data[key] for key in data.files}
        observation = np.asarray((initial_gap, initial_speed, initial_closing))
        self.multi_initial_state = initialize_filter(self.hsmm[self.style_id], observation)


def _sample_parameters(model: str, resources: Resources, rng: np.random.Generator) -> dict[str, Any]:
    if model in {"b_idm", "ma_idm"}:
        posterior = resources.b_posterior if model == "b_idm" else resources.ma_posterior
        values = sample_driver_joint(posterior, rng)
        return {
            "theta": values[:5], "sigma": float(values[5]), "ell": float(values[6]),
            "iid_sigma": float(values[7]) if len(values) > 7 else 0.1,
        }
    if model == "dynamic_ar5":
        draw = int(rng.integers(len(resources.dynamic["sigma_draws"])))
        driver = int(rng.integers(resources.dynamic["theta_draws"].shape[1]))
        return {
            "theta": resources.dynamic["theta_draws"][draw, driver],
            "rho": np.asarray(resources.dynamic["rho_map"], float),
            "sigma": float(resources.dynamic["sigma_draws"][draw]),
            "draw": draw, "driver": driver,
        }
    if model == "multi_regime":
        draw = int(rng.integers(len(resources.multi["theta_draws"])))
        return {
            "theta": np.asarray(resources.multi["theta_draws"][draw], float),
            "sigma": np.asarray(resources.multi["sigma_draws"][draw], float),
            "draw": draw, "style": resources.style_id,
        }
    raise ValueError(model)


def _noise(model: str, parameters: dict[str, Any], scenario: Scenario, rng: np.random.Generator) -> dict[str, np.ndarray]:
    count = scenario.decisions
    if model == "ma_idm":
        return {"process_z": rng.standard_normal(count), "iid_z": rng.standard_normal(count)}
    if model == "b_idm":
        return {"process_z": rng.standard_normal(count)}
    if model == "dynamic_ar5":
        return {"process_z": rng.standard_normal(count)}
    if model == "multi_regime":
        return {"z": rng.standard_normal(count), "u": rng.random(count)}
    raise ValueError(model)


def simulate_branch(model: str, parameters: dict[str, Any], noise: dict[str, np.ndarray],
                    leader: dict[str, np.ndarray], scenario: Scenario, resources: Resources) -> dict[str, np.ndarray]:
    x, v = 0.0, float(resources.anchor.follower_v[0])
    action = 0.0
    position = np.empty(scenario.frames)
    speed = np.empty(scenario.frames)
    acceleration = np.empty(scenario.frames)
    requested = np.empty(scenario.frames)
    gap_values = np.empty(scenario.frames)
    closing_values = np.empty(scenario.frames)
    ttc = np.empty(scenario.frames)
    residual = np.empty(scenario.frames)
    regimes = np.full(scenario.frames, -1, dtype=int)
    if model in {"b_idm", "ma_idm"}:
        driver: Any = BayesianIDMDriver(
            model,
            parameters["theta"],
            parameters["sigma"],
            lengthscale_s=parameters["ell"],
            iid_sigma=parameters["iid_sigma"] if model == "ma_idm" else 0.0,
            memory_s=None,
        )
    elif model == "dynamic_ar5":
        driver = DynamicIDMDriver(
            parameters["theta"], parameters["rho"], parameters["sigma"]
        )
    else:
        driver = MultiRegimeIDMDriver(
            resources.hsmm[resources.style_id], parameters["theta"], parameters["sigma"]
        )
        driver.state = _copy_filter(resources.multi_initial_state)
    for frame in range(scenario.frames):
        decision = frame // scenario.hold_frames
        gap = float(leader["x"][frame] - x - scenario.vehicle_length_m)
        closing = float(v - leader["v"][frame])
        if frame % scenario.hold_frames == 0:
            if model == "multi_regime":
                result = driver.decision(
                    gap_m=gap,
                    speed_mps=v,
                    leader_speed_mps=float(leader["v"][frame]),
                    standard_normal=float(noise["z"][decision]),
                    regime_uniform=float(noise["u"][decision]),
                )
                regime = result.regime
                disturbance = float(parameters["sigma"][regime] * noise["z"][decision])
                raw = result.requested_acceleration_mps2
            elif model in {"b_idm", "ma_idm"}:
                result = driver.decision(
                    gap_m=gap,
                    speed_mps=v,
                    leader_speed_mps=float(leader["v"][frame]),
                    process_standard_normal=float(noise["process_z"][decision]),
                    iid_standard_normal=(float(noise["iid_z"][decision]) if model == "ma_idm" else None),
                )
                regime = -1
                disturbance = result.residual_acceleration_mps2
                raw = result.requested_acceleration_mps2
            else:
                regime = -1
                raw = driver.decision(
                    gap_m=gap,
                    speed_mps=v,
                    leader_speed_mps=float(leader["v"][frame]),
                    standard_normal=float(noise["process_z"][decision]),
                )
                disturbance = float(driver.state.residual_history[0])
            action = float(np.clip(raw, scenario.min_acceleration_mps2, scenario.max_acceleration_mps2))
        # Every stored value is a frame-start state/action.  Mixing a frame-start
        # leader with a frame-end follower shifts the displayed gap by roughly
        # one native-frame travel distance at highway speed.
        position[frame], speed[frame], acceleration[frame] = x, v, action
        requested[frame] = raw
        residual[frame] = disturbance
        regimes[frame] = regime
        gap_values[frame] = gap
        closing_values[frame] = closing
        ttc[frame] = gap / closing if gap > 0.0 and closing > 1.0e-6 else np.inf
        x += v * scenario.dt_s + 0.5 * action * scenario.dt_s**2
        v = max(0.0, v + action * scenario.dt_s)
        if model == "multi_regime" and (frame + 1) % scenario.hold_frames == 0 and frame + 1 < scenario.frames:
            new_gap = float(leader["x"][frame + 1] - x - scenario.vehicle_length_m)
            driver.observe(
                gap_m=new_gap,
                speed_mps=v,
                leader_speed_mps=float(leader["v"][frame + 1]),
            )
    return {
        "position": position, "speed": speed, "acceleration": acceleration,
        "requested_acceleration": requested, "gap": gap_values, "closing": closing_values, "ttc": ttc,
        "residual": residual, "regime": regimes,
    }


def frozen_action_branch(
    natural: dict[str, np.ndarray], leader: dict[str, np.ndarray],
    scenario: Scenario, resources: Resources,
) -> dict[str, np.ndarray]:
    """Replay no-intervention follower actions under the intervened leader.

    This is the IDM analogue of CIH-WM's frozen factual-transition baseline:
    it receives the same braking leader as the reactive branch but cannot alter
    its action sequence in response.
    """
    x, v = 0.0, float(resources.anchor.follower_v[0])
    position = np.empty(scenario.frames)
    speed = np.empty(scenario.frames)
    gap_values = np.empty(scenario.frames)
    closing_values = np.empty(scenario.frames)
    ttc = np.empty(scenario.frames)
    for frame, action in enumerate(natural["acceleration"]):
        gap = float(leader["x"][frame] - x - scenario.vehicle_length_m)
        closing = float(v - leader["v"][frame])
        position[frame], speed[frame] = x, v
        gap_values[frame], closing_values[frame] = gap, closing
        ttc[frame] = gap / closing if gap > 0.0 and closing > 1.0e-6 else np.inf
        x += v * scenario.dt_s + 0.5 * float(action) * scenario.dt_s**2
        v = max(0.0, v + float(action) * scenario.dt_s)
    return {
        "position": position,
        "speed": speed,
        "acceleration": natural["acceleration"].copy(),
        "requested_acceleration": natural["requested_acceleration"].copy(),
        "gap": gap_values,
        "closing": closing_values,
        "ttc": ttc,
        "residual": natural["residual"].copy(),
        "regime": natural["regime"].copy(),
    }


def _onset(delta_action: np.ndarray, scenario: Scenario) -> float:
    decisions = np.asarray(delta_action[::scenario.hold_frames])
    active = decisions <= -scenario.response_threshold_mps2
    consecutive = active[:-1] & active[1:]
    candidates = np.flatnonzero(consecutive)
    if not len(candidates):
        return float("nan")
    onset = float(candidates[0] * scenario.decision_dt_s)
    return max(0.0, onset - scenario.brake_start_s)


def _metrics(
    natural: dict[str, np.ndarray], frozen: dict[str, np.ndarray], reactive: dict[str, np.ndarray],
    scenario: Scenario, anchor: HighDAnchor,
) -> dict[str, float]:
    delta = reactive["acceleration"] - frozen["acceleration"]
    decision_delta = delta[::scenario.hold_frames]
    finite_ttc = reactive["ttc"][np.isfinite(reactive["ttc"])]
    regime_valid = natural["regime"] >= 0
    gap_benefit = reactive["gap"] - frozen["gap"]
    return {
        "response_latency_s": _onset(delta, scenario),
        "peak_additional_braking_mps2": float(np.max(np.maximum(0.0, -decision_delta))),
        "additional_brake_dose_mps": float(np.sum(np.maximum(0.0, -decision_delta)) * scenario.decision_dt_s),
        "minimum_gap_m": float(np.min(reactive["gap"])),
        "frozen_minimum_gap_m": float(np.min(frozen["gap"])),
        "terminal_gap_benefit_m": float(gap_benefit[-1]),
        "peak_gap_benefit_m": float(np.max(gap_benefit)),
        "terminal_closing_reduction_mps": float(frozen["closing"][-1] - reactive["closing"][-1]),
        "natural_gap_rmse_to_highd_m": float(np.sqrt(np.mean(np.square(natural["gap"] - anchor.logged_gap)))),
        "natural_terminal_gap_drift_m": float(natural["gap"][-1] - natural["gap"][0]),
        "minimum_ttc_s": float(np.min(finite_ttc)) if len(finite_ttc) else float("inf"),
        "collision": float(np.any(reactive["gap"] <= 0.0)),
        "frozen_collision": float(np.any(frozen["gap"] <= 0.0)),
        "residual_branch_difference_mps2": float(np.max(np.abs(reactive["residual"] - natural["residual"]))),
        "regime_branch_difference": float(np.mean(reactive["regime"][regime_valid] != natural["regime"][regime_valid])) if np.any(regime_valid) else 0.0,
    }


def _summarize(rows: list[dict[str, float]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key in rows[0]:
        values = np.asarray([row[key] for row in rows], float)
        if key == "response_latency_s":
            finite = values[np.isfinite(values)]
            summary["response_probability"] = float(len(finite) / len(values))
            summary[key] = {
                "median": float(np.median(finite)) if len(finite) else None,
                "p10": float(np.quantile(finite, 0.1)) if len(finite) else None,
                "p90": float(np.quantile(finite, 0.9)) if len(finite) else None,
            }
        elif key == "minimum_ttc_s":
            finite = values[np.isfinite(values)]
            summary[key] = {
                "median": float(np.median(finite)) if len(finite) else None,
                "p10": float(np.quantile(finite, 0.1)) if len(finite) else None,
                "finite_share": float(len(finite) / len(values)),
            }
        elif key in {"collision", "frozen_collision"}:
            summary[f"{key}_probability"] = float(np.mean(values))
        else:
            summary[key] = {
                "median": float(np.median(values)), "p10": float(np.quantile(values, 0.1)),
                "p90": float(np.quantile(values, 0.9)), "mean": float(np.mean(values)),
            }
    return summary


def run_experiment(output_dir: str | Path, *, scenario: Scenario = Scenario(), futures: int = 256,
                   seed: int = 20260916) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if futures < 8:
        raise ValueError("at least eight paired futures are required")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    anchor = load_highd_anchor(scenario)
    resources = Resources(scenario, anchor)
    leaders = {dose: leader_trajectory(scenario, dose, anchor) for dose in (0.0, *scenario.brake_doses_mps2)}
    ensemble: dict[str, np.ndarray] = {}
    report: dict[str, Any] = {
        "experiment": "highD-anchored frozen-action versus closed-loop braking response probe",
        "purpose": "mechanism audit, not observational accuracy or causal-identification evidence",
        "scenario": asdict(scenario), "futures": int(futures), "seed": int(seed),
        "highd_anchor": {
            "split": "validation", "row": anchor.row, "recording_id": anchor.recording_id,
            "leader_id": anchor.leader_id, "follower_id": anchor.follower_id,
            "initial_leader_speed_mps": float(anchor.leader_v[0]),
            "initial_follower_speed_mps": float(anchor.follower_v[0]),
            "initial_gap_m": float(anchor.logged_gap[0]),
            "source": "canonical highD row and frozen CIH-WM reaction-event evidence",
            "selection": (
                "supported ego-led validation event nearest shared model speed/gap support using "
                "pre-intervention state only; future response outcomes were not used"
            ),
        },
        "common_random_numbers": True,
        "response_detection": "paired action delta <= -0.15 m/s^2 for two consecutive 5 Hz decisions",
        "branch_contract": (
            "both displayed branches receive the identical logged-control-plus-brake leader; the frozen branch replays "
            "the no-intervention follower actions while the reactive branch recomputes actions in closed loop"
        ),
        "multi_regime_style_id": int(resources.style_id),
        "multi_regime_initialization": "modal train-only style and one current observation; no five-second prefix is available for this canonical row",
        "models": {},
        "interpretation_guardrails": [
            "The synthetic intervention has no individual-level observed counterfactual ground truth.",
            "The highD future is factual context only and is not a target after the added leader brake.",
            "Natural-rollout gap RMSE reports anchor mismatch; it is not removed by selecting a convenient trajectory.",
            "B-IDM, MA-IDM and Dynamic-AR residual processes are exogenous to the leader command.",
            "Multi-regime emissions observe gap/speed/closing but its states were not trained as emergency-response states.",
            "Absolute cross-model scores combine different retained posterior protocols; use them as response diagnostics, not a ranking.",
        ],
    }
    model_seeds = np.random.SeedSequence(seed).spawn(len(MODEL_NAMES))
    for model_index, model in enumerate(MODEL_NAMES):
        rng = np.random.default_rng(model_seeds[model_index])
        keys = ("position", "speed", "acceleration", "gap", "closing", "residual", "regime")
        natural_store = {key: [] for key in keys}
        frozen_store = {dose: {key: [] for key in keys} for dose in scenario.brake_doses_mps2}
        reactive_store = {dose: {key: [] for key in keys} for dose in scenario.brake_doses_mps2}
        metric_rows = {dose: [] for dose in scenario.brake_doses_mps2}
        for _ in range(futures):
            parameters = _sample_parameters(model, resources, rng)
            innovations = _noise(model, parameters, scenario, rng)
            natural = simulate_branch(model, parameters, innovations, leaders[0.0], scenario, resources)
            for key in keys:
                natural_store[key].append(natural[key])
            for dose in scenario.brake_doses_mps2:
                frozen = frozen_action_branch(natural, leaders[dose], scenario, resources)
                reactive = simulate_branch(model, parameters, innovations, leaders[dose], scenario, resources)
                for key in keys:
                    frozen_store[dose][key].append(frozen[key])
                    reactive_store[dose][key].append(reactive[key])
                metric_rows[dose].append(_metrics(natural, frozen, reactive, scenario, anchor))
        for key, values in natural_store.items():
            ensemble[f"{model}__natural__{key}"] = np.asarray(values)
        for dose in scenario.brake_doses_mps2:
            dose_key = str(dose).replace(".", "p")
            for key in keys:
                ensemble[f"{model}__dose_{dose_key}__frozen__{key}"] = np.asarray(frozen_store[dose][key])
                ensemble[f"{model}__dose_{dose_key}__reactive__{key}"] = np.asarray(reactive_store[dose][key])
        report["models"][model] = {
            "posterior_source": {
                "b_idm": "all-251-pair B-IDM deployment posterior; OOF folds are evaluation-only",
                "ma_idm": "all-251-pair MA-IDM deployment posterior; OOF folds are evaluation-only",
                "dynamic_ar5": "251-pair MAP/Laplace theta draws with stable fitted MAP rho",
                "multi_regime": f"retained style-{resources.style_id} hierarchical NUTS posterior and causal finite-HSMM filter",
            }[model],
            "dose_response": {str(dose): _summarize(metric_rows[dose]) for dose in scenario.brake_doses_mps2},
        }
    medium = str(float(scenario.brake_doses_mps2[1]))
    report["primary_findings_at_medium_dose"] = {
        "dose_mps2": float(scenario.brake_doses_mps2[1]),
        "exogenous_residual_models": {
            model: {
                "response_probability": report["models"][model]["dose_response"][medium]["response_probability"],
                "median_latency_s": report["models"][model]["dose_response"][medium]["response_latency_s"]["median"],
                "median_peak_additional_braking_mps2": report["models"][model]["dose_response"][medium]["peak_additional_braking_mps2"]["median"],
                "mean_residual_branch_difference_mps2": report["models"][model]["dose_response"][medium]["residual_branch_difference_mps2"]["mean"],
            }
            for model in ("b_idm", "ma_idm", "dynamic_ar5")
        },
        "multi_regime": {
            "response_probability": report["models"]["multi_regime"]["dose_response"][medium]["response_probability"],
            "median_latency_s": report["models"]["multi_regime"]["dose_response"][medium]["response_latency_s"]["median"],
            "median_peak_additional_braking_mps2": report["models"]["multi_regime"]["dose_response"][medium]["peak_additional_braking_mps2"]["median"],
            "mean_regime_branch_difference_share": report["models"]["multi_regime"]["dose_response"][medium]["regime_branch_difference"]["mean"],
        },
        "interpretation": (
            "B-IDM, MA-IDM and Dynamic-AR react only through the deterministic IDM state feedback: their paired "
            "residual paths are exactly unchanged by the leader intervention. Multi-regime has a branch-dependent "
            "latent-state path and the strongest median peak response in this anchor, but this does not establish "
            "human counterfactual validity or an absolute cross-model ranking."
        ),
    }
    ensemble["highd_logged_leader_x"] = anchor.leader_x
    ensemble["highd_logged_leader_v"] = anchor.leader_v
    ensemble["highd_logged_follower_x"] = anchor.follower_x
    ensemble["highd_logged_follower_v"] = anchor.follower_v
    ensemble["highd_logged_follower_a"] = anchor.follower_a
    ensemble["highd_logged_gap"] = anchor.logged_gap
    ensemble["leader_natural_x"] = leaders[0.0]["x"]
    ensemble["leader_natural_v"] = leaders[0.0]["v"]
    ensemble["leader_natural_a"] = leaders[0.0]["a"]
    for dose in scenario.brake_doses_mps2:
        dose_key = str(dose).replace(".", "p")
        ensemble[f"leader_dose_{dose_key}_x"] = leaders[dose]["x"]
        ensemble[f"leader_dose_{dose_key}_v"] = leaders[dose]["v"]
        ensemble[f"leader_dose_{dose_key}_a"] = leaders[dose]["a"]
    np.savez_compressed(output / "response_ensembles.npz", **ensemble)
    (output / "response_metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report, ensemble
