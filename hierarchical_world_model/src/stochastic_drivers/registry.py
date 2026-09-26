"""Retained-artifact registry and constructors for driver sessions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from external_model_baselines.models.active_inference_driver.config import ActiveInferenceConfig
from external_model_baselines.models.active_inference_driver.model import ActiveInferenceDriver
from external_model_baselines.models.active_inference_driver.official_wrapper import OfficialPOMDP25Hz
from external_model_baselines.models.bayesian_ma_idm.src.driver import BayesianIDMDriver
from external_model_baselines.models.bayesian_ma_idm.src.evaluation import (
    condition_driver_joint_from_prefix,
)
from external_model_baselines.models.bayesian_ma_idm.src.model import load_posterior
from external_model_baselines.models.dynamic_ar_idm.model import DynamicIDMDriver
from external_model_baselines.models.multi_regime_bidm.src.driver import MultiRegimeIDMDriver, load_hsmm_models

from .interface import DriverSession, LongitudinalPrefix


ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class DriverSpec:
    artifact: str | None
    evidence_status: str
    default_allowed: bool
    paper_relation: str


DRIVER_SPECS: dict[str, DriverSpec] = {
    "b_idm": DriverSpec(
        "external_model_baselines/models/bayesian_ma_idm/evidence/deployment/b_idm_all_251.npz",
        "usable_routine_following_baseline", True, "paper mechanism; full local cohort fit",
    ),
    "ma_idm": DriverSpec(
        "external_model_baselines/models/bayesian_ma_idm/evidence/deployment/ma_idm_all_251.npz",
        "usable_routine_following", True, "paper mechanism; full local cohort fit",
    ),
    "dynamic_ar5": DriverSpec(
        "external_model_baselines/models/dynamic_ar_idm/artifacts/full_posterior/dynamic_ar5_full_posterior.npz",
        "engineering_map_laplace_approximation", True, "paper model; NUTS did not converge",
    ),
    "multi_regime": DriverSpec(
        "external_model_baselines/models/multi_regime_bidm/evidence/posterior/style_{style_id}_nuts_posterior.npz",
        "rejected_not_well_calibrated", False, "documented finite-HSMM adaptation",
    ),
    "active_inference_longitudinal": DriverSpec(
        None, "failed_official_bridge", False, "independent longitudinal adaptation",
    ),
    "active_inference_official": DriverSpec(
        None, "passed_requires_external_pinned_source", True, "official v1.0.0 wrapper",
    ),
}


def _require_allowed(model_id: str, allow_unaccepted: bool) -> DriverSpec:
    try:
        spec = DRIVER_SPECS[model_id]
    except KeyError as error:
        raise ValueError(f"unknown driver model {model_id!r}") from error
    if not spec.default_allowed and not allow_unaccepted:
        raise RuntimeError(
            f"{model_id} is not deployment-approved: {spec.evidence_status}; "
            "pass allow_unaccepted=True only for research diagnostics"
        )
    return spec


def create_driver_session(
    model_id: str,
    *,
    seed: int = 0,
    style_id: int = 1,
    allow_unaccepted: bool = False,
    official_source_dir: str | Path | None = None,
    official_following_dir: str | Path | None = None,
    official_device: str = "cpu",
    active_config: ActiveInferenceConfig | None = None,
    posterior_path: str | Path | None = None,
    prefix_observations: LongitudinalPrefix | np.ndarray | None = None,
    personalize_from_prefix: bool = False,
    personalization_candidates: int = 128,
) -> DriverSession:
    """Create one episode-level driver with a fresh joint posterior draw."""
    spec = _require_allowed(model_id, allow_unaccepted)
    if model_id in {"b_idm", "ma_idm"}:
        artifact = ROOT / spec.artifact if posterior_path is None else Path(posterior_path)
        if not artifact.is_absolute():
            artifact = ROOT / artifact
        prefix = (
            prefix_observations.as_array()
            if isinstance(prefix_observations, LongitudinalPrefix)
            else None
            if prefix_observations is None
            else np.asarray(prefix_observations, float)
        )
        if personalize_from_prefix:
            if prefix is None or len(prefix) < 3:
                raise ValueError(
                    "prefix personalization requires at least three completed actions"
                )
            rng = np.random.default_rng(seed)
            values = condition_driver_joint_from_prefix(
                load_posterior(artifact),
                prefix,
                rng,
                candidates=int(personalization_candidates),
            )
            driver = BayesianIDMDriver(
                model_id,
                values[:5],
                float(values[5]),
                lengthscale_s=float(values[6]),
                iid_sigma=float(values[7]) if len(values) > 7 and model_id == "ma_idm" else 0.0,
                seed=seed,
            )
        else:
            driver = BayesianIDMDriver.from_population_posterior(
                artifact, model=model_id, seed=seed,
            )
        return DriverSession(
            model_id,
            driver,
            kind="bayesian_idm",
            metadata={
                "posterior_artifact": str(artifact.resolve()),
                "parameter_mode": (
                    "prefix_personalized" if personalize_from_prefix else "population"
                ),
                "prefix_memory_initialized": bool(prefix is not None and model_id == "ma_idm"),
            },
            default_prefix_observations=prefix_observations,
        )
    if model_id == "dynamic_ar5":
        return DriverSession(
            model_id, DynamicIDMDriver.from_posterior(ROOT / spec.artifact, seed=seed)
        )
    if model_id == "multi_regime":
        if style_id not in (0, 1, 2):
            raise ValueError("multi-regime style_id must be 0, 1 or 2")
        models = load_hsmm_models(
            ROOT / "external_model_baselines/models/multi_regime_bidm/evidence/segmentation/stage_a_finite_hsmm_models.npz"
        )
        driver = MultiRegimeIDMDriver.from_posterior(
            models[style_id], ROOT / spec.artifact.format(style_id=style_id), seed=seed,
        )
        return DriverSession(model_id, driver, kind="multi_regime")
    if model_id == "active_inference_longitudinal":
        return DriverSession(
            model_id,
            ActiveInferenceDriver(active_config or ActiveInferenceConfig()),
            kind="active_inference",
        )
    if official_source_dir is None or official_following_dir is None:
        raise ValueError(
            "official active inference requires official_source_dir and official_following_dir"
        )
    official = OfficialPOMDP25Hz(
        source_dir=official_source_dir,
        following_dir=official_following_dir,
        device=official_device,
    )
    return DriverSession(model_id, official, kind="official_active_inference")


def verify_driver_assets() -> dict[str, dict[str, Any]]:
    """Return machine-readable availability without treating rejection as success."""
    result: dict[str, dict[str, Any]] = {}
    for model_id, spec in DRIVER_SPECS.items():
        artifact = None if spec.artifact is None else spec.artifact.format(style_id=1)
        result[model_id] = {
            **asdict(spec),
            "artifact": artifact,
            "artifact_present": None if artifact is None else (ROOT / artifact).is_file(),
        }
    return result
