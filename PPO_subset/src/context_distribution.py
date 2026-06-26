"""Process-highD distribution-backed tail context sampling."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tools.context import context_from_npz, load_context_npz


FOLLOWING_EMPIRICAL_SOURCE = "highd_independent_tail_peak"
CUTIN_EMPIRICAL_SOURCE = "highd_evt_independent_tail_peak"
FOLLOWING_DISTRIBUTION_SOURCE = "highd_tail_distribution_sample"
CUTIN_DISTRIBUTION_SOURCE = "highd_evt_tail_distribution_sample"


def _string_array(value: np.ndarray) -> np.ndarray:
    return np.asarray([str(item) for item in value.reshape(-1)], dtype=object)


def _feature_from_context(context: dict[str, Any], *, event_type: str) -> np.ndarray:
    cond = np.asarray(context["scenario_conditions"], dtype=np.float64).reshape(-1)
    gap = max(float(cond[1]), 0.2)
    if event_type == "cut_in":
        return np.asarray(
            [
                float(cond[0]),
                np.log(gap),
                float(cond[2]),
                float(cond[3]),
                float(cond[4]),
                float(cond[5]),
                float(cond[6]),
                float(cond[7]),
                float(cond[8]),
                float(cond[9]),
            ],
            dtype=np.float64,
        )
    return np.asarray(
        [
            float(cond[0]),
            np.log(gap),
            float(cond[2]),
            float(cond[3]),
            float(cond[4]),
            float(cond[5]),
            float(cond[6]),
        ],
        dtype=np.float64,
    )


def _base_rows(rows: list[dict[str, Any]], *, event_type: str) -> list[dict[str, Any]]:
    source = (
        CUTIN_EMPIRICAL_SOURCE
        if event_type == "cut_in"
        else FOLLOWING_EMPIRICAL_SOURCE
    )
    empirical = [row for row in rows if str(row.get("source_type", "")) == source]
    if len(empirical) < 2:
        raise ValueError(
            "Tail context distribution requires at least two empirical "
            f"independent tail rows with source_type={source}; "
            f"got {len(empirical)}. "
            "Rebuild process_highD tail contexts."
        )
    return empirical


def _load_context_rows(path: Path, *, event_type: str) -> list[dict[str, Any]]:
    raw = load_context_npz(path)
    count = int(raw["scenario_conditions"].shape[0])
    rows = [context_from_npz(raw, idx) for idx in range(count)]
    if not rows:
        raise ValueError(f"Tail context file is empty: {path}")
    if "source_type" in raw:
        source_types = set(_string_array(raw["source_type"]))
        empirical = (
            CUTIN_EMPIRICAL_SOURCE
            if event_type == "cut_in"
            else FOLLOWING_EMPIRICAL_SOURCE
        )
        if empirical not in source_types:
            raise ValueError(
                f"{path} does not contain empirical independent tail rows "
                f"({empirical}); rebuild process_highD tail contexts"
            )
    return rows


def _load_distribution(path: Path, *, event_type: str) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(
            "Scenario-condition distribution not found: "
            f"{path}. Run process_highD tail-context selection first."
        )
    raw = dict(np.load(path, allow_pickle=True))
    required = (
        "copula_correlation",
        "copula_variable_mask",
        "copula_marginal_values",
        "copula_marginal_clip_quantile",
    )
    missing = [key for key in required if key not in raw]
    if missing:
        raise KeyError(f"{path} is missing distribution keys: {missing}")
    marginal = np.asarray(raw["copula_marginal_values"], dtype=np.float64)
    expected_dim = 10 if event_type == "cut_in" else 7
    if marginal.ndim != 2 or int(marginal.shape[1]) != expected_dim:
        raise ValueError(
            f"{path} marginal feature shape {tuple(marginal.shape)} does not "
            f"match {event_type} expected dim {expected_dim}"
        )
    variable = np.asarray(raw["copula_variable_mask"], dtype=bool).reshape(-1)
    if int(variable.size) != expected_dim:
        raise ValueError(
            f"{path} copula_variable_mask has dim {variable.size}, "
            f"expected {expected_dim}"
        )
    if not np.any(variable):
        raise ValueError(f"{path} has no variable copula dimensions")
    corr = np.asarray(raw["copula_correlation"], dtype=np.float64)
    if corr.shape == (expected_dim, expected_dim):
        corr = corr[np.ix_(variable, variable)]
    elif corr.shape != (int(np.sum(variable)), int(np.sum(variable))):
        raise ValueError(
            f"{path} copula_correlation shape {tuple(corr.shape)} is incompatible "
            f"with variable dim {int(np.sum(variable))}"
        )
    corr = np.nan_to_num(np.atleast_2d(corr), nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 1.0)
    return {
        **raw,
        "copula_correlation": corr,
        "copula_variable_mask": variable,
        "copula_marginal_values": marginal,
        "copula_marginal_clip_quantile": np.asarray(
            raw["copula_marginal_clip_quantile"],
            dtype=np.float64,
        ),
    }


def _reconstruct_context(
    base: dict[str, Any],
    feature: np.ndarray,
    *,
    event_type: str,
    base_context_index: int,
    distance: float,
    sample_index: int,
    dt: float,
) -> dict[str, Any]:
    context = dict(base)
    states = np.asarray(base["initial_states"], dtype=np.float32).copy()
    ego_length = float(base["ego_length"])
    adv_length = float(base["adv_length"])

    if event_type == "cut_in":
        ego_vx = max(float(feature[0]), 0.0)
        gap = float(np.exp(float(feature[1])))
        lateral_offset = float(feature[2])
        delta_vx = float(feature[3])
        states[0, 2] = np.float32(ego_vx)
        states[1, 0] = np.float32(states[0, 0] + 0.5 * (ego_length + adv_length) + gap)
        states[1, 1] = np.float32(states[0, 1] + lateral_offset)
        states[1, 2] = np.float32(max(ego_vx - delta_vx, 0.0))
        states[1, 4] = np.float32(np.clip(float(feature[4]), -8.0, 4.0))
        states[1, 3] = np.float32(float(feature[5]))
        states[1, 5] = np.float32(np.clip(float(feature[6]), -4.0, 4.0))
        conditions = np.asarray(
            [
                ego_vx,
                gap,
                lateral_offset,
                delta_vx,
                float(feature[4]),
                float(feature[5]),
                float(feature[6]),
                float(feature[7]),
                float(feature[8]),
                float(feature[9]),
            ],
            dtype=np.float32,
        )
        context["source_type"] = CUTIN_DISTRIBUTION_SOURCE
        context["event_id"] = f"subset_cutin_distribution_{sample_index:010d}"
        context["risk_start_index"] = max(
            0,
            int(round(float(feature[8]) / max(float(dt), 1.0e-6))) - 1,
        )
    else:
        ego_vx = max(float(feature[0]), 0.0)
        gap = float(np.exp(float(feature[1])))
        delta_v = float(feature[2])
        states[0, 2] = np.float32(ego_vx)
        states[1, 0] = np.float32(states[0, 0] + 0.5 * (ego_length + adv_length) + gap)
        states[1, 2] = np.float32(max(ego_vx - delta_v, 0.0))
        states[1, 4] = np.float32(np.clip(float(feature[3]), -8.0, 4.0))
        conditions = np.asarray(
            [
                ego_vx,
                gap,
                delta_v,
                float(states[1, 4]),
                float(feature[4]),
                float(feature[5]),
                max(float(feature[6]), 0.0),
            ],
            dtype=np.float32,
        )
        context["source_type"] = FOLLOWING_DISTRIBUTION_SOURCE
        context["event_id"] = f"subset_following_distribution_{sample_index:010d}"

    context["scenario_conditions"] = conditions
    context["initial_states"] = states.astype(np.float32)
    context["base_event_id"] = str(base.get("event_id", ""))
    context["base_context_index"] = int(base_context_index)
    context["synthetic_context"] = 1
    context["context_model_method"] = "process_highd_gaussian_copula_distribution"
    context["context_feature_distance"] = float(distance)
    context["initial_gap"] = float(conditions[1])
    context["initial_closing_speed"] = float(conditions[3] if event_type == "cut_in" else conditions[2])
    context["risk_score"] = float("nan")
    context["evt_tail_probability"] = float("nan")
    return context


@dataclass
class TailContextDistribution:
    """Deterministic sampler using process_highD's saved tail condition model."""

    context_rows: list[dict[str, Any]]
    distribution: dict[str, np.ndarray]
    event_type: str
    seed: int = 42
    population_size: int = 2_147_483_647
    dt: float = 0.04

    def __post_init__(self) -> None:
        self.base_rows = _base_rows(self.context_rows, event_type=self.event_type)
        self.base_features = np.stack(
            [
                _feature_from_context(row, event_type=self.event_type)
                for row in self.base_rows
            ],
            axis=0,
        )
        self.marginal_values = np.asarray(
            self.distribution["copula_marginal_values"],
            dtype=np.float64,
        )
        self.variable = np.asarray(
            self.distribution["copula_variable_mask"],
            dtype=bool,
        )
        self.corr = np.asarray(
            self.distribution["copula_correlation"],
            dtype=np.float64,
        )
        clip_raw = np.asarray(
            self.distribution["copula_marginal_clip_quantile"],
            dtype=np.float64,
        )
        clip_value = (
            clip_raw.item()
            if clip_raw.ndim == 0
            else clip_raw.reshape(-1)[0]
        )
        self.clip_quantile = min(
            max(float(clip_value), 0.0),
            0.49,
        )
        variable_values = self.marginal_values[:, self.variable]
        self.variable_lower = np.quantile(
            variable_values,
            self.clip_quantile,
            axis=0,
        )
        self.variable_upper = np.quantile(
            variable_values,
            1.0 - self.clip_quantile,
            axis=0,
        )
        self.center = np.median(self.marginal_values, axis=0)
        self.scale = np.std(self.marginal_values, axis=0)
        self.scale = np.where(self.scale > 1.0e-6, self.scale, 1.0)
        self.standardized_base = (self.base_features - self.center) / self.scale
        if self.event_type == "cut_in":
            self.time_to_cross_support = np.unique(self.marginal_values[:, 8])
        else:
            self.time_to_cross_support = np.asarray([], dtype=np.float64)

    def __len__(self) -> int:
        return int(self.population_size)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        from scipy.special import ndtr

        sample_index = int(idx)
        rng = np.random.default_rng(int(self.seed) + sample_index)
        feature = self.center.copy()
        sampled_z = rng.multivariate_normal(
            np.zeros(int(np.sum(self.variable)), dtype=np.float64),
            self.corr,
            check_valid="ignore",
        )
        sampled_u = np.clip(ndtr(sampled_z), 1.0e-6, 1.0 - 1.0e-6)
        variable_cols = np.flatnonzero(self.variable)
        for out_col, feature_col in enumerate(variable_cols):
            feature[feature_col] = np.quantile(
                self.marginal_values[:, feature_col],
                sampled_u[out_col],
            )
            feature[feature_col] = np.clip(
                feature[feature_col],
                self.variable_lower[out_col],
                self.variable_upper[out_col],
            )
        if self.event_type == "cut_in":
            feature[7] = np.clip(feature[7], -1.0, 1.0)
            if self.time_to_cross_support.size:
                nearest = int(np.argmin(np.abs(self.time_to_cross_support - feature[8])))
                feature[8] = float(self.time_to_cross_support[nearest])
        target_standardized = (feature - self.center) / self.scale
        distance = np.sum(
            (self.standardized_base - target_standardized[None, :]) ** 2,
            axis=1,
        )
        base_idx = int(np.argmin(distance))
        return _reconstruct_context(
            self.base_rows[base_idx],
            feature,
            event_type=self.event_type,
            base_context_index=base_idx,
            distance=float(np.sqrt(distance[base_idx])),
            sample_index=sample_index,
            dt=self.dt,
        )


def load_tail_context_distribution(
    context_path: str | Path,
    distribution_path: str | Path,
    *,
    event_type: str,
    seed: int,
    population_size: int,
    dt: float,
) -> TailContextDistribution:
    context_file = Path(context_path)
    distribution_file = Path(distribution_path)
    if not context_file.exists():
        raise FileNotFoundError(f"Tail context file not found: {context_file}")
    rows = _load_context_rows(context_file, event_type=event_type)
    distribution = _load_distribution(distribution_file, event_type=event_type)
    return TailContextDistribution(
        context_rows=rows,
        distribution=distribution,
        event_type=event_type,
        seed=int(seed),
        population_size=int(population_size),
        dt=float(dt),
    )
