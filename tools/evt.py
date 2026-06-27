"""Peak-over-threshold EVT calibration for event risk scores."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import genpareto


RETURN_PERIODS = (20, 50, 100)


def _finite_sorted(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        raise ValueError("EVT calibration requires at least one finite value")
    return np.sort(arr)


def empirical_survival(values: np.ndarray, y: np.ndarray | float) -> np.ndarray:
    """Return P(X > y) using finite empirical calibration values."""
    sorted_values = _finite_sorted(values)
    query = np.asarray(y, dtype=np.float64)
    right = np.searchsorted(sorted_values, query, side="right")
    survival = (sorted_values.size - right) / float(sorted_values.size)
    return np.asarray(survival, dtype=np.float64)


@dataclass(frozen=True)
class GPDTailModel:
    """Serializable POT/GPD tail model for event-level risk scores."""

    u: float
    xi: float
    beta: float
    exceedance_rate: float
    calibration_values: np.ndarray
    return_levels: dict[str, float]
    return_level_ci: dict[str, dict[str, float]]
    threshold_candidates: list[dict[str, float]]
    threshold_selection: dict[str, Any] | None = None
    survival_eps: float = 1.0e-12

    def survival(self, y: np.ndarray | float) -> np.ndarray:
        query = np.asarray(y, dtype=np.float64)
        out = empirical_survival(self.calibration_values, query)
        exceedance = query > float(self.u)
        if np.any(exceedance):
            excess = np.maximum(query - float(self.u), 0.0)
            tail = genpareto.sf(
                excess,
                c=float(self.xi),
                loc=0.0,
                scale=max(float(self.beta), 1.0e-12),
            )
            out = np.where(exceedance, float(self.exceedance_rate) * tail, out)
        return np.maximum(out, 0.0)

    def score(self, y: np.ndarray | float) -> np.ndarray:
        survival = self.survival(y)
        return -np.log(np.maximum(survival, float(self.survival_eps)))

    def return_level(self, period: int | float) -> float:
        key = f"z{int(period)}"
        if key in self.return_levels:
            return float(self.return_levels[key])
        return event_return_level(
            self.calibration_values,
            period=float(period),
            u=float(self.u),
            xi=float(self.xi),
            beta=float(self.beta),
            exceedance_rate=float(self.exceedance_rate),
        )

    def to_dict(self, model_type: str | None = None) -> dict[str, Any]:
        return {
            "model_type": str(model_type or "gpd_pot_event_risk"),
            "u": float(self.u),
            "xi": float(self.xi),
            "beta": float(self.beta),
            "exceedance_rate": float(self.exceedance_rate),
            "num_calibration_values": int(self.calibration_values.size),
            "calibration_values": [
                float(value) for value in np.asarray(self.calibration_values)
            ],
            "return_levels": {
                key: float(value) for key, value in self.return_levels.items()
            },
            "return_level_ci": self.return_level_ci,
            "threshold_candidates": self.threshold_candidates,
            "threshold_selection": self.threshold_selection or {},
            "survival_eps": float(self.survival_eps),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GPDTailModel":
        return cls(
            u=float(payload["u"]),
            xi=float(payload["xi"]),
            beta=float(payload["beta"]),
            exceedance_rate=float(payload["exceedance_rate"]),
            calibration_values=np.asarray(
                payload["calibration_values"],
                dtype=np.float64,
            ),
            return_levels={
                str(key): float(value)
                for key, value in payload.get("return_levels", {}).items()
            },
            return_level_ci={
                str(key): {
                    str(inner_key): float(inner_value)
                    for inner_key, inner_value in dict(inner).items()
                }
                for key, inner in payload.get("return_level_ci", {}).items()
            },
            threshold_candidates=[
                {str(key): float(value) for key, value in dict(row).items()}
                for row in payload.get("threshold_candidates", [])
            ],
            threshold_selection=dict(payload.get("threshold_selection", {})),
            survival_eps=float(payload.get("survival_eps", 1.0e-12)),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "GPDTailModel":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def to_json(self, path: str | Path, model_type: str | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                self.to_dict(model_type=model_type),
                f,
                indent=2,
                ensure_ascii=False,
            )


def fit_gpd_excess(excess: np.ndarray) -> tuple[float, float]:
    values = np.asarray(excess, dtype=np.float64)
    values = values[np.isfinite(values) & (values > 0.0)]
    if values.size < 5:
        raise ValueError("GPD fitting requires at least five positive exceedances")
    xi, _loc, beta = genpareto.fit(values, floc=0.0)
    beta = max(float(beta), 1.0e-12)
    return float(xi), beta


def return_level_from_params(
    *,
    period: float,
    u: float,
    xi: float,
    beta: float,
    exceedance_rate: float,
) -> float:
    if period <= 1.0:
        raise ValueError("return period must be greater than one event")
    tail_probability = 1.0 / float(period)
    if tail_probability >= float(exceedance_rate):
        return float(u)
    conditional_survival = max(tail_probability / float(exceedance_rate), 1.0e-15)
    if abs(float(xi)) < 1.0e-8:
        excess = -float(beta) * math.log(conditional_survival)
    else:
        excess = (
            float(beta)
            / float(xi)
            * (conditional_survival ** (-float(xi)) - 1.0)
        )
    return float(u + excess)


def gpd_conditional_survival(
    y: float,
    *,
    u: float,
    xi: float,
    beta: float,
) -> float:
    """Return P(X > y | X > u) under a fitted GPD POT tail."""
    y_value = float(y)
    u_value = float(u)
    if y_value <= u_value:
        return 1.0
    excess = y_value - u_value
    beta_value = max(float(beta), 1.0e-12)
    xi_value = float(xi)
    if abs(xi_value) < 1.0e-8:
        return float(math.exp(-excess / beta_value))
    support = 1.0 + xi_value * excess / beta_value
    if support <= 0.0:
        return 0.0
    return float(support ** (-1.0 / xi_value))


def return_level_for_tail_exposure(
    *,
    expected_tail_exceedances: float,
    u: float,
    xi: float,
    beta: float,
) -> float:
    """Return level for a distance with expected POT exceedance count."""
    expected = float(expected_tail_exceedances)
    if expected <= 1.0:
        return float(u)
    conditional_survival = 1.0 / expected
    beta_value = max(float(beta), 1.0e-12)
    xi_value = float(xi)
    if abs(xi_value) < 1.0e-8:
        excess = -beta_value * math.log(conditional_survival)
    else:
        excess = (
            beta_value
            / xi_value
            * (conditional_survival ** (-xi_value) - 1.0)
        )
    return float(float(u) + max(excess, 0.0))


def event_return_level(
    values: np.ndarray,
    *,
    period: float,
    u: float,
    xi: float,
    beta: float,
    exceedance_rate: float,
) -> float:
    tail_probability = 1.0 / float(period)
    if tail_probability >= float(exceedance_rate):
        return float(np.quantile(_finite_sorted(values), 1.0 - tail_probability))
    return return_level_from_params(
        period=period,
        u=u,
        xi=xi,
        beta=beta,
        exceedance_rate=exceedance_rate,
    )


def threshold_stability(
    values: np.ndarray,
    *,
    min_exceedances: int = 20,
    max_tail_fraction: float = 0.25,
    max_threshold_candidates: int | None = None,
) -> list[dict[str, float]]:
    sorted_values = _finite_sorted(values)
    n = int(sorted_values.size)
    max_exceedances = min(n - 1, max(min_exceedances, int(n * max_tail_fraction)))
    min_k = max(5, int(min_exceedances))
    if max_exceedances < min_k:
        raise ValueError(
            "Not enough values for EVT threshold scan: "
            f"n={n}, min_exceedances={min_exceedances}"
        )
    if max_threshold_candidates is None or max_threshold_candidates <= 0:
        k_values = np.arange(min_k, max_exceedances + 1, dtype=np.int64)
    else:
        k_values = np.unique(
            np.round(
                np.linspace(
                    min_k,
                    max_exceedances,
                    int(max_threshold_candidates),
                )
            ).astype(np.int64)
        )
    rows: list[dict[str, float]] = []
    for k in k_values:
        u = float(sorted_values[n - k - 1])
        exceedances = sorted_values[sorted_values > u] - u
        if exceedances.size < min_k:
            continue
        try:
            xi, beta = fit_gpd_excess(exceedances)
        except ValueError:
            continue
        rows.append(
            {
                "k": float(exceedances.size),
                "u": float(u),
                "xi": float(xi),
                "beta": float(beta),
                "modified_scale": float(beta - xi * u),
                "endpoint": float(u - beta / xi) if float(xi) < 0.0 else float("inf"),
                "z1000": event_return_level(
                    sorted_values,
                    period=1000.0,
                    u=float(u),
                    xi=float(xi),
                    beta=float(beta),
                    exceedance_rate=float(exceedances.size / n),
                ),
                "exceedance_rate": float(exceedances.size / n),
            }
        )
    if not rows:
        raise RuntimeError("No valid GPD threshold candidates were fitted")
    return rows


def _selection_start_index(rows: list[dict[str, float]]) -> int:
    if len(rows) <= 2:
        return 0
    return max(1, min(len(rows) - 1, int(round(0.01 * len(rows)))))


def _select_threshold(
    rows: list[dict[str, float]],
    *,
    beta_power: float = 0.25,
) -> dict[str, float]:
    xis = np.asarray([row["xi"] for row in rows], dtype=np.float64)
    scores: list[tuple[float, int]] = []
    start_idx = _selection_start_index(rows)
    for idx in range(start_idx, len(rows)):
        prefix = xis[: idx + 1]
        weights = np.power(
            np.arange(1, idx + 2, dtype=np.float64),
            float(beta_power),
        )
        score = float(np.sum(weights * np.square(prefix - xis[idx])) / np.sum(weights))
        scores.append((score, idx))
    return dict(rows[min(scores, key=lambda item: item[0])[1]])


def fit_evt_model(
    values: np.ndarray,
    *,
    return_periods: tuple[int, ...] = RETURN_PERIODS,
    min_exceedances: int = 20,
    max_tail_fraction: float = 0.25,
    max_threshold_candidates: int | None = None,
    min_threshold_exceedance_rate: float | None = 0.10,
    bootstrap_samples: int = 200,
    random_seed: int = 42,
    survival_eps: float = 1.0e-12,
) -> GPDTailModel:
    calibration_values = _finite_sorted(values)
    rows = threshold_stability(
        calibration_values,
        min_exceedances=min_exceedances,
        max_tail_fraction=max_tail_fraction,
        max_threshold_candidates=max_threshold_candidates,
    )
    if min_threshold_exceedance_rate is None:
        candidate_rows = rows
        min_rate = None
    else:
        min_rate = max(float(min_threshold_exceedance_rate), 0.0)
        candidate_rows = [
            row for row in rows if float(row["exceedance_rate"]) >= min_rate
        ]
        if not candidate_rows:
            raise RuntimeError(
                "No EVT threshold candidates satisfy "
                f"min_threshold_exceedance_rate={min_rate:.6g}"
            )
    chosen = _select_threshold(candidate_rows)
    u = float(chosen["u"])
    excess = calibration_values[calibration_values > u] - u
    xi, beta = fit_gpd_excess(excess)
    exceedance_rate = float(excess.size / calibration_values.size)
    return_levels = {
        f"z{int(period)}": event_return_level(
            calibration_values,
            period=float(period),
            u=u,
            xi=xi,
            beta=beta,
            exceedance_rate=exceedance_rate,
        )
        for period in return_periods
    }
    ci = bootstrap_return_level_ci(
        calibration_values,
        chosen_k=int(chosen["k"]),
        return_periods=return_periods,
        bootstrap_samples=bootstrap_samples,
        random_seed=random_seed,
    )
    return GPDTailModel(
        u=u,
        xi=xi,
        beta=beta,
        exceedance_rate=exceedance_rate,
        calibration_values=calibration_values,
        return_levels=return_levels,
        return_level_ci=ci,
        threshold_candidates=rows,
        threshold_selection={
            "candidate_count": float(len(rows)),
            "eligible_candidate_count": float(len(candidate_rows)),
            "min_threshold_exceedance_rate": (
                float(min_rate) if min_rate is not None else float("nan")
            ),
            "max_tail_fraction": float(max_tail_fraction),
            "min_exceedances": float(min_exceedances),
            "max_threshold_candidates": (
                float(max_threshold_candidates)
                if max_threshold_candidates is not None
                else float("nan")
            ),
        },
        survival_eps=float(survival_eps),
    )


def bootstrap_return_level_ci(
    values: np.ndarray,
    *,
    chosen_k: int,
    return_periods: tuple[int, ...],
    bootstrap_samples: int,
    random_seed: int,
) -> dict[str, dict[str, float]]:
    sorted_values = _finite_sorted(values)
    n = int(sorted_values.size)
    rng = np.random.default_rng(int(random_seed))
    samples: dict[str, list[float]] = {f"z{period}": [] for period in return_periods}
    if bootstrap_samples <= 0:
        return {
            key: {"lower": float("nan"), "upper": float("nan")}
            for key in samples
        }
    for _ in range(int(bootstrap_samples)):
        boot = np.sort(rng.choice(sorted_values, size=n, replace=True))
        k = min(max(5, int(chosen_k)), n - 1)
        u = float(boot[n - k - 1])
        excess = boot[boot > u] - u
        if excess.size < 5:
            continue
        try:
            xi, beta = fit_gpd_excess(excess)
        except ValueError:
            continue
        exceedance_rate = float(excess.size / n)
        for period in return_periods:
            key = f"z{period}"
            try:
                samples[key].append(
                    event_return_level(
                        boot,
                        period=float(period),
                        u=u,
                        xi=xi,
                        beta=beta,
                        exceedance_rate=exceedance_rate,
                    )
                )
            except ValueError:
                continue
    return {
        key: {
            "lower": float(np.quantile(vals, 0.05)) if vals else float("nan"),
            "upper": float(np.quantile(vals, 0.95)) if vals else float("nan"),
        }
        for key, vals in samples.items()
    }


def load_evt_model(path: str | Path) -> GPDTailModel:
    return GPDTailModel.from_json(path)
