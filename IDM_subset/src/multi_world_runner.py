"""Orchestration and comparable reporting across IDM world-model backends."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from tools.plot_style import get_pyplot
from world_model.src.core.utils import file_sha256, load_json, load_yaml, save_json
from world_model.src.core.evaluation_scope import (
    evaluation_scope_contract,
    require_evaluation_scope,
)

from .world_model_registry import WorldModelSpec, get_world_model


ESTIMATORS = ("subset", "monte_carlo")
_CONFIG_SECTION = {"subset": "subset_simulation", "monte_carlo": "monte_carlo"}


def _resolve_path(value: str | Path, base: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def load_model_configs(
    suite: dict[str, Any], suite_dir: Path, selected: Iterable[str] | None = None
) -> dict[str, tuple[WorldModelSpec, dict[str, Any], Path]]:
    """Load configured model adapters without importing either model stack."""
    requested = None if selected is None else {get_world_model(x).model_id for x in selected}
    loaded: dict[str, tuple[WorldModelSpec, dict[str, Any], Path]] = {}
    for entry in suite.get("world_models", []):
        if not bool(entry.get("enabled", True)):
            continue
        spec = get_world_model(entry["id"])
        if requested is not None and spec.model_id not in requested:
            continue
        config_path = _resolve_path(entry.get("config", spec.default_config), suite_dir)
        if spec.model_id in loaded:
            raise ValueError(f"duplicate world model in suite: {spec.model_id}")
        loaded[spec.model_id] = (spec, load_yaml(config_path), config_path)
    if requested is not None:
        missing = requested.difference(loaded)
        if missing:
            raise ValueError(f"selected models are not enabled in suite: {sorted(missing)}")
    if not loaded:
        raise ValueError("the IDM model suite contains no enabled world models")
    return loaded


def validate_comparison_contract(
    models: dict[str, tuple[WorldModelSpec, dict[str, Any], Path]]
) -> dict[str, Any]:
    """Fail before execution if model-independent IDM protocol fields differ."""
    contracts: dict[str, dict[str, Any]] = {}
    output_dirs: set[Path] = set()
    for model_id, (_, config, path) in models.items():
        require_evaluation_scope(config)
        config_dir = path.parent
        evt_path = _resolve_path(config["paths"]["evt_model"], config_dir)
        idm_path = _resolve_path(config["paths"]["idm_ego_config"], config_dir)
        common_key = (
            "world_model_config"
            if model_id == "hierarchical"
            else "common_world_config"
        )
        common_path = _resolve_path(config["paths"][common_key], config_dir)
        subset = config["subset_simulation"]
        monte_carlo = config["monte_carlo"]
        contracts[model_id] = {
            "steps": int(config["simulation"].get("steps", 149)),
            "execution_backend": str(
                config["simulation"].get("execution_backend", "local_highway_env")
            ),
            "return_period": int(config["failure_event"].get("return_period", 100)),
            "evt_model_sha256": file_sha256(evt_path),
            "idm_ego_config_sha256": file_sha256(idm_path),
            "common_initial_prior_config_sha256": file_sha256(common_path),
            "subset_num_samples": int(subset["num_samples"]),
            "subset_p0": float(subset["p0"]),
            "subset_max_levels": int(subset["max_levels"]),
            "subset_pcn_beta": float(subset["pcn_beta"]),
            "subset_mcmc_steps": int(subset["mcmc_steps"]),
            "monte_carlo_num_samples": int(monte_carlo["num_samples"]),
            "evaluation_scope": evaluation_scope_contract(),
            "test_space": config.get("test_space", {"kind": "full_prior"}),
        }
        for estimator in ESTIMATORS:
            section = _CONFIG_SECTION[estimator]
            output_dir = _resolve_path(config[section]["output_dir"], config_dir)
            if output_dir in output_dirs:
                raise ValueError(
                    f"world models must not share result directories: {output_dir}"
                )
            output_dirs.add(output_dir)
    reference_id = next(iter(contracts))
    reference = contracts[reference_id]
    for model_id, contract in contracts.items():
        mismatches = {
            key: (reference[key], value)
            for key, value in contract.items()
            if value != reference[key]
        }
        if mismatches:
            details = ", ".join(
                f"{key}: {reference_id}={left!r}, {model_id}={right!r}"
                for key, (left, right) in mismatches.items()
            )
            raise ValueError(f"incomparable IDM evaluation configs: {details}")
    return {
        **reference,
        "dt_s": 0.04,
        "ego_controller": "native HighwayEnv IDMVehicle",
        "failure_metric": "shared trajectory event risk mapped by shared EVT model",
        "initial_scene_distribution": "shared declared test-space distribution",
        "comparison_semantics": (
            "protocol-matched independent estimates; model-specific latent priors "
            "are intentionally not treated as paired random variables"
        ),
    }


def _summary_path(config: dict[str, Any], config_path: Path, estimator: str) -> Path:
    output = _resolve_path(
        config[_CONFIG_SECTION[estimator]]["output_dir"], config_path.parent
    )
    filename = (
        "world_subset_summary.json"
        if estimator == "subset"
        else "world_monte_carlo_summary.json"
    )
    return output / filename


def _record(
    model_id: str,
    spec: WorldModelSpec,
    estimator: str,
    path: Path,
    summary: dict[str, Any],
) -> dict[str, Any]:
    uncertainty = summary.get("uncertainty", {})
    record = {
        "world_model_id": model_id,
        "world_model": spec.display_name,
        "estimator": estimator,
        "summary": str(path),
        "probability": float(summary["probability"]),
        "probability_standard_error": uncertainty.get("probability_standard_error"),
        "probability_ci95_lower": uncertainty.get("probability_ci95_lower"),
        "probability_ci95_upper": uncertainty.get("probability_ci95_upper"),
        "world_evaluations": int(
            summary.get("simulation_counts", {}).get("world_evaluations", 0)
        ),
    }
    if estimator == "subset":
        levels = summary.get("level_statistics", [])
        final = levels[-1] if levels else {}
        record.update(
            num_levels=int(summary.get("num_levels", 0)),
            final_failure_fraction=float(summary.get("final_failure_fraction", np.nan)),
            final_population_collision_fraction=final.get("collision_fraction"),
        )
    else:
        evt = summary.get("evt_score_summary", {})
        record.update(
            failure_count=int(summary.get("failure_count", 0)),
            collision_fraction=summary.get("collision_fraction"),
            numerical_valid_fraction=summary.get("numerical_valid_fraction"),
            evt_score_mean=evt.get("mean"),
            evt_score_p95=evt.get("p95"),
            evt_score_max=evt.get("max"),
        )
    return record


def _verify_result_contract(
    summaries: list[tuple[str, str, dict[str, Any]]], contract: dict[str, Any]
) -> None:
    expected_failure: dict[str, Any] | None = None
    expected_flow_sha256: str | None = None
    for model_id, estimator, summary in summaries:
        if summary.get("world_model_id") != model_id:
            raise ValueError(
                f"result world_model_id mismatch for {model_id}/{estimator}"
            )
        failure = summary.get("failure_event")
        if expected_failure is None:
            expected_failure = failure
        elif failure != expected_failure:
            raise ValueError(
                f"result failure threshold differs for {model_id}/{estimator}"
            )
        provenance = summary.get("provenance", {})
        flow_sha256 = provenance.get("flow_checkpoint_sha256")
        if not isinstance(flow_sha256, str) or not flow_sha256:
            raise ValueError(
                f"result is missing initial Flow provenance for {model_id}/{estimator}"
            )
        if expected_flow_sha256 is None:
            expected_flow_sha256 = flow_sha256
        elif flow_sha256 != expected_flow_sha256:
            raise ValueError(
                f"result initial Flow checkpoint differs for {model_id}/{estimator}"
            )
        evaluation_contract = summary.get("evaluation_contract", {})
        if int(evaluation_contract.get("steps", -1)) != int(contract["steps"]):
            raise ValueError(f"result step count differs for {model_id}/{estimator}")
        if evaluation_contract.get("population_scope") != evaluation_scope_contract():
            raise ValueError(
                f"result population scope differs for {model_id}/{estimator}"
            )
        checks = {
            "execution_backend": contract["execution_backend"],
            "evt_model_sha256": contract["evt_model_sha256"],
            "idm_ego_config_sha256": contract["idm_ego_config_sha256"],
        }
        for key, expected in checks.items():
            if provenance.get(key) != expected:
                raise ValueError(
                    f"result provenance mismatch for {model_id}/{estimator}: {key}"
                )
        initial_key = (
            "world_model_config_sha256"
            if model_id == "hierarchical"
            else "common_world_config_sha256"
        )
        if provenance.get(initial_key) != contract["common_initial_prior_config_sha256"]:
            raise ValueError(
                f"result initial-scene prior mismatch for {model_id}/{estimator}"
            )
        if estimator == "subset":
            populations = {
                int(level.get("num_samples", -1))
                for level in summary.get("level_statistics", [])
            }
            if populations != {int(contract["subset_num_samples"])}:
                raise ValueError(
                    f"result AMS population size differs for {model_id}/{estimator}"
                )
        elif int(summary.get("simulation_counts", {}).get("world_evaluations", -1)) != int(
            contract["monte_carlo_num_samples"]
        ):
            raise ValueError(
                f"result Monte-Carlo sample count differs for {model_id}/{estimator}"
            )


def _write_records_csv(path: Path, records: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for record in records:
        for key in record:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)


def _render_comparison(path: Path, records: list[dict[str, Any]]) -> None:
    estimators = [x for x in ESTIMATORS if any(r["estimator"] == x for r in records)]
    figure, axes = get_pyplot().subplots(
        1, len(estimators), figsize=(6.2 * len(estimators), 4.6), dpi=150
    )
    axes_list = [axes] if len(estimators) == 1 else list(axes)
    for axis, estimator in zip(axes_list, estimators):
        selected = [r for r in records if r["estimator"] == estimator]
        names = [r["world_model"] for r in selected]
        values = np.asarray([r["probability"] for r in selected], np.float64)
        lower = np.asarray([
            r["probability"]
            if r.get("probability_ci95_lower") is None
            else r["probability_ci95_lower"]
            for r in selected
        ], np.float64)
        upper = np.asarray([
            r["probability"]
            if r.get("probability_ci95_upper") is None
            else r["probability_ci95_upper"]
            for r in selected
        ], np.float64)
        positions = np.arange(len(selected))
        colors = [get_world_model(r["world_model_id"]).background_color for r in selected]
        axis.bar(positions, values, color=colors, alpha=0.82, width=0.62)
        axis.errorbar(
            positions,
            values,
            yerr=np.vstack([values - lower, upper - values]),
            fmt="none",
            color="#222222",
            capsize=4,
            linewidth=1.2,
        )
        axis.set_xticks(positions, names, rotation=12, ha="right")
        axis.set_ylabel("estimated failure probability")
        axis.set_title("AMS / subset simulation" if estimator == "subset" else "Independent Monte Carlo")
        axis.grid(axis="y", alpha=0.25)
        for position, value in zip(positions, values):
            axis.text(position, value, f"{value:.4g}", ha="center", va="bottom", fontsize=8)
    figure.suptitle("IDM risk under protocol-matched world models")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight")
    get_pyplot().close(figure)


def build_comparison_report(
    suite: dict[str, Any],
    suite_dir: Path,
    *,
    selected: Iterable[str] | None = None,
    require_all: bool = True,
) -> Path:
    models = load_model_configs(suite, suite_dir, selected)
    contract = validate_comparison_contract(models)
    summaries: list[tuple[str, str, dict[str, Any]]] = []
    records: list[dict[str, Any]] = []
    missing: list[str] = []
    for model_id, (spec, config, config_path) in models.items():
        for estimator in ESTIMATORS:
            path = _summary_path(config, config_path, estimator)
            if not path.is_file():
                missing.append(str(path))
                continue
            summary = load_json(path)
            summaries.append((model_id, estimator, summary))
            records.append(_record(model_id, spec, estimator, path, summary))
    if require_all and missing:
        raise FileNotFoundError("missing comparison result(s): " + ", ".join(missing))
    if not records:
        raise FileNotFoundError("no completed IDM world-model results are available")
    _verify_result_contract(summaries, contract)
    output_dir = _resolve_path(
        suite.get("output", {}).get("comparison_dir", "../results/comparisons"),
        suite_dir,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "world_model_comparison.csv"
    figure_path = output_dir / "world_model_comparison.png"
    _write_records_csv(csv_path, records)
    _render_comparison(figure_path, records)
    report = {
        "schema": "idm_multi_world_model_comparison_v1",
        "role": "protocol-matched comparison; each model retains its native latent prior",
        "comparison_contract": contract,
        "records": records,
        "missing_results": missing,
        "artifacts": {
            "table_csv": str(csv_path),
            "comparison_figure": str(figure_path),
        },
    }
    path = output_dir / "world_model_comparison.json"
    save_json(report, path)
    return path


def run_model_suite(
    suite: dict[str, Any],
    suite_dir: Path,
    *,
    selected: Iterable[str] | None = None,
    estimators: Iterable[str] = ESTIMATORS,
    formal: bool = True,
) -> list[Path]:
    models = load_model_configs(suite, suite_dir, selected)
    validate_comparison_contract(models)
    requested = tuple(estimators)
    unknown = set(requested).difference(ESTIMATORS)
    if unknown:
        raise ValueError(f"unknown estimator(s): {sorted(unknown)}")
    # Validate every selected release before the first expensive simulation.
    # This prevents a suite from writing one model and only then discovering
    # that another model is dirty, missing, or points at a stale checkpoint.
    if formal:
        for _, (spec, config, config_path) in models.items():
            spec.validate_formal_provenance(config, config_path.parent)
    results: list[Path] = []
    for _, (spec, config, config_path) in models.items():
        if "subset" in requested:
            results.append(spec.run_subset(config, config_path.parent, formal=formal))
        if "monte_carlo" in requested:
            results.append(spec.run_monte_carlo(config, config_path.parent, formal=formal))
    return results


def build_explicit_subset_comparison(
    hierarchical_summary: Path,
    trafficbots_summary: Path,
    output_dir: Path,
) -> Path:
    """Compare an existing hierarchical AMS result with a TrafficBots AMS result.

    This explicit-source entry point supports the maintained historical
    hierarchical directory while new model-organized runs are being produced.
    It never upgrades development provenance to a formal result.
    """
    sources = {
        "hierarchical": hierarchical_summary.resolve(),
        "trafficbots": trafficbots_summary.resolve(),
    }
    summaries = {model_id: load_json(path) for model_id, path in sources.items()}
    hierarchy = summaries["hierarchical"]
    trafficbots = summaries["trafficbots"]
    if hierarchy.get("schema") != "highway_env_idm_subset_simulation":
        raise ValueError("hierarchical source is not an IDM subset summary")
    if trafficbots.get("schema") != "trafficbots_highway_env_idm_subset_simulation":
        raise ValueError("TrafficBots source is not an IDM subset summary")
    if hierarchy.get("failure_event") != trafficbots.get("failure_event"):
        raise ValueError("subset summaries use different failure thresholds")
    if (
        hierarchy.get("evaluation_contract", {}).get("test_space")
        != trafficbots.get("evaluation_contract", {}).get("test_space")
    ):
        raise ValueError(
            "subset summaries use different test spaces; a fixed-K_GT empirical "
            "hierarchical result cannot be ranked against a TrafficBots prior result"
        )
    for model_id, summary in summaries.items():
        scope = summary.get("evaluation_contract", {}).get("population_scope")
        if scope != evaluation_scope_contract():
            raise ValueError(
                f"subset summary uses a stale population scope: {model_id}"
            )
    shared_provenance = (
        "flow_checkpoint_sha256",
        "evt_model_sha256",
        "idm_ego_config_sha256",
        "execution_backend",
    )
    for key in shared_provenance:
        if hierarchy.get("provenance", {}).get(key) != trafficbots.get(
            "provenance", {}
        ).get(key):
            raise ValueError(f"subset summaries are not comparable: {key}")
    hierarchy_steps = int(hierarchy.get("dimensions", {}).get("response_steps", -1))
    trafficbots_steps = int(
        trafficbots.get("evaluation_contract", {}).get("steps", -1)
    )
    if hierarchy_steps != trafficbots_steps or hierarchy_steps < 1:
        raise ValueError("subset summaries use different rollout horizons")
    settings = {}
    for model_id, summary in summaries.items():
        levels = summary.get("level_statistics", [])
        sample_sizes = {int(level.get("num_samples", -1)) for level in levels}
        if len(sample_sizes) != 1 or -1 in sample_sizes:
            raise ValueError(f"invalid AMS population metadata for {model_id}")
        kernel = summary.get("mutation_kernel", {})
        settings[model_id] = {
            "num_samples": sample_sizes.pop(),
            "pcn_beta": float(kernel.get("pcn_beta", np.nan)),
            "mcmc_steps": int(kernel.get("mcmc_steps", -1)),
        }
    if settings["hierarchical"] != settings["trafficbots"]:
        raise ValueError(f"subset estimator settings differ: {settings}")

    records = []
    for model_id, summary in summaries.items():
        spec = get_world_model(model_id)
        uncertainty = summary["uncertainty"]
        formal = bool(
            summary.get("formal", not summary.get("provenance", {}).get("worktree_dirty", True))
        )
        record = {
                "world_model_id": model_id,
                "world_model": spec.display_name,
                "formal": formal,
                "probability": float(summary["probability"]),
                "probability_standard_error": float(
                    uncertainty["probability_standard_error"]
                ),
                "probability_ci95_lower": float(
                    uncertainty["probability_ci95_lower"]
                ),
                "probability_ci95_upper": float(
                    uncertainty["probability_ci95_upper"]
                ),
                "num_levels": int(summary["num_levels"]),
                "world_evaluations": int(
                    summary["simulation_counts"]["world_evaluations"]
                ),
                "prior_level_failure_fraction": float(
                    summary["level_statistics"][0]["failure_fraction"]
                ),
                "prior_level_collision_fraction": float(
                    summary["level_statistics"][0]["collision_fraction"]
                ),
                "final_population_collision_fraction": float(
                    summary["level_statistics"][-1]["collision_fraction"]
                ),
                "summary": str(sources[model_id]),
            }
        manifest_path = sources[model_id].parent / "playbacks/playback_manifest.json"
        if manifest_path.is_file():
            episodes = load_json(manifest_path).get("episodes", [])
            collision_times = [
                float(episode["first_collision_time_s"])
                for episode in episodes
                if episode.get("first_collision_time_s") is not None
            ]
            record.update(
                top_case_playback_manifest=str(manifest_path),
                top_case_count=len(episodes),
                top_case_collision_fraction=float(
                    np.mean([bool(episode.get("collision")) for episode in episodes])
                )
                if episodes
                else None,
                top_case_first_collision_time_min_s=float(min(collision_times))
                if collision_times
                else None,
                top_case_first_collision_time_median_s=float(
                    np.median(collision_times)
                )
                if collision_times
                else None,
                top_case_first_collision_time_max_s=float(max(collision_times))
                if collision_times
                else None,
            )
        records.append(record)
    by_id = {record["world_model_id"]: record for record in records}
    h_probability = by_id["hierarchical"]["probability"]
    t_probability = by_id["trafficbots"]["probability"]
    effect = {
        "trafficbots_minus_hierarchical": float(t_probability - h_probability),
        "trafficbots_over_hierarchical": float(t_probability / h_probability),
        "trafficbots_relative_change": float(
            (t_probability - h_probability) / h_probability
        ),
        "ci95_overlap": bool(
            by_id["trafficbots"]["probability_ci95_lower"]
            <= by_id["hierarchical"]["probability_ci95_upper"]
            and by_id["hierarchical"]["probability_ci95_lower"]
            <= by_id["trafficbots"]["probability_ci95_upper"]
        ),
    }
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "subset_comparison.csv"
    _write_records_csv(csv_path, records)
    figure_path = output_dir / "subset_comparison.png"
    plt = get_pyplot()
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.6), dpi=150)
    positions = np.arange(2)
    values = np.asarray([record["probability"] for record in records])
    lower = np.asarray([record["probability_ci95_lower"] for record in records])
    upper = np.asarray([record["probability_ci95_upper"] for record in records])
    colors = [get_world_model(record["world_model_id"]).background_color for record in records]
    labels = [record["world_model"] for record in records]
    axes[0].bar(positions, values, color=colors, width=0.62, alpha=0.85)
    axes[0].errorbar(
        positions,
        values,
        yerr=np.vstack([values - lower, upper - values]),
        fmt="none",
        color="#222222",
        capsize=4,
    )
    axes[0].set_xticks(positions, labels, rotation=10, ha="right")
    axes[0].set_ylabel("estimated failure probability")
    axes[0].set_title("AMS probability with reported 95% interval")
    axes[0].grid(axis="y", alpha=0.25)
    for position, value in zip(positions, values):
        axes[0].text(position, value, f"{value:.5f}", ha="center", va="bottom")
    for record, color in zip(records, colors):
        levels = summaries[record["world_model_id"]]["level_statistics"]
        axes[1].plot(
            [int(level["level"]) for level in levels],
            [float(level["failure_fraction"]) for level in levels],
            marker="o",
            linewidth=2.0,
            color=color,
            label=record["world_model"],
        )
    axes[1].set(
        xlabel="AMS level",
        ylabel="fraction above final failure threshold",
        title="Tail enrichment across AMS levels",
        ylim=(-0.02, 1.02),
    )
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    figure.suptitle("TrafficBots vs hierarchical world model under the shared IDM protocol")
    figure.tight_layout()
    figure.savefig(figure_path, bbox_inches="tight")
    plt.close(figure)
    publishable = bool(all(record["formal"] for record in records))
    report = {
        "schema": "idm_explicit_subset_world_model_comparison_v1",
        "publishable": publishable,
        "status": "formal" if publishable else "development_mixed_provenance",
        "warning": None
        if publishable
        else (
            "At least one source was produced from a dirty worktree; use for "
            "engineering analysis only, not as a final paper result."
        ),
        "shared_contract": {
            "population_scope": evaluation_scope_contract(),
            "steps": hierarchy_steps,
            "dt_s": 0.04,
            "failure_event": hierarchy["failure_event"],
            "estimator": "adaptive_multilevel_splitting_pcn_subset_simulation",
            **settings["hierarchical"],
            "shared_provenance": {
                key: hierarchy["provenance"][key] for key in shared_provenance
            },
        },
        "records": records,
        "effect": effect,
        "interpretation_limits": [
            "Reported AMS intervals use the existing final-level binomial approximation and do not account for MCMC correlation.",
            "Final-population collision fractions and top-case playback statistics are tail-conditioned diagnostics, not unconditional probabilities.",
            "The two models share p(M,C0) as a distribution but these runs are independent rather than sample-wise paired.",
        ],
        "artifacts": {
            "table_csv": str(csv_path),
            "comparison_figure": str(figure_path),
        },
    }
    path = output_dir / "subset_comparison.json"
    save_json(report, path)
    return path
