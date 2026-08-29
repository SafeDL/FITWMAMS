#!/usr/bin/env python3
"""Render model-specific HighwayEnv playbacks for retained AMS cases."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from IDM_subset.src.world_model_registry import get_world_model  # noqa: E402
from hierarchical_world_model.src.visualization import (  # noqa: E402
    EGO_COLOR,
    ROAD_COLOR,
    _draw_lane_markings,
    _draw_vehicle,
)
from tools.plot_style import get_pyplot  # noqa: E402
from world_model.src.core.utils import load_json, load_yaml, save_json  # noqa: E402
from world_model.src.core.evaluation_scope import (  # noqa: E402
    evaluation_scope_contract,
    scoped_agent_valid,
)


def _rank(values: np.ndarray) -> np.ndarray:
    """Return deterministic normalized ranks without assuming metric scales."""
    value = np.asarray(values, np.float64)
    if len(value) <= 1:
        return np.zeros(len(value), np.float64)
    order = np.argsort(value, kind="stable")
    result = np.empty(len(value), np.float64)
    result[order] = np.linspace(0.0, 1.0, len(value))
    return result


def _diverse_tail_indices(
    evt_score: np.ndarray,
    event_risk: np.ndarray,
    collision: np.ndarray,
    min_gap_m: np.ndarray,
    *,
    failure_threshold: float,
    count: int,
) -> np.ndarray:
    """Pick representative AMS failures, rather than repeatedly picking maxima.

    The AMS final population is a conditional tail distribution.  Selection by
    risk alone therefore concentrates figures on one near-gap mechanism.  This
    selector reserves some collision examples and then uses farthest-point
    sampling over score, risk and realized clearance ranks for non-collision
    failures.  It is a visualization policy only; it never changes an AMS
    estimate or its retained population.
    """
    score = np.asarray(evt_score, np.float64)
    risk = np.asarray(event_risk, np.float64)
    hit = np.asarray(collision, bool)
    gap = np.asarray(min_gap_m, np.float64)
    if not (score.shape == risk.shape == hit.shape == gap.shape):
        raise ValueError("tail-selection arrays must have matching shapes")
    failures = np.flatnonzero(score >= float(failure_threshold))
    if not len(failures) or int(count) < 1:
        return np.empty(0, np.int64)
    limit = min(int(count), len(failures))
    selected: list[int] = []
    # Collisions are a distinct failure mechanism.  Preserve up to three
    # representatives over its severity range, rather than filling every GIF
    # with them.
    collisions = failures[hit[failures]]
    if len(collisions):
        ordered = collisions[np.argsort(score[collisions], kind="stable")]
        for offset in np.linspace(0, len(ordered) - 1, min(3, len(ordered))).round().astype(int):
            candidate = int(ordered[offset])
            if candidate not in selected and len(selected) < limit:
                selected.append(candidate)
    pool = np.asarray([index for index in failures if index not in selected], np.int64)
    if not len(pool):
        return np.asarray(selected, np.int64)
    descriptor = np.stack(
        (
            _rank(score[pool]),
            _rank(risk[pool]),
            _rank(np.minimum(gap[pool], 30.0)),
        ),
        axis=1,
    )
    # Anchor the non-collision set at a near-threshold failure, then maximize
    # coverage in metric space.  Stable tie-breaking preserves reproducibility.
    anchor = int(np.argmin(score[pool]))
    selected.append(int(pool[anchor]))
    chosen = [anchor]
    while len(selected) < limit and len(chosen) < len(pool):
        distances = np.linalg.norm(
            descriptor[:, None, :] - descriptor[np.asarray(chosen)][None, :, :],
            axis=-1,
        ).min(axis=1)
        distances[np.asarray(chosen)] = -np.inf
        next_index = int(np.argmax(distances))
        selected.append(int(pool[next_index]))
        chosen.append(next_index)
    return np.asarray(selected, np.int64)


def _diverse_hierarchical_cases(
    subset_dir: Path, count: int, evaluator: Any, *, overwrite: bool
) -> list[dict[str, Any]]:
    """Screen every AMS failure and materialize semantic-diverse tail cases."""
    from hierarchical_world_model.src.execution import rollout_world
    from hierarchical_world_model.src.randomness import WorldExogenousState
    from hierarchical_world_model.src.empirical_context import EmpiricalKWorldState

    summary = load_json(subset_dir / "world_subset_summary.json")
    if summary.get("evaluation_contract", {}).get("population_scope") != evaluation_scope_contract():
        raise ValueError("subset summary uses a stale population scope")
    with np.load(subset_dir / "world_subset_final_population.npz") as population:
        common = (
            "diffusion_noise", "scene_innovations", "agent_response_innovations",
            "evt_score", "event_risk", "collision", "min_gap_m",
        )
        empirical = "test_row_uniform" in population.files
        random_fields = (
            ("test_row_uniform", *common[:3])
            if empirical
            else (
                "scenario_uniform", "c0_base_latent", "k_base_latent",
                *common[:3],
            )
        )
        required = (*random_fields, *common[3:])
        if not set(required).issubset(population.files):
            raise ValueError("final population lacks diversity-selection fields")
        values = {name: np.asarray(population[name]) for name in required}
    threshold = float(summary["failure_event"]["evt_score_threshold"])
    failures = np.flatnonzero(values["evt_score"] >= threshold)
    world_type = EmpiricalKWorldState if empirical else WorldExogenousState
    worlds = world_type(**{name: values[name] for name in random_fields})
    records: list[dict[str, Any]] = []
    for start in range(0, len(failures), int(evaluator.batch_size)):
        indices = failures[start : start + int(evaluator.batch_size)]
        rollout = rollout_world(
            evaluator.sampler, worlds.select(indices), evaluator.policy,
            steps=evaluator.steps, evt_model=evaluator.evt_model,
        )
        for local, index in enumerate(indices):
            records.append({
                "population_index": int(index),
                "semantic_tags": _scenario_semantics(
                    rollout.states[local], rollout.initial_valid[local]
                ),
            })
    by_index = {item["population_index"]: item for item in records}
    selected: list[int] = []
    # Prefer semantically distinctive mechanisms.  A mechanism receives a
    # low- and high-severity representative when both exist.
    for tag in ("cut_in", "cut_out", "collision", "lateral_activity", "initial_close_front", "longitudinal_tail"):
        candidates = [
            item["population_index"] for item in records
            if tag in item["semantic_tags"] and item["population_index"] not in selected
        ]
        if not candidates:
            continue
        ordered = sorted(candidates, key=lambda item: float(values["evt_score"][item]))
        for offset in np.linspace(0, len(ordered) - 1, min(2, len(ordered))).round().astype(int):
            candidate = int(ordered[offset])
            if candidate not in selected and len(selected) < int(count):
                selected.append(candidate)
    # Fill the remaining display budget with metric-space coverage from the
    # complete failure cohort, never just the pre-existing top-ten archive.
    for candidate in _diverse_tail_indices(
        values["evt_score"], values["event_risk"], values["collision"], values["min_gap_m"],
        failure_threshold=threshold, count=len(failures),
    ):
        if int(candidate) not in selected and len(selected) < int(count):
            selected.append(int(candidate))
    selected_dir = subset_dir / "failure_cases"
    selected_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for path in selected_dir.iterdir():
            if path.is_file():
                path.unlink()
    cases: list[dict[str, Any]] = []
    for ordinal, index in enumerate(selected, start=1):
        world = world_type(
            **{name: values[name][index : index + 1] for name in random_fields}
        )
        case_id = f"hierarchical_semantic_tail_{ordinal:02d}"
        world_path = selected_dir / f"{case_id}.npz"
        world.save(world_path)
        cases.append({
            "case_id": case_id,
            "world_exogenous_state": str(world_path.resolve()),
            "event_risk": float(values["event_risk"][index]),
            "evt_score": float(values["evt_score"][index]),
            "collision": bool(values["collision"][index]),
            "min_gap_m": float(values["min_gap_m"][index]),
            "selection_population_index": int(index),
            "semantic_tags": by_index[int(index)]["semantic_tags"],
        })
    counts = {
        tag: int(sum(tag in item["semantic_tags"] for item in records))
        for tag in ("cut_in", "cut_out", "collision", "lateral_activity", "initial_close_front", "longitudinal_tail")
    }
    save_json(
        {
            "schema": "hierarchical_ams_semantic_tail_cases_v1",
            "selection": "semantic-strata representatives plus score/risk/clearance coverage",
            "source_population": str(subset_dir / "world_subset_final_population.npz"),
            "failure_threshold": threshold,
            "failure_worlds_screened": int(len(failures)),
            "semantic_counts": counts,
            "cases": cases,
        },
        selected_dir / "semantic_cases.json",
    )
    return cases


def _diverse_empirical_sweep_cases(
    sweep_dir: Path, count: int, evaluator: Any
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Select semantic-diverse exact failures from an exhaustive test sweep.

    Unlike the AMS final population, this source has one fixed CRN rollout for
    every held-out context.  All listed cases have already crossed the fixed
    EVT threshold; replay is only used to attach transparent geometric tags
    before choosing figures.
    """
    from IDM_subset.src.world_model_registry import get_world_model

    cases = load_json(sweep_dir / "test_sweep_failure_cases.json")
    if not cases or int(count) < 1:
        return [], {tag: 0 for tag in (
            "cut_in", "cut_out", "collision", "lateral_activity",
            "initial_close_front", "longitudinal_tail",
        )}
    spec = get_world_model("hierarchical")
    records: list[dict[str, Any]] = []
    for case in cases:
        case_path = Path(case["world_exogenous_state"])
        if not case_path.is_absolute():
            case_path = (sweep_dir / case_path).resolve()
        rollout, _ = spec.replay_case(evaluator, case_path)
        record = dict(case)
        record["semantic_tags"] = _scenario_semantics(
            rollout.states[0], rollout.initial_valid[0]
        )
        records.append(record)
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    for tag in (
        "cut_in", "cut_out", "collision", "lateral_activity",
        "initial_close_front", "longitudinal_tail",
    ):
        candidates = [item for item in records if tag in item["semantic_tags"]]
        if not candidates:
            continue
        candidates.sort(key=lambda item: float(item["evt_score"]))
        for offset in np.linspace(0, len(candidates) - 1, min(2, len(candidates))).round().astype(int):
            candidate = candidates[int(offset)]
            if candidate["case_id"] not in selected_ids and len(selected) < int(count):
                selected.append(candidate)
                selected_ids.add(candidate["case_id"])
    values = {
        name: np.asarray([item[name] for item in records])
        for name in ("evt_score", "event_risk", "collision", "min_gap_m")
    }
    threshold = float(load_json(sweep_dir / "test_context_sweep_summary.json")["failure_event"]["evt_score_threshold"])
    for index in _diverse_tail_indices(
        values["evt_score"], values["event_risk"], values["collision"],
        values["min_gap_m"], failure_threshold=threshold, count=len(records),
    ):
        candidate = records[int(index)]
        if candidate["case_id"] not in selected_ids and len(selected) < int(count):
            selected.append(candidate)
            selected_ids.add(candidate["case_id"])
    counts = {
        tag: int(sum(tag in item["semantic_tags"] for item in records))
        for tag in (
            "cut_in", "cut_out", "collision", "lateral_activity",
            "initial_close_front", "longitudinal_tail",
        )
    }
    return selected, counts


def _local_coordinates(states: np.ndarray) -> np.ndarray:
    local = np.asarray(states, np.float32).copy()
    local[..., :2] -= local[0:1, 0:1, :2]
    return local


def _collision_mask(states: np.ndarray, valid: np.ndarray) -> np.ndarray:
    ego = states[:, :, :1]
    background = states[:, :, 1:]
    valid = np.asarray(scoped_agent_valid(valid), bool)
    active = valid[:, 1:][:, None]
    longitudinal = background[..., 0] - ego[..., 0]
    lateral = np.abs(background[..., 1] - ego[..., 1])
    return active & (np.abs(longitudinal) < 4.8) & (lateral < 1.8)


def _scenario_semantics(states: np.ndarray, valid: np.ndarray) -> list[str]:
    """Assign transparent geometric labels to one generated IDM rollout.

    These are display strata, not ground-truth causal labels.  In particular,
    a cut-in means an initially adjacent-lane background vehicle enters the
    ego lane while longitudinally relevant during the generated rollout.
    """
    values = np.asarray(states, np.float32)
    present = np.asarray(scoped_agent_valid(valid), bool)
    relative_x = values[:, 1:, 0] - values[:, :1, 0]
    relative_y = values[:, 1:, 1] - values[:, :1, 1]
    active = present[1:]
    same_lane = np.abs(relative_y) < 1.8
    initially_same = same_lane[0] & active
    entered = (~initially_same) & active & np.any(
        same_lane & (relative_x > -4.8), axis=0
    )
    exited = initially_same & np.any(~same_lane, axis=0)
    initial_front_gap = np.where(
        initially_same & (relative_x[0] > 0.0),
        np.maximum(relative_x[0] - 4.8, 0.0),
        np.inf,
    )
    tags: list[str] = []
    if np.any(entered):
        tags.append("cut_in")
    if np.any(exited):
        tags.append("cut_out")
    if np.any(initial_front_gap < 10.0):
        tags.append("initial_close_front")
    if np.any(np.abs(relative_y - relative_y[:1]) > 1.0, axis=0)[active].any():
        tags.append("lateral_activity")
    if _collision_mask(values[None], present[None]).any():
        tags.append("collision")
    return tags or ["longitudinal_tail"]


def _render_case(
    *,
    path: Path,
    case: dict[str, Any],
    states: np.ndarray,
    valid: np.ndarray,
    frame_stride: int,
    model_name: str,
    background_label: str,
    background_color: str,
    background_color_name: str,
) -> dict[str, Any]:
    valid = np.asarray(scoped_agent_valid(valid), bool)
    local = _local_coordinates(states)
    collision = _collision_mask(states[None], valid[None])[0]
    frames = np.arange(0, len(states), max(int(frame_stride), 1))
    if frames[-1] != len(states) - 1:
        frames = np.append(frames, len(states) - 1)
    collision_frames = np.flatnonzero(collision.any(axis=1))
    first_collision = None if not len(collision_frames) else int(collision_frames[0])
    semantics = _scenario_semantics(states, valid)
    plt = get_pyplot()
    figure, axis = plt.subplots(figsize=(12.0, 4.8), dpi=100)
    figure.subplots_adjust(left=0.065, right=0.965, bottom=0.18, top=0.84)
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(
        path,
        mode="I",
        duration=40 * max(int(frame_stride), 1),
        loop=0,
    ) as writer:
        for frame in frames:
            axis.clear()
            axis.set_facecolor(ROAD_COLOR)
            center_x = float(local[frame, 0, 0])
            _draw_lane_markings(axis)
            axis.set(
                xlim=(center_x - 60.0, center_x + 60.0),
                ylim=(-8.2, 8.2),
                xlabel="relative longitudinal position [m]",
                ylabel="relative lateral position [m]",
                aspect="equal",
                title=(
                    f"{model_name} + IDM | {case['case_id']} | t={frame * 0.04:.2f}s | "
                    f"S_EVT={float(case['evt_score']):.2f}"
                ),
            )
            trail_start = max(0, frame - 45)
            axis.plot(
                local[trail_start : frame + 1, 0, 0],
                local[trail_start : frame + 1, 0, 1],
                color=EGO_COLOR,
                linewidth=2.0,
                label="IDM ego",
            )
            for slot in np.flatnonzero(valid[1:]):
                axis.plot(
                    local[trail_start : frame + 1, slot + 1, 0],
                    local[trail_start : frame + 1, slot + 1, 1],
                    color=background_color,
                    linewidth=1.35,
                    alpha=0.85,
                )
                colliding = bool(collision[frame, slot])
                _draw_vehicle(
                    axis,
                    local[frame, slot + 1],
                    color="#ffbf00" if colliding else background_color,
                    label=f"collision background b{slot + 1}" if colliding else None,
                    filled=True,
                    alpha=0.9 if colliding else 0.52,
                )
            _draw_vehicle(
                axis,
                local[frame, 0],
                color="#ffbf00" if collision[frame].any() else EGO_COLOR,
                label="IDM ego",
                filled=True,
                alpha=0.92,
            )
            axis.text(
                0.01,
                0.02,
                f"red: IDM ego | {background_color_name}: {background_label} | "
                "yellow: footprint overlap",
                transform=axis.transAxes,
                fontsize=7.5,
                va="bottom",
                ha="left",
                bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none"},
            )
            axis.tick_params(labelsize=8)
            figure.canvas.draw()
            writer.append_data(
                np.asarray(figure.canvas.buffer_rgba())[:, :, :3].copy()
            )
    plt.close(figure)
    return {
        "case_id": case["case_id"],
        "gif": str(path),
        "world_exogenous_state": str(case["world_exogenous_state"]),
        "event_risk": float(case["event_risk"]),
        "evt_score": float(case["evt_score"]),
        "collision": bool(case["collision"]),
        "semantic_tags": semantics,
        "first_collision_frame": first_collision,
        "first_collision_time_s": None
        if first_collision is None
        else float(first_collision * 0.04),
        "frame_stride": int(frame_stride),
        "playback_frames": int(len(frames)),
    }


def render_subset_playbacks(
    *,
    model_id: str,
    config_path: Path,
    top_k: int = 10,
    frame_stride: int = 2,
    output_dir: Path | None = None,
    selection: str = "risk",
    overwrite: bool = False,
) -> Path:
    """Replay and render exact retained cases for one registered world model."""
    spec = get_world_model(model_id)
    config_path = config_path.resolve()
    config = load_yaml(config_path)
    subset_dir = (
        config_path.parent / config["subset_simulation"]["output_dir"]
    ).resolve()
    if selection not in {"risk", "semantic_diverse", "test_sweep_diverse"}:
        raise ValueError(
            "selection must be 'risk', 'semantic_diverse', or "
            "'test_sweep_diverse'"
        )
    target_dir = output_dir.resolve() if output_dir is not None else subset_dir / "playbacks"
    if overwrite and target_dir.exists():
        for path in target_dir.iterdir():
            if path.is_file():
                path.unlink()
    semantic_counts = None
    source_summary = subset_dir / "world_subset_summary.json"
    if selection == "risk":
        cases = load_json(subset_dir / "world_subset_top_cases.json")[: max(int(top_k), 0)]
    elif selection == "semantic_diverse" and spec.model_id == "hierarchical":
        evaluator, provenance = spec.build_evaluator(config, config_path.parent)
        cases = _diverse_hierarchical_cases(
            subset_dir, max(int(top_k), 0), evaluator, overwrite=overwrite
        )
    elif selection == "test_sweep_diverse" and spec.model_id == "hierarchical":
        sweep_dir = (config_path.parent / config["test_sweep"]["output_dir"]).resolve()
        if output_dir is None:
            target_dir = sweep_dir / "playbacks"
        evaluator, provenance = spec.build_evaluator(config, config_path.parent)
        cases, semantic_counts = _diverse_empirical_sweep_cases(
            sweep_dir, max(int(top_k), 0), evaluator
        )
        source_summary = sweep_dir / "test_context_sweep_summary.json"
    else:
        raise NotImplementedError(
            "semantic-diverse playbacks are currently available for hierarchical worlds"
        )
    if selection in {"risk", "semantic_diverse"}:
        evaluator, provenance = spec.build_evaluator(config, config_path.parent)
    episodes = []
    for case in cases:
        case_path = Path(case["world_exogenous_state"])
        if not case_path.is_absolute():
            case_path = (subset_dir / case_path).resolve()
        rollout, _ = spec.replay_case(evaluator, case_path)
        episodes.append(
            _render_case(
                path=target_dir / f"{case['case_id']}.gif",
                case=case,
                states=rollout.states[0],
                valid=rollout.initial_valid[0],
                frame_stride=frame_stride,
                model_name=spec.display_name,
                background_label=spec.background_label,
                background_color=spec.background_color,
                background_color_name=spec.background_color_name,
            )
        )
    manifest = target_dir / "playback_manifest.json"
    save_json(
        {
            "schema": "highway_env_idm_subset_playbacks_v4",
            "role": (
                "model-specific playbacks from the exhaustive held-out test sweep"
                if selection == "test_sweep_diverse"
                else "model-specific playbacks from the retained AMS final population"
            ),
            "selection": selection,
            "world_model_id": spec.model_id,
            "world_model": spec.display_name,
            "evaluation_scope": evaluation_scope_contract(),
            "source_summary": str(source_summary),
            "semantic_counts_over_source_failures": semantic_counts,
            "visual_contract": {
                "coordinate_frame": "S0 ego-relative",
                "x_window_m": [-60.0, 60.0],
                "y_window_m": [-8.2, 8.2],
                "ego_color": EGO_COLOR,
                "background_color": spec.background_color,
                "background_color_name": spec.background_color_name,
                "collision_color": "#ffbf00",
            },
            "provenance": provenance,
            "episodes": episodes,
        },
        manifest,
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render GIFs from the exact retained AMS subset worlds."
    )
    parser.add_argument(
        "--model",
        default="hierarchical",
        help="registered world model (hierarchical or trafficbots)",
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--selection",
        choices=("risk", "semantic_diverse", "test_sweep_diverse"),
        default="risk",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    spec = get_world_model(args.model)
    config_path = (
        args.config.resolve() if args.config is not None else spec.default_config
    )
    manifest = render_subset_playbacks(
        model_id=spec.model_id,
        config_path=config_path,
        top_k=args.top_k,
        frame_stride=args.frame_stride,
        output_dir=args.output_dir,
        selection=args.selection,
        overwrite=args.overwrite,
    )
    episodes = load_json(manifest)["episodes"]
    print(f"rendered {len(episodes)} {spec.model_id} playbacks in {manifest.parent}")


if __name__ == "__main__":
    main()
