"""Shared command-line entrypoints for A2C subset scripts."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from diffusion.src.utils import load_yaml, setup_logging
from A2C_subset.src.latent_subset_runner import (
    run_monte_carlo_from_config,
    run_subset_from_config,
)
from A2C_subset.src.result_payload import compact_a2c_result


logger = logging.getLogger(__name__)
MAX_MONTE_CARLO_SAMPLES = 200_000


def _override_if_set(
    config: dict[str, Any],
    section: str,
    key: str,
    value: Any,
) -> None:
    if value is not None:
        config.setdefault(section, {})[key] = value


def run_subset_entrypoint(
    *,
    default_config_path: Path,
    event_type: str,
    label: str,
    result_filename: str,
) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(default_config_path))
    parser.add_argument("--checkpoint_path", help="A2C stable-baselines3 checkpoint path.")
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use deterministic policy inference.",
    )
    parser.add_argument("--seed", type=int, help="Shared simulation/policy seed.")
    parser.add_argument("--num_samples", type=int, help="Subset simulation N.")
    parser.add_argument("--p0", type=float, help="Subset conditional level probability.")
    parser.add_argument("--max_levels", type=int, help="Maximum subset levels.")
    parser.add_argument("--proposal_std", type=float, help="Latent random-walk proposal std.")
    parser.add_argument(
        "--context_refresh_prob",
        type=float,
        help="Probability of refreshing scenario context during MH proposal.",
    )
    parser.add_argument(
        "--mh_retries_per_sample",
        type=int,
        help="MH proposal retries per generated sample.",
    )
    args = parser.parse_args()
    setup_logging("INFO")
    config_path = Path(args.config).resolve()
    config = load_yaml(config_path)

    if args.checkpoint_path:
        config.setdefault("a2c_policy", {})["checkpoint_path"] = args.checkpoint_path
    if args.deterministic is not None:
        config.setdefault("a2c_policy", {})["deterministic"] = bool(args.deterministic)
    if args.seed is not None:
        config.setdefault("training", {})["seed"] = int(args.seed)
        config.setdefault("context_sampling", {})["seed"] = int(args.seed)

    _override_if_set(config, "subset_simulation", "num_samples", args.num_samples)
    _override_if_set(config, "subset_simulation", "p0", args.p0)
    _override_if_set(config, "subset_simulation", "max_levels", args.max_levels)
    _override_if_set(config, "subset_simulation", "proposal_std", args.proposal_std)
    _override_if_set(
        config,
        "subset_simulation",
        "context_refresh_prob",
        args.context_refresh_prob,
    )
    _override_if_set(
        config,
        "subset_simulation",
        "mh_retries_per_sample",
        args.mh_retries_per_sample,
    )

    summary_path = run_subset_from_config(
        config,
        config_path.parent,
        expected_event_type=event_type,
    )
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    result_payload = compact_a2c_result(
        summary,
        summary_path=summary_path,
        config=config,
        config_dir=config_path.parent,
    )
    result_path = summary_path.with_name(result_filename)
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result_payload, f, indent=2, sort_keys=True)
        f.write("\n")

    counts = dict(summary.get("simulation_counts", {}) or {})
    reliability = dict(summary.get("reliability", {}) or {})
    logger.info(
        (
            "%s subset summary: probability=%.8g se=%.3g "
            "levels=%s stop_reason=%s reliability=%s"
        ),
        label,
        float(summary.get("probability", float("nan"))),
        float(summary.get("probability_standard_error", float("nan"))),
        summary.get("num_levels"),
        summary.get("stop_reason"),
        reliability.get("status"),
    )
    logger.info("%s A2C result output: %s", label, result_path)
    logger.info(
        (
            "%s subset actual simulated scenario count: "
            "closed_loop_evaluations=%s stored_level_samples=%s "
            "unique_context_indices_all_levels=%s "
            "unique_context_indices_final_level=%s"
        ),
        label,
        counts.get("closed_loop_evaluations"),
        counts.get("stored_level_samples"),
        counts.get("unique_context_indices_all_levels"),
        counts.get("unique_context_indices_final_level"),
    )


def run_monte_carlo_entrypoint(
    *,
    default_config_path: Path,
    event_type: str,
) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(default_config_path))
    parser.add_argument(
        "--num_samples",
        type=int,
        help="Override monte_carlo.num_samples for this run.",
    )
    parser.add_argument("--seed", type=int, help="Override monte_carlo.seed.")
    args = parser.parse_args()
    setup_logging("INFO")
    config_path = Path(args.config).resolve()
    config = load_yaml(config_path)
    if args.num_samples is not None:
        if int(args.num_samples) > MAX_MONTE_CARLO_SAMPLES:
            parser.error(f"--num_samples must be <= {MAX_MONTE_CARLO_SAMPLES}")
        config.setdefault("monte_carlo", {})["num_samples"] = int(args.num_samples)
    if args.seed is not None:
        config.setdefault("monte_carlo", {})["seed"] = int(args.seed)
    run_monte_carlo_from_config(
        config,
        config_path.parent,
        expected_event_type=event_type,
    )


def play_final_level_entrypoint(
    *,
    default_config_path: Path,
    event_type: str,
    label: str,
) -> None:
    from A2C_subset.src import final_level_playback as playback

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(default_config_path))
    parser.add_argument("--samples-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--num-cases",
        type=int,
        default=10,
        help="Maximum final-level failure test scenarios to render.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Seed for random final-level failure case selection.",
    )
    parser.add_argument("--level", type=int, default=-1)
    parser.add_argument("--no-gif", action="store_true", help="Render overview PNGs only.")
    parser.add_argument(
        "--background-config",
        default=None,
        help="Path to process_highD config used to replay background traffic.",
    )
    parser.add_argument(
        "--no-background",
        action="store_true",
        help="Do not overlay highD background traffic in GIF playback.",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    playback.SCRIPT_DEFAULTS.update(
        {
            "samples_path": args.samples_path,
            "output_dir": args.output_dir,
            "num_cases": int(args.num_cases),
            "random_seed": int(args.random_seed),
            "level": int(args.level),
            "unique_test_scenarios": True,
            "render_gif": not bool(args.no_gif),
            "render_background": not bool(args.no_background),
            "log_level": str(args.log_level),
        }
    )
    if args.background_config:
        playback.SCRIPT_DEFAULTS["background_config_path"] = args.background_config

    setup_logging(str(playback.SCRIPT_DEFAULTS["log_level"]))
    config_path = Path(args.config).resolve()
    manifest_path = playback.replay_final_level(
        load_yaml(config_path),
        config_path.parent,
        expected_event_type=event_type,
    )
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    logger.info(
        "%s final-level playback manifest: %s cases=%s level=%s threshold=%.6g",
        label,
        manifest_path,
        manifest.get("num_cases"),
        manifest.get("level"),
        float(manifest.get("failure_threshold", float("nan"))),
    )
