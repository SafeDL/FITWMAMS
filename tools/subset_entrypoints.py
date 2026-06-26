"""Shared CLI entrypoints for ADS subset-evaluation scripts."""

from __future__ import annotations

import argparse
import importlib
import json
import logging
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)


def _override_if_set(
    config: dict[str, Any],
    section: str,
    key: str,
    value: Any,
) -> None:
    if value is not None:
        config.setdefault(section, {})[key] = value


def _load_yaml_and_setup_logging(config_path: Path) -> dict[str, Any]:
    from diffusion.src.utils import load_yaml, setup_logging

    setup_logging("INFO")
    return load_yaml(config_path)


def run_subset_entrypoint(
    *,
    subset_name: str,
    default_config_path: Path,
    event_type: str,
    label: str,
    result_filename: str | None = None,
    compact_result: tuple[str, str] | None = None,
    policy_section: str | None = None,
    policy_label: str | None = None,
    include_deterministic: bool = False,
    include_subset_overrides: bool = True,
    seed_policy: bool = False,
) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(default_config_path))
    if policy_section is not None:
        parser.add_argument(
            "--checkpoint_path",
            help=f"{policy_label or policy_section} policy weights path.",
        )
    if include_deterministic:
        parser.add_argument(
            "--deterministic",
            action=argparse.BooleanOptionalAction,
            default=None,
            help="Use deterministic policy inference.",
        )
    if include_subset_overrides:
        parser.add_argument("--seed", type=int, help="Shared simulation/policy seed.")
        parser.add_argument("--num_samples", type=int, help="Subset simulation N.")
        parser.add_argument(
            "--p0",
            type=float,
            help="Subset conditional level probability.",
        )
        parser.add_argument("--max_levels", type=int, help="Maximum subset levels.")
        parser.add_argument(
            "--proposal_std",
            type=float,
            help="Latent random-walk proposal std.",
        )
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

    config_path = Path(args.config).resolve()
    config = _load_yaml_and_setup_logging(config_path)
    if policy_section is not None and args.checkpoint_path:
        config.setdefault(policy_section, {})["checkpoint_path"] = args.checkpoint_path
    if (
        include_deterministic
        and policy_section is not None
        and args.deterministic is not None
    ):
        config.setdefault(policy_section, {})["deterministic"] = bool(args.deterministic)
    seed = getattr(args, "seed", None)
    if seed is not None:
        config.setdefault("training", {})["seed"] = int(seed)
        config.setdefault("context_sampling", {})["seed"] = int(seed)
        if seed_policy and policy_section is not None:
            config.setdefault(policy_section, {})["seed"] = int(seed)

    if include_subset_overrides:
        _override_if_set(config, "subset_simulation", "num_samples", args.num_samples)
        _override_if_set(config, "subset_simulation", "p0", args.p0)
        _override_if_set(config, "subset_simulation", "max_levels", args.max_levels)
        _override_if_set(
            config,
            "subset_simulation",
            "proposal_std",
            args.proposal_std,
        )
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

    runner = importlib.import_module(f"{subset_name}.src.latent_subset_runner")
    summary_path = runner.run_subset_from_config(
        config,
        config_path.parent,
        expected_event_type=event_type,
    )
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    if result_filename and compact_result:
        module_name, function_name = compact_result
        compact_fn = getattr(importlib.import_module(module_name), function_name)
        result_payload = compact_fn(
            summary,
            summary_path=summary_path,
            config=config,
            config_dir=config_path.parent,
        )
        result_path = summary_path.with_name(result_filename)
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result_payload, f, indent=2, sort_keys=True)
            f.write("\n")
        logger.info("%s %s result output: %s", label, policy_label, result_path)

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
    subset_name: str,
    default_config_path: Path,
    event_type: str,
    max_samples: int,
    allow_output_dir: bool = False,
) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(default_config_path))
    parser.add_argument(
        "--num_samples",
        type=int,
        help="Override monte_carlo.num_samples for this run.",
    )
    parser.add_argument("--seed", type=int, help="Override monte_carlo.seed.")
    if allow_output_dir:
        parser.add_argument("--output_dir", help="Override monte_carlo.output_dir.")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = _load_yaml_and_setup_logging(config_path)
    if args.num_samples is not None:
        if int(args.num_samples) > int(max_samples):
            parser.error(f"--num_samples must be <= {max_samples}")
        config.setdefault("monte_carlo", {})["num_samples"] = int(args.num_samples)
    if args.seed is not None:
        config.setdefault("monte_carlo", {})["seed"] = int(args.seed)
    if allow_output_dir and args.output_dir:
        config.setdefault("monte_carlo", {})["output_dir"] = args.output_dir

    runner = importlib.import_module(f"{subset_name}.src.latent_subset_runner")
    runner.run_monte_carlo_from_config(
        config,
        config_path.parent,
        expected_event_type=event_type,
    )


def play_final_level_entrypoint(
    *,
    subset_name: str,
    default_config_path: Path,
    event_type: str,
    label: str,
) -> None:
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

    from diffusion.src.utils import load_yaml, setup_logging

    playback = importlib.import_module(f"{subset_name}.src.final_level_playback")
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
