#!/usr/bin/env python3
"""Convert causal rollout traces into response-gate supervision."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from hierarchical_world_model.src.response_gate import (
    LONGITUDINAL_RESPONSE_THRESHOLD_MPS2,
    MANEUVER_RESPONSE_THRESHOLD_RPS,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.traces) as values:
        features = np.asarray(values["features"], np.float32)
        base = np.asarray(values["base_actions"], np.float32)
        teacher = np.asarray(values["teacher_actions"], np.float32)
        rule = np.asarray(
            values["rule_actions"]
            if "rule_actions" in values
            else values["teacher_actions"],
            np.float32,
        )
        source = np.asarray(values["source"] if "source" in values else np.full(len(features), "teacher"))
        metadata = {
            name: np.asarray(values[name])
            for name in (
                "event_id", "source_event_id", "recording_id",
                "interaction_group", "frame_index", "actor_index",
            )
            if name in values
        }
        provenance = {
            name: np.asarray(values[name])
            for name in ("protocol_id", "protocol_lock_sha256")
            if name in values
        }
        target_authority = (
            np.asarray(values["target_authority"], np.float32)
            if "target_authority" in values
            else np.full(len(features), np.nan, np.float32)
        )
    if (
        features.shape[:-1] != base.shape[:-1]
        or base.shape != teacher.shape
        or rule.shape != base.shape
        or base.shape[-1] != 2
    ):
        raise ValueError("trace feature/action shapes are inconsistent")
    delta = teacher - base
    mode = np.where(
        np.abs(delta[..., 1]) > MANEUVER_RESPONSE_THRESHOLD_RPS,
        2,
        np.where(
            np.abs(delta[..., 0])
            > LONGITUDINAL_RESPONSE_THRESHOLD_MPS2,
            1,
            0,
        ),
    )
    candidate_delta = rule - base
    denominator = candidate_delta[..., 0]
    authority = np.divide(
        delta[..., 0],
        denominator,
        out=np.ones_like(denominator, dtype=np.float32),
        where=np.abs(denominator) > 0.05,
    )
    # Opposite-signed candidates cannot reproduce the observed target; use
    # full authority and let closed-loop evaluation expose the bottleneck.
    authority = np.where(authority > 0.0, authority, 1.0)
    gate = np.where(
        mode == 0,
        0.0,
        np.clip(authority, 0.05, 1.0),
    ).astype(np.float32)
    source_flat = np.asarray(source).reshape(-1)
    if len(source_flat) not in (1, mode.size):
        raise ValueError("source labels must be scalar or match the action samples")
    if len(source_flat) == 1:
        source_flat = np.repeat(source_flat, mode.size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        features=features.reshape(-1, features.shape[-1]),
        mode=mode.reshape(-1),
        gate=gate.reshape(-1),
        base_actions=base.reshape(-1, 2),
        rule_actions=rule.reshape(-1, 2),
        target_actions=teacher.reshape(-1, 2),
        target_authority=target_authority.reshape(-1),
        source=source_flat,
        **{name: value.reshape(-1) for name, value in metadata.items()},
        **provenance,
    )


if __name__ == "__main__":
    main()
