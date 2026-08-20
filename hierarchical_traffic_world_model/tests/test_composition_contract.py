"""Seed partition tests for the public sampler."""

from __future__ import annotations

from hierarchical_traffic_world_model.src.composition import (
    split_motion_seed,
)


def test_seed_partition_is_deterministic_and_independent():
    assert split_motion_seed(41) == split_motion_seed(41)
    assert split_motion_seed(41) != split_motion_seed(42)
    left, right = split_motion_seed(41)
    assert left != right
