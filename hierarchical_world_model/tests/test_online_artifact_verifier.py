"""Checks for the maintained online-world provenance verifier."""

import numpy as np
import pytest

from hierarchical_world_model.scripts.verify_online_artifacts import _recording_ids


def test_recording_ids_are_read_from_highd_sequence_identifiers() -> None:
    assert _recording_ids(np.asarray(["nat_04_001", "nat_49_002", "nat_04_003"])) == {4, 49}


def test_unexpected_sequence_identifiers_are_rejected() -> None:
    with pytest.raises(ValueError, match="unexpected highD sequence ID"):
        _recording_ids(np.asarray(["unknown_04_001"]))
