from __future__ import annotations

import argparse

from bayesian_ma_idm.scripts.run_highway_env_ring import apply_protocol


def test_source_code_protocol_resolves_coupled_author_config() -> None:
    values = argparse.Namespace(protocol="source_code", vehicles=37, radius=128., fixed_profile="paper_recommended",
                                update_mode="synchronous", idm_semantics="equation")
    resolved = apply_protocol(values)
    assert (resolved.vehicles, resolved.radius, resolved.fixed_profile) == (32, 137., "donor_ring")
    assert (resolved.update_mode, resolved.idm_semantics) == ("source_sequential", "donor_ring")
