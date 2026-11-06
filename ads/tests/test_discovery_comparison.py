"""Common motion catalog uses stored collisions, with no additional executions."""

from ads.search.comparison import motion_catalog


def row(identity, x, *, collision=True, time=0):
    return {
        "instance_hash": identity,
        "collision": collision,
        "critical_slot": 1,
        "descriptor": [x, 0, 0, 0, time],
    }


def test_catalog_is_method_blind_time_invariant_and_retains_nearby_assignments():
    a, b = row("a", 0), row("b", 3)
    journal = {
        "first": [a, row("c", 0.1, time=1)],
        "second": [b, a, row("d", 9, collision=False)],
    }
    result = motion_catalog(journal, 0.2)
    assert result["representatives"] == ["a", "b"]
    assert result["methods"]["first"]["covered_states"] == 1
    assert result["methods"]["second"]["covered_states"] == 2
    swapped = motion_catalog(
        {"second": journal["second"], "first": journal["first"]}, 0.2
    )
    assert result == swapped


def test_exact_catalog_handles_empty_and_duplicate_motion_states():
    assert motion_catalog({"a": []}, 0)["representatives"] == []
    result = motion_catalog({"a": [row("a", 0), row("b", 0, time=10)]}, 0)
    assert result["methods"]["a"]["covered_states"] == 1
