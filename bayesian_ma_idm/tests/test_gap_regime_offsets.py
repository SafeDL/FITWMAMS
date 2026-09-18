from bayesian_ma_idm.src.full_evaluation import regime_interval_offset


def test_gap_regime_offsets_use_training_bin_edges() -> None:
    spec = {"gap_bin_edges_m": [5., 10.], "offsets_m": [.1, .2, .3]}
    assert regime_interval_offset(spec, 4.9) == .1
    assert regime_interval_offset(spec, 5.) == .2
    assert regime_interval_offset(spec, 99.) == .3
    assert regime_interval_offset(None, 5.) == 0.
    try:
        regime_interval_offset({"gap_bin_edges_m": [5.], "offsets_m": [.1]}, 5.)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid bin specification must fail")
