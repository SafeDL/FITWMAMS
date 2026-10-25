import numpy as np
from matplotlib.patches import Patch, StepPatch

from hierarchical_world_model.src.npc_experiments.figures import (
    STAGES,
    canvas,
    clustered_ratio,
    density_histogram,
)
from tools.plot_style import get_pyplot


def test_histogram_preserves_physical_bins_and_tail_mass_without_zoom_renormalizing():
    plt = get_pyplot()
    figure, axis = plt.subplots()
    edges = np.array([-np.inf, 0, 1, 3, np.inf])
    counts = {name: np.array([1, 2, 4, 3]) for name in ("real", *STAGES)}
    original = {name: value.copy() for name, value in counts.items()}
    density_histogram(axis, counts, edges, view=(1, 2))
    bars = axis.containers[0].patches
    np.testing.assert_allclose([bar.get_x() for bar in bars], [0, 1])
    np.testing.assert_allclose([bar.get_width() for bar in bars], [1, 2])
    np.testing.assert_allclose([bar.get_height() for bar in bars], [0.2, 0.2])
    assert np.isclose(sum(bar.get_height() * bar.get_width() for bar in bars), 0.6)
    outlines = [patch for patch in axis.patches if isinstance(patch, StepPatch)]
    assert len(outlines) == len(STAGES)
    for step in outlines:
        np.testing.assert_allclose(step.get_data().values, [0.2, 0.2])
        np.testing.assert_allclose(step.get_data().edges, [0, 1, 3])
        assert not step.get_fill()
    for name, values in counts.items():
        np.testing.assert_array_equal(values, original[name])
    assert axis.get_xlim() == (1, 2)
    plt.close(figure)


def test_recording_ratio_pools_exposure_instead_of_averaging_run_ratios():
    values = np.array([[1, 2], [9, 10], [2, 8]], float)
    value, interval = clustered_ratio(values, [1, 1, 2])
    assert value == 12 / 20
    assert interval[0] == 2 / 8
    assert interval[1] == 10 / 12
    np.testing.assert_array_equal(values, [[1, 2], [9, 10], [2, 8]])


def test_histogram_legend_identifies_observed_bars_not_a_smooth_density_fit():
    plt, figure, _ = canvas("Distributions", "Fixed bins", histogram=True)
    handles = figure.legends[0].legend_handles
    assert isinstance(handles[0], Patch)
    assert next(text.get_text() for text in figure.legends[0].texts) == "Observed highD"
    assert len(handles) == 1 + len(STAGES)
    plt.close(figure)


def test_recording_ratio_with_no_requests_is_unestimated_not_zero():
    value, interval = clustered_ratio(np.zeros((2, 2)), [1, 2])
    assert np.isnan(value)
    assert np.isnan(interval).all()


def test_supplementary_canvas_has_no_observed_zero_score_in_legend():
    plt, figure, axes = canvas(
        "Stability",
        "Matched windows",
        STAGES,
        shape=(2, 2),
        figsize=(13.8, 9.4),
        include_real=False,
    )
    assert axes.shape == (2, 2)
    assert [text.get_text() for text in figure.legends[0].texts] == [
        "NPC response model"
    ]
    assert "Observed highD" not in [text.get_text() for text in figure.legends[0].texts]
    plt.close(figure)
