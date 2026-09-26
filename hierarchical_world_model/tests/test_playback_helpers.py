from __future__ import annotations

import numpy as np

from hierarchical_world_model.scripts.playback_helpers import _draw_world
from tools.plot_style import get_pyplot


def test_current_playback_draws_road_ads_and_npc() -> None:
    plt = get_pyplot()
    figure, axis = plt.subplots()
    states = np.zeros((2, 7, 6), dtype=np.float32)
    states[:, :, 2] = 20.0
    states[:, 1, 0] = 12.0
    valid = np.asarray((True, True, False, False, False, False, False))

    _draw_world(
        axis,
        states=states,
        valid=valid,
        frame=1,
        title="playback",
        focus_slot=0,
    )

    assert len(axis.patches) == 2
    assert axis.get_title() == "playback"
    figure.canvas.draw()
    plt.close(figure)
