import numpy as np

from npc_behavior_benchmark.data.lateral_response_dataset import _advance_stimulus
from npc_behavior_benchmark.evaluation.stimuli import lateral_stimulus_trajectories


def test_single_lateral_stimulus_matches_shared_dynamics():
    state = np.asarray([10.0, -3.5, 20.0, 0.0, 0.0, 0.0], np.float32)
    states = np.zeros((1, 7, 6), np.float32)
    states[0, 2] = state
    valid = np.zeros((1, 7), bool)
    valid[0, 2] = True
    natural, changed = lateral_stimulus_trajectories(
        states,
        valid,
        np.asarray([2]),
        np.asarray([0.0]),
        np.asarray([0.9]),
        horizon_frames=20,
    )
    assert np.allclose(
        _advance_stimulus(state, 0.0, 0.9, 20, intervene=False),
        natural[0, :, 2],
        atol=1.0e-5,
    )
    assert np.allclose(
        _advance_stimulus(state, 0.0, 0.9, 20, intervene=True),
        changed[0, :, 2],
        atol=1.0e-5,
    )
