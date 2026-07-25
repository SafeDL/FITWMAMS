import numpy as np
from test_ramp_snapshot_restore import _environment


def test_explicit_uniform_is_a_replayable_world_variable():
    left, right = _environment(), _environment()
    one = left.step(np.zeros(6, np.float32)); two = right.step(np.zeros(6, np.float32))
    assert one["candidate_index"] == two["candidate_index"]
    assert one["trace"]["plan_uniform_random_numbers"] == two["trace"]["plan_uniform_random_numbers"]
