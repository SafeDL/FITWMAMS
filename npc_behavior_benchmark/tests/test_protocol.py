import numpy as np

from npc_behavior_benchmark.data.manifest import choose_causal_window
from npc_behavior_benchmark.evaluation.control import apply_control_limits
from npc_behavior_benchmark.policies.interface import RandomKey


def test_causal_window_prefers_legacy_start_then_shifts_for_tail():
    window = choose_causal_window(100, np.asarray([1, 50]), np.asarray([400, 400]))
    assert window is not None and window.history_start_frame == 100
    shifted = choose_causal_window(100, np.asarray([1, 50]), np.asarray([260, 260]))
    assert shifted is not None and shifted.history_start_frame == 87
    assert shifted.decision_frame == 111 and shifted.future_end_frame == 260


def test_short_identity_set_is_ineligible():
    assert choose_causal_window(100, np.asarray([100]), np.asarray([249])) is None


def test_random_key_is_stable_and_stream_separated():
    key = RandomKey("b", "s", 1, 2)
    assert key.seed() == RandomKey("b", "s", 1, 2).seed()
    assert key.seed() != RandomKey("b", "s", 1, 3).seed()


def test_common_limits_report_requested_rewrites():
    request = np.asarray([[8.0, 1.0], [1.0, 0.1], [9.0, 2.0]], np.float32)
    state = np.zeros((3, 6), np.float32)
    state[:, 2] = 20.0
    applied, diagnostic = apply_control_limits(
        request, state, np.asarray([True, True, False])
    )
    assert np.allclose(applied[0], (4.0, 0.2))
    assert np.allclose(applied[1], (1.0, 0.1))
    assert np.allclose(applied[2], 0.0)
    assert diagnostic["rewritten"].tolist() == [True, False, False]
