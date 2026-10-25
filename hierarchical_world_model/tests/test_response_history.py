import numpy as np
import torch

from hierarchical_world_model.src.stochastic_drivers import raw_data
from hierarchical_world_model.src.stochastic_drivers.driver_policy import (
    response_history_features,
)


def test_own_acceleration_cannot_reenter_the_driver_correction():
    current = torch.zeros(2, 2, 6)
    current[:, 1, 2] = 20
    history = current[:, None].repeat(1, 6, 1, 1)
    history[:, 0, 1, 4] = torch.tensor([-5.0, 4.0])
    valid = torch.zeros(2, 6, 2, dtype=torch.bool)
    valid[:, :, 1] = True
    features = response_history_features(
        current,
        history,
        valid,
        torch.full((2, 1), -1),
        torch.tensor([4.8, 1.8]).expand(2, 2, 2),
    )
    torch.testing.assert_close(features, torch.zeros(2, 1, 2), atol=1e-6, rtol=0)


def test_raw_history_never_joins_gaps_or_changed_leaders(monkeypatch):
    d = {
        "recording": np.ones(4, int),
        "vehicle": np.ones(4, int),
        "frame": np.array([0, 5, 15, 20]),
        "free": np.zeros(4, bool),
        "leader_vehicle": np.array([2, 2, 2, 3]),
        "acceleration": np.array([0.1, 0.2, 0.3, 0.4]),
        "leader_speed": np.array([20.0, 19.8, 19.5, 25.0]),
        "speed": np.full(4, 21.0),
        "gap": np.full(4, 30.0),
    }
    monkeypatch.setattr(
        raw_data,
        "load_group",
        lambda *args: {"train": {key: value.copy() for key, value in d.items()}},
    )
    result = raw_data.load_following()["train"]
    np.testing.assert_array_equal(
        result["leader_history_available"], [False, True, False, False]
    )
    np.testing.assert_allclose(result["leader_acceleration"], [0, -1, 0, 0])
    assert "past_acceleration" not in result
