import torch

from npc_behavior_benchmark.evaluation.pnc import fixed_pnc_action, structured_bc_pnc_action


def test_cautious_idm_brakes_for_close_stopped_lead():
    states = torch.zeros((1, 7, 6))
    states[0, 0, 2] = 20.0
    states[0, 1, 0] = 8.0
    active = torch.zeros((1, 7), dtype=torch.bool)
    active[:, :2] = True
    lengths = torch.full((1, 7), 4.5)
    action = fixed_pnc_action(
        states,
        active,
        lengths,
        torch.tensor([20.0]),
        torch.tensor([0.0]),
        "idm_cautious",
    )
    assert action[0, 0] < -1.0


def test_cruise_steers_toward_initial_lane_center():
    states = torch.zeros((1, 7, 6))
    states[0, 0, 1] = -1.0
    states[0, 0, 2] = 10.0
    active = torch.zeros((1, 7), dtype=torch.bool)
    active[:, 0] = True
    action = fixed_pnc_action(
        states,
        active,
        torch.full((1, 7), 4.5),
        torch.tensor([10.0]),
        torch.tensor([0.0]),
        "cruise",
    )
    assert action[0, 1] > 0.0


def test_trajectory_mpc_brakes_for_close_stopped_lead():
    states = torch.zeros((1, 7, 6))
    states[0, 0, 2] = 20.0
    states[0, 1, 0] = 9.0
    active = torch.zeros((1, 7), dtype=torch.bool)
    active[:, :2] = True
    action = fixed_pnc_action(
        states,
        active,
        torch.full((1, 7), 4.5),
        torch.tensor([20.0]),
        torch.tensor([0.0]),
        "trajectory_mpc",
        widths=torch.full((1, 7), 1.9),
    )
    assert action[0, 0] < -1.0


def test_trajectory_mpc_is_future_blind_and_steers_to_lane_center():
    states = torch.zeros((2, 7, 6))
    states[:, 0, 1] = -1.0
    states[:, 0, 2] = 10.0
    active = torch.zeros((2, 7), dtype=torch.bool)
    active[:, 0] = True
    action = fixed_pnc_action(
        states,
        active,
        torch.full((2, 7), 4.5),
        torch.full((2,), 10.0),
        torch.zeros(2),
        "trajectory_mpc",
        widths=torch.full((2, 7), 1.9),
    )
    assert torch.equal(action[0], action[1])
    assert action[0, 1] > 0.0


def test_structured_bc_pnc_applies_common_bounds():
    class FixedModel(torch.nn.Module):
        def forward(self, features, history_valid):
            del history_valid
            return torch.tensor([[20.0, 20.0]], dtype=features.dtype).expand(
                len(features), -1
            )

    features = torch.zeros((2, 25, 7, 12))
    valid = torch.ones((2, 25, 7), dtype=torch.bool)
    current = torch.zeros((2, 7, 6))
    current[:, 0, 2] = 20.0
    active = valid[:, -1]
    checkpoint = {
        "feature_mean": torch.zeros(12),
        "feature_std": torch.ones(12),
        "action_mean": torch.zeros(2),
        "action_std": torch.ones(2),
    }
    action = structured_bc_pnc_action(
        FixedModel(), checkpoint, features, valid, current, active
    )
    assert torch.all(action[:, 0] == 4.0)
    assert torch.allclose(action[:, 1], torch.full((2,), 0.2))
