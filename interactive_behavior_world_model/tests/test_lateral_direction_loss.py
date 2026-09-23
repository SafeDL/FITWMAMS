import torch

from interactive_behavior_world_model.scripts.train_action_b2rlp import _lateral_direction_losses


def test_merge_braking_can_satisfy_event_margin_but_not_yaw_margin():
    response = torch.tensor([[[-0.2, 0.0], [-0.2, 0.0], [-0.2, 0.0]]])
    event, yaw = _lateral_direction_losses(
        response,
        torch.tensor([0]),
        torch.tensor([1.0]),
    )
    assert torch.isclose(event, torch.tensor(0.0))
    assert torch.isclose(yaw, torch.tensor(1.0))


def test_away_yaw_satisfies_both_merge_margins():
    response = torch.tensor([[[0.0, -0.015], [0.0, -0.015], [0.0, -0.015]]])
    event, yaw = _lateral_direction_losses(
        response,
        torch.tensor([0]),
        torch.tensor([-1.0]),
    )
    assert torch.isclose(event, torch.tensor(0.0))
    assert torch.isclose(yaw, torch.tensor(0.0))


def test_cutout_does_not_contribute_to_merge_yaw_margin():
    response = torch.tensor([[[0.2, 0.0], [0.2, 0.0], [0.2, 0.0]]])
    event, yaw = _lateral_direction_losses(
        response,
        torch.tensor([1]),
        torch.tensor([1.0]),
    )
    assert torch.isclose(event, torch.tensor(0.0))
    assert torch.isclose(yaw, torch.tensor(0.0))
