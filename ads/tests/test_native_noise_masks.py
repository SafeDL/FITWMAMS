"""Native corruption paths retain noise where future labels are unavailable."""

from copy import deepcopy

import numpy as np
import pytest
import torch

from ads.common.setup import load_policy
from ads.nuplan.assets import official_path
from ads.nuplan.losses import OBJECTIVES, targets
from ads.nuplan.teacher import demonstration


@pytest.fixture(params=("diffusion_planner", "flow_planner"))
def native_policy(request):
    torch.set_num_threads(1)
    policy = load_policy(request.param, official_path(request.param), device="cpu")
    policy.network.train()
    return policy


def sample_for(instance, algorithm):
    states = np.tile(instance.event.initial_states, (150, 1, 1))
    states[:, :, :2] += np.arange(150)[:, None, None] * 0.04 * states[:, :, 2:4]
    return demonstration(instance, states, algorithm)[0]


def decoder_for(policy):
    if policy.algorithm == "diffusion_planner":
        return policy.network.decoder.decoder.dit
    return policy.network.model_decoder


def decoder_valid(policy, sample):
    from ads.flow_planner.vendor.flow_planner.model.model_utils.traj_tool import (
        traj_chunking,
    )

    agents = 11 if policy.algorithm == "diffusion_planner" else 1
    _, valid = targets([sample], policy.device, agents)
    if policy.algorithm == "diffusion_planner":
        current = torch.ones_like(valid[:, :, :1])
        return torch.cat((current, valid), dim=2)[..., None].expand(-1, -1, -1, 4)
    return torch.cat(traj_chunking(valid[..., None], 20, 10), dim=1).expand(
        -1, -1, -1, 4
    )


def test_unobserved_future_keeps_noise_but_has_no_label_gradient(
    native_policy, instance
):
    policy = native_policy
    sample = sample_for(instance, policy.algorithm)
    captured = {}

    def capture_input(module, args):
        captured["input"] = args[0].detach().clone()

    def capture_output(module, args, output):
        output.retain_grad()
        captured["output"] = output

    decoder = decoder_for(policy)
    before = decoder.register_forward_pre_hook(capture_input)
    after = decoder.register_forward_hook(capture_output)
    torch.manual_seed(20260926)
    loss = OBJECTIVES[policy.algorithm](policy, [sample])
    loss.backward()
    before.remove()
    after.remove()
    valid = decoder_valid(policy, sample)
    noisy = captured["input"].reshape(valid.shape)
    gradient = captured["output"].grad.reshape(valid.shape)
    # Present ego's unavailable tail is stochastic input, never a zero trajectory.
    ego_noisy = noisy[:, :1] if policy.algorithm == "diffusion_planner" else noisy
    ego_valid = valid[:, :1] if policy.algorithm == "diffusion_planner" else valid
    assert torch.count_nonzero(ego_noisy[~ego_valid]) > 0
    assert torch.count_nonzero(gradient[~valid]) == 0
    assert torch.count_nonzero(gradient[valid]) > 0


def test_unknown_label_values_cannot_change_corruption_or_loss(native_policy, instance):
    policy = native_policy
    sample = sample_for(instance, policy.algorithm)
    changed = deepcopy(sample)
    changed["target"][~changed["future_valid"]] = 100000
    inputs = []
    handle = decoder_for(policy).register_forward_pre_hook(
        lambda module, args: inputs.append(args[0].detach().clone())
    )
    with torch.no_grad():
        torch.manual_seed(20260926)
        original = OBJECTIVES[policy.algorithm](policy, [sample])
        torch.manual_seed(20260926)
        modified = OBJECTIVES[policy.algorithm](policy, [changed])
    handle.remove()
    torch.testing.assert_close(original, modified, rtol=0, atol=0)
    torch.testing.assert_close(inputs[0], inputs[1], rtol=0, atol=0)


def test_diffusion_partial_neighbor_future_matches_official_objective(instance):
    from ads.diffusion_planner.vendor.diffusion_planner.loss import diffusion_loss_func

    torch.set_num_threads(1)
    policy = load_policy(
        "diffusion_planner", official_path("diffusion_planner"), device="cpu"
    )
    sample = sample_for(instance, policy.algorithm)
    sample["future_valid"][0] = True
    sample["future_valid"][1, 40:] = False
    sample["target"][0, :, 0] = np.arange(1, 81) * 2.0
    policy.network.train()
    inputs = []
    handle = decoder_for(policy).register_forward_pre_hook(
        lambda module, args: inputs.append(args[0].detach().clone())
    )
    with torch.no_grad():
        torch.manual_seed(20260926)
        actual = OBJECTIVES[policy.algorithm](policy, [sample])
        target, valid = targets([sample], policy.device, 11)
        future = torch.cat(
            (target[..., :2], target[..., 2:3].cos(), target[..., 2:3].sin()), -1
        )
        future = torch.where(valid[..., None], future, 0)
        torch.manual_seed(20260926)
        losses, _ = diffusion_loss_func(
            policy.network,
            policy.inputs([sample["features"]]),
            policy.network.sde.marginal_prob,
            (future[:, 0], future[:, 1:], ~valid[:, 1:]),
            policy.args.state_normalizer,
            {},
            policy.args.diffusion_model_type,
        )
    handle.remove()
    expected = losses["ego_planning_loss"] + losses["neighbor_prediction_loss"]
    torch.testing.assert_close(inputs[0], inputs[1], rtol=0, atol=0)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
