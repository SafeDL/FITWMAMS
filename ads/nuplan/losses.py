"""Published planner objectives with masks for the actually observed future.

The native diffusion parameterization and planning/prediction loss terms
are retained. Unobserved labels are removed before corruption and excluded
from losses; their input noise is retained as in native diffusion sampling.
"""

import torch


def mean_valid(error, valid):
    return error[valid].mean() if valid.any() else error.sum() * 0


def targets(samples, device, agents):
    target = torch.zeros((len(samples), agents, 80, 3), device=device)
    valid = torch.zeros((len(samples), agents, 80), dtype=torch.bool, device=device)
    for index, sample in enumerate(samples):
        n = min(agents, len(sample["target"]))
        valid[index, :n] = torch.as_tensor(sample["future_valid"][:n], device=device)
        values = torch.as_tensor(sample["target"][:n], device=device)
        target[index, :n] = torch.where(valid[index, :n, :, None], values, 0)
    return target, valid


def diffusion_batch(policy, samples, *, time=None, noise=None):
    inputs = policy.inputs([s["features"] for s in samples])
    agents = 1 + policy.args.predicted_neighbor_num
    target, valid = targets(samples, policy.device, agents)
    future = torch.cat(
        (target[..., :2], target[..., 2:3].cos(), target[..., 2:3].sin()), -1
    )
    future = torch.where(valid[..., None], policy.args.state_normalizer(future), 0)
    current = torch.cat(
        (
            inputs["ego_current_state"][:, None, :4],
            inputs["neighbor_agents_past"][:, : agents - 1, -1, :4],
        ),
        1,
    )
    if time is None:
        time = torch.rand(len(samples), device=policy.device) * 0.999 + 0.001
    if noise is None:
        noise = torch.randn_like(future)
    mean, std = policy.network.sde.marginal_prob(future, time)
    std = std.reshape(len(samples), 1, 1, 1)
    # Missing labels do not remove the native corruption noise. A zero noisy tail
    # creates a training-only cue absent from the full-horizon inference sampler.
    noised = mean + std * noise
    return (
        inputs,
        torch.cat((current[:, :, None], noised), 2),
        time,
        future,
        valid,
        std,
        noise,
    )


def diffusion_loss(policy, samples, *, denominators=None, components=None):
    inputs, trajectories, time, future, valid, std, noise = diffusion_batch(
        policy, samples
    )
    prediction = policy.denoise(inputs, trajectories, time)[:, :, 1:]
    error = (
        (prediction - future).square()
        if policy.args.diffusion_model_type == "x_start"
        else (prediction * std + noise).square()
    )
    error = error.sum(-1)
    if denominators is None:
        ego = mean_valid(error[:, 0], valid[:, 0])
        neighbors = mean_valid(error[:, 1:], valid[:, 1:])
    else:
        ego = (error[:, 0] * valid[:, 0]).sum() / max(denominators[0], 1)
        neighbors = (error[:, 1:] * valid[:, 1:]).sum() / max(denominators[1], 1)
    if components is not None:
        for name, value in (("ego", ego), ("neighbors", neighbors)):
            components[name] = components.get(name, 0.0) + float(value.detach())
    return ego + neighbors


OBJECTIVES = {"diffusion_planner": diffusion_loss}
