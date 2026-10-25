"""Published planner objectives with masks for the actually observed future.

The native networks, noise/path parameterizations, candidate generators and loss
terms are retained. Unobserved labels are removed before corruption and excluded
from losses; their input noise is retained as in native diffusion/flow sampling.
"""

import numpy as np
import torch
from torch.nn import functional as F

from ads.nuplan.policy import collate


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


def diffusion_batch(policy, samples):
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
    time = torch.rand(len(samples), device=policy.device) * 0.999 + 0.001
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


def diffusion_loss(policy, samples):
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
    return mean_valid(error[:, 0], valid[:, 0]) + mean_valid(error[:, 1:], valid[:, 1:])


def flow_loss(policy, samples):
    from ads.flow_planner.vendor.flow_planner.model.model_utils.traj_tool import (
        traj_chunking,
    )

    network = policy.network
    target, valid = targets(samples, policy.device, 1)
    data = policy.inputs([s["features"] for s in samples], future=target[:, 0])
    flags = (torch.rand(len(samples), 1, device=policy.device) > network.cfg_prob).int()
    inputs, gt = network.prepare_model_input(
        flags, data, use_cfg=False, is_training=True
    )
    gt_future = torch.where(valid[..., None], gt[:, :, 1:], 0)
    encoder = network.encoder(**network.extract_encoder_inputs(inputs))
    extra = network.extract_decoder_inputs(encoder, inputs)
    noised, expected, time = network.flow_ode.sample(gt_future, network.model_type)
    length, overlap = (
        network.planner_params[key] for key in ("action_len", "action_overlap")
    )

    def chunk(value):
        return torch.cat(traj_chunking(value, length, overlap), dim=1)

    mask = chunk(valid[..., None]).squeeze(-1)
    prediction = network.decoder(chunk(noised), time, **extra)
    loss = float(policy.native_config.core.ego_planning_loss) * mean_valid(
        (prediction - chunk(expected)).square().sum(-1), mask
    )
    if overlap:
        consistency = []
        # Same adjacent pairs as the published objective.
        for index in range(prediction.shape[1] - 2):
            shared = mask[:, index, -overlap:] & mask[:, index + 1, :overlap]
            error = (
                (prediction[:, index, -overlap:] - prediction[:, index + 1, :overlap])
                .square()
                .sum(-1)
            )
            consistency.append(mean_valid(error, shared))
        if consistency:
            loss = (
                loss
                + float(policy.native_config.core.consistency_loss)
                * torch.stack(consistency).mean()
            )
    return loss


def dtpp_candidates(policy, sample):
    """Run only the causal native candidate/tree planner; no world execution."""
    with torch.no_grad():
        policy.tree, policy.map_api = None, None
        policy._plan(sample["observation"])
    result = []
    target = sample["target"][0]
    observed = sample["future_valid"][0]
    for proposals in (
        policy.tree.first_stage_candidates,
        policy.tree.second_stage_candidates,
    ):
        bank = np.zeros((30, 80, 6), np.float32)
        for index, trajectory in enumerate(proposals[:30]):
            values = trajectory.detach().cpu().numpy()[:, :6]
            bank[index, : len(values)] = values[:80]
        available = np.any(bank != 0, -1)
        score = np.linalg.norm(bank[..., :2] - target[None, :, :2], axis=-1)
        mask = available & observed[None]
        distance = (score * mask).sum(-1) / mask.sum(-1).clip(1)
        distance[~mask.any(-1)] = np.inf
        order = np.argsort(distance, kind="stable")
        result.append(bank[order])
    return result


def dtpp_loss(policy, samples):
    inputs = collate([s["features"] for s in samples], policy.device)
    target, valid = targets(samples, policy.device, 11)
    encoder = policy.network["encoder"](inputs)
    loss = target.sum() * 0
    for stage, horizon, coefficient in ((0, 30, 1.0), (1, 80, 0.2)):
        bank = torch.as_tensor(
            np.stack([s["dtpp_candidates"][stage] for s in samples]),
            device=policy.device,
        )
        bank = bank * valid[:, :1, :, None]
        # The native branch embedding and learned costs both index the stage's
        # terminal time. Bucket by actual label horizon so padding is never a
        # terminal stopping candidate or an extra interaction-cost interval.
        lengths = valid[:, 0].sum(-1).clamp(max=horizon)
        for count in lengths.unique():
            steps = int(count)
            rows = torch.where(lengths == count)[0]
            encoded = {key: value[rows] for key, value in encoder.items()}
            neighbors, scores, ego, weights = policy.network["decoder"](
                encoded, bank[rows], inputs["neighbor_agents_past"][rows], steps
            )
            usable = (
                valid[rows, :, :steps]
                & (bank[rows, 0, :steps].abs().sum(-1) > 0)[:, None]
            )
            forecast = F.smooth_l1_loss(
                neighbors[:, 0, :, :steps], target[rows, 1:, :steps], reduction="none"
            )
            prediction = (forecast * usable[:, 1:, :, None]).sum() / usable[
                :, 0
            ].sum().clamp_min(1)
            regularization = mean_valid(
                F.smooth_l1_loss(
                    ego[:, :steps], target[rows, 0, :steps], reduction="none"
                ).sum(-1),
                usable[:, 0],
            )
            classification = F.cross_entropy(
                scores, torch.zeros(len(rows), device=policy.device, dtype=torch.long)
            )
            loss = loss + coefficient * len(rows) / len(samples) * (
                prediction
                + classification
                + 0.1 * regularization
                + 0.01 * weights.square().mean()
            )
    return loss


def str2_loss(policy, samples):
    from ads.str2.vendor.transformer4planning.models.decoder.base import (
        mean_circular_error,
    )

    network = policy.network
    target, valid = targets(samples, policy.device, 1)
    mask = valid[:, 0]
    label = F.pad(target[:, 0, :, :2], (0, 2))
    label[..., 3] = target[:, 0, :, 2]
    inputs = policy.inputs([s["features"] for s in samples], target=label)
    embedded, info = network.encoder(
        is_training=network.training, future_valid=mask, **inputs
    )
    attention = torch.ones(embedded.shape[:2], dtype=torch.bool, device=policy.device)
    start = info["context_length"] + network.config.use_proposal
    attention[:, start : start + 80] = mask
    indices = info["selected_indices"]
    kp_valid = mask[:, indices]
    attention[:, start + 80 : start + 80 + len(indices)] = kp_valid
    embedded = embedded * attention[..., None]
    hidden = network.embedding_to_hidden(
        embedded, attention_mask=attention, return_dict=True
    )
    state = hidden["last_hidden_state"]
    _, trajectory = network.traj_decoder.compute_traj_loss(state, label, info)
    planning = mean_valid(
        (trajectory[..., :2] - label[..., :2]).square().mean(-1), mask
    )
    planning += mean_valid(
        mean_circular_error(trajectory[..., 3], label[..., 3], mean=False), mask
    )
    distances = info["future_traj_diff"]
    eligible = distances <= 30
    empty = ~eligible.any(-1)
    eligible[empty, distances[empty].argmin(-1)] = True
    probability = torch.softmax(-distances.masked_fill(~eligible, float("inf")), -1)
    proposal_state = state[:, info["context_length"] + info["candi_proposal_num"] - 1]
    classification = F.cross_entropy(
        network.proposal_decoder.proposal_cls_decoder(proposal_state), probability
    )
    _, keypoints, _ = network.key_points_decoder.compute_keypoint_loss(state, info)
    kp_loss = mean_valid(
        (keypoints[..., :2] - info["future_key_points"][..., :2]).square().mean(-1),
        kp_valid,
    )
    auxiliary = network.load_balancing_loss_func(
        hidden["router_logits"],
        network.config.num_local_experts,
        network.config.num_experts_per_token,
    )
    return (
        planning * network.config.trajectory_loss_rescale
        + classification
        + kp_loss * network.config.kp_loss_rescale
        + network.config.router_aux_loss_coef * auxiliary
    )


OBJECTIVES = {
    "diffusion_planner": diffusion_loss,
    "dtpp": dtpp_loss,
    "flow_planner": flow_loss,
    "str2": str2_loss,
}
