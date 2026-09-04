"""Longitudinal human-action prior for dynamic causal NPC relations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


HUMAN_PRIOR_ROLE_DIM = 5
# The prior is explicitly autoregressive at the physical-control level.  The
# last realized child acceleration is part of the observation; without it a
# per-tick Gaussian can match marginal acceleration while independently
# resampling every 40 ms, which produces an unrealistically large jerk tail.
HUMAN_PRIOR_FEATURE_DIM = 25 * 6 + 5 + 1 + HUMAN_PRIOR_ROLE_DIM + 4


def realized_parent_controls(
    history: torch.Tensor, parent_index: torch.Tensor, *, dt_s: float = .04,
) -> torch.Tensor:
    """Recover the last five causal longitudinal controls of each parent."""
    source = history[:, -6:]
    if source.shape[1] < 6:
        source = torch.cat((
            source.new_zeros((len(source), 6 - source.shape[1], 7, 6)), source,
        ), dim=1)
    index = parent_index.clamp_min(0)[:, None, None].expand(-1, source.shape[1], 6)
    parent = torch.gather(source, 2, index[:, :, None]).squeeze(2)
    speed = torch.linalg.vector_norm(parent[..., 2:4], dim=-1)
    controls = source.new_zeros((len(source), 5, 2))
    controls[..., 0] = (speed[:, 1:] - speed[:, :-1]) / float(dt_s)
    return controls


def human_prior_features(
    history: torch.Tensor,
    committed_ego_controls: torch.Tensor,
    *,
    target_slot_index: int = 1,
    parent_index: torch.Tensor | None = None,
    role: torch.Tensor | int | None = None,
) -> torch.Tensor:
    """Causal parent--child features, intentionally excluding HiQR previews."""
    history = history[:, -25:]
    if history.shape[1] < 25:
        history = torch.cat((history.new_zeros((len(history), 25 - history.shape[1], 7, 6)), history), dim=1)
    child = history[:, :, target_slot_index + 1]
    if parent_index is None:
        parent = history[:, :, 0]
    else:
        index = parent_index.clamp_min(0)[:, None, None].expand(-1, history.shape[1], 6)
        parent = torch.gather(history, 2, index[:, :, None]).squeeze(2)
    relative = torch.stack((child[..., 0] - parent[..., 0], child[..., 1] - parent[..., 1], child[..., 2] - parent[..., 2], child[..., 3] - parent[..., 3], child[..., 4], parent[..., 4]), dim=-1)
    temporal = relative.reshape(len(history), -1)
    control = committed_ego_controls[:, -5:, 0]
    if control.shape[1] < 5:
        control = torch.cat((control.new_zeros((len(control), 5 - control.shape[1])), control), dim=1)
    now_parent, now_child = parent[:, -1], child[:, -1]
    gap = now_parent[:, 0] - now_child[:, 0] - 4.8
    closing = now_child[:, 2] - now_parent[:, 2]
    ttc = torch.where((gap > 0) & (closing > 1e-4), gap / closing.clamp_min(1e-4), torch.full_like(gap, 10.)).clamp(0., 10.)
    if role is None:
        role_tensor = torch.zeros(len(history), dtype=torch.long, device=history.device)
    else:
        role_tensor = torch.as_tensor(role, dtype=torch.long, device=history.device)
        if role_tensor.ndim == 0:
            role_tensor = role_tensor.expand(len(history))
    role_one_hot = torch.nn.functional.one_hot(
        role_tensor.clamp(0, HUMAN_PRIOR_ROLE_DIM - 1),
        num_classes=HUMAN_PRIOR_ROLE_DIM,
    ).to(history.dtype)
    scalars = torch.stack((gap / 60., closing / 25., ttc / 10., now_child[:, 2] / 40.), dim=-1)
    # Causal temporal anchor for the action distribution.  This uses only the
    # two latest realized child states (the current state and its predecessor)
    # and is therefore available both in highD expert windows and after each
    # HighwayEnv tick.  It also lets the GAIL generator learn smooth
    # continuation/recovery rather than treating each action as iid noise.
    previous_child_ax = (
        torch.linalg.vector_norm(child[:, -1, 2:4], dim=-1)
        - torch.linalg.vector_norm(child[:, -2, 2:4], dim=-1)
    ) / .04
    previous_child_ax = previous_child_ax.clamp(-8., 4.)[:, None] / 8.
    # Keep the four physical scalars last: held-out tooling reads them without
    # knowing the internal history/role layout.
    return torch.cat((temporal / 20., control / 8., previous_child_ax, role_one_hot, scalars), dim=-1)


class HumanActionPrior(nn.Module):
    """V4 paper-aligned bounded Gaussian longitudinal driving prior."""

    def __init__(self, hidden_dim: int = 128) -> None:
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(HUMAN_PRIOR_FEATURE_DIM, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim), nn.SiLU())
        self.mean = nn.Linear(hidden_dim, 1)
        self.log_std = nn.Linear(hidden_dim, 1)
        # At 25 Hz, the old exp(-2.5) floor mapped to roughly 0.5 m/s² of
        # independent action noise and created artificial 10--20 m/s³ jerk.
        # Start with a narrow but still learnable Gaussian; highD braking
        # tails can widen it through the state-dependent head.
        nn.init.zeros_(self.log_std.weight)
        nn.init.constant_(self.log_std.bias, -3.5)
        self.value = nn.Sequential(nn.Linear(HUMAN_PRIOR_FEATURE_DIM, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))
        self.discriminator = nn.Sequential(
            nn.Linear(HUMAN_PRIOR_FEATURE_DIM + 1, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.discriminator_head = nn.Linear(hidden_dim, 1)
        nodes, weights = np.polynomial.hermite.hermgauss(12)
        self.register_buffer("quadrature_nodes", torch.as_tensor(nodes, dtype=torch.float32))
        self.register_buffer(
            "quadrature_weights",
            torch.as_tensor(weights / np.sqrt(np.pi), dtype=torch.float32),
        )

    def distribution(self, features: torch.Tensor) -> torch.distributions.Normal:
        hidden = self.actor(features)
        mean = 3.0 * torch.tanh(self.mean(hidden).squeeze(-1))
        std = self.log_std(hidden).squeeze(-1).clamp(-4.5, 0.5).exp()
        return torch.distributions.Normal(mean, std)

    def action_from_raw(self, raw: torch.Tensor) -> torch.Tensor:
        return -2.0 + 6.0 * torch.tanh(raw)

    @staticmethod
    def raw_from_action(action: torch.Tensor) -> torch.Tensor:
        return torch.atanh(((action + 2.0) / 6.0).clamp(-0.999, 0.999))

    def action_log_prob(self, features: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        raw = self.raw_from_action(action)
        jacobian = (6.0 * (1.0 - torch.tanh(raw).square())).clamp_min(1.0e-5)
        return self.distribution(features).log_prob(raw) - torch.log(jacobian)

    def forward(self, features: torch.Tensor, *, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution = self.distribution(features)
        raw = distribution.mean if deterministic else distribution.sample()
        return self.action_from_raw(raw), raw, distribution.log_prob(raw), self.value(features).squeeze(-1)

    def discriminator_logits(self, features: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if features.ndim == 2:
            features = features[:, None]
            action = action.reshape(-1, 1)
        if features.ndim != 3 or action.shape != features.shape[:2]:
            raise ValueError("sequence discriminator expects features [B,T,F] and action [B,T]")
        # Average per-step evidence rather than giving a recurrent
        # discriminator enough capacity to identify trivial rollout details.
        return self.discriminator_step_logits(features, action).mean(-1)

    def discriminator_step_logits(self, features: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if features.ndim == 2:
            features = features[:, None]
            action = action.reshape(-1, 1)
        if features.ndim != 3 or action.shape != features.shape[:2]:
            raise ValueError("sequence discriminator expects features [B,T,F] and action [B,T]")
        hidden = self.discriminator(torch.cat((features, action[..., None] / 8.0), dim=-1))
        return self.discriminator_head(hidden).squeeze(-1)

    def forward_kl_to(self, features: torch.Tensor, final_mean: torch.Tensor, final_std: torch.Tensor) -> torch.Tensor:
        """Forward KL in the shared *raw* action space.

        ``HumanActionPrior.distribution`` is Gaussian before its bounded
        ``tanh`` action transform.  A reaction controller, conversely,
        reports its final mean and scale in physical m/s².  Comparing those
        numbers directly made the old KL dimensionally invalid and forced
        almost every naturalness reward to zero.  We map the controller's
        physical Gaussian locally through the inverse/derivative of the same
        transform before evaluating the analytic Gaussian KL.
        """
        human = self.distribution(features)
        raw = human.mean[:, None] + np.sqrt(2.0) * human.stddev[:, None] * self.quadrature_nodes[None]
        action = self.action_from_raw(raw)
        log_h = self.action_log_prob(
            features[:, None].expand(-1, raw.shape[1], -1).reshape(-1, features.shape[-1]),
            action.reshape(-1),
        ).reshape_as(raw)
        final = torch.distributions.Normal(final_mean[:, None], final_std[:, None].clamp_min(.05))
        log_final = final.log_prob(action)
        return ((log_h - log_final) * self.quadrature_weights[None]).sum(-1).clamp_min(0.0)

    def forward_kl_to_policy_interval(
        self,
        features: torch.Tensor,
        *,
        base_action: torch.Tensor,
        nominal_lower: torch.Tensor,
        nominal_upper: torch.Tensor,
        execute_lower: torch.Tensor,
        execute_upper: torch.Tensor,
        minimum_gate: torch.Tensor,
        authority: torch.Tensor,
        policy_gate_mean: torch.Tensor,
        policy_gate_std: torch.Tensor,
        policy_raw_mean: torch.Tensor,
        policy_raw_std: torch.Tensor,
        decreasing: torch.Tensor,
    ) -> torch.Tensor:
        """MC forward KL to the controller's actual bounded final density.

        The previous implementation conditioned on one sampled gate and
        therefore overstated KL whenever the gate was narrow.  V3 integrates
        over both Gaussian actor coordinates and evaluates the marginal
        density of the action actually executable by HighwayEnv.
        """
        human = self.distribution(features)
        nodes = self.quadrature_nodes.to(base_action)
        weights = self.quadrature_weights.to(base_action)
        human_raw = human.mean[:, None] + np.sqrt(2.0) * human.stddev[:, None] * nodes[None]
        human_action = self.action_from_raw(human_raw)
        expanded_features = features[:, None].expand(-1, len(nodes), -1).reshape(-1, features.shape[-1])
        log_h_action = self.action_log_prob(expanded_features, human_action.reshape(-1)).reshape_as(human_action)

        gate_raw = policy_gate_mean[:, None] + np.sqrt(2.0) * policy_gate_std[:, None] * nodes[None]
        gate = authority[:, None] * torch.sigmoid(gate_raw)
        gate = torch.maximum(gate, minimum_gate[:, None]).clamp_min(1.e-6)
        # Rebuild the exact conditional target interval for every integrated
        # gate value.  This is the same inverse constraint used by
        # dynamic_bounded_action_distribution, including safety and jerk.
        inverse_lower = (
            execute_lower[:, None] - (1.0 - gate) * base_action[:, None]
        ) / gate
        inverse_upper = (
            execute_upper[:, None] - (1.0 - gate) * base_action[:, None]
        ) / gate
        lower = torch.maximum(nominal_lower[:, None], inverse_lower)
        upper = torch.minimum(nominal_upper[:, None], inverse_upper)
        feasible = lower <= upper
        midpoint = ((lower + upper) * .5).clamp(-8.0, 4.0)
        lower = torch.where(feasible, lower, midpoint)
        upper = torch.where(feasible, upper, midpoint)
        target_span = (upper - lower).clamp_min(1.e-5)
        final_lower = (1.0 - gate) * base_action[:, None] + gate * lower
        final_upper = (1.0 - gate) * base_action[:, None] + gate * upper
        final_span = (final_upper - final_lower).clamp_min(1.e-8)
        increasing_q = (
            human_action[:, :, None] - final_lower[:, None, :]
        ) / final_span[:, None, :]
        decreasing_q = (
            final_upper[:, None, :] - human_action[:, :, None]
        ) / final_span[:, None, :]
        q = torch.where(decreasing[:, None, None], decreasing_q, increasing_q)
        inside = (q > 1.e-5) & (q < 1.0 - 1.e-5) & (authority[:, None, None] > 1.e-5)
        q_safe = q.clamp(1.e-5, 1.0 - 1.e-5)
        policy_raw = torch.logit(q_safe)
        policy = torch.distributions.Normal(
            policy_raw_mean[:, None, None],
            policy_raw_std[:, None, None].clamp_min(1.e-4),
        )
        final_jacobian = (
            final_span[:, None, :] * q_safe * (1.0 - q_safe)
        ).clamp_min(1.e-8)
        log_final = policy.log_prob(policy_raw) - torch.log(final_jacobian)
        log_final = torch.where(inside, log_final, torch.full_like(log_final, -60.0))
        # Quadrature over the gate produces the marginal final-action density.
        log_final_marginal = torch.logsumexp(
            log_final + torch.log(weights.clamp_min(1.e-12))[None, None, :], dim=-1
        )
        return ((log_h_action - log_final_marginal) * weights[None]).sum(-1).clamp_min(0.0)


def _relation_candidates(
    states: np.ndarray, valid: np.ndarray, *, sequence_frames: int = 50,
) -> list[tuple[int, int, int, int, bool]]:
    """One auditable continuous crop per observed parent/child relation.

    The tuple is ``(frame, child agent index, parent agent index, role, risk)``.
    Role 1 is ego/direct following, role 2 is an observed adjacent cut-in
    conflict, and role 4 is an NPC-to-NPC secondary following relation.
    """
    last = min(173, states.shape[0] - 1)
    first = 24 + max(int(sequence_frames), 1) - 1
    if last <= first:
        return []
    frame_states = states[first:last]
    frame_valid = valid[first:last]
    child_next_valid = valid[first + 1:last + 1]
    gap = frame_states[:, :, None, 0] - frame_states[:, None, :, 0] - 4.8
    lateral = frame_states[:, :, None, 1] - frame_states[:, None, :, 1]
    closing = frame_states[:, None, :, 2] - frame_states[:, :, None, 2]
    future_lateral = lateral + (
        frame_states[:, :, None, 3] - frame_states[:, None, :, 3]
    ) * 1.5
    pair_valid = frame_valid[:, :, None] & frame_valid[:, None, :] & child_next_valid[:, None, :]
    pair_valid &= ~np.eye(7, dtype=bool)[None]
    # Child 0 is the external ADS ego and never an expert generator target.
    pair_valid[:, :, 0] = False
    following = pair_valid & (gap > .1) & (gap < 60.) & (np.abs(lateral) < 1.8)
    cutin = pair_valid & (gap > .1) & (gap < 35.) & (np.abs(lateral) >= 1.8) & (
        np.minimum(np.abs(lateral), np.abs(future_lateral)) < 1.8
    )
    cost = np.where(following, gap, np.where(cutin, gap + 5., np.inf))
    parent = cost.argmin(axis=1)
    best = np.take_along_axis(cost, parent[:, None], axis=1)[:, 0]
    by_relation: dict[tuple[int, int, bool], list[tuple[int, int, int, int, bool]]] = {}
    for local_frame, child in np.argwhere(np.isfinite(best)):
        parent_index = int(parent[local_frame, child])
        frame = int(local_frame + first)
        role = 2 if bool(cutin[local_frame, parent_index, child]) else 1 if parent_index == 0 else 4
        relation_gap = float(gap[local_frame, parent_index, child])
        relation_closing = float(closing[local_frame, parent_index, child])
        risk = bool(relation_closing > 1.0e-3 and relation_gap / relation_closing < 4.0)
        item = (frame, int(child), parent_index, role, risk)
        by_relation.setdefault((int(child), parent_index, role, risk), []).append(item)
    # Preserve multimodality without allowing long recordings to dominate:
    # each child/role/risk relation contributes its most compressed crop.
    return [min(items, key=lambda item: states[item[0], item[2], 0] - states[item[0], item[1], 0])
            for items in by_relation.values()]


def build_human_expert_samples(
    arrays: dict[str, np.ndarray], rows: np.ndarray, *, max_samples: int, seed: int,
    return_metadata: bool = False, sequence_frames: int = 50,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Build causal highD parent--child longitudinal action samples.

    Formal mode (``max_samples <= 0``) scans every supplied highD recording
    and retains each distinct direct-following, cut-in-conflict and secondary
    relation/risk stratum.  A positive bound is only for diagnostics and uses
    a deterministic reservoir over that same scan.
    """
    states_all = np.asarray(arrays["agent_states"])
    valid_all = np.asarray(arrays["agent_valid"])
    rng = np.random.default_rng(seed)
    records: list[tuple[np.ndarray, float, int, float, float, float, int, int, int, int]] = []
    seen = 0
    for scanned, row in enumerate(np.asarray(rows, np.int64), start=1):
        states, valid = states_all[int(row)], valid_all[int(row)]
        for frame, child, parent, role, risk in _relation_candidates(
            states, valid, sequence_frames=sequence_frames,
        ):
            history = torch.from_numpy(states[None, frame - 24:frame + 1].astype(np.float32))
            controls = torch.zeros(1, 5, 2)
            parent_speed = np.linalg.norm(states[frame - 5:frame + 1, parent, 2:4], axis=-1)
            controls[0, :, 0] = torch.from_numpy(np.diff(parent_speed).astype(np.float32) / .04)
            feature = human_prior_features(
                history, controls, target_slot_index=child - 1,
                parent_index=torch.tensor((parent,), dtype=torch.long),
                role=torch.tensor((role,), dtype=torch.long),
            ).numpy()[0]
            speed = np.linalg.norm(states[frame:frame + 2, child, 2:4], axis=-1)
            action = float(np.clip((speed[1] - speed[0]) / .04, -8., 4.))
            gap = float(states[frame, parent, 0] - states[frame, child, 0] - 4.8)
            closing = float(states[frame, child, 2] - states[frame, parent, 2])
            ttc = float(np.clip(gap / closing, 0., 10.)) if closing > 1.e-4 else 10.0
            record = (
                feature, action, role, gap, closing, ttc,
                int(row), int(frame), int(child), int(parent),
            )
            seen += 1
            if int(max_samples) <= 0 or len(records) < int(max_samples):
                records.append(record)
            else:
                replacement = int(rng.integers(seen))
                if replacement < int(max_samples):
                    records[replacement] = record
        if scanned % 5000 == 0:
            print(f"[human-prior] expert scan {scanned}/{len(rows)} rows, retained={len(records)}", flush=True)
    if not records:
        raise RuntimeError("no causal parent--child expert samples for human prior")
    features = np.asarray([item[0] for item in records], np.float32)
    actions = np.asarray([item[1] for item in records], np.float32)
    if not return_metadata:
        return features, actions
    metadata = {
        "role": np.asarray([item[2] for item in records], np.int8),
        "gap_m": np.asarray([item[3] for item in records], np.float32),
        "closing_mps": np.asarray([item[4] for item in records], np.float32),
        "ttc_s": np.asarray([item[5] for item in records], np.float32),
        "row": np.asarray([item[6] for item in records], np.int64),
        "frame": np.asarray([item[7] for item in records], np.int16),
        "child": np.asarray([item[8] for item in records], np.int8),
        "parent": np.asarray([item[9] for item in records], np.int8),
    }
    return features, actions, metadata


def build_human_reference_samples(
    arrays: dict[str, np.ndarray], rows: np.ndarray, *, max_samples: int, seed: int, min_per_ttc_bin: int = 256,
) -> dict[str, np.ndarray]:
    """Balanced held-out human action *and jerk* samples with causal context.

    This intentionally mirrors the expert sampler's ordinary/critical balance
    but also retains the physical conditioning variables.  It is evaluation
    evidence only and never feeds the prior or PPO optimiser.
    """
    states = np.asarray(arrays["agent_states"])[np.asarray(rows, np.int64)]
    valid = np.asarray(arrays["agent_valid"])[np.asarray(rows, np.int64)]
    rng = np.random.default_rng(seed)
    # Critical following is a long-tail event in a factual highD test split.
    # Equal-size TTC strata provide enough samples for a *conditional*
    # distribution comparison without pretending that the strata prevalence
    # itself is natural.  Prevalence remains a separate reported statistic.
    buckets = {0: ([], [], [], [], []), 1: ([], [], [], [], []), 2: ([], [], [], [], [])}
    quota = max(1, int(min_per_ttc_bin))
    attempts, maximum = 0, max(int(max_samples) * 50, quota * 5000)
    last_frame = min(173, states.shape[1] - 1)
    while min(len(values[0]) for values in buckets.values()) < quota and attempts < maximum:
        attempts += 1
        sequence, frame = int(rng.integers(len(states))), int(rng.integers(25, last_frame))
        if not (valid[sequence, frame - 1, 2] and valid[sequence, frame, 0] and valid[sequence, frame, 2] and valid[sequence, frame + 1, 2]):
            continue
        ego, rear = states[sequence, frame, 0], states[sequence, frame, 2]
        gap = float(ego[0] - rear[0] - 4.8)
        closing = float(rear[2] - ego[2])
        if not (gap > .1 and abs(float(ego[1] - rear[1])) < 1.8 and rear[0] < ego[0]):
            continue
        ttc = min(max(gap / closing, 0.), 10.) if closing > 1.e-4 else 10.
        index = 0 if ttc < 2. else 1 if ttc < 4. else 2
        if len(buckets[index][0]) >= quota:
            continue
        rear_v = states[sequence, frame - 1:frame + 2, 2, 2]
        now = float(np.clip((rear_v[2] - rear_v[1]) / .04, -8., 4.))
        previous = float(np.clip((rear_v[1] - rear_v[0]) / .04, -8., 4.))
        values = buckets[index]
        values[0].append(now); values[1].append(abs(now - previous) / .04)
        values[2].append(gap); values[3].append(closing); values[4].append(ttc)
    action, jerk, gap_values, closing_values, ttc_values = ([], [], [], [], [])
    for values in buckets.values():
        action.extend(values[0]); jerk.extend(values[1]); gap_values.extend(values[2]); closing_values.extend(values[3]); ttc_values.extend(values[4])
    if not action:
        raise RuntimeError("no held-out same-rear human reference samples")
    return {
        "final_ax_mps2": np.asarray(action, np.float32), "jerk_mps3": np.asarray(jerk, np.float32),
        "gap_m": np.asarray(gap_values, np.float32), "closing_mps": np.asarray(closing_values, np.float32),
        "ttc_s": np.asarray(ttc_values, np.float32),
    }


@dataclass(frozen=True)
class HumanPriorConfig:
    bc_epochs: int = 30
    gail_epochs: int = 20
    batch_size: int = 2048
    actor_learning_rate: float = 1e-4
    critic_learning_rate: float = 1e-3
    discriminator_learning_rate: float = 1e-4
    gail_bc_initial_weight: float = .1
    gail_bc_decay_passes: int = 3
    # Small temporal naturalness regularizer used only while refining the
    # generator.  The expert action remains the adversarial target; this term
    # matches one-step jerk to the paired highD sequence and prevents a
    # discriminator loophole based on iid action noise.
    gail_jerk_weight: float = .10
    gamma: float = .99
    gae_lambda: float = .95
    clip_ratio: float = .2
    ppo_epochs: int = 4
    patience: int = 5
    max_expert_samples: int = 0
    trajectory_frames: int = 50
    highway_rollout_batch: int = 32
    refinement_source_rows: int = 0
    seed: int = 20260902


def train_human_prior(
    features: np.ndarray,
    actions: np.ndarray,
    *,
    config: HumanPriorConfig,
    device: torch.device,
    validation_features: np.ndarray | None = None,
    validation_actions: np.ndarray | None = None,
) -> tuple[HumanActionPrior, dict[str, Any]]:
    """Probability-BC initialization for the closed-loop GAIL stage."""
    torch.manual_seed(config.seed)
    model = HumanActionPrior().to(device)
    expert_x, expert_a = torch.from_numpy(features).to(device), torch.from_numpy(actions).to(device)
    actor_parameters = [
        *model.actor.parameters(), *model.mean.parameters(),
        *model.log_std.parameters(),
    ]
    actor_optimizer = torch.optim.Adam(actor_parameters, lr=config.actor_learning_rate)
    history: list[dict[str, float]] = []
    val_x = None if validation_features is None else torch.from_numpy(validation_features).to(device)
    val_a = None if validation_actions is None else torch.from_numpy(validation_actions).to(device)
    best_state, best_val, stale = None, float("inf"), 0
    for epoch in range(int(config.bc_epochs)):
        order = torch.randperm(len(expert_x), device=device)
        values: list[float] = []
        for start in range(0, len(order), config.batch_size):
            index = order[start:start + config.batch_size]
            x, expert = expert_x[index], expert_a[index]
            loss = -model.action_log_prob(x, expert).mean()
            actor_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor_parameters, 1.0)
            actor_optimizer.step()
            values.append(float(loss.detach()))
        with torch.no_grad():
            val_nll = (
                float(-model.action_log_prob(val_x, val_a).mean())
                if val_x is not None else float(np.mean(values))
            )
        history.append({
            "phase": "bc", "epoch": float(epoch),
            "loss": float(np.mean(values)), "validation_nll": val_nll,
        })
        if val_nll < best_val - 1.0e-5:
            best_val, stale = val_nll, 0
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        else:
            stale += 1
            if stale >= 5:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"schema": "longitudinal_gail_human_prior_v4", "history": history,
                   "expert_samples": int(len(features)), "best_validation_nll": float(best_val)}
