from __future__ import annotations

import numpy as np
import torch

from normalizing_flow.src.constraints import (
    constraint_from_model_coordinates,
    constraint_to_model_coordinates,
)
from normalizing_flow.src.features import (
    SLOT_NAMES,
    build_feature_schema,
    feature_index,
)
from normalizing_flow.src.metrics import physical_validity_flags
from normalizing_flow.src.scenario import DirectScenarioFlow, project_constraint


class _GaussianFlow(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(()))

    def log_prob(self, values, context=None):
        del context
        return -0.5 * values.square().sum(dim=1) + self.bias


def _model() -> DirectScenarioFlow:
    dimensions = np.zeros((2, 7, 2), np.float32)
    dimensions[..., 0] = 4.8
    dimensions[..., 1] = 1.8
    return DirectScenarioFlow(
        c0_flow=_GaussianFlow(),
        constraint_flow=_GaussianFlow(),
        mask_log_prob=torch.tensor([0.0] + [float("-inf")] * 63, dtype=torch.float32),
        dimension_bank=torch.from_numpy(dimensions),
        dimension_pattern=torch.zeros(2, dtype=torch.long),
        schema={"feature_names": list(build_feature_schema().feature_names)},
    )


def test_compact_contract_dimensions() -> None:
    assert len(build_feature_schema().feature_names) == 26


def test_joint_log_prob_is_sum_of_factors() -> None:
    model = _model()
    terms = model.log_prob_tensors(
        c0_normalized=torch.zeros(2, 26),
        slot_mask=torch.zeros(2, 6, dtype=torch.bool),
        constraint_normalized=torch.zeros(2, 6, 6),
    )
    torch.testing.assert_close(
        terms["joint_log_prob"],
        terms["mask_log_prob"] + terms["c0_log_prob"] + terms["k_log_prob"],
    )


def test_future_constraint_does_not_change_initial_factor() -> None:
    model = _model()
    arguments = {
        "c0_normalized": torch.randn(3, 26),
        "slot_mask": torch.zeros(3, 6, dtype=torch.bool),
    }
    first = model.log_prob_tensors(
        **arguments, constraint_normalized=torch.zeros(3, 6, 6)
    )
    second = model.log_prob_tensors(
        **arguments, constraint_normalized=torch.ones(3, 6, 6)
    )
    torch.testing.assert_close(first["c0_log_prob"], second["c0_log_prob"])
    assert not torch.equal(first["k_log_prob"], second["k_log_prob"])


def test_speed_change_coordinates_round_trip() -> None:
    c0 = np.zeros((2, 26), np.float32)
    c0[:, feature_index(None, "ego_vx_mps")] = [20.0, 30.0]
    for name in SLOT_NAMES:
        c0[:, feature_index(name, "rel_vx_mps")] = 1.0
    values = np.zeros((2, 6, 6), np.float32)
    values[..., (0, 2, 4)] = (42.0, 86.0, 130.0)
    values[..., (1, 3, 5)] = (0.1, 0.2, 0.3)
    model = constraint_to_model_coordinates(values, c0)
    restored = constraint_from_model_coordinates(model, c0)
    np.testing.assert_allclose(restored, values, atol=1.0e-5)


def test_position_projection_produces_legal_interval_speeds() -> None:
    c0 = np.zeros((1, 26), np.float32)
    c0[:, feature_index(None, "ego_vx_mps")] = 20.0
    values = np.zeros((1, 6, 6), np.float32)
    values[..., (0, 2, 4)] = (100.0, 20.0, -5.0)
    slots = np.ones((1, 6), bool)
    projected = project_constraint(values, slots, c0)
    positions = np.concatenate(
        (np.zeros((1, 6, 1)), projected[..., (0, 2, 4)]), axis=-1
    )
    speed = np.diff(positions, axis=-1) / np.diff((0.0, 2.0, 4.0, 5.96))
    assert ((speed >= 0.0) & (speed <= 70.0)).all()


def test_empirical_dimensions_follow_sampled_mask() -> None:
    model = _model()
    slots, dimensions = model._structure_from_uniform(np.asarray([0.25, 0.75]))
    assert not slots.any()
    np.testing.assert_allclose(dimensions[..., 0], 4.8)
    np.testing.assert_allclose(dimensions[..., 1], 1.8)


def test_dimension_metrics_ignore_empty_slots() -> None:
    from types import SimpleNamespace

    from normalizing_flow.src.evaluation import _dimension_metrics

    slots = np.ones((4, 6), bool)
    slots[0] = False
    dimensions = np.broadcast_to([4.8, 1.8], (4, 7, 2)).copy()
    generated = dimensions.copy()
    dimensions[0, 1:] = 0.0
    generated[0, 1:] = 100.0
    metrics = _dimension_metrics(
        {"vehicle_dimensions": dimensions, "slot_mask": slots},
        np.arange(4),
        SimpleNamespace(vehicle_dimensions=generated, slot_mask=slots),
    )
    assert len(metrics["per_feature"]) == 14
    assert metrics["mean_ks"] == metrics["mean_wasserstein"] == 0.0


def test_empirical_dimensions_use_the_matching_mask_bank() -> None:
    dimensions = np.zeros((2, 7, 2), np.float32)
    dimensions[0] = (4.0, 1.7)
    dimensions[1] = (5.0, 2.0)
    model = DirectScenarioFlow(
        c0_flow=_GaussianFlow(),
        constraint_flow=_GaussianFlow(),
        mask_log_prob=torch.tensor(
            [float("-inf"), np.log(0.5), float("-inf"), np.log(0.5)]
            + [float("-inf")] * 60,
            dtype=torch.float32,
        ),
        dimension_bank=torch.from_numpy(dimensions),
        dimension_pattern=torch.tensor([1, 3]),
        schema={"feature_names": list(build_feature_schema().feature_names)},
    )
    slots, sampled = model._structure_from_uniform(np.asarray([0.0, 0.75]))
    assert slots.tolist() == [
        [True, False, False, False, False, False],
        [True, True, False, False, False, False],
    ]
    np.testing.assert_allclose(sampled[:, 0, 0], [4.0, 5.0])


def test_initial_overlap_checks_npc_pairs() -> None:
    c0 = np.zeros((1, 26), np.float32)
    c0[:, feature_index(None, "ego_vx_mps")] = 25.0
    slots = np.zeros((1, 6), bool)
    for name, x in (("left_front", 2.0), ("left_rear", -2.0)):
        slots[:, SLOT_NAMES.index(name)] = True
        c0[:, feature_index(name, "rel_x_m")] = x
        c0[:, feature_index(name, "rel_y_left_m")] = 3.6
    dimensions = np.tile([4.8, 1.8], (1, 7, 1)).astype(np.float32)
    invalid, reasons, _ = physical_validity_flags(c0, slots, dimensions)
    assert invalid.item()
    assert reasons["npc_pair_box_overlap"] == 1
