"""Coverage, ancestry weighting and AdamW-displacement retention checks."""

from types import SimpleNamespace

import numpy as np
import torch

from ads.diffusion_planner.constraints import constrain_step, project_displacement
from ads.diffusion_planner.repair import valid_counts
from ads.diffusion_planner.sampling import BalancedSampler


def test_full_coverage_preserves_reference_learning_rate_mass_and_ema_age():
    from ads.diffusion_planner.repair import normalized_schedule

    reference, learning_rate, decay = 250, 1e-5, 0.999
    mass = learning_rate * np.maximum(0.1, 1 - np.arange(reference) / reference).sum()
    for updates in (250, 500, 21817):
        schedule = normalized_schedule(updates, reference, learning_rate, decay)
        actual = (
            schedule["learning_rate"]
            * np.maximum(0.1, 1 - np.arange(updates) / updates).sum()
        )
        np.testing.assert_allclose(actual, mass, rtol=1e-12)
        np.testing.assert_allclose(
            schedule["ema_decay"] ** updates, decay**reference, rtol=1e-11
        )


def test_scene_coverage_precedes_repetition_and_resume_is_exact():
    samples = [
        {"scene_id": str(i), "instance_id": str(i), "frame_id": j}
        for i in range(20)
        for j in range(3)
    ]
    sampler = BalancedSampler(samples, 20260926)
    first = sampler.draw(20)
    assert len({s["scene_id"] for s in first}) == 20
    state = sampler.state_dict()
    restored = BalancedSampler(samples, 0, state=state)
    assert sampler.draw(61) == restored.draw(61)


def test_more_descendants_do_not_increase_ancestor_weight():
    samples = [
        {"scene_id": root, "instance_id": root + str(i)}
        for root, count in (("a", 100), ("b", 1), ("c", 4))
        for i in range(count)
    ]
    sampler = BalancedSampler(samples, 5)
    sampler.draw(len(samples))
    counts = sampler.report()["root_visits"]
    assert sampler.report()["visited_instances"] == len(samples)
    sampler.draw(600)
    assert {
        n - counts[root] for root, n in sampler.report()["root_visits"].items()
    } == {200}


def test_new_query_material_extends_ancestor_sampler_without_resetting_exposure():
    samples = [{"scene_id": "a", "instance_id": "initial"}]
    sampler = BalancedSampler(samples, 5)
    sampler.draw(5)
    extended = samples + [{"scene_id": "a", "instance_id": "query"}]
    resumed = BalancedSampler(extended, 5, state=sampler.state_dict())
    drawn = resumed.draw(4)
    assert {s["instance_id"] for s in drawn} == {"initial", "query"}
    assert resumed.report()["exposures"] == 9


def test_projection_is_closest_feasible_point_for_two_constraints():
    delta = torch.tensor([2.0, 3.0])
    gradients = [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])]
    projected, report = project_displacement(delta, gradients)
    assert torch.allclose(projected, torch.zeros(2, dtype=torch.float64))
    assert report["active_constraints"] == 2
    projected, _ = project_displacement(torch.tensor([-1.0, 2.0]), gradients)
    assert torch.allclose(projected, torch.tensor([-1.0, 0.0], dtype=torch.float64))
    opposite = [torch.tensor([1.0, 0.0]), torch.tensor([-1.0, 0.0])]
    projected, _ = project_displacement(delta, opposite)
    assert torch.allclose(projected, torch.tensor([0.0, 3.0], dtype=torch.float64))


def test_constraint_applies_after_adamw_preconditioning_and_decay():
    parameter = torch.nn.Parameter(torch.tensor([2.0, -3.0]))
    optimizer = torch.optim.AdamW([parameter], lr=0.1, weight_decay=0.1)
    gradient = torch.tensor([1.0, 1.0])
    before = parameter.detach().clone()
    parameter.grad = torch.tensor([-1.0, 2.0])
    optimizer.step()
    unconstrained = parameter.detach().clone()
    report = constrain_step([parameter], before, [gradient, torch.zeros(2)])
    assert gradient.dot(parameter.detach() - before) <= 1e-7
    assert np.isfinite(report["actual_step_norm"])
    assert not torch.equal(parameter, before)
    assert not torch.equal(unconstrained, before)


def test_fp32_boundary_rounding_preserves_high_dimensional_updates():
    """A small inward correction keeps useful updates without relaxing GEM."""
    generator = torch.Generator().manual_seed(20260926)
    for _ in range(12):
        before = torch.randn(4096, generator=generator) * 2
        gradients = [torch.randn(4096, generator=generator) for _ in range(2)]
        proposal = torch.randn(4096, generator=generator) * 1e-5
        for gradient in gradients:
            proposal += 1e-4 * gradient / gradient.norm()
        parameter = torch.nn.Parameter(before + proposal)
        ideal, _ = project_displacement(
            parameter.detach().double() - before.double(), gradients
        )
        report = constrain_step([parameter], before, gradients)
        actual = parameter.detach().double() - before.double()
        assert actual.norm() > 0.99 * ideal.norm()
        assert (actual - ideal).norm() < 1e-5
        assert max(report["actual_normalized_dots"]) <= 1e-9
        assert not report.get("rejected_for_rounding", False)


def test_opposite_constraints_keep_representable_tangential_updates():
    parameter = torch.nn.Parameter(torch.tensor([2.25, 0.751]))
    before = torch.tensor([2.0, 0.75])
    report = constrain_step(
        [parameter], before, [torch.tensor([1.0, 0.0]), torch.tensor([-1.0, 0.0])]
    )
    assert parameter[0] == before[0]
    assert parameter[1] > before[1]
    assert max(report["actual_normalized_dots"]) <= 1e-9


def test_variable_horizon_gradient_accumulation_matches_full_source(monkeypatch):
    from ads.nuplan import losses

    weight = torch.nn.Parameter(torch.tensor(0.7))
    samples = []
    for length in (80, 40, 8, 60):
        valid = np.zeros((2, 80), bool)
        valid[0, :length] = True
        valid[1, : length // 2] = True
        samples.append({"future_valid": valid})

    def batch(policy, rows):
        mask = torch.as_tensor(np.stack([s["future_valid"] for s in rows]))
        future = torch.ones((len(rows), 2, 80, 4))
        trajectory = torch.zeros((len(rows), 2, 81, 4))
        return {}, trajectory, None, future, mask, None, None

    monkeypatch.setattr(losses, "diffusion_batch", batch)
    policy = SimpleNamespace(
        args=SimpleNamespace(diffusion_model_type="x_start"),
        denoise=lambda inputs, trajectory, time: torch.ones_like(trajectory) * weight,
    )
    losses.diffusion_loss(policy, samples).backward()
    expected = weight.grad.clone()
    weight.grad = None
    counts = valid_counts(samples)
    for offset in (0, 2):
        losses.diffusion_loss(
            policy, samples[offset : offset + 2], denominators=counts
        ).backward()
    torch.testing.assert_close(weight.grad, expected)


def test_all_key_and_other_entries_are_covered_and_resume_is_exact():
    samples = [
        {
            "scene_id": str(root),
            "instance_id": str(root),
            "frame_id": frame,
            "critical_window": frame in (1, 2),
        }
        for root in range(12)
        for frame in range(8)
    ]
    sampler = BalancedSampler(samples, 20260926, key_windows=True)
    first = sampler.draw(48)
    assert len({(r["instance_id"], r["frame_id"]) for r in first[:24]}) == 24
    assert all(row["critical_window"] for row in first[:24])
    restored = BalancedSampler(samples, 0, key_windows=True, state=sampler.state_dict())
    assert sampler.draw(96) == restored.draw(96)
    report = sampler.report()
    assert report["unvisited_entries"] == 0
    assert report["minimum_critical_visits"] >= 2


def test_exhausted_ancestors_yield_until_the_entire_pool_is_covered():
    samples = [
        {
            "scene_id": root,
            "instance_id": root + str(i),
            "frame_id": frame,
            "critical_window": frame == 0,
        }
        for root, count in [("many", 600)] + [(f"other{i}", 1) for i in range(31)]
        for i in range(count)
        for frame in (0, 5)
    ]
    sampler = BalancedSampler(samples, 20260926, key_windows=True)
    first = sampler.draw(1262)
    assert len({(r["instance_id"], r["frame_id"]) for r in first}) == 1262
    assert sampler.report()["unvisited_entries"] == 0
    sampler.draw(1262)
    counts = sampler.report()["root_visits"]
    sampler.draw(640)
    increments = [
        n - counts[root] for root, n in sampler.report()["root_visits"].items()
    ]
    assert max(increments) - min(increments) <= 2
    assert sum(increments) == 640


def test_added_queries_are_learned_without_resetting_old_visits():
    samples = [
        {
            "scene_id": "a",
            "instance_id": "initial",
            "frame_id": 0,
            "critical_window": True,
        }
    ]
    sampler = BalancedSampler(samples, 2, key_windows=True)
    sampler.draw(20)
    state = sampler.state_dict()
    extended = samples + [
        {
            "scene_id": "a",
            "instance_id": "query",
            "frame_id": 20,
            "critical_window": True,
        }
    ]
    restored = BalancedSampler(extended, 0, key_windows=True, state=state)
    assert restored.draw(1)[0]["instance_id"] == "query"
    assert restored.report()["exposures"] == 21


def test_full_data_budget_exceeds_old_cap_and_transfers_empty_quota():
    from ads.diffusion_planner.sampling import phase_budget

    native = [None] * 12685
    keep = [None] * 30000
    corrections = [{"critical_window": True} for _ in range(9000)]
    plan = phase_budget(native, keep, corrections)
    assert plan["updates"] == 938
    assert plan["critical_quota"] == 32 and plan["other_quota"] == 0
    assert plan["entries"] == {
        "native": 12685,
        "keep": 30000,
        "critical": 9000,
        "other_correction": 0,
    }
