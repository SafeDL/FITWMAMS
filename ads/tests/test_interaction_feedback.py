"""Coverage targets, state-dependent kernels, snapshots and deterministic waves."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from ads.search.engine import Branch, SearchEngine
from ads.search.feedback import (
    BehaviorCatalog,
    CoverageTarget,
    coverage_radius,
    proposal_configuration,
    risk_gain,
    systematic_resample,
    trajectory_hash,
    transition_acceptance,
)
from ads.tests.test_search import evaluate, small_config
from hierarchical_world_model.src.randomness import WORLD_RANDOM_BLOCKS
from hierarchical_world_model.src.scene_geometry import SceneGeometry


@pytest.mark.parametrize(
    "phase,cooldown,entered,adopted,y,width,expected",
    [
        (0, 0, False, False, 0.0, 4.0, True),
        (3, 1, False, False, 0.0, 4.0, True),
        (3, 2, False, False, 0.0, 4.0, False),
        (2, 0, False, False, 0.0, 4.0, False),
        (0, 0, True, True, 0.0, 4.0, False),
        (0, 0, True, False, 0.0, 4.0, True),
        (0, 0, True, None, 0.0, 4.0, False),
        (None, None, False, None, 0.0, 4.0, False),
        (0, 0, False, False, 0.75, 3.5, False),
        (0, 0, False, False, 0.65, 3.5, True),
    ],
)
def test_maneuver_dependencies_follow_observed_policy_stage(
    instance, phase, cooldown, entered, adopted, y, width, expected
):
    from ads.common.evaluation import annotate_dependencies

    result = evaluate(instance)
    event = instance.event
    geometry = SceneGeometry.straight_template(
        lane_centers_y=np.array((-4.0, 0.0, 4.0)), lane_widths=width
    )
    result.instance = SimpleNamespace(
        event=SimpleNamespace(
            exogenous_state=event.exogenous_state, valid=event.valid, geometry=geometry
        )
    )
    result.states = np.repeat(event.initial_states[None], 2, axis=0)
    result.states[:, 1, 1] = y
    result.controls = np.zeros((1, 2), np.float32)
    result.record.update(critical_slot=1, critical_frame=1, physics_steps=1)
    diagnostic = {
        "takeover_owned": np.array((True, False, False, False, False, False)),
        "takeover_started": np.array((entered, False, False, False, False, False)),
    }
    if phase is not None:
        diagnostic["input_maneuver_phase"] = np.array((phase, 0, 0, 0, 0, 0))
        diagnostic["input_maneuver_cooldown"] = np.array((cooldown, 0, 0, 0, 0, 0))
    if adopted is not None:
        diagnostic["input_maneuver_adopted"] = np.array(
            (adopted, False, False, False, False, False)
        )
    result.diagnostics = [diagnostic]
    controller = SimpleNamespace(
        driver_correction=None,
        mobil_onset_calibration={},
        mobil_parameter_variation=False,
        lane_style=object(),
    )
    annotate_dependencies(result, controller)
    response = np.zeros_like(event.exogenous_state.policy_response_innovations, bool)
    response.reshape(-1)[
        result.record["input_dependencies"]["policy_response_innovations"]
    ] = True
    assert bool(response[0, 0, 0, 0]) == expected
    assert int(response.sum()) == int(expected)
    assert result.record["maneuver_unobserved_frames"] == ([0] if phase is None else [])
    assert result.record["maneuver_unobserved_admissions"] == (
        [[0, 0]] if entered and adopted is None else []
    )
    if adopted:
        assert result.record["maneuver_adoptions"] == [[0, 0]]
        style = np.zeros_like(event.exogenous_state.maneuver_latent, bool)
        style.reshape(-1)[result.record["input_dependencies"]["maneuver_latent"]] = True
        np.testing.assert_array_equal(style[0, 0], (False, False, True, True))


def point(source, x, failure=True):
    result = evaluate(source.sample(1)[0])
    result.states = np.full((1, 7, 6), x, np.float32)
    result.record.update(
        descriptor=[x, 0, 0, 0],
        collision=failure,
        safety_failure=failure,
        critical_slot=1,
    )
    result.record["trajectory_hash"] = trajectory_hash(result)
    return result


def test_catalog_batch_freeze_duplicates_and_diminishing_gain(source):
    a, b, c = (point(source, x) for x in (0, 0.1, 2))
    catalog = BehaviorCatalog(0.2)
    catalog.update([a])
    target = catalog.target([a], 1)
    gains = catalog.gains([a], [a, b, c, c], target)
    assert gains[0] == gains[3] == 0
    assert gains[2] > gains[1] > 0
    assert target.counts == {0: 1}
    assert target.cell(c) == -1
    catalog.update([b, c])
    following = catalog.target([a, b, c, c], 1)
    assert following.counts == {0: 2, 1: 1}
    assert target.version < following.version
    assert following.weight(a) < following.weight(c)


def test_missing_descriptor_receives_base_only(source):
    a = point(source, 0)
    a.record["critical_slot"] = -1
    catalog = BehaviorCatalog(0.1)
    target = catalog.target([], 1)
    assert target.weight(a) == 0.05
    assert catalog.gains([], [a, a], target) == [0.05, 0]


def test_joint_keeps_history_but_single_operators_only_mutate_written_response(source):
    a = point(source, 0)
    omega = a.instance.event.exogenous_state
    selected = np.zeros_like(omega.policy_response_innovations, bool)
    selected[0, 0, 0, 1] = selected[0, 35, 0, 1] = True
    a.record.update(
        critical_frame=40,
        input_dependencies={
            "policy_response_innovations": np.flatnonzero(selected).tolist()
        },
        static_input_components={"maneuver_latent": [0, 2, 3]},
    )
    proposal = proposal_configuration([a], dict.fromkeys(WORLD_RANDOM_BLOCKS, 0))
    response = [
        mask
        for name, mask in proposal.operators
        if name == "policy_response_innovations"
    ]
    assert len(response) == 2
    assert np.array_equal(np.logical_or.reduce(response), selected)
    assert proposal.joint_masks["policy_response_innovations"][0, 1, 0, 1]
    assert not proposal.joint_masks["policy_calibration_innovations"][:, 1:].any()
    assert not any(mask[0, :, 1].any() for mask in response)
    assert all(
        mask[..., 1].sum() == 0
        for name, mask in proposal.operators
        if name == "maneuver_latent"
    )
    assert proposal.probabilities(a)[0] == 0.2
    frozen = [mask.copy() for _, mask in proposal.operators]
    a.record["input_dependencies"] = {}
    assert all(
        np.array_equal(mask, original)
        for (_, mask), original in zip(proposal.operators, frozen)
    )
    assert all(proposal.probabilities(a) > 0)


def test_combined_acceptance_enforces_detailed_balance(source):
    a, b = point(source, -1), point(source, 1)
    target = CoverageTarget(
        1,
        0.1,
        np.array([[-1, 0, 0, 0], [1, 0, 0, 0]]),
        {0: 20, 1: 0},
        1,
    )

    class Operators:
        def probabilities(self, value):
            return np.array([0.2, 0.1 if value is a else 0.7])

    operators = Operators()
    forward, reverse = 0.1, 0.7
    ab = transition_acceptance(a, b, target, operators, 1, forward)
    ba = transition_acceptance(b, a, target, operators, 1, reverse)
    assert np.isclose(target.weight(a) * forward * ab, target.weight(b) * reverse * ba)
    b.record["safety_failure"] = False
    assert transition_acceptance(a, b, target, operators, 1, forward) == 0


def test_incremental_target_ratios_do_not_repeatedly_multiply_coverage(source):
    a, b = point(source, 0), point(source, 2)
    catalog = BehaviorCatalog(0.1)
    catalog.update([a, b])
    old = catalog.target([a, b], 1)
    current = catalog.target([a, b, point(source, 0.01)], 1)
    weights = np.array([current.weight(r) / old.weight(r) for r in (a, b)])
    rng = np.random.default_rng(1)
    draws = systematic_resample(weights, 10000, rng)
    assert abs(np.mean(draws == 0) - weights[0] / weights.sum()) < 0.001
    np.testing.assert_allclose(
        [current.weight(r) / current.weight(r) for r in (a, b)], 1
    )


def test_restoration_retains_complete_old_target_population(source):
    engine = SearchEngine(
        source,
        evaluate,
        config=small_config(budget=80, batch_size=20, subset_max_levels=8),
    )
    population = [point(source, x, False) for x in np.linspace(-1, 1, 12)]
    radius, _ = coverage_radius(population, 0.1)
    engine.coverage = BehaviorCatalog(min(radius, 0.001))
    engine.coverage.update(population)
    target = engine.coverage.target(population, 0)
    path = Branch("first", population, depth=2, threshold=0, target=target)
    low = population[0]
    engine._remember([low], path)
    restored = engine._restore([], 0)
    assert restored.population == population
    assert restored.target is target
    engine._advance(restored, "guided")
    level = engine.nodes[restored.identity]["levels"][0]
    assert level["threshold"] == 0 and level["restored_first_batch"]
    assert len(level["incremental_weights"]) == len(population)
    assert restored.depth == 2  # Same-layer work does not consume a risk level.


def test_positive_progress_reward_is_normalized_and_failure_saturated(source):
    a, b = [evaluate(i) for i in source.sample(2)]
    a.record.update(safety_failure=False, event_risk=2)
    b.record.update(safety_failure=False, event_risk=3)
    assert np.isclose(risk_gain(a, b), 0.25)
    assert risk_gain(b, a) == 0
    b.record["safety_failure"] = True
    assert risk_gain(a, b) == 1 and risk_gain(b, a) == 0


@pytest.mark.parametrize("method", ("monte_carlo", "subset", "guided"))
def test_parallel_waves_preserve_serial_proposals_chains_and_feedback(source, method):
    from concurrent.futures import ThreadPoolExecutor

    from ads.common.instances import GeneratedSource

    class BatchEvaluator:
        def __call__(self, instance):
            return evaluate(instance)

        def many(self, instances):
            return [evaluate(instance) for instance in instances]

    results = []
    for workers in (1, 4):
        fresh = GeneratedSource(
            source.sampler, source.assets, master_seed=123, purpose="search"
        )
        with ThreadPoolExecutor(max_workers=4) as pool:
            results.append(
                SearchEngine(
                    fresh,
                    BatchEvaluator(),
                    config=small_config(
                        budget=173,
                        initial_samples=24,
                        batch_size=20,
                        generation_batch_size=8,
                        terminal_batch_size=5,
                        terminal_proposals=15,
                        subset_max_levels=3,
                    ),
                    evaluation_batch_size=workers,
                    generation_executor=pool if workers > 1 else None,
                ).run(method)
            )
    assert [r.instance.instance_hash for r in results[0].records] == [
        r.instance.instance_hash for r in results[1].records
    ]
    assert [r.record for r in results[0].records] == [
        r.record for r in results[1].records
    ]
    assert results[0].branches == results[1].branches
    assert results[0].costs == results[1].costs


def test_fixed_weighted_target_matches_integrated_gaussian_law(source):
    """Both coverage and state-dependent operator ratios affect stationary mass."""
    result = point(source, -1)
    target = CoverageTarget(
        1,
        float("inf"),
        np.array([[-1, 0, 0, 0], [1, 0, 0, 0]]),
        {0: 0, 1: 20},
        1,
    )

    class Operators:
        def probabilities(self, value):
            return (
                np.array([0.2, 0.8])
                if value.record["descriptor"][0] < 0
                else np.array([0.7, 0.3])
            )

    proposal, rng, samples = Operators(), np.random.default_rng(20260926), []
    x = -1.0
    for step in range(42000):
        operator = int(rng.choice(2, p=proposal.probabilities(result)))
        beta = (0.3, 0.9)[operator]
        y = np.sqrt(1 - beta * beta) * x + beta * rng.normal()
        child = replace(result, record={**result.record, "descriptor": [y, 0, 0, 0]})
        probability = transition_acceptance(
            result,
            child,
            target,
            proposal,
            operator,
            proposal.probabilities(result)[operator],
        )
        if rng.random() < probability:
            x, result = y, child
        if step >= 2000:
            samples.append(x)
    negative = target.delta + np.log(2)
    positive = target.delta + np.log(22 / 21)
    expected_probability = positive / (negative + positive)
    expected_mean = (positive - negative) / (positive + negative) * np.sqrt(2 / np.pi)
    assert abs(np.mean(np.asarray(samples) > 0) - expected_probability) < 0.015
    assert abs(np.mean(samples) - expected_mean) < 0.035
