"""Finite piecewise derivative qualification, weak signals and sampled branches."""

from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from autoformalism.fitting import piecewise_assessment as p


def oracle(jacobian=None):
    """Two physical coordinates, including an explicitly fitted initializer."""
    return SimpleNamespace(
        names=("a", "init_x"),
        units=np.array([2.0, 0.5]),
        lower=np.array([-20.0, -20.0]),
        upper=np.array([20.0, 20.0]),
        system=SimpleNamespace(initial_parameter_names=("init_x",), has_piecewise=True),
        j=np.eye(2) if jacobian is None else np.asarray(jacobian),
    )


def evidence(o, x, function, policy=None, *, alternate=None, active=None):
    """Evaluate the whole predetermined plan without any adaptive step choice."""
    policy = policy or p.PiecewiseAssessmentPolicy()
    active = active or o.names
    alternate = alternate or function
    plan = p.perturbation_plan(o, x, active, policy, jacobian=o.j)
    evaluations = {
        point.key: p.PerturbationEvaluation(
            residual=tuple(function(np.asarray(point.vector))),
            alternate_residual=tuple(alternate(np.asarray(point.vector))),
        )
        for point in plan
        if point.in_domain
    }
    result = p.assess_derivatives(
        o,
        x,
        active,
        policy,
        function(x),
        o.j,
        evaluations,
        base_alternate_residual=alternate(x),
    )
    return plan, evaluations, result


@pytest.mark.parametrize("steps", [(0.001,), (0.001, 0.001), (0.0003, 0.001), (1, 0.1)])
def test_policy_requires_two_frozen_decreasing_steps(steps):
    with pytest.raises(ValidationError, match="decreasing steps"):
        p.PiecewiseAssessmentPolicy(steps=steps)


def test_plan_uses_fixed_units_canonical_order_and_all_initials():
    o, x, policy = oracle(), np.array([0.7, 1.1]), p.PiecewiseAssessmentPolicy()
    plan = p.perturbation_plan(o, x, ("init_x", "a"), policy)
    assert plan == p.perturbation_plan(o, x, o.names, policy)
    assert len(plan) == 8 and len({point.key for point in plan}) == 8
    np.testing.assert_allclose(plan[0].vector, x - [0.002, 0])
    assert plan[4].parameter == "init_x" and plan[4].initial_parameter
    np.testing.assert_allclose(plan[4].vector, x - [0, 0.0005])
    np.testing.assert_array_equal(x, [0.7, 1.1])


def test_supported_joint_evidence_is_local_and_not_calibrated_confidence():
    o, x = oracle(), np.array([0.7, 1.1])
    _, _, result = evidence(
        o,
        x,
        lambda value: o.j @ value,
        p.PiecewiseAssessmentPolicy(include_weak_direction=True),
    )
    assert result.status == "supported"
    assert result.initial_parameters_checked == ("init_x",)
    assert result.perturbations_required == result.perturbations_available == 12
    assert result.checked_direction_scope == "coordinate_axes_and_weak_joint"
    assert all(d.converged_across_steps for d in result.directions)
    assert not result.supported_classical and not result.supported_global
    assert not result.calibrated_confidence_claimed
    assert not result.solver_discrepancy_is_rigorous_error_bound
    assert (
        p.PiecewiseAssessmentResult.model_validate_json(result.model_dump_json())
        == result
    )


def test_bounds_are_explicit_infeasible_points_and_never_clipped():
    o, x = oracle(), np.array([0.0, 1.0])
    o.lower[0] = 0
    plan, _, result = evidence(o, x, lambda value: o.j @ value)
    assert plan[0].vector[0] < 0 and not plan[0].in_domain
    assert plan[0].reason == "outside_declared_bounds"
    assert result.status == "infeasible_direction"
    assert result.perturbations_available == 6
    assert result.directions[1].status == "supported"


def test_unrepresentable_step_does_not_become_a_zero_derivative():
    o, x = oracle(), np.array([1e15, 1.0])
    o.upper[0] = np.inf
    plan = p.perturbation_plan(o, x, o.names, p.PiecewiseAssessmentPolicy())
    assert not plan[0].in_domain and plan[0].reason == "unrepresentable_step"
    assert all(
        not point.in_domain and point.reason == "unrepresentable_step"
        for point in plan
        if point.parameter == "a"
    )


def test_negligible_joint_component_can_round_away_in_optimizer_units():
    o, x = oracle([[0.5, 2e-14], [0, 4.0]]), np.ones(2)
    policy = p.PiecewiseAssessmentPolicy(include_weak_direction=True)
    plan = p.perturbation_plan(o, x, o.names, policy, jacobian=o.j)
    weak_points = [point for point in plan if point.direction == "weak_joint"]
    assert weak_points and all(point.in_domain for point in weak_points)
    for point in weak_points:
        direction = np.asarray(point.direction_vector)
        assert 0 < abs(direction[1]) < 1e-12
        assert point.vector[1] == x[1]
        actual = (np.asarray(point.vector) - x) / o.units
        assert np.linalg.norm(actual - point.sign * point.step * direction) <= (
            1e-8 * point.step
        )


def test_zero_central_difference_at_abs_tie_does_not_hide_one_sided_kink():
    o, x = oracle([[0, 0], [0, 1]]), np.array([0.0, 1.0])
    _, _, result = evidence(o, x, lambda value: np.array([abs(value[0]), value[1]]))
    kink = result.directions[0]
    assert result.status == kink.status == "unsupported_kink"
    assert all(s.central_agrees and not s.sides_agree for s in kink.steps)


def test_weak_column_cannot_pass_using_absolute_tolerance():
    o, x = oracle([[1e-12, 0], [0, 1]]), np.array([0.7, 1.1])
    _, _, result = evidence(o, x, lambda value: o.j @ value)
    assert result.status == result.directions[0].status == "numerical_noise"
    assert all(not s.noise_resolved for s in result.directions[0].steps)


def test_weak_joint_direction_is_checked_even_when_each_axis_is_resolved():
    o, x = oracle([[1, 1], [2, 2]]), np.array([0.7, 1.1])
    policy = p.PiecewiseAssessmentPolicy(include_weak_direction=True)
    plan, _, result = evidence(o, x, lambda value: o.j @ value, policy)
    assert all(d.status == "supported" for d in result.directions[:2])
    assert result.directions[-1].direction == "weak_joint"
    assert result.directions[-1].status == result.status == "numerical_noise"
    direction = np.array(plan[-1].direction_vector)
    assert np.linalg.norm(o.j @ (direction * o.units)) < 1e-14
    assert np.linalg.norm(direction) == pytest.approx(1)


@pytest.mark.parametrize("rows", [1, 20])
def test_weak_direction_avoids_quadratic_tall_svd_but_retains_wide_nullspace(
    rows, monkeypatch
):
    o = oracle(np.arange(1, rows * 2 + 1, dtype=float).reshape(rows, 2))
    modes, original = [], np.linalg.svd

    def record(matrix, *, full_matrices):
        modes.append(full_matrices)
        return original(matrix, full_matrices=full_matrices)

    monkeypatch.setattr(p.np.linalg, "svd", record)
    plan = p.perturbation_plan(
        o,
        np.zeros(2),
        o.names,
        p.PiecewiseAssessmentPolicy(include_weak_direction=True),
        jacobian=o.j,
    )
    assert modes == [rows < 2]
    weak = np.asarray(plan[-1].direction_vector)
    assert np.linalg.norm(weak) == pytest.approx(1)
    if rows < 2:
        assert np.linalg.norm((o.j * o.units) @ weak) < 1e-14


def test_solver_discrepancy_prevents_apparent_finite_difference_agreement():
    o, x = oracle(), np.array([0.7, 1.1])
    _, _, result = evidence(
        o,
        x,
        lambda value: o.j @ value,
        alternate=lambda value: o.j @ value + 1e-3,
    )
    assert result.status == "numerical_noise"


def test_missing_alternate_or_failed_point_never_qualifies():
    o, x, policy = oracle(), np.array([0.7, 1.1]), p.PiecewiseAssessmentPolicy()
    _, evaluations, _ = evidence(o, x, lambda value: o.j @ value, policy)
    args = (o, x, o.names, policy, x, o.j, evaluations)
    assert p.assess_derivatives(*args).status == "unavailable"
    evaluations[next(iter(evaluations))] = p.PerturbationEvaluation(
        status="unavailable"
    )
    assert (
        p.assess_derivatives(*args, base_alternate_residual=x).status == "unavailable"
    )


def test_wrong_ad_and_nonconverged_steps_remain_unqualified():
    o, x = oracle([[1.2, 0], [0, 1]]), np.array([0.7, 1.1])
    assert evidence(o, x, lambda value: value)[2].status == "derivative_mismatch"
    o, x = oracle([[0, 0], [0, 1]]), np.array([0.0, 1.0])
    result = evidence(o, x, lambda value: np.array([100 * value[0] ** 3, value[1]]))[2]
    assert result.status == "derivative_mismatch"
    assert not result.directions[0].converged_across_steps


def test_nonfinite_or_incomplete_evidence_rejected_before_support():
    o, x, policy = oracle(), np.array([0.7, 1.1]), p.PiecewiseAssessmentPolicy()
    with pytest.raises(ValueError, match="joint AD"):
        p.assess_derivatives(o, x, o.names, policy, x, np.ones((2, 1)), {})
    with pytest.raises(ValueError, match="nonfinite saved"):
        p.assess_derivatives(o, x, o.names, policy, [np.nan, 1], o.j, {})
    with pytest.raises(ValidationError):
        p.PerturbationEvaluation(residual=(np.nan,))
    with pytest.raises(ValueError, match="saved joint Jacobian"):
        p.perturbation_plan(
            o, x, o.names, p.PiecewiseAssessmentPolicy(include_weak_direction=True)
        )


def branch_oracle(rhs="a - 0.2*max(x-10,0)", observation="x", process=None):
    """Build a tiny genuine integrated model with expanded process references."""
    pytest.importorskip("casadi")
    from autoformalism.data import DatasetSplit, SplitName, Trajectory
    from autoformalism.expressions import ValidationContext, compile_candidate
    from autoformalism.fitting.sensitivity_probe import SymbolicODE
    from autoformalism.schemas import CandidateModel

    payload = {
        "candidate_id": "assessment_control",
        "parent_candidate_id": None,
        "states": [{"name": "x", "kind": "latent"}],
        "state_equations": [{"state": "x", "rhs": rhs}],
        "observation_mappings": [{"channel": "v01", "expression": observation}],
        "parameters": [{"name": "a", "scope": "global", "role": "rate"}],
        "initial_conditions": [{"state": "x", "scope": "global", "fixed_value": 9}],
    }
    if process is not None:
        payload["processes"] = [{"name": "q", "expression": process}]
    model = compile_candidate(
        CandidateModel.model_validate(payload), ValidationContext(targets=("v01",))
    )
    times = np.linspace(0, 3, 16)
    row = Trajectory("training", times, {"v01": np.zeros(len(times))}, {}, {}, {}, {})
    return SimpleNamespace(
        model=model,
        system=SymbolicODE(model, allow_piecewise=True),
        train=DatasetSplit(SplitName.TRAIN, (row,), "synthetic_assessment"),
        names=("a",),
        units=np.ones(1),
        lower=np.array([0.0]),
        upper=np.array([20.0]),
    )


def test_clean_integrated_crossing_supports_only_local_numerics():
    from autoformalism.fitting import FitConfig
    from autoformalism.fitting.sensitivity_probe import symbolic_rollout

    o, x = branch_oracle(), np.array([1.0])
    row = o.train.trajectories[0]
    settings = FitConfig(
        integration_method="DOP853", relative_tolerance=1e-11, absolute_tolerance=1e-13
    )

    def integrate(value, method, sensitivities=False):
        return symbolic_rollout(
            o.system,
            row,
            value,
            settings.model_copy(update={"integration_method": method}),
            None,
            sensitivities=sensitivities,
        )

    base, jacobian, states, _ = integrate(x, "DOP853", True)
    o.j = jacobian.reshape(-1, 1)
    _, evaluations, result = evidence(
        o,
        x,
        lambda value: integrate(value, "DOP853")[0].ravel(),
        alternate=lambda value: integrate(value, "Radau")[0].ravel(),
    )
    assert result.status == "supported"
    summary = p.sampled_branch_summary(
        o, x, {row.trajectory_id: states}, p.PiecewiseAssessmentPolicy()
    )
    assert summary.sampled_activity_changes > 0
    assert summary.near_tie_samples > 0
    assert not summary.between_sample_behavior_certified
    checked = p.assess_derivatives(
        o,
        x,
        o.names,
        p.PiecewiseAssessmentPolicy(),
        base.ravel(),
        o.j,
        evaluations,
        base_alternate_residual=integrate(x, "Radau")[0].ravel(),
        branch_summary=summary,
    )
    assert checked.status == "supported" and not checked.supported_classical


def test_branch_probe_expands_processes_and_retains_every_multiargument_fold():
    o = branch_oracle(rhs="a - q", observation="abs(x-10)", process="max(x-10,0,2)")
    row = o.train.trajectories[0]
    states = np.linspace(9, 13, len(row.time))[:, None]
    result = p.sampled_branch_summary(
        o, np.ones(1), {row.trajectory_id: states}, p.PiecewiseAssessmentPolicy()
    )
    maximum = [b for b in result.branches if b.primitive == "max"]
    assert len(maximum) == 2 and [b.fold for b in maximum] == [0, 1]
    assert all(b.location == "equation:x" for b in maximum)
    assert len([b for b in result.branches if b.primitive == "abs"]) == 1
    assert all(b.samples == len(row.time) for b in result.branches)
    assert result.sampled_activity_changes > 0
    with pytest.raises(ValueError, match="stored branch states"):
        p.sampled_branch_summary(
            o,
            np.ones(1),
            {row.trajectory_id: states[:-1]},
            p.PiecewiseAssessmentPolicy(),
        )


def test_no_sampled_tie_never_certifies_between_samples():
    o = branch_oracle(rhs="a", observation="min(x-10,0,2)")
    row = o.train.trajectories[0]
    states = np.full((len(row.time), 1), 11.0)
    result = p.sampled_branch_summary(
        o, np.ones(1), {row.trajectory_id: states}, p.PiecewiseAssessmentPolicy()
    )
    assert result.near_tie_samples == 0 and result.sampled_activity_changes == 0
    assert not result.between_sample_behavior_certified


def test_branch_probe_includes_causal_initializer_primitives():
    from autoformalism.fitting.initialization import (
        LatentInitializationPlan,
        apply_initialization_plan,
    )
    from autoformalism.fitting.sensitivity_probe import SymbolicODE

    o = branch_oracle(rhs="a", observation="x+1")
    plan = LatentInitializationPlan.model_validate(
        {
            "rules": {
                "x": {
                    "initial": {
                        "mode": "map",
                        "expression": "abs(b)",
                        "parameters": [{"name": "b", "guess": 0}],
                    }
                }
            }
        }
    )
    o.model, _, _ = apply_initialization_plan(o.model, plan)
    o.system = SymbolicODE(o.model, allow_piecewise=True)
    o.names, o.units = o.system.names, np.ones(2)
    o.lower, o.upper = np.full(2, -20.0), np.full(2, 20.0)
    row = o.train.trajectories[0]
    result = p.sampled_branch_summary(
        o,
        np.array([1.0, 0.0]),
        {row.trajectory_id: row.time[:, None]},
        p.PiecewiseAssessmentPolicy(),
    )
    assert len(result.branches) == 1
    assert result.branches[0].location == "initial_condition:x"
    assert result.branches[0].samples == result.near_tie_samples == 1
    with pytest.raises(ValueError, match="every trajectory"):
        p.sampled_branch_summary(o, np.ones(2), {}, p.PiecewiseAssessmentPolicy())
