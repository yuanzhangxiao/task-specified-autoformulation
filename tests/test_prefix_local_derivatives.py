"""Bounded local derivative probes preserve evidence without wide excursions."""

from itertools import pairwise
from time import monotonic

import numpy as np
import pytest
from pydantic import ValidationError

from autoformalism.fitting import prefix_fit_adequacy as adequacy
from autoformalism.fitting import prefix_local_derivatives as local
from autoformalism.fitting.recovery_numerics import NumericalInvalidTrial


def check(evaluate, x=(0.0,), lower=(-2.0,), upper=(2.0,), **kwargs):
    """Use the explicit new policy without affecting historical test helpers."""
    return adequacy.derivative_check(
        evaluate,
        x,
        lower,
        upper,
        kwargs.pop("deadline", monotonic() + 5),
        policy=kwargs.pop(
            "policy", adequacy.Policy(precision_aware=True, local_adaptive=True)
        ),
        **kwargs,
    )


def test_local_policy_is_opt_in_and_preserves_old_policy_payloads():
    for precision in (False, True):
        policy = adequacy.Policy(precision_aware=precision)
        assert not any(k.startswith("local_") for k in policy.model_dump())
        assert adequacy.Policy.model_validate_json(policy.model_dump_json()) == policy
    with pytest.raises(ValidationError, match="require precision_aware"):
        adequacy.Policy(local_adaptive=True)
    policy = adequacy.Policy(precision_aware=True, local_adaptive=True)
    assert adequacy.Policy.model_validate_json(policy.model_dump_json()) == policy


def test_smooth_pair_finishes_without_six_scale_probes():
    points = []

    def smooth(x, deadline):
        points.append(x.copy())
        return np.array([np.exp(x[0]), x[0] ** 2]), np.array(
            [[np.exp(x[0])], [2 * x[0]]]
        )

    result = check(smooth, (0.5,))
    assert result.status == "passed" and result.search_qualified
    assert result.parameter_information_qualified
    assert result.calls_started == result.calls_completed == 5
    assert len(result.local_probe_attempts) == 2
    assert [p.multiplier for p in result.directions[0].precision_probes] == [1, 0.5]
    assert max(abs(p[0] - 0.5) for p in points) <= 0.5e-4 * (1 + 1e-10)


def test_small_time_constant_uses_local_magnitude_not_large_optimizer_units():
    tau, points = 0.0015, []

    def stiff(x, deadline):
        points.append(float(x[0]))
        value = np.exp(-tau / x[0])
        return np.array([value]), np.array([[value * tau / x[0] ** 2]])

    result = check(stiff, (tau,), lower=(1e-10,), upper=(1e3,), coordinate_units=[100])
    assert result.status == "passed"
    assert result.directions[0].scheme == "central"
    assert max(abs(v - tau) for v in points) <= tau * 1.000001e-4
    assert all(v > 0 for v in points)


def test_curvature_causes_shrinking_until_a_local_pair_agrees():
    def curve(x, deadline):
        return np.array([np.sin(1e4 * x[0])]), np.array([[1e4 * np.cos(1e4 * x[0])]])

    result = check(curve)
    assert result.status == "passed"
    assert len(result.local_probe_attempts) > 2
    steps = [p.step for p in result.local_probe_attempts]
    assert all(a / 2 == b for a, b in pairwise(steps))
    assert result.calls_started <= 1 + 2 * result.policy.local_max_scales


@pytest.mark.parametrize("slope,offset", [(1, 1), (1e-12, 0)])
def test_stable_wrong_derivative_fails_without_negligible_gradient_escape(
    slope, offset
):
    def wrong(x, deadline):
        return np.array([offset + slope * x[0]]), np.array([[-slope]])

    result = check(wrong)
    assert result.status == "failed" and not result.search_qualified
    assert result.failed_indices == (0,)
    assert result.calls_started == 5


@pytest.mark.parametrize("amplitude", [1.0, 1e-12])
def test_actual_kink_withholds_even_if_analytic_and_central_derivative_are_zero(
    amplitude,
):
    def kink(x, deadline):
        return np.array([amplitude * abs(x[0])]), np.zeros((1, 1))

    result = check(kink)
    assert result.status == "failed" and not result.search_qualified
    assert result.directions[0].one_sided_disagreement[1] > 0
    assert len(result.local_probe_attempts) == result.policy.local_max_scales


@pytest.mark.parametrize("point,scheme", [(0, "forward"), (1, "backward")])
def test_one_sided_probes_respect_bounds_and_have_two_local_sizes(point, scheme):
    points = []

    def bounded(x, deadline):
        points.append(float(x[0]))
        assert 0 <= x[0] <= 1
        return np.array([np.exp(x[0])]), np.array([[np.exp(x[0])]])

    result = check(bounded, (point,), lower=(0,), upper=(1,))
    assert result.status == "passed"
    assert result.directions[0].scheme == scheme
    assert result.calls_started == result.calls_completed == 3
    assert max(abs(v - point) for v in points) <= 1.00001e-4


@pytest.mark.parametrize("offset", [0.0, 1.0])
def test_exact_zero_direction_is_search_only_when_gradient_envelope_is_small(offset):
    def constant(x, deadline):
        return np.array([offset]), np.zeros((1, 1))

    result = check(constant)
    assert result.status == "inconclusive" and result.search_qualified
    assert result.inconclusive_indices == (0,)
    assert not result.parameter_information_qualified
    assert result.gradient_error_bound_inf <= 1e-9


def test_tiny_unresolved_column_large_residual_cannot_pass_search():
    def tiny(x, deadline):
        return np.array([1e6 + 1e-14 * x[0]]), np.array([[1e-14]])

    result = check(tiny)
    assert result.status == "inconclusive" and not result.search_qualified
    assert result.gradient_error_bound_inf > 1e-8
    assert result.calls_started == 5  # Smaller probes would amplify rounding.


def test_nonfinite_initial_probe_shrinks_and_preserves_error_receipt():
    def narrow(x, deadline):
        if abs(x[0]) > 6e-5:
            raise NumericalInvalidTrial("transition overflow")
        return x.copy(), np.ones((1, 1))

    result = check(narrow)
    assert result.status == "passed"
    assert result.local_probe_attempts[0].status == "numerical_failure"
    assert "transition overflow" in result.local_probe_attempts[0].error
    assert [p.step for p in result.local_probe_attempts] == [1e-4, 5e-5, 2.5e-5]
    assert result.error is None
    assert result.calls_started == 6 and result.calls_completed == 5


def test_no_large_probe_is_attempted_after_valid_local_pair():
    points = []

    def guarded(x, deadline):
        points.append(float(x[0]))
        if abs(x[0]) > 0.001:
            raise NumericalInvalidTrial("wide probe is not local")
        return np.array([np.sin(x[0])]), np.array([[np.cos(x[0])]])

    result = check(guarded)
    assert result.status == "passed"
    assert max(abs(x) for x in points) <= 1e-4
    assert all(p.status == "complete" for p in result.local_probe_attempts)


@pytest.mark.parametrize("failure", ["shape", "changed_shape", "programming"])
def test_only_numerical_errors_are_shrunk_contract_errors_abort(failure):
    def broken(x, deadline):
        if x[0] != 0:
            if failure == "shape":
                return np.empty(0), np.empty((0, 1))
            if failure == "changed_shape":
                return np.ones(2), np.ones((2, 1))
            raise ValueError("undeclared model input")
        return np.ones(1), np.ones((1, 1))

    result = check(broken)
    assert result.status == "incomplete" and not result.search_qualified
    assert result.calls_started == 2
    assert result.local_probe_attempts[0].status == "unavailable"
    assert result.omitted_indices == (0,)


def test_partial_timeout_preserves_completed_direction_and_failed_probe(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(local, "monotonic", lambda: clock[0])

    def slow(x, deadline):
        clock[0] += 1
        return x.copy(), np.eye(2)

    result = check(slow, (0, 0), lower=(-1, -1), upper=(1, 1), deadline=6.5)
    assert result.status == "incomplete" and not result.search_qualified
    assert result.checked_indices == (0,) and result.omitted_indices == (1,)
    assert result.directions[0].passed
    assert result.local_probe_attempts[-1].status == "deadline"
    assert result.calls_started == result.calls_completed == 7
    assert result.elapsed_seconds == 7


def test_partial_direction_retains_finite_probe_before_timeout(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(local, "monotonic", lambda: clock[0])

    def slow(x, deadline):
        clock[0] += 1
        return x.copy(), np.ones((1, 1))

    result = check(slow, deadline=4.5)
    assert result.status == "incomplete" and not result.search_qualified
    assert result.checked_indices == ()
    assert [p.status for p in result.local_probe_attempts] == ["complete", "deadline"]


def test_residual_only_callback_avoids_jacobian_probes_and_binds_base():
    def identity(x, deadline):
        return x.copy(), np.eye(len(x))

    result = check(identity, residual_only=lambda x, d: x.copy())
    assert result.status == "passed"
    assert result.jacobian_calls_started == 1
    assert result.residual_only_calls_started == 5
    bad = check(identity, residual_only=lambda x, d: x + 1)
    assert bad.status == "incomplete" and not bad.search_qualified
    assert "differs" in bad.error


def test_selected_or_capped_directions_never_qualify():
    result = check(
        lambda x, d: (x.copy(), np.eye(2)),
        (0, 0),
        lower=(-1, -1),
        upper=(1, 1),
        directions=[0],
    )
    assert result.status == "incomplete" and not result.search_qualified
    assert result.omitted_indices == (1,)


def test_unrepresentable_local_step_cannot_be_mistaken_for_zero_sensitivity():
    result = check(
        lambda x, d: (x.copy(), np.ones((1, 1))),
        (1,),
        lower=(1,),
        upper=(np.nextafter(1.0, np.inf),),
    )
    assert result.status == "incomplete" and not result.search_qualified
    assert result.checked_indices == ()
    assert "not representable" in result.error


def test_old_audit_serialization_does_not_gain_new_fields():
    def identity(x, deadline):
        return x.copy(), np.ones((1, 1))

    for precision in (False, True):
        result = adequacy.derivative_check(
            identity,
            [0],
            [-1],
            [1],
            monotonic() + 5,
            policy=adequacy.Policy(precision_aware=precision),
        )
        assert "local_probe_attempts" not in result.model_dump()
        assert (
            adequacy.DerivativeCheck.model_validate_json(result.model_dump_json())
            == result
        )


def test_opt_in_assessment_binds_exact_local_policy_and_zero_column_is_not_identified():
    from tests.test_prefix_fit_adequacy import history, verified

    def constant(x, deadline):
        return np.ones(1), np.zeros((1, 1))

    result = check(constant)
    options = {
        "verification": verified([0], 1),
        "derivative": result,
        "stop_reason": "gradient_converged",
        "budget_exhausted": False,
        "history": history(1),
        "policy": result.policy,
    }
    assessment = adequacy.assess([0], [1], [[0]], [-2], [2], **options)
    assert assessment.classification == "locally_converged_poor_fit"
    assert assessment.parameter_information == "derivative_precision_unresolved"
    assert not assessment.parameter_information_qualified
    options["policy"] = adequacy.Policy(precision_aware=True)
    stale = adequacy.assess([0], [1], [[0]], [-2], [2], **options)
    assert stale.classification == "optimization_unresolved"


def test_roundoff_bound_is_invariant_under_consistent_coordinate_scaling():
    def tiny(x, deadline):
        return np.array([1 + 1e-14 * x[0]]), np.array([[1e-14]])

    def scaled(x, deadline):
        return np.array([1 + 1e-14 * x[0] / 10]), np.array([[1e-15]])

    first = check(tiny)
    second = check(scaled, lower=(-20,), upper=(20,), coordinate_units=[10])
    assert first.gradient_error_bound_inf == pytest.approx(
        second.gradient_error_bound_inf
    )
    assert first.status == second.status == "inconclusive"


def test_shape_valid_nonfinite_probe_is_typed_numerical_failure():
    def overflow(x, deadline):
        if abs(x[0]) > 6e-5:
            return np.array([np.inf]), np.ones((1, 1))
        return x.copy(), np.ones((1, 1))

    result = check(overflow)
    assert result.status == "passed"
    assert result.local_probe_attempts[0].status == "numerical_failure"
    assert result.calls_started == result.calls_completed + 1


def test_missing_finite_pair_records_failures_without_inventing_a_zero_column():
    def impossible(x, deadline):
        if x[0] != 0:
            raise NumericalInvalidTrial("unsupported transition")
        return np.ones(1), np.zeros((1, 1))

    result = check(impossible)
    assert result.status == "incomplete"
    assert not result.search_qualified
    assert not result.parameter_information_qualified
    assert result.omitted_indices == (0,)
    assert len(result.local_probe_attempts) == result.policy.local_max_scales
    assert all(p.status == "numerical_failure" for p in result.local_probe_attempts)
    assert result.calls_started == 1 + result.policy.local_max_scales


def test_later_numerical_probe_failure_does_not_discard_completed_pair():
    def oscillatory(x, deadline):
        if 1e-5 < abs(x[0]) < 3e-5:
            raise NumericalInvalidTrial("a finite-precision transition failed")
        return np.array([np.sin(1e4 * x[0])]), np.array([[1e4 * np.cos(1e4 * x[0])]])

    result = check(oscillatory)
    assert any(p.status == "numerical_failure" for p in result.local_probe_attempts)
    assert result.checked_indices == (0,)
    assert result.directions[0].precision_probes
    assert result.status == "passed"


def test_remaining_direction_reserve_uses_probe_cost_not_initial_sensitivity_cost(
    monkeypatch,
):
    clock = [0.0]
    monkeypatch.setattr(local, "monotonic", lambda: clock[0])

    def residual(x):
        return np.array([np.sin(1e4 * x[0]), x[1]])

    def sensitivity(x, deadline):
        clock[0] += 100
        return residual(x), np.diag([1e4 * np.cos(1e4 * x[0]), 1.0])

    def cheap(x, deadline):
        clock[0] += 1
        return residual(x)

    result = check(
        sensitivity,
        (0, 0),
        lower=(-1, -1),
        upper=(1, 1),
        residual_only=cheap,
        deadline=130,
    )
    assert result.status == "passed"
    assert result.checked_indices == (0, 1)
    assert len(result.directions[0].precision_probes) > 2
    assert result.jacobian_calls_started == 1
    assert result.elapsed_seconds < 130
