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


def bounds_check(evaluate, **kwargs):
    """Use the separate bounds-aware policy with explicitly declared radii."""
    return check(
        evaluate,
        policy=adequacy.Policy(
            precision_aware=True, local_adaptive=True, local_bounds_aware=True
        ),
        **kwargs,
    )


def test_bounds_aware_policy_preserves_earlier_local_payload_and_requires_opt_in():
    old = adequacy.Policy(precision_aware=True, local_adaptive=True)
    assert "local_bounds_aware" not in old.model_dump()
    with pytest.raises(ValidationError, match="require local_adaptive"):
        adequacy.Policy(precision_aware=True, local_bounds_aware=True)
    with pytest.raises(ValueError, match="require local_bounds_aware"):
        check(lambda x, d: (x, np.eye(1)), local_coordinate_scales=[1])
    result = check(lambda x, d: (x, np.eye(1)))
    assert "local_coordinate_scales" not in result.model_dump()


@pytest.mark.parametrize(
    "value,scheme", [(0.0, "forward"), (1e-16, "forward"), (1 - 1e-16, "backward")]
)
def test_meaningful_inward_local_probe_resolves_near_bound_gain(value, scheme):
    points = []

    def gain(x, deadline):
        points.append(float(x[0]))
        assert 0 <= x[0] <= 1
        return np.array([1 + 0.2 * x[0]]), np.array([[0.2]])

    result = bounds_check(
        gain, x=(value,), lower=(0,), upper=(1,), local_coordinate_scales=[1]
    )
    assert result.status == "passed"
    assert result.directions[0].scheme == scheme
    assert result.directions[0].steps[0] == 5e-5
    assert max(abs(p - value) for p in points) <= 1.00001e-4
    assert result.calls_started == result.calls_completed == 5
    assert result.local_coordinate_scales == (1,)
    assert (
        adequacy.DerivativeCheck.model_validate_json(result.model_dump_json()) == result
    )


def test_inward_second_order_stencil_resolves_curved_equation_at_bound():
    def quadratic(x, deadline):
        return np.array([1 + x[0] + 1e3 * x[0] ** 2]), np.array([[1 + 2e3 * x[0]]])

    result = bounds_check(
        quadratic, x=(1e-16,), lower=(0,), upper=(1,), local_coordinate_scales=[1]
    )
    assert result.status == "passed" and result.calls_started == 5
    assert result.directions[0].relative_errors[-1] < 1e-8


def test_small_nonlinear_time_constant_keeps_local_probes_in_mixed_coordinates():
    tau, points = 1.5e-6, []

    def mixed(x, deadline):
        points.append(x.copy())
        value = np.exp(-tau / x[0])
        return np.array([value, 1 + x[1]]), np.diag([value * tau / x[0] ** 2, 1])

    result = bounds_check(
        mixed,
        x=(tau, 1e-16),
        lower=(1e-12, 0),
        upper=(1, 10),
        coordinate_units=[1, 1],
        local_coordinate_scales=[tau, 1],
    )
    assert result.status == "passed" and result.checked_indices == (0, 1)
    assert result.directions[0].scheme == "central"
    assert result.directions[1].scheme == "forward"
    assert max(abs(p[0] - tau) for p in points) <= tau * 1.00001e-4
    assert max(abs(p[1] - 1e-16) for p in points) <= 1.00001e-4


def test_bounds_aware_without_declared_scale_remains_conservative_for_small_tau():
    tau, points = 1.5e-6, []

    def stiff(x, deadline):
        points.append(x[0])
        r = np.exp(-tau / x[0])
        return np.array([r]), np.array([[r * tau / x[0] ** 2]])

    result = bounds_check(stiff, x=(tau,), lower=(1e-12,), upper=(1,))
    assert result.status == "passed" and result.local_coordinate_scales == (tau,)
    assert max(abs(p - tau) for p in points) <= tau * 1.00001e-4


@pytest.mark.parametrize("scales", [[0], [-1], [np.nan], [np.inf], [1, 2], [[1]]])
def test_invalid_semantic_scales_are_not_silently_ignored(scales):
    with pytest.raises(ValueError, match="finite positive full vector"):
        bounds_check(lambda x, d: (x, np.eye(1)), local_coordinate_scales=scales)


def test_bound_scaled_wrong_derivative_is_rejected():
    result = bounds_check(
        lambda x, d: (np.array([1 + x[0]]), np.array([[-1.0]])),
        x=(1e-16,),
        lower=(0,),
        upper=(1,),
        local_coordinate_scales=[1],
    )
    assert result.status == "failed" and not result.search_qualified
    assert result.failed_indices == (0,)


def test_bound_scaled_kink_still_withholds_information_and_search():
    result = bounds_check(
        lambda x, d: (np.array([abs(x[0])]), np.zeros((1, 1))),
        local_coordinate_scales=[1],
    )
    assert result.status == "failed" and not result.search_qualified
    assert not result.parameter_information_qualified


def test_bound_scaled_exact_zero_is_not_claimed_identifiable():
    result = bounds_check(
        lambda x, d: (np.ones(1), np.zeros((1, 1))),
        x=(1e-16,),
        lower=(0,),
        upper=(1,),
        local_coordinate_scales=[1],
    )
    assert (
        result.status == "inconclusive" and not result.parameter_information_qualified
    )
    # The empirical rounding envelope still applies despite meaningful steps.
    assert result.directions[0].gradient_error_bound > 0


def test_bound_scaled_stencil_timeout_preserves_finite_completed_columns(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(local, "monotonic", lambda: clock[0])

    def slow(x, deadline):
        clock[0] += 1
        return x.copy(), np.eye(2)

    result = bounds_check(
        slow,
        x=(1e-16, 1e-16),
        lower=(0, 0),
        upper=(1, 1),
        local_coordinate_scales=[1, 1],
        deadline=6.5,
    )
    assert result.status == "incomplete" and result.checked_indices == (0,)
    assert result.omitted_indices == (1,) and not result.search_qualified
    assert result.local_probe_attempts[-1].status == "deadline"


def test_second_order_inward_probe_failure_shrinks_without_budget_reset():
    def narrow(x, deadline):
        if x[0] > 6e-5:
            raise NumericalInvalidTrial("transition unavailable")
        return x.copy(), np.eye(1)

    result = bounds_check(
        narrow, x=(1e-16,), lower=(0,), upper=(1,), local_coordinate_scales=[1]
    )
    assert result.status == "passed"
    assert result.local_probe_attempts[0].status == "numerical_failure"
    assert result.calls_started == result.calls_completed + 1
    assert result.calls_started == 7


def test_smaller_noisier_pair_does_not_replace_more_resolvable_local_evidence():
    def cancellation(x, deadline):
        value = x[0]
        return np.array(
            [4e6 + 1e-3 * value + 1e3 * value**3 - 1e13 * value**5]
        ), np.array([[1e-3 + 3e3 * value**2 - 5e13 * value**4]])

    result = bounds_check(cancellation, local_coordinate_scales=[1])
    row = result.directions[0]
    assert len(row.precision_probes) == 3
    assert row.reference_probe_index == 1
    assert row.steps == (1e-4, 5e-5)
    # The finest pair has closer analytic agreement, but twice the rounding
    # envelope. Preserve numerical evidence, rather than cherry-picking agreement.
    assert row.precision_probes[-1].error_rms < row.precision_probes[1].error_rms
    assert row.precision_probes[-1].roundoff_rms > row.precision_probes[1].roundoff_rms
    assert result.status == "inconclusive" and not result.search_qualified


def test_nonlinear_inward_stencil_must_shrink_before_it_qualifies():
    points = []

    def steep(x, deadline):
        points.append(x[0])
        result = np.exp(1e4 * x[0])
        return np.array([result]), np.array([[1e4 * result]])

    result = bounds_check(
        steep, x=(1e-16,), lower=(0,), upper=(1,), local_coordinate_scales=[1]
    )
    assert result.status == "passed"
    assert len(result.local_probe_attempts) > 2
    assert all(0 <= x <= 1.00001e-4 for x in points)
    assert result.calls_started <= 1 + 2 * result.policy.local_max_scales


def test_large_physical_scale_is_relative_and_nonlinear_curvature_still_shrinks():
    value, points = 3e5, []

    def nonlinear(x, deadline):
        points.append(float(x[0]))
        result = np.exp(1e4 * (x[0] / value - 1))
        return np.array([result]), np.array([[result * 1e4 / value]])

    result = bounds_check(
        nonlinear,
        x=(value,),
        lower=(0,),
        upper=(1e6,),
        coordinate_units=[1.5],
        local_coordinate_scales=[value],
    )
    assert result.status == "passed"
    probes = result.local_probe_attempts
    assert probes[0].step == value * 1e-4
    assert len(probes) > 2
    assert all(a.step / 2 == b.step for a, b in pairwise(probes))
    assert max(abs(p - value) for p in points) <= value * 1.00001e-4
    assert result.directions[0].steps[0] < probes[0].step


def test_physical_scale_never_overrides_finite_declared_domain_width():
    points = []
    value = 3e5

    def quadratic(x, deadline):
        points.append(float(x[0]))
        return np.array([(x[0] - value) ** 2 + x[0] - value]), np.array(
            [[2 * (x[0] - value) + 1]]
        )

    result = bounds_check(
        quadratic,
        x=(value,),
        lower=(value,),
        upper=(value + 1,),
        coordinate_units=[1.5],
        local_coordinate_scales=[value],
    )
    assert result.status == "passed"
    assert max(p - value for p in points) <= 1.00001e-4


def test_old_local_mode_keeps_optimizer_unit_cap_for_large_coordinates():
    value, points = 3e5, []

    def affine(x, deadline):
        points.append(float(x[0]))
        return np.array([x[0] / value]), np.array([[1 / value]])

    result = check(affine, x=(value,), lower=(0,), upper=(1e6,), coordinate_units=[1.5])
    assert result.local_probe_attempts[0].step == pytest.approx(1.5e-4)
    assert max(abs(p - value) for p in points) <= 1.500001e-4
    assert "local_coordinate_scales" not in result.model_dump()
