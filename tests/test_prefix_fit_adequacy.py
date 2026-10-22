"""Reduced-oracle derivative coverage and conservative local adequacy gates."""

from time import monotonic

import numpy as np
import pytest
from pydantic import ValidationError

from autoformalism.fitting import prefix_fit_adequacy as adequacy


def history(loss, *, count=3, step=1e-8):
    return [
        adequacy.HistoryPoint(
            training_nmse=loss,
            coordinate_step_norm=step,
            evaluation_count=i + 1,
        )
        for i in range(count)
    ]


def verified(x, loss):
    return adequacy.OriginalVerification(
        reduced_vector=tuple(x),
        training_nmse=loss,
        alternate_training_nmse=loss,
        solver_loss_discrepancy=0,
        solver_prediction_discrepancy=0,
        original_equations_verified=True,
    )


def run(evaluate, x, *, lower=None, upper=None, units=None, **kwargs):
    x = np.asarray(x, dtype=float)
    lower = np.full_like(x, -np.inf) if lower is None else lower
    upper = np.full_like(x, np.inf) if upper is None else upper
    residual, jacobian = evaluate(x, monotonic() + 5)
    loss = float(np.mean(np.asarray(residual) ** 2))
    check = adequacy.derivative_check(
        evaluate,
        x,
        lower,
        upper,
        monotonic() + 5,
        coordinate_units=units,
    )
    options = {
        "verification": verified(x, loss),
        "derivative": check,
        "stop_reason": "stationary_poor_fit",
        "budget_exhausted": False,
        "history": history(loss),
        "coordinate_units": units,
    } | kwargs
    return adequacy.assess(x, residual, jacobian, lower, upper, **options)


def linear(x, deadline):
    return np.array([x[0], x[0]]), np.ones((2, 1))


def incompatible(x, deadline):
    # The best scalar prediction of opposing measurements is x=0, MSE=1.
    return np.array([x[0] - 1, x[0] + 1]), np.ones((2, 1))


def test_verified_accurate_and_poor_local_solutions_are_distinct():
    good = run(linear, [0], stop_reason="numerical_target_reached")
    poor = run(incompatible, [0])
    assert good.classification == "locally_converged_accurate_fit"
    assert poor.classification == "locally_converged_poor_fit"
    assert poor.parameter_rank == 1 and poor.parameter_information == "full_local_rank"
    assert not poor.global_minimum_certified and not poor.structural_revision_justified


@pytest.mark.parametrize(
    "stop,exhausted",
    [
        ("budget_limited", True),
        ("budget_limited", False),
        ("wall_clock_limit", True),
        ("stationary_poor_fit", None),
        (None, False),
        ("unrecognized", False),
    ],
)
def test_budget_unknown_or_unsupported_stop_never_certifies(stop, exhausted):
    result = run(incompatible, [0], stop_reason=stop, budget_exhausted=exhausted)
    assert result.classification == "optimization_unresolved"


def test_step_only_and_isolated_stationary_start_are_unresolved():
    for endpoint, trace in [
        ([0], []),
        ([0], history(1, count=1)),
        ([0.2], history(1.04)),
    ]:
        result = run(
            incompatible,
            endpoint,
            history=trace,
            stop_reason="step_or_gradient_converged",
        )
        assert result.classification == "optimization_unresolved"
    result = run(incompatible, [0], history=history(1, step=0))
    assert "accepted_progress_history_insufficient" in result.reasons


def test_tiny_jacobian_cannot_make_a_large_descent_correlation_pass():
    def weak(x, deadline):
        return np.array([1 + 1e-10 * x[0]]), np.array([[1e-10]])

    result = run(weak, [0])
    assert result.projected_gradient_inf < 1e-8
    assert result.maximum_residual_column_correlation > 0.99
    assert "residual_column_correlation_not_small" in result.reasons
    assert result.classification == "optimization_unresolved"


@pytest.mark.parametrize("x,lower,upper,sign", [(0, 0, 3, 1), (0, -3, 0, -1)])
def test_active_bound_blocks_only_infeasible_descent(x, lower, upper, sign):
    def bounded(v, deadline):
        return np.array([v[0] + sign]), np.ones((1, 1))

    result = run(bounded, [x], lower=[lower], upper=[upper])
    assert result.classification == "locally_converged_poor_fit"
    assert result.active_bound_indices == (0,)
    assert (
        result.projected_gradient_inf == result.maximum_residual_column_correlation == 0
    )

    def inward(v, deadline):
        return np.array([v[0] - sign]), np.ones((1, 1))

    other = run(inward, [x], lower=[lower], upper=[upper])
    assert other.classification == "optimization_unresolved"
    assert other.projected_gradient_inf > 0


def test_rank_deficiency_is_separate_from_local_fit_adequacy():
    def redundant(x, deadline):
        return np.array([x.sum() - 1, x.sum() + 1]), np.ones((2, 2))

    result = run(redundant, [0, 0])
    assert result.classification == "locally_converged_poor_fit"
    assert result.parameter_information == "rank_deficient"
    assert result.parameter_rank == 1 and result.parameter_condition_number is None


def test_wrong_missing_stale_or_inaccurate_original_verification_withholds():
    v = verified([0], 1)
    for invalid in [
        None,
        v.model_copy(update={"original_equations_verified": False}),
        v.model_copy(update={"reduced_vector": (1,)}),
        v.model_copy(update={"solver_prediction_discrepancy": 1e-3}),
        v.model_copy(update={"alternate_training_nmse": 1.1}),
        verified([0], 2),
    ]:
        result = run(incompatible, [0], verification=invalid)
        assert not result.original_rollout_verified
        assert result.classification == "optimization_unresolved"


def test_audit_must_match_current_vector_residual_and_coordinate_units():
    stale = adequacy.derivative_check(incompatible, [1], [-2], [2], monotonic() + 5)
    result = run(incompatible, [0], derivative=stale)
    assert "reduced_derivative_audit_unqualified_or_mismatched" in result.reasons
    wrong_units = adequacy.derivative_check(
        incompatible, [0], [-2], [2], monotonic() + 5, coordinate_units=[2]
    )
    assert (
        run(incompatible, [0], derivative=wrong_units).classification
        == "optimization_unresolved"
    )


def test_history_must_be_stable_accepted_nonduplicate_and_at_endpoint():
    bad_histories = [
        history(1, step=0.01),
        history(2),
        [p.model_copy(update={"accepted": False}) for p in history(1)],
        history(1) * 2,
    ]
    for trace in bad_histories:
        assert (
            run(incompatible, [0], history=trace).classification
            == "optimization_unresolved"
        )


def test_restarts_cannot_override_bad_local_evidence_or_claim_global_minimum():
    restarts = [
        {
            "start_id": "a",
            "predeclared": True,
            "distinct_from_incumbent": True,
            "status": "complete",
            "verified_training_nmse": 1.0,
        }
    ]
    result = run(incompatible, [0], restarts=restarts)
    assert result.complete_predeclared_restarts == 1
    assert result.classification == "locally_converged_poor_fit"
    assert not result.global_minimum_certified
    unresolved = run(incompatible, [1], restarts=restarts)
    assert unresolved.improving_predeclared_restarts == 1
    assert unresolved.classification == "optimization_unresolved"
    unplanned = [dict(restarts[0], predeclared=False, verified_training_nmse=0.0)]
    assert run(incompatible, [0], restarts=unplanned).complete_predeclared_restarts == 0
    with pytest.raises(ValueError, match="unique"):
        run(incompatible, [0], restarts=restarts * 2)


def test_all_columns_two_scales_actual_nonlinear_reduced_oracle():
    def nonlinear(x, deadline):
        # Example of a reconstructed/profiled coordinate, including chain rule.
        return np.array([np.exp(x[0]) + x[1] ** 2, x[0] * x[1]]), np.array(
            [
                [np.exp(x[0]), 2 * x[1]],
                [x[1], x[0]],
            ]
        )

    result = adequacy.derivative_check(
        nonlinear,
        [0.2, 0.4],
        [-1, -1],
        [1, 1],
        monotonic() + 5,
        coordinate_units=[2, 0.5],
    )
    assert result.status == "passed"
    assert result.calls_started == result.calls_completed == 9
    assert result.checked_indices == (0, 1) and not result.omitted_indices
    assert all(row.scheme == "central" for row in result.directions)


def test_wrong_derivative_does_not_pass_absolute_floor_for_tiny_column():
    def wrong(x, deadline):
        return np.array([1e-12 * x[0]]), np.zeros((1, 1))

    result = adequacy.derivative_check(wrong, [1], [-2], [2], monotonic() + 5)
    assert result.status == "failed"
    assert result.directions[0].relative_errors == pytest.approx((1, 1))


def test_residual_only_probes_preserve_derivative_result_and_accounting():
    result = adequacy.derivative_check(
        incompatible,
        [0],
        [-2],
        [2],
        monotonic() + 5,
        residual_only=lambda x, deadline: incompatible(x, deadline)[0],
    )
    assert result.status == "passed"
    assert result.calls_started == result.calls_completed == 6
    assert result.jacobian_calls_started == 1
    assert result.residual_only_calls_started == 5
    assert len(result.base_residual_sha256) == len(result.base_jacobian_sha256) == 64
    assert result.base_training_nmse == 1
    assert "base_jacobian" not in result.model_dump()
    wrong = adequacy.derivative_check(
        incompatible,
        [0],
        [-2],
        [2],
        monotonic() + 5,
        residual_only=lambda x, deadline: incompatible(x, deadline)[0] + 1,
    )
    assert wrong.status == "incomplete"
    assert "differs" in wrong.error


def test_central_cancellation_does_not_hide_a_kink_but_accepts_smooth_curvature():
    def kink(x, deadline):
        return np.array([abs(x[0])]), np.array([[0.0]])

    def smooth(x, deadline):
        return np.array([x[0] ** 2]), np.array([[2 * x[0]]])

    bad = adequacy.derivative_check(kink, [0], [-1], [1], monotonic() + 5)
    good = adequacy.derivative_check(smooth, [0], [-1], [1], monotonic() + 5)
    assert bad.status == "failed"
    assert bad.directions[0].one_sided_disagreement == pytest.approx((2, 2))
    assert good.status == "passed"


def test_looser_audit_policy_cannot_qualify_stricter_assessment():
    check = adequacy.derivative_check(
        incompatible,
        [0],
        [-1],
        [1],
        monotonic() + 5,
        policy=adequacy.Policy(derivative_relative_tolerance=0.01),
    )
    assert check.status == "passed"
    assert (
        run(incompatible, [0], derivative=check).classification
        == "optimization_unresolved"
    )


def test_one_sided_bound_probe_never_leaves_domain():
    def bounded(x, deadline):
        assert 0 <= x[0] <= 1
        return np.array([x[0]]), np.ones((1, 1))

    for point, scheme in [(0, "forward"), (1, "backward")]:
        result = adequacy.derivative_check(bounded, [point], [0], [1], monotonic() + 5)
        assert result.status == "passed"
        assert result.directions[0].scheme == scheme
        assert result.calls_started == result.calls_completed == 3


def test_capped_or_selected_direction_coverage_is_incomplete():
    def two(x, deadline):
        return x.copy(), np.eye(2)

    for kwargs in [{"directions": [1]}, {"policy": adequacy.Policy(max_directions=1)}]:
        result = adequacy.derivative_check(
            two, [0, 0], [-1, -1], [1, 1], monotonic() + 5, **kwargs
        )
        assert result.status == "incomplete"
        assert len(result.omitted_indices) == 1
        assert result.calls_started == 5


def test_deadline_and_failed_calls_are_counted(monkeypatch):
    expired = adequacy.derivative_check(linear, [0], [-1], [1], monotonic() - 1)
    assert expired.status == "unavailable" and expired.calls_started == 0
    calls = 0

    def fails(x, deadline):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("probe unavailable")
        return linear(x, deadline)

    result = adequacy.derivative_check(fails, [0], [-1], [1], monotonic() + 5)
    assert result.status == "incomplete"
    assert result.calls_started == 2 and result.calls_completed == 1
    assert result.omitted_indices == (0,)
    clock = [0.0]
    monkeypatch.setattr(adequacy, "monotonic", lambda: clock[0])

    def late(x, deadline):
        clock[0] = 2
        return linear(x, deadline)

    result = adequacy.derivative_check(late, [0], [-1], [1], 1)
    assert result.status == "unavailable"
    assert result.calls_started == result.calls_completed == 1
    assert "deadline" in result.error


def test_fixed_problem_has_no_search_dimensions_but_requires_verification():
    def fixed(x, deadline):
        return np.ones(2), np.zeros((2, 1))

    result = run(
        fixed, [1], lower=[1], upper=[1], history=[], stop_reason="no_free_coordinates"
    )
    assert result.classification == "locally_converged_poor_fit"
    assert result.parameter_information == "no_free_coordinates"
    assert not result.global_minimum_certified


@pytest.mark.parametrize(
    "kwargs",
    [
        {"gradient_tolerance": float("nan")},
        {"accuracy_nmse": float("inf")},
        {"derivative_absolute_step": 0},
        {"max_directions": 0},
        {"history_window": 1},
    ],
)
def test_policy_rejects_nonfinite_or_invalid_thresholds(kwargs):
    with pytest.raises(ValidationError):
        adequacy.Policy(**kwargs)


def test_invalid_arrays_and_directions_are_rejected():
    for x, low, high in [
        ([2], [0], [1]),
        ([float("nan")], [0], [1]),
        ([0], [1], [0]),
        ([0], [], [1]),
    ]:
        with pytest.raises(ValueError):
            adequacy.derivative_check(linear, x, low, high, monotonic() + 5)
    with pytest.raises(ValueError, match="distinct"):
        adequacy.derivative_check(
            linear, [0], [-1], [1], monotonic() + 5, directions=[0, 0]
        )


def test_scoped_affine_oracle_audit_covers_shared_maps_and_per_row_roots():
    pytest.importorskip("casadi")
    from autoformalism.fitting.prefix_affine import TrainingOracle
    from autoformalism.fitting.prefix_initialization import NumericalPolicy
    from tests.test_prefix_initialization import problem, training

    oracle = TrainingOracle(problem(), training(), NumericalPolicy())
    x = oracle.vector(oracle.start)
    result = adequacy.derivative_check(
        oracle.evaluate,
        x,
        oracle.lower,
        oracle.upper,
        monotonic() + 10,
        coordinate_units=oracle.units,
        residual_only=lambda v, deadline: oracle.evaluate(v, deadline, jacobian=False)[
            0
        ],
    )
    assert result.status == "passed"
    assert len(result.checked_indices) == len(oracle.names) == 4
    assert result.jacobian_calls_started == 1
    assert result.residual_only_calls_started == 17


def precision_check(evaluate, x=(0,), lower=(-2,), upper=(2,), **kwargs):
    """Run the predeclared precision audit without changing legacy test helpers."""
    return adequacy.derivative_check(
        evaluate,
        x,
        lower,
        upper,
        monotonic() + 5,
        policy=adequacy.Policy(precision_aware=True),
        **kwargs,
    )


def test_precision_opt_in_preserves_legacy_serialization_and_probe_counts():
    policy = adequacy.Policy()
    assert not any(name.startswith("precision_") for name in policy.model_dump())
    assert adequacy.Policy.model_validate(policy.model_dump()) == policy
    check = adequacy.derivative_check(linear, [0], [-2], [2], monotonic() + 5)
    assert check.status == "passed" and check.calls_started == 5
    assert not check.directions[0].precision_probes
    precision = precision_check(linear)
    assert precision.status == "passed" and precision.search_qualified
    assert precision.calls_started == precision.calls_completed == 13
    assert [probe.multiplier for probe in precision.directions[0].precision_probes] == [
        1,
        0.5,
        16,
        8,
        1024,
        512,
    ]
    assert (
        adequacy.Policy.model_validate_json(precision.policy.model_dump_json())
        == precision.policy
    )


def test_unresolved_nearzero_column_qualifies_search_without_parameter_confidence():
    def tiny(x, deadline):
        return np.array([1 + 1e-14 * x[0]]), np.array([[1e-14]])

    legacy = adequacy.derivative_check(tiny, [0], [-2], [2], monotonic() + 5)
    check = precision_check(tiny)
    assert legacy.status == "failed"
    assert check.status == "inconclusive" and check.search_qualified
    assert check.inconclusive_indices == (0,) and check.failed_indices == ()
    assert not check.parameter_information_qualified
    assert 0 < check.gradient_error_bound_inf < 1e-9
    assert check.directions[0].reference_probe_index == 5
    result = run(
        tiny, [0], lower=[-2], upper=[2], derivative=check, policy=check.policy
    )
    assert result.classification == "locally_converged_poor_fit"
    assert result.parameter_information == "derivative_precision_unresolved"
    assert not result.parameter_information_qualified
    assert result.parameter_condition_number is None
    assert result.maximum_residual_column_correlation == pytest.approx(1)
    assert result.correlation_unresolved_indices == (0,)
    assert result.projected_gradient_upper_bound < check.policy.gradient_tolerance
    # The certificate is specific to this opt-in and exact declared domain.
    assert run(tiny, [0], derivative=check).classification == "optimization_unresolved"
    mismatched = run(tiny, [0], derivative=check, policy=check.policy)
    assert "reduced_derivative_audit_unqualified_or_mismatched" in mismatched.reasons


@pytest.mark.parametrize("slope,offset", [(1e-12, 0), (1e-9, 1), (1, 1)])
def test_precision_policy_rejects_resolvable_wrong_derivatives(slope, offset):
    def wrong(x, deadline):
        return np.array([offset + slope * x[0]]), np.zeros((1, 1))

    result = precision_check(wrong, x=(1,))
    assert result.status == "failed" and not result.search_qualified
    assert result.failed_indices == (0,) and not result.inconclusive_indices
    assert result.directions[0].precision_status == "failed"


def test_precision_tiny_kink_cannot_hide_behind_absolute_or_gradient_floor():
    def kink(x, deadline):
        return np.array([1e-12 * abs(x[0])]), np.zeros((1, 1))

    result = precision_check(kink)
    assert result.status == "failed" and not result.search_qualified
    assert result.gradient_error_bound_inf == 0


def test_unresolved_column_must_bound_mean_loss_gradient_in_fixed_units():
    def tiny(x, deadline):
        return np.array([1 + 1e-14 * x[0]]), np.array([[1e-14]])

    ordinary = precision_check(tiny)

    # Change physical coordinates consistently while preserving dimensionless
    # fixed optimizer units; the uncertainty bound must remain the same.
    def rescaled(x, deadline):
        return np.array([1 + 1e-14 * x[0] / 10]), np.array([[1e-15]])

    rescaled_check = precision_check(
        rescaled, lower=(-20,), upper=(20,), coordinate_units=[10]
    )
    assert rescaled_check.gradient_error_bound_inf == pytest.approx(
        ordinary.gradient_error_bound_inf, rel=1e-8
    )

    def huge_residual(x, deadline):
        return np.array([1e6 + 1e-14 * x[0]]), np.array([[1e-14]])

    huge = precision_check(huge_residual)
    assert huge.status == "inconclusive" and not huge.search_qualified
    assert huge.gradient_error_bound_inf > 1e-8


@pytest.mark.parametrize("point,scheme", [(0, "forward"), (1, "backward")])
def test_precision_bound_probes_are_inward_and_gradient_uncertainty_is_projected(
    point,
    scheme,
):
    sign = 1 if point == 0 else -1

    def bounded(x, deadline):
        assert 0 <= x[0] <= 1
        return np.array([1 + sign * 1e-14 * x[0]]), np.array([[sign * 1e-14]])

    check = precision_check(bounded, x=(point,), lower=(0,), upper=(1,))
    assert check.status == "inconclusive" and check.search_qualified
    assert check.directions[0].scheme == scheme
    assert check.calls_started == check.calls_completed == 7
    result = run(
        bounded, [point], lower=[0], upper=[1], derivative=check, policy=check.policy
    )
    assert result.projected_gradient_inf == 0
    # Nominal outward descent does not discard the possible inward gradient.
    assert 0 < result.projected_gradient_upper_bound < 1e-8


def test_precision_missing_coverage_never_qualifies_and_calls_remain_accounted():
    def two(x, deadline):
        return np.ones(2) + x * 1e-14, np.eye(2) * 1e-14

    check = precision_check(
        two,
        x=(0, 0),
        lower=(-2, -2),
        upper=(2, 2),
        directions=[0],
        residual_only=lambda x, deadline: two(x, deadline)[0],
    )
    assert check.status == "incomplete" and not check.search_qualified
    assert check.inconclusive_indices == (0,) and check.omitted_indices == (1,)
    assert check.jacobian_calls_started == 1
    assert check.residual_only_calls_started == 13
    assert check.gradient_error_bound_inf is None


def test_precision_tiny_wrong_sign_is_a_failure_not_negligible_permission():
    def wrong_sign(x, deadline):
        return np.array([1e-12 * x[0]]), np.array([[-1e-12]])

    check = precision_check(wrong_sign, x=(1,))
    assert check.status == "failed" and not check.search_qualified
    assert check.gradient_error_bound_inf < 1e-9


def test_precision_unrepresentable_probe_fails_closed_and_narrow_domain_is_respected():
    point = 1.0
    upper = np.nextafter(point, np.inf)
    unavailable = precision_check(linear, x=(point,), lower=(point,), upper=(upper,))
    assert unavailable.status == "incomplete" and not unavailable.search_qualified
    assert unavailable.calls_started == unavailable.calls_completed == 1
    assert "not representable" in unavailable.error
    assert unavailable.omitted_indices == (0,)

    def narrow(x, deadline):
        assert 1 <= x[0] <= 1 + 1e-8
        return x.copy(), np.ones((1, 1))

    check = precision_check(narrow, x=(1,), lower=(1,), upper=(1 + 1e-8,))
    assert check.status == "passed" and check.search_qualified
    assert check.directions[0].scheme == "forward"
    assert len(check.directions[0].precision_probes) == 6


def test_precision_assessment_binds_frozen_envelope_policy():
    check = precision_check(incompatible)
    changed = adequacy.Policy(precision_aware=True, precision_roundoff_factor=64)
    result = run(
        incompatible, [0], lower=[-2], upper=[2], derivative=check, policy=changed
    )
    assert result.classification == "optimization_unresolved"
    assert "reduced_derivative_audit_unqualified_or_mismatched" in result.reasons
