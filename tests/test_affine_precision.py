"""Bounded precision escalation preserves transitions and local derivatives."""

from dataclasses import replace
from time import monotonic

import numpy as np
import pytest
from scipy.linalg import expm, expm_frechet

pytest.importorskip("mpmath")

from autoformalism.fitting import affine_precision as precision


def test_extreme_fast_slow_cancellation_recovers_transition_and_derivative():
    """The exact slow mode is constant despite trillion-scale opposing rates."""
    matrix = np.array([[-1e12, 1e12], [1e12, -1e12]])
    evaluator = precision.TransitionEvaluator(precision.AffinePrecisionPolicy())
    value, derivative = evaluator(matrix, [-np.eye(2), np.zeros((2, 2))])
    assert value == pytest.approx(np.full((2, 2), 0.5), abs=1e-15)
    assert derivative[0] == pytest.approx(-value, abs=1e-15)
    assert derivative[1] == pytest.approx(np.zeros((2, 2)), abs=0)
    # This is the specific double-roundoff failure addressed by the fallback.
    assert np.max(np.abs(expm(matrix) - value)) > 1e-7
    assert evaluator.audit["multiprecision_completed"] == 1
    assert evaluator.audit["multiprecision_passes_completed"] == 2
    assert evaluator.audit["double_started"] == 0
    assert evaluator.audit["maximum_agreement_ratio"] < 1


def test_nonnormal_transition_and_frechet_against_independent_block_exponential():
    mpmath = pytest.importorskip("mpmath")
    context = mpmath.mp.clone()
    context.dps = 110
    # Distinct nearly cancelling fast/slow modes and a noncommuting direction.
    matrix = np.array([[-1e9 - 1, 2e9], [0.5e9, -1e9 - 1]])
    direction = np.array([[0.3, -0.7], [0.2, 0.4]])
    block = context.zeros(4)
    block[:2, :2] = context.matrix(matrix.tolist())
    block[2:, 2:] = context.matrix(matrix.tolist())
    block[:2, 2:] = context.matrix(direction.tolist())
    expected = context.expm(block)
    evaluator = precision.TransitionEvaluator(precision.AffinePrecisionPolicy())
    value, derivative = evaluator(matrix, [direction])
    assert value == pytest.approx(np.array(expected[:2, :2].tolist(), float), abs=1e-14)
    assert derivative[0] == pytest.approx(
        np.array(expected[:2, 2:].tolist(), float), abs=1e-14
    )


@pytest.mark.parametrize("directions", [[], [np.eye(2), np.zeros((2, 2))]])
def test_moderate_matrices_retain_double_backend(directions):
    matrix = np.array([[-2.0, 3], [0.1, -1]])
    evaluator = precision.TransitionEvaluator(precision.AffinePrecisionPolicy())
    value, derivative = evaluator(matrix, directions)
    assert value == pytest.approx(expm(matrix), abs=0)
    for actual, direction in zip(derivative, directions, strict=True):
        assert actual == pytest.approx(
            expm_frechet(matrix, direction, compute_expm=False), abs=1e-15
        )
    assert evaluator.audit["double_completed"] == 1
    assert evaluator.audit["multiprecision_started"] == 0


def test_multiprecision_zero_generator_has_exact_nonzero_frechet_direction():
    policy = replace(
        precision.AffinePrecisionPolicy(), dimensionless_norm_threshold=1e-20
    )
    # Direct recurrence check also covers zero norm, which normally stays double.
    direction = np.array([[1.0, 3], [-2, 0.0]])
    _, value, derivative = precision._multiprecision(
        np.zeros((2, 2)), [direction], 60, policy, monotonic() + 10
    )
    assert np.array(value.tolist(), float) == pytest.approx(np.eye(2), abs=0)
    assert np.array(derivative[0].tolist(), float) == pytest.approx(direction, abs=0)


@pytest.mark.parametrize(
    "change,message",
    [
        ({"maximum_dimension": 1}, "dimension allowance"),
        ({"maximum_directions": 1}, "dimension allowance"),
        ({"maximum_scalings": 1}, "scaling allowance"),
        ({"maximum_taylor_terms": 1}, "Taylor allowance"),
    ],
)
def test_precision_caps_fail_without_partial_packet(change, message):
    policy = replace(precision.AffinePrecisionPolicy(), **change)
    evaluator = precision.TransitionEvaluator(policy)
    with pytest.raises(precision.AffinePrecisionUnavailable, match=message):
        evaluator(np.diag([-1e9, -2]), [np.eye(2), -np.eye(2)])
    assert evaluator.audit["failures"] == 1
    assert evaluator.audit["multiprecision_completed"] == 0


def test_precision_disagreement_is_unavailable(monkeypatch):
    actual = precision._multiprecision

    def disagree(matrix, directions, digits, policy, deadline):
        context, value, derivative = actual(
            matrix, directions, digits, policy, deadline
        )
        if digits == 90:
            value[0, 0] += context.mpf("0.01")
        return context, value, derivative

    monkeypatch.setattr(precision, "_multiprecision", disagree)
    evaluator = precision.TransitionEvaluator(precision.AffinePrecisionPolicy())
    with pytest.raises(precision.AffinePrecisionUnavailable, match="disagreement"):
        evaluator(np.diag([-1e9, -2]), [])
    assert evaluator.audit["maximum_agreement_ratio"] > 1
    assert evaluator.audit["multiprecision_completed"] == 0


def test_overflowing_norm_is_bounded_and_audit_remains_json_finite():
    import json

    evaluator = precision.TransitionEvaluator(precision.AffinePrecisionPolicy())
    with pytest.raises(precision.AffinePrecisionUnavailable, match="scaling allowance"):
        evaluator(np.full((2, 2), 1e308), [])
    assert evaluator.audit["norm_overflows"] == 1
    json.dumps(evaluator.audit, allow_nan=False)


def test_nonfinite_input_is_rejected_before_backend():
    evaluator = precision.TransitionEvaluator(precision.AffinePrecisionPolicy())
    with pytest.raises(precision.NumericalInvalidTrial, match="nonfinite"):
        evaluator(np.array([[float("nan")]]), [])
    assert evaluator.audit["double_started"] == 0
    assert evaluator.audit["multiprecision_started"] == 0


def test_deadline_overrun_cannot_return_transition(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(precision, "monotonic", lambda: clock[0])
    original = precision._multiprecision

    def overrun(*args, **kwargs):
        result = original(*args, **kwargs)
        clock[0] = 102.0
        return result

    monkeypatch.setattr(precision, "_multiprecision", overrun)
    evaluator = precision.TransitionEvaluator(precision.AffinePrecisionPolicy())
    with pytest.raises(TimeoutError, match="wall-clock"):
        evaluator(np.diag([-1e9, -2]), [], 101.0)
    assert evaluator.audit["multiprecision_passes_started"] == 1
    assert evaluator.audit["multiprecision_completed"] == 0


@pytest.mark.parametrize("matrix,directions", [([1, 2], []), (np.eye(2), [np.eye(3)])])
def test_invalid_shapes_are_programming_errors(matrix, directions):
    evaluator = precision.TransitionEvaluator(precision.AffinePrecisionPolicy())
    with pytest.raises(ValueError) as caught:
        evaluator(matrix, directions)
    assert not isinstance(caught.value, precision.AffinePrecisionUnavailable)
    assert evaluator.audit["double_started"] == 0


@pytest.mark.parametrize(
    "change",
    [
        {"decimal_digits": (90, 60)},
        {"decimal_digits": (True, 60)},
        {"transition_seconds": 0},
        {"maximum_dimension": True},
        {"dimensionless_norm_threshold": float("inf")},
    ],
)
def test_invalid_policy_rejected(change):
    with pytest.raises(ValueError):
        precision.AffinePrecisionPolicy(**change)


def test_training_and_prefix_opt_in_policy_preserved_in_recompiled_oracle():
    pytest.importorskip("casadi")
    from autoformalism.fitting import prefix_affine as affine
    from autoformalism.fitting import prefix_initialization as kernel
    from autoformalism.fitting.public_fitting import content_sha256
    from tests.test_prefix_initialization import problem, training

    model, data = problem(), training()
    policy = precision.AffinePrecisionPolicy(dimensionless_norm_threshold=1e-9)
    oracle = affine.TrainingOracle(
        model, data, kernel.NumericalPolicy(), precision_policy=policy
    )
    vector = oracle.vector(oracle.start)
    residual, jacobian = oracle.evaluate(vector, monotonic() + 20)
    legacy = affine.TrainingOracle(model, data, kernel.NumericalPolicy())
    expected, derivative = legacy.evaluate(vector, monotonic() + 20)
    assert residual == pytest.approx(expected, abs=1e-13)
    assert jacobian == pytest.approx(derivative, abs=1e-13)
    replay = oracle.with_numerical_policy(kernel.NumericalPolicy())
    assert replay.precision_policy == policy
    assert replay.affine_audit["transition_precision"]["multiprecision_started"] == 0
    assert oracle.affine_audit["transition_precision"]["multiprecision_completed"] > 0
    assert "transition_precision" not in legacy.affine_audit
    parameters = oracle.parameters(vector)
    fit = {
        "problem_sha256": content_sha256(model),
        "fit_success": True,
        "shared_parameters": {name: parameters[name] for name in oracle.shared},
        "training_scales": oracle.scales,
    }
    prefix = kernel.make_prefix(data.rows[0], kernel.PrefixPolicy(duration=0.5))
    prefix_oracle = affine.PrefixOracle(
        model, fit, prefix, kernel.NumericalPolicy(), precision_policy=policy
    )
    point = prefix_oracle.vector(prefix_oracle.start)
    value, gradient = prefix_oracle.evaluate(point, monotonic() + 20)
    other, other_gradient = prefix_oracle.evaluate(
        point, monotonic() + 20, method="Radau"
    )
    assert value == pytest.approx(other, abs=1e-9)
    assert gradient == pytest.approx(other_gradient, abs=1e-9)
