"""Complete MP arithmetic, exact local nodes, finite budgets and no promotion."""

import json
from copy import deepcopy
from time import monotonic

import casadi as ca
import mpmath
import numpy as np
import pytest

from autoformalism.fitting import prefix_affine as affine
from autoformalism.fitting import prefix_construction_recovery as recovery
from autoformalism.fitting import prefix_precision_probe as probe
from autoformalism.fitting.prefix_initialization import NumericalPolicy
from autoformalism.fitting.public_fitting import content_sha256
from autoformalism.fitting.recovery_numerics import NumericalInvalidTrial
from tests import test_prefix_affine as affine_tests
from tests.test_prefix_initialization import problem, training
from tests.test_prefix_structural_reduction import structural_case, wrapped


def context():
    result = mpmath.mp.clone()
    result.dps = 80
    return result


def ordinary_case():
    model, data = problem(), training()
    original = affine.TrainingOracle(model, data, NumericalPolicy())
    return recovery.make_oracle(
        model, data, NumericalPolicy(), original.start, structural_reduction=True
    )


def test_full_precision_generator_and_subtraction_do_not_cast_through_double():
    p = ca.SX.sym("p")
    expression = 1 / (1 + p) - 1
    numeric = ca.Function("cancellation", [p], [expression])
    assert float(numeric(1e-20)) == 0
    value = probe._Expression(expression, 100)(context(), {"p": context().mpf(1e-20)})[
        0
    ]
    assert float(value) == pytest.approx(-1e-20, rel=1e-12, abs=1e-40)


def test_full_residual_preserves_initial_dag_roots_outputs_and_forcing():
    model, data, original, vector = affine_tests.coupled_case.__wrapped__()
    before = content_sha256(model), content_sha256(data), vector.copy()
    reference, _ = original.evaluate(vector, monotonic() + 10)
    residual = probe._Residual(original, probe.PrecisionProbePolicy()).evaluate(
        vector, context(), monotonic() + 10
    )
    assert np.asarray(residual, float) == pytest.approx(reference, abs=1e-12)
    assert before[:2] == (content_sha256(model), content_sha256(data))
    assert np.array_equal(before[2], vector)


def test_structural_lift_is_evaluated_symbolically_before_coefficient_arithmetic():
    _, oracle = wrapped(structural_case())
    point = oracle.vector(oracle.start)
    q = ca.SX.sym("q", len(point))
    expression = probe._Expression(probe._symbolic_lift(oracle.reduction, q), 1000)
    ctx = context()
    actual = expression(ctx, {q[i].name(): ctx.mpf(v) for i, v in enumerate(point)})
    expected = oracle.reduction.lift(point)
    assert np.asarray(list(actual), float) == pytest.approx(expected, abs=1e-13)
    residue = probe._Residual(oracle, probe.PrecisionProbePolicy()).evaluate(
        point, ctx, monotonic() + 10
    )
    baseline, _ = oracle.evaluate(point, monotonic() + 10)
    assert np.asarray(residue, float) == pytest.approx(baseline, abs=1e-12)


def test_complete_fixed_point_comparison_is_json_finite_and_does_not_promote():
    original, oracle = ordinary_case()
    saved, checkpoints = deepcopy(oracle.start), []
    result = probe.compare(oracle, saved, 10, checkpoints.append, indices=(0,))
    assert result["status"] == "complete", result
    assert result["calls_started"] == result["calls_completed"] == 15
    assert result["calls_failed"] == 0
    assert result["search_performed"] is result["qualification_override"] is False
    assert result["parameters_sha256"] == content_sha256(saved)
    assert result["original_parameters_sha256"] == content_sha256(
        original.parameters(oracle.reduction.lift(oracle.vector(saved)))
    )
    assert oracle.start == saved
    assert len(checkpoints) >= 4
    json.dumps(result, allow_nan=False)
    coarse, fine = result["rows"][0]["scales"]
    assert coarse["step"] == 2 * fine["step"]
    assert fine["multiprecision_discrepancy_rms"][-1] < 1e-7
    assert fine["precision_difference_rms"] < 1e-30
    assert result["timing"]["numerical_active_wall_seconds"] > 0


def test_output_and_residual_cancellation_is_repaired_before_float_conversion():
    model = problem(shared=True).model_dump(mode="json")
    model["base_candidate"]["state_equations"] = [
        {"state": name, "rhs": "0"} for name in ("x", "z", "r")
    ]
    model["base_candidate"]["observation_mappings"][0]["expression"] = "x+theta"
    model["base_candidate"]["initial_conditions"][0]["fixed_value"] = 1e16
    model["graph"]["rules"]["x"] = {
        "initial": {"mode": "value", "scope": "shared_training", "guess": 1e16}
    }
    model = affine.PrefixFitProblem.model_validate(model)
    data = training()
    original = affine.TrainingOracle(model, data, NumericalPolicy())
    _, oracle = recovery.make_oracle(model, data, NumericalPolicy(), original.start)
    result = probe.compare(oracle, oracle.start, 10, None, indices=(0,))
    assert result["status"] == "complete", result
    row = result["rows"][0]["scales"][-1]
    assert row["guarded_discrepancy_rms"] > 0.1
    assert row["multiprecision_discrepancy_rms"][-1] < 1e-14


def test_bounds_use_inward_quadratic_same_nodes_for_every_precision():
    _, oracle = ordinary_case()
    saved = oracle.start | {"theta": float(oracle.lower[0])}
    result = probe.compare(oracle, saved, 10, None, indices=(0,))
    assert result["status"] == "complete", result
    for step in result["rows"][0]["scales"]:
        assert step["scheme"] == "one_sided_quadratic"
        assert all(v > 0 for v in step["actual_offsets"])
        assert step["multiprecision_discrepancy_rms"][-1] < 1e-6


def test_partial_timeout_retains_completed_scale_and_charges_failed_call(monkeypatch):
    _, oracle = ordinary_case()
    previous = probe._Residual.evaluate
    calls = 0

    def fail_later(self, vector, ctx, deadline):
        nonlocal calls
        calls += 1
        if calls == 7:
            raise TimeoutError("injected before second scale")
        return previous(self, vector, ctx, deadline)

    monkeypatch.setattr(probe._Residual, "evaluate", fail_later)
    result = probe.compare(oracle, oracle.start, 10, None, indices=(0,))
    assert result["status"] == "incomplete"
    assert len(result["rows"][0]["scales"]) == 1
    assert result["calls_started"] == result["calls_completed"] + 1
    assert result["calls_failed"] == 1
    assert "TimeoutError" in result["error"]


def test_interruption_preserves_callback_and_does_not_silently_resume(monkeypatch):
    _, oracle = ordinary_case()
    checkpoints = []

    def fail(*args):
        raise KeyboardInterrupt("interrupted")

    monkeypatch.setattr(probe._Residual, "evaluate", fail)
    with pytest.raises(KeyboardInterrupt):
        probe.compare(oracle, oracle.start, 10, checkpoints.append, indices=(0,))
    assert checkpoints[-1]["calls_failed"] == 1
    assert checkpoints[-1]["status"] == "incomplete"


@pytest.mark.parametrize("indices", [(), (0, 0), (-1,), (100,), (True,)])
def test_invalid_targeted_coordinate_roster_fails_before_calls(indices):
    _, oracle = ordinary_case()
    with pytest.raises(ValueError, match="coordinate indices"):
        probe.compare(oracle, oracle.start, 10, None, indices=indices)


def test_unsupported_symbolic_operation_and_work_limits_fail_closed():
    x = ca.SX.sym("x")
    with pytest.raises(ValueError, match="unsupported"):
        probe._Expression(ca.erf(x), 100)
    with pytest.raises(ValueError, match="node allowance"):
        probe._Expression(x + ca.sin(x), 1)
    with pytest.raises(ValueError, match="precisions"):
        probe.PrecisionProbePolicy(decimal_digits=(70, 40))
    with pytest.raises(ValueError, match="positive integer"):
        probe.PrecisionProbePolicy(maximum_columns=0)
    ctx = context()
    with pytest.raises(NumericalInvalidTrial, match="scaling allowance"):
        probe._exponential(
            ctx,
            ctx.matrix([[1e30]]),
            probe.PrecisionProbePolicy(maximum_scalings=1),
            monotonic() + 1,
        )


def test_comparison_never_modifies_legacy_policy_serialization():
    from autoformalism.fitting.prefix_fit_adequacy import Policy

    before = Policy().model_dump_json()
    probe.policy_receipt()
    assert Policy().model_dump_json() == before
    assert "diagnostic_precision" not in before


def test_saved_small_steps_are_reused_without_enlargement():
    _, oracle = ordinary_case()
    result = probe.compare(
        oracle, oracle.start, 10, None, indices=(0,), steps={0: (1e-7, 5e-8)}
    )
    assert result["status"] == "complete", result
    assert result["step_source"] == "saved_local_pairs"
    assert [s["step"] for s in result["rows"][0]["scales"]] == [1e-7, 5e-8]
    result = probe.compare(
        oracle, oracle.start, 10, None, indices=(0,), steps={0: (1.0, 0.5)}
    )
    assert result["status"] == "incomplete"
    assert "exceed declared local radius" in result["error"]


@pytest.mark.parametrize(
    "steps", [{1: (1e-7, 5e-8)}, {0: (-1, -0.5)}, {0: (1e-5, 1e-6)}]
)
def test_invalid_saved_steps_fail_before_numerical_work(steps):
    _, oracle = ordinary_case()
    with pytest.raises(ValueError, match="saved precision steps"):
        probe.compare(oracle, oracle.start, 10, None, indices=(0,), steps=steps)


def test_partial_step_mapping_preserves_selected_pairs_and_marks_fallback():
    _, oracle = ordinary_case()
    result = probe.compare(
        oracle, oracle.start, 10, None, indices=(0, 1), steps={0: (1e-7, 5e-8)}
    )
    assert result["status"] == "complete", result
    assert [r["step_source"] for r in result["rows"]] == [
        "saved_local_pair",
        "initial_local_pair",
    ]


def test_incorrect_analytic_derivative_is_not_hidden_by_precision(monkeypatch):
    _, oracle = ordinary_case()
    evaluate = oracle.evaluate

    def wrong(vector, deadline, *, jacobian=True, method=None):
        residual, derivative = evaluate(
            vector, deadline, jacobian=jacobian, method=method
        )
        return residual, derivative * 2 if jacobian else None

    monkeypatch.setattr(oracle, "evaluate", wrong)
    result = probe.compare(oracle, oracle.start, 10, None, indices=(0,))
    assert result["status"] == "complete", result
    for record in result["rows"][0]["scales"]:
        assert record["multiprecision_discrepancy_rms"][-1] > 1
        assert record["precision_difference_rms"] < 1e-30
    assert result["qualification_override"] is False
