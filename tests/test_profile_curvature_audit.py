"""Local curvature evidence stays separate from global fit or confidence claims."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from autoformalism.fitting import profile_curvature_audit as curvature
from autoformalism.fitting.recovery_numerics import NumericalInvalidTrial
from tests import test_prefix_meal_joint_profile as joint_tests


class Oracle:
    names = ("a", "b", "root")
    units = np.ones(3)
    lower = np.array([-np.inf, -np.inf, 0.0])
    upper = np.full(3, np.inf)
    reduction = SimpleNamespace(blocks=())

    def vector(self, parameters):
        point = np.array([parameters[n] for n in self.names])
        if (
            not np.isfinite(point).all()
            or np.any(point < self.lower)
            or np.any(point > self.upper)
        ):
            raise ValueError("invalid point")
        return point

    def parameters(self, vector):
        return dict(zip(self.names, map(float, vector), strict=True))


def factory(residual, *, active=None, fail=None, expose=True):
    class Profiler:
        inner = (2,)
        outer = (0, 1)

        def __init__(self, oracle):
            self.oracle = oracle
            self.audit = {"sha256": "synthetic"}
            self.calls = 0

        def profile(self, values, deadline):
            del deadline
            self.calls += 1
            self.oracle.vector(values)
            if fail:
                fail(values, self.calls)
            vector = np.asarray(residual(values), dtype=float)
            if expose:
                self.last_residual = vector
            return {
                "training_nmse": float(np.mean(vector**2)),
                "parameters": dict(values) | {"root": 0.0},
                "joint_solve": {"active_mask": [active(values) if active else 0]},
            }

    return Profiler


def run(residual, *, start=None, oracle=None, factory_options=None, **kwargs):
    records = []
    start = start or {"a": 0.0, "b": 0.0, "root": 1.0}
    result = curvature.curvature_audit(
        oracle or Oracle(),
        start,
        10,
        records.append,
        profiler_factory=factory(residual, **(factory_options or {})),
        **kwargs,
    )
    return result, records


def test_quadratic_optimum_has_curvature_without_false_step_sensitive_flag():
    start = {"a": 0.0, "b": 0.0, "root": 1.0}
    original = deepcopy(start)
    result, records = run(lambda p: [1, p["a"], 2 * p["b"]], start=start)
    assert result["complete"]
    assert result["status"] == "no_resolved_descent_in_audit"
    assert result["mixed_curvature"]["qualified_local_model"]
    assert all(c["status"] == "two_scale_consistent" for c in result["coordinates"])
    assert result["mixed_curvature"]["hessians"][1] == pytest.approx(
        np.diag([2 / 3, 8 / 3])
    )
    assert result["residual_jacobian"]["rank"] == 2
    assert not result["material_improvement"]
    assert result["best_candidate"] is None
    assert not result["stationarity_certified"]
    assert not result["parameter_confidence_certified"]
    assert start == original
    assert records[-1]["calls_completed"] == result["calls_completed"]
    assert any(r["status"] == "running" and r["calls_completed"] > 1 for r in records)


def test_actual_mixed_curvature_detects_saddle_despite_positive_gauss_newton():
    def residual(p):
        a, b = p["a"], p["b"]
        return [1 + a * a + b * b - 4 * a * b, 0.1 * a, 0.1 * b]

    result, records = run(residual)
    audit = result["mixed_curvature"]
    assert result["complete"]
    assert audit["resolved_negative_curvature"]
    assert audit["eigenvalues"][0] < -1
    assert (
        np.linalg.eigvalsh(result["residual_jacobian"]["gauss_newton_hessian"])[0] > 0
    )
    assert result["status"] == "material_descent_detected"
    assert (
        result["best_candidate"]["training_nmse"]
        < result["baseline"]["training_nmse"] - 1e-4
    )
    assert result["best_origin"] == "coupled_candidate"
    assert any(r["best_candidate"] is not None for r in records)
    assert result["independent_original_verification_required"]


def test_boundary_stencil_respects_domain_and_reports_one_sided_derivatives():
    oracle = Oracle()
    oracle.lower = np.array([0.0, -np.inf, 0.0])
    oracle.upper = np.array([0.0004, np.inf, np.inf])
    result, _ = run(lambda p: [1 + p["a"], p["b"]], oracle=oracle)
    assert result["complete"]
    row = result["coordinates"][0]
    assert row["side"] == 1
    assert row["step"] == pytest.approx(0.0002)
    assert row["scales"][1]["gradient"] == pytest.approx(1.0, abs=1e-9)
    assert not result["material_improvement"]
    assert all(0 <= s["displacement"][0] <= 0.0004 for s in result["samples"])


def test_active_set_switch_withholds_derivative_conclusions_not_actual_improvement():
    result, _ = run(
        lambda p: [1 - p["a"], p["b"]],
        factory_options={"active": lambda p: int(p["a"] > 0)},
    )
    assert result["complete"]
    assert result["material_improvement"]
    assert not result["mixed_curvature"]["active_set_stable"]
    assert not result["mixed_curvature"]["qualified_local_model"]
    assert result["residual_jacobian"]["status"] == "derivative_conclusion_deferred"
    assert result["best_candidate"] is not None


def test_profiled_and_fixed_coordinates_are_excluded_from_all_mixed_points():
    result, _ = run(lambda p: [1, p["a"], p["b"]], fixed={"a": 0.0})
    assert result["complete"]
    assert [c["name"] for c in result["coordinates"]] == ["b"]
    assert result["inner_coordinates"] == ["root"]
    assert result["best"]["parameters"]["a"] == 0
    assert all(len(s["displacement"]) == 1 for s in result["samples"])


@pytest.mark.parametrize("fixed", [{"root": 1.0}, {"unknown": 0}, {"a": 2.0}])
def test_invalid_fixed_identity_never_silently_changes_endpoint(fixed):
    result, _ = run(lambda p: [1, p["a"]], fixed=fixed)
    assert result["status"] == "incomplete"
    assert result["calls_started"] == 0


def test_nonfinite_trials_are_accounted_and_do_not_certify_no_descent():
    def fail(values, calls):
        del calls
        if values["a"] > 0:
            raise NumericalInvalidTrial("pairing differs")

    result, _ = run(lambda p: [1, p["a"]], factory_options={"fail": fail})
    assert result["status"] == "numerically_unresolved"
    assert result["calls_failed"] > 0
    assert result["calls_started"] == result["calls_completed"] + result["calls_failed"]
    assert result["mixed_curvature"]["status"] == "numerically_unresolved"


def test_call_cap_and_timeout_preserve_the_best_finite_evaluation():
    result, records = run(lambda p: [1 + p["a"], p["b"]], maximum_calls=2)
    assert result["status"] == "incomplete"
    assert result["calls_started"] == result["calls_completed"] == 2
    assert (
        result["best_candidate"]["training_nmse"] < result["baseline"]["training_nmse"]
    )
    assert any(r["best_candidate"] is not None for r in records)

    def timeout(values, calls):
        del values
        if calls == 3:
            raise TimeoutError("synthetic deadline")

    result, _ = run(lambda p: [1 + p["a"]], factory_options={"fail": timeout})
    assert result["status"] == "incomplete"
    assert result["calls_completed"] == 2
    assert result["calls_failed"] == 1
    assert result["best_candidate"] is not None
    assert "timing" in result


def test_persistence_failure_stops_further_calls():
    records = []

    def checkpoint(result):
        records.append(result)
        if len(records) == 2:
            raise OSError("disk unavailable")

    result = curvature.curvature_audit(
        Oracle(),
        {"a": 0.0, "b": 0.0, "root": 1.0},
        10,
        checkpoint,
        profiler_factory=factory(lambda _: [1]),
    )
    assert result["status"] == "incomplete"
    assert result["calls_started"] == 1
    assert len(records) == 2


def test_expiry_before_profile_start_has_distinct_attempted_count(monkeypatch):
    budget = curvature.execution._budget(10, 250)
    original = budget.numerical
    calls = []

    def numerical(function):
        calls.append(function)
        if len(calls) == 2:
            raise TimeoutError("expired before profile")
        return original(function)

    monkeypatch.setattr(budget, "numerical", numerical)
    monkeypatch.setattr(curvature.execution, "_budget", lambda *_: budget)
    result, _ = run(lambda _: [1])
    assert result["status"] == "incomplete"
    assert result["calls_attempted"] == 1
    assert (
        result["calls_started"]
        == result["calls_completed"]
        == result["calls_failed"]
        == 0
    )


def test_no_residual_interface_is_explicit_and_has_no_fabricated_svd():
    result, _ = run(lambda p: [1, p["a"], p["b"]], factory_options={"expose": False})
    assert result["complete"]
    assert result["residual_jacobian"]["status"] == "paired_residual_unavailable"
    assert "rank" not in result["residual_jacobian"]
    assert result["samples"][1]["loss_difference_method"] == "loss_subtraction"


def test_residual_difference_avoids_subtraction_of_large_nearly_equal_losses():
    base = {"training_nmse": 1e16}
    r0 = np.array([1e8, 1.0])
    r1 = np.array([1e8, 1.001])
    delta, method, floor = curvature._differences(base, base, r1, r0)
    assert delta == pytest.approx((1.001**2 - 1) / 2)
    assert method == "paired_residual_difference"
    assert floor < abs(delta) * 1e-10


def test_real_joint_profile_exposes_only_ephemeral_paired_residual():
    _, oracle, _ = joint_tests.fixture()
    profiler = joint_tests.factory(oracle)
    chart = profiler.meal
    result = curvature.curvature_audit(
        oracle,
        oracle.start,
        30,
        lambda _: None,
        profiler_factory=joint_tests.factory,
        coordinate_chart=chart,
    )
    assert result["complete"], result.get("error")
    assert result["calls_failed"] == 0
    assert result["coordinates"][0]["name"] == "k"
    assert result["coordinates"][0]["coordinate"] == "log_ratio"
    assert (
        result["samples"][1]["loss_difference_method"] == "paired_residual_difference"
    )
    assert result["residual_jacobian"]["residual_arrays_persisted"] is False
    assert "last_residual" not in str(result)


def test_all_outer_coordinates_fixed_needs_only_one_profile_call():
    result, _ = run(lambda _: [1], fixed={"a": 0.0, "b": 0.0})
    assert result["complete"]
    assert result["calls_started"] == 1
    assert result["mixed_curvature"]["status"] == "no_free_outer_coordinates"


def test_regularized_correction_is_tested_at_its_predicted_magnitude():
    result, _ = run(lambda p: [1, 1000 * (p["a"] - 0.0002), p["b"]])
    rows = [
        r
        for r in result["coupled_probes"]
        if r["direction"] == "regularized_local_correction"
    ]
    assert rows
    assert max(row["radius"] for row in rows) == pytest.approx(0.0002, rel=1e-6)
    assert result["best_candidate"]["parameters"]["a"] == pytest.approx(
        0.0002, rel=1e-6
    )


def test_expiry_in_final_callback_revokes_complete_but_keeps_candidate(monkeypatch):
    budget = curvature.execution._budget(10, 250)
    original = budget.check
    state = {"final": False}

    def check(*, numerical=True):
        if state["final"]:
            budget.timeout_reason = "overall_wall_limit"
            raise TimeoutError("final callback expired")
        original(numerical=numerical)

    def checkpoint(result):
        if result["complete"]:
            state["final"] = True

    monkeypatch.setattr(budget, "check", check)
    monkeypatch.setattr(curvature.execution, "_budget", lambda *_: budget)
    result = curvature.curvature_audit(
        Oracle(),
        {"a": 0.0, "b": 0.0, "root": 1.0},
        10,
        checkpoint,
        profiler_factory=factory(lambda p: [1 - p["a"], p["b"]]),
    )
    assert result["status"] == "incomplete"
    assert result["complete"] is False
    assert result["best_candidate"] is not None
    assert result["timing"]["timeout_reason"] == "overall_wall_limit"


def test_unstable_weak_jacobian_column_is_not_hidden_by_a_large_column():
    a = np.array([[1e6, 0.1], [1e6, 0.1]])
    b = np.array([[1e6, 0.2], [1e6, 0.2]])
    result = curvature._jacobian_audit([a, b], np.eye(2), True, np.ones(2))
    assert result["column_two_scale_consistent"] == [True, False]
    assert not result["two_scale_consistent"]
    assert result["status"] == "derivative_conclusion_deferred"


def test_negative_chart_curvature_at_a_bound_is_not_a_feasible_saddle_claim():
    oracle = Oracle()
    oracle.lower = np.array([0.0, -np.inf, 0.0])
    # At a=0 the feasible outward derivative is positive. Negative curvature
    # alone does not make this endpoint a constrained saddle or descent point.
    result, _ = run(
        lambda p: [np.sqrt(1 + p["a"] - p["a"] ** 2)],
        oracle=oracle,
        fixed={"b": 0.0},
    )
    assert result["complete"]
    audit = result["mixed_curvature"]
    assert audit["resolved_negative_curvature"]
    assert not audit["negative_curvature_implies_constrained_saddle"]
    assert not audit["negative_curvature_implies_feasible_descent"]
    assert audit["actual_feasible_objective_decrease_required"]
    assert not result["material_improvement"]
    assert result["stencil_coverage"]["axial_completed"] == 2
    assert result["stencil_coverage"]["mixed_completed"] == 0


def test_partial_stencil_coverage_is_explicit_after_call_exhaustion():
    result, _ = run(lambda p: [1 + p["a"], p["b"]], maximum_calls=2)
    assert result["status"] == "incomplete"
    assert result["stencil_coverage"] == {
        "axial_planned": 4,
        "axial_attempted": 1,
        "axial_completed": 0,
        "mixed_planned": 2,
        "mixed_attempted": 0,
        "mixed_completed": 0,
        "scope": "two-scale derivative stencils; excludes finite candidate probes",
    }
