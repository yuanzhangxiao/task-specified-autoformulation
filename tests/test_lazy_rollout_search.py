"""Small deterministic controls for optional residual-only TRF trials."""

from copy import deepcopy
from time import monotonic
from types import SimpleNamespace

import numpy as np
import pytest

from autoformalism.fitting import lazy_rollout_search as lazy
from autoformalism.fitting import poor_fit
from autoformalism.fitting import recovery_numerics as recovery


class NonlinearOracle:
    """Newton overshoots a scalar square-law root before TRF shrinks its step."""

    names = ("a",)
    lower, upper, units = np.array([-np.inf]), np.array([np.inf]), np.ones(1)

    def __init__(self):
        self.calls = []

    def vector(self, parameters):
        return np.array([parameters[n] for n in self.names], dtype=float)

    def parameters(self, vector):
        return dict(zip(self.names, vector.tolist(), strict=True))

    def evaluate(self, vector, deadline, *, jacobian=True):
        self.calls.append((vector.copy(), deadline, jacobian))
        return vector[:1] ** 2 - 0.25, np.array([[2 * vector[0]]]) if jacobian else None


def run(oracle=None, **options):
    """Keep these controls cheap while exercising the installed real TRF."""
    oracle = oracle or NonlinearOracle()
    saved = []
    kwargs = {
        "calls": 100,
        "target": -1.0,
        "objective_multiplier": 1.0,
        "record_steps": True,
        "record_full_steps": True,
        "telemetry": True,
        "lazy_jacobian": True,
    } | options
    result = recovery.search(
        oracle,
        oracle.parameters(np.array([0.1] * len(oracle.names))),
        monotonic() + 5,
        lambda item: saved.append(deepcopy(item)),
        **kwargs,
    )
    return oracle, result, saved


@pytest.mark.parametrize(
    "scaling", ["jacobian", "fixed_coordinates", "guarded_relative"]
)
@pytest.mark.parametrize("normalized", [False, True])
def test_real_trf_matches_eager_accepted_trajectory_with_fewer_jacobians(
    scaling,
    normalized,
):
    eager_oracle, eager, _ = run(
        lazy_jacobian=False,
        optimizer_scaling=scaling,
        normalized_objective=normalized,
    )
    oracle, result, _ = run(
        optimizer_scaling=scaling,
        normalized_objective=normalized,
    )
    assert result["best"] == eager["best"]
    assert result["stop_reason"] == eager["stop_reason"]
    assert result["calls"] == eager["calls"]
    assert [s["reduced_vector"] for s in result["accepted_steps"]] == [
        s["reduced_vector"] for s in eager["accepted_steps"]
    ]
    assert result["jacobian_calls"] < len(eager_oracle.calls)
    assert result["calls"] - result["jacobian_calls"] >= 1
    unrequested = [
        r for r in result["evaluations"] if r["gradient_status"] == "not_requested"
    ]
    assert unrequested and all(r["projected_gradient_inf"] is None for r in unrequested)
    assert len(oracle.calls) == result["oracle_calls"]
    assert len({d for _, d, _ in oracle.calls}) == 1


class HeldOracle(NonlinearOracle):
    """A deliberately unusable held derivative cannot poison active evidence."""

    names = ("a", "held")
    lower, upper, units = np.full(2, -np.inf), np.full(2, np.inf), np.ones(2)
    active_indices = (0,)

    def evaluate(self, vector, deadline, *, jacobian=True):
        residual, _ = super().evaluate(vector, deadline, jacobian=jacobian)
        return residual, np.array([[2 * vector[0], np.nan]]) if jacobian else None


@pytest.mark.parametrize("policy", [None, poor_fit.Policy()])
def test_held_derivative_excluded_independently_of_poor_fit(policy):
    oracle, result, _ = run(
        HeldOracle(),
        poor_fit=policy,
        optimizer_scaling="guarded_relative",
    )
    assert result["stop_reason"] == "step_or_gradient_converged"
    assert result["best"]["parameters"] == pytest.approx({"a": 0.5, "held": 0.1})
    assert result["solver_scaling"]["scales"] == [1.0]
    assert result["stationarity_scope"] == "active_block"
    assert result["full_endpoint_audit_required"]
    assert all(vector[1] == 0.1 for vector, _, _ in oracle.calls)
    assert all(step["reduced_vector"][1] == 0.1 for step in result["accepted_steps"])
    assert result["history_coordinate_names"] == ["a", "held"]


def test_empty_active_block_uses_one_residual_and_no_derivative():
    oracle = HeldOracle()
    oracle.active_indices = ()
    _, result, _ = run(
        oracle, optimizer_scaling="guarded_relative", normalized_objective=True
    )
    assert result["stop_reason"] == "output_problem_solved"
    assert result["solution_scope"] == "fixed_active_block"
    assert result["calls"] == 1 and result["jacobian_calls"] == 0
    assert result["last_accepted_parameters"] == {"a": 0.1, "held": 0.1}
    assert result["stationarity_scope"] == "active_block"


@pytest.mark.parametrize("failure", ["nan", "error", "timeout", "shape"])
def test_derivative_failure_preserves_residual_checkpoint(failure):
    class Failing(NonlinearOracle):
        def evaluate(self, vector, deadline, *, jacobian=True):
            r, j = super().evaluate(vector, deadline, jacobian=jacobian)
            if jacobian:
                if failure == "error":
                    raise RuntimeError("derivative unavailable")
                if failure == "timeout":
                    raise TimeoutError("derivative deadline")
                j = np.array([[np.nan]]) if failure == "nan" else np.ones((1, 2))
            return r, j

    _, result, saved = run(Failing())
    assert result["best"]["training_nmse"] == pytest.approx(0.24**2)
    assert result["stop_reason"] == (
        "budget_limited" if failure == "timeout" else "numerical_failure"
    )
    assert result["failed_jacobian_calls"] == result["jacobian_calls"] == 1
    assert result["evaluations"][0]["projected_gradient_inf"] is None
    assert result["evaluations"][0]["gradient_status"] == "failed"
    assert any(s["best"] == result["best"] and s["jacobian_calls"] == 0 for s in saved)


def test_improvement_is_checkpointed_before_later_jacobian_timeout():
    class LaterFailure(NonlinearOracle):
        def evaluate(self, vector, deadline, *, jacobian=True):
            if jacobian and vector[0] != 0.1:
                self.calls.append((vector.copy(), deadline, jacobian))
                raise TimeoutError("accepted derivative expired")
            return super().evaluate(vector, deadline, jacobian=jacobian)

    _, result, saved = run(LaterFailure())
    assert result["best"]["training_nmse"] < 0.24**2
    assert result["stop_reason"] == "budget_limited"
    assert result["last_accepted_parameters"] == result["best"]["parameters"]
    assert len(result["accepted_steps"]) == 1
    assert any(s["best"] == result["best"] and s["jacobian_calls"] == 1 for s in saved)


@pytest.mark.parametrize("failure", ["nan", "typed", "shape", "timeout", "error"])
def test_invalid_trial_handling_and_failed_costs(failure):
    class FailingTrial(NonlinearOracle):
        def evaluate(self, vector, deadline, *, jacobian=True):
            r, j = super().evaluate(vector, deadline, jacobian=jacobian)
            if vector[0] > 0.8:
                assert not jacobian
                if failure == "typed":
                    raise recovery.NumericalInvalidTrial("overflow")
                if failure == "timeout":
                    raise TimeoutError("expired")
                if failure == "error":
                    raise ValueError("oracle contract")
                return np.array([np.nan]) if failure == "nan" else np.ones((1, 1)), None
            return r, j

    _, result, _ = run(FailingTrial(), reject_invalid_trials=True)
    assert result["failed_calls"] >= 1
    if failure in {"nan", "typed"}:
        assert result["best"]["parameters"]["a"] == pytest.approx(0.5)
        assert result["rejected_calls"] == result["failed_calls"]
        assert result["stop_reason"] == "step_or_gradient_converged"
    else:
        assert result["rejected_calls"] == 0
        assert result["stop_reason"] == (
            "budget_limited" if failure == "timeout" else "numerical_failure"
        )


def test_cache_and_rejected_curvature_do_not_request_extra_jacobians(monkeypatch):
    def optimizer(fun, x0, *, jac, callback, **kwargs):
        assert np.array_equal(fun(x0), fun(x0))
        assert np.array_equal(jac(x0), jac(x0))
        rejected = np.array([1.0])
        fun(rejected)
        fun(x0)
        jac(x0)
        callback(SimpleNamespace(x=x0))
        return SimpleNamespace(success=False, message="done", optimality=1.0)

    monkeypatch.setattr(lazy, "least_squares", optimizer)
    oracle, result, _ = run()
    assert result["calls"] == 2 and result["jacobian_calls"] == 1
    assert len(oracle.calls) == 3
    assert result["accepted_steps"] == []


def test_no_stationarity_inference_from_a_rejected_residual(monkeypatch):
    class FlatTrial(NonlinearOracle):
        def evaluate(self, vector, deadline, *, jacobian=True):
            r, j = super().evaluate(vector, deadline, jacobian=jacobian)
            if vector[0] > 1:
                return np.array([2.0]), np.zeros((1, 1)) if jacobian else None
            return r, j

    def optimizer(fun, x0, *, jac, **kwargs):
        fun(x0)
        jac(x0)
        fun(np.array([2.0]))
        return SimpleNamespace(success=False, message="rejected", optimality=1.0)

    monkeypatch.setattr(lazy, "least_squares", optimizer)
    _, result, _ = run(FlatTrial(), poor_fit=poor_fit.Policy())
    assert result["stop_reason"] == "budget_limited"
    assert "stationarity" not in result
    assert result["evaluations"][-1]["projected_gradient_inf"] is None


def test_active_stationarity_is_explicitly_not_full_model_evidence():
    class FlatActive(HeldOracle):
        def evaluate(self, vector, deadline, *, jacobian=True):
            self.calls.append((vector.copy(), deadline, jacobian))
            return np.ones(1), np.array([[0.0, 99.0]]) if jacobian else None

    _, result, _ = run(FlatActive(), poor_fit=poor_fit.Policy())
    assert result["stop_reason"] == "active_block_stationary_poor_fit"
    assert result["stationarity"]["scope"] == "active_block"
    assert not result["stationarity"]["full_model_stationarity_certified"]


def test_residual_call_budget_does_not_renew_for_derivatives():
    oracle, result, _ = run(calls=2)
    assert result["stop_reason"] == "budget_limited"
    assert result["calls"] == result["residual_calls"] == 2
    assert result["jacobian_calls"] == 1
    assert len(oracle.calls) == 3


def test_shared_deadline_includes_checkpoint_callbacks(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(lazy, "monotonic", lambda: clock[0])
    oracle = NonlinearOracle()

    def checkpoint(value):
        if value["best"] is not None:
            clock[0] = 2.0

    result = recovery.search(
        oracle,
        {"a": 0.1},
        1.0,
        checkpoint,
        calls=100,
        target=-1,
        lazy_jacobian=True,
    )
    assert result["best"] is not None
    assert result["stop_reason"] == "budget_limited"
    assert result["jacobian_calls"] == 0
    assert len(oracle.calls) == 1


def test_profile_laziness_is_explicitly_unsupported_before_work():
    with pytest.raises(ValueError, match="does not support projected profiles"):
        run(profile=SimpleNamespace(outer=[0]))


@pytest.mark.parametrize("normalized", [False, True])
def test_matched_eager_and_lazy_controls_have_same_accepted_rosenbrock_path(normalized):
    class Rosenbrock(NonlinearOracle):
        names = ("x", "y")
        lower, upper, units = np.full(2, -np.inf), np.full(2, np.inf), np.ones(2)

        def evaluate(self, vector, deadline, *, jacobian=True):
            self.calls.append((vector.copy(), deadline, jacobian))
            x, y = vector
            return np.array([100 * (y - x * x), 1 - x]), (
                np.array([[-200 * x, 100], [-1, 0]]) if jacobian else None
            )

    values = []
    for residual_only in (False, True):
        oracle = Rosenbrock()
        result = lazy.search(
            oracle,
            {"x": -1.2, "y": 1.0},
            monotonic() + 5,
            lambda _: None,
            calls=200,
            target=-1,
            objective_multiplier=1.0,
            optimizer_scaling="fixed_coordinates",
            record_steps=True,
            record_full_steps=True,
            telemetry=True,
            poor_fit=poor_fit.Policy(),
            normalized_objective=normalized,
            residual_only_trials=residual_only,
        )
        assert result["best"]["parameters"] == pytest.approx({"x": 1, "y": 1})
        assert result["oracle_calls"] == len(oracle.calls)
        assert result["jacobian_calls"] == sum(jac for _, _, jac in oracle.calls)
        values.append(result)
    eager, deferred = values
    assert eager["stop_reason"] == deferred["stop_reason"]
    assert eager["calls"] == deferred["calls"]
    assert eager["best"] == deferred["best"]
    assert [s["reduced_vector"] for s in eager["accepted_steps"]] == [
        s["reduced_vector"] for s in deferred["accepted_steps"]
    ]
    # Several real finite trials are rejected; the saved work is not a zero-J
    # or linear oracle artifact, and no extra derivative rollout is hidden.
    assert eager["jacobian_calls"] - deferred["jacobian_calls"] >= 5
    assert deferred["jacobian_calls"] <= 0.8 * eager["jacobian_calls"]
    assert deferred["completed_jacobian_calls"] == deferred["jacobian_calls"]


@pytest.mark.parametrize("residual_only", [False, True])
def test_matched_controls_ignore_wrong_held_columns_and_need_no_j_when_fixed(
    residual_only,
):
    oracle = HeldOracle()
    result = lazy.search(
        oracle,
        {"a": 0.1, "held": 0.1},
        monotonic() + 5,
        lambda _: None,
        calls=100,
        target=-1,
        optimizer_scaling="guarded_relative",
        objective_multiplier=1,
        residual_only_trials=residual_only,
    )
    assert result["best"]["parameters"]["a"] == pytest.approx(0.5)
    oracle.active_indices = ()
    fixed = lazy.search(
        oracle,
        {"a": 0.1, "held": 0.1},
        monotonic() + 5,
        lambda _: None,
        calls=100,
        target=-1,
        residual_only_trials=residual_only,
    )
    assert fixed["jacobian_calls"] == 0


@pytest.mark.parametrize(
    "raw", [np.array([]), np.ones((1, 1)), np.array(np.nan), np.array([np.nan])]
)
def test_invalid_initial_residual_cannot_supply_curvature(raw):
    class Invalid(NonlinearOracle):
        def evaluate(self, vector, deadline, *, jacobian=True):
            assert not jacobian
            self.calls.append((vector.copy(), deadline, jacobian))
            return raw, None

    _, result, _ = run(Invalid(), reject_invalid_trials=True)
    assert result["stop_reason"] == "numerical_failure"
    assert result["best"] is None
    assert result["rejected_calls"] == result["jacobian_calls"] == 0
    assert result["failed_calls"] == 1


def test_invalid_cached_trial_never_supplies_fabricated_curvature(monkeypatch):
    class Invalid(NonlinearOracle):
        def evaluate(self, vector, deadline, *, jacobian=True):
            if vector[0] > 0.8:
                return np.array([np.inf]), None
            return super().evaluate(vector, deadline, jacobian=jacobian)

    def optimizer(fun, x0, *, jac, **kwargs):
        fun(x0)
        jac(x0)
        q = np.ones(1)
        assert np.isinf(fun(q)).all()
        assert np.isinf(fun(q)).all()
        with pytest.raises(recovery.NumericalInvalidTrial, match="Jacobian requested"):
            jac(q)
        return SimpleNamespace(success=False, message="done", optimality=1)

    monkeypatch.setattr(lazy, "least_squares", optimizer)
    _, result, _ = run(Invalid(), reject_invalid_trials=True)
    assert result["calls"] == 2
    assert result["jacobian_calls"] == 1
    assert result["rejected_calls"] == 1


def test_accuracy_gate_prevents_unearned_early_stop():
    _, result, _ = run(calls=2, target=1, acceptable=lambda _: False)
    assert result["stop_reason"] == "budget_limited"
    assert result["calls"] == 2


def test_derivative_residual_mismatch_fails_without_losing_checkpoint():
    class Inconsistent(NonlinearOracle):
        def evaluate(self, vector, deadline, *, jacobian=True):
            residual, derivative = super().evaluate(vector, deadline, jacobian=jacobian)
            if jacobian:
                residual = residual + 1e-12
            return residual, derivative

    _, result, saved = run(Inconsistent(), optimizer_scaling="guarded_relative")
    assert result["stop_reason"] == "numerical_failure"
    assert "changed the cached residual" in result["message"]
    assert result["best"]["parameters"] == {"a": 0.1}
    assert result["best"]["training_nmse"] == pytest.approx(0.24**2)
    assert result["completed_jacobian_calls"] == 0
    assert result["failed_jacobian_calls"] == 1
    assert result["evaluations"][0]["projected_gradient_inf"] is None
    assert any(s["best"] == result["best"] and s["jacobian_calls"] == 0 for s in saved)
