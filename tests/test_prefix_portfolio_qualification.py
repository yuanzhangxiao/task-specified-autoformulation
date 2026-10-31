"""Independent replay budgets and complete, identity-checked local coverage."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("casadi")

from autoformalism.fitting import prefix_continuation as continuation
from autoformalism.fitting import prefix_portfolio as portfolio
from autoformalism.fitting import prefix_portfolio_qualification as q
from autoformalism.fitting.sensitivity_probe import SymbolicRolloutFailure
from tests.test_prefix_initialization import problem, training
from tests.test_prefix_portfolio import composite_case


class LinearFixture:
    """Two local directions with an exact, inexpensive paired Jacobian."""

    names = ("a", "b")
    lower, upper, units = np.full(2, -2.0), np.full(2, 2.0), np.ones(2)

    def __init__(self):
        self.start = {"a": 0.1, "b": 0.2}
        self.original = SimpleNamespace(settings={"rtol": 1e-11, "atol": 1e-13})
        self.calls = []

    def parameters(self, vector):
        return dict(zip(self.names, np.asarray(vector).tolist(), strict=True))

    def vector(self, parameters):
        return np.array([parameters[name] for name in self.names])

    def local_coordinate_scales(self, vector):
        return self.units.copy()

    def evaluate(self, vector, deadline, *, jacobian=True, method=None):
        self.calls.append((jacobian, method))
        derivative = np.array([[1.0, 2.0], [2.0, -1.0], [1.0, 1.0]])
        return derivative @ vector, derivative if jacobian else None


@pytest.fixture
def oracle():
    return q.FixedRadauQualifier(
        portfolio.PortfolioOracle(LinearFixture(), {"test": True})
    )


def test_coordinate_journals_require_complete_matching_coverage(oracle):
    first = q.qualify_coordinate(oracle, 0, 5, lambda _: None)
    assert first["passed"]
    partial = q.aggregate_qualification(oracle, [first])
    assert not partial["complete"] and not partial["search_qualified"]
    assert partial["missing_indices"] == [1]
    second = q.qualify_coordinate(oracle, 1, 5, lambda _: None)
    assert second["passed"]
    full = q.aggregate_qualification(oracle, [second, first])
    assert full["passed"] and full["search_qualified"]
    assert full["checked_indices"] == [0, 1]
    assert not full["parameter_information_qualified"]
    # Only one baseline state/Jacobian pair is needed in this process. Every
    # stencil still evaluates its own state-only residual under the same method.
    assert sum(jacobian for jacobian, _ in oracle.oracle.oracle.calls) == 1
    assert {method for _, method in oracle.oracle.oracle.calls} == {"Radau"}
    resumed = q.FixedRadauQualifier(
        portfolio.PortfolioOracle(LinearFixture(), {"test": True})
    )
    replayed = q.qualify_coordinate(resumed, 1, 5, lambda _: None)
    assert q.aggregate_qualification(resumed, [first, replayed])["search_qualified"]


@pytest.mark.parametrize(
    "corruption", ["duplicate", "identity", "policy", "hash", "array", "direction"]
)
def test_aggregate_rejects_mixed_or_tampered_evidence(oracle, corruption):
    first = q.qualify_coordinate(oracle, 0, 5, lambda _: None)
    second = q.qualify_coordinate(oracle, 1, 5, lambda _: None)
    if corruption == "duplicate":
        second = first
    elif corruption == "identity":
        second["identity"]["parameters_sha256"] = "other"
    elif corruption == "policy":
        second["derivative_check"]["policy"]["derivative_relative_tolerance"] = 0.1
    elif corruption == "hash":
        second["derivative_check"]["base_jacobian_sha256"] = "other"
    elif corruption == "array":
        second["base_residual"][0] += 1
    else:
        second["derivative_check"]["directions"][0]["index"] = 0
    with pytest.raises(ValueError):
        q.aggregate_qualification(oracle, [first, second])


def test_failed_or_timed_out_coordinate_does_not_qualify_aggregate(oracle):
    first = q.qualify_coordinate(oracle, 0, 5, lambda _: None)
    second = q.qualify_coordinate(oracle, 1, 5, lambda _: None)
    failed = deepcopy(second)
    failed["status"] = "incomplete"
    failed["passed"] = False
    result = q.aggregate_qualification(oracle, [first, failed])
    assert result["missing_indices"] == [1]
    assert not result["search_qualified"]
    # Resolved derivative disagreement remains distinct from unattempted work.
    second["derivative_check"]["directions"][0].update(
        precision_status="failed", passed=False
    )
    result = q.aggregate_qualification(oracle, [first, second])
    assert result["complete"] and result["failed_indices"] == [1]
    assert not result["search_qualified"]


def test_independent_state_replays_preserve_failure_and_attempt_second_solver(
    monkeypatch,
):
    calls = []
    diagnostic = {
        "trajectory_id": "row-0",
        "integration_interval": [0.0, 1.0],
        "last_rhs_attempt": {"time": 0.5, "states": {"x": 1e8}},
        "attempt_is_accepted_solver_state": False,
    }

    class Replay(LinearFixture):
        def __init__(self, problem, training, numerical):
            super().__init__()
            assert numerical.rtol == 1e-11 and numerical.atol == 1e-13

        def evaluate(self, vector, deadline, *, jacobian=True, method=None):
            calls.append((jacobian, method))
            if method == "Radau":
                raise SymbolicRolloutFailure("step unavailable", diagnostic)
            return np.array([0.2, 0.3]), None

    monkeypatch.setattr(q.kernel, "TrainingOracle", Replay)
    failed = q.verify_start(problem(), training(), "Radau", 5, lambda _: None)
    completed = q.verify_start(problem(), training(), "BDF", 5, lambda _: None)
    assert calls == [(False, "Radau"), (False, "BDF")]
    assert not failed["replay_completed"]
    assert failed["failure_diagnostic"] == diagnostic
    assert failed["failure_cause"] == "not_established"
    assert completed["replay_completed"] and completed["residual"] == [0.2, 0.3]
    assert not completed["original_equations_verified"]
    assert failed["execution_policy"]["numerical_seconds"] == 5
    assert completed["execution_policy"]["numerical_seconds"] == 5


def test_diagnostic_opt_in_preserves_historical_invalidation_and_bounds_payload():
    diagnostic = {
        "trajectory_id": "r",
        "states": {"x": float("inf")},
        "many": list(range(1000)),
    }
    error = SymbolicRolloutFailure("failed", diagnostic)
    old, new = {}, {}
    continuation._invalidate(old, error, "verification")
    continuation._invalidate(new, error, "verification", preserve_diagnostic=True)
    assert "failure_diagnostic" not in old
    assert new["failure_diagnostic"]["states"]["x"] == "inf"
    assert len(new["failure_diagnostic"]["many"]) == 64
    assert diagnostic["states"]["x"] == float("inf")


def test_real_nonlinear_smoke_uses_original_chart_and_tight_state_replays():
    model, data = composite_case(True)
    before = model.model_dump_json(), data.model_dump_json()
    oracle = q.make_qualifier(model, data)
    assert oracle.portfolio_audit["route"] == "original_ODE"
    assert len(oracle.names) < len(oracle.original.names)
    replay = [
        q.verify_start(model, data, method, 10, lambda _: None) for method in q.METHODS
    ]
    assert all(row["replay_completed"] for row in replay)
    assert (
        np.max(np.abs(np.array(replay[0]["residual"]) - replay[1]["residual"])) < 1e-7
    )
    direction = q.qualify_coordinate(oracle, 0, 10, lambda _: None)
    assert direction["passed"]
    assert before == (model.model_dump_json(), data.model_dump_json())
    with pytest.raises(ValueError, match="training only"):
        q.make_qualifier(model, data.model_copy(update={"name": "validation"}))
