"""Mixed equation routing, exact charts, objective pairing and lazy retention."""

from copy import deepcopy
from time import monotonic
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("casadi")

from autoformalism.fitting import prefix_continuation as continuation
from autoformalism.fitting import prefix_initialization as kernel
from autoformalism.fitting import prefix_portfolio as p
from autoformalism.fitting.recovery_numerics import NumericalInvalidTrial
from autoformalism.schemas.prefix_initialization import PrefixFitProblem, PrefixPolicy
from autoformalism.schemas.public_fitting import PublicSplit
from tests.test_prefix_gain_reduction import oracle_for
from tests.test_prefix_initialization import problem, training


def composite_case(nonlinear):
    """A composite readout and continuously supplied auxiliary survive routing."""
    source = oracle_for(observation="x+z")
    payload = source.problem.model_dump(mode="json")
    payload["context"]["auxiliaries"] = ["w"]
    payload["base_candidate"]["state_equations"][0]["rhs"] = (
        "-x*x+g*q+0.2*w" if nonlinear else "-x+g*q+0.2*w"
    )
    data = source.training.model_dump(mode="json")
    for row in data["rows"]:
        row["auxiliaries"] = {"w": [1 + t for t in row["time"]]}
    return PrefixFitProblem.model_validate(payload), PublicSplit.model_validate(data)


@pytest.mark.parametrize("nonlinear", [False, True])
def test_routing_reductions_roots_auxiliaries_and_composite_outputs(nonlinear):
    model, data = composite_case(nonlinear)
    before = model.model_dump_json(), data.model_dump_json()
    original, reduced = p.make_oracle(model, data, kernel.NumericalPolicy())
    route = "original_ODE" if nonlinear else "certified_affine"
    assert original.portfolio_audit["route"] == route
    assert reduced.portfolio_audit["forcing_symbols"] == ["w"]
    assert len(reduced.names) == len(original.names) - 1
    assert "g" not in reduced.names
    assert len(original.roots) == 1
    assert sum(n.startswith("trajectory:") for n in reduced.names) == 2
    start = continuation.to_effective(original, reduced, original.start)
    lifted = continuation.to_original(original, reduced, start)
    assert lifted["g"] == 1.0
    point = reduced.vector(start)
    residual, _ = reduced.evaluate(point, monotonic() + 10, jacobian=False)
    second, derivative = reduced.evaluate(point, monotonic() + 10)
    assert np.array_equal(residual, second)
    raw, raw_j = original.evaluate(original.vector(lifted), monotonic() + 10)
    assert residual == pytest.approx(raw, abs=1e-7)
    assert derivative == pytest.approx(
        raw_j @ reduced.reduction.lift_jacobian(point), abs=1e-7
    )
    for index in range(len(point)):
        h = np.eye(len(point))[index] * 1e-5
        plus = reduced.evaluate(point + h, monotonic() + 10, jacobian=False)[0]
        minus = reduced.evaluate(point - h, monotonic() + 10, jacobian=False)[0]
        assert derivative[:, index] == pytest.approx((plus - minus) / 2e-5, abs=2e-6)
    prefix = kernel.make_prefix(
        data.rows[0], PrefixPolicy(duration=0.5, future_auxiliaries=("w",))
    )
    kernel._validate_future(original.system, prefix)
    missing = kernel.make_prefix(data.rows[0], PrefixPolicy(duration=0.5))
    with pytest.raises(ValueError, match="future target/auxiliary"):
        kernel._validate_future(original.system, missing)
    verified = p.verify_endpoint(original, reduced, start, 10, lambda _: None)
    assert verified["original_equations_verified"]
    assert verified["methods"] == [route, "original_Radau"]
    assert before == (model.model_dump_json(), data.model_dump_json())


def test_wrong_split_and_non_eligibility_errors_are_not_nonlinear_fallback(monkeypatch):
    model, data = composite_case(True)
    with pytest.raises(ValueError, match="training only"):
        p.make_oracle(
            model,
            data.model_copy(update={"name": "validation"}),
            kernel.NumericalPolicy(),
        )

    def broken(*args):
        raise ValueError("certificate implementation failure")

    monkeypatch.setattr(p.prefix_affine, "_Propagation", broken)
    with pytest.raises(ValueError, match="certificate implementation failure"):
        p.make_oracle(model, data, kernel.NumericalPolicy())


class PairingFixture:
    """Separate integrations deliberately return slightly different baselines."""

    names = ("theta",)
    units = np.ones(1)
    lower = np.array([-2.0])
    upper = np.array([2.0])
    def __init__(self, difference=0.0):
        self.start = {"theta": 0.0}
        self.original = SimpleNamespace(settings={"tolerance": 1e-9})
        self.difference = difference
        self.calls = []

    def vector(self, point):
        return np.asarray([point["theta"]])

    def parameters(self, point):
        return {"theta": float(point[0])}

    def evaluate(self, point, deadline, *, jacobian=True, method=None):
        self.calls.append((jacobian, method))
        residual = np.array([point[0] - 1, 2 * (point[0] - 1)])
        return (
            residual + (self.difference if jacobian else 0),
            np.array([[1.0], [2.0]]) if jacobian else None,
        )


def test_canonical_pairing_records_gap_and_cache_is_bounded_and_keyed():
    base = PairingFixture(difference=1e-9)
    oracle = p.PortfolioOracle(base, {"fixture": True})
    point = np.array([0.0])
    residual = oracle.evaluate(point, monotonic() + 10, jacobian=False)[0]
    paired = oracle.evaluate(point, monotonic() + 10)[0]
    assert np.array_equal(residual, paired)
    assert base.calls == [(False, None), (True, None)]
    assert oracle.execution_receipt()["pairing_checks"] == 1
    assert oracle.maximum_pairing_discrepancy == pytest.approx(1e-9)
    oracle.evaluate(point, monotonic() + 10, jacobian=False, method="Radau")
    base.original.settings["tolerance"] = 1e-11
    oracle.evaluate(point, monotonic() + 10, jacobian=False)
    assert base.calls[-2:] == [(False, "Radau"), (False, None)]
    for value in np.linspace(-1, 1, p.CACHE_POINTS + 3):
        oracle.evaluate(np.array([value]), monotonic() + 10, jacobian=False)
    assert len(oracle._cache) == p.CACHE_POINTS
    with pytest.raises(TimeoutError):
        oracle.evaluate(np.array([1.0]), monotonic() - 1, jacobian=False)


def test_pairing_disagreement_is_not_hidden_by_cached_residual():
    oracle = p.PortfolioOracle(PairingFixture(difference=1e-3), {})
    point = np.array([0.0])
    baseline = oracle.evaluate(point, monotonic() + 10, jacobian=False)[0]
    with pytest.raises(NumericalInvalidTrial, match="pairing policy"):
        oracle.evaluate(point, monotonic() + 10)
    assert oracle.execution_receipt()["pairing_failures"] == 1
    assert np.array_equal(
        baseline, oracle.evaluate(point, monotonic() + 10, jacobian=False)[0]
    )


def test_lazy_search_qualifies_synthetic_fit_and_records_distinct_costs():
    original, oracle = p.make_oracle(problem(), training(), kernel.NumericalPolicy())
    start = continuation.to_effective(original, oracle, original.start)
    qualified = p.qualification(original, oracle, start, 10, lambda _: None)
    assert qualified["passed"], qualified
    checkpoints = []
    searched = p.search(oracle, start, 15, 100, checkpoints.append)
    assert searched["best"]["training_nmse"] < 1e-8
    assert searched["budget_exhausted"] is False
    assert searched["lazy_jacobian"]
    assert searched["jacobian_calls"] > 0
    assert searched["residual_calls"] > 0
    assert not searched["global_minimum_certified"]
    assert searched["accepted_history_protocol"] == "full-coordinate-accepted-steps-1"
    assert checkpoints
    endpoint = searched["best"]["parameters"]
    verification = p.verify_endpoint(original, oracle, endpoint, 10, lambda _: None)
    assert verification["original_equations_verified"]
    mismatched_history = deepcopy(searched)
    mismatched_history["last_accepted_parameters"] = start
    diagnosis = p.diagnose(
        original, oracle, endpoint, verification, mismatched_history, 10, lambda _: None
    )
    assert not diagnosis["endpoint_matches_accepted_history"]
    assert diagnosis["assessment"]["classification"] == "optimization_unresolved"


def test_failed_initial_jacobian_retains_observed_residual_without_confidence():
    oracle = p.PortfolioOracle(PairingFixture(difference=1e-3), {})
    searched = p.search(oracle, oracle.start, 10, 10, lambda _: None)
    assert searched["best"]["parameters"] == oracle.start
    assert searched["stop_reason"] == "numerical_failure"
    assert searched["budget_exhausted"] is False
    assert not searched["global_minimum_certified"]
    assert searched["failed_jacobian_calls"] == 1


def test_failed_checkpoint_prevents_more_numerical_work():
    base = PairingFixture()
    oracle = p.PortfolioOracle(base, {})

    def failed(_):
        raise OSError("disk unavailable")

    result = p.search(oracle, oracle.start, 10, 10, failed)
    assert result["status"] == "incomplete"
    assert not base.calls


@pytest.mark.parametrize(
    ("receipt", "expected"),
    [
        ({"status": "complete", "stop_reason": "step_or_gradient_converged"}, False),
        ({"status": "complete", "stop_reason": "budget_limited"}, True),
        ({"stop_reason": "step_or_gradient_converged"}, None),
        ({"status": "incomplete", "stop_reason": "step_or_gradient_converged"}, None),
        ({"status": "complete", "stop_reason": "interrupted"}, None),
        ({"status": "complete", "stop_reason": "unknown"}, None),
        ({"budget_exhausted": "false"}, None),
        ({"budget_exhausted": True}, True),
        ({"budget_exhausted": False}, False),
    ],
)
def test_search_budget_evidence_preserves_unknown(receipt, expected):
    before = deepcopy(receipt)
    assert p.budget_status(receipt) is expected
    assert receipt == before


@pytest.mark.parametrize(
    ("stop", "expected"),
    [
        ("step_or_gradient_converged", False),
        ("numerical_target_reached", False),
        ("budget_limited", True),
        ("numerical_failure", False),
    ],
)
def test_every_new_terminal_search_records_budget_status(monkeypatch, stop, expected):
    monkeypatch.setattr(
        p.recovery_numerics,
        "search",
        lambda *args, **kwargs: {"stop_reason": stop},
    )
    oracle = p.PortfolioOracle(PairingFixture(), {})
    result = p.search(oracle, oracle.start, 5, 5, lambda _: None)
    assert result["budget_exhausted"] is expected
