"""Small feasible probes give bounded evidence, never convergence by fiat."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from autoformalism.fitting import prefix_portfolio_root_profile as roots
from autoformalism.fitting import profile_endpoint_probe as probe
from autoformalism.fitting.recovery_numerics import NumericalInvalidTrial
from tests.test_prefix_meal_coordinates import certified, fixture


class Oracle:
    names = ("a", "b", "root")
    units = np.ones(3)
    lower = np.array([0.0, -np.inf, 0.0])
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


def factory(loss, *, fail=None):
    class Profiler:
        inner = (2,)
        outer = (0, 1)

        def __init__(self, oracle):
            self.oracle = oracle
            self.audit = {"sha256": "synthetic"}

        def profile(self, values, deadline):
            del deadline
            self.oracle.vector(values)
            if fail and fail(values):
                raise NumericalInvalidTrial("synthetic pairing failure")
            return {
                "training_nmse": float(loss(values)),
                "parameters": dict(values) | {"root": 0.0},
            }

    return Profiler


def run(loss, *, start=None, **kwargs):
    records = []
    start = start or {"a": 0.0, "b": 0.0, "root": 1.0}
    result = probe.profile_endpoint_probe(
        Oracle(),
        start,
        10,
        records.append,
        profiler_factory=factory(loss),
        **kwargs,
    )
    return result, records


def test_feasible_descent_is_resolved_without_overclaiming_or_mutating_start():
    start = {"a": 0.0, "b": 0.0, "root": 1.0}
    before = deepcopy(start)
    result, records = run(lambda p: (p["a"] - 1) ** 2 + p["b"] ** 2, start=start)
    assert result["complete"]
    assert result["status"] == "feasible_descent_detected"
    assert result["calls_completed"] == result["planned_calls"] == 7
    assert result["coordinates"][0]["directions"][0]["status"] == "bound_blocked"
    assert result["coordinates"][0]["directions"][1]["status"] == "descent_detected"
    assert result["best_probe"]["training_nmse"] < 1
    assert not result["stationarity_certified"]
    assert not result["parameter_confidence_certified"]
    assert start == before
    assert records


def test_correct_boundary_and_flat_directions_do_not_claim_stationarity():
    result, _ = run(lambda p: 1 + p["a"])
    assert result["status"] == "no_resolved_coordinate_descent"
    assert not result["stationarity_certified"]


def test_profiled_and_fixed_coordinates_are_not_probed():
    result, _ = run(lambda p: (p["a"] - 1) ** 2, fixed={"a": 0.0})
    assert [row["name"] for row in result["coordinates"]] == ["b"]
    assert result["planned_calls"] == 5
    assert result["inner_coordinates"] == ["root"]


@pytest.mark.parametrize("fixed", [{"root": 1.0}, {"unknown": 0}, {"a": 1.0}])
def test_invalid_held_coordinate_is_incomplete_not_silently_overwritten(fixed):
    result, _ = run(lambda p: 1, fixed=fixed)
    assert not result["complete"]
    assert result["status"] == "incomplete"
    assert result["calls_started"] == 0


def test_step_dependence_and_pairing_failure_are_not_non_descent_certificates():
    result, _ = run(lambda p: 1 + 100 * p["a"] ** 2)
    assert result["status"] == "step_resolution_limited"
    failed = probe.profile_endpoint_probe(
        Oracle(),
        {"a": 0.0, "b": 0.0, "root": 1.0},
        10,
        lambda _: None,
        profiler_factory=factory(lambda p: 1, fail=lambda p: p["a"] > 0),
    )
    assert failed["status"] == "numerically_unresolved"
    assert failed["calls_failed"] == 1


def test_incomplete_work_keeps_prior_points_and_cost(monkeypatch):
    original = factory(lambda p: (p["a"] - 1) ** 2)
    calls = []

    class Interrupted(original):
        def profile(self, values, deadline):
            calls.append(values)
            if len(calls) == 3:
                raise TimeoutError("test deadline")
            return super().profile(values, deadline)

    result = probe.profile_endpoint_probe(
        Oracle(),
        {"a": 0.0, "b": 0.0, "root": 1.0},
        10,
        lambda _: None,
        profiler_factory=Interrupted,
    )
    assert result["status"] == "incomplete"
    assert result["calls_completed"] == 2
    assert result["calls_failed"] == 1
    assert result["best_probe"]["training_nmse"] < 1
    assert "timing" in result


def test_finite_upper_bound_and_nonrepresentable_steps_are_respected():
    oracle = Oracle()
    oracle.upper = np.array([0.0001, np.inf, np.inf])
    result = probe.profile_endpoint_probe(
        oracle,
        {"a": 0.0, "b": 0.0, "root": 1.0},
        10,
        lambda _: None,
        profiler_factory=factory(lambda p: (1 - p["a"]) ** 2),
    )
    samples = result["coordinates"][0]["directions"][1]["samples"]
    assert [s["distance"] for s in samples] == [0.0001, 0.00005]


def test_real_profile_and_positive_meal_chart_pair_with_original_model():
    _, oracle = fixture()
    chart = certified(oracle)
    result = probe.profile_endpoint_probe(
        oracle,
        oracle.start,
        10,
        lambda _: None,
        profiler_factory=roots.RootProfile,
        coordinate_chart=chart,
    )
    assert result["complete"]
    assert result["calls_failed"] == 0
    by_name = {c["name"]: c for c in result["coordinates"]}
    assert by_name[chart.rate]["coordinate"] == "log_ratio"
    assert all(not c["name"].startswith("trajectory:") for c in result["coordinates"])
    assert not result["stationarity_certified"]


def test_failed_persistence_stops_further_numerics():
    records = []

    def callback(value):
        records.append(value)
        if len(records) == 2:
            raise OSError("disk unavailable")

    result = probe.profile_endpoint_probe(
        Oracle(),
        {"a": 0.0, "b": 0.0, "root": 1.0},
        10,
        callback,
        profiler_factory=factory(lambda _: 1),
    )
    assert result["status"] == "incomplete"
    assert result["calls_started"] == 1
    assert len(records) == 2


def test_expiry_before_profile_call_is_attempted_but_not_started(monkeypatch):
    budget = probe.execution._budget(10, 130)
    original = budget.numerical
    invocations = []

    def numerical(function):
        invocations.append(function)
        if len(invocations) == 2:
            raise TimeoutError("deadline before profile lambda")
        return original(function)

    monkeypatch.setattr(budget, "numerical", numerical)
    monkeypatch.setattr(probe.execution, "_budget", lambda *_: budget)
    result, _ = run(lambda _: 1)
    assert result["status"] == "incomplete"
    assert result["calls_attempted"] == 1
    assert (
        result["calls_started"]
        == result["calls_completed"]
        == result["calls_failed"]
        == 0
    )
