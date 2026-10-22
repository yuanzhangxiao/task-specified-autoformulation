"""Terminal checkpoint costs and expired continuation results remain explicit."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("casadi")

from autoformalism.fitting import prefix_continuation as c
from autoformalism.fitting import prefix_verification as v


class Clock:
    """Advance wall and CPU time without sleeping or numerical integration."""

    wall = 100.0
    cpu = 10.0

    def advance(self, wall, cpu=0.0):
        self.wall += wall
        self.cpu += cpu


class Oracle:
    """One deterministic residual used by the real stage and derivative timers."""

    names = ("a",)

    def __init__(self, clock):
        self.lower, self.upper, self.units = (
            np.array([-np.inf]),
            np.array([np.inf]),
            np.ones(1),
        )
        self.start = {"a": 0.5}
        self.affine_audit = {"eligible": True}
        self.system = SimpleNamespace(channels=("y",))
        self.clock, self.calls, self.elapsed = clock, 0, 0.05

    def vector(self, values):
        return np.array([values["a"]])

    def parameters(self, vector):
        return {"a": float(vector[0])}

    def evaluate(self, vector, deadline, *, jacobian=True, method=None):
        self.calls += 1
        self.clock.advance(self.elapsed, 0.01)
        return vector.copy(), np.ones((1, 1)) if jacobian else None


@pytest.fixture
def context(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(v, "monotonic", lambda: clock.wall)
    monkeypatch.setattr(v, "process_time", lambda: clock.cpu)
    monkeypatch.setattr(c.adequacy, "monotonic", lambda: clock.wall)

    def search(oracle, start, deadline, checkpoint, **kwargs):
        vector = oracle.vector(start)
        residual, _ = oracle.evaluate(vector, deadline)
        result = {
            "best": {
                "parameters": deepcopy(start),
                "training_nmse": float(residual @ residual),
            },
            "calls": 1,
            "stop_reason": "gradient_converged",
            "accepted_steps": [],
        }
        checkpoint(result)
        return result

    monkeypatch.setattr(c.recovery_numerics, "search", search)
    return clock, Oracle(clock)


def verification():
    """A verified vector matching the deterministic residual."""
    return {
        "reduced_vector": [0.5],
        "training_nmse": 0.25,
        "alternate_training_nmse": 0.25,
        "solver_loss_discrepancy": 0.0,
        "solver_prediction_discrepancy": 0.0,
        "original_equations_verified": True,
    }


def run(kind, oracle, checkpoint, search_result=None):
    """Exercise each public stage with no fitter setup or remote calls."""
    if kind == "search":
        return c.search(oracle, oracle.start, 1, 2, checkpoint)
    if kind == "verification":
        return c.verify_endpoint(oracle, oracle, oracle.start, 1, checkpoint)
    if kind == "qualification":
        return c.qualification(oracle, oracle, oracle.start, 1, checkpoint)
    return c.diagnose(
        oracle,
        oracle.start,
        verification(),
        search_result or {"stop_reason": "gradient_converged", "accepted_steps": []},
        1,
        checkpoint,
    )


def final(kind, snapshot):
    """Identify the terminal callback, independently of progress callback count."""
    if kind == "search":
        return "optimizer_start" in snapshot
    if kind == "verification":
        return "verification_calls_completed" in snapshot
    return True


def incomplete(kind, result):
    """Require revoked scientific qualification while retaining useful candidates."""
    assert result["status"] == "incomplete"
    if kind == "search":
        assert result["best"]["parameters"] == {"a": 0.5}
        assert result["stop_reason"] in {
            "budget_limited",
            "numerical_failure",
            "interrupted",
        }
    elif kind == "verification":
        assert not result["original_equations_verified"]
    elif kind == "qualification":
        assert not result["passed"]
    else:
        assert result["assessment"] is None


@pytest.mark.parametrize(
    "kind", ["search", "verification", "qualification", "diagnosis"]
)
def test_terminal_write_is_measured_and_excluded_only_from_active_time(context, kind):
    clock, oracle = context
    saved = []

    def checkpoint(snapshot):
        saved.append(deepcopy(snapshot))
        clock.advance(2, 0.2)

    result = run(kind, oracle, checkpoint)
    timing = result["timing"]
    assert result["status"] == "complete"
    assert final(kind, saved[-1])
    assert timing["callback_calls"] == len(saved)
    assert timing["callback_wall_seconds"] == pytest.approx(2 * len(saved))
    assert timing["callback_cpu_seconds"] == pytest.approx(0.2 * len(saved))
    assert timing["numerical_active_wall_seconds"] == pytest.approx(0.05 * oracle.calls)
    assert timing["elapsed_wall_seconds"] == pytest.approx(
        timing["numerical_active_wall_seconds"] + timing["callback_wall_seconds"]
    )
    assert timing["elapsed_wall_seconds"] > saved[-1]["timing"]["elapsed_wall_seconds"]
    assert timing["timeout_reason"] is None


@pytest.mark.parametrize(
    "kind", ["search", "verification", "qualification", "diagnosis"]
)
@pytest.mark.parametrize("failure", ["timeout", "write_error", "interrupt"])
def test_final_write_failure_cannot_preserve_qualification(context, kind, failure):
    clock, oracle = context
    saved = []

    def checkpoint(snapshot):
        saved.append(deepcopy(snapshot))
        if final(kind, snapshot):
            clock.advance(200 if failure == "timeout" else 0.2, 0.1)
            if failure == "write_error":
                raise OSError("disk write failed")
            if failure == "interrupt":
                raise KeyboardInterrupt

    if failure == "interrupt":
        with pytest.raises(KeyboardInterrupt) as raised:
            run(kind, oracle, checkpoint)
        result = raised.value.execution_receipt
    else:
        result = run(kind, oracle, checkpoint)
    incomplete(kind, result)
    assert sum(final(kind, snapshot) for snapshot in saved) == 1
    assert result["timing"]["callback_calls"] == len(saved)
    assert result["timing"]["callback_wall_seconds"] == pytest.approx(
        200 if failure == "timeout" else 0.2
    )
    if failure == "timeout":
        assert result["timing"]["timeout_reason"] == "overall_wall_limit"
    else:
        assert result["timing"]["callback_failures"] == 1


def test_initial_verification_checkpoint_expiry_does_not_count_an_oracle_call(context):
    clock, oracle = context
    saved = []

    def checkpoint(snapshot):
        saved.append(snapshot)
        clock.advance(200)

    result = run("verification", oracle, checkpoint)
    incomplete("verification", result)
    assert result["calls_started"] == oracle.calls == 0
    assert len(saved) == 1


@pytest.mark.parametrize("kind", ["verification", "qualification", "diagnosis"])
def test_numerical_expiry_returns_measured_incomplete_receipt(context, kind):
    _, oracle = context
    oracle.elapsed = 2
    saved = []
    result = run(kind, oracle, saved.append)
    incomplete(kind, result)
    assert oracle.calls == 1
    assert result["timing"]["numerical_active_wall_seconds"] == pytest.approx(2)
    assert result["timing"]["timeout_reason"] == "numerical_active_wall_limit"
    assert final(kind, saved[-1])


def test_search_retains_last_best_when_progress_or_final_save_cannot_return(
    context, monkeypatch
):
    clock, oracle = context
    best = {"parameters": {"a": 0.25}, "training_nmse": 0.0625}

    def search(oracle, start, deadline, checkpoint, **kwargs):
        checkpoint({"best": best, "calls": 1})
        clock.advance(200)
        checkpoint({"best": best, "calls": 2})

    monkeypatch.setattr(c.recovery_numerics, "search", search)
    saved = []
    result = run("search", oracle, saved.append)
    assert result["status"] == "incomplete"
    assert result["best"] == best
    assert result["calls"] == 2
    assert result["stop_reason"] == "budget_limited"
    assert len(saved) == 1


def test_failed_progress_callback_is_not_retried(context, monkeypatch):
    _, oracle = context
    calls = []

    def checkpoint(value):
        calls.append(value)
        raise OSError("persistent write failure")

    def search(oracle, start, deadline, save, **kwargs):
        point = {"best": {"parameters": start, "training_nmse": 0.25}, "calls": 1}
        try:
            save(point)
        except RuntimeError:
            save(point)

    monkeypatch.setattr(c.recovery_numerics, "search", search)
    result = run("search", oracle, checkpoint)
    incomplete("search", result)
    assert len(calls) == 1
    assert result["timing"]["callback_failures"] == 1


@pytest.mark.parametrize(
    "search_result",
    [
        {"stop_reason": "interrupted"},
        {"stop_reason": "gradient_converged", "budget_exhausted": True},
    ],
)
def test_diagnosis_preserves_recovered_search_budget_status(
    context, monkeypatch, search_result
):
    _, oracle = context
    seen = []

    def assess(*args, **kwargs):
        seen.append(kwargs["budget_exhausted"])
        return {"classification": "optimization_unresolved"}

    monkeypatch.setattr(c.adequacy, "assess", assess)
    result = run("diagnosis", oracle, lambda _: None, search_result)
    assert seen == [True]
    assert result["assessment"]["classification"] == "optimization_unresolved"
