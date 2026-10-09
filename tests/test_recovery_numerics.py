"""Bound restoration, incumbent dominance, diverse starts and assessment gates."""

from time import monotonic
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from autoformalism.fitting import recovery_numerics as numerical
from autoformalism.fitting.fitting_assessment import FittingAssessment, assess


def point(loss, *, discrepancy=0.0, parameters=None):
    return {
        "parameters": parameters or {"a": 1.0, "initial": 0.0},
        "training_nmse": loss,
        "alternate_training_nmse": loss + discrepancy,
        "solver_loss_discrepancy": abs(discrepancy),
        "solver_prediction_discrepancy": 1e-10,
    }


def test_observed_lower_bound_roundoff_is_recorded_not_materially_clipped():
    anchor = np.array([4.0])
    lower, upper, units = np.array([0.01]), np.array([30.0]), np.ones(1)
    q = lower - anchor
    events = []
    assert (anchor + q)[0] < lower[0]
    value = numerical.restore(anchor, units, q, lower, upper, ["a4"], events)
    assert value[0] == lower[0]
    event = events[0]
    assert event["action"] == "roundoff_projected"
    assert event["violations"][0]["excess"] == pytest.approx(2.133709875e-16)
    with pytest.raises(numerical.DomainViolation):
        numerical.restore(anchor, units, q - 1e-6, lower, upper, ["a4"], events)
    assert events[-1]["action"] == "rejected"
    with pytest.raises(ValueError, match="nonfinite"):
        numerical.restore(anchor, units, np.array([np.nan]), lower, upper, ["a"], [])


def test_retention_does_not_require_stringent_uncertainty_precision():
    incumbent = point(0.0002, discrepancy=4.8e-13)
    assert numerical.usable(incumbent)
    assert not numerical.improves(point(0.0003), incumbent)
    assert not numerical.improves(incumbent, incumbent)
    assert numerical.improves(point(0.0001), incumbent)
    # Overlapping two-integrator loss intervals cannot establish improvement.
    assert not numerical.improves(point(0.0002 - 1e-13, discrepancy=8e-13), incumbent)
    assert not numerical.usable(point(0.01, discrepancy=1e-6))
    assert not numerical.usable(point(float("nan")))
    assert not numerical.improves(None, incumbent)


def test_tight_verification_restores_settings_after_failure(monkeypatch):
    class Settings:
        def model_copy(self, *, update):
            return SimpleNamespace(**update)

    settings = Settings()
    oracle = SimpleNamespace(settings=settings)

    def fail(*args):
        assert oracle.settings.relative_tolerance == 1e-12
        raise TimeoutError("unavailable")

    monkeypatch.setattr(numerical, "verify_outputs", fail)
    with pytest.raises(TimeoutError):
        numerical.check_point(oracle, {}, monotonic() + 1, lambda _: None, tight=True)
    assert oracle.settings is settings


class QuadraticOracle:
    names = ("a", "initial")
    units = np.array([2.0, 1.0])
    lower, upper = np.array([0.01, -10.0]), np.array([30.0, 10.0])

    def vector(self, parameters):
        v = np.array([parameters[n] for n in self.names])
        if np.any(v < self.lower) or np.any(v > self.upper):
            raise ValueError("initial vector outside domain")
        return v

    def parameters(self, vector):
        return dict(zip(self.names, vector.tolist(), strict=True))

    def evaluate(self, vector, deadline):
        return vector - [2, 3], np.eye(2)


def test_search_target_budget_initial_domain_and_failed_residual(monkeypatch):
    oracle, progress = QuadraticOracle(), []
    start = {"a": 1.0, "initial": 0.0}
    r = numerical.search(
        oracle, start, monotonic() + 5, progress.append, calls=100, target=1e-20
    )
    assert r["stop_reason"] == "numerical_target_reached"
    assert r["best"]["parameters"] == pytest.approx({"a": 2, "initial": 3})
    assert r["calls"] > 1 and progress[-1]["best"] == r["best"]
    short = numerical.search(
        oracle, start, monotonic() + 5, lambda _: None, calls=2, target=1e-20
    )
    assert short["stop_reason"] == "budget_limited" and short["calls"] == 2
    with pytest.raises(ValueError, match="initial vector"):
        numerical.search(
            oracle,
            start | {"a": -1.0},
            monotonic() + 5,
            lambda _: None,
            calls=2,
            target=1e-20,
        )
    monkeypatch.setattr(oracle, "evaluate", lambda *a: (np.array([np.nan]), np.eye(2)))
    failure = numerical.search(
        oracle, start, monotonic() + 5, lambda _: None, calls=2, target=1e-20
    )
    assert failure["stop_reason"] == "numerical_failure" and failure["best"] is None


def test_profiled_search_returns_inner_solution():
    oracle = QuadraticOracle()

    class Profile:
        outer = (0,)

        def evaluate(self, vector, deadline):
            fitted = vector.copy()
            fitted[1] = 3.0
            return fitted, fitted - [2, 3], np.array([[1.0], [0.0]]), {"inner": True}

    r = numerical.search(
        oracle,
        {"a": 1.0, "initial": 0.0},
        monotonic() + 5,
        lambda _: None,
        calls=100,
        target=1e-20,
        profile=Profile(),
    )
    assert r["best"]["parameters"] == pytest.approx({"a": 2, "initial": 3})
    assert r["projection"] == {"inner": True}


def test_additional_accuracy_gate_can_prevent_average_loss_early_stop():
    r = numerical.search(
        QuadraticOracle(),
        {"a": 1.0, "initial": 0.0},
        monotonic() + 5,
        lambda _: None,
        calls=2,
        target=100.0,
        acceptable=lambda _: False,
    )
    assert r["stop_reason"] == "budget_limited" and r["calls"] == 2


def test_portfolio_reproducible_distinct_and_bounded():
    oracle = QuadraticOracle()
    problem = SimpleNamespace(start={"a": 1.0, "initial": 0.0})
    starts = numerical.diverse_starts(oracle, problem, 3, 19)
    assert starts == numerical.diverse_starts(oracle, problem, 3, 19)
    assert starts != numerical.diverse_starts(oracle, problem, 3, 20)
    assert len({tuple(p.values()) for p in starts}) == 3
    assert all(p != problem.start for p in starts)
    assert len({int(3 * np.log(p["a"] / 0.01) / np.log(3000)) for p in starts}) == 3
    assert all(-2 <= p["initial"] <= 2 for p in starts)
    assert numerical.diverse_starts(oracle, problem, 0, 0) == []
    oracle.lower, oracle.upper = np.array([0.01, -np.inf]), np.array([30, np.inf])
    unbounded = numerical.diverse_starts(oracle, problem, 3, 19)
    assert unbounded == starts
    oracle.lower = np.array([np.nan, -10])
    with pytest.raises(ValueError, match="ordered declared"):
        numerical.diverse_starts(oracle, problem, 3, 19)


def test_assessment_separates_information_from_fit_and_search():
    sensitivity = {
        "parameters": ["a", "initial"],
        "units": [1, 1],
        "rank": 2,
        "parameter_count": 2,
        "condition_number": 2e8,
    }
    good = point(1e-22)
    a = assess(good, sensitivity, [], [], 1e-20)
    assert a["fit_quality"] == "numerical_target_reached"
    assert a["parameter_information"] == "weak_sensitivity"
    assert a["recommended_action"] == "assess_parameter_uncertainty"
    assert a["probability"] is None and not a["global_identifiability_certified"]
    alternatives = [point(1e-15, parameters={"a": 1.2, "initial": 0.0})]
    a = assess(good, sensitivity, alternatives, [], 1e-20)
    assert a["parameter_information"] == "alternative_vectors_found"
    assert a["alternative_witnesses"][0]["parameter"] == "a"
    poor = assess(point(0.1), sensitivity, [], ["budget_limited"], 1e-20)
    assert poor["recommended_action"] == "diversify_or_extend_search"
    loose = assess(point(0.1, discrepancy=4.8e-13), sensitivity, [], [], 1e-20)
    assert loose["recommended_action"] == "verify_numerics"
    assert loose["fit_quality"] == "above_numerical_target"
    assert loose["parameter_information"] == "unavailable"
    missing = assess(None, None, [], [], 1e-20)
    assert missing["search_status"] == "verification_needed"
    resolved = assess(good, sensitivity | {"condition_number": 10}, [], [], 1e-20)
    assert resolved["recommended_action"] == "retain_with_local_evidence"
    with pytest.raises(ValidationError):
        FittingAssessment.model_validate(a | {"coefficient_error": 0.0})
