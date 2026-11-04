"""Optimizer telemetry must refer to a finite evaluated root-profile point."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from autoformalism.fitting import prefix_portfolio_root_profile as roots
from tests.test_prefix_meal_coordinates import certified, fixture
from tests.test_prefix_portfolio_timescale import wrapped


def returned(q, loss, *, success=True):
    return SimpleNamespace(
        success=success, message="synthetic termination", nit=1, x=q.copy(), fun=loss
    )


def test_finite_evaluated_terminal_is_qualified_without_extra_calls(monkeypatch):
    _, oracle = wrapped()
    calls_at_return = []

    def fit(objective, q, *, callback, **kwargs):
        loss = objective(q)
        callback(q)
        calls_at_return.append(loss)
        return returned(q, loss)

    monkeypatch.setattr(roots, "minimize", fit)
    result = roots.root_profile_search(oracle, oracle.start, 10, 100, lambda _: None)
    terminal = result["optimizer_terminal"]
    assert result["optimizer_raw_success"] and result["optimizer_success"]
    assert terminal["status"] == "verified"
    assert terminal["reported_training_nmse"] == calls_at_return[0]
    assert terminal["evaluation_attempt"] == result["search_calls_completed"]
    assert terminal["extra_profile_calls"] == 0
    assert not terminal["stationarity_certified"]
    assert len(result["accepted_shared_iteration_history"]) == 1
    assert result["optimizer_callback_audit"]["verified"] == 1


@pytest.mark.parametrize("reported", [np.inf, np.nan, None])
def test_nonfinite_optimizer_success_does_not_qualify(monkeypatch, reported):
    _, oracle = wrapped()

    def fit(objective, q, **kwargs):
        objective(q)
        return returned(q, reported)

    monkeypatch.setattr(roots, "minimize", fit)
    result = roots.root_profile_search(oracle, oracle.start, 10, 100, lambda _: None)
    assert result["optimizer_raw_success"] and not result["optimizer_success"]
    assert result["stop_reason"] == "optimizer_terminal_unqualified"
    assert result["optimizer_terminal"]["status"] == "missing_or_nonfinite_objective"
    assert result["optimizer_terminal"]["reported_training_nmse"] is None
    assert np.isfinite(result["best"]["training_nmse"])
    json.dumps(result, allow_nan=False)


def test_finite_objective_must_match_evaluation_receipt(monkeypatch):
    _, oracle = wrapped()

    def fit(objective, q, **kwargs):
        return returned(q, objective(q) + 1)

    monkeypatch.setattr(roots, "minimize", fit)
    result = roots.root_profile_search(oracle, oracle.start, 10, 100, lambda _: None)
    assert result["optimizer_terminal"]["status"] == "objective_receipt_mismatch"
    assert not result["optimizer_success"]


def test_unevaluated_callback_and_terminal_cannot_be_accepted(monkeypatch):
    _, oracle = wrapped()

    def fit(objective, q, *, callback, **kwargs):
        trial = q + 0.123456789
        callback(trial)
        return returned(trial, 0.0)

    monkeypatch.setattr(roots, "minimize", fit)
    result = roots.root_profile_search(oracle, oracle.start, 10, 100, lambda _: None)
    assert result["optimizer_terminal"]["status"] == "not_evaluated"
    assert not result["optimizer_success"]
    assert result["accepted_shared_iteration_history"] == []
    assert result["optimizer_callback_audit"]["unqualified"] == 1
    assert result["search_calls_completed"] == 1 + 2 * len(result["outer_coordinates"])


def test_invalid_chart_callback_preserves_best_and_does_not_raise(monkeypatch):
    _, oracle = fixture()
    chart = certified(oracle)

    def fit(objective, q, *, callback, **kwargs):
        trial = q.copy()
        trial[0] = -1e9
        assert np.isinf(objective(trial))
        for _ in range(12):
            callback(trial)
        return returned(trial, np.inf)

    monkeypatch.setattr(roots, "minimize", fit)
    result = roots.root_profile_search(
        oracle, oracle.start, 10, 100, lambda _: None, coordinate_chart=chart
    )
    assert result["optimizer_raw_success"] and not result["optimizer_success"]
    assert result["accepted_shared_iteration_history"] == []
    callbacks = result["optimizer_callback_audit"]
    assert callbacks["unqualified"] == 12
    assert len(callbacks["unqualified_examples"]) == 8
    assert callbacks["omitted_examples"] == 4
    assert callbacks["unqualified_examples"][0]["status"] == "invalid_coordinates"
    assert (
        result["best"]["training_nmse"]
        <= result["unoptimized_control"]["training_nmse"]
    )
    assert result["invalid_trials"] == 1
    json.dumps(result, allow_nan=False)


def test_rejected_evaluation_supersedes_earlier_completed_point(monkeypatch):
    _, oracle = wrapped()
    original = roots.RootProfile.profile

    def fit(objective, q, *, callback, **kwargs):
        loss = objective(q)

        def rejected(*args, **kw):
            raise roots.NumericalInvalidTrial("injected precision failure")

        monkeypatch.setattr(roots.RootProfile, "profile", rejected)
        assert np.isinf(objective(q))
        callback(q)
        return returned(q, loss)

    monkeypatch.setattr(roots.RootProfile, "profile", original)
    monkeypatch.setattr(roots, "minimize", fit)
    result = roots.root_profile_search(oracle, oracle.start, 10, 100, lambda _: None)
    assert not result["optimizer_success"]
    assert result["optimizer_terminal"]["status"] == "rejected_evaluation"
    assert result["accepted_shared_iteration_history"] == []
    assert result["best"]["parameters"]


def test_budget_exhaustion_does_not_gain_a_terminal_evaluation(monkeypatch):
    _, oracle = wrapped()

    def unexpected(*args, **kwargs):
        pytest.fail("optimizer must not run after exhausted initial sweep")

    monkeypatch.setattr(roots, "minimize", unexpected)
    result = roots.root_profile_search(oracle, oracle.start, 10, 2, lambda _: None)
    assert result["search_calls_started"] == result["search_calls_completed"] == 2
    assert not result["optimizer_success"] and result["optimizer_raw_success"] is None
    assert result["optimizer_terminal"] is None
    assert result["budget_exhausted"]


def test_all_fixed_control_reports_evaluation_without_optimizer_call(monkeypatch):
    _, oracle = wrapped()
    profiler = roots.RootProfile(oracle)
    fixed = {oracle.names[i]: oracle.start[oracle.names[i]] for i in profiler.outer}
    monkeypatch.setattr(
        roots, "minimize", lambda *a, **kw: pytest.fail("no free coordinates")
    )
    result = roots.root_profile_search(
        oracle, oracle.start, 10, 2, lambda _: None, fixed=fixed
    )
    assert result["optimizer_raw_success"] is None
    assert result["optimizer_terminal"]["feasible_evaluation_verified"]
    assert result["search_calls_started"] == 1


def test_feasible_terminal_does_not_override_unsuccessful_optimizer(monkeypatch):
    _, oracle = wrapped()

    def fit(objective, q, **kwargs):
        return returned(q, objective(q), success=False)

    monkeypatch.setattr(roots, "minimize", fit)
    result = roots.root_profile_search(oracle, oracle.start, 10, 100, lambda _: None)
    assert result["optimizer_terminal"]["feasible_evaluation_verified"]
    assert not result["optimizer_raw_success"] and not result["optimizer_success"]


def test_nonfinite_profile_loss_never_becomes_a_completed_receipt(monkeypatch):
    _, oracle = wrapped()
    real = roots.RootProfile.profile

    def fit(objective, q, *, callback, **kwargs):
        monkeypatch.setattr(
            roots.RootProfile, "profile",
            lambda *a, **kw: {"training_nmse": np.inf},
        )
        assert np.isinf(objective(q))
        callback(q)
        return returned(q, 0.0)

    monkeypatch.setattr(roots.RootProfile, "profile", real)
    monkeypatch.setattr(roots, "minimize", fit)
    result = roots.root_profile_search(oracle, oracle.start, 10, 100, lambda _: None)
    assert result["invalid_trials"] == result["search_calls_failed"] == 1
    assert result["optimizer_terminal"]["status"] == "rejected_evaluation"
    assert not result["optimizer_success"]
    assert np.isfinite(result["best"]["training_nmse"])
    json.dumps(result, allow_nan=False)
