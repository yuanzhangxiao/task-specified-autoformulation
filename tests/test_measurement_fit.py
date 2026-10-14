"""Weighted retention, solver checks, immutable resume and shared allowances."""

from copy import deepcopy
from time import monotonic
from types import SimpleNamespace

import numpy as np
import pytest

from autoformalism.benchmarks.audited_release import read_seal
from autoformalism.fitting import measurement_fit as fit
from autoformalism.schemas.measurement_fitting import MeasurementProblem
from tests.test_measurement_contract import measurement_payload


@pytest.fixture(autouse=True)
def stable_source(monkeypatch):
    monkeypatch.setattr(
        fit.public, "_source_identity", lambda: "measurement-test-source"
    )


@pytest.fixture
def problem():
    payload = measurement_payload()
    candidate = payload["request"]["base_candidate"]
    candidate["state_equations"][0]["rhs"] = "-0.8*x+rate*u"
    candidate["parameters"][0]["bounds"] = {"lower": 0, "upper": 5}
    return MeasurementProblem.model_validate(payload)


@pytest.fixture
def policy():
    return fit.Policy(
        restart_count=0, search_seconds=5, verification_seconds=5, maximum_calls=30
    )


@pytest.mark.parametrize("strategy", ["direct", "profiled"])
def test_real_weighted_fit_matches_analytic_optimum(
    problem, policy, tmp_path, strategy
):
    result = fit.run(problem, policy, tmp_path, strategy=strategy)
    decay = np.exp(-0.8 * np.arange(3))
    design, offset = 2 / 0.8 * (1 - decay), 4 * decay - [3, 5, 7]
    expected = np.clip(-design @ offset / (design @ design), 0, 5)
    loss = np.sum(((design * expected + offset) / 0.2) ** 2) / 2
    assert result["status"] == "complete"
    assert result["parameters"]["rate"] == pytest.approx(expected, abs=1e-7)
    assert result["objective_half_sum_squares"] == pytest.approx(loss, abs=1e-6)
    assert result["verification"]["passed"]
    assert result["negative_log_likelihood"] == pytest.approx(
        loss + 3 * np.log(0.2) + 1.5 * np.log(2 * np.pi), abs=1e-6
    )
    assert result["confidence_probability"] is None
    assert (
        result["data_fault_verdict"]
        == result["specification_verdict"]
        == "not_assessed"
    )
    assert result["cost"]["search_calls_started"] <= policy.maximum_calls
    assert result["search"]["stop_reason"] != "numerical_target_reached"
    if strategy == "profiled":
        assert result["search"]["stop_reason"] == "bounded_affine_problem_solved"
        assert result["cost"]["search_calls_started"] == 1
    audit = result["residual_evidence"]["reading_audit"]
    assert sum(row["quadratic_contribution"] for row in audit) == pytest.approx(loss)
    assert audit[0]["raw_value"] == 1
    assert audit[0]["prediction_analysis"] == pytest.approx(4)
    assert audit[0]["standardized_residual"] == pytest.approx(5)


def test_completed_resume_is_noop_and_changed_sigma_rejected(
    problem, policy, tmp_path, monkeypatch
):
    first = fit.run(problem, policy, tmp_path)
    monkeypatch.setattr(fit.rollout, "build", lambda *a: pytest.fail("repeat setup"))
    assert fit.run(problem, policy, tmp_path) == first
    changed = problem.model_dump(mode="json")
    changed["manifest"]["noise"]["sigma"][0] *= 2
    with pytest.raises(ValueError, match="differs"):
        fit.run(MeasurementProblem.model_validate(changed), policy, tmp_path)


def test_unverified_improvement_cannot_displace_verified_incumbent(
    problem, policy, tmp_path, monkeypatch
):
    verify = fit._verify

    def fail_child(oracle, parameters, *args):
        result = verify(oracle, parameters, *args)
        if parameters != problem.incumbent:
            result["passed"] = False
        return result

    monkeypatch.setattr(fit, "_verify", fail_child)
    result = fit.run(problem, policy, tmp_path)
    assert result["parameters"] == problem.incumbent
    assert result["selected_origin"] == "incumbent"
    assert result["search"]["attempts"][0]["best"]["parameters"] != problem.incumbent


def test_profile_failure_falls_back_with_remaining_call_and_time_budget(
    problem, policy, tmp_path, monkeypatch
):
    build = fit.rollout.build
    deadlines = []

    def broken_profile(vector, deadline):
        deadlines.append(deadline)
        raise ValueError("rank-deficient weighted profile")

    def override(*args):
        oracle, _, route = build(*args)
        return oracle, SimpleNamespace(outer=(), evaluate=broken_profile), route

    search = fit.numerical.search

    def record(*args, **kwargs):
        assert args[2] == deadlines[0]
        assert kwargs["calls"] == policy.maximum_calls - 1
        return search(*args, **kwargs)

    monkeypatch.setattr(fit.rollout, "build", override)
    monkeypatch.setattr(fit.numerical, "search", record)
    result = fit.run(problem, policy, tmp_path)
    attempt = result["search"]["attempts"][0]
    assert attempt["profile_failures"] == ["rank-deficient weighted profile"]
    assert result["parameters"]["rate"] > 1
    assert result["cost"]["search_calls_started"] <= policy.maximum_calls


def test_interrupted_search_never_restarts_or_loses_verified_point(
    problem, policy, tmp_path, monkeypatch
):
    original = fit._search

    def interrupt(oracle, profile, start, policy, calls, deadline, save):
        save({"calls": 1, "completed_calls": 0, "best": {"parameters": dict(start)}})
        raise KeyboardInterrupt

    monkeypatch.setattr(fit, "_search", interrupt)
    with pytest.raises(KeyboardInterrupt):
        fit.run(problem, policy, tmp_path)
    monkeypatch.setattr(fit, "_search", lambda *a: pytest.fail("renewed search"))
    result = fit.run(problem, policy, tmp_path)
    assert result["parameters"] == problem.incumbent
    assert not result["cost"]["accounting_complete"]
    assert result["cost"]["search_calls_started"] is None
    assert result["cost"]["budget_charge_seconds"] >= policy.search_seconds
    assert result["search"]["stop_reason"] == "interrupted"
    assert read_seal(tmp_path / "search-0/result.json")["budget_restarted"] is False
    monkeypatch.setattr(fit, "_search", original)
    assert fit.run(problem, policy, tmp_path) == result


def test_problem_revalidation_and_policy_limits(problem, policy, tmp_path):
    corrupted = deepcopy(problem.model_dump(mode="json"))
    corrupted["manifest"]["noise"]["sigma"][0] = 0
    with pytest.raises(ValueError):
        MeasurementProblem.model_validate(corrupted)
    with pytest.raises(ValueError, match="each frozen start"):
        fit.Policy(maximum_calls=2, restart_count=3)
    with pytest.raises(ValueError, match="strategy"):
        fit.run(problem, policy, tmp_path, strategy="huber")


@pytest.mark.parametrize("interrupted", [False, True])
def test_fallback_preserves_best_progress_and_full_profile_attempt(
    policy, monkeypatch, interrupted
):
    best = {"parameters": {"rate": 2.0}, "mean_squared_standardized_residual": 1.0}
    worse = {"parameters": {"rate": 1.0}, "mean_squared_standardized_residual": 4.0}
    saved = []

    def search(oracle, initial, deadline, save, **kwargs):
        is_profile = kwargs["profile"] is not None
        result = {
            "calls": 2,
            "completed_calls": 1,
            "best": best if is_profile else worse,
            "stop_reason": "numerical_failure" if is_profile else "budget_limited",
            "evaluation_trace": [{"audit": "profile" if is_profile else "joint"}],
        }
        save(result)
        if interrupted and not is_profile:
            raise KeyboardInterrupt
        return result

    monkeypatch.setattr(fit.numerical, "search", search)

    def run():
        return fit._search(
            None,
            SimpleNamespace(outer=(0,)),
            {},
            policy,
            8,
            monotonic() + 5,
            saved.append,
        )

    if interrupted:
        with pytest.raises(KeyboardInterrupt):
            run()
    else:
        result = run()
        assert len(result["subattempts"]) == 2
        assert result["calls"] == 4
        assert result["completed_calls"] == 2
    assert saved[-1]["best"] == best
    assert saved[-1]["subattempts"][0]["evaluation_trace"] == [{"audit": "profile"}]
    assert saved[-1]["profile_failures"]


def test_interrupted_setup_has_unknown_cost_on_resume(
    problem, policy, tmp_path, monkeypatch
):
    build = fit.rollout.build

    def interrupt(*args):
        raise KeyboardInterrupt

    monkeypatch.setattr(fit.rollout, "build", interrupt)
    with pytest.raises(KeyboardInterrupt):
        fit.run(problem, policy, tmp_path)
    monkeypatch.setattr(fit.rollout, "build", build)
    result = fit.run(problem, policy, tmp_path)
    assert result["status"] == "complete"
    assert not result["cost"]["accounting_complete"]
    assert result["cost"]["interrupted_setup_attempts"] == 1
    assert len(list((tmp_path / "setup").glob("*/started.json"))) == 2


def test_missing_completion_counts_cannot_claim_complete_accounting():
    cost = fit._cost(
        [
            {
                "stage": "search",
                "accounting_complete": True,
                "wall_seconds": 1,
                "cpu_seconds": 1,
                "calls": 2,
                "budget_charge_seconds": 1,
                "value": None,
            }
        ],
        {"wall": 1, "cpu": 1, "complete": True, "interrupted": 0},
    )
    assert cost["search_calls_started"] == 2
    assert cost["search_calls_completed"] is None
    assert not cost["accounting_complete"]
