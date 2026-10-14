"""Opt-in robust retention, raw numerical verification and bounded accounting."""

import json
from time import monotonic
from types import SimpleNamespace

import numpy as np
import pytest

from autoformalism.benchmarks.audited_release import read_seal
from autoformalism.fitting import measurement_fit as fit
from autoformalism.fitting import measurement_fixtures as fixtures
from autoformalism.fitting import robust_measurement as robust
from autoformalism.schemas.measurement_fitting import MeasurementProblem


@pytest.fixture(autouse=True)
def stable_source(monkeypatch):
    monkeypatch.setattr(
        fit.public, "_source_identity", lambda: "robust-fit-test-source"
    )


@pytest.fixture
def policy():
    return fit.Policy(
        restart_count=0, search_seconds=8, verification_seconds=8, maximum_calls=50
    )


def problem_with_extreme_reading(case="exact_varied"):
    """Keep one declared acquisition extreme; no observation is excluded."""
    payload = fixtures.make(case, 0)[0].model_dump(mode="json")
    reading = payload["training"]["rows"][0]["readings"][12]
    assert reading["channel"] == "y"
    reading["raw_value"] += 50
    return MeasurementProblem.model_validate(payload)


def huber_sum(residual, delta=1.345):
    """Independent literal Huber definition for controller comparisons."""
    values = np.abs(residual)
    return float(
        np.where(values <= delta, values**2 / 2, delta * values - delta**2 / 2).sum()
    )


@pytest.mark.parametrize("strategy", ["direct", "profiled"])
def test_explicit_gaussian_preserves_m34_numerical_result(policy, tmp_path, strategy):
    problem = fixtures.make("exact_varied", 0)[0]
    legacy = fit.run(problem, policy, tmp_path / "legacy", strategy=strategy)
    explicit = fit.run(
        problem,
        policy,
        tmp_path / "explicit",
        strategy=strategy,
        loss=robust.Loss(kind="gaussian"),
    )
    assert explicit["parameters"] == pytest.approx(legacy["parameters"], abs=1e-8)
    assert explicit["objective_value"] == pytest.approx(
        legacy["objective_half_sum_squares"], abs=1e-8
    )
    assert explicit["negative_log_likelihood"] == pytest.approx(
        legacy["negative_log_likelihood"], abs=1e-8
    )
    assert legacy["protocol"] == "measurement-fit-result-1"
    assert "objective_definition" not in legacy["identity"]
    assert "objective_value" not in legacy
    assert explicit["identity"] != legacy["identity"]


def test_huber_retention_can_prefer_a_larger_gaussian_quadratic(
    policy, tmp_path, monkeypatch
):
    problem = fixtures.make("exact_varied", 0)[0]
    count = len(problem.manifest.noise.sigma)
    initial = np.r_[np.full(3, 4.0), np.zeros(count - 3)]
    candidate = np.r_[8.0, np.zeros(count - 1)]
    child = {"g": 1.2}

    def verify(oracle, parameters, policy, deadline, save):
        residual = initial if parameters == problem.incumbent else candidate
        save({"calls": 2, "completed_calls": 2})
        quadratic = float(residual @ residual / 2)
        objective = huber_sum(residual)
        return {
            "parameters": parameters,
            "passed": True,
            "primary_half_sum_squares": quadratic,
            "alternate_half_sum_squares": quadratic,
            "primary_objective_value": objective,
            "alternate_objective_value": objective,
            "standardized_residuals": residual.tolist(),
        }

    def search(oracle, profile, start, policy, calls, deadline, save):
        result = {
            "calls": 1,
            "completed_calls": 1,
            "best": {
                "parameters": child,
                "mean_twice_objective": 2 * huber_sum(candidate) / count,
            },
            "stop_reason": "step_or_gradient_converged",
        }
        save(result)
        return result

    monkeypatch.setattr(fit, "_verify", verify)
    monkeypatch.setattr(fit, "_search", search)
    result = fit.run(problem, policy, tmp_path, loss=robust.Loss(kind="huber"))
    assert result["parameters"] == child
    assert result["selected_origin"] == "verify-0"
    assert result["objective_half_sum_squares"] > initial @ initial / 2
    assert result["objective_value"] < huber_sum(initial)
    assert result["negative_log_likelihood"] is None
    assert result["selection_metric"] == "objective_value"


def test_independent_solver_check_uses_raw_standardized_residuals(policy):
    loss = robust.Loss(kind="huber")
    primary = np.array([1e8, 0, 0.0])
    alternate = primary + np.array([1e-4, 0, 0])
    derivative = np.array([[1.0], [2.0], [3.0]])
    seen = []

    def evaluate(vector, deadline, *, method, jacobian=True):
        seen.append((method, jacobian))
        return (primary, derivative) if method == "DOP853" else (alternate, None)

    base = SimpleNamespace(
        vector=lambda values: np.asarray([values["g"]]),
        evaluate=evaluate,
        units=np.ones(1),
        names=("g",),
    )
    wrapped = SimpleNamespace(loss=loss, base=base)
    checked = fit._verify(wrapped, {"g": 1.0}, policy, monotonic() + 5, lambda _: None)
    assert not checked["passed"]
    assert checked["maximum_standardized_solver_difference"] > 1e-5
    assert checked["standardized_residuals"] == primary.tolist()
    assert checked["joint_scaled_singular_values"] == pytest.approx([np.sqrt(14)])
    assert checked["primary_objective_value"] == pytest.approx(huber_sum(primary))
    assert seen == [("DOP853", True), ("Radau", False)]
    # The transformed objective residual would hide this numerical discrepancy.
    transformed_difference = abs(
        np.sqrt(2 * huber_sum(primary)) - np.sqrt(2 * huber_sum(alternate))
    )
    assert transformed_difference < policy.maximum_solver_difference


@pytest.mark.parametrize("case", ["exact_varied", "nonlinear_decay"])
def test_direct_and_profiled_huber_fit_agree_without_gaussian_claims(
    policy, tmp_path, case
):
    problem = problem_with_extreme_reading(case)
    loss = robust.Loss(kind="huber")
    direct = fit.run(problem, policy, tmp_path / "direct", strategy="direct", loss=loss)
    profiled = fit.run(problem, policy, tmp_path / "profiled", loss=loss)
    assert direct["status"] == profiled["status"] == "complete"
    assert profiled["objective_value"] == pytest.approx(
        direct["objective_value"], abs=2e-6
    )
    assert profiled["parameters"] == pytest.approx(direct["parameters"], abs=3e-5)
    assert (
        direct["negative_log_likelihood"] is profiled["negative_log_likelihood"] is None
    )
    assert profiled["confidence_probability"] is None
    assert (
        profiled["data_fault_verdict"]
        == profiled["specification_verdict"]
        == "not_assessed"
    )
    evidence = profiled["residual_evidence"]
    assert evidence["downweighted_readings"] > 0
    assert evidence["robust_confidence_assessed"] is False
    assert "neither confirmed" in evidence["downweighting_interpretation"]
    rows = evidence["reading_audit"]
    assert sum(row["objective_contribution"] for row in rows) == pytest.approx(
        profiled["objective_value"]
    )
    assert sum(row["quadratic_contribution"] for row in rows) == pytest.approx(
        profiled["objective_half_sum_squares"]
    )
    assert len(rows) == len(problem.manifest.noise.sigma)
    assert (
        "mean_squared_standardized_residual"
        not in profiled["search"]["attempts"][0]["best"]
    )
    if case == "exact_varied":
        assert profiled["search"]["stop_reason"] == "bounded_affine_problem_solved"
        assert (
            profiled["search"]["attempts"][0]["solution_scope"]
            == "fixed_loss_convex_affine_output_family"
        )
    assert profiled["cost"]["search_calls_started"] <= policy.maximum_calls


def test_fixed_threshold_resume_identity_rejects_changes_before_numerical_work(
    policy, tmp_path, monkeypatch
):
    problem = fixtures.make("exact_varied", 0)[0]
    loss = robust.Loss(kind="huber")
    result = fit.run(problem, policy, tmp_path, loss=loss)
    monkeypatch.setattr(robust, "build", lambda *args: pytest.fail("repeated setup"))
    assert fit.run(problem, policy, tmp_path, loss=loss) == result
    changed = loss.model_dump(mode="json") | {"delta": 1.5}
    with pytest.raises(ValueError, match="differs"):
        fit.run(problem, policy, tmp_path, loss=robust.Loss.model_validate(changed))
    with pytest.raises(ValueError, match="differs"):
        fit.run(problem, policy, tmp_path, loss=robust.Loss(kind="gaussian"))


def test_failed_inner_solve_is_charged_and_fallback_uses_remaining_budget(
    policy, tmp_path, monkeypatch
):
    problem = fixtures.make("exact_varied", 0)[0]
    build, numerical_search = robust.build, fit.numerical.search
    deadlines = []
    ledger = []
    inner_audit = {
        "status": "unavailable",
        "reason": "inner KKT not converged",
        "iterations": 3,
    }

    class InnerFailure(ValueError):
        audit = inner_audit

    def fail(vector, deadline):
        deadlines.append(deadline)
        ledger.append(inner_audit)
        raise InnerFailure("inner KKT not converged")

    def override(*args):
        oracle, _, route = build(*args)
        return (
            oracle,
            SimpleNamespace(outer=(), evaluate=fail, inner_audits=ledger),
            route,
        )

    def search(*args, **kwargs):
        assert args[2] == deadlines[0]
        assert kwargs["calls"] == policy.maximum_calls - 1
        return numerical_search(*args, **kwargs)

    monkeypatch.setattr(robust, "build", override)
    monkeypatch.setattr(fit.numerical, "search", search)
    result = fit.run(problem, policy, tmp_path, loss=robust.Loss(kind="huber"))
    attempt = result["search"]["attempts"][0]
    assert attempt["inner_audits"] == [inner_audit]
    assert attempt["subattempts"][0]["inner_audit"] == inner_audit
    assert attempt["profile_failures"] == ["inner KKT not converged"]
    assert attempt["calls"] == attempt["completed_calls"] + 1
    assert result["cost"]["search_calls_started"] <= policy.maximum_calls
    progress = json.loads((tmp_path / "search-0/progress.json").read_text())
    assert progress["inner_audits"] == [inner_audit]


def test_interrupted_robust_search_keeps_known_ledger_and_never_renews_budget(
    policy, tmp_path, monkeypatch
):
    problem = fixtures.make("exact_varied", 0)[0]
    loss = robust.Loss(kind="huber")
    inner = [{"status": "complete", "iterations": 2, "known_partial": True}]

    def interrupt(oracle, profile, start, policy, calls, deadline, save):
        save(
            {
                "calls": 2,
                "completed_calls": 1,
                "best": {"parameters": dict(start)},
                "inner_audits": inner,
            }
        )
        raise KeyboardInterrupt

    monkeypatch.setattr(fit, "_search", interrupt)
    with pytest.raises(KeyboardInterrupt):
        fit.run(problem, policy, tmp_path, loss=loss)
    monkeypatch.setattr(fit, "_search", lambda *args: pytest.fail("renewed search"))
    result = fit.run(problem, policy, tmp_path, loss=loss)
    assert result["parameters"] == problem.incumbent
    assert not result["cost"]["accounting_complete"]
    assert result["cost"]["search_calls_started"] is None
    assert result["cost"]["budget_charge_seconds"] >= policy.search_seconds
    assert result["search"]["stop_reason"] == "interrupted"
    assert read_seal(tmp_path / "search-0/result.json")["budget_restarted"] is False
    progress = json.loads((tmp_path / "search-0/progress.json").read_text())
    assert progress["inner_audits"] == inner
    assert result["cost"]["profile_iterations_observed"] == 2
    assert result["cost"]["profile_iterations_total"] is None
    assert fit.run(problem, policy, tmp_path, loss=loss) == result


def test_profile_ledger_counts_only_new_work_in_each_restart(policy):
    previous = {"status": "certified", "iterations": 5}
    ledger = [previous]
    oracle = SimpleNamespace(
        loss=robust.Loss(kind="huber"),
        vector=lambda values: np.array([values["g"]]),
        parameters=lambda vector: {"g": float(vector[0])},
    )

    def evaluate(vector, deadline):
        audit = {"status": "certified", "iterations": len(ledger)}
        ledger.append(audit)
        return vector, np.array([0.2, 0.5]), np.zeros((2, 0)), audit

    profile = SimpleNamespace(outer=(), evaluate=evaluate, inner_audits=ledger)
    first = fit._search(
        oracle, profile, {"g": 1.0}, policy, 3, monotonic() + 5, lambda _: None
    )
    second = fit._search(
        oracle, profile, {"g": 1.0}, policy, 3, monotonic() + 5, lambda _: None
    )
    assert first["inner_audits"] == [{"status": "certified", "iterations": 1}]
    assert second["inner_audits"] == [{"status": "certified", "iterations": 2}]
    totals = fit._inner_cost([first, second], True)
    assert totals["profile_iterations_total"] == 3
    assert totals["profile_audits_recorded"] == 2
    assert totals["profile_inner_evaluations_total"] is None


def test_nonlinear_profile_failure_retains_better_huber_progress(policy, monkeypatch):
    profile = SimpleNamespace(outer=(0,), inner_audits=[])
    oracle = SimpleNamespace(loss=robust.Loss(kind="huber"))
    saved = []
    best = {"parameters": {"g": 1.0}, "training_nmse": 2.0}
    worse = {"parameters": {"g": 2.0}, "training_nmse": 8.0}
    qualified = {"status": "certified", "iterations": 2}
    failed = {"status": "unavailable", "iterations": 3, "reason": "inlier rank"}
    limit = monotonic() + 5

    def search(oracle, initial, deadline, save, **kwargs):
        assert deadline == limit
        is_profile = kwargs["profile"] is not None
        if is_profile:
            profile.inner_audits.extend([qualified, failed])
        else:
            assert kwargs["calls"] == 6
            assert initial == best["parameters"]
        result = {
            "calls": 2,
            "completed_calls": 1 if is_profile else 2,
            "best": best if is_profile else worse,
            "stop_reason": "numerical_failure" if is_profile else "budget_limited",
            "message": "inlier rank" if is_profile else "limited",
        }
        save(result)
        return result

    monkeypatch.setattr(fit.numerical, "search", search)
    result = fit._search(oracle, profile, {"g": 0.5}, policy, 8, limit, saved.append)
    assert result["best"] == {"parameters": {"g": 1.0}, "mean_twice_objective": 2.0}
    assert result["calls"] == 4 and result["completed_calls"] == 3
    assert result["inner_audits"] == [qualified, failed]
    assert result["profile_failures"] == ["inlier rank"]
    assert len(result["subattempts"]) == 2
    assert saved[-1]["best"] == result["best"]


def test_invalid_partial_counts_stay_unknown_in_inner_totals(tmp_path):
    progress = {
        "inner_audits": [
            {"iterations": 2, "inner_evaluations": 3, "inner_seconds": 0.25},
            {
                "iterations": -1,
                "inner_evaluations": True,
                "inner_seconds": float("nan"),
            },
        ]
    }
    (tmp_path / "progress.json").write_text(json.dumps(progress))
    recovered = fit._progress_inner_audits(tmp_path)
    assert all(audit["partial_progress_record"] for audit in recovered)
    assert "iterations" not in recovered[1]
    assert "inner_evaluations" not in recovered[1]
    assert "inner_seconds" not in recovered[1]
    totals = fit._inner_cost([{"inner_audits": recovered}], True)
    assert totals["profile_iterations_observed"] == 2
    assert totals["profile_inner_evaluations_observed"] == 3
    assert totals["profile_inner_seconds_observed"] == 0.25
    assert all(
        totals[name] is None
        for name in (
            "profile_iterations_total",
            "profile_inner_evaluations_total",
            "profile_inner_seconds_total",
        )
    )
