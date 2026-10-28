"""Staged checks spend strict-audit work only on authenticated plausible endpoints."""

from copy import deepcopy

import numpy as np
import pytest

from autoformalism.fitting import llm_sr_detention_continuation as chart
from autoformalism.fitting import llm_sr_detention_lazy_controls as controls
from autoformalism.fitting import llm_sr_detention_polishing as polishing
from autoformalism.fitting import llm_sr_detention_recovery as previous
from autoformalism.fitting import llm_sr_detention_refit as original
from autoformalism.fitting import llm_sr_detention_staged_polishing as kernel
from autoformalism.fitting import public_fitting as public
from tests.test_llm_sr_detention_polishing import analytic, gate, verification
from tests.test_llm_sr_detention_refit import example


def with_error(problem, qualified, *, error, inconclusive=False):
    """Construct authenticated scalar uncertainty fixtures on real raw arrays."""
    records = deepcopy(qualified["directions"])
    record = next(row for row in records if row["name"] == "beta")
    check = record["derivative"]
    check["directions"][0]["gradient_error_bound"] = error
    check["directions"][0]["projected_gradient_error_bound"] = error
    check["gradient_error_bound_inf"] = error
    if inconclusive:
        check["status"] = "inconclusive"
        check["inconclusive_indices"] = [0]
        check["directions"][0]["passed"] = False
        check["directions"][0]["precision_status"] = "inconclusive"
    record["evidence_sha256"] = public.content_sha256(
        {key: value for key, value in record.items() if key != "evidence_sha256"}
    )
    return chart.qualification(
        *problem, qualified["parameters"], records, mode=qualified["mode"]
    )


def decide(problem, qualified, *, verified=None):
    """Use the real assessment/authentication contract for every decision test."""
    parameters = qualified["parameters"]
    return kernel.audit_decision(
        *problem,
        parameters,
        qualification=qualified,
        verification=verification(parameters, qualified)
        if verified is None
        else verified,
        mode=qualified["mode"],
    )


@pytest.mark.parametrize("mode", ["original", "reduced_model15_consistent"])
def test_one_segment_reuses_each_independent_starts_exact_arrays(monkeypatch, mode):
    problem = example(15)
    calls = analytic(monkeypatch, problem[0].parameters | {"beta": 0.2})
    for beta in (0.7, 0.8):
        parameters = problem[0].parameters | {"beta": beta}
        qualified = gate(problem, parameters, mode)
        calls.clear()
        saved = []

        def search(
            oracle,
            initial,
            deadline,
            checkpoint,
            expected_parameters=parameters,
            expected_qualification=qualified,
            **kwargs,
        ):
            assert initial == expected_parameters
            assert kwargs["residual_only_trials"]
            residual, matrix = oracle.evaluate(oracle.vector(initial), deadline)
            assert np.array_equal(residual, expected_qualification["base_residual"])
            assert np.array_equal(matrix, expected_qualification["jacobian"])
            assert not calls
            return {"best": None, "stop_reason": "fixture_complete"}

        monkeypatch.setattr(controls.backend, "search", search)
        result = kernel.fit(
            *problem,
            parameters,
            mode=mode,
            qualification=qualified,
            checkpoint=saved.append,
        )
        assert result["allowance_seconds"] == 1800
        assert result["maximum_segments"] == 1
        assert result["staged_polishing_protocol"] == kernel.PROTOCOL
        assert not result["historical_allowances_renewed"]
        assert result["initial_sample_reuse"]["jacobian_reuse_hits"] == 1
        assert result["initial_sample_reuse"]["seeded_active_columns"] == len(
            qualified["coordinate_names"]
        )
        assert result["best"]["parameters"] == parameters
        assert result["best"]["independent_verification_required"]
        assert saved and all(row["allowance_seconds"] == 1800 for row in saved)


def test_partial_search_keeps_improvement_and_exact_held_parameters_on_timeout(
    monkeypatch,
):
    problem = example(15)
    parameters = problem[0].parameters
    analytic(monkeypatch, parameters | {"beta": 0.2})
    qualified, saved = gate(problem), []

    def interrupted(*args):
        raise TimeoutError("later derivative interrupted")

    monkeypatch.setattr(previous, "_column", interrupted)
    result = kernel.fit(
        *problem,
        parameters,
        qualification=qualified,
        active_names=["beta"],
        seconds=5,
        checkpoint=saved.append,
    )
    assert result["stop_reason"] == "budget_limited"
    assert result["best"]["training_nmse"] < np.mean(
        np.asarray(qualified["base_residual"]) ** 2
    )
    assert all(
        value == result["best"]["parameters"][name]
        for name, value in parameters.items()
        if name != "beta"
    )
    assert any(
        row.get("best", {}).get("parameters") == result["best"]["parameters"]
        and row["best"]["training_nmse"] == result["best"]["training_nmse"]
        for row in saved
    )
    assert not result["full_model_convergence_claimed"]


@pytest.mark.parametrize(
    "seconds", [0, -1, 1801, float("nan"), float("inf"), True, "1800"]
)
def test_segment_allowance_rejects_invalid_values(seconds):
    with pytest.raises(ValueError, match="at most 1800"):
        kernel.fit(None, None, None, {}, qualification={}, seconds=seconds)


def test_old_polishing_allowance_remains_600_seconds():
    with pytest.raises(ValueError, match="at most 600"):
        polishing.fit(None, None, None, {}, qualification={}, seconds=601)


@pytest.mark.parametrize("kind", ["array", "policy", "endpoint", "data", "bounds"])
def test_invalid_initial_evidence_never_enters_search(monkeypatch, kind):
    problem = list(example(15))
    parameters = problem[0].parameters
    analytic(monkeypatch, parameters)
    qualified, arguments = gate(problem), {}
    if kind == "array":
        qualified["jacobian"][0][0] += 1
    elif kind == "policy":
        arguments["policy"] = chart.Policy(search_gradient_relative_error=0.02)
    elif kind == "endpoint":
        parameters = parameters | {"beta": 0.9}
    elif kind == "data":
        problem[2] = problem[2].model_copy(update={"fingerprint": "other"})
    else:
        parameters = parameters | {"beta": -1}
    qualified["qualification_sha256"] = public.content_sha256(
        {
            key: value
            for key, value in qualified.items()
            if key != "qualification_sha256"
        }
    )
    monkeypatch.setattr(
        controls.backend, "search", lambda *args, **kwargs: pytest.fail("search")
    )
    with pytest.raises(ValueError):
        kernel.fit(*problem, parameters, qualification=qualified, **arguments)


@pytest.mark.parametrize("mode", ["original", "reduced_model15_consistent"])
def test_search_only_near_endpoint_allocates_strict_checks_without_promotion(
    monkeypatch, tmp_path, mode
):
    problem = example(15)
    calls = analytic(monkeypatch, problem[0].parameters)
    qualified = gate(problem, mode=mode, root=tmp_path)
    before, count = deepcopy(qualified), len(calls)
    result = decide(problem, qualified)
    assert qualified == before and len(calls) == count
    assert result["strict_audit_required"]
    assert result["decision"] == "strict_audit_required"
    assert result["near_stationarity_plausible"]
    assert result["status"] == "unresolved"
    assert not result["qualified_full_chart_stationarity"]
    assert not result["parameter_information_qualified"]
    strict = gate(problem, mode=mode, root=tmp_path, search_only=False)
    resolved = decide(problem, strict)
    assert resolved["decision"] == "strict_stationarity_already_qualified"
    assert resolved["qualified_full_chart_stationarity"]
    assert not resolved["strict_audit_required"]


def test_reliable_descent_skips_strict_checks_even_with_missing_other_coordinates(
    monkeypatch, tmp_path
):
    problem = example(15)
    parameters = problem[0].parameters
    analytic(monkeypatch, parameters | {"beta": 0.2})
    full = gate(problem, root=tmp_path)
    partial = chart.qualification(
        *problem,
        parameters,
        [row for row in full["directions"] if row["name"] == "beta"],
        mode="original",
    )
    for qualified in (full, partial):
        result = decide(problem, qualified)
        assert result["decision"] == "reliable_descent_remains"
        assert not result["strict_audit_required"]
        assert not result["near_stationarity_plausible"]


@pytest.mark.parametrize("kind", ["zero", "kink"])
def test_local_zero_can_allocate_checks_but_kink_failure_cannot(
    monkeypatch, tmp_path, kind
):
    problem = example(15)
    analytic(monkeypatch, problem[0].parameters, kind=kind)
    qualified = gate(problem, root=tmp_path)
    result = decide(problem, qualified)
    assert result["strict_audit_required"] is (kind == "zero")
    assert not result["qualified_full_chart_stationarity"]
    assert not result["parameter_confidence_claimed"]


@pytest.mark.parametrize("bound_index", [0, 1])
@pytest.mark.parametrize("offset", [-0.5, 0.5])
def test_escalation_uses_feasible_bound_projection(
    monkeypatch, tmp_path, offset, bound_index
):
    problem = example(15)
    bound = original.domain_policy(*problem)["bounds"]["beta"][bound_index]
    parameters = problem[0].parameters | {"beta": bound}
    analytic(monkeypatch, parameters | {"beta": bound + offset})
    qualified = gate(problem, parameters, root=tmp_path)
    result = decide(problem, qualified)
    feasible = offset > 0 if bound_index == 0 else offset < 0
    assert result["strict_audit_required"] is (not feasible)
    assert result["status"] == (
        "reliable_descent_remains" if feasible else "unresolved"
    )


def test_existing_search_error_limit_controls_uncertainty(monkeypatch, tmp_path):
    problem = example(15)
    parameters = problem[0].parameters
    analytic(monkeypatch, parameters | {"beta": parameters["beta"] - 4e-7})
    qualified = gate(problem, root=tmp_path)
    admitted = with_error(problem, qualified, error=1e-7)
    unresolved = with_error(problem, qualified, error=1e-3)
    assert admitted["search_qualified"]
    assert decide(problem, admitted)["strict_audit_required"]
    assert not unresolved["search_qualified"]
    result = decide(problem, unresolved)
    assert result["decision"] == "unresolved"
    assert not result["strict_audit_required"]


def test_inconclusive_nonzero_interval_cannot_allocate_strict_stationarity_checks(
    monkeypatch, tmp_path
):
    problem = example(15)
    analytic(monkeypatch, problem[0].parameters | {"beta": 0.8})
    qualified = with_error(
        problem, gate(problem, root=tmp_path), error=1e-12, inconclusive=True
    )
    result = decide(problem, qualified)
    assert qualified["search_qualified"]
    assert result["status"] == "unresolved"
    assert not result["strict_audit_required"]
    assert (
        result["maximum_lower_empirical_descent_magnitude"]
        > result["gradient_tolerance"]
    )


def test_missing_near_endpoint_evidence_cannot_allocate_strict_checks(monkeypatch):
    problem = example(15)
    analytic(monkeypatch, problem[0].parameters)
    full = gate(problem)
    partial = chart.qualification(
        *problem, full["parameters"], full["directions"][:-1], mode="original"
    )
    result = decide(problem, partial)
    assert result["decision"] == "unresolved"
    assert not result["strict_audit_required"]
    assert result["maximum_lower_empirical_descent_magnitude"] is None


@pytest.mark.parametrize("kind", ["unavailable", "unusable", "disagreement"])
def test_unavailable_independent_verification_blocks_escalation(monkeypatch, kind):
    problem = example(15)
    analytic(monkeypatch, problem[0].parameters)
    qualified = gate(problem)
    verified = verification(qualified["parameters"], qualified)
    if kind == "unavailable":
        verified["status"] = "unavailable"
    elif kind == "unusable":
        verified["usable_for_retention"] = False
    else:
        verified["solver_loss_discrepancy"] = 1
    result = decide(problem, qualified, verified=verified)
    assert result["decision"] == "unavailable"
    assert not result["strict_audit_required"]
    assert not result["qualified_full_chart_stationarity"]


@pytest.mark.parametrize(
    "kind", ["array", "gradient", "receipt", "point", "loss", "malformed"]
)
def test_forged_or_mismatched_evidence_is_rejected(monkeypatch, kind):
    problem = example(15)
    analytic(monkeypatch, problem[0].parameters)
    qualified = gate(problem)
    verified = verification(qualified["parameters"], qualified)
    if kind == "array":
        qualified["jacobian"][0][0] += 1
    elif kind == "gradient":
        qualified["rows"][0]["gradient"] += 1
    elif kind == "receipt":
        qualified["directions"][0]["integration_settings"]["absolute_tolerance"] *= 2
    elif kind == "point":
        verified["parameters"]["beta"] += 0.1
    elif kind == "loss":
        verified["training_nmse"] += 1
    else:
        qualified["directions"][0].pop("derivative")
    qualified["qualification_sha256"] = public.content_sha256(
        {
            key: value
            for key, value in qualified.items()
            if key != "qualification_sha256"
        }
    )
    with pytest.raises(ValueError):
        decide(problem, qualified, verified=verified)
