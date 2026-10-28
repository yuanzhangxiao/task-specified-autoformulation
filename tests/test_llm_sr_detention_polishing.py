"""Polishing reuses exact qualified samples without promoting search precision."""

from copy import deepcopy

import numpy as np
import pytest

from autoformalism.fitting import llm_sr_detention_continuation as chart
from autoformalism.fitting import llm_sr_detention_lazy_controls as controls
from autoformalism.fitting import llm_sr_detention_polishing as kernel
from autoformalism.fitting import llm_sr_detention_probe_recovery as probe
from autoformalism.fitting import llm_sr_detention_recovery as previous
from autoformalism.fitting import llm_sr_detention_refit as original
from autoformalism.fitting import public_fitting as public
from tests.test_llm_sr_detention_refit import example


def analytic(monkeypatch, target, *, kind="smooth"):
    """Provide dimension-matching cheap residuals and honest native call counters."""
    calls = []

    def residual(oracle, point, deadline, method="Radau"):
        calls.append((np.asarray(point).copy(), method))
        oracle.cost["residual_calls_started"] += 1
        oracle.cost["residual_calls_completed"] += 1
        oracle.cost["trajectory_integrations"] += 1
        result = np.asarray(point) - np.asarray([target[n] for n in oracle.names])
        if kind == "zero":
            result[oracle.names.index("h_max")] = 0
        elif kind == "kink":
            index = oracle.names.index("eps_s")
            result[index] = 1 + abs(result[index])
        return np.resize(
            result, sum(len(row.time) for row in oracle.split.trajectories)
        )

    monkeypatch.setattr(original.DetentionOracle, "residual", residual)
    return calls


def gate(problem, parameters=None, mode="original", *, root=None, search_only=True):
    """Build real authenticated finite-difference records without ODE integrations."""
    parameters = parameters or problem[0].parameters
    count = len(chart.domain(*problem, parameters, mode)["names"])
    records = [
        chart.direction(*problem, parameters, i, mode=mode, seconds=5)
        if root is None
        else probe.direction(
            *problem,
            parameters,
            i,
            mode=mode,
            seconds=5,
            cache_root=root,
            search_only=search_only,
        )
        for i in range(count)
    ]
    return chart.qualification(*problem, parameters, records, mode=mode)


def verification(parameters, qualified):
    """Represent an externally authenticated matching independent replay."""
    loss = float(np.mean(np.asarray(qualified["directions"][0]["base_residual"]) ** 2))
    return {
        "status": "complete",
        "parameters": dict(parameters),
        "usable_for_retention": True,
        "training_nmse": loss,
        "alternate_training_nmse": loss,
        "solver_loss_discrepancy": 0.0,
        "solver_prediction_discrepancy": 0.0,
    }


@pytest.mark.parametrize("mode", ["original", "reduced_model15_consistent"])
def test_exact_initial_reuse_then_fresh_active_columns_at_changed_point(
    monkeypatch, mode
):
    problem = example(15)
    parameters = problem[0].parameters
    calls = analytic(monkeypatch, parameters | {"beta": 0.2})
    qualified = gate(problem, mode=mode)
    calls.clear()
    columns, saved = [], []
    actual_column = previous._column

    def column(oracle, point, index, deadline):
        columns.append(index)
        return actual_column(oracle, point, index, deadline)

    def search(oracle, initial, deadline, checkpoint, **kwargs):
        assert kwargs["residual_only_trials"]
        point = oracle.vector(initial)
        residual, matrix = oracle.evaluate(point, deadline)
        assert not calls and not columns
        assert np.array_equal(residual, qualified["base_residual"])
        assert np.isnan(matrix[:, oracle.held_indices]).all()
        assert np.array_equal(
            matrix[:, oracle.active_indices],
            np.asarray(qualified["jacobian"])[:, oracle.active_indices],
        )
        changed = point.copy()
        changed[oracle.names.index("beta")] += 1e-3
        oracle.evaluate(changed, deadline)
        assert calls and columns == [oracle.names.index("beta")]
        return {"best": None, "calls": 2, "stop_reason": "test_complete"}

    monkeypatch.setattr(previous, "_column", column)
    monkeypatch.setattr(controls.backend, "search", search)
    result = kernel.fit(
        *problem,
        parameters,
        mode=mode,
        qualification=qualified,
        active_names=["beta"],
        checkpoint=saved.append,
    )
    assert result["initial_sample_reuse"]["seeded_active_columns"] == 1
    assert result["initial_sample_reuse"]["seeded_held_columns"] == 0
    assert result["initial_sample_reuse"]["jacobian_reuse_hits"] == 1
    assert result["search_calls_started"] == len(calls)
    assert result["numerical_cost"]["jacobian_calls"] == 1
    assert saved and all(row["maximum_segments"] == 1 for row in saved)
    assert result["best"]["independent_verification_required"]


def test_default_control_does_not_reuse_saved_samples(monkeypatch):
    problem = example(15)
    calls = analytic(monkeypatch, problem[0].parameters)
    qualified = gate(problem)
    calls.clear()

    def search(oracle, initial, deadline, *args, **kwargs):
        oracle.evaluate(oracle.vector(initial), deadline)
        return {"best": None, "stop_reason": "test_complete"}

    monkeypatch.setattr(controls.backend, "search", search)
    result = controls.fit_control(
        *problem,
        problem[0].parameters,
        lazy_jacobian=True,
        active_names=["beta"],
        qualification=qualified,
    )
    assert calls and result["numerical_cost"]["jacobian_calls"] == 1
    assert "initial_sample_reuse" not in result


def test_seeded_lazy_search_improves_and_keeps_checkpoint_on_later_failure(monkeypatch):
    problem = example(15)
    parameters = problem[0].parameters
    analytic(monkeypatch, parameters | {"beta": 0.2})
    qualified = gate(problem)
    saved = []

    def fails(*args):
        raise TimeoutError("new point derivative interrupted")

    monkeypatch.setattr(previous, "_column", fails)
    result = kernel.fit(
        *problem,
        parameters,
        qualification=qualified,
        active_names=["beta"],
        checkpoint=saved.append,
        seconds=5,
    )
    assert result["stop_reason"] == "budget_limited"
    assert result["best"]["training_nmse"] < np.mean(
        np.asarray(qualified["base_residual"]) ** 2
    )
    assert any(
        row.get("best", {}).get("parameters") == result["best"]["parameters"]
        and row["best"]["training_nmse"] == result["best"]["training_nmse"]
        for row in saved
    )
    assert not result["full_model_convergence_claimed"]


def test_completed_polishing_matches_unseeded_search(monkeypatch):
    problem = example(15)
    parameters = problem[0].parameters
    analytic(monkeypatch, parameters | {"beta": 0.2})
    qualified = gate(problem)
    arguments = {"qualification": qualified, "active_names": ["beta"], "seconds": 5}
    seeded = kernel.fit(*problem, parameters, **arguments)
    plain = controls.fit_control(*problem, parameters, lazy_jacobian=True, **arguments)
    assert seeded["best"]["parameters"] == plain["best"]["parameters"]
    assert seeded["best"]["training_nmse"] < 1e-8
    assert seeded["search_calls_started"] < plain["search_calls_started"]


def test_seeded_samples_do_not_cross_methods_or_settings(monkeypatch):
    problem = example(15)
    parameters = problem[0].parameters
    calls = analytic(monkeypatch, parameters)
    qualified = gate(problem)
    calls.clear()

    def search(oracle, initial, deadline, *args, **kwargs):
        point = oracle.vector(initial)
        oracle.evaluate(point, deadline, jacobian=False, method="DOP853")
        assert len(calls) == 1 and calls[0][1] == "DOP853"
        with pytest.raises(ValueError, match="matching Radau"):
            oracle.evaluate(point, deadline, method="DOP853")
        oracle.settings = oracle.settings.model_copy(
            update={"absolute_tolerance": oracle.settings.absolute_tolerance / 10}
        )
        oracle.evaluate(point, deadline, jacobian=False)
        assert len(calls) == 2
        with pytest.raises(ValueError, match="matching Radau"):
            oracle.evaluate(point, deadline)
        return {"best": None, "stop_reason": "test_complete"}

    monkeypatch.setattr(controls.backend, "search", search)
    result = kernel.fit(*problem, parameters, qualification=qualified)
    assert result["initial_sample_reuse"]["jacobian_reuse_hits"] == 0
    assert result["initial_sample_reuse"]["residual_reuse_hits"] == 0


@pytest.mark.parametrize("kind", ["array", "policy", "point", "data", "settings"])
def test_tampering_and_mismatched_contracts_fail_before_search(monkeypatch, kind):
    problem = list(example(15))
    parameters = problem[0].parameters
    analytic(monkeypatch, parameters)
    qualified = gate(problem)
    kwargs = {}
    if kind == "array":
        qualified["jacobian"][0][0] += 1
    elif kind == "settings":
        qualified["directions"][0]["integration_settings"]["absolute_tolerance"] *= 2
    elif kind == "policy":
        kwargs["policy"] = chart.Policy(search_gradient_relative_error=0.02)
    elif kind == "point":
        parameters = parameters | {"beta": 0.9}
    else:
        problem[2] = problem[2].model_copy(update={"fingerprint": "other"})
    qualified["qualification_sha256"] = public.content_sha256(
        {
            key: value
            for key, value in qualified.items()
            if key != "qualification_sha256"
        }
    )
    monkeypatch.setattr(
        controls.backend, "search", lambda *a, **k: pytest.fail("search")
    )
    with pytest.raises(ValueError):
        kernel.fit(*problem, parameters, qualification=qualified, **kwargs)


@pytest.mark.parametrize("seconds", [0, -1, 601, float("nan"), True])
def test_polishing_allowance_is_bounded(seconds):
    with pytest.raises(ValueError, match="at most 600"):
        kernel.fit(None, None, None, {}, qualification={}, seconds=seconds)


@pytest.mark.parametrize("kind", ["smooth", "zero", "kink"])
def test_strict_endpoint_assessment_does_not_promote_zero_or_kink(monkeypatch, kind):
    problem = example(15)
    parameters = problem[0].parameters
    analytic(monkeypatch, parameters, kind=kind)
    qualified = gate(problem)
    before = deepcopy(qualified)
    result = kernel.assess_endpoint(
        *problem,
        parameters,
        qualification=qualified,
        verification=verification(parameters, qualified),
    )
    assert qualified == before
    assert result["status"] == (
        "qualified_local_stationarity" if kind == "smooth" else "unresolved"
    )
    assert not result["parameter_confidence_claimed"]
    assert not result["full_model_convergence_claimed"]
    assert result["original_parameter_uniqueness"] == "excluded_by_exact_ambiguity"


def test_search_only_precision_can_show_descent_but_never_stationarity(
    monkeypatch, tmp_path
):
    problem = example(15)
    parameters = problem[0].parameters
    analytic(monkeypatch, parameters | {"beta": 0.2})
    qualified = gate(problem, root=tmp_path / "descent")
    result = kernel.assess_endpoint(
        *problem,
        parameters,
        qualification=qualified,
        verification=verification(parameters, qualified),
    )
    assert result["status"] == "reliable_descent_remains"
    assert result["supporting_coordinate"]["coordinate"] == "beta"
    analytic(monkeypatch, parameters)
    qualified = gate(problem, root=tmp_path / "near_zero")
    result = kernel.assess_endpoint(
        *problem,
        parameters,
        qualification=qualified,
        verification=verification(parameters, qualified),
    )
    assert result["status"] == "unresolved"


def test_strict_cached_reaudit_reuses_raw_samples_without_search_stopping(
    monkeypatch, tmp_path
):
    problem = example(15)
    parameters = problem[0].parameters
    calls = analytic(monkeypatch, parameters)
    gate(problem, root=tmp_path)
    before = len(calls)
    qualified = gate(problem, root=tmp_path, search_only=False)
    assert len(calls) == before
    assert all(
        row["probe_recovery"]["search_error_limits"] is None
        for row in qualified["directions"]
    )
    result = kernel.assess_endpoint(
        *problem,
        parameters,
        qualification=qualified,
        verification=verification(parameters, qualified),
    )
    assert result["status"] == "qualified_local_stationarity"


def test_incomplete_chart_can_support_descent_but_not_full_gradient_bound(monkeypatch):
    problem = example(15)
    parameters = problem[0].parameters
    analytic(monkeypatch, parameters | {"beta": 0.2})
    full = gate(problem)
    qualified = chart.qualification(
        *problem,
        parameters,
        [row for row in full["directions"] if row["name"] == "beta"],
        mode="original",
    )
    result = kernel.assess_endpoint(
        *problem,
        parameters,
        qualification=qualified,
        verification=verification(parameters, full),
    )
    assert result["status"] == "reliable_descent_remains"
    assert result["projected_gradient_upper_bound"] is None
    assert not result["qualified_full_chart_stationarity"]


@pytest.mark.parametrize("target_offset", [-0.5, 0.5])
def test_lower_bound_projection_excludes_infeasible_descent(monkeypatch, target_offset):
    problem = example(15)
    lower = original.domain_policy(*problem)["bounds"]["beta"][0]
    parameters = problem[0].parameters | {"beta": lower}
    analytic(monkeypatch, parameters | {"beta": lower + target_offset})
    qualified = gate(problem, parameters)
    result = kernel.assess_endpoint(
        *problem,
        parameters,
        qualification=qualified,
        verification=verification(parameters, qualified),
    )
    beta = next(
        row for row in result["coordinate_evidence"] if row["coordinate"] == "beta"
    )
    assert not beta["decrease_feasible"] and beta["increase_feasible"]
    if target_offset > 0:
        assert result["status"] == "reliable_descent_remains"
        assert beta["descent_direction_sign"] == 1
    else:
        assert result["status"] != "reliable_descent_remains"
        assert beta["projected_gradient_interval"] == [0, 0]


@pytest.mark.parametrize("tamper", ["point", "loss", "unavailable"])
def test_final_assessment_requires_matching_independent_verification(
    monkeypatch, tamper
):
    problem = example(15)
    parameters = problem[0].parameters
    analytic(monkeypatch, parameters | {"beta": 0.2})
    qualified = gate(problem)
    verified = verification(parameters, qualified)
    if tamper == "point":
        verified["parameters"]["beta"] += 0.1
    elif tamper == "loss":
        verified["training_nmse"] += 1
    else:
        verified["status"] = "unavailable"
    if tamper == "unavailable":
        assert (
            kernel.assess_endpoint(
                *problem,
                parameters,
                qualification=qualified,
                verification=verified,
            )["status"]
            == "unresolved"
        )
    else:
        with pytest.raises(ValueError):
            kernel.assess_endpoint(
                *problem,
                parameters,
                qualification=qualified,
                verification=verified,
            )
