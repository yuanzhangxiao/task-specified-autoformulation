"""Reliable blocks hold omitted coordinates and retain verifiable checkpoints."""

from copy import deepcopy
from time import monotonic

import numpy as np
import pytest

from autoformalism.fitting import llm_sr_detention_active_blocks as active
from autoformalism.fitting import llm_sr_detention_continuation as chart
from autoformalism.fitting import llm_sr_detention_recovery as previous
from autoformalism.fitting import llm_sr_detention_refit as original
from autoformalism.fitting import public_fitting as public
from tests.test_llm_sr_detention_refit import example


def analytic(monkeypatch, *, kink=False, observed=None):
    """Use a two-coordinate known residual while keeping the real chart/search."""

    def residual(oracle, point, deadline, method="Radau"):
        p = oracle.parameters(point)
        if observed is not None:
            observed.append(dict(p))
        return np.asarray(
            [p["beta"] - 0.2, 1 + abs(p["eps_s"] - 1) if kink else p["eps_s"] - 0.7]
        )

    monkeypatch.setattr(original.DetentionOracle, "residual", residual)


def gate(problem, parameters=None, policy=None):
    """Build actual complete derivative receipts on the analytic residual."""
    parameters = parameters or problem[0].parameters
    directions = [
        chart.direction(
            *problem,
            parameters,
            i,
            mode=active.MODE,
            seconds=10,
            policy=policy,
        )
        for i in range(8)
    ]
    return chart.qualification(
        *problem, parameters, directions, mode=active.MODE, policy=policy
    )


def test_partial_search_does_not_evaluate_held_derivatives_and_can_release(monkeypatch):
    observed = []
    analytic(monkeypatch, observed=observed)
    problem = example(15)
    initial = problem[0].parameters
    qualified = gate(problem)
    names = active.select_active(*problem, initial, qualification=qualified)
    assert len(names) == 8
    selected = tuple(n for n in names if n != "eps_s")
    actual_column = previous._column
    searched_indices = []

    def column(oracle, point, index, deadline):
        searched_indices.append(index)
        assert oracle.names[index] != "eps_s"
        return actual_column(oracle, point, index, deadline)

    monkeypatch.setattr(previous, "_column", column)
    observed.clear()
    result = active.fit_active(
        *problem, initial, qualification=qualified, active_names=selected, seconds=10
    )
    assert result["best"]["training_nmse"] < 0.046
    assert searched_indices and 2 not in searched_indices
    assert all(p["eps_s"] == initial["eps_s"] for p in observed)
    assert result["held_names"] == ["eps_s"]
    assert result["held_chart_values"] == {"eps_s": initial["eps_s"]}
    assert result["temporary_coordinate_restriction"]
    assert not result["search_covers_full_chart"]
    assert result["history_coordinate_names"] == list(names)
    assert not result["full_model_convergence_claimed"]
    assert not result["parameter_information_qualified"]
    assert result["stationarity_scope"] == "active_coordinates_only"
    monkeypatch.setattr(previous, "_column", actual_column)
    # A fresh full audit at the new endpoint can release the held coordinate.
    retained = result["best"]["parameters"]
    rechecked = gate(problem, retained)
    released = active.fit_active(
        *problem,
        retained,
        qualification=rechecked,
        active_names=active.select_active(*problem, retained, qualification=rechecked),
        seconds=10,
    )
    assert released["held_names"] == []
    assert released["search_covers_full_chart"]
    assert released["best"]["training_nmse"] < 1e-8
    assert released["best"]["parameters"]["eps_s"] == pytest.approx(0.7, abs=1e-4)
    # New checkpoints cannot use the initial derivative receipts.
    with pytest.raises(ValueError, match="endpoint"):
        active.select_active(*problem, retained, qualification=qualified)


def test_failed_coordinate_is_explicitly_held_not_silently_qualified(monkeypatch):
    analytic(monkeypatch, kink=True)
    problem = example(15)
    initial = problem[0].parameters
    qualified = gate(problem)
    assert qualified["complete"] and not qualified["search_qualified"]
    names = active.select_active(*problem, initial, qualification=qualified)
    assert len(names) == 7 and "eps_s" not in names
    with pytest.raises(ValueError, match="qualified chart rows"):
        active.fit_active(
            *problem,
            initial,
            qualification=qualified,
            active_names=qualified["coordinate_names"],
            seconds=10,
        )
    result = active.fit_active(
        *problem, initial, qualification=qualified, active_names=names, seconds=10
    )
    assert result["best"]["training_nmse"] < 0.501
    assert result["best"]["parameters"]["eps_s"] == initial["eps_s"]


def test_all_active_matches_unrestricted_corrected_chart_search(monkeypatch):
    analytic(monkeypatch)
    problem = example(15)
    initial = problem[0].parameters
    qualified = gate(problem)
    ordinary = chart.fit(
        *problem,
        initial,
        mode=active.MODE,
        qualification=qualified,
        seconds=10,
    )
    block = active.fit_active(
        *problem,
        initial,
        qualification=qualified,
        active_names=qualified["coordinate_names"],
        seconds=10,
    )
    assert block["best"]["training_nmse"] == ordinary["best"]["training_nmse"]
    assert block["best"]["parameters"] == ordinary["best"]["parameters"]
    assert block["calls"] == ordinary["calls"]


@pytest.mark.parametrize(
    "names", [("beta", "beta"), ("bogus",), ("eps_s", "beta"), "beta", (2,)]
)
def test_invalid_active_sets_are_rejected_before_search(monkeypatch, names):
    analytic(monkeypatch)
    problem = example(15)
    qualified = gate(problem)
    monkeypatch.setattr(
        active.recovery, "search", lambda *a, **k: pytest.fail("search")
    )
    with pytest.raises(ValueError, match="active"):
        active.fit_active(
            *problem,
            problem[0].parameters,
            qualification=qualified,
            active_names=names,
            seconds=10,
        )


def test_empty_active_set_is_an_explicit_no_search_result(monkeypatch):
    analytic(monkeypatch)
    problem = example(15)
    qualified = gate(problem)
    monkeypatch.setattr(
        active.recovery, "search", lambda *a, **k: pytest.fail("search")
    )
    result = active.fit_active(
        *problem, problem[0].parameters, qualification=qualified, active_names=[]
    )
    assert result["calls"] == 0 and result["best"] is None
    assert result["stop_reason"] == "no_qualified_coordinates"
    assert len(result["held_names"]) == 8


def test_missing_forged_and_other_policy_evidence_rejected(monkeypatch):
    analytic(monkeypatch)
    problem = example(15)
    initial = problem[0].parameters
    qualified = gate(problem)
    partial = chart.qualification(
        *problem, initial, qualified["directions"][:-1], mode=active.MODE
    )
    with pytest.raises(ValueError, match="all eight"):
        active.select_active(*problem, initial, qualification=partial)
    forged = deepcopy(qualified)
    forged["rows"][0]["search_qualified"] = False
    forged["qualification_sha256"] = public.content_sha256(
        {k: v for k, v in forged.items() if k != "qualification_sha256"}
    )
    with pytest.raises(ValueError, match="reconstructed evidence"):
        active.select_active(*problem, initial, qualification=forged)
    with pytest.raises(ValueError, match="contract"):
        active.select_active(
            *problem, initial, qualification=qualified, policy=active.precision_policy()
        )


def test_higher_precision_preserves_derivative_settings_and_actual_steps(monkeypatch):
    analytic(monkeypatch)
    problem = example(15)
    policy = active.Policy()
    tighter = active.precision_policy(policy)
    before = policy.model_dump(mode="json")
    after = tighter.model_dump(mode="json")
    for name in ("tight_absolute_tolerance", "tight_relative_tolerance"):
        assert after["numerical"].pop(name) == before["numerical"].pop(name) / 100
    assert before == after
    records = [
        chart.direction(
            *problem,
            problem[0].parameters,
            2,
            mode=active.MODE,
            seconds=10,
            policy=p,
        )
        for p in (policy, tighter)
    ]
    assert (
        records[0]["optimizer_fd_relative_step"]
        == records[1]["optimizer_fd_relative_step"]
    )
    assert [p["step"] for p in records[0]["derivative"]["local_probe_attempts"]] == [
        p["step"] for p in records[1]["derivative"]["local_probe_attempts"]
    ]
    assert records[1]["integration_settings"] == {
        "absolute_tolerance": policy.numerical.tight_absolute_tolerance / 100,
        "relative_tolerance": policy.numerical.tight_relative_tolerance / 100,
    }
    assert (
        records[0]["identity"]["policy_sha256"]
        != records[1]["identity"]["policy_sha256"]
    )


def test_base_rollout_checkpoint_survives_derivative_interruption(monkeypatch):
    analytic(monkeypatch)
    problem = example(15)
    initial = problem[0].parameters
    qualified = gate(problem)
    saved = []

    def interrupted(*args, **kwargs):
        raise TimeoutError("derivative interrupted after finite base rollout")

    monkeypatch.setattr(previous, "_column", interrupted)
    result = active.fit_active(
        *problem,
        initial,
        qualification=qualified,
        active_names=("beta",),
        seconds=10,
        checkpoint=saved.append,
    )
    assert result["stop_reason"] == "budget_limited"
    assert result["best"]["parameters"] == initial
    assert result["best"]["independent_verification_required"]
    assert any(row.get("best") for row in saved)
    assert all(row["active_names"] == ["beta"] for row in saved)


def test_oracle_rejects_held_changes_and_never_supplies_inactive_curvature(monkeypatch):
    analytic(monkeypatch)
    problem = example(15)
    initial = problem[0].parameters
    oracle = active.ActiveOracle(
        *problem, initial, active.MODE, active.Policy(), "tight", active_names=("beta",)
    )
    point = oracle.vector(initial)
    _, matrix = oracle.evaluate(point, monotonic() + 10)
    assert np.isfinite(matrix[:, 1]).all()
    assert np.isnan(matrix[:, [0, 2, 3, 4, 5, 6, 7]]).all()
    point[2] += 0.1
    with pytest.raises(ValueError, match="held coordinate"):
        oracle.evaluate(point, monotonic() + 10)
