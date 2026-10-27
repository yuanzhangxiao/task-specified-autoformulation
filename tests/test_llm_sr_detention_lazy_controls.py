"""Matched schedules retain complete evidence and exact held coordinates."""

from copy import deepcopy
from time import monotonic

import numpy as np
import pytest

from autoformalism.fitting import llm_sr_detention_continuation as chart
from autoformalism.fitting import llm_sr_detention_lazy_controls as control
from autoformalism.fitting import llm_sr_detention_recovery as previous
from autoformalism.fitting import llm_sr_detention_refit as original
from autoformalism.fitting import public_fitting as public
from tests.test_llm_sr_detention_refit import example

MODES = ("original", "reduced_model15_consistent")


def analytic(monkeypatch, *, kink=False, observed=None):
    """Replace integrations with a known residual, preserving real chart/search."""

    def residual(oracle, point, deadline, method="Radau"):
        parameters = oracle.parameters(point)
        if observed is not None:
            observed.append(parameters)
        return np.asarray(
            [
                parameters["beta"] - 0.2,
                1 + abs(parameters["eps_s"] - 1)
                if kink
                else parameters["eps_s"] - 0.7,
            ]
        )

    monkeypatch.setattr(original.DetentionOracle, "residual", residual)


def gate(problem, mode="original", *, precision="tight"):
    """Use actual per-column audits and authentication on cheap analytic data."""
    point = problem[0].parameters
    directions = [
        chart.direction(
            *problem, point, i, mode=mode, precision=precision, seconds=10
        )
        for i in range(len(chart.domain(*problem, point, mode)["names"]))
    ]
    return chart.qualification(
        *problem, point, directions, mode=mode, precision=precision
    )


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("held", [False, True])
def test_eager_lazy_match_full_and_held_charts(monkeypatch, mode, held):
    observed = []
    analytic(monkeypatch, observed=observed)
    problem = example(15)
    initial = problem[0].parameters
    qualified = gate(problem, mode)
    names = qualified["coordinate_names"]
    selected = [name for name in names if name != "eps_s"] if held else None
    actual_column, searched = previous._column, []

    def column(oracle, point, index, deadline):
        searched.append(oracle.names[index])
        assert not held or oracle.names[index] != "eps_s"
        return actual_column(oracle, point, index, deadline)

    monkeypatch.setattr(previous, "_column", column)
    observed.clear()
    results = [
        control.fit_control(
            *problem,
            initial,
            mode=mode,
            lazy_jacobian=lazy,
            active_names=selected,
            qualification=qualified,
            seconds=10,
        )
        for lazy in (False, True)
    ]
    eager, lazy = results
    assert searched
    assert eager["best"]["parameters"] == lazy["best"]["parameters"]
    assert eager["best"]["training_nmse"] == lazy["best"]["training_nmse"]
    assert eager["best"]["training_nmse"] < (0.046 if held else 1e-8)
    assert eager["solver_scaling"] == lazy["solver_scaling"]
    for result in results:
        assert len(result["best"]["parameters"]) == 9
        assert len(result["history_coordinate_names"]) == (
            9 if mode == "original" else 8
        )
        assert result["held_names"] == (["eps_s"] if held else [])
        assert result["search_covers_full_chart"] == (not held)
        assert result["temporary_coordinate_restriction"] == held
        assert not result["full_model_convergence_claimed"]
        assert not result["parameter_information_qualified"]
        assert not result["parameter_rank_qualified"]
        assert result["stationarity_scope"] == "active_coordinates_only"
        assert result["integration_settings"] == qualified["directions"][0][
            "integration_settings"
        ]
        if held:
            assert result["held_chart_values"] == {"eps_s": initial["eps_s"]}
            for step in result["accepted_steps"]:
                assert step["reduced_vector"][names.index("eps_s")] == initial["eps_s"]
    if held:
        assert all(point["eps_s"] == initial["eps_s"] for point in observed)


@pytest.mark.parametrize("mode", MODES)
def test_failed_held_direction_is_explicit_and_does_not_block_subset(monkeypatch, mode):
    analytic(monkeypatch, kink=True)
    problem = example(15)
    initial = problem[0].parameters
    qualified = gate(problem, mode)
    assert qualified["complete"] and not qualified["search_qualified"]
    selected = control.select_active(
        *problem, initial, mode=mode, qualification=qualified
    )
    assert "eps_s" not in selected
    blocked = control.fit_control(
        *problem, initial, mode=mode, lazy_jacobian=True, qualification=qualified
    )
    assert blocked["stop_reason"] == "derivatives_unqualified"
    assert blocked["calls"] == blocked["search_calls_started"] == 0
    result = control.fit_control(
        *problem,
        initial,
        mode=mode,
        lazy_jacobian=True,
        active_names=selected,
        qualification=qualified,
    )
    assert result["best"]["parameters"]["eps_s"] == initial["eps_s"]
    assert result["best"]["training_nmse"] < 0.501
    with pytest.raises(ValueError, match="qualified chart rows"):
        control.fit_control(
            *problem,
            initial,
            mode=mode,
            lazy_jacobian=True,
            active_names=qualified["coordinate_names"],
            qualification=qualified,
        )


@pytest.mark.parametrize("lazy", [False, True])
def test_active_metrics_ignore_arbitrary_held_derivative_columns(monkeypatch, lazy):
    analytic(monkeypatch)
    problem = example(15)
    qualified = gate(problem)
    arguments = {
        "lazy_jacobian": lazy,
        "active_names": ("beta",),
        "qualification": qualified,
        "seconds": 10,
    }
    expected = control.fit_control(*problem, problem[0].parameters, **arguments)
    actual_evaluate = control.ControlOracle.evaluate

    def perturbed(oracle, *args, **kwargs):
        residual, jacobian = actual_evaluate(oracle, *args, **kwargs)
        if jacobian is not None:
            jacobian[:, oracle.held_indices] = 1e250
        return residual, jacobian

    monkeypatch.setattr(control.ControlOracle, "evaluate", perturbed)
    actual = control.fit_control(*problem, problem[0].parameters, **arguments)
    assert actual["best"] == expected["best"]
    assert actual["solver_scaling"] == expected["solver_scaling"]
    assert actual["stop_reason"] == expected["stop_reason"]
    for key in ("evaluations", "accepted_steps"):
        assert [
            {k: v for k, v in row.items() if k != "elapsed_seconds"}
            for row in actual[key]
        ] == [
            {k: v for k, v in row.items() if k != "elapsed_seconds"}
            for row in expected[key]
        ]


@pytest.mark.parametrize("lazy", [False, True])
def test_finite_rollout_survives_derivative_failure(monkeypatch, lazy):
    analytic(monkeypatch)
    problem = example(15)
    initial = problem[0].parameters
    qualified = gate(problem)
    saved = []

    def interrupted(*args, **kwargs):
        raise TimeoutError("interrupted derivative")

    monkeypatch.setattr(previous, "_column", interrupted)
    result = control.fit_control(
        *problem,
        initial,
        lazy_jacobian=lazy,
        active_names=("beta",),
        qualification=qualified,
        checkpoint=saved.append,
        seconds=10,
    )
    assert result["stop_reason"] == "budget_limited"
    assert result["best"]["parameters"] == initial
    assert result["best"]["independent_verification_required"]
    assert any(row.get("best") for row in saved)
    assert all(row["active_names"] == ["beta"] for row in saved)


def test_residual_only_checkpoint_and_exact_held_vector(monkeypatch):
    analytic(monkeypatch)
    problem = example(15)
    initial = problem[0].parameters
    saved = []
    oracle = control.ControlOracle(
        *problem,
        initial,
        "original",
        control.Policy(),
        "tight",
        saved.append,
        active_names=("beta",),
    )
    oracle.retain_base_checkpoints = True
    point = oracle.vector(initial)
    point[oracle.names.index("beta")] = 0.2
    residual, jacobian = oracle.evaluate(point, monotonic() + 10, jacobian=False)
    assert jacobian is None
    assert oracle.cost["jacobian_calls"] == 0
    assert saved[-1]["best_rollout_checkpoint"]["training_nmse"] == np.mean(residual**2)
    point[oracle.names.index("eps_s")] += 1e-5
    with pytest.raises(ValueError, match="held coordinate"):
        oracle.evaluate(point, monotonic() + 10, jacobian=False)


@pytest.mark.parametrize(
    "names", [("beta", "beta"), ("bogus",), ("eps_s", "beta"), "beta", (2,)]
)
def test_invalid_masks_fail_before_search(monkeypatch, names):
    analytic(monkeypatch)
    problem = example(15)
    qualified = gate(problem)
    monkeypatch.setattr(
        control.backend, "search", lambda *a, **k: pytest.fail("search")
    )
    with pytest.raises(ValueError, match="active"):
        control.fit_control(
            *problem,
            problem[0].parameters,
            lazy_jacobian=True,
            active_names=names,
            qualification=qualified,
        )


def test_empty_mask_and_native_accounting_remain_explicit(monkeypatch):
    analytic(monkeypatch)
    problem = example(15)
    qualified = gate(problem)
    monkeypatch.setattr(
        control.backend, "search", lambda *a, **k: pytest.fail("search")
    )
    result = control.fit_control(
        *problem,
        problem[0].parameters,
        lazy_jacobian=True,
        active_names=[],
        qualification=qualified,
    )
    assert result["stop_reason"] == "no_qualified_coordinates"
    assert result["calls"] == result["search_calls_started"] == 0
    assert result["best"] is None
    assert result["numerical_cost"]["jacobian_calls"] == 0


def test_malformed_missing_mixed_and_forged_qualifications_rejected(monkeypatch):
    analytic(monkeypatch)
    problem = example(15)
    initial = problem[0].parameters
    qualified = gate(problem)
    partial = chart.qualification(
        *problem, initial, qualified["directions"][:-1], mode="original"
    )
    with pytest.raises(ValueError, match="all full-chart"):
        control.select_active(*problem, initial, qualification=partial)
    with pytest.raises(ValueError, match="endpoint"):
        control.select_active(
            *problem, initial | {"beta": 0.5}, qualification=qualified
        )
    with pytest.raises(ValueError, match="tight complete-chart"):
        control.select_active(
            *problem, initial, qualification=gate(problem, precision="standard")
        )
    forged = deepcopy(qualified)
    forged["rows"][0]["search_qualified"] = False
    forged["qualification_sha256"] = public.content_sha256(
        {k: v for k, v in forged.items() if k != "qualification_sha256"}
    )
    with pytest.raises(ValueError, match="reconstructed evidence"):
        control.select_active(*problem, initial, qualification=forged)
    malformed = deepcopy(qualified)
    malformed["directions"][0].pop("derivative")
    with pytest.raises(ValueError):
        control.select_active(*problem, initial, qualification=malformed)
    with pytest.raises(ValueError, match="consistent reduced chart"):
        control.select_active(
            *problem, initial, mode="reduced_model15", qualification=qualified
        )


def test_native_cost_and_stationarity_scope_are_not_callback_cost_or_confidence(
    monkeypatch,
):
    analytic(monkeypatch)
    problem = example(15)
    qualified = gate(problem)

    def search(oracle, *args, **kwargs):
        assert kwargs["residual_only_trials"]
        assert kwargs["normalized_objective"]
        oracle.cost.update(
            residual_calls_started=13,
            residual_calls_completed=12,
            residual_calls_failed=1,
            jacobian_calls=2,
        )
        return {
            "calls": 3,
            "best": None,
            "stop_reason": "stationary_poor_fit",
            "stationarity": {"projected_gradient_inf": 1e-8},
        }

    monkeypatch.setattr(control.backend, "search", search)
    result = control.fit_control(
        *problem,
        problem[0].parameters,
        lazy_jacobian=True,
        active_names=("beta",),
        qualification=qualified,
    )
    assert result["calls"] == 3
    assert result["search_calls_started"] == 13
    assert result["search_calls_completed"] == 12
    assert result["search_calls_failed"] == 1
    assert result["numerical_cost"]["jacobian_calls"] == 2
    assert result["stop_reason"] == "active_block_stationary_poor_fit"
    assert result["active_block_stationarity"]["projected_gradient_inf"] == 1e-8
    assert "stationarity" not in result
    assert not result["full_model_convergence_claimed"]
