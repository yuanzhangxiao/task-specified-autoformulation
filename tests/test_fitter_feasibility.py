"""Feasible-start recovery and threshold coverage without hidden labels."""

import numpy as np
import pytest

pytest.importorskip("casadi")

from autoformalism.expressions import ValidationContext, compile_candidate
from autoformalism.fitting.collocation_sensitivity import (
    CollocationSensitivityConfig,
    fit_collocation_forward_sensitivity,
)
from autoformalism.fitting.feasibility import (
    EvaluationBudget,
    GuardedOracle,
    SensitivityUnavailable,
    restart_points,
)
from autoformalism.fitting.initialization import (
    LatentInitializationPlan,
    apply_initialization_plan,
)
from autoformalism.fitting.sensitivity_probe import SymbolicODE
from autoformalism.fitting.start_design import BranchCatalog, designed_starts
from autoformalism.rebuttal.fitter_diagnostic import read_json
from autoformalism.rebuttal.initialization_campaign import synthetic_problem
from autoformalism.rebuttal.piecewise_campaign import unpack_split
from autoformalism.schemas import CandidateModel


def problem(kind="piecewise"):
    raw, _ = synthetic_problem(kind, 0.0, 0)
    model = compile_candidate(
        CandidateModel.model_validate(raw["candidate"]),
        ValidationContext.model_validate(raw["context"]),
    )
    plan = LatentInitializationPlan.model_validate(raw["initialization_plan"])
    compiled, guesses, _ = apply_initialization_plan(model, plan)
    return raw, model, compiled, plan, {**raw["start"], **guesses}


def test_branch_starts_cross_threshold_and_preserve_domains():
    raw, _, model, _, starts = problem()
    system = SymbolicODE(model, allow_piecewise=True)
    training = unpack_split(raw["splits"]["train"])
    theta = np.array([starts[n] for n in system.names])
    lower = np.array([1e-12 if n == "rate" else -np.inf for n in system.names])
    upper = np.full(len(theta), np.inf)
    points, audit = designed_starts(
        system, training, theta, lower, upper, branches=True
    )
    catalog = BranchCatalog(system)
    selected = [p for p in points if p["source"].startswith("branch:")]
    assert selected
    assert any(
        catalog.initial_values(
            training.trajectories[0],
            np.array([p["parameters"][n] for n in system.names]),
        )[0]
        > 0
        for p in selected
    )
    assert all(p["parameters"]["rate"] > 0 for p in points)
    assert (points, audit) == designed_starts(
        system, training, theta, lower, upper, branches=True
    )
    # Observed y(0) stays measured; only declared fitted parameters can move it.
    for p in selected:
        x = system.initial_for(
            training.trajectories[0],
            np.array([p["parameters"][n] for n in system.names]),
        )
        assert x[system.model.state_names.index("y")] == 0


def test_checkpoint_candidates_are_retained_but_not_certified():
    checkpoint = {"parameters": {"a": 2.0}, "physical_rollout_verified": False}
    rows = restart_points(
        {"progress": {"checkpoints": {"latest": checkpoint}}}, {"a": 0.0}, []
    )
    assert rows[0]["parameters"] == {"a": 2.0}
    assert rows[1]["source"] == "ordinary"


def test_node_branch_start_preserves_known_initial_and_observed_nodes():
    from autoformalism.fitting.matching_probe import observed_node_guess
    from autoformalism.fitting.start_design import target_node_branch

    raw, model, _, _, _ = problem()
    system = SymbolicODE(model, allow_piecewise=True)
    data = unpack_split(raw["splits"]["train"]).trajectories[0]
    theta = np.array([raw["start"][n] for n in system.names])
    ordinary = observed_node_guess(system, data, theta)
    alternative, report = target_node_branch(system, data, theta, ordinary, (0, 1))
    assert np.array_equal(alternative[0], ordinary[0])
    y_index = system.model.state_names.index("y")
    assert np.array_equal(alternative[:, y_index], ordinary[:, y_index])
    assert report["coverage"][0]["positive"]


def test_nary_branch_compares_with_all_competitors():
    raw, _, _, _, _ = problem()
    raw["candidate"]["state_equations"][1]["rhs"] = "max(m-1, m-2, 0)-0.7*y"
    model = compile_candidate(
        CandidateModel.model_validate(raw["candidate"]),
        ValidationContext.model_validate(raw["context"]),
    )
    system = SymbolicODE(model, allow_piecewise=True)
    catalog = BranchCatalog(system)
    data = unpack_split(raw["splits"]["train"]).trajectories[0]
    theta = np.array([raw["start"][n] for n in system.names])
    assert len(catalog.branches) == 3
    x = system.initial_for(data, theta)
    x[system.model.state_names.index("m")] = 3
    margins = catalog.values(data, 0, x, theta)
    assert margins[0] > 0 and margins[1] < 0 and margins[2] < 0


def test_failed_augmented_call_raises_instead_of_exposing_zero_jacobian(
    tmp_path, monkeypatch
):
    from time import monotonic

    from autoformalism.fitting import sensitivity_probe

    raw, _, model, _, starts = problem("shared")
    system = SymbolicODE(model)
    training = unpack_split(raw["splits"]["train"])
    config = CollocationSensitivityConfig()
    budget = EvaluationBudget(monotonic() + 10, 3)
    oracle = GuardedOracle(
        system,
        training,
        1.0,
        config.fit_config(),
        tmp_path,
        budget.deadline,
        sensitivities=True,
        budget=budget,
        point_seconds=5,
    )

    def failed(*args, **kwargs):
        raise ValueError("synthetic integration failure")

    monkeypatch.setattr(sensitivity_probe, "symbolic_rollout", failed)
    with pytest.raises(SensitivityUnavailable):
        oracle(oracle.vector(starts))
    assert oracle.last_jac is None
    assert budget.calls == 1 and oracle.valid_calls == 0


def test_branch_collocation_recovers_inactive_start_and_saves_progress(tmp_path):
    raw, model, _, plan, _ = problem()
    result = fit_collocation_forward_sensitivity(
        model,
        unpack_split(raw["splits"]["train"]),
        unpack_split(raw["splits"]["val"]),
        CollocationSensitivityConfig(
            initializer_seconds=30,
            refinement_seconds=25,
            maximum_function_evaluations=60,
            collocation_node_start="rollout_or_observed",
            recovery_policy="branch_aware",
            collocation_diagnostics=True,
            least_squares_ftol=None,
        ),
        tmp_path,
        initial_parameters=raw["start"],
        initialization_plan=plan,
    )
    assert result["status"] == "complete", result
    assert result["validation"]["normalized_mse"] < 1e-5
    progress = read_json(tmp_path / "collocation/progress.json")
    assert progress["iterations"] and progress["checkpoints"]
    assert progress["iterations"][-1]["constraint_maximum"] >= 0
    assert not progress["physical_rollout_verified"]
    assert result["refinement"]["actual_residual_calls"] <= 60


def test_recovery_changes_an_invalid_ordinary_start(tmp_path, monkeypatch):
    from autoformalism.fitting import collocation_sensitivity as adapter

    raw, _, _, plan, _ = problem("shared")
    raw["candidate"]["state_equations"][0]["rhs"] = "rate*m**2 + gain*u01"
    model = compile_candidate(
        CandidateModel.model_validate(raw["candidate"]),
        ValidationContext.model_validate(raw["context"]),
    )
    monkeypatch.setattr(
        adapter, "bounded_latent_start", lambda *a, **kw: {"success": False}
    )
    result = fit_collocation_forward_sensitivity(
        model,
        unpack_split(raw["splits"]["train"]),
        unpack_split(raw["splits"]["val"]),
        CollocationSensitivityConfig(
            initializer_seconds=1,
            refinement_seconds=20,
            maximum_function_evaluations=25,
            recovery_policy="feasible",
            recovery_probe_seconds=2,
        ),
        tmp_path,
        initial_parameters={"rate": 1.0, "gain": 1.0, "init_m_value": 2.0},
        initialization_plan=plan,
    )
    r = result["refinement"]
    assert r["actual_residual_calls"] > 1 and r["actual_residual_calls"] <= 25
    assert r["screens"][0]["valid"] is False
    assert any(p["valid"] for p in r["screens"])
    assert r["parameters"] is not None


@pytest.mark.parametrize("policy", ["legacy", "feasible"])
def test_all_diagnostic_arms_enforce_actual_call_cap(tmp_path, monkeypatch, policy):
    from autoformalism.fitting import collocation_sensitivity as adapter

    raw, model, _, plan, _ = problem("shared")
    monkeypatch.setattr(
        adapter, "bounded_latent_start", lambda *args, **kwargs: {"success": False}
    )
    result = fit_collocation_forward_sensitivity(
        model,
        unpack_split(raw["splits"]["train"]),
        unpack_split(raw["splits"]["val"]),
        CollocationSensitivityConfig(
            initializer_seconds=1,
            refinement_seconds=10,
            maximum_function_evaluations=1,
            recovery_policy=policy,
            collocation_diagnostics=True,
        ),
        tmp_path,
        initial_parameters=raw["start"],
        initialization_plan=plan,
    )
    assert result["refinement"]["actual_residual_calls"] == 1
