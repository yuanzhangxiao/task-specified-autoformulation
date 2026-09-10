"""Conditional linearity, physical domains and paired initializer checkpoints."""

from time import monotonic

import numpy as np
import pytest

pytest.importorskip("casadi")

from autoformalism.data import DatasetSplit, SplitName, Trajectory
from autoformalism.expressions import compile_candidate
from autoformalism.fitting.alternating_probe import bounded_separable_start
from autoformalism.fitting.sensitivity_probe import SymbolicODE
from autoformalism.fitting.separable_collocation import (
    CollocationProblem,
    LiftedSystem,
    linear_update,
)
from autoformalism.fitting.stagnation import RolloutOracle
from autoformalism.rebuttal.fitter_recovery import CONTEXT, recovery_candidate
from tests.test_fitter_methods import affine_training, small_plan


def hidden_training():
    system = SymbolicODE(compile_candidate(recovery_candidate(), CONTEXT))
    time = np.arange(11) * 0.2
    row = Trajectory(
        "training", time, {"v01": np.sin(time)}, {}, {"u01": np.sin(time * 2)}, {}, {}
    )
    return system, DatasetSplit(SplitName.TRAIN, (row,), "hidden-input-only")


def layout(system, training, tmp_path):
    return RolloutOracle(
        system.model,
        training,
        {"v01": 1},
        small_plan().settings(),
        tmp_path / "layout",
        monotonic() + 30,
    )


def test_lifting_and_rate_conversion_expose_full_coefficient_block(tmp_path):
    system, training = hidden_training()
    assert not system.affine
    problem = CollocationProblem(
        system, training, np.ones(7), 1, small_plan().settings(), 100, monotonic() + 20
    )
    oracle = layout(system, training, tmp_path)
    low, high = problem.lifted.bounds(oracle.lower, oracle.upper)
    beta = np.arange(1, 8) / 3
    matrix, target = problem.design(problem.z0)
    expected = np.asarray(matrix) @ beta - np.asarray(target).ravel()
    actual = np.asarray(problem.residual(problem.z0, beta)).ravel()
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    np.testing.assert_allclose(
        list(problem.lifted.parameters(problem.lifted.coefficients(beta)).values()),
        beta,
    )
    assert (
        len(problem.z0) == 81
    )  # 3 states + output at two stages, plus initial output.
    assert low[system.names.index("c")] == -np.inf
    assert low[system.names.index("tau")] > 0
    with pytest.raises(ValueError, match="invalid physical"):
        problem.lifted.parameters(np.zeros(7))
    proposal, report = linear_update(problem, problem.z0, low, high)
    assert report["success"], report
    assert problem.loss(problem.z0, proposal) <= problem.loss(problem.z0, problem.beta0)


def test_nonseparable_internal_shape_parameter_is_rejected():
    from autoformalism.rebuttal.fitter_methods import affine_candidate
    from autoformalism.schemas import CandidateModel

    payload = affine_candidate().model_dump(mode="json")
    payload["state_equations"][0]["rhs"] = "-a*x + b*tanh(c*x) + d"
    system = SymbolicODE(
        compile_candidate(CandidateModel.model_validate(payload), CONTEXT)
    )
    with pytest.raises(ValueError, match="not affine"):
        LiftedSystem(system)


def test_validation_is_rejected_before_any_rollout():
    system, train = hidden_training()
    validation = DatasetSplit(SplitName.VALIDATION, train.trajectories, "validation")
    with pytest.raises(ValueError, match="training split"):
        CollocationProblem(
            system,
            validation,
            np.ones(7),
            1,
            small_plan().settings(),
            100,
            monotonic() + 2,
        )


def test_linear_solver_preserves_signs_and_reports_rank_deficiency():
    class Problem:
        def design(self, _):
            return np.array(
                [[1.0, 1.0, 1.0], [1.0, -1.0, -1.0], [1.0, 0.0, 0.0]]
            ), np.array([-2.0, -4.0, -3.0])

    beta, report = linear_update(
        Problem(), None, np.array([-np.inf, 0.0, 0.0]), np.full(3, np.inf)
    )
    assert report["success"] and report["rank"] == 2
    assert beta[0] == pytest.approx(-3)
    assert beta[1] >= 0 and beta[2] >= 0
    assert beta[1] + beta[2] == pytest.approx(1)


def test_native_joint_and_alternating_share_objective_and_resume(tmp_path, monkeypatch):
    system, training = affine_training(small_plan())
    oracle = layout(system, training, tmp_path)
    arguments = {
        "training": training,
        "lower": oracle.lower,
        "upper": oracle.upper,
        "start": np.ones(4),
        "scale": 1,
        "settings": small_plan().settings(),
        "seconds": 20,
        "penalty": 100,
        "cycles": 3,
        "state_iterations": 8,
        "joint_iterations": 40,
        "identity": "paired-test",
    }
    outcomes = []
    for method in ("joint_collocation", "alternating_collocation"):
        directory = tmp_path / method
        result = bounded_separable_start(
            system, **arguments, method=method, directory=directory
        )
        assert result["success"], result
        assert result["initializer_objective"] <= result["initial_objective"]
        assert not result["initial_conditions_optimized"]
        assert not result["hidden_labels_used"] and not result["rollout_verified"]
        assert all(
            e["after"] <= e["before"] for e in result["progress"] if e["accepted"]
        )
        outcomes.append(result)
    assert (
        outcomes[0]["initial_nodes_identity"] == outcomes[1]["initial_nodes_identity"]
    )
    assert outcomes[0]["initial_objective"] == outcomes[1]["initial_objective"]
    method = "alternating_collocation"
    resumed = bounded_separable_start(
        system, **arguments, method=method, directory=tmp_path / method
    )
    assert resumed["parameters"] == outcomes[1]["parameters"]
    assert resumed["seconds"] >= outcomes[1]["seconds"]
    with pytest.raises(ValueError, match="identity"):
        bounded_separable_start(
            system,
            **{**arguments, "penalty": 1000},
            method=method,
            directory=tmp_path / method,
        )
    # A native timeout after an accepted checkpoint must retain that point and
    # must not grant another initializer budget when the task is resumed.
    from autoformalism.fitting import alternating_probe
    from autoformalism.rebuttal.fitter_diagnostic import read_json, write_json

    saved_path = tmp_path / method / "checkpoint.json"
    saved = read_json(saved_path)
    saved["seconds"] = arguments["seconds"]
    write_json(saved_path, saved)

    def unexpected_restart(*args, **kwargs):
        raise AssertionError("exhausted initializer must not start new native work")

    monkeypatch.setattr(alternating_probe, "bounded_latent_start", unexpected_restart)
    recovered = bounded_separable_start(
        system, **arguments, method=method, directory=tmp_path / method
    )
    assert recovered["success"]
    assert recovered["acceptance"] == "last_accepted_point_after_native_limit"
    assert recovered["parameters"] == resumed["parameters"]
    assert recovered["seconds"] == arguments["seconds"]
    assert not recovered["last_nonlinear_block_success"]
    assert not recovered["rollout_verified"]
    path = tmp_path / method / resumed["checkpoint_array"]
    path.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="digest"):
        bounded_separable_start(
            system, **arguments, method=method, directory=tmp_path / method
        )


def test_rate_conversion_preserves_existing_small_denominator_guard():
    system, _training = hidden_training()
    lifted = LiftedSystem(system)
    physical = np.ones(7)
    physical[lifted.rates] = 1e-16
    beta = lifted.coefficients(physical)
    x, u, t = np.array([0.2, 0.3, 0.4]), np.array([0.5]), 0.1
    q = lifted.guess_processes(t, x, beta, u)
    rhs, _, observed = lifted.point(t, x, q, beta, u)
    np.testing.assert_allclose(rhs, system.rhs(t, x, physical, u), rtol=1e-13)
    np.testing.assert_allclose(observed, system.observe(t, x, physical, u), rtol=1e-13)
    np.testing.assert_allclose(beta[lifted.rates], 1e12)


def test_v4_matrix_and_stress_start_are_frozen(tmp_path):
    from pathlib import Path

    from autoformalism.rebuttal import fitter_methods as campaign
    from autoformalism.rebuttal.fitter_diagnostic import read_json

    path = Path(__file__).resolve().parents[1] / "configs/fitter_methods_v4.json"
    plan = campaign.MethodsPlan.model_validate(read_json(path))
    frozen = campaign.prepare_methods(plan, tmp_path)
    assert len(frozen["tasks"]) == 41
    assert sum(t["kind"] == "fit" for t in frozen["tasks"]) == 39
    assert not any(t["kind"] == "initializer" for t in frozen["tasks"])
    stress = [t for t in frozen["tasks"] if t.get("start_kind") == "stress"]
    assert len(stress) == 3
    assert all(campaign._start(frozen, t) == plan.stress_start for t in stress)
    with pytest.raises(ValueError, match="invalid prespecified"):
        campaign.MethodsPlan.model_validate(
            {**plan.model_dump(), "stress_start": {"k": -1}}
        )


def test_v4_paired_end_to_end_and_completed_fit_resume(tmp_path):
    from autoformalism.rebuttal import fitter_methods as campaign

    payload = small_plan().model_dump(mode="json")
    payload.update(
        protocol="fitter-methods-4",
        fit_seconds=60,
        initializer_seconds=15,
        alternating_cycles=3,
        state_step_iterations=8,
        initializer_iterations=40,
        collocation_penalty=10000,
    )
    plan = campaign.MethodsPlan.model_validate(payload)
    frozen = campaign.prepare_methods(plan, tmp_path)
    assert len(frozen["tasks"]) == 4
    assert campaign.execute_methods(tmp_path, 0)["status"] == "complete"
    for index in (1, 2, 3):
        result = campaign.execute_methods(tmp_path, index)
        assert result["status"] == "complete", result
        assert result["total_fit_seconds"] <= 60
        assert result["training_only_optimization"]
        assert not result["hidden_labels_used"]
        assert campaign.execute_methods(tmp_path, index) == result
    report = campaign.summarize_methods(tmp_path)["paired_summary"]
    assert report["pairing_checks"][0]["same_data_and_initializer"]
    assert len(report["block_diagnostics"]) == 2
    assert "Matched-formulation" in (tmp_path / "summary.md").read_text()
