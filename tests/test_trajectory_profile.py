"""Conditional affinity, fitted initials, bounded steps and reusable node graphs."""

from copy import deepcopy
from time import monotonic

import numpy as np
import pytest

from autoformalism.fitting import nonlinear_rollout
from autoformalism.fitting.confidence_checks import TrainingProblem
from autoformalism.fitting.trajectory_profile import TrajectoryProblem
from autoformalism.fitting.trajectory_profile_fit import optimize
from tests.profiled_fixture import make_base


def problem():
    base, _ = make_base()
    return TrainingProblem.model_validate(
        {k: v for k, v in base.items() if k != "nodes"} | {"incumbent": base["start"]}
    )


def setup():
    p = problem()
    oracle, profile, route = nonlinear_rollout.build(p)
    graph = TrajectoryProblem(oracle, p, 100, 4, 2, monotonic() + 30)
    return p, oracle, profile, route, graph


def test_strong_nonlinear_baseline_keeps_exact_terminal_profiling():
    p, oracle, profile, route, _ = setup()
    assert route["selected"] == "terminal_output_profile"
    assert set(profile.kernel.gain_names) == {"c", "d"}
    vector, residual, jac, _ = profile.evaluate(
        oracle.vector(p.start), monotonic() + 30
    )
    independent, _ = oracle.evaluate(vector, monotonic() + 30, jacobian=False)
    np.testing.assert_allclose(residual, independent, atol=2e-8)
    assert jac.shape == (len(residual), len(profile.outer))
    point = nonlinear_rollout.verify(
        oracle, oracle.parameters(vector), monotonic() + 30, lambda _: None
    )
    assert point["usable_for_retention"]
    assert point["verification"].startswith("original_equations_DOP853_vs_Radau")


def test_conditional_residual_affine_in_coefficients_not_hidden_initial():
    p, _, _, _, graph = setup()
    assert set(graph.partition["names"]) == {"a", "b", "c", "d"}
    assert graph.partition["outer_indices"] == [
        graph.system.names.index("init_z_value")
    ]
    z, q = graph.guesses(p.start)
    a, residual = graph.design(z, q, 100)
    changed = q.copy()
    changed[graph.linear] += 0.01
    np.testing.assert_allclose(
        graph.residual(z, changed, 100),
        residual + a @ (changed[graph.linear] - q[graph.linear]),
        atol=1e-10,
    )
    # The first boundary depends on the fitted initializer, never a frozen z0.
    q2 = q.copy()
    q2[graph.outer] += 0.2
    assert np.linalg.norm(np.asarray(graph.residual(z, q2, 100) - residual)) > 1e-3
    update, audit = graph.linear_step(z, q, 100)
    assert audit["accepted"]
    np.testing.assert_array_equal(update[graph.outer], q[graph.outer])
    assert audit["after_loss"] <= audit["before_loss"]
    graph.oracle.vector(graph.parameters(update))
    assert graph.audit["all_observations_retained"]
    assert graph.audit["input_interpolation_preserved"]


def test_optimizer_checkpoint_and_graph_reuse_do_not_reuse_trial_solution(tmp_path):
    p, _, _, _, graph = setup()
    z, q = graph.guesses(p.start)
    initial = float(np.sum(np.asarray(graph.residual(z, q, 100)) ** 2) / 2)
    saved = []
    first = optimize(
        graph,
        p.start,
        100,
        monotonic() + 15,
        saved.append,
        tmp_path,
        "test",
        cycles=2,
        block_iterations=8,
    )
    assert first["objective"] <= initial
    assert not first["rollout_verified"] and (tmp_path / "checkpoint.json").exists()
    reset_z, reset_q = graph.guesses(p.start)
    np.testing.assert_array_equal(reset_z, z)
    np.testing.assert_array_equal(reset_q, q)
    fine = TrajectoryProblem(graph.oracle, p, 500, 4, 2, monotonic() + 30)
    zz, qq = fine.guesses(first["parameters"], first["trajectories"])
    assert np.isfinite(zz).all()
    assert fine.parameters(qq) == pytest.approx(first["parameters"])


def test_feedback_rejects_terminal_profile_without_rejecting_joint_rollout():
    p = problem().model_dump(mode="json")
    p["request"]["base_candidate"]["state_equations"][1]["rhs"] = "-a*z+u+y"
    _, profile, route = nonlinear_rollout.build(TrainingProblem.model_validate(p))
    assert profile is None and route["selected"] == "joint_rollout"
    assert "feeds back" in route["inapplicable"][-1]["reason"]


def test_training_schema_rejects_evaluator_and_validation():
    raw = problem().model_dump(mode="json")
    with pytest.raises(ValueError):
        TrainingProblem.model_validate(raw | {"reference_parameters": {"a": 0.7}})
    raw = deepcopy(raw)
    raw["training"]["name"] = "val"
    with pytest.raises(ValueError):
        TrainingProblem.model_validate(raw)
