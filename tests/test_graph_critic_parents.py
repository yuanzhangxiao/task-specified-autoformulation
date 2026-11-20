"""Frozen graph critic intake preserves the original states, roots, and trial scope."""

from copy import deepcopy

import pytest

from scripts import graph_critic_parents as intake
from tests.test_prefix_initialization import problem, training


def test_intake_restores_per_trajectory_roots_and_linked_initials():
    p = problem()
    base = intake.compile_candidate(p.base_candidate, p.context)
    lowered, _, info = intake.apply_initialization_graph(base, p.graph)
    split = training()
    vector = {
        "theta": 0.7,
        "init_z_phi": 1.3,
        "trajectory:a:init_r_value": 0.4,
        "trajectory:b:init_r_value": 1.7,
    }
    point = {"training_nmse": 0.0, "parameters": vector}
    roots = info["trajectory_parameter_names"]
    actual = intake._materialize_point(base, lowered, roots, split, point, {})
    assert actual.parameters == {"theta": 0.7}
    for name, root in (("a", 0.4), ("b", 1.7)):
        realized = dict(zip(base.state_names, actual.initial_states[name], strict=True))
        assert realized == pytest.approx({"x": 1.0, "z": 1.3 * root, "r": root})
    with pytest.raises(ValueError, match="public training"):
        intake._materialize_point(
            base, lowered, roots, split.model_copy(update={"name": "val"}), point, {}
        )
    del vector["trajectory:a:init_r_value"]
    with pytest.raises(ValueError, match="incomplete"):
        intake._materialize_point(base, lowered, roots, split, point, {})


def test_trial_control_is_independent_of_better_saved_incumbent():
    name, label = "candidate-10", "saturation-0p1"
    task = {
        "candidate_id": name,
        "seed": label,
        "problem_sha256": "model",
        "experiment": {"fixed": {"k": 0.1}},
    }
    trial = {"original_equations_verified": True, "parameters": {"k": 0.1}}
    terminal = {
        "identity": {"task": task},
        "status": "complete",
        "test_data_opened": False,
        "trial_endpoint": trial,
        "retained": {"parameters": {"k": 1.0}},
        "search_termination": "numerical_active_wall_limit",
        "search_budget_exhausted": True,
    }
    point, status = intake._endpoint(name, label, terminal, "model")
    assert point == trial
    assert (
        status["budget_exhausted"] and status["search_scope"] == "fixed_coordinate_fit"
    )
    bad = deepcopy(terminal)
    bad["trial_endpoint"]["parameters"]["k"] = 0.2
    with pytest.raises(ValueError, match="different restricted fit"):
        intake._endpoint(name, label, bad, "model")


def test_roster_disallows_revision_or_traversal():
    with pytest.raises(ValueError, match="unknown frozen parent"):
        intake.source_paths("../c15-2-seed109-correction3")


def test_sibling_control_is_labeled_separately_and_uses_own_verified_fit():
    from autoformalism.fitting.public_fitting import content_sha256

    name = intake.TRANSFER_CONTROL
    assert name not in intake.ROSTER
    assert {p.name for p in intake.source_paths(name)} == {
        "orion-fitting-handoff.zip",
        "fitted-models.json",
        "result.json",
    }
    parameters = {"k": 1.2}
    point = {
        "parameters": parameters,
        "parameter_sha256": content_sha256(parameters),
        "two_original_solvers_verified": True,
    }
    last = {"solver": {"message": "ftol"}, "call_cap": 2000, "calls": 291}
    terminal = {
        "status": "complete",
        "identity": {"task": "long-memory"},
        "branch": name,
        "best_admissible": point,
        "history": [last],
    }
    got, status = intake._endpoint(name, "long-memory", terminal, "model")
    assert got == point and not status["budget_exhausted"]
    with pytest.raises(ValueError, match="sibling fit task differs"):
        intake._endpoint(name, "short-memory", terminal, "model")
