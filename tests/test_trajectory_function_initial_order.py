"""The function campaign reports time-zero order without changing fitting."""

import pytest

from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.research import trajectory_function_fitting as fitting
from autoformalism.staged_topology import content_hash
from scripts import trajectory_functions as trial
from tests.test_trajectory_function_fitting import problem


@pytest.fixture
def qualified_fit(tmp_path, monkeypatch):
    graph, formula, rows, _ = problem()
    formula["equations"]["output"] = "gain*y+sin(u)"
    formula["parameters"]["gain"] = -1.0
    case = {
        "evidence": {
            "context": {"targets": ["output"]},
            "contract": {"absolute_error_bounds": {"output": 0.01}},
            "trajectories": rows,
        }
    }
    row = {
        "case": "c",
        "dataset": "d",
        "structure_sha256": content_hash(graph),
        "graph": graph,
    }
    data = {
        "train": {"d": rows},
        "cases": {"c": case},
        "graphs": [row],
        "profile": trial.FIXED,
    }
    budget = {"starts": 1, "wall_seconds": 30, "max_evaluations": 100}
    plan = {"artifact_sha256": "plan", "maximum_fit_slots": 6, "fit_budget": budget}
    monkeypatch.setattr(trial, "load", lambda _: (plan, data))
    result = sealed_write(
        tmp_path / "candidates/000.json",
        {
            "plan": "plan",
            "index": 0,
            "status": "eligible",
            "graph": content_hash(graph),
            "formula": formula,
        },
    )
    sealed_write(
        tmp_path / "generation.json",
        {
            "plan": "plan",
            "candidates": [result["artifact_sha256"]] * 6,
        },
    )
    return tmp_path, graph, formula, case


def test_fit_persists_start_before_optimization_and_rechecks_selected_vector(
    qualified_fit,
    monkeypatch,
):
    root, _, _, _ = qualified_fit
    original = fitting.fit_formula
    calls = []

    def bounded(*args, **kwargs):
        saved = sealed_read(root / "fits/000/reservation.json")
        assert (
            saved["initial_order_start"]["targets"]["output"]["status"]
            == "ordering_conflict"
        )
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(fitting, "fit_formula", bounded)
    saved = trial.fit(root, 0)
    assert trial.fit(root, 0) == saved
    result = saved["result"]
    assert result["status"] == "fitted" and result["training_nmse"] < 1e-15
    assert result["budget"] == calls[0] and len(calls) == 1
    reports = result["initial_algebraic_order"]
    assert reports["start"]["targets"]["output"]["reversed_pairs"] == 1
    final = reports["endpoint"]["targets"]["output"]
    assert final["status"] == "no_ordering_conflict" and final["max_abs_error"] < 1e-6


def test_no_endpoint_or_missing_public_bound_is_reported(qualified_fit, monkeypatch):
    root, graph, formula, case = qualified_fit
    case["evidence"]["contract"].clear()
    advisory = trial.initial_order_advisory(graph, formula, case)
    assert advisory["status"] == "unverified"
    assert "bound" in advisory["targets"]["output"]["reason"]
    monkeypatch.setattr(
        fitting,
        "fit_formula",
        lambda *args, **kwargs: {
            "status": "fit_failed",
            "parameters": None,
        },
    )
    failed = trial.fit(root, 0)["result"]
    assert failed["status"] == "fit_failed"
    assert failed["initial_algebraic_order"]["start"] == advisory
    assert failed["initial_algebraic_order"]["endpoint"]["status"] == "unverified"


def test_nonzero_initial_time_or_no_comparable_pair_abstains(qualified_fit):
    _, graph, formula, case = qualified_fit
    case["evidence"]["trajectories"] = case["evidence"]["trajectories"][:1]
    result = trial.initial_order_advisory(graph, formula, case)["targets"]["output"]
    assert result["status"] == "unverified"
    assert result["event_count"] == 1 and result["max_abs_error"] > 0
    case["evidence"]["trajectories"][0]["time"][0] = -1
    result = trial.initial_order_advisory(graph, formula, case)["targets"]["output"]
    assert result["status"] == "unverified" and "zero" in result["reason"]
