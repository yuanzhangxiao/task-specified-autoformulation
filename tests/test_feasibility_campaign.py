"""Controlled-feature equality, source provenance and deterministic resume."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("casadi")

from autoformalism.fitting.collocation_sensitivity import CollocationSensitivityConfig
from autoformalism.rebuttal.feasibility_campaign import (
    FeasibilityExperiment,
    coupled_problem,
    execute,
    prepare,
    verify,
)
from autoformalism.rebuttal.fitter_diagnostic import read_json, sha256, write_json
from autoformalism.rebuttal.initialization_campaign import synthetic_problem
from autoformalism.staged_topology import content_hash


def plan():
    return FeasibilityExperiment(
        fit=CollocationSensitivityConfig(
            initializer_seconds=20,
            refinement_seconds=20,
            maximum_function_evaluations=40,
            collocation_node_start="rollout_or_observed",
            collocation_diagnostics=True,
            least_squares_ftol=None,
        )
    )


def test_controlled_coordinate_and_start_changes_preserve_observations():
    baseline, reference = coupled_problem("quadratic")
    for variant in ("scaled", "safe_start"):
        other, other_reference = coupled_problem(variant)
        assert reference == other_reference
        assert baseline["splits"] == other["splits"]
    safe, _ = coupled_problem("safe_start")
    assert (
        baseline["candidate"]["state_equations"] == safe["candidate"]["state_equations"]
    )
    assert baseline["start"] != safe["start"]
    scaled, _ = coupled_problem("scaled")
    assert baseline["start"] == scaled["start"]
    assert (
        baseline["candidate"]["state_equations"]
        != scaled["candidate"]["state_equations"]
    )


def test_prepare_freezes_30_control_arms_and_detects_drift(tmp_path):
    frozen = prepare(plan(), tmp_path)
    assert len(frozen["tasks"]) == 30
    assert sum(t["arm"] == "branch_aware" for t in frozen["tasks"]) == 4
    assert prepare(plan(), tmp_path) == frozen
    path = tmp_path / "problems/000.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="frozen input differs"):
        verify(tmp_path)


def test_public_import_preserves_frozen_initial_plan(tmp_path):
    source = tmp_path / "source"
    problem, _ = synthetic_problem("shared", 0.0, 0)
    path = source / "problems/000.json"
    write_json(path, problem)
    frozen = {
        "plan": {"protocol": "physical-initialization-comparison-1"},
        "cases": [{"index": 0, "synthetic": False, "label": "public"}],
        "assets": {"problems/000.json": sha256(path)},
        "test_data_opened": False,
    }
    write_json(source / "freeze.json", {**frozen, "identity": content_hash(frozen)})
    result = prepare(plan(), tmp_path / "out", source)
    assert len(result["tasks"]) == 32
    assert read_json(tmp_path / "out/problems/013.json") == problem
    write_json(path, {**problem, "start": {}})
    with pytest.raises(ValueError, match="public problem hash differs"):
        prepare(plan(), tmp_path / "out2", source)


def test_completed_fit_reused_and_stdlib_summary(tmp_path, monkeypatch):
    from autoformalism.rebuttal import feasibility_campaign as campaign

    prepare(plan(), tmp_path)
    result = execute(tmp_path, 1)
    assert result["status"] == "complete", result
    assert result["recovered"]

    def forbidden(*args, **kwargs):
        raise AssertionError("completed fit must not rerun")

    monkeypatch.setattr(campaign, "fit_collocation_forward_sensitivity", forbidden)
    assert execute(tmp_path, 1) == result
    script = Path(__file__).resolve().parents[1] / "scripts/run_feasibility_campaign.py"
    subprocess.run(
        [sys.executable, "-S", str(script), "summarize", "--output", str(tmp_path)],
        env={**os.environ, "PYTHONPATH": ""},
        check=True,
        capture_output=True,
    )
    summary = read_json(tmp_path / "summary.json")
    assert summary["statuses"] == {"complete": 1, "missing": 29}
    assert len(json.loads((tmp_path / "diagnostics.json").read_text())) == 30


def test_interrupted_fit_does_not_gain_budget(tmp_path):
    frozen = prepare(plan(), tmp_path)
    identity = content_hash([frozen["identity"], frozen["tasks"][0]])
    write_json(tmp_path / "results/task_000/fit_started.json", {"identity": identity})
    result = execute(tmp_path, 0)
    assert result["status"] == "interrupted"
    assert "fit" not in result
