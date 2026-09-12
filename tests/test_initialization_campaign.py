"""Frozen experiment provenance, independent replay and checkpoint behavior."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("casadi")

from autoformalism.fitting.collocation_sensitivity import CollocationSensitivityConfig
from autoformalism.rebuttal.fitter_diagnostic import read_json, write_json
from autoformalism.rebuttal.initialization_campaign import (
    InitializationExperiment,
    execute,
    prepare,
    verify,
)


def plan():
    return InitializationExperiment(
        starts=1,
        noise=(0.0,),
        fit=CollocationSensitivityConfig(
            initializer_seconds=25,
            refinement_seconds=25,
            maximum_function_evaluations=60,
            collocation_node_start="rollout_or_observed",
            least_squares_ftol=None,
        ),
    )


def test_freeze_detects_changed_input_and_resume_plan(tmp_path):
    frozen = prepare(plan(), tmp_path)
    assert len(frozen["tasks"]) == 8
    assert prepare(plan(), tmp_path) == frozen
    file = tmp_path / "problems/000.json"
    original = read_json(file)
    changed = {**original, "start": {"rate": 10.0, "gain": 10.0}}
    write_json(file, changed)
    with pytest.raises(ValueError, match="input differs"):
        verify(tmp_path)


def test_fit_replay_and_completed_resume_are_frozen(tmp_path, monkeypatch):
    from autoformalism.rebuttal import initialization_campaign as campaign

    prepare(plan(), tmp_path)
    result = execute(tmp_path, 1)
    assert result["status"] == "complete", result
    assert result["recovered"]
    assert result["fit"]["initializer"]["initial_conditions_optimized"]

    def forbidden(*args, **kwargs):
        raise AssertionError("completed checkpoint must not refit")

    monkeypatch.setattr(campaign, "fit_collocation_forward_sensitivity", forbidden)
    assert execute(tmp_path, 1) == result
    env = {**os.environ, "PYTHONPATH": ""}
    script = (
        Path(__file__).resolve().parents[1] / "scripts/run_initialization_campaign.py"
    )
    # -S disables site packages: the summary must remain standard-library-only.
    subprocess.run(
        [sys.executable, "-S", str(script), "summarize", "--output", str(tmp_path)],
        env=env,
        check=True,
        capture_output=True,
    )
    summary = read_json(tmp_path / "summary.json")
    assert summary["statuses"] == {"complete": 1, "missing": 7}


def test_interrupted_native_fit_never_receives_fresh_budget(tmp_path):
    frozen = prepare(plan(), tmp_path)
    write_json(tmp_path / "results/task_000/fit_started.json", {"identity": "started"})
    result = execute(tmp_path, 0)
    assert result["status"] == "interrupted"
    assert not result.get("fit")
    assert frozen["test_data_opened"] is False


def test_public_source_requires_verified_problem_hash(tmp_path):
    from autoformalism.rebuttal.fitter_diagnostic import sha256
    from autoformalism.rebuttal.initialization_campaign import synthetic_problem
    from autoformalism.staged_topology import content_hash

    source = tmp_path / "source"
    problem, _ = synthetic_problem("shared", 0.0, 0)
    problem_file = source / "problems/000.json"
    problem.pop("initialization_plan")
    write_json(problem_file, problem)
    freeze = {
        "plan": {"protocol": "piecewise-fitting-comparison-1"},
        "cases": [{"index": 0, "synthetic": False, "label": "public_control"}],
        "assets": {"problems/000.json": sha256(problem_file)},
        "test_data_opened": False,
    }
    write_json(source / "freeze.json", {**freeze, "identity": content_hash(freeze)})
    frozen = prepare(plan(), tmp_path / "out", source)
    assert len(frozen["tasks"]) == 10
    public = read_json(tmp_path / "out/problems/004.json")
    assert public["initialization_plan"]["rules"]["m"]["initial"]["guess"] == 0
    problem_file.write_text(json.dumps({**problem, "start": {}}))
    with pytest.raises(ValueError, match="hash differs"):
        prepare(plan(), tmp_path / "out2", source)
