"""Mesh ablations, provenance and bounded deterministic resume."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from autoformalism.rebuttal.fitter_diagnostic import read_json, sha256, write_json
from autoformalism.staged_topology import content_hash

pytest.importorskip("casadi")

from autoformalism.fitting.collocation_sensitivity import CollocationSensitivityConfig
from autoformalism.rebuttal.initialization_campaign import synthetic_problem
from autoformalism.rebuttal.mesh_campaign import (
    MeshExperiment,
    arm_config,
    execute,
    matched_problem,
    prepare,
    verify,
)
from autoformalism.rebuttal.piecewise_campaign import unpack_split


def source_at(root, count=4):
    problem, _ = synthetic_problem("shared", 0.0, 0)
    write_json(root / "problems/000.json", problem)
    frozen = {
        "plan": {"protocol": "fitter-feasibility-comparison-1"},
        "cases": [
            {"index": 0, "synthetic": False, "label": f"public{i}"}
            for i in range(count)
        ],
        "assets": {"problems/000.json": sha256(root / "problems/000.json")},
        "test_data_opened": False,
    }
    write_json(root / "freeze.json", {**frozen, "identity": content_hash(frozen)})
    return problem


def test_ablation_changes_only_declared_factors():
    plan = MeshExperiment()
    previous = arm_config(plan, "previous", False).model_dump()
    handoff = arm_config(plan, "handoff", False).model_dump()
    assert {k for k in previous if previous[k] != handoff[k]} == {
        "recovery_handoff",
        "sensitivity_invalid_trials",
    }
    mapped = arm_config(plan, "mapped", False).model_dump()
    assert {k for k in handoff if handoff[k] != mapped[k]} == {"collocation_assembly"}
    coarse = arm_config(plan, "mesh12000", False).model_dump()
    assert {k for k in mapped if mapped[k] != coarse[k]} == {
        "collocation_target_variables"
    }
    more = arm_config(plan, "more_time", False).model_dump()
    assert {k for k in mapped if mapped[k] != more[k]} == {
        "initializer_seconds",
        "collocation_maximum_iterations",
    }
    assert arm_config(plan, "previous", True).recovery_policy == "branch_aware"
    with pytest.raises(ValueError, match="unknown comparison"):
        arm_config(plan, "unknown", False)
    with pytest.raises(ValueError, match="variable target"):
        CollocationSensitivityConfig(collocation_target_variables=100)


def test_public_freeze_47_arms_and_detects_drift(tmp_path):
    source, output = tmp_path / "source", tmp_path / "out"
    problem = source_at(source)
    frozen = prepare(MeshExperiment(), output, source)
    assert len(frozen["tasks"]) == 47
    assert len(frozen["cases"]) == 10
    assert frozen == prepare(MeshExperiment(), output, source)
    assert read_json(output / "problems/004.json") == problem
    assert verify(output) == frozen
    write_json(output / "problems/004.json", {**problem, "start": {}})
    with pytest.raises(ValueError, match="frozen input differs"):
        verify(output)
    write_json(source / "problems/000.json", {})
    with pytest.raises(ValueError, match="public problem hash differs"):
        prepare(MeshExperiment(), tmp_path / "changed", source)


def test_matched_controls_preserve_shape_and_coordinate_equivalence(tmp_path):
    template = source_at(tmp_path)
    ordinary, clean = matched_problem("quadratic", template)
    scaled, other_clean = matched_problem("scaled", template)
    assert ordinary["splits"] == scaled["splits"]
    assert clean == other_clean
    for name in ("train", "val"):
        a, b = [unpack_split(x["splits"][name]) for x in (template, ordinary)]
        assert len(a.trajectories) == len(b.trajectories)
        for original, control in zip(a.trajectories, b.trajectories, strict=True):
            np.testing.assert_array_equal(original.time, control.time)
            assert np.ptp(control.targets["v01"]) > 0.5
    assert ordinary["feature_control"]["public_forcing_matched"] is False


def test_interrupted_checkpoint_retains_budget_and_stdlib_summary(tmp_path):
    frozen = prepare(MeshExperiment(), tmp_path)
    task = frozen["tasks"][0]
    identity = content_hash([frozen["identity"], task])
    write_json(tmp_path / "results/task_000/fit_started.json", {"identity": identity})
    result = execute(tmp_path, 0)
    assert result["status"] == "interrupted"
    assert execute(tmp_path, 0) == result
    script = Path(__file__).resolve().parents[1] / "scripts/run_mesh_campaign.py"
    subprocess.run(
        [sys.executable, "-S", str(script), "summarize", "--output", str(tmp_path)],
        env={**os.environ, "PYTHONPATH": ""},
        check=True,
        capture_output=True,
    )
    assert read_json(tmp_path / "summary.json")["statuses"] == {
        "interrupted": 1,
        "missing": 11,
    }


def test_saved_fit_replay_reused_without_optimizer(tmp_path, monkeypatch):
    from autoformalism.rebuttal import mesh_campaign as campaign

    frozen = prepare(MeshExperiment(), tmp_path)
    task = frozen["tasks"][2]
    identity = content_hash([frozen["identity"], task])
    root = tmp_path / "results/task_002"
    fit = {"status": "fit_failed", "parameters": None}
    write_json(root / "fit.json", {"identity": identity, "fit": fit})

    def forbidden(*args, **kwargs):
        raise AssertionError("saved fit must not rerun")

    monkeypatch.setattr(campaign, "fit_collocation_forward_sensitivity", forbidden)
    result = execute(tmp_path, 2)
    assert result["status"] == "fit_failed"
    assert execute(tmp_path, 2) == result


def test_launcher_dependencies_and_duplicate_submission(tmp_path):
    """Exercise the real launcher using local git/sbatch stubs; submit no jobs."""
    source, output = tmp_path / "source", tmp_path / "out"
    source_at(source)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    git = bin_dir / "git"
    git.write_text(
        '#!/bin/sh\ncase "$1" in\nstatus) exit 0;;\n'
        "rev-parse) echo test-commit;;\n*) exit 2;;\nesac\n"
    )
    batch = bin_dir / "sbatch"
    batch.write_text(
        "#!/bin/sh\n"
        'printf "%s\\n" "$*" >> "$AF_BATCH_TEST_LOG"\n'
        'wc -l < "$AF_BATCH_TEST_LOG" | tr -d " "\n'
    )
    git.chmod(0o755)
    batch.chmod(0o755)
    root = Path(__file__).resolve().parents[1]
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "AF_REPO_ROOT": str(root),
        "AF_PYTHON": sys.executable,
        "AF_CASADI_ROOT": str(tmp_path),
        "AF_SOURCE_ROOT": str(source),
        "AF_OUTPUT_ROOT": str(output),
        "AF_ARRAY_CONCURRENCY": "2",
        "AF_BATCH_TEST_LOG": str(tmp_path / "batch.log"),
    }
    command = ["bash", str(root / "scripts/hpc/submit_mesh_delta.sh")]
    subprocess.run(command, env=env, check=True, capture_output=True)
    log = (tmp_path / "batch.log").read_text().splitlines()
    assert len(log) == 3
    assert "--dependency=afterok:1" in log[1]
    assert "--array=0-46%2" in log[1]
    assert "--dependency=afterany:2" in log[2]
    assert read_json(output / "submission.json")["submission_complete"] is True
    subprocess.run(command, env=env, check=True, capture_output=True)
    assert (tmp_path / "batch.log").read_text().splitlines() == log
