"""Bounded campaign execution, source isolation, and immutable resume behavior."""

from __future__ import annotations

import importlib.util
import io
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("casadi")

from autoformalism.fitting.conditional_collocation import common_initialization
from autoformalism.fitting.conditional_optimizer import load_point, save_point
from autoformalism.fitting.conditional_scaling import (
    partition_and_exponents,
    system_for,
)
from autoformalism.rebuttal.fitter_diagnostic import (
    _write_bytes,
    read_json,
    sha256,
    write_json,
)
from autoformalism.rebuttal.fitter_methods import identity_runtime
from autoformalism.rebuttal.piecewise_campaign import unpack_split
from autoformalism.rebuttal.scaled_alternating import (
    ScaledAlternatingPlan,
    code_identity,
    common_data,
    prepare,
    verify,
)
from autoformalism.rebuttal.scaled_alternating_gate import small_problem
from autoformalism.rebuttal.scaled_alternating_stages import best_of, interrupted_result
from autoformalism.staged_topology import content_hash

REPO = Path(__file__).resolve().parents[1]


def script(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def freeze_control(output):
    """A small explicit plumbing fixture, never a production reference source."""
    problem = small_problem()
    system = system_for(problem)
    train = unpack_split(problem["splits"]["train"])
    plan = ScaledAlternatingPlan(
        initializer_seconds=10,
        screen_seconds=10,
        refinement_seconds=20,
        replay_seconds=10,
        initializer_iterations=20,
        alternating_cycles=2,
        block_iterations=10,
        target_variables=300,
        minimum_intervals=8,
        warmup_seconds=0.001,
    )
    write_json(output / "problem.json", problem)
    frozen = {
        "tasks": ["joint", "alternating"],
        "plan": plan.model_dump(mode="json"),
        "assets": {"problem.json": sha256(output / "problem.json")},
        "code": code_identity(),
        "runtime": identity_runtime(),
        "test_fixture": "not a production reference",
    }
    frozen["identity"] = content_hash(frozen)
    write_json(output / "freeze.json", frozen)
    common, nodes = common_initialization(
        system,
        train,
        np.ones(5),
        plan.settings(),
        partition_and_exponents(system),
        target_variables=300,
        minimum_intervals=8,
        warmup_seconds=0.001,
    )
    common.pop("identity")
    common["parameter_names"] = list(system.names)
    common["identity"] = content_hash(common)
    write_json(output / "common.json", common)
    buf = io.BytesIO()
    np.save(buf, nodes, allow_pickle=False)
    _write_bytes(output / "nodes.npy", buf.getvalue())
    write_json(
        output / "gate/result.json",
        {
            "identity": frozen["identity"],
            "pass": True,
            "common_identity": common["identity"],
            "nodes_sha256": sha256(output / "nodes.npy"),
        },
    )
    return frozen


@pytest.mark.parametrize("index", [0, 1])
def test_real_stage_supervision_and_resume(tmp_path, index):
    freeze_control(tmp_path)
    env = {
        **os.environ,
        "PYTHONPATH": str(REPO / "src"),
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    command = [
        sys.executable,
        str(REPO / "scripts/run_scaled_alternating.py"),
        "run",
        "--output",
        str(tmp_path),
        "--task-index",
        str(index),
    ]
    completed = subprocess.run(
        command, cwd=REPO, env=env, capture_output=True, text=True, timeout=100
    )
    assert completed.returncode == 0, (
        completed.stdout
        + completed.stderr
        + (tmp_path / f"results/task_{index:03d}/worker.log").read_text()
    )
    root = tmp_path / f"results/task_{index:03d}"
    result = read_json(root / "result.json")
    assert result["status"] == "complete"
    assert result["strict"]
    assert result["refinement"]["calls"] <= 118
    assert result["screen"]["calls"] <= 2
    assert not result["hidden_trajectory_labels_used"]
    before = {
        str(p): sha256(p) for p in root.rglob("*.json") if p.name != "supervisor.json"
    }
    assert (
        subprocess.run(
            command, cwd=REPO, env=env, capture_output=True, timeout=20
        ).returncode
        == 0
    )
    assert before == {
        str(p): sha256(p) for p in root.rglob("*.json") if p.name != "supervisor.json"
    }
    report = script("scaled_alternating_report").report(tmp_path)
    assert report["counts"] == {"complete": 1, "missing": 1}


def test_interrupted_handoff_retains_incumbent_and_consumes_stage(tmp_path):
    point = {"parameters": {"k": 1.0}, "cost": 0.2, "call": 1}
    write_json(tmp_path / "refinement/calls/best_evaluated.json", point)
    interrupted = interrupted_result(tmp_path, "refinement", "case", 124)
    assert interrupted["status"] == "interrupted" and interrupted["best"] == point
    assert best_of({"best": {"cost": 0.1}}, interrupted)["cost"] == 0.1
    assert best_of({"best": {"cost": 0.3}}, interrupted) == point
    save_point(
        tmp_path / "initializer/native",
        "case",
        np.ones(2),
        np.ones(1),
        {"parameters": {"k": 1.0}},
    )
    saved = interrupted_result(tmp_path, "initializer", "case", 124)
    assert saved["initializer"]["parameters"] == {"k": 1.0}
    assert "no fresh" in saved["message"].lower()
    record = read_json(tmp_path / "initializer/native/checkpoint.json")
    record["parameters"]["k"] = 2.0
    write_json(tmp_path / "initializer/native/checkpoint.json", record)
    with pytest.raises(ValueError, match="metadata digest"):
        load_point(tmp_path / "initializer/native", "case")


def test_stale_assets_common_nodes_and_budget_expansion_rejected(tmp_path):
    frozen = freeze_control(tmp_path)
    verify(tmp_path)
    common_data(tmp_path, frozen)
    with pytest.raises(ValueError, match="iteration"):
        ScaledAlternatingPlan(block_iterations=10)
    with pytest.raises(ValueError):
        ScaledAlternatingPlan(refinement_seconds=4000)
    data = read_json(tmp_path / "problem.json")
    data["start"]["r"] = 2.0
    write_json(tmp_path / "problem.json", data)
    with pytest.raises(ValueError, match="asset changed"):
        verify(tmp_path)
    (tmp_path / "nodes.npy").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="common initialization"):
        common_data(tmp_path, frozen)


def test_source_contract_rejects_wrong_family_and_test_access(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    old = {
        "plan": {"protocol": "other"},
        "test_data_opened": False,
        "proposer_access": False,
    }
    old["identity"] = content_hash(old)
    write_json(source / "freeze.json", old)
    with pytest.raises(ValueError, match="isolated"):
        prepare(source, tmp_path / "output", ScaledAlternatingPlan())
    old.pop("identity")
    old["plan"]["protocol"] = "final-fitter-alternatives-1"
    old["test_data_opened"] = True
    old["identity"] = content_hash(old)
    write_json(source / "freeze.json", old)
    with pytest.raises(ValueError, match="isolated"):
        prepare(source, tmp_path / "output", ScaledAlternatingPlan())
    with pytest.raises(ValueError, match="separate"):
        prepare(source, source / "output", ScaledAlternatingPlan())


def test_supervisor_hard_kills_child(tmp_path, monkeypatch):
    runner = script("run_scaled_alternating")
    fake = tmp_path / "child.py"
    fake.write_text("import time\ntime.sleep(10)\n")
    monkeypatch.setattr(runner, "__file__", str(fake))
    with (tmp_path / "child.log").open("w") as log:
        assert runner.child([], log, 0.1, group=True) == 124


def test_launcher_records_two_tasks_without_duplicate_submission(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    output = tmp_path / "output"
    binpath = tmp_path / "bin"
    binpath.mkdir()
    launcher = REPO / "scripts/hpc/submit_scaled_alternating_delta.sh"
    fakegit = binpath / "git"
    fakegit.write_text(
        '#!/bin/bash\nif [[ "$1" == rev-parse ]]; then echo pinned; fi\n'
    )
    fakegit.chmod(0o755)
    fakepython = binpath / "python"
    fakepython.write_text(
        "#!/bin/bash\n"
        'if [[ "$1" == scripts/run_scaled_alternating.py ]]; then exit 0; fi\n'
        f'exec "{sys.executable}" "$@"\n'
    )
    fakepython.chmod(0o755)
    sbatch = binpath / "sbatch"
    sbatch.write_text('#!/bin/bash\necho "$*" >> "$JOB_LOG"\necho 12345\n')
    sbatch.chmod(0o755)
    env = {
        **os.environ,
        "PATH": str(binpath) + ":" + os.environ["PATH"],
        "AF_REPO_ROOT": str(repo),
        "AF_PYTHON": str(fakepython),
        "AF_OUTPUT_ROOT": str(output),
        "AF_ARRAY_CONCURRENCY": "2",
        "JOB_LOG": str(tmp_path / "jobs"),
        "AF_CONFIG": "unused",
    }
    for _ in range(2):
        done = subprocess.run(
            ["bash", str(launcher)], env=env, capture_output=True, text=True
        )
        assert done.returncode == 0, done.stderr
    jobs = (tmp_path / "jobs").read_text().splitlines()
    assert len(jobs) == 3 and "--array=0-1%2" in jobs[1]
    record = json.loads((output / "submission.json").read_text())
    assert record["fit_tasks"] == 2 and record["submission_complete"]


def test_positive_source_provenance_and_changed_start_rejected(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from autoformalism.rebuttal import scaled_alternating as module

    source = tmp_path / "source"
    problem = small_problem()
    # Mock only the production size gate to exercise packaging with a small fixture.
    monkeypatch.setattr(
        module,
        "system_for",
        lambda _: SimpleNamespace(names=tuple(range(48)), state_count=6),
    )
    write_json(source / "problems/004.json", problem)
    task = {
        "index": 4,
        "family": "free_shapes",
        "strategy": "collocation_exact",
        "oracle_weight_start": False,
        "oracle_shapes": False,
        "oracle_initials": True,
        "starts": [problem["start"]],
    }
    old = {
        "plan": {"protocol": "final-fitter-alternatives-1"},
        "tasks": [task],
        "runtime": identity_runtime(),
        "test_data_opened": False,
        "proposer_access": False,
        "assets": {"problems/004.json": sha256(source / "problems/004.json")},
    }
    old["identity"] = content_hash(old)
    write_json(source / "freeze.json", old)
    write_json(source / "gate/result.json", {"identity": old["identity"], "pass": True})
    output = tmp_path / "output"
    frozen = prepare(source, output, ScaledAlternatingPlan())
    assert prepare(source, output, ScaledAlternatingPlan()) == frozen
    assert set(frozen["assets"]) == {
        "problem.json",
        "provenance/source_freeze.json",
        "provenance/source_gate.json",
    }
    assert (
        frozen["tasks"] == ["joint", "alternating"]
        and not frozen["historical_estimates_used"]
    )
    old.pop("identity")
    task["starts"] = [{**problem["start"], "r": 3.0}]
    old["identity"] = content_hash(old)
    write_json(source / "freeze.json", old)
    write_json(source / "gate/result.json", {"identity": old["identity"], "pass": True})
    with pytest.raises(ValueError, match="ordinary start"):
        prepare(source, tmp_path / "changed", ScaledAlternatingPlan())


def test_numerical_failure_not_reported_as_interruption(tmp_path):
    from autoformalism.rebuttal.scaled_alternating_stages import failed_result

    result = failed_result(
        tmp_path,
        "replay-val-BDF",
        "case",
        1,
        status="numerical_failed",
        message="integration failed at a finite step",
    )
    assert result["status"] == "numerical_failed" and result["message"].startswith(
        "integration"
    )
