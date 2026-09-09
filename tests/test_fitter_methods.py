"""Method eligibility, train-only initializers, matched fits and bounded resume."""

from __future__ import annotations

import os
from pathlib import Path
from time import monotonic

import numpy as np
import pytest

pytest.importorskip("casadi")

from autoformalism.data import DatasetSplit, SplitName, Trajectory
from autoformalism.expressions import compile_candidate
from autoformalism.fitting.matching_probe import (
    bounded_latent_start,
    latent_start,
    matching_start,
)
from autoformalism.fitting.sensitivity_probe import SymbolicODE
from autoformalism.rebuttal import fitter_methods as campaign
from autoformalism.rebuttal.fitter_diagnostic import read_json, write_json
from scripts import run_fitter_methods as runner

CONFIG = Path(__file__).resolve().parents[1] / "configs/fitter_methods_v1.json"


def small_plan():
    payload = read_json(CONFIG)
    payload.update(
        noise_fractions=[0],
        fit_seconds=30,
        initializer_seconds=5,
        guard_seconds=100,
        replay_seconds=5,
        grace_seconds=5,
    )
    reference = payload["reference"]
    reference["cases"] = reference["cases"][:1]
    reference["maximum_nfev"] = 50
    reference["inputs"] = [
        {
            "name": "train_one",
            "split": "train",
            "times": [0, 0.8, 1.2, 2.8, 3.2, 4],
            "values": [0, 0, 1, 1, -0.5, -0.5],
        },
        {
            "name": "train_two",
            "split": "train",
            "times": [0, 0.8, 1.2, 2.8, 3.2, 4],
            "values": [0, 0, -1, -1, 1, 1],
        },
        {
            "name": "validation",
            "split": "validation",
            "times": [0, 1.2, 1.6, 4],
            "values": [0, 0, 0.8, 0.8],
        },
    ]
    return campaign.MethodsPlan.model_validate(payload)


def affine_training(plan):
    truth = {"a": 1.2, "b": 0.5, "c": 0.8, "d": -0.4}
    model = compile_candidate(campaign.affine_candidate(), campaign.CONTEXT)
    rows = []
    for forcing in plan.reference.inputs:
        if forcing.split != "train":
            continue
        time = np.arange(round(forcing.times[-1] / 0.2) + 1) * 0.2
        reference = campaign._affine_reference(
            forcing, truth, time, "Radau", monotonic() + 5
        )
        rows.append(
            Trajectory(
                forcing.name,
                time,
                {"v01": reference[:, 2]},
                {},
                {"u01": reference[:, 1]},
                {},
                {},
            )
        )
    return SymbolicODE(model), DatasetSplit(SplitName.TRAIN, tuple(rows), "test")


def test_default_matrix_and_freeze_fail_closed(tmp_path):
    plan = campaign.MethodsPlan.model_validate(read_json(CONFIG))
    frozen = campaign.prepare_methods(plan, tmp_path)
    assert len(frozen["tasks"]) == 35
    assert sum(t["kind"] == "fit" for t in frozen["tasks"]) == 32
    assert campaign.verify_methods(tmp_path) == frozen
    assert campaign.prepare_methods(plan, tmp_path) == frozen
    with pytest.raises(ValueError, match="unknown task"):
        campaign.execute_methods(tmp_path, -1)
    for task in frozen["tasks"]:
        if task["kind"] == "fit":
            assert (
                task["method"] not in campaign.MATCHING
                or task["case"] == "affine_observed"
            )
    path = tmp_path / "candidate_moderate.json"
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="candidate asset"):
        campaign.verify_methods(tmp_path)


@pytest.mark.parametrize("method", campaign.MATCHING)
def test_linear_matching_preserves_bounds_and_estimates_gain_and_offset(method):
    system, train = affine_training(small_plan())
    lower, upper = np.array([0, 0, 0, -np.inf]), np.full(4, np.inf)
    result = matching_start(system, train, lower, upper, method)
    assert result["success"] and result["design_rank"] == 4
    assert result["training_only"] and not result["hidden_labels_used"]
    theta = np.array([result["parameters"][n] for n in system.names])
    assert np.all(theta >= lower)
    # Coarse quadrature/smoothing is an initializer, not an exact truth estimator.
    assert abs(theta[2] - 0.8) < 0.25
    assert abs(theta[3] + 0.4) < 0.25


def test_matching_declines_hidden_model_and_too_short_data():
    system, train = affine_training(small_plan())
    hidden = SymbolicODE(
        compile_candidate(campaign.recovery_candidate(), campaign.CONTEXT)
    )
    with pytest.raises(ValueError, match="fully observed"):
        matching_start(hidden, train, np.zeros(7), np.ones(7), "integral_init")
    with pytest.raises(ValueError, match="sufficiently long"):
        matching_start(
            system, train, np.zeros(4), np.ones(4), "weak_init", window_intervals=100
        )


@pytest.mark.parametrize("method", campaign.LATENT)
def test_latent_initializer_solves_actual_nonlinear_program(tmp_path, method):
    system, train = affine_training(small_plan())
    result = latent_start(
        system,
        train,
        np.array([0, 0, 0, -np.inf]),
        np.full(4, np.inf),
        np.array([1, 0.4, 0.6, -0.2]),
        1.0,
        small_plan().settings(),
        method,
        15,
        tmp_path,
    )
    assert result["success"], result
    assert not result["hidden_labels_used"]
    assert not result["initial_conditions_optimized"]
    assert abs(result["parameters"]["c"] - 0.8) < 0.03
    assert result["initializer_objective"] < 1e-4


def test_native_initializer_has_a_hard_wall_limit(tmp_path):
    system, train = affine_training(small_plan())
    before = monotonic()
    result = bounded_latent_start(
        system,
        training=train,
        lower=np.array([0, 0, 0, -np.inf]),
        upper=np.full(4, np.inf),
        start=np.ones(4),
        scale=1,
        settings=small_plan().settings(),
        method="shooting_init",
        seconds=0.01,
        directory=tmp_path,
    )
    assert monotonic() - before < 5
    assert not result["success"]
    assert "wall-clock" in result["message"]


def test_guard_noisy_data_resume_and_forward_fit(tmp_path):
    plan = small_plan()
    frozen = campaign.prepare_methods(plan, tmp_path)
    guard = campaign.execute_methods(tmp_path, 0)
    assert guard["status"] == "complete", guard
    assert guard["coefficient_audit"]["rhs_parameter_affine_certified"]
    assert campaign.execute_methods(tmp_path, 0) == guard
    root = tmp_path / "results" / "guard_affine_observed" / "reference"
    first = campaign._data(
        tmp_path,
        "affine_observed",
        guard["records"],
        root,
        1,
        0.03,
        guard["baseline"]["training_scale"],
        plan.seed,
    )
    second = campaign._data(
        tmp_path,
        "affine_observed",
        guard["records"],
        root,
        1,
        0.03,
        guard["baseline"]["training_scale"],
        plan.seed,
    )
    for a, b in zip(first.train.trajectories, second.train.trajectories, strict=True):
        np.testing.assert_array_equal(a.targets["v01"], b.targets["v01"])
        assert not a.auxiliaries and set(a.targets) == {"v01"}
    index = next(
        i
        for i, t in enumerate(frozen["tasks"])
        if t.get("method") == "forward_sensitivity"
    )
    result = campaign.execute_methods(tmp_path, index)
    assert result["status"] == "complete", result
    assert result["clean_signal_nmse"]["train"] < 1e-9
    assert result["clean_signal_nmse"]["validation"] < 1e-9
    assert not result["initializer"]["success"]
    assert campaign.execute_methods(tmp_path, index) == result
    summary = campaign.summarize_methods(tmp_path)
    assert sum(r["status"] == "missing" for r in summary["rows"]) == 11
    # An unverified guard must block every dependent fit.
    guard["status"] = "guard_failed"
    write_json(root.parent / "result.json", guard)
    blocked = campaign.execute_methods(tmp_path, index + 1)
    assert blocked["status"] == "guard_failed"


def test_supervisor_hard_timeout_preserves_partial_checkpoint(tmp_path, monkeypatch):
    campaign.prepare_methods(small_plan(), tmp_path)

    class Stalled:
        pid = os.getpid()

        def wait(self, timeout=None):
            if timeout is not None:
                raise runner.subprocess.TimeoutExpired("worker", timeout)
            return -9

    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **kw: Stalled())
    killed = []
    monkeypatch.setattr(runner.os, "killpg", lambda pid, sig: killed.append(pid))
    result = runner.run_supervised(tmp_path, 0)
    assert result["status"] == "timeout" and killed == [os.getpid()]
    assert runner.run_supervised(tmp_path, 0) == result


@pytest.mark.parametrize("method", campaign.LATENT)
def test_bounded_initializer_accepts_only_observed_outputs_of_hidden_system(
    tmp_path, method
):
    plan = small_plan()
    truth = plan.reference.cases[0].truth.model_dump()
    model = compile_candidate(campaign.recovery_candidate(), campaign.CONTEXT)
    system = SymbolicODE(model)
    rows = []
    for forcing in plan.reference.inputs:
        if forcing.split != "train":
            continue
        time = np.arange(21) * 0.2
        reference = campaign.reference_rollout(
            forcing, truth, time, "Radau", monotonic() + 10
        )
        rows.append(
            Trajectory(
                forcing.name,
                time,
                {"v01": reference[:, 2]},
                {},
                {"u01": reference[:, 1]},
                {},
                {},
            )
        )
    train = DatasetSplit(SplitName.TRAIN, tuple(rows), "hidden-test")
    start = np.array([truth[n] * 1.02 for n in system.names])
    lower = np.array([-np.inf if n == "c" else 1e-12 for n in system.names])
    result = bounded_latent_start(
        system,
        training=train,
        lower=lower,
        upper=np.full(7, np.inf),
        start=start,
        scale=1.0,
        settings=plan.settings(),
        method=method,
        seconds=30,
        directory=tmp_path,
    )
    assert result["success"], result
    assert result["initializer_objective"] < 1e-4
    assert not result["hidden_labels_used"]


def test_launcher_accounts_for_matrix_and_rejects_partial_submission(tmp_path):
    import subprocess
    import sys

    root = CONFIG.parent.parent
    binaries = tmp_path / "bin"
    binaries.mkdir()
    git = binaries / "git"
    git.write_text('#!/bin/sh\nif [ "$1" = rev-parse ]; then echo abc123; fi\n')
    git.chmod(0o755)
    sbatch = binaries / "sbatch"
    sbatch.write_text(
        f"#!{sys.executable}\nimport os, sys\nfrom pathlib import Path\n"
        'p=Path(os.environ["SUBMISSION_LOG"])\n'
        'old=p.read_text() if p.exists() else ""\n'
        'p.write_text(old+" ".join(sys.argv[1:])+"\\n")\n'
        "print(100+len(old.splitlines()))\n"
    )
    sbatch.chmod(0o755)
    log = tmp_path / "submission.log"
    output = tmp_path / "output"
    env = {
        **os.environ,
        "AF_REPO_ROOT": str(root),
        "AF_PYTHON": sys.executable,
        "AF_OUTPUT_ROOT": str(output),
        "SUBMISSION_LOG": str(log),
        "PATH": str(binaries) + os.pathsep + os.environ["PATH"],
    }
    command = ["bash", str(root / "scripts/hpc/submit_fitter_methods_delta.sh")]
    one = subprocess.run(command, env=env, text=True, capture_output=True)
    assert one.returncode == 0, one.stderr
    jobs = read_json(output / "submission.json")
    assert jobs["guard_tasks"] == 3 and jobs["fit_tasks"] == 32
    assert jobs["submission_complete"]
    text = log.read_text()
    assert len(text.splitlines()) == 3 and "--array=0,1,2%2" in text
    assert "afterany:100" in text and "afterany:101" in text
    two = subprocess.run(command, env=env, text=True, capture_output=True)
    assert two.returncode == 0 and log.read_text() == text
    jobs["submission_complete"] = False
    write_json(output / "submission.json", jobs)
    three = subprocess.run(command, env=env, text=True, capture_output=True)
    assert three.returncode != 0 and "Partial submission" in three.stderr
    assert log.read_text() == text
