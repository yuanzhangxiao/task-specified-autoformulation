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


@pytest.mark.parametrize("version", [1, 2])
def test_launcher_accounts_for_matrix_and_rejects_partial_submission(tmp_path, version):
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
    launcher = (
        "submit_fitter_methods_delta.sh"
        if version == 1
        else "submit_fitter_methods_v2_delta.sh"
    )
    command = ["bash", str(root / "scripts/hpc" / launcher)]
    one = subprocess.run(command, env=env, text=True, capture_output=True)
    assert one.returncode == 0, one.stderr
    jobs = read_json(output / "submission.json")
    assert jobs["guard_tasks"] == (3 if version == 1 else 2)
    assert jobs["fit_tasks"] == (32 if version == 1 else 36)
    assert jobs["initializer_tasks"] == (0 if version == 1 else 12)
    assert jobs["submission_complete"]
    text = log.read_text()
    assert len(text.splitlines()) == (3 if version == 1 else 4)
    assert ("--array=0,1,2%2" if version == 1 else "--array=0,1%2") in text
    assert "afterany:100" in text and "afterany:101" in text
    if version == 2:
        assert "afterany:102" in text and jobs["initializer_job_id"] == "101"
    two = subprocess.run(command, env=env, text=True, capture_output=True)
    assert two.returncode == 0 and log.read_text() == text
    jobs["submission_complete"] = False
    write_json(output / "submission.json", jobs)
    three = subprocess.run(command, env=env, text=True, capture_output=True)
    assert three.returncode != 0 and "Partial submission" in three.stderr
    assert log.read_text() == text


@pytest.mark.parametrize("slope", [-3.0, 0.0, 3.0])
@pytest.mark.parametrize("intervals", [8, 10])
def test_weak_quadrature_preserves_constant_drift(slope, intervals):
    """Integration by parts must agree on a polynomial, including signed offsets."""
    from autoformalism.schemas import CandidateModel

    payload = campaign.affine_candidate().model_dump(mode="json")
    payload["state_equations"][0]["rhs"] = "d"
    payload["parameters"] = [p for p in payload["parameters"] if p["name"] == "d"]
    system = SymbolicODE(
        compile_candidate(CandidateModel.model_validate(payload), campaign.CONTEXT)
    )
    time = np.arange(31) * 0.2
    train = DatasetSplit(
        SplitName.TRAIN,
        (Trajectory("linear", time, {"v01": 2 + slope * time}, {}, {}, {}, {}),),
        "analytic",
    )
    result = matching_start(
        system,
        train,
        np.array([-np.inf]),
        np.array([np.inf]),
        "weak_init",
        window_intervals=intervals,
    )
    assert result["success"]
    assert result["parameters"]["d"] == pytest.approx(slope, abs=1e-12)
    assert result["weak_quadrature"] == "gauss5-linear-v2"
    if intervals == 8:
        old = matching_start(
            system,
            train,
            np.array([-np.inf]),
            np.array([np.inf]),
            "weak_init",
            weak_quadrature="trapezoid-v1",
        )
        assert old["parameters"]["d"] == pytest.approx(slope * 12 / 13, abs=1e-12)
    with pytest.raises(ValueError, match="quadrature policy"):
        matching_start(
            system,
            train,
            np.array([-np.inf]),
            np.array([np.inf]),
            "weak_init",
            weak_quadrature="unknown",
        )


def test_known_memory_alias_preserves_output_and_maps_states():
    from autoformalism.rebuttal.fitter_methods_report import parameter_equivalence

    plan = small_plan()
    truth = plan.reference.cases[0].truth.model_dump()
    case = {"observed_state": False, "truth": truth}
    report = parameter_equivalence(case, truth)
    alias = report["known_admissible_aliases"]["memory_pole_swap"]
    assert (alias["tau"], alias["tau_p"], alias["k_p"]) == pytest.approx((4, 2, 0.15))
    assert parameter_equivalence(case, alias)["scaled_linf_distance"] == 0
    for forcing in plan.reference.inputs:
        time = np.arange(21) * 0.2
        a = campaign.reference_rollout(forcing, truth, time, "Radau", monotonic() + 10)
        b = campaign.reference_rollout(forcing, alias, time, "Radau", monotonic() + 10)
        np.testing.assert_allclose(a[:, 2], b[:, 2], atol=1e-9, rtol=0)
        np.testing.assert_allclose(
            b[:, 3],
            a[:, 3] + (1 / truth["tau"] - 1 / truth["tau_p"]) * a[:, 4],
            atol=1e-9,
        )
        np.testing.assert_allclose(a[:, 4:], b[:, 4:], atol=1e-9, rtol=0)
    separated = campaign.MethodsPlan.model_validate(read_json(CONFIG)).reference.cases[
        1
    ]
    truth = separated.truth.model_dump()
    assert (
        len(
            parameter_equivalence({"observed_state": False, "truth": truth}, truth)[
                "known_admissible_aliases"
            ]
        )
        == 1
    )


def test_v2_freezes_paired_starts_and_noise_without_truth_dependence(tmp_path):
    payload = read_json(CONFIG.with_name("fitter_methods_v2.json"))
    plan = campaign.MethodsPlan.model_validate(payload)
    frozen = campaign.prepare_methods(plan, tmp_path / "one")
    assert len(frozen["tasks"]) == 50
    assert len({t["pair"] for t in frozen["tasks"] if t["kind"] == "fit"}) == 12
    for pair in {t["pair"] for t in frozen["tasks"] if t["kind"] == "fit"}:
        tasks = [t for t in frozen["tasks"] if t.get("pair") == pair]
        assert len(tasks) == 4
        assert len({t["noise_seed"] for t in tasks}) == 1
        assert len({t["start_seed"] for t in tasks}) == 1
        assert len({t["initializer_task"] for t in tasks}) == 1
        assert {t["method"] for t in tasks if t["kind"] == "fit"} == set(
            campaign.PAIRED
        )
    payload["reference"]["cases"][0]["truth"]["k"] = 99
    changed = campaign.prepare_methods(
        campaign.MethodsPlan.model_validate(payload), tmp_path / "two"
    )
    assert frozen["paired_starts"] == changed["paired_starts"]
    summary = campaign.summarize_methods(tmp_path / "one")
    assert all(r["status"] == "missing" for r in summary["rows"])
    assert all(
        a["planned"] == 3 and a["verified"] == 0
        for a in summary["paired_summary"]["aggregates"]
    )
    assert (tmp_path / "one/details.md").exists()
    assert (
        "Both latent cases expose v01 only" in (tmp_path / "one/details.md").read_text()
    )
    assert len((tmp_path / "one/summary.md").read_text().splitlines()) < 80


def test_paired_collocation_refines_same_checkpoint_and_resumes(tmp_path):
    payload = small_plan().model_dump(mode="json")
    payload.update(protocol="fitter-methods-2", initializer_seconds=20, fit_seconds=60)
    plan = campaign.MethodsPlan.model_validate(payload)
    frozen = campaign.prepare_methods(plan, tmp_path)
    guard = campaign.execute_methods(tmp_path, 0)
    assert guard["status"] == "complete", guard
    init = campaign.execute_methods(tmp_path, 1)
    assert init["status"] == "complete", init
    reports = {}
    for index, task in enumerate(frozen["tasks"]):
        if task["kind"] == "fit" and task["method"] != "forward_sensitivity":
            result = campaign.execute_methods(tmp_path, index)
            assert result["status"] == "complete", result
            assert not result["initializer_fallback"]
            assert result["refinement_start"] == init["initializer"]["parameters"]
            assert result["total_fit_seconds"] == pytest.approx(
                init["initializer"]["seconds"] + result["fit"]["fit_seconds"]
            )
            assert campaign.execute_methods(tmp_path, index) == result
            reports[task["method"]] = result
    assert reports["collocation_sensitivity"]["symbolic_solver_counts"]
    assert reports["collocation_init"]["symbolic_solver_counts"] is None
    summary = campaign.summarize_methods(tmp_path)
    assert summary["paired_summary"]["pairing_checks"][0]["same_data_and_initializer"]
    assert reports["collocation_sensitivity"]["clean_signal_nmse"]["train"] < 1e-6
    init["initializer"]["seconds"] += 1
    write_json(tmp_path / "results" / frozen["tasks"][1]["name"] / "result.json", init)
    with pytest.raises(ValueError, match="initializer changed"):
        campaign.summarize_methods(tmp_path)


def test_shared_initializer_missing_and_timeout_are_distinct(tmp_path):
    payload = small_plan().model_dump(mode="json")
    payload.update(protocol="fitter-methods-2")
    plan = campaign.MethodsPlan.model_validate(payload)
    frozen = campaign.prepare_methods(plan, tmp_path)
    task = next(
        t for t in frozen["tasks"] if t.get("method") == "collocation_sensitivity"
    )
    with pytest.raises(ValueError, match="missing or blocked"):
        campaign._paired_initializer(tmp_path, frozen, task, plan)
    init = frozen["tasks"][1]
    key = campaign.content_hash([frozen["freeze_sha256"], init])
    write_json(
        tmp_path / "results" / init["name"] / "result.json",
        {"identity": key, "task": init, "status": "timeout", "error": "wall limit"},
    )
    result = campaign._paired_initializer(tmp_path, frozen, task, plan)
    assert not result["success"] and result["parameters"] is None
    assert result["seconds"] == plan.initializer_seconds
