"""Fixed-function certification, paired starts, isolation and bounded resume."""

import json
import os
import subprocess
import sys
from pathlib import Path
from time import monotonic

import numpy as np
import pytest

pytest.importorskip("casadi")

from autoformalism.expressions.diagnostics import ModelValidationError
from autoformalism.fitting.collocation_sensitivity import (
    CollocationSensitivityConfig,
    fit_collocation_forward_sensitivity,
)
from autoformalism.rebuttal import parameter_freedom as freedom
from autoformalism.rebuttal.attainability_campaign import AttainabilityPlan
from autoformalism.rebuttal.attainability_controls import system_for
from autoformalism.rebuttal.fitter_diagnostic import read_json, write_json
from autoformalism.rebuttal.resolution_campaign import prepare_resolution, profile_point
from autoformalism.staged_topology import content_hash
from tests.test_attainability_campaign import resolution_source


def source_at(tmp_path):
    source = resolution_source(tmp_path / "v2")
    v3 = tmp_path / "v3"
    frozen = prepare_resolution(
        source,
        v3,
        AttainabilityPlan.model_validate_json(
            Path("configs/fitter_resolution_v3.json").read_text()
        ),
    )
    write_json(v3 / "preflight.json", {"identity": frozen["identity"], "pass": True})
    return v3


def plan_for(source):
    problem = read_json(source / "generated/000/synthetic.json")
    truth = read_json(source / "generated/000/truth.json")
    _, shapes = freedom.fixed_basis(problem, truth)
    raw = json.loads(Path("configs/fitter_parameter_freedom_v1.json").read_text())
    raw.update(
        expected_parameters=len(truth), expected_weights=len(truth) - len(shapes)
    )
    return freedom.FreedomPlan.model_validate(raw)


def prepared(tmp_path):
    source = source_at(tmp_path)
    output = tmp_path / "out"
    frozen = freedom.prepare(source, output, plan_for(source))
    return source, output, frozen


def test_freezing_preserves_equations_and_shared_weight_starts(tmp_path):
    source = source_at(tmp_path)
    problem = read_json(source / "generated/000/synthetic.json")
    truth = read_json(source / "generated/000/truth.json")
    tasks, problems, fixed = freedom.task_matrix(problem, truth)
    assert len(tasks) == 6
    assert all(t["oracle_initials"] for t in tasks)
    for left, right in ((0, 1), (3, 4)):
        a, b = problems[left], problems[right]
        sa, sb = system_for(a)[0], system_for(b)[0]
        assert sa.affine and not sb.affine
        assert {**a["start"], **fixed} == b["start"]
        for state in (np.zeros(6), np.arange(6) * 0.25, -np.arange(6) * 0.5):
            va = sa.rhs(
                0, state, [a["start"][n] for n in sa.names], np.zeros(len(sa.inputs))
            )
            vb = sb.rhs(
                0, state, [b["start"][n] for n in sb.names], np.zeros(len(sb.inputs))
            )
            np.testing.assert_allclose(va, vb, atol=1e-13)
    assert problems[2]["start"] == profile_point(problem, truth, "ordinary")
    assert problems[5]["start"] == profile_point(problem, truth, "near")
    assert not tasks[2]["oracle_shapes"]
    assert tasks[5]["oracle_shapes"] and not tasks[5]["exact_shape_start"]
    for p in problems:
        assert p["splits"] == problem["splits"]
        assert p["initialization_plan"] == problem["initialization_plan"]


def test_unknown_parameters_and_nonlinear_remaining_weights_fail(tmp_path):
    source = source_at(tmp_path)
    problem = read_json(source / "generated/000/synthetic.json")
    truth = read_json(source / "generated/000/truth.json")
    with pytest.raises(ValueError, match="parameter values"):
        freedom.fixed_basis(problem, {**truth, "unused": 1.0})
    problem["candidate"]["state_equations"][0]["rhs"] += "+decay0**2"
    with pytest.raises(ValueError, match="not affine"):
        freedom.fixed_basis(problem, truth)
    problem["candidate"]["state_equations"][0]["rhs"] = "__import__('os')"
    with pytest.raises(ModelValidationError, match="UNSUPPORTED_FUNCTION"):
        freedom.fixed_basis(problem, truth)


def test_paired_rollout_and_derivative_gate(tmp_path):
    source = source_at(tmp_path)
    problem = read_json(source / "generated/000/synthetic.json")
    truth = read_json(source / "generated/000/truth.json")
    _, problems, _ = freedom.task_matrix(problem, truth)
    for p in problems[:2]:
        row = p["splits"]["train"]["rows"][0]
        row.update(
            time=[0, 0.1, 0.2, 0.3],
            targets={"v01": [0, 0.1, 0.2, 0.3]},
            external_inputs={"u01": [0, 0.1, 0, 0]},
        )
        p["splits"]["train"]["rows"] = [row]
    result = freedom.paired_check(
        problems[0], problems[1], plan_for(source).fit, monotonic() + 30
    )
    assert result["pass"] and result["same_mesh"]
    problems[1]["start"]["input0"] *= 2
    result = freedom.paired_check(
        problems[0], problems[1], plan_for(source).fit, monotonic() + 30
    )
    assert not result["pass"]


def test_prepare_is_idempotent_and_checks_source_and_output(tmp_path):
    source, output, frozen = prepared(tmp_path)
    assert freedom.prepare(source, output, plan_for(source)) == frozen
    assert (output / "inputs/synthetic.json").read_bytes() == (
        source / "generated/000/synthetic.json"
    ).read_bytes()
    assert frozen["test_data_opened"] is False and frozen["proposer_access"] is False
    write_json(output / "problems/000.json", {})
    with pytest.raises(ValueError, match="frozen asset"):
        freedom.verify(output)
    write_json(
        source / "preflight.json",
        {"identity": read_json(source / "freeze.json")["identity"], "pass": False},
    )
    with pytest.raises(ValueError, match="source preflight"):
        freedom.prepare(source, tmp_path / "blocked", plan_for(source))


def test_interrupted_gate_and_fit_do_not_restart(tmp_path, monkeypatch):
    _, output, frozen = prepared(tmp_path)
    for index in (-1, 6):
        with pytest.raises(ValueError, match="task index"):
            freedom.execute(output, index)
    write_json(output / "gate/started.json", {"identity": frozen["identity"]})
    monkeypatch.setattr(
        freedom, "paired_check", lambda *a: pytest.fail("restarted gate")
    )
    assert freedom.gate(output)["status"] == "interrupted"
    assert freedom.gate(output)["status"] == "interrupted"
    with pytest.raises(ValueError, match="preflight"):
        freedom.execute(output, 0)
    write_json(
        output / "gate/result.json", {"identity": frozen["identity"], "pass": True}
    )
    identity = content_hash([frozen["identity"], frozen["tasks"][0]])
    write_json(output / "results/task_000/fit_started.json", {"identity": identity})
    monkeypatch.setattr(
        freedom,
        "fit_collocation_forward_sensitivity",
        lambda *a, **k: pytest.fail("restarted fit"),
    )
    assert freedom.execute(output, 0)["status"] == "interrupted"
    assert freedom.execute(output, 0)["status"] == "interrupted"


def test_completed_fit_is_retained_and_interrupted_replay_not_repeated(
    tmp_path, monkeypatch
):
    _, output, frozen = prepared(tmp_path)
    write_json(
        output / "gate/result.json", {"identity": frozen["identity"], "pass": True}
    )
    identity = content_hash([frozen["identity"], frozen["tasks"][0]])
    p = read_json(output / "problems/000.json")
    root = output / "results/task_000"
    write_json(
        root / "fit.json",
        {
            "identity": identity,
            "fit": {"parameters": p["start"]},
            "estimation_seconds": 12,
        },
    )
    write_json(root / "replay_train_started.json", {"identity": identity})
    write_json(
        root / "replay_val.json", {"identity": identity, "check": {"pass": True}}
    )
    monkeypatch.setattr(
        freedom,
        "fit_collocation_forward_sensitivity",
        lambda *a, **k: pytest.fail("refitted"),
    )
    monkeypatch.setattr(
        freedom, "checked_replay", lambda *a, **k: pytest.fail("replayed")
    )
    row = freedom.execute(output, 0)
    assert row["status"] == "replay_unverified"
    assert row["replays"]["train"]["status"] == "interrupted"
    assert freedom.execute(output, 0) == row


def test_extended_budgets_require_explicit_profile():
    with pytest.raises(ValueError, match="extended_diagnostic"):
        CollocationSensitivityConfig(refinement_seconds=3600)
    raw = json.loads(Path("configs/fitter_parameter_freedom_v1.json").read_text())
    plan = freedom.FreedomPlan.model_validate(raw)
    assert (
        plan.fit.refinement_seconds == 3600 and plan.fit.recovery_screen_seconds == 240
    )
    raw["supervisor_seconds"] = 1200
    with pytest.raises(ValueError, match="supervisor"):
        freedom.FreedomPlan.model_validate(raw)


def test_deferred_fit_never_scores_validation(tmp_path, monkeypatch):
    from autoformalism.fitting import collocation_sensitivity as adapter
    from autoformalism.rebuttal.initialization_campaign import synthetic_problem
    from autoformalism.rebuttal.piecewise_campaign import unpack_split

    p, _ = synthetic_problem("shared", 0.0, 0)
    system, _ = system_for(p)
    monkeypatch.setattr(
        adapter,
        "bounded_latent_start",
        lambda *a, **k: {"success": False, "seconds": 0},
    )
    monkeypatch.setattr(
        adapter,
        "recover_refinement",
        lambda *a, **k: {"parameters": dict.fromkeys(system.names, 0.1)},
    )
    monkeypatch.setattr(
        adapter,
        "evaluate_fitted_candidate",
        lambda *a, **k: pytest.fail("unbounded replay"),
    )
    fit = fit_collocation_forward_sensitivity(
        system.model,
        unpack_split(p["splits"]["train"]),
        unpack_split(p["splits"]["val"]),
        CollocationSensitivityConfig(
            recovery_policy="feasible", defer_production_replay=True
        ),
        tmp_path,
    )
    assert fit["status"] == "verification_pending" and fit["validation"] is None
    assert fit["validation_used_for_fitting"] is False


def test_standard_library_report_preserves_missing_and_blocked(tmp_path):
    _, output, frozen = prepared(tmp_path)
    write_json(
        output / "gate/result.json", {"identity": frozen["identity"], "pass": False}
    )
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "scripts/run_parameter_freedom.py",
            "summarize",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert read_json(output / "summary.json")["counts"] == {"blocked_by_gate": 6}


def test_launcher_dependencies_and_repeat_submission(tmp_path):
    source = source_at(tmp_path)
    plan = plan_for(source)
    config = tmp_path / "config.json"
    write_json(config, plan.model_dump(mode="json"))
    binpath = tmp_path / "bin"
    binpath.mkdir()
    (binpath / "git").write_text(
        '#!/bin/sh\n[ "$1" = rev-parse ] && echo fake-commit\nexit 0\n'
    )
    (binpath / "git").chmod(0o755)
    calls = tmp_path / "calls.json"
    sbatch = binpath / "sbatch"
    sbatch.write_text(
        f"#!{sys.executable}\nimport json,sys\nfrom pathlib import Path\n"
        f"p=Path({str(calls)!r})\n"
        "r=json.loads(p.read_text()) if p.exists() else []\n"
        "r.append(sys.argv[1:]);p.write_text(json.dumps(r));print(40000+len(r))\n"
    )
    sbatch.chmod(0o755)
    env = {
        **os.environ,
        "PATH": str(binpath) + os.pathsep + os.environ["PATH"],
        "AF_REPO_ROOT": str(Path.cwd()),
        "AF_PYTHON": sys.executable,
        "AF_CASADI_ROOT": "",
        "AF_SOURCE_ROOT": str(source),
        "AF_OUTPUT_ROOT": str(tmp_path / "jobs"),
        "AF_CONFIG": str(config),
    }
    for _ in range(2):
        result = subprocess.run(
            ["bash", "scripts/hpc/submit_parameter_freedom_delta.sh"],
            env=env,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
    rows = json.loads(calls.read_text())
    assert len(rows) == 4
    assert "--dependency=afterok:40002" in rows[2] and "--array=0-5%2" in rows[2]
    assert "--dependency=afterany:40003:40002" in rows[3]
