"""M43 fixed controls, causal boundaries, durable resume and CPU-only launchers."""

import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from autoformalism.benchmarks.audited_release import seal
from autoformalism.fitting import initialization_qualification as q
from scripts import submit_phase_c_initialization_qualification as submitter


@pytest.fixture(autouse=True)
def stable_source(monkeypatch):
    monkeypatch.setattr(q.public, "_source_identity", lambda: "a" * 64)


@pytest.fixture
def prepared(tmp_path):
    root = tmp_path / "campaign"
    assert q.prepare(root, None, q.Policy())["tasks"] == 4
    return root


def test_roster_same_y0_different_roots_and_dependent_coordinates():
    first = q.make_inputs()
    assert first == q.make_inputs()
    assert len(q._validate_inputs(first)) == 4
    for case in q.CASES:
        a, b = (first["records"][f"{case}_s{seed}"] for seed in q.SEEDS)
        assert a["training"] == b["training"] and a["validation"] == b["validation"]
        problem = q.kernel.PrefixFitProblem.model_validate(a["problem"])
        training = q.PublicSplit.model_validate(a["training"])
        oracle = q.kernel.TrainingOracle(problem, training, q.Policy().numerical)
        assert "init_z_phi" in oracle.shared
        assert "init_z_value" not in oracle.names
        if case == "trajectory_root":
            assert len(oracle.names) == 5 and len(oracle.roots) == 1
            assert all(row.targets["y"][0] == 0.4 for row in training.rows)
            assert (
                training.rows[0].targets["y"][10] != training.rows[1].targets["y"][10]
            )
        else:
            assert len(oracle.names) == 3 and not oracle.roots
        assert "evaluation" not in problem.model_dump()


@pytest.mark.parametrize("theta", [0.35, 1.0, 2.0, 1.0 + 1e-12])
def test_analytic_replay_handles_equal_poles(theta):
    matrix = np.array([[-theta, 1, 1], [0, -1, 0], [0, 0, -2]])
    time = np.linspace(0, 4, 41)
    expected = solve_ivp(
        lambda _, values: matrix @ values,
        (0, 4),
        [0.4, 0.7 * 0.8, 0.8],
        t_eval=time,
        method="DOP853",
        rtol=1e-12,
        atol=1e-14,
    ).y[0]
    np.testing.assert_allclose(
        q.analytic(time, 0.4, theta, 0.7, 0.8), expected, atol=1e-12
    )


def test_prepare_resume_identity_and_missing_denominators(prepared, monkeypatch):
    monkeypatch.setattr(q, "make_inputs", lambda: pytest.fail("regenerated"))
    assert q.prepare(prepared, None, q.Policy())["tasks"] == 4
    summary = q.report(prepared)
    assert summary["status"] == "incomplete" and summary["expected"] == 4
    assert summary["recorded"] == 0 and summary["status_counts"] == {"missing": 4}
    assert summary["cost"]["wall_seconds_total"] is None
    assert not summary["cost"]["accounting_complete"]
    with pytest.raises(ValueError, match="sealed artifact differs"):
        q.prepare(prepared, None, q.Policy(prefix=q.kernel.PrefixPolicy(duration=0.5)))
    monkeypatch.setattr(q.public, "_source_identity", lambda: "b" * 64)
    with pytest.raises(ValueError, match="source/runtime"):
        q.verify(prepared)
    assert q.report(prepared) == summary


@pytest.mark.parametrize("change", ["fingerprint", "model", "overlapping_ids"])
def test_imported_inputs_reject_changed_design_and_split_identity(change):
    data = q.make_inputs()
    record = data["records"]["shared_root_s0"]
    if change == "fingerprint":
        record["training"]["rows"][0]["targets"]["y"][0] += 1
    elif change == "model":
        record["problem"]["base_candidate"]["state_equations"][0]["rhs"] = "-theta*x"
    else:
        record["validation"]["rows"][0]["trajectory_id"] = "train_0"
        record["validation"]["fingerprint"] = q.public.content_sha256(
            {"name": "val", "rows": record["validation"]["rows"]}
        )
    with pytest.raises(ValueError, match="qualification"):
        q._validate_inputs(data)


@pytest.mark.parametrize("index", [0, 2])
def test_real_known_equation_fit_prefix_future_and_terminal_resume(
    prepared, index, monkeypatch
):
    result = q.run_task(prepared, index)
    assert result["qualification_passed"] and result["status"] == "complete"
    assert result["training_fit"]["normalized_mse"] < 1e-12
    assert result["training_fit"]["rank_diagnostic"]["full_column_rank"]
    assert all(
        abs(error) < 1e-6
        for error in result["evaluation"]["prefix_root_errors_evaluator_only"].values()
    )
    assert all(item["score"]["sample_count"] == 30 for item in result["forecasts"])
    assert all(item["forecast"]["forecast_success"] for item in result["forecasts"])
    assert all(
        item["inference"]["shared_parameters"]
        == result["training_fit"]["shared_parameters"]
        for item in result["forecasts"]
    )
    if index == 0:
        assert all(
            item["inference"]["search"]["calls"] == 0 for item in result["forecasts"]
        )
    else:
        values = [
            item["inference"]["roots"]["init_r_value"]
            for item in result["forecasts"]
            if item["split"] == "validation"
        ]
        assert values == pytest.approx([0.65, 1.45], abs=1e-6)
    assert result["cost"]["verification_calls_started_total"] == 22
    assert result["cost"]["verification_calls_completed_total"] == 22
    assert (
        result["cost"]["search_calls_started_total"]
        == result["cost"]["search_calls_completed_total"]
    )
    paths = {
        str(path.relative_to(prepared)): path.read_bytes()
        for path in (prepared / "results").rglob("*.json")
    }
    for name in ("fit_training", "infer_prefix", "forecast", "score_forecast"):
        monkeypatch.setattr(
            q.kernel,
            name,
            lambda *a, **kw: pytest.fail("terminal computation repeated"),
        )
    assert q.run_task(prepared, index) == result
    assert paths == {
        str(path.relative_to(prepared)): path.read_bytes()
        for path in (prepared / "results").rglob("*.json")
    }
    summary = q.report(prepared)
    assert summary["recorded"] == 1 and not summary["qualification_passed"]
    assert len(summary["prefix_forecasts"]["training"]) == 3
    assert len(summary["prefix_forecasts"]["validation"]) == 2
    folder = prepared / "results" / result["task_id"]
    changed = deepcopy(result)
    changed["training_fit"] = deepcopy(changed["training_fit"])
    changed["training_fit"]["shared_parameters"]["theta"] += 0.1
    (folder / "result.json").unlink()
    seal(folder / "result.json", changed)
    with pytest.raises(ValueError, match="fitted endpoint"):
        q.run_task(prepared, index)


def test_interrupted_fit_keeps_observed_work_and_never_restarts_optimizer(
    prepared, monkeypatch
):
    def interrupted(problem, training, policy, checkpoint):
        checkpoint(
            {"calls": 2, "completed_calls": 1, "best": {"parameters": {"theta": 0.2}}}
        )
        raise KeyboardInterrupt

    monkeypatch.setattr(q.kernel, "fit_training", interrupted)
    with pytest.raises(KeyboardInterrupt):
        q.run_task(prepared, 0)
    summary = q.report(prepared)
    assert summary["cost"]["budget_charge_seconds"] == 40
    assert summary["cost"]["search_calls_started_observed"] == 2
    assert summary["cost"]["search_calls_completed_observed"] == 1
    assert summary["cost"]["search_calls_started_total"] is None
    assert summary["cost"]["wall_seconds_total"] is None
    monkeypatch.setattr(
        q.kernel, "fit_training", lambda *a, **kw: pytest.fail("restarted")
    )
    result = q.run_task(prepared, 0)
    assert result["status"] == "incomplete" and not result["qualification_passed"]
    assert result["operations"]["training"]["status"] == "interrupted"
    assert result["cost"]["budget_charge_seconds"] == 40
    assert result["cost"]["calls_total"] is None
    assert q.run_task(prepared, 0) == result


def test_failed_forecast_is_unavailable_and_never_analytically_replayed(
    prepared, monkeypatch
):
    def unavailable(problem, fit, inference, schedule, policy, checkpoint):
        checkpoint({"verification_calls_started": 1, "verification_calls_completed": 0})
        return {
            "forecast_success": False,
            "predictions": {},
            "states": {},
            "verification_calls_started": 1,
            "verification_calls_completed": 0,
        }

    monkeypatch.setattr(q.kernel, "forecast", unavailable)
    result = q.run_task(prepared, 0)
    assert result["status"] == "incomplete"
    assert all(item["score"] is None for item in result["forecasts"])
    assert result["evaluation"]["forecast_analytic_checks"] == []
    assert result["cost"]["verification_calls_started_total"] == 17
    assert result["cost"]["verification_calls_completed_total"] == 12


def test_fitter_receives_only_training_and_no_evaluator_payload(prepared, monkeypatch):
    def failed(problem, training, policy, checkpoint):
        assert isinstance(problem, q.kernel.PrefixFitProblem)
        assert training.name == "train"
        assert not {"evaluation", "validation", "truth"} & problem.model_dump().keys()
        checkpoint({"calls": 0, "completed_calls": 0})
        return {"fit_success": False, "search": {"calls": 0, "completed_calls": 0}}

    monkeypatch.setattr(q.kernel, "fit_training", failed)
    monkeypatch.setattr(q, "_evaluate", lambda *a: pytest.fail("evaluated failed fit"))
    assert q.run_task(prepared, 1)["status"] == "incomplete"
    with pytest.raises(ValueError, match="unknown"):
        q.run_task(prepared, 4)


def test_scheduler_uses_four_cpu_tasks_and_idempotent_receipts(tmp_path, monkeypatch):
    root, config = tmp_path / "scheduled", tmp_path / "policy.json"
    config.write_text(q.Policy().model_dump_json())
    monkeypatch.setenv("AF_PYTHON", sys.executable)
    calls = []

    def submit(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=0, stdout=str(100 + len(calls)), stderr="")

    monkeypatch.setattr(submitter.scheduler.subprocess, "run", submit)
    monkeypatch.setattr(
        submitter.scheduler.subprocess, "check_output", lambda *a, **kw: "commit"
    )
    first = submitter.submit(root, config)
    assert submitter.submit(root, config) == first
    assert first["array_tasks"] == 4 and first["gpus"] == 0
    assert len(calls) == 3
    assert "--array=0-3%2" in calls[1]
    assert "--cpus-per-task=1" in calls[1] and "--time=00:10:00" in calls[1]
    (root / "submission/run.id").unlink()
    with pytest.raises(ValueError, match="Unconfirmed run"):
        submitter.submit(root, config)
    assert len(calls) == 3


def test_launcher_shell_syntax_defaults_and_cli_help():
    repo = Path(__file__).resolve().parents[1]
    scripts = [
        repo / "scripts/hpc" / f"{verb}_phase_c_initialization_qualification_delta.sh"
        for verb in ("submit", "run")
    ]
    subprocess.run(["bash", "-n", *map(str, scripts)], check=True)
    text = scripts[0].read_text()
    assert "/projects/bibo/yxiao2/venvs/autoformalism-v21/bin/python" in text
    assert "/projects/bibo/yxiao2/venvs/fitter-methods-v1-deps" in text
    assert "AF_CONCURRENCY:-2" in text and "OPENBLAS_NUM_THREADS=1" in text
    result = subprocess.run(
        [
            sys.executable,
            str(repo / "scripts/phase_c_initialization_qualification.py"),
            "--help",
        ],
        capture_output=True,
        text=True,
        check=True,
        env=os.environ | {"PYTHONPATH": str(repo / "src")},
    )
    assert "prepare,verify,run,report" in result.stdout
    payload = json.loads(
        (repo / "configs/phase_c_initialization_qualification_v1.json").read_text()
    )
    assert q.Policy.model_validate(payload).numerical.maximum_calls == 100
