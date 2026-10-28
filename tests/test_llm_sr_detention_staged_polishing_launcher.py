"""Two staged tasks, sealed CPU ceilings, safe submission and compact inspection."""

import json
import os
import subprocess
import sys
import tarfile
from types import SimpleNamespace

import pytest

from autoformalism.benchmarks.audited_release import seal
from autoformalism.fitting import public_fitting as public
from scripts import run_llm_sr_detention_staged_polishing as launcher


def ready(root, monkeypatch):
    """Separate scheduler contract tests from numerical campaign execution."""
    plan = {
        "protocol": "llm-sr-detention-staged-polishing-1",
        "source_sha256": "a" * 64,
        "tasks": [{"index": index, "kind": "polishing"} for index in range(2)],
    }
    context = SimpleNamespace(plan=plan)
    monkeypatch.setattr(launcher.campaign, "load_run", lambda root: context)
    monkeypatch.setattr(launcher.campaign, "verify", lambda root: (plan, None, None))
    monkeypatch.setattr(launcher.public, "_runtime", lambda: {"python": "test"})
    monkeypatch.setenv("AF_PYTHON", sys.executable)
    monkeypatch.setenv("AF_OUTPUT_ROOT", "/unrelated/campaign")
    seal(root / "plan.json", plan)
    seal(root / "inputs.json", {"protocol": "portable-training-endpoints"})
    return context


def record_submission(directory, key, options, worker, stage, index):
    """Write the same durable evidence as a successful mocked sbatch reply."""
    job = "101" if stage == "run" else "102"
    argv = ["sbatch", "--parsable", *options, str(worker), stage, str(index)]
    public._write(directory / f"{key}.intent.json", {"argv": argv})
    public._write(
        directory / f"{key}.reply.json",
        {"returncode": 0, "stdout": job + "\n", "stderr": ""},
    )
    (directory / f"{key}.id").write_text(job + "\n")
    return job


def test_submission_limits_deduplication_and_immediate_progress(
    tmp_path, monkeypatch, capsys
):
    context = ready(tmp_path, monkeypatch)
    calls = []
    loads = []

    def load(root):
        loads.append(root)
        return context

    def submit(*args):
        calls.append(args[1])
        return record_submission(*args)

    def forbidden_verify(*args, **kwargs):
        raise AssertionError("already authenticated campaign context")

    monkeypatch.setattr(launcher.campaign, "load_run", load)
    monkeypatch.setattr(launcher.campaign, "verify", forbidden_verify)
    monkeypatch.setattr(launcher, "submit_job", submit)
    result = launcher.submit(tmp_path)
    messages = capsys.readouterr().err
    assert messages.index("Input authentication complete") < messages.index(
        "Submitting run"
    )
    assert messages.index("Slurm accepted run: job 101") < messages.index(
        "Submitting report"
    )
    assert launcher.submit(tmp_path) == result
    assert calls == ["run", "report"]
    assert loads == [tmp_path, tmp_path]
    assert result["tasks"] == 2 and result["maximum_parallel_tasks"] == 2
    assert result["gpus"] == result["llm_calls"] == 0
    assert result["validation_or_test_access"] is False
    identity = launcher.execution_identity(tmp_path, verified_plan=context.plan)
    assert identity["numerical_reservation_ceiling_seconds"] == 8400
    allowances = identity["maximum_numerical_allowances"]
    assert allowances == {
        "search_segments": 1,
        "search_seconds": 1800,
        "independent_verification_seconds": 120,
        "search_audit_coordinates": 9,
        "search_audit_seconds_per_coordinate": 360,
        "strict_audit_coordinates_if_required": 9,
        "strict_audit_seconds_per_coordinate": 360,
    }
    assert 1800 + 120 + 2 * 9 * 360 == 8400 < 4 * 3600
    directory = tmp_path / launcher.SUBMISSIONS
    run = json.loads((directory / "run.intent.json").read_text())["argv"]
    report = json.loads((directory / "report.intent.json").read_text())["argv"]
    assert "--array=0-1%2" in run and "--time=04:00:00" in run
    assert "--cpus-per-task=1" in run and "--mem=8G" in run
    assert "--time=00:30:00" in report and "--dependency=afterany:101" in report
    assert os.environ["AF_LLMSR_DETENTION_STAGED_POLISHING_ROOT"] == str(tmp_path)
    assert os.environ["AF_OUTPUT_ROOT"] == "/unrelated/campaign"


@pytest.mark.parametrize("interrupted_stage", ["run", "report"])
def test_accepted_job_recovery_does_not_submit_twice(
    tmp_path, monkeypatch, interrupted_stage
):
    """A crash after durable acceptance reuses that ID before completing submit."""
    ready(tmp_path, monkeypatch)
    calls = []

    def submit(*args):
        stage = args[4]
        calls.append(stage)
        job = record_submission(*args)
        if stage == interrupted_stage:
            raise KeyboardInterrupt()
        return job

    monkeypatch.setattr(launcher, "submit_job", submit)
    with pytest.raises(KeyboardInterrupt):
        launcher.submit(tmp_path)
    result = launcher.submit(tmp_path)
    assert result["jobs"] == {"run": "101", "report": "102"}
    assert calls == ["run", "report"]


@pytest.mark.parametrize("uncertain_stage", ["run", "report"])
def test_uncertain_intent_never_resubmits_array_or_report(
    tmp_path, monkeypatch, uncertain_stage
):
    ready(tmp_path, monkeypatch)
    calls = []

    def submit(directory, key, options, worker, stage, index):
        calls.append(stage)
        if stage == uncertain_stage:
            public._write(directory / f"{key}.intent.json", {"argv": []})
            raise ValueError("uncertain scheduler reply")
        return record_submission(directory, key, options, worker, stage, index)

    monkeypatch.setattr(launcher, "submit_job", submit)
    with pytest.raises(ValueError, match="uncertain"):
        launcher.submit(tmp_path)
    with pytest.raises(ValueError, match="uncertain"):
        launcher.submit(tmp_path)
    assert calls == (["run"] if uncertain_stage == "run" else ["run", "report"])


def test_execution_identity_rejects_changed_dependencies(tmp_path, monkeypatch):
    ready(tmp_path, monkeypatch)
    seal(tmp_path / "launcher_execution.json", launcher.execution_identity(tmp_path))
    monkeypatch.setenv("AF_CASADI_ROOT", "/different/dependencies")
    with pytest.raises(ValueError, match="launcher identity differs"):
        launcher.verify_execution(tmp_path)


def test_execution_identity_rejects_changed_source(tmp_path, monkeypatch):
    context = ready(tmp_path, monkeypatch)
    seal(tmp_path / "launcher_execution.json", launcher.execution_identity(tmp_path))
    context.plan["source_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="launcher identity differs"):
        launcher.verify_execution(tmp_path)


def test_execution_identity_rejects_unselected_python(tmp_path, monkeypatch):
    ready(tmp_path, monkeypatch)
    monkeypatch.delenv("AF_PYTHON")
    with pytest.raises(ValueError, match="selected AF_PYTHON"):
        launcher.execution_identity(tmp_path)


def test_submit_prints_progress_before_authenticating_inputs(
    tmp_path, monkeypatch, capsys
):
    ready(tmp_path, monkeypatch)

    def load(root):
        assert "Authenticating compact staged-polishing inputs" in (
            capsys.readouterr().err
        )
        raise ValueError("invalid compact inputs")

    def forbidden(*args, **kwargs):
        raise AssertionError("unverified inputs must never reach the scheduler")

    monkeypatch.setattr(launcher.campaign, "load_run", load)
    monkeypatch.setattr(launcher, "submit_job", forbidden)
    with pytest.raises(ValueError, match="invalid compact inputs"):
        launcher.submit(tmp_path)


def test_execution_identity_rejects_changed_worker(tmp_path, monkeypatch):
    ready(tmp_path, monkeypatch)
    seal(tmp_path / "launcher_execution.json", launcher.execution_identity(tmp_path))
    original_hash = launcher.file_hash
    worker = (
        launcher.REPO / "scripts/hpc/run_llm_sr_detention_staged_polishing_delta.sh"
    )
    monkeypatch.setattr(
        launcher,
        "file_hash",
        lambda path: "b" * 64 if path == worker else original_hash(path),
    )
    with pytest.raises(ValueError, match="launcher identity differs"):
        launcher.verify_execution(tmp_path)


def test_prepare_freezes_one_search_budget_without_fitting(
    tmp_path, monkeypatch, capsys
):
    source = tmp_path / "input.json"
    inputs = {"protocol": "portable-training-endpoints"}
    seal(source, inputs)

    def prepare(root, raw, policy):
        assert root == tmp_path / "polishing" and raw == inputs
        assert policy.segments == 1
        assert policy.segment_seconds == 1800
        assert policy.verification_seconds == 120
        assert policy.direction_seconds == 360
        return {"tasks": 2, "fit_tasks": 2}

    def forbidden(*args, **kwargs):
        raise AssertionError("preparation must not fit or submit")

    monkeypatch.setattr(launcher.campaign, "prepare", prepare)
    monkeypatch.setattr(launcher.campaign, "run_task", forbidden)
    monkeypatch.setattr(launcher, "submit_job", forbidden)
    launcher.main(
        ["prepare", "--root", str(tmp_path / "polishing"), "--inputs", str(source)]
    )
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {"tasks": 2, "fit_tasks": 2}
    assert "freezing two staged-polishing tasks" in captured.err


def test_worker_shares_one_authenticated_context(tmp_path, monkeypatch, capsys):
    context = ready(tmp_path, monkeypatch)
    calls = []

    def load(root):
        assert root == tmp_path
        calls.append("load")
        return context

    def verify(root, *, verified_plan):
        assert root == tmp_path and verified_plan is context.plan
        calls.append("execution")

    def run(root, index, *, context):
        assert root == tmp_path and index == 1
        assert context is shared_context
        calls.append("run")
        return {"status": "complete", "retained": {"training_nmse": 0.1}}

    shared_context = context
    monkeypatch.setattr(launcher.campaign, "load_run", load)
    monkeypatch.setattr(launcher, "verify_execution", verify)
    monkeypatch.setattr(launcher.campaign, "run_task", run)
    launcher.main(["run", "--root", str(tmp_path), "--index", "1"])
    assert calls == ["load", "execution", "run"]
    captured = capsys.readouterr()
    assert json.loads(captured.out)["training_nmse"] == 0.1
    assert "Task 1: authentication complete" in captured.err


def test_lightweight_status_does_not_load_inputs_or_recompute_report(
    tmp_path, monkeypatch
):
    context = ready(tmp_path, monkeypatch)
    plan = context.plan
    receipt = {
        "identity": {
            "plan_sha256": public.content_sha256(plan),
            "task": plan["tasks"][0],
        },
        "status": "complete",
        "diagnostic_status": "passed",
        "initial_training_nmse": 0.1,
        "endpoint_assessment": {"status": "reliable_descent_remains"},
        "audit_decision": {"status": "skip_strict", "strict_audit_required": False},
        "strict_audit": {
            "status": "skipped",
            "performed": False,
            "skipped_reason": "reliable_descent_remains",
        },
        "full_model_stationarity_qualified": False,
        "initial_sample_reuse": {"seeded_active_columns": 9},
        "optimization": {"stop_reason": "budget_limited"},
        "retained": {"training_nmse": 0.01, "usable_for_retention": True},
    }
    seal(tmp_path / "tasks/0/result.json", receipt)
    seal(tmp_path / "tasks/1/operations/initial/started.json", {"reserved": 360})
    (tmp_path / "inputs.json").unlink()

    def forbidden(*args, **kwargs):
        raise AssertionError("lightweight status must not replay/authenticate history")

    for name in ("report", "verify", "load_run", "run_task"):
        monkeypatch.setattr(launcher.campaign, name, forbidden)
    result = launcher.status(tmp_path)
    assert result["status"] == "incomplete" and result["recorded"] == 1
    assert result["tasks"][0]["training_nmse"] == 0.01
    assert result["tasks"][0]["initial_training_nmse"] == 0.1
    assert result["tasks"][0]["optimizer_stop_reason"] == "budget_limited"
    assert (
        result["tasks"][0]["endpoint_assessment_status"] == "reliable_descent_remains"
    )
    assert result["tasks"][0]["initial_sample_reuse"] == {"seeded_active_columns": 9}
    assert result["tasks"][0]["strict_audit_performed"] is False
    assert result["tasks"][0]["strict_audit_skipped_reason"] == (
        "reliable_descent_remains"
    )
    assert result["tasks"][0]["retained_independently_verified"] is True
    assert result["tasks"][1]["started_operations"] == 1
    assert result["fitting_performed"] is False
    assert result["submission_performed"] is False


def test_compact_report_keeps_search_stopping_separate_from_endpoint_assessment():
    task = {"index": 1, "kind": "polishing", "source_index": 1}
    result = launcher.compact_report(
        {
            "protocol": "llm-sr-detention-staged-polishing-1",
            "status": "complete",
            "expected": 2,
            "recorded": 2,
            "tasks": [
                {
                    "identity": {"task": task},
                    "status": "complete",
                    "stop_reason": "single_search_allowance_consumed",
                    "optimization": {
                        "stop_reason": "active_block_stationary_poor_fit",
                        "initial_sample_reuse": {"jacobian_reuse_hits": 1},
                    },
                    "initial_training_nmse": 0.1,
                    "retained": {
                        "training_nmse": 0.05,
                        "usable_for_retention": True,
                    },
                    "candidate_verification": {
                        "status": "complete",
                        "usable_for_retention": True,
                    },
                    "verified_improvement": True,
                    "endpoint_assessment": {"status": "unresolved"},
                    "audit_decision": {
                        "decision": "strict_audit_required",
                        "strict_audit_required": True,
                        "coordinate_evidence": [{"large_evidence": [1.0] * 1000}],
                    },
                    "strict_audit": {"status": "complete", "performed": True},
                    "full_model_stationarity_qualified": False,
                }
            ],
        }
    )
    row = result["tasks"][0]
    assert row["stop_reason"] == "single_search_allowance_consumed"
    assert row["optimizer_stop_reason"] == "active_block_stationary_poor_fit"
    assert row["endpoint_assessment_status"] == "unresolved"
    assert row["audit_decision"]["strict_audit_required"] is True
    assert row["audit_decision"]["decision"] == "strict_audit_required"
    assert "coordinate_evidence" not in row["audit_decision"]
    assert row["strict_audit_status"] == "complete"
    assert row["strict_audit_performed"] is True
    assert row["strict_audit_skipped_reason"] is None
    assert row["full_model_stationarity_qualified"] is False
    assert row["candidate_verification_status"] == "complete"
    assert row["retained_independently_verified"] is True
    assert row["verified_improvement"] is True
    assert row["initial_sample_reuse"] == {"jacobian_reuse_hits": 1}
    assert row["initial_training_nmse"] == 0.1 and row["training_nmse"] == 0.05


@pytest.mark.parametrize("changed_field", ["plan_sha256", "task"])
def test_status_rejects_receipt_for_another_frozen_task(
    tmp_path, monkeypatch, changed_field
):
    plan = ready(tmp_path, monkeypatch).plan
    identity = {
        "plan_sha256": public.content_sha256(plan),
        "task": plan["tasks"][0],
    }
    identity[changed_field] = "wrong"
    seal(tmp_path / "tasks/0/result.json", {"identity": identity, "status": "complete"})
    with pytest.raises(ValueError, match="receipt differs"):
        launcher.status(tmp_path)


def test_inspect_includes_probe_caches_and_operation_receipts(tmp_path, monkeypatch):
    root = tmp_path / "polishing"
    ready(root, monkeypatch)
    summary = {"status": "incomplete", "expected": 2, "recorded": 0}
    monkeypatch.setattr(launcher.campaign, "report", lambda root: summary)
    seal(root / "tasks/0/operations/direction/started.json", {"reserved": 360})
    seal(root / "tasks/0/probe-cache/endpoint/base.json", {"residual": [1.0]})
    seal(root / "tasks/0/probe-cache/search/coordinate-1.json", {"status": "complete"})
    seal(root / "tasks/0/probe-cache/strict/coordinate-1.json", {"status": "complete"})
    seal(root / "tasks/0/audit-decision.json", {"strict_audit_required": True})
    seal(root / "invocations/task-0.json", {"stage": "run"})
    seal(root / "submissions/run.intent.json", {"argv": []})
    result = launcher.inspect(root)
    assert result["status"] == "incomplete"
    with tarfile.open(result["archive"]) as archive:
        names = archive.getnames()
    assert "polishing/inputs.json" in names
    assert "polishing/tasks/0/operations/direction/started.json" in names
    assert "polishing/tasks/0/probe-cache/endpoint/base.json" in names
    assert "polishing/tasks/0/probe-cache/search/coordinate-1.json" in names
    assert "polishing/tasks/0/probe-cache/strict/coordinate-1.json" in names
    assert "polishing/tasks/0/audit-decision.json" in names
    assert "polishing/report/summary.json" in names
    assert "polishing/invocations/task-0.json" in names
    assert "polishing/submissions/run.intent.json" in names


@pytest.mark.parametrize("index, expected_code", [("0", 0), ("1", 0), ("2", 2)])
def test_shell_worker_selects_array_index_and_rejects_out_of_roster(
    tmp_path, index, expected_code
):
    """Exercise actual shell routing without importing numerical dependencies."""
    fake_python = tmp_path / "python"
    fake_python.write_text('#!/bin/bash\nprintf "%s\\n" "$@"\n')
    fake_python.chmod(0o755)
    env = {
        **os.environ,
        "AF_REPO_ROOT": str(launcher.REPO),
        "AF_LLMSR_DETENTION_STAGED_POLISHING_ROOT": str(tmp_path / "campaign"),
        "AF_PYTHON": str(fake_python),
        "SLURM_ARRAY_TASK_ID": index,
    }
    result = subprocess.run(
        [
            "bash",
            str(
                launcher.REPO
                / "scripts/hpc/run_llm_sr_detention_staged_polishing_delta.sh"
            ),
            "run",
            "0",
        ],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == expected_code
    if expected_code == 0:
        assert result.stdout.splitlines()[-2:] == ["--index", index]
        assert "Detention staged polishing: stage=run" in result.stderr
    else:
        assert "Expected task index 0 through 1" in result.stderr
        assert not result.stdout


def test_shell_start_uses_distinct_root_and_compact_input(tmp_path):
    fake_python = tmp_path / "python"
    fake_python.write_text('#!/bin/bash\nprintf "%s\\n" "$@"\n')
    fake_python.chmod(0o755)
    env = {
        **os.environ,
        "AF_PYTHON": str(fake_python),
        "AF_OUTPUT_ROOT": "/unrelated/campaign",
        "AF_LLMSR_DETENTION_PROBE_RECOVERY_ROOT": "/historical/probe-recovery",
    }
    env.pop("AF_LLMSR_DETENTION_STAGED_POLISHING_ROOT", None)
    env.pop("AF_LLMSR_DETENTION_STAGED_POLISHING_INPUTS", None)
    result = subprocess.run(
        [
            "bash",
            str(
                launcher.REPO
                / "scripts/hpc/start_llm_sr_detention_staged_polishing_delta.sh"
            ),
            "run",
        ],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0
    argv = result.stdout.splitlines()
    assert argv.count("prepare") == argv.count("submit") == 1
    assert argv.count(
        "/work/hdd/bibo/yxiao2/phase_c/llm-sr-detention-staged-polishing-v1"
    ) == 2
    assert str(
        launcher.REPO / "inputs/llm-sr-detention-staged-polishing-input.json"
    ) in argv
    assert "/unrelated/campaign" not in argv
    assert "/historical/probe-recovery" not in argv
