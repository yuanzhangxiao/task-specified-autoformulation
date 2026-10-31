"""CPU portfolio scheduling preserves identities and never retries ambiguous jobs."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import public_fitting as public
from scripts import run_construction_portfolio_continuation as cli
from scripts import submit_review_continuation as scheduler

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    """Use scheduler fixtures, not numerical fits or cluster sessions."""
    root, code = tmp_path / "results", tmp_path / "code"
    plan = {
        "protocol": "construction-portfolio-continuation-1",
        "source_sha256": "frozen-source",
        "tasks": [
            {
                "index": i,
                "task_id": f"candidate-{candidate:02d}",
                "candidate_id": f"candidate-{candidate:02d}",
                "kind": "nonlinear_prerequisites"
                if candidate in (3, 10, 14)
                else "affine_continuation",
            }
            for i, candidate in enumerate((3, 10, 12, 14, 15))
        ],
    }
    seal(root / "plan.json", plan)
    monkeypatch.setattr(cli.campaign, "verify", lambda *a, **k: (plan, {}))
    monkeypatch.setattr(cli.campaign, "reservation_ceiling", lambda plan: 9600)
    monkeypatch.setattr(cli, "runtime_identity", lambda: {"mode": "native"})
    for name in cli.ENTRYPOINTS:
        path = code / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# frozen entrypoint\n")
    monkeypatch.setattr(cli, "REPO", code)
    monkeypatch.setenv("AF_REPO_ROOT", str(code))
    monkeypatch.setenv("AF_CONSTRUCTION_PORTFOLIO_CONTINUATION_ROOT", str(root))
    monkeypatch.delenv("AF_ACCOUNT", raising=False)
    monkeypatch.delenv(
        "AF_CONSTRUCTION_PORTFOLIO_CONTINUATION_CONCURRENCY", raising=False
    )
    return root, code, plan


def accepting(monkeypatch):
    """Exercise the real intent/reply/id writer against mocked Slurm replies."""
    commands = []

    def accept(argv, **kwargs):
        commands.append(argv)
        return SimpleNamespace(
            returncode=0, stdout=f"{100 + len(commands)}\n", stderr=""
        )

    monkeypatch.setattr(scheduler.subprocess, "run", accept)
    return commands


def test_five_cpu_tasks_afterany_report_and_deduplicated_submission(
    prepared, monkeypatch, capsys
):
    root, _, _ = prepared
    commands = accepting(monkeypatch)
    monkeypatch.setenv("AF_OUTPUT_ROOT", "/unrelated/campaign")
    result = cli.submit(root)
    assert result["jobs"] == {"run": "101", "report": "102"}
    assert result["tasks"] == 5 and result["maximum_parallel_tasks"] == 2
    assert result["gpus"] == result["llm_calls"] == 0
    assert result["validation_or_test_access"] is False
    assert "--array=0-4%2" in commands[0]
    assert "--time=04:00:00" in commands[0]
    assert "--dependency=afterany:101" in commands[1]
    assert "--time=00:30:00" in commands[1]
    assert commands[0][-2:] == ["worker-run", "0"]
    assert commands[1][-2:] == ["worker-report", "0"]
    for command in commands:
        assert "--account=bibo-delta-cpu" in command
        assert "--partition=cpu" in command
        assert "--cpus-per-task=1" in command and "--mem=8G" in command
        assert not any("gpu" in part for part in command)
    assert os.environ["AF_CONSTRUCTION_PORTFOLIO_CONTINUATION_ROOT"] == str(root)
    assert os.environ["AF_OUTPUT_ROOT"] == "/unrelated/campaign"
    messages = capsys.readouterr().err
    assert messages.index("Slurm accepted run: job 101") < messages.index(
        "Submitting report"
    )
    assert cli.submit(root) == result
    assert len(commands) == 2
    assert cli.verify_execution(root)["source_sha256"] == "frozen-source"


def test_submit_verifies_scientific_inputs_once(prepared, monkeypatch):
    root, _, plan = prepared
    accepting(monkeypatch)
    calls = []
    monkeypatch.setattr(
        cli.campaign, "verify", lambda *a, **k: (calls.append(a), (plan, {}))[1]
    )
    cli.submit(root)
    assert calls == [(root,)]


def test_continuation_identity_cannot_reuse_historical_submission(
    prepared, monkeypatch
):
    root, _, _ = prepared
    accepting(monkeypatch)
    result = cli.submit(root)
    identity = read_seal(root / "launcher_execution.json")
    assert identity["protocol"] == "construction-portfolio-continuation-cpu-execution-1"
    assert identity["maximum_reserved_seconds_per_task"] == 9600
    assert cli.SUBMISSIONS == "submissions/cpu-construction-portfolio-continuation-1"
    assert read_seal(root / cli.SUBMISSIONS / "manifest.json") == result
    assert not (root / "submissions/cpu-construction-portfolio-recovery-1").exists()


@pytest.mark.parametrize("seconds", [13201, 0, float("nan"), float("inf")])
def test_oversized_or_invalid_reservations_cannot_submit(
    prepared, monkeypatch, seconds
):
    root, _, _ = prepared
    commands = accepting(monkeypatch)
    monkeypatch.setattr(cli.campaign, "reservation_ceiling", lambda plan: seconds)
    with pytest.raises(ValueError, match="four-hour worker envelope"):
        cli.submit(root)
    assert not commands


def test_serial_execution_preserves_bounded_worker_resources(prepared, monkeypatch):
    root, _, _ = prepared
    commands = accepting(monkeypatch)
    monkeypatch.setenv("AF_CONSTRUCTION_PORTFOLIO_CONTINUATION_CONCURRENCY", "1")
    result = cli.submit(root)
    assert result["maximum_parallel_tasks"] == 1
    assert "--array=0-4%1" in commands[0]


@pytest.mark.parametrize("failure", ["blank", "nonzero", "timeout"])
def test_ambiguous_reply_is_never_resubmitted(prepared, monkeypatch, failure):
    root, _, _ = prepared
    calls = []

    def fail(argv, **kwargs):
        calls.append(argv)
        if failure == "timeout":
            raise subprocess.TimeoutExpired(argv, 60)
        return SimpleNamespace(
            returncode=int(failure == "nonzero"), stdout="", stderr="uncertain"
        )

    monkeypatch.setattr(scheduler.subprocess, "run", fail)
    with pytest.raises(ValueError, match=r"uncertain|unconfirmed"):
        cli.submit(root)
    with pytest.raises(ValueError, match="uncertain"):
        cli.submit(root)
    assert len(calls) == 1
    assert not (root / cli.SUBMISSIONS / "report.intent.json").exists()


def test_array_confirmation_survives_ambiguous_report(prepared, monkeypatch):
    root, _, _ = prepared
    calls = []

    def reply(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(
            returncode=0, stdout="103\n" if len(calls) == 1 else "", stderr=""
        )

    monkeypatch.setattr(scheduler.subprocess, "run", reply)
    with pytest.raises(ValueError, match="unconfirmed"):
        cli.submit(root)
    with pytest.raises(ValueError, match="uncertain report"):
        cli.submit(root)
    assert len(calls) == 2
    assert (root / cli.SUBMISSIONS / "run.id").read_text().strip() == "103"


@pytest.mark.parametrize("change", ["source", "entrypoint", "runtime", "concurrency"])
def test_frozen_execution_rejects_changes(prepared, monkeypatch, change):
    root, code, plan = prepared
    commands = accepting(monkeypatch)
    cli.submit(root)
    if change == "source":
        plan["source_sha256"] = "new-source"
    elif change == "entrypoint":
        (code / cli.ENTRYPOINTS[1]).write_text("changed worker\n")
    elif change == "concurrency":
        monkeypatch.setenv("AF_CONSTRUCTION_PORTFOLIO_CONTINUATION_CONCURRENCY", "1")
    else:
        monkeypatch.setattr(cli, "runtime_identity", lambda: {"mode": "changed"})
    with pytest.raises(ValueError, match="identity differs"):
        cli.verify_execution(root)
    with pytest.raises(ValueError):
        cli.submit(root)
    assert len(commands) == 2


@pytest.mark.parametrize("value", ["0", "3", "24", "-1", "two"])
def test_invalid_concurrency_cannot_submit(prepared, monkeypatch, value):
    root, _, _ = prepared
    commands = accepting(monkeypatch)
    monkeypatch.setenv("AF_CONSTRUCTION_PORTFOLIO_CONTINUATION_CONCURRENCY", value)
    with pytest.raises(ValueError, match="CONCURRENCY"):
        cli.submit(root)
    assert not commands


def test_empty_array_cannot_submit(prepared, monkeypatch):
    root, _, plan = prepared
    plan["tasks"] = []
    commands = accepting(monkeypatch)
    with pytest.raises(ValueError, match="nonempty task array"):
        cli.submit(root)
    assert not commands


def test_manifest_never_bypasses_scheduler_receipt_checks(prepared, monkeypatch):
    root, _, _ = prepared
    commands = accepting(monkeypatch)
    cli.submit(root)
    path = root / cli.SUBMISSIONS / "run.reply.json"
    path.write_text(json.dumps({"returncode": 0, "stdout": "999\n"}))
    with pytest.raises(ValueError, match="unconfirmed run"):
        cli.submit(root)
    assert len(commands) == 2


@pytest.mark.parametrize(
    "continuation_status",
    ["continued_verified", "qualified_for_search", "derivatives_unqualified"],
)
def test_status_uses_only_bound_terminal_receipts(
    prepared, monkeypatch, continuation_status
):
    root, _, plan = prepared
    task = plan["tasks"][0]
    terminal = root / "results" / task["task_id"] / "result.json"
    identity = {"plan_sha256": public.content_sha256(plan), "task": task}
    seal(
        terminal,
        {
            "identity": identity,
            "status": "complete",
            "continuation_status": continuation_status,
        },
    )
    monkeypatch.setattr(
        cli.campaign, "report", lambda *a: pytest.fail("expensive report")
    )
    monkeypatch.setattr(cli.campaign, "verify", lambda *a: pytest.fail("input load"))
    result = cli.status(root)
    assert result["expected"] == 5 and result["recorded"] == 1
    assert [row["candidate_id"] for row in result["tasks"]] == [
        "candidate-03",
        "candidate-10",
        "candidate-12",
        "candidate-14",
        "candidate-15",
    ]
    assert result["tasks"][0]["index"] == 0
    assert result["tasks"][0]["kind"] == "nonlinear_prerequisites"
    assert result["tasks"][0]["status"] == "complete"
    assert result["tasks"][0]["continuation_status"] == continuation_status
    assert "fitting_status" not in result["tasks"][0]
    assert result["tasks"][0]["terminal_receipt_present"]
    changed = read_seal(terminal)
    changed["identity"]["plan_sha256"] = "different"
    public._write(
        terminal, {"value": changed, "sha256": public.content_sha256(changed)}
    )
    with pytest.raises(ValueError, match="status receipt differs"):
        cli.status(root)


def test_run_checks_execution_before_fitting(tmp_path, monkeypatch):
    def rejected(root):
        raise ValueError("launcher identity differs")

    monkeypatch.setattr(cli, "verify_execution", rejected)
    monkeypatch.setattr(cli.campaign, "run_task", lambda *a: pytest.fail("fitted"))
    with pytest.raises(ValueError, match="identity differs"):
        cli.main(["run", "--root", str(tmp_path), "--index", "0"])


def test_run_prints_continuation_status_from_completed_campaign(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(cli, "verify_execution", lambda root: {})
    monkeypatch.setattr(
        cli.campaign,
        "run_task",
        lambda root, index: {
            "status": "complete",
            "continuation_status": "continued_verified",
        },
    )
    cli.main(["run", "--root", str(tmp_path), "--index", "0"])
    assert json.loads(capsys.readouterr().out) == {
        "index": 0,
        "status": "complete",
        "continuation_status": "continued_verified",
    }


def test_status_rejects_path_traversal_before_reading_results(tmp_path):
    seal(tmp_path / "plan.json", {"tasks": [{"task_id": "../foreign"}]})
    with pytest.raises(ValueError, match="unsafe portfolio continuation task path"):
        cli.status(tmp_path)


def test_inspect_prints_exact_archive_path(tmp_path, monkeypatch, capsys):
    archive = tmp_path / "inspection.tar.gz"
    monkeypatch.setattr(cli.campaign, "inspection", lambda root: archive)
    cli.main(["inspect", "--root", str(tmp_path)])
    assert json.loads(capsys.readouterr().out) == {"archive": str(archive)}


def test_prepare_uses_source_override_without_fitting_or_submission(
    tmp_path, monkeypatch, capsys
):
    root, source = tmp_path / "continuation", tmp_path / "historical"
    calls = []
    monkeypatch.setenv("AF_CONSTRUCTION_PORTFOLIO_CONTINUATION_SOURCE", str(source))
    monkeypatch.setattr(
        cli.campaign,
        "prepare",
        lambda *args: calls.append(args) or {"tasks": 5},
    )
    monkeypatch.setattr(cli, "submit", lambda *args: pytest.fail("submitted"))
    monkeypatch.setattr(cli.campaign, "run_task", lambda *args: pytest.fail("fitted"))
    cli.main(["prepare", "--root", str(root)])
    assert calls[0][:2] == (root, source)
    assert calls[0][2].population == "scaled-readout-5"
    assert json.loads(capsys.readouterr().out) == {"tasks": 5}


def test_verify_is_portable_but_execution_verification_pins_runtime(
    prepared, monkeypatch, capsys
):
    root, _, plan = prepared
    calls = []

    def verify(*args, **kwargs):
        calls.append((args, kwargs))
        return plan, {}

    monkeypatch.setattr(cli.campaign, "verify", verify)
    cli.main(["verify", "--root", str(root)])
    assert calls == [((root,), {"runtime": False})]
    assert json.loads(capsys.readouterr().out) == {"verified": True, "tasks": 5}
    cli.execution_identity(root)
    assert calls[-1] == ((root,), {})


def test_default_source_uses_recovery_campaign_and_honors_override(
    tmp_path, monkeypatch
):
    remote, bundle = tmp_path / "remote", tmp_path / "inputs/portfolio-source"
    monkeypatch.setattr(cli, "REPO", tmp_path)
    monkeypatch.setattr(cli, "DEFAULT_SOURCE", remote)
    monkeypatch.delenv("AF_CONSTRUCTION_PORTFOLIO_CONTINUATION_SOURCE", raising=False)
    monkeypatch.setenv("AF_CONSTRUCTION_PORTFOLIO_SOURCE", "/old/portfolio-fitting")
    assert cli.default_source() == remote
    bundle.mkdir(parents=True)
    assert cli.default_source() == remote
    remote.mkdir()
    assert cli.default_source() == remote
    explicit = tmp_path / "explicit"
    monkeypatch.setenv("AF_CONSTRUCTION_PORTFOLIO_CONTINUATION_SOURCE", str(explicit))
    assert cli.default_source() == explicit


def shell_environment(tmp_path):
    """Record shell argument routing without importing science or contacting Slurm."""
    python = tmp_path / "python"
    python.write_text('#!/bin/bash\nprintf "%s\\n" "$@" >> "$AF_ARGUMENT_LOG"\n')
    python.chmod(0o755)
    source = tmp_path / "source"
    source.mkdir()
    (source / "plan.json").write_text("{}")
    (source / "inputs.json").write_text("{}")
    env = {k: v for k, v in os.environ.items() if not k.startswith("AF_")}
    env.update(
        AF_CONSTRUCTION_PORTFOLIO_CONTINUATION_ROOT=str(tmp_path / "campaign"),
        AF_CONSTRUCTION_PORTFOLIO_CONTINUATION_SOURCE=str(source),
        AF_CONSTRUCTION_PORTFOLIO_RECOVERY_ROOT=str(tmp_path / "historical-recovery"),
        AF_CONSTRUCTION_PORTFOLIO_ROOT=str(tmp_path / "historical-campaign"),
        AF_OUTPUT_ROOT=str(tmp_path / "wrong-campaign"),
        AF_PYTHON=str(python),
        AF_ARGUMENT_LOG=str(tmp_path / "args.txt"),
    )
    return env


def test_shell_run_prepares_then_submits_specific_campaign(tmp_path):
    env = shell_environment(tmp_path)
    wrapper = REPO / "scripts/hpc/start_construction_portfolio_continuation_delta.sh"
    subprocess.run(["bash", "-n", str(wrapper)], check=True)
    subprocess.run(["bash", str(wrapper), "run"], env=env, check=True)
    args = (tmp_path / "args.txt").read_text().splitlines()
    assert args.index("prepare") < args.index("submit")
    assert env["AF_CONSTRUCTION_PORTFOLIO_CONTINUATION_ROOT"] in args
    assert env["AF_CONSTRUCTION_PORTFOLIO_CONTINUATION_SOURCE"] in args
    assert env["AF_CONSTRUCTION_PORTFOLIO_ROOT"] not in args
    assert env["AF_CONSTRUCTION_PORTFOLIO_RECOVERY_ROOT"] not in args
    assert env["AF_OUTPUT_ROOT"] not in args
    assert str(REPO / "configs/construction_portfolio_continuation_v1.json") in args
    assert not {"pytest", "run", "--index"} & set(args)


@pytest.mark.parametrize("command", ["status", "report", "inspect"])
def test_shell_inspection_never_prepares_or_submits(tmp_path, command):
    env = shell_environment(tmp_path)
    wrapper = REPO / "scripts/hpc/start_construction_portfolio_continuation_delta.sh"
    subprocess.run(["bash", str(wrapper), command], env=env, check=True)
    args = (tmp_path / "args.txt").read_text().splitlines()
    assert command in args
    assert not {"prepare", "submit", "run", "--index"} & set(args)


def test_worker_uses_array_index_and_rejects_invalid_index(tmp_path):
    env = shell_environment(tmp_path) | {
        "SLURM_ARRAY_TASK_ID": "4",
        "AF_REPO_ROOT": str(REPO),
    }
    wrapper = REPO / "scripts/hpc/start_construction_portfolio_continuation_delta.sh"
    subprocess.run(["bash", str(wrapper), "worker-run", "0"], env=env, check=True)
    args = (tmp_path / "args.txt").read_text().splitlines()
    assert args[args.index("--index") + 1] == "4"
    assert "prepare" not in args and "submit" not in args
    env["SLURM_ARRAY_TASK_ID"] = "invalid"
    result = subprocess.run(
        ["bash", str(wrapper), "worker-run"], env=env, capture_output=True, text=True
    )
    assert result.returncode != 0 and "nonnegative task index" in result.stderr


def copied_slurm_script(tmp_path):
    """Mirror sbatch's copied script location, independently of the source tree."""
    original = REPO / "scripts/hpc/start_construction_portfolio_continuation_delta.sh"
    copied = tmp_path / "spool/slurmd/job23332572/slurm_script"
    copied.parent.mkdir(parents=True)
    copied.write_bytes(original.read_bytes())
    return copied


@pytest.mark.parametrize(
    "stage,command", [("worker-run", "run"), ("worker-report", "report")]
)
def test_spool_copy_workers_preserve_exported_repository(tmp_path, stage, command):
    env = shell_environment(tmp_path) | {
        "AF_REPO_ROOT": str(REPO),
        "SLURM_ARRAY_TASK_ID": "4",
    }
    copied = copied_slurm_script(tmp_path)
    subprocess.run(["bash", str(copied), stage, "0"], env=env, check=True)
    args = (tmp_path / "args.txt").read_text().splitlines()
    assert args[0] == str(REPO / "scripts/run_construction_portfolio_continuation.py")
    assert args[1] == command
    assert (
        args[args.index("--root") + 1]
        == env["AF_CONSTRUCTION_PORTFOLIO_CONTINUATION_ROOT"]
    )
    assert "prepare" not in args and "submit" not in args
    if stage == "worker-run":
        assert args[args.index("--index") + 1] == "4"
    else:
        assert "--index" not in args


@pytest.mark.parametrize("stage", ["worker-run", "worker-report"])
@pytest.mark.parametrize("missing", [True, False])
def test_spool_copy_requires_nonempty_exported_repository(tmp_path, stage, missing):
    env = shell_environment(tmp_path)
    if not missing:
        env["AF_REPO_ROOT"] = ""
    copied = copied_slurm_script(tmp_path)
    result = subprocess.run(
        ["bash", str(copied), stage, "0"], env=env, capture_output=True, text=True
    )
    assert result.returncode != 0
    assert "Slurm workers require exported AF_REPO_ROOT" in result.stderr
    assert not (tmp_path / "args.txt").exists()


@pytest.mark.parametrize("stage", ["run", "status"])
def test_login_stage_ignores_stale_inherited_repository(tmp_path, stage):
    env = shell_environment(tmp_path) | {
        "AF_REPO_ROOT": str(tmp_path / "unrelated-old-checkout")
    }
    wrapper = REPO / "scripts/hpc/start_construction_portfolio_continuation_delta.sh"
    subprocess.run(["bash", str(wrapper), stage], env=env, check=True)
    args = (tmp_path / "args.txt").read_text().splitlines()
    assert args[0] == str(REPO / "scripts/run_construction_portfolio_continuation.py")
    assert env["AF_REPO_ROOT"] not in args


def test_continuation_launcher_defaults_to_separate_result_root(tmp_path):
    env = shell_environment(tmp_path)
    env.pop("AF_CONSTRUCTION_PORTFOLIO_CONTINUATION_ROOT")
    wrapper = REPO / "scripts/hpc/start_construction_portfolio_continuation_delta.sh"
    subprocess.run(["bash", str(wrapper), "status"], env=env, check=True)
    args = (tmp_path / "args.txt").read_text().splitlines()
    assert args[args.index("--root") + 1] == (
        "/work/hdd/bibo/yxiao2/phase_c/construction-portfolio-continuation-v1"
    )
    assert env["AF_OUTPUT_ROOT"] not in args


@pytest.mark.parametrize("missing", ["directory", "plan", "inputs"])
def test_missing_source_stops_before_prepare_or_submit(tmp_path, missing):
    env = shell_environment(tmp_path)
    if missing == "directory":
        env["AF_CONSTRUCTION_PORTFOLIO_CONTINUATION_SOURCE"] = str(tmp_path / "missing")
    else:
        (
            Path(env["AF_CONSTRUCTION_PORTFOLIO_CONTINUATION_SOURCE"])
            / f"{missing}.json"
        ).unlink()
    wrapper = REPO / "scripts/hpc/start_construction_portfolio_continuation_delta.sh"
    result = subprocess.run(
        ["bash", str(wrapper), "run"], env=env, capture_output=True, text=True
    )
    assert (
        result.returncode != 0 and "complete saved recovery campaign" in result.stderr
    )
    assert not (tmp_path / "args.txt").exists()


@pytest.mark.parametrize("remote_present", [True, False])
def test_code_bundle_requires_existing_default_source(tmp_path, remote_present):
    env = shell_environment(tmp_path)
    env.pop("AF_CONSTRUCTION_PORTFOLIO_CONTINUATION_SOURCE")
    repo = tmp_path / "extracted"
    bundled, remote = repo / "inputs/portfolio-source", tmp_path / "remote"
    for source in (bundled, remote) if remote_present else (bundled,):
        source.mkdir(parents=True)
        (source / "plan.json").write_text("{}")
        (source / "inputs.json").write_text("{}")
    wrapper = repo / "scripts/hpc/start_construction_portfolio_continuation_delta.sh"
    wrapper.parent.mkdir(parents=True)
    original = REPO / "scripts/hpc/start_construction_portfolio_continuation_delta.sh"
    wrapper.write_text(
        original.read_text().replace(str(cli.DEFAULT_SOURCE), str(remote))
    )
    script = repo / "scripts/run_construction_portfolio_continuation.py"
    script.write_text("# fake interpreter records arguments\n")
    result = subprocess.run(
        ["bash", str(wrapper), "prepare"], env=env, capture_output=True, text=True
    )
    if not remote_present:
        assert result.returncode != 0
        assert "complete saved recovery campaign" in result.stderr
        assert not (tmp_path / "args.txt").exists()
        return
    assert result.returncode == 0
    args = (tmp_path / "args.txt").read_text().splitlines()
    assert args[args.index("--source") + 1] == str(remote)
    assert args[0] == str(script)
    assert not {"submit", "run", "--index"} & set(args)
