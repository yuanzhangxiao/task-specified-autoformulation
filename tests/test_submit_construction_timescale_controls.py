"""Portable CPU continuation scheduling never retries uncertain submissions."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_construction_timescale_controls as cli
from scripts import submit_review_continuation as scheduler

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    root, code = tmp_path / "results", tmp_path / "bundle"
    plan = {
        "tasks": [{"model": i // 2, "arm": str(i % 2)} for i in range(10)],
        "source_sha256": "frozen-source",
    }
    monkeypatch.setattr(cli.campaign, "verify", lambda *a, **k: (plan, {}))
    monkeypatch.setattr(cli, "runtime_identity", lambda: {"mode": "native"})
    for name in cli.ENTRYPOINTS:
        path = code / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# frozen entrypoint\n")
    monkeypatch.setattr(cli, "REPO", code)
    monkeypatch.setenv("AF_PLATFORM", "delta")
    monkeypatch.delenv("AF_ACCOUNT", raising=False)
    return root, code, plan


def accepting(monkeypatch):
    commands = []

    def accept(argv, **kwargs):
        commands.append(argv)
        return SimpleNamespace(
            returncode=0, stdout=f"{100 + len(commands)}\n", stderr=""
        )

    monkeypatch.setattr(scheduler.subprocess, "run", accept)
    return commands


def test_derived_ten_cpu_tasks_afterany_report_and_idempotent_receipts(
    prepared, monkeypatch
):
    root, _, _ = prepared
    commands = accepting(monkeypatch)
    monkeypatch.setenv("AF_OUTPUT_ROOT", "/unrelated/historical/campaign")
    result = cli.submit(root)
    assert result["jobs"] == {"run": "101", "report": "102"}
    assert result["tasks"] == 10 and result["maximum_parallel_tasks"] == 2
    assert "--array=0-9%2" in commands[0]
    assert "--time=01:00:00" in commands[0]
    assert "--time=00:15:00" in commands[1]
    assert "--dependency=afterany:101" in commands[1]
    for command in commands:
        assert "--account=bibo-delta-cpu" in command
        assert "--partition=cpu" in command
        assert "--cpus-per-task=1" in command
        assert "--mem=8G" in command
        assert not any("gpu" in part or "--exclude" in part for part in command)
    assert os.environ["AF_TIMESCALE_CONTROLS_ROOT"] == str(root)
    assert os.environ["AF_OUTPUT_ROOT"] == "/unrelated/historical/campaign"
    assert cli.submit(root) == result
    assert len(commands) == 2
    assert cli.verify_execution(root)["source_sha256"] == "frozen-source"


def test_optional_aces_platform_and_array_size_are_bound_to_execution(
    prepared, monkeypatch
):
    root, _, plan = prepared
    plan["tasks"] = plan["tasks"][:4]
    monkeypatch.setenv("AF_PLATFORM", "aces")
    commands = accepting(monkeypatch)
    result = cli.submit(root)
    assert result["tasks"] == 4 and "--array=0-3%2" in commands[0]
    assert "--account=156264627414" in commands[0]
    assert "--exclude=ac042" in commands[0]
    monkeypatch.setenv("AF_PLATFORM", "delta")
    with pytest.raises(ValueError, match="identity differs"):
        cli.verify_execution(root)


@pytest.mark.parametrize("failure", ["blank", "nonzero", "timeout"])
def test_uncertain_reply_never_renews_submission(prepared, monkeypatch, failure):
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
    assert (root / cli.SUBMISSIONS / "run.reply.json").exists()
    assert not (root / cli.SUBMISSIONS / "report.intent.json").exists()


def test_confirmed_array_is_reused_when_report_submission_is_uncertain(
    prepared, monkeypatch
):
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


@pytest.mark.parametrize("change", ["source", "entrypoint", "runtime"])
def test_changed_execution_cannot_submit_or_execute(prepared, monkeypatch, change):
    root, code, plan = prepared
    commands = accepting(monkeypatch)
    cli.submit(root)
    if change == "source":
        plan["source_sha256"] = "different-source"
    elif change == "entrypoint":
        (code / cli.ENTRYPOINTS[1]).write_text("changed worker\n")
    else:
        monkeypatch.setattr(cli, "runtime_identity", lambda: {"mode": "changed"})
    with pytest.raises(ValueError, match="identity differs"):
        cli.verify_execution(root)
    with pytest.raises(ValueError):
        cli.submit(root)
    assert len(commands) == 2


def test_manifest_does_not_bypass_confirmed_scheduler_receipts(prepared, monkeypatch):
    root, _, _ = prepared
    commands = accepting(monkeypatch)
    cli.submit(root)
    receipt = root / cli.SUBMISSIONS / "run.reply.json"
    receipt.write_text(json.dumps({"returncode": 0, "stdout": "999\n"}))
    with pytest.raises(ValueError, match="unconfirmed run"):
        cli.submit(root)
    assert len(commands) == 2


def test_run_cli_requires_execution_verification_before_numerical_work(
    tmp_path, monkeypatch
):
    calls = []

    def rejected(root):
        calls.append("verify")
        raise ValueError("launcher identity differs")

    monkeypatch.setattr(cli, "verify_execution", rejected)
    monkeypatch.setattr(
        cli.campaign, "run_task", lambda *args: pytest.fail("run before verification")
    )
    with pytest.raises(ValueError, match="identity differs"):
        cli.main(["run", "--root", str(tmp_path), "--index", "1"])
    assert calls == ["verify"]


def test_export_cli_needs_source_and_destination_but_no_campaign_root(
    tmp_path, monkeypatch, capsys
):
    source, destination = tmp_path / "source", tmp_path / "portable/inputs.json"
    calls = []
    monkeypatch.setattr(
        cli.campaign, "export_inputs", lambda *args: calls.append(args) or {"models": 8}
    )
    cli.main(["export", "--source", str(source), "--destination", str(destination)])
    assert calls == [(source, destination)]
    assert json.loads(capsys.readouterr().out) == {"models": 8}


def shell_environment(tmp_path):
    """Exercise shell dispatch without running Python science or contacting Slurm."""
    python = tmp_path / "python"
    python.write_text('#!/bin/bash\nprintf "%s\\n" "$@" >> "$AF_ARGUMENT_LOG"\n')
    python.chmod(0o755)
    squeue = tmp_path / "squeue"
    squeue.write_text("#!/bin/bash\nexit 0\n")
    squeue.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if not k.startswith("AF_")}
    env.update(
        AF_REPO_ROOT=str(REPO),
        AF_TIMESCALE_CONTROLS_ROOT=str(tmp_path / "continuation"),
        AF_OUTPUT_ROOT=str(tmp_path / "stale-unrelated"),
        AF_PYTHON=str(python),
        AF_ARGUMENT_LOG=str(tmp_path / "args.txt"),
        PATH=f"{tmp_path}:{env['PATH']}",
    )
    return env


def test_shell_launcher_uses_portable_inputs_and_ignores_stale_output_root(tmp_path):
    env = shell_environment(tmp_path)
    launcher = REPO / "scripts/hpc/start_construction_timescale_controls.sh"
    worker = REPO / "scripts/hpc/run_construction_timescale_controls.sh"
    for path in (launcher, worker):
        subprocess.run(["bash", "-n", str(path)], check=True)
    subprocess.run(["bash", str(launcher), "run"], env=env, check=True)
    args = (tmp_path / "args.txt").read_text().splitlines()
    assert "prepare" in args and "submit" in args
    assert str(REPO / "inputs/timescale-controls-inputs.json") in args
    assert str(REPO / "configs/construction_timescale_controls_v1.json") in args
    assert env["AF_TIMESCALE_CONTROLS_ROOT"] in args
    assert env["AF_OUTPUT_ROOT"] not in args
    assert not {"pytest", "qualify"} & set(args)
    assert args.count("-c") == 4


@pytest.mark.parametrize("command", ["status", "report", "inspect"])
def test_shell_readonly_commands_do_not_route_to_prepare_submit_or_fit(
    tmp_path, command
):
    env = shell_environment(tmp_path)
    launcher = REPO / "scripts/hpc/start_construction_timescale_controls.sh"
    subprocess.run(["bash", str(launcher), command], env=env, check=True)
    args = (tmp_path / "args.txt").read_text().splitlines()
    assert command in args
    assert not {"run", "prepare", "submit", "--index"} & set(args)


def test_worker_uses_array_index_and_rejects_invalid_index(tmp_path):
    env = shell_environment(tmp_path) | {
        "SLURM_ARRAY_TASK_ID": "7",
        "AF_PLATFORM": "aces",
    }
    worker = REPO / "scripts/hpc/run_construction_timescale_controls.sh"
    subprocess.run(["bash", str(worker), "run", "0"], env=env, check=True)
    args = (tmp_path / "args.txt").read_text().splitlines()
    assert args[args.index("--index") + 1] == "7"
    assert env["AF_OUTPUT_ROOT"] not in args
    env["SLURM_ARRAY_TASK_ID"] = "invalid"
    result = subprocess.run(
        ["bash", str(worker), "run"], env=env, capture_output=True, text=True
    )
    assert result.returncode != 0 and "nonnegative task index" in result.stderr


def test_worker_report_discards_scheduler_index_placeholder(tmp_path):
    env = shell_environment(tmp_path)
    worker = REPO / "scripts/hpc/run_construction_timescale_controls.sh"
    subprocess.run(["bash", str(worker), "report", "0"], env=env, check=True)
    args = (tmp_path / "args.txt").read_text().splitlines()
    assert "report" in args and "0" not in args and "--index" not in args


def test_runtime_identity_binds_precision_package_source(tmp_path, monkeypatch):
    import sys

    package = tmp_path / "mpmath"
    package.mkdir()
    module = package / "__init__.py"
    module.write_text("# first\n")
    monkeypatch.setitem(
        sys.modules,
        "mpmath",
        SimpleNamespace(
            __file__=str(module),
            __version__="1.3.0",
        ),
    )
    monkeypatch.setenv("AF_FIT_RUNTIME", "native")
    monkeypatch.setattr(cli, "_runtime_identity", lambda: {"mode": "native"})
    before = cli.runtime_identity()
    assert before["mpmath_version"] == "1.3.0"
    module.write_text("# second\n")
    assert (
        before["mpmath_source_sha256"] != cli.runtime_identity()["mpmath_source_sha256"]
    )
