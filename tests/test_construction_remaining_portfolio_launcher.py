"""Frozen 22-model CPU submission, recovery safety and shell routing."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import public_fitting as public
from scripts import run_construction_remaining_portfolio as cli
from scripts import submit_review_continuation as scheduler

REPO = Path(__file__).resolve().parents[1]
START = REPO / "scripts/hpc/start_construction_remaining_portfolio_delta.sh"
WORKER = REPO / "scripts/hpc/run_construction_remaining_portfolio_delta.sh"


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    """Freeze scheduler fixtures without numerical work or cluster access."""
    root, code = tmp_path / "campaign", tmp_path / "code"
    plan = {
        "source_sha256": "authenticated-compact-source",
        "tasks": [
            {
                "index": i,
                "task_id": f"candidate-{j:02}",
                "candidate_id": f"candidate-{j:02}",
                "route": "affine",
            }
            for i, j in enumerate(j for j in range(23) if j != 15)
        ],
    }
    seal(root / "plan.json", plan)
    monkeypatch.setattr(cli.campaign, "verify", lambda *a, **kw: (plan, {}))
    monkeypatch.setattr(cli.campaign, "reservation_ceiling", lambda plan: 9000)
    monkeypatch.setattr(cli, "runtime_identity", lambda: {"mode": "native"})
    for name in cli.ENTRYPOINTS:
        path = code / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# frozen entrypoint\n")
    monkeypatch.setattr(cli, "REPO", code)
    monkeypatch.delenv("AF_ACCOUNT", raising=False)
    monkeypatch.delenv("AF_REMAINING_PORTFOLIO_CONCURRENCY", raising=False)
    return root, code, plan


def accepting(monkeypatch):
    """Run real receipt handling against deterministic fake Slurm responses."""
    calls = []

    def accept(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=0, stdout=f"{100 + len(calls)}\n", stderr="")

    monkeypatch.setattr(scheduler.subprocess, "run", accept)
    return calls


def test_submit_all_22_cpu_tasks_once_and_afterany_report(prepared, monkeypatch):
    root, _, _ = prepared
    calls = accepting(monkeypatch)
    result = cli.submit(root)
    assert result["jobs"] == {"run": "101", "report": "102"}
    assert result["tasks"] == 22 and result["maximum_parallel_tasks"] == 2
    assert result["gpus"] == result["llm_calls"] == 0
    assert result["validation_or_test_access"] is False
    assert "--array=0-21%2" in calls[0]
    assert "--time=04:00:00" in calls[0]
    assert "--dependency=afterany:101" in calls[1]
    assert "--time=01:00:00" in calls[1]
    assert calls[0][-2:] == ["run", "0"]
    assert calls[1][-2:] == ["report", "0"]
    for call in calls:
        assert "--account=bibo-delta-cpu" in call
        assert "--cpus-per-task=1" in call and "--mem=8G" in call
        assert "--partition=cpu" in call
        assert call[-3].endswith("run_construction_remaining_portfolio_delta.sh")
    assert cli.submit(root) == result
    assert len(calls) == 2
    assert read_seal(root / cli.SUBMISSIONS / "manifest.json") == result
    assert cli.verify_execution(root)["maximum_reserved_seconds_per_task"] == 9000


def test_submit_authenticates_inputs_once(prepared, monkeypatch):
    root, _, plan = prepared
    accepting(monkeypatch)
    calls = []
    monkeypatch.setattr(
        cli.campaign, "verify", lambda *a, **k: (calls.append(a), (plan, {}))[1]
    )
    cli.submit(root)
    assert calls == [(root,)]


@pytest.mark.parametrize("failure", ["blank", "nonzero", "timeout"])
def test_ambiguous_submission_never_repeats(prepared, monkeypatch, failure):
    root, _, _ = prepared
    calls = []

    def fail(argv, **kwargs):
        calls.append(argv)
        if failure == "timeout":
            raise subprocess.TimeoutExpired(argv, 60)
        return SimpleNamespace(
            returncode=int(failure == "nonzero"), stdout="", stderr=""
        )

    monkeypatch.setattr(scheduler.subprocess, "run", fail)
    with pytest.raises(ValueError, match=r"uncertain|unconfirmed"):
        cli.submit(root)
    with pytest.raises(ValueError, match="uncertain"):
        cli.submit(root)
    assert len(calls) == 1
    assert not (root / cli.SUBMISSIONS / "report.intent.json").exists()


def test_confirmed_array_survives_ambiguous_report(prepared, monkeypatch):
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


@pytest.mark.parametrize("change", ["source", "runtime", "entrypoint", "concurrency"])
def test_execution_rejects_changed_identity(prepared, monkeypatch, change):
    root, code, plan = prepared
    calls = accepting(monkeypatch)
    cli.submit(root)
    if change == "source":
        plan["source_sha256"] = "different"
    elif change == "entrypoint":
        (code / cli.ENTRYPOINTS[2]).write_text("changed\n")
    elif change == "runtime":
        monkeypatch.setattr(cli, "runtime_identity", lambda: {"mode": "different"})
    else:
        monkeypatch.setenv("AF_REMAINING_PORTFOLIO_CONCURRENCY", "1")
    with pytest.raises(ValueError, match="identity differs"):
        cli.verify_execution(root)
    with pytest.raises(ValueError):
        cli.submit(root)
    assert len(calls) == 2


@pytest.mark.parametrize("value", ["0", "3", "-1", "two"])
def test_invalid_concurrency_never_submits(prepared, monkeypatch, value):
    root, _, _ = prepared
    calls = accepting(monkeypatch)
    monkeypatch.setenv("AF_REMAINING_PORTFOLIO_CONCURRENCY", value)
    with pytest.raises(ValueError, match="CONCURRENCY"):
        cli.submit(root)
    assert calls == []


@pytest.mark.parametrize("seconds", [0, -1, 13201, float("nan"), float("inf")])
def test_reservation_overflow_never_submits(prepared, monkeypatch, seconds):
    root, _, _ = prepared
    calls = accepting(monkeypatch)
    monkeypatch.setattr(cli.campaign, "reservation_ceiling", lambda plan: seconds)
    with pytest.raises(ValueError, match="four-hour worker envelope"):
        cli.submit(root)
    assert calls == []


def test_partial_roster_never_submits(prepared, monkeypatch):
    root, _, plan = prepared
    calls = accepting(monkeypatch)
    plan["tasks"].pop()
    with pytest.raises(ValueError, match="all 22"):
        cli.submit(root)
    assert calls == []


def test_saved_manifest_does_not_bypass_scheduler_receipts(prepared, monkeypatch):
    root, _, _ = prepared
    calls = accepting(monkeypatch)
    cli.submit(root)
    (root / cli.SUBMISSIONS / "run.reply.json").write_text(
        json.dumps({"returncode": 0, "stdout": "999\n"})
    )
    with pytest.raises(ValueError, match="unconfirmed run"):
        cli.submit(root)
    assert len(calls) == 2


def test_status_reads_receipts_without_fitting_or_large_inputs(prepared, monkeypatch):
    root, _, plan = prepared
    task = plan["tasks"][0]
    terminal = root / "results" / task["task_id"] / "result.json"
    seal(
        terminal,
        {
            "identity": {"plan_sha256": public.content_sha256(plan), "task": task},
            "status": "complete",
            "search_status": "verified_improvement",
            "retained_origin": "fit_00_shared_verification",
            "source_anchor_nmse": 2.0,
            "training_reconstruction_nmse": 1.5,
            "per_target_normalized_mse": {"Gp": 1.0, "I": 2.0},
            "feedback": {"passed": True, "large_detail": [1, 2, 3]},
            "optimization_converged": False,
        },
    )
    monkeypatch.setattr(cli.campaign, "verify", lambda *a: pytest.fail("input load"))
    monkeypatch.setattr(cli.campaign, "report", lambda *a: pytest.fail("report"))
    result = cli.status(root)
    assert result["recorded"] == 1 and result["expected"] == 22
    assert result["tasks"][0]["search_status"] == "verified_improvement"
    assert result["tasks"][0]["training_reconstruction_nmse"] == 1.5
    assert result["tasks"][0]["per_target_normalized_mse"] == {"Gp": 1.0, "I": 2.0}
    assert result["tasks"][0]["feedback_passed"] is True
    assert result["tasks"][0]["optimization_converged"] is False
    assert not {"kind", "arm", "seed", "mode", "search_termination", "feedback"} & (
        result["tasks"][0].keys()
    )
    assert result["tasks"][1]["terminal_receipt_present"] is False
    changed = read_seal(terminal)
    changed["identity"]["plan_sha256"] = "wrong"
    public._write(
        terminal, {"value": changed, "sha256": public.content_sha256(changed)}
    )
    with pytest.raises(ValueError, match="status receipt differs"):
        cli.status(root)


def test_status_rejects_foreign_path(tmp_path):
    seal(tmp_path / "plan.json", {"tasks": [{"task_id": "../foreign"}]})
    with pytest.raises(ValueError, match="unsafe"):
        cli.status(tmp_path)


def test_run_checks_identity_before_fitting(tmp_path, monkeypatch):
    def rejected(root):
        raise ValueError("launcher identity differs")

    monkeypatch.setattr(cli, "verify_execution", rejected)
    monkeypatch.setattr(cli.campaign, "run_task", lambda *a: pytest.fail("fit"))
    with pytest.raises(ValueError, match="identity differs"):
        cli.main(["run", "--root", str(tmp_path), "--index", "0"])


def test_run_prints_retained_metrics_and_feedback(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "verify_execution", lambda *a: {})
    row = {
        "status": "complete",
        "search_status": "bounded_search_completed",
        "candidate_id": "candidate-00",
        "route": "affine",
        "training_reconstruction_nmse": 1.5,
        "per_target_normalized_mse": {"Gp": 1.5},
        "feedback": {"passed": False},
        "retained": {"parameters": {"k": 123}},
    }
    monkeypatch.setattr(cli.campaign, "run_task", lambda *a: row)
    cli.main(["run", "--root", str(tmp_path), "--index", "0"])
    result = json.loads(capsys.readouterr().out)
    assert result["candidate_id"] == "candidate-00"
    assert result["training_reconstruction_nmse"] == 1.5
    assert result["feedback_passed"] is False
    assert "retained" not in result


def test_inspect_returns_download_path(tmp_path, monkeypatch, capsys):
    archive = tmp_path / "inspection.tar.gz"
    monkeypatch.setattr(cli.campaign, "inspection", lambda root: archive)
    cli.main(["inspect", "--root", str(tmp_path)])
    assert json.loads(capsys.readouterr().out) == {"archive": str(archive)}


def test_prepare_accepts_compact_source_without_submitting(
    tmp_path, monkeypatch, capsys
):
    source, config = tmp_path / "intake.json", tmp_path / "policy.json"
    source.write_text("{}")
    config.write_text("{}")
    calls = []
    policy = object()
    monkeypatch.setattr(
        cli.campaign,
        "Policy",
        SimpleNamespace(model_validate_json=lambda text: policy),
    )
    monkeypatch.setattr(
        cli.campaign,
        "prepare",
        lambda *args: (calls.append(args), {"tasks": 22})[1],
    )
    monkeypatch.setattr(cli, "submit", lambda *args: pytest.fail("submission"))
    cli.main(
        [
            "prepare",
            "--root",
            str(tmp_path / "new-root"),
            "--source",
            str(source),
            "--config",
            str(config),
        ]
    )
    assert calls == [(tmp_path / "new-root", source, policy)]
    result = capsys.readouterr()
    assert "Authenticating historical source receipts" in result.err
    assert json.loads(result.out) == {"tasks": 22}


def test_portable_verify_does_not_require_original_runtime(
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
    assert json.loads(capsys.readouterr().out) == {"verified": True, "tasks": 22}


def shell_environment(tmp_path):
    """Use an argument recorder as Python to test wrappers without Slurm."""
    python = tmp_path / "python"
    python.write_text('#!/bin/bash\nprintf "%s\\n" "$@" >> "$AF_ARGUMENT_LOG"\n')
    python.chmod(0o755)
    source = tmp_path / "source"
    source.mkdir()
    (source / "inputs.json").write_text("{}")
    env = {k: v for k, v in os.environ.items() if not k.startswith("AF_")}
    return env | {
        "AF_REMAINING_PORTFOLIO_ROOT": str(tmp_path / "campaign"),
        "AF_REMAINING_PORTFOLIO_SOURCE": str(source),
        "AF_PYTHON": str(python),
        "AF_ARGUMENT_LOG": str(tmp_path / "args.txt"),
        "AF_OUTPUT_ROOT": str(tmp_path / "unrelated"),
    }


def test_login_run_prepares_then_submits_without_numerical_work(tmp_path):
    env = shell_environment(tmp_path)
    subprocess.run(["bash", "-n", str(START)], check=True)
    subprocess.run(["bash", "-n", str(WORKER)], check=True)
    subprocess.run(["bash", str(START), "run"], env=env, check=True)
    args = (tmp_path / "args.txt").read_text().splitlines()
    assert args.index("prepare") < args.index("submit")
    assert env["AF_REMAINING_PORTFOLIO_SOURCE"] in args
    assert env["AF_REMAINING_PORTFOLIO_ROOT"] in args
    assert env["AF_OUTPUT_ROOT"] not in args
    assert not {"pytest", "--index"} & set(args)


@pytest.mark.parametrize("stage", ["status", "report", "inspect"])
def test_readonly_shell_never_prepares_or_submits(tmp_path, stage):
    env = shell_environment(tmp_path) | {"AF_REPO_ROOT": "/stale/checkout"}
    subprocess.run(["bash", str(START), stage], env=env, check=True)
    args = (tmp_path / "args.txt").read_text().splitlines()
    assert args[0] == str(REPO / "scripts/run_construction_remaining_portfolio.py")
    assert args[1] == stage
    assert not {"prepare", "submit", "run", "--index"} & set(args)


@pytest.mark.parametrize("stage", ["run", "report"])
def test_slurm_spool_copy_preserves_exported_repo(tmp_path, stage):
    env = shell_environment(tmp_path) | {
        "AF_REPO_ROOT": str(REPO),
        "SLURM_ARRAY_TASK_ID": "21",
    }
    copied = tmp_path / "spool/slurm_script"
    copied.parent.mkdir()
    copied.write_bytes(WORKER.read_bytes())
    subprocess.run(["bash", str(copied), stage, "0"], env=env, check=True)
    args = (tmp_path / "args.txt").read_text().splitlines()
    assert args[0] == str(REPO / "scripts/run_construction_remaining_portfolio.py")
    assert args[1] == stage
    if stage == "run":
        assert args[args.index("--index") + 1] == "21"
    assert "prepare" not in args and "submit" not in args


@pytest.mark.parametrize("bad", ["missing_repo", "invalid_index"])
def test_worker_rejects_unbound_execution(tmp_path, bad):
    env = shell_environment(tmp_path) | {"AF_REPO_ROOT": str(REPO)}
    if bad == "missing_repo":
        del env["AF_REPO_ROOT"]
    else:
        env["SLURM_ARRAY_TASK_ID"] = "wrong"
    result = subprocess.run(
        ["bash", str(WORKER), "run", "0"], env=env, capture_output=True, text=True
    )
    assert result.returncode != 0
    assert not (tmp_path / "args.txt").exists()


def test_inspect_from_foreign_directory_uses_recorded_repo(tmp_path):
    env = shell_environment(tmp_path)
    root = Path(env["AF_REMAINING_PORTFOLIO_ROOT"])
    seal(root / "launcher_execution.json", {"repo": str(REPO)})
    foreign = tmp_path / "new-empty-folder"
    foreign.mkdir()
    subprocess.run(
        [
            "bash",
            "-c",
            """
set -euo pipefail
AF_REPO=$(jq -er '.value.repo' "$AF_REMAINING_PORTFOLIO_ROOT/launcher_execution.json")
bash "$AF_REPO/scripts/hpc/start_construction_remaining_portfolio_delta.sh" inspect
""",
        ],
        cwd=foreign,
        env=env,
        check=True,
    )
    args = (tmp_path / "args.txt").read_text().splitlines()
    assert args[1] == "inspect"
    assert args[args.index("--root") + 1] == str(root)


def test_missing_compact_source_prevents_submission(tmp_path):
    env = shell_environment(tmp_path)
    (Path(env["AF_REMAINING_PORTFOLIO_SOURCE"]) / "inputs.json").unlink()
    result = subprocess.run(
        ["bash", str(START), "run"], env=env, capture_output=True, text=True
    )
    assert result.returncode != 0
    assert "authenticated compact intake" in result.stderr
    assert not (tmp_path / "args.txt").exists()
