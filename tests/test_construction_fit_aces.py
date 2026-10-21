"""Bounded CPU scheduling retains uncertain receipts and verifies runtime choice."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from autoformalism.fitting.public_fitting import content_sha256
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from scripts import submit_construction_fit_aces as submission
from scripts import submit_review_continuation as scheduler

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    root = tmp_path / "results"
    plan = sealed_write(
        root / "plan.json",
        {"tasks": [{"task_id": str(index)} for index in range(8)]},
    )
    monkeypatch.setattr(submission, "campaign_plan", lambda _: plan)
    monkeypatch.setattr(submission, "runtime_identity", lambda: {"mode": "native"})
    code = tmp_path / "repo"
    for name in submission.ENTRYPOINTS:
        path = code / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# frozen entrypoint\n")
    monkeypatch.setattr(submission, "REPO", code)
    monkeypatch.setenv("AF_ACCOUNT", "156264627414")
    return root, code


def test_eight_cpu_tasks_and_afterany_report_are_idempotent(prepared, monkeypatch):
    root, _ = prepared
    commands = []

    def accept(argv, **kwargs):
        commands.append(argv)
        return SimpleNamespace(
            returncode=0, stdout=f"{100 + len(commands)}\n", stderr=""
        )

    monkeypatch.setattr(scheduler.subprocess, "run", accept)
    result = submission.submit(root)
    assert result["jobs"] == {"fit": "101", "report": "102"}
    assert result["tasks"] == 8 and result["maximum_parallel_tasks"] == 4
    assert result["gpus"] == result["llm_calls"] == 0
    assert not result["test_data_opened"] and not result["automatic_followup"]
    assert "--array=0-7%4" in commands[0]
    assert "--dependency=afterany:101" in commands[1]
    assert "--time=01:00:00" in commands[0]
    for command in commands:
        assert "--cpus-per-task=1" in command
        assert "--mem=8G" in command
        assert "--partition=cpu" in command
        assert "--exclude=ac042" in command
        assert not any("gpu" in token for token in command)
    assert submission.submit(root) == result
    assert len(commands) == 2
    assert submission.verify_execution(root)["plan_sha256"] == content_sha256(
        sealed_read(root / "plan.json")
    )


@pytest.mark.parametrize("failure", ["blank_success", "nonzero", "timeout"])
def test_unconfirmed_scheduler_reply_is_retained_and_never_retried(
    prepared, monkeypatch, failure
):
    root, _ = prepared
    calls = []

    def fail(argv, **kwargs):
        calls.append(argv)
        if failure == "timeout":
            raise subprocess.TimeoutExpired(argv, 60)
        return SimpleNamespace(
            returncode=0 if failure == "blank_success" else 1,
            stdout="\n",
            stderr="sbatch: error: QOSMaxSubmitJobPerUserLimit\n",
        )

    monkeypatch.setattr(scheduler.subprocess, "run", fail)
    with pytest.raises(ValueError, match=r"unconfirmed|uncertain"):
        submission.submit(root)
    directory = root / "submissions/cpu-fit-1"
    assert (directory / "fit.intent.json").exists()
    assert (directory / "fit.reply.json").exists()
    assert not (directory / "report.intent.json").exists()
    with pytest.raises(ValueError, match="uncertain"):
        submission.submit(root)
    assert len(calls) == 1


def test_confirmed_array_survives_uncertain_report_without_duplicate(
    prepared, monkeypatch
):
    root, _ = prepared
    commands = []

    def reply(argv, **kwargs):
        commands.append(argv)
        return SimpleNamespace(
            returncode=0,
            stdout="103\n" if len(commands) == 1 else "\n",
            stderr="" if len(commands) == 1 else "site wrapper rejected report",
        )

    monkeypatch.setattr(scheduler.subprocess, "run", reply)
    with pytest.raises(ValueError, match="unconfirmed"):
        submission.submit(root)
    with pytest.raises(ValueError, match="uncertain report"):
        submission.submit(root)
    assert len(commands) == 2
    assert (root / "submissions/cpu-fit-1/fit.id").read_text().strip() == "103"


def test_changed_worker_or_runtime_cannot_reuse_execution(prepared, monkeypatch):
    root, code = prepared
    monkeypatch.setattr(
        scheduler.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0, stdout="104\n", stderr=""
        ),
    )
    submission.submit(root)
    worker = code / "scripts/hpc/run_construction_fit_aces.sh"
    worker.write_text("changed\n")
    with pytest.raises(ValueError):
        submission.submit(root)
    with pytest.raises(ValueError, match="identity differs"):
        submission.verify_execution(root)
    worker.write_text("# frozen entrypoint\n")
    monkeypatch.setattr(submission, "runtime_identity", lambda: {"mode": "container"})
    with pytest.raises(ValueError):
        submission.submit(root)


def test_forged_confirmed_receipt_is_rejected(prepared, monkeypatch):
    root, _ = prepared
    directory = root / "submissions/cpu-fit-1"
    directory.mkdir(parents=True)
    (directory / "fit.intent.json").write_text("{}")
    (directory / "fit.id").write_text("101\n")
    (directory / "fit.reply.json").write_text(
        json.dumps({"returncode": 0, "stdout": "999\n"})
    )
    monkeypatch.setattr(
        scheduler.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("no submission"),
    )
    with pytest.raises(ValueError, match="unconfirmed fit"):
        submission.submit(root)


def worker_environment(tmp_path):
    """Record launcher commands without importing a fitter or contacting Slurm."""
    python = tmp_path / "python"
    python.write_text('#!/bin/bash\nprintf "%s\\n" "$@" >> "$AF_ARGUMENT_LOG"\n')
    python.chmod(0o755)
    environment = {
        key: value for key, value in os.environ.items() if not key.startswith("AF_")
    }
    environment.update(
        AF_REPO_ROOT=str(REPO),
        AF_OUTPUT_ROOT=str(tmp_path / "results"),
        AF_GROUP=str(tmp_path),
        AF_PYTHON=str(python),
        AF_ARGUMENT_LOG=str(tmp_path / "arguments.txt"),
    )
    return environment


def test_worker_preflight_and_index_routing_use_cpu_python(tmp_path):
    environment = worker_environment(tmp_path)
    environment["SLURM_ARRAY_TASK_ID"] = "6"
    worker = REPO / "scripts/hpc/run_construction_fit_aces.sh"
    subprocess.run(["bash", "-n", str(worker)], check=True)
    subprocess.run(["bash", str(worker), "run", "0"], env=environment, check=True)
    args = (tmp_path / "arguments.txt").read_text().splitlines()
    assert args[0] == "-c" and "ctypes" in args[1] and "pandas" in args[1]
    assert args[args.index("--index") + 1] == "6"
    assert str(REPO / "scripts/submit_construction_fit_aces.py") in args
    assert str(REPO / "scripts/qualify_construction_fitting.py") in args


def test_native_import_failure_loads_compiler_before_python(tmp_path):
    environment = worker_environment(tmp_path)
    python = Path(environment["AF_PYTHON"])
    python.write_text(
        "#!/bin/bash\n"
        'if [[ "${AF_MODULE_LOADED:-}" != yes ]]; then exit 1; fi\n'
        'printf "%s\\n" "$@" >> "$AF_ARGUMENT_LOG"\n'
    )
    worker = REPO / "scripts/hpc/run_construction_fit_aces.sh"
    script = (
        'module() { printf "%s\\n" "$@" >> "$AF_ARGUMENT_LOG"; '
        'export AF_MODULE_LOADED=yes; }; export -f module; exec bash "$1" verify'
    )
    subprocess.run(
        ["bash", "-c", script, "unused", str(worker)], env=environment, check=True
    )
    args = (tmp_path / "arguments.txt").read_text().splitlines()
    assert args[:3] == ["load", "GCCcore/13.2.0", "Python/3.11.5"]
    assert "verify" in args


def test_wrapper_prepares_bundled_inputs_and_submits_to_scratch(tmp_path):
    environment = worker_environment(tmp_path)
    environment.pop("AF_OUTPUT_ROOT")
    wrapper = REPO / "scripts/hpc/start_construction_fit_aces.sh"
    subprocess.run(["bash", "-n", str(wrapper)], check=True)
    subprocess.run(["bash", str(wrapper), "run"], env=environment, check=True)
    args = (tmp_path / "arguments.txt").read_text().splitlines()
    assert str(tmp_path / "phase-c-construction-fitting-v1") in args
    assert str(REPO / "inputs/source-inspection.tar.gz") in args
    assert str(REPO / "inputs/public-cell") in args
    assert str(REPO / "configs/construction_fit_qualification_v1.json") in args
    assert "submit" in args


def test_explicit_cpu_container_preflight_uses_no_gpu_or_new_server(tmp_path):
    environment = worker_environment(tmp_path)
    runtime = tmp_path / "apptainer"
    runtime.write_text('#!/bin/bash\nprintf "%s\\n" "$@" >> "$AF_ARGUMENT_LOG"\n')
    runtime.chmod(0o755)
    image = tmp_path / "scientific.sif"
    image.write_text("fake scientific image")
    environment.update(
        PATH=f"{tmp_path}:{environment['PATH']}",
        AF_FIT_RUNTIME="container",
        AF_CPU_IMAGE=str(image),
    )
    worker = REPO / "scripts/hpc/run_construction_fit_aces.sh"
    subprocess.run(["bash", str(worker), "verify"], env=environment, check=True)
    args = (tmp_path / "arguments.txt").read_text().splitlines()
    assert args.count("exec") == 2
    assert args.count(str(image)) == 2
    assert "--nv" not in args
    assert "verify" in args


def test_invalid_array_index_stops_before_any_optimization(tmp_path):
    environment = worker_environment(tmp_path)
    environment["SLURM_ARRAY_TASK_ID"] = "8"
    worker = REPO / "scripts/hpc/run_construction_fit_aces.sh"
    result = subprocess.run(
        ["bash", str(worker), "run", "0"],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "Expected task index 0..7" in result.stderr
    args = (tmp_path / "arguments.txt").read_text().splitlines()
    assert "run" not in args


def test_actual_eight_model_campaign_plan_binds_without_artifact_field(
    tmp_path, monkeypatch
):
    from autoformalism.research import construction_fit_qualification as campaign
    from tests.test_construction_fit_qualification import eight_inputs

    root = tmp_path / "qualification"
    campaign.prepare(
        root, eight_inputs(), campaign.Policy(prefix=campaign.PrefixPolicy(duration=1))
    )
    plan = submission.campaign_plan(root)
    assert "artifact_sha256" not in plan
    assert len(plan["tasks"]) == 8
    monkeypatch.setattr(submission, "runtime_identity", lambda: {"mode": "test"})
    commands = []

    def accept(argv, **kwargs):
        commands.append(argv)
        return SimpleNamespace(
            returncode=0, stdout=f"{200 + len(commands)}\n", stderr=""
        )

    monkeypatch.setattr(scheduler.subprocess, "run", accept)
    assert submission.submit(root)["jobs"] == {"fit": "201", "report": "202"}
    assert submission.verify_execution(root)["plan_sha256"] == content_sha256(plan)
    assert len(commands) == 2


def test_status_reads_campaign_report_directory(tmp_path):
    environment = worker_environment(tmp_path)
    queue = tmp_path / "squeue"
    queue.write_text("#!/bin/bash\nprintf 'empty queue\\n'\n")
    queue.chmod(0o755)
    environment["PATH"] = f"{tmp_path}:{environment['PATH']}"
    report = Path(environment["AF_OUTPUT_ROOT"]) / "report/summary.json"
    report.parent.mkdir(parents=True)
    report.write_text('{"completed_tasks": 8}\n')
    wrapper = REPO / "scripts/hpc/start_construction_fit_aces.sh"
    result = subprocess.run(
        ["bash", str(wrapper), "status"],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert '"completed_tasks": 8' in result.stdout
    assert not (tmp_path / "arguments.txt").exists()
