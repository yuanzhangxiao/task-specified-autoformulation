"""Run the D3 batch script locally against stubs.

Six submissions were lost to shell and orchestration errors that `bash -n`
cannot see: a mktemp against a missing parent, a subcommand read from a
variable name, a loop variable shadowing a readonly, an exit status that
ignored what the task recorded. Each cost a queue slot and a model load to
discover.

This executes the real script with the cluster replaced: fake nvidia-smi,
apptainer and server, and a stub worker in a temporary checkout. It exercises
the orchestration only -- argument handling, gating, the task loop, exit
codes -- never the science.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

BATCH = Path("scripts/hpc/phase_b_d3_vllm_batch.slurm").resolve()

WORKER = '''#!/usr/bin/env python3
"""Stand-in for scripts/phase_b_d3.py that records what it was asked."""
import json, os, sys
from pathlib import Path

root = Path(sys.argv[sys.argv.index("--root") + 1])
command = sys.argv[1]
(root / "invocations.log").open("a").write(" ".join(sys.argv[1:]) + "\\n")

if command == "prepare":
    assert "--provider" in sys.argv, "the provider must be declared"
    (root / "plan.json").write_text(json.dumps({"rows": [{}], "expected": 1}))
elif command == "run":
    index = sys.argv[sys.argv.index("--index") + 1]
    assert os.environ.get("AF_VLLM_BASE_URL", "").startswith("http"), \\
        "the worker needs a dialable endpoint"
    # Per-index outcomes, so a "partial failure" test can actually mix them.
    failing = set(filter(None, os.environ.get("AF_STUB_FAIL_INDICES", "").split(",")))
    outcome = (
        os.environ.get("AF_STUB_FAIL_OUTCOME", "discovery_failed")
        if index in failing
        else os.environ.get("AF_STUB_OUTCOME", "complete")
    )
    if outcome == "crash":
        sys.exit(3)
    directory = root / "results" / index
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "result.json").write_text(json.dumps({"status": outcome}))
'''


def _fake(directory: Path, name: str, body: str) -> None:
    path = directory / name
    path.write_text("#!/bin/bash\n" + body, encoding="utf-8")
    path.chmod(0o755)


@pytest.fixture
def cluster(tmp_path: Path) -> dict:
    """A checkout and a PATH where the cluster is replaced by stubs."""
    repo = tmp_path / "repo"
    (repo / "scripts" / "hpc").mkdir(parents=True)
    shutil.copy(BATCH, repo / "scripts" / "hpc" / BATCH.name)
    worker = repo / "scripts" / "phase_b_d3.py"
    worker.write_text(WORKER, encoding="utf-8")
    worker.chmod(0o755)

    stubs = tmp_path / "bin"
    stubs.mkdir()
    _fake(stubs, "nvidia-smi", 'echo "FAKE GPU, 46068 MiB"\n')
    _fake(stubs, "apptainer", 'echo "apptainer $*"\nsleep 600 &\nwait $!\n')
    # the readiness probe succeeds once the script has started its "server"
    _fake(stubs, "curl", "exit 0\n")
    _fake(stubs, "df", 'echo "Filesystem 1K-blocks Used Available Use%"\n'
                       'echo "fake 100 1 999999999 1%"\n')
    # flock is util-linux and absent on macOS; the cluster has the real one,
    # so the stub only has to preserve "lock then run".
    _fake(stubs, "flock", 'shift\nexec "$@"\n')

    output = tmp_path / "out"
    output.mkdir()
    environment = {
        **os.environ,
        "PATH": f"{stubs}:{os.environ['PATH']}",
        "AF_REPO_ROOT": str(repo),
        "AF_D3_OUTPUT_ROOT": str(output),
        "AF_PYTHON": shutil.which("python3") or "python3",
        "AF_PROJECT": str(tmp_path / "project"),
        "AF_WORK": str(tmp_path / "work"),
        "AF_VLLM_IMAGE": str(tmp_path / "image.sif"),
        "SLURM_ARRAY_TASK_ID": "0",
        "SLURM_CPUS_PER_TASK": "8",
        "SLURM_JOB_ID": "1",
        "AF_TASK_INDICES": "0",
        "AF_BATCH_SIZE": "1",
        "AF_TASK_CONCURRENCY": "1",
    }
    (tmp_path / "image.sif").write_text("", encoding="utf-8")
    return {"env": environment, "repo": repo, "output": output}


def _run(cluster: dict, **overrides: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(cluster["repo"] / "scripts" / "hpc" / BATCH.name)],
        cwd=cluster["repo"],
        env={**cluster["env"], **overrides},
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def test_a_batch_runs_end_to_end_and_reports_success(cluster: dict) -> None:
    result = _run(cluster)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "batch job starting" in result.stdout
    assert "task 0 complete" in result.stdout
    invoked = (cluster["output"] / "invocations.log").read_text(encoding="utf-8")
    # the plan is frozen once, then the task runs against it
    assert invoked.splitlines()[0].startswith("prepare")
    assert any(line.startswith("run") for line in invoked.splitlines())
    assert (cluster["output"] / "plan.json").is_file()


def test_a_recorded_failure_is_not_reported_as_success(cluster: dict) -> None:
    """The worker exits zero while recording discovery_failed."""
    result = _run(cluster, AF_STUB_OUTCOME="discovery_failed")
    assert result.returncode == 1
    assert "recorded discovery_failed" in result.stderr
    assert "every task in batch" in result.stderr


def test_a_crashing_worker_is_counted_as_a_failure(cluster: dict) -> None:
    result = _run(cluster, AF_STUB_OUTCOME="crash")
    assert result.returncode == 1
    assert "task 0 failed" in result.stderr


def test_an_existing_plan_is_reused_rather_than_refrozen(cluster: dict) -> None:
    assert _run(cluster).returncode == 0
    second = _run(cluster)
    assert second.returncode == 0
    assert "reusing the sealed plan" in second.stdout
    invoked = (cluster["output"] / "invocations.log").read_text(encoding="utf-8")
    assert sum(line.startswith("prepare") for line in invoked.splitlines()) == 1


def test_a_partially_failing_batch_still_succeeds(cluster: dict) -> None:
    """One bad task must not abandon the rest of its batch.

    This previously gave both tasks the same successful outcome, so it proved
    only that two successes succeed. Index 1 now genuinely fails.
    """
    result = _run(
        cluster, AF_TASK_INDICES="0,1", AF_BATCH_SIZE="2", AF_STUB_FAIL_INDICES="1"
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "task 0 complete" in result.stdout
    assert "task 1 recorded discovery_failed" in result.stderr
    assert "completed with 1 failed task(s) of 2" in result.stdout


def test_a_missing_required_variable_fails_before_any_gpu_work(
    cluster: dict,
) -> None:
    environment = dict(cluster["env"])
    del environment["AF_D3_OUTPUT_ROOT"]
    result = subprocess.run(
        ["bash", str(cluster["repo"] / "scripts" / "hpc" / BATCH.name)],
        cwd=cluster["repo"], env=environment, capture_output=True, text=True,
        timeout=60, check=False,
    )
    assert result.returncode != 0
    assert "AF_D3_OUTPUT_ROOT" in result.stderr
    assert "apptainer" not in result.stdout
