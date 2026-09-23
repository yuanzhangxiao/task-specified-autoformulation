"""Run the LLM-SR batch script locally against stubs.

The same orchestration harness as the D3 batch, plus the one thing this
campaign adds: the claim that the search came from a specific upstream
revision. A checkout at the wrong commit has to fail before the queue slot and
the model load are spent, not after.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

BATCH = Path("scripts/hpc/phase_b_llm_sr_vllm_batch.slurm").resolve()

WORKER = '''#!/usr/bin/env python3
"""Stand-in for scripts/phase_b_llm_sr.py that records what it was asked."""
import json, os, sys
from pathlib import Path

root = Path(sys.argv[sys.argv.index("--root") + 1])
command = sys.argv[1]
(root / "invocations.log").open("a").write(" ".join(sys.argv[1:]) + "\\n")

if command == "prepare":
    (root / "plan.json").write_text(json.dumps({"rows": [{}], "expected": 1}))
elif command == "run":
    index = sys.argv[sys.argv.index("--index") + 1]
    assert os.environ.get("AF_VLLM_BASE_URL", "").startswith("http"), \\
        "the worker needs a dialable endpoint"
    assert os.environ.get("AF_LLM_SR_ROOT"), \\
        "the worker needs the pinned checkout"
    outcome = os.environ.get("AF_STUB_OUTCOME", "complete")
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


def _checkout(path: Path) -> str:
    """A real git repo shaped like the upstream checkout; returns its HEAD."""
    (path / "llmsr").mkdir(parents=True)
    (path / "llmsr" / "pipeline.py").write_text("", encoding="utf-8")
    run = lambda *args: subprocess.run(  # noqa: E731
        args, cwd=path, check=True, capture_output=True
    )
    run("git", "init", "-q")
    run("git", "config", "user.email", "t@example.com")
    run("git", "config", "user.name", "t")
    run("git", "add", "-A")
    run("git", "commit", "-qm", "pinned")
    return subprocess.run(
        ("git", "rev-parse", "HEAD"), cwd=path, check=True,
        capture_output=True, text=True,
    ).stdout.strip()


@pytest.fixture
def cluster(tmp_path: Path) -> dict:
    """A checkout and a PATH where the cluster is replaced by stubs."""
    repo = tmp_path / "repo"
    (repo / "scripts" / "hpc").mkdir(parents=True)
    shutil.copy(BATCH, repo / "scripts" / "hpc" / BATCH.name)
    worker = repo / "scripts" / "phase_b_llm_sr.py"
    worker.write_text(WORKER, encoding="utf-8")
    worker.chmod(0o755)

    upstream = tmp_path / "llm-ode"
    commit = _checkout(upstream)
    config = tmp_path / "campaign.json"
    config.write_text(
        json.dumps({"upstream": {"commit": commit}}), encoding="utf-8"
    )

    stubs = tmp_path / "bin"
    stubs.mkdir()
    _fake(stubs, "nvidia-smi", 'echo "FAKE GPU, 81920 MiB"\n')
    _fake(stubs, "apptainer", 'echo "apptainer $*"\nsleep 600 &\nwait $!\n')
    _fake(stubs, "curl", "exit 0\n")
    _fake(stubs, "df", 'echo "Filesystem 1K-blocks Used Available Use%"\n'
                       'echo "fake 100 1 999999999 1%"\n')
    _fake(stubs, "flock", 'shift\nexec "$@"\n')

    output = tmp_path / "out"
    output.mkdir()
    environment = {
        **os.environ,
        "PATH": f"{stubs}:{os.environ['PATH']}",
        "AF_REPO_ROOT": str(repo),
        "AF_LLM_SR_OUTPUT_ROOT": str(output),
        "AF_LLM_SR_ROOT": str(upstream),
        "AF_LLM_SR_CONFIG": str(config),
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
    return {"env": environment, "repo": repo, "output": output, "commit": commit}


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
    assert f"commit={cluster['commit']}" in result.stdout
    assert "task 0 complete" in result.stdout
    invoked = (cluster["output"] / "invocations.log").read_text(encoding="utf-8")
    assert invoked.splitlines()[0].startswith("prepare")
    assert any(line.startswith("run") for line in invoked.splitlines())


def test_a_checkout_at_the_wrong_commit_fails_before_the_model_loads(
    cluster: dict, tmp_path: Path
) -> None:
    """Faithfulness is a claim about a revision, so it is checked, not assumed."""
    config = tmp_path / "other.json"
    config.write_text(json.dumps({"upstream": {"commit": "d" * 40}}), encoding="utf-8")
    result = _run(cluster, AF_LLM_SR_CONFIG=str(config))
    assert result.returncode == 1
    assert "not the pinned" in result.stderr
    # nothing was served and no task ran
    assert "apptainer" not in result.stdout
    assert not (cluster["output"] / "invocations.log").exists()


def test_a_recorded_failure_is_not_reported_as_success(cluster: dict) -> None:
    result = _run(cluster, AF_STUB_OUTCOME="inexpressible")
    assert result.returncode == 1
    assert "recorded inexpressible" in result.stderr


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


def test_a_missing_checkout_fails_before_any_gpu_work(cluster: dict) -> None:
    environment = dict(cluster["env"])
    del environment["AF_LLM_SR_ROOT"]
    result = subprocess.run(
        ["bash", str(cluster["repo"] / "scripts" / "hpc" / BATCH.name)],
        cwd=cluster["repo"], env=environment, capture_output=True, text=True,
        timeout=60, check=False,
    )
    assert result.returncode != 0
    assert "AF_LLM_SR_ROOT" in result.stderr
    assert "apptainer" not in result.stdout


def test_requested_modules_are_loaded_before_the_work(cluster: dict) -> None:
    """A venv from a module-provided Python needs that module in the job."""
    stubs = Path(cluster["env"]["PATH"].split(":")[0])
    loaded = stubs.parent / "module.log"
    _fake(stubs, "module", f'echo "$@" >> {loaded}\n')
    result = _run(cluster, AF_MODULES="python/3.13.5-gcc13.3.1")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "modules=python/3.13.5-gcc13.3.1" in result.stdout
    assert "load python/3.13.5-gcc13.3.1" in loaded.read_text()


def test_an_unavailable_module_command_fails_before_the_model_loads(
    cluster: dict,
) -> None:
    """Silently skipping the load would fail later and far less clearly."""
    # a real PATH, but without the stub and with no lmod init present
    result = _run(cluster, AF_MODULES="python/3.13.5-gcc13.3.1", PATH="/usr/bin:/bin")
    assert result.returncode != 0
    assert "'module' is unavailable" in result.stderr


def test_no_modules_requested_changes_nothing(cluster: dict) -> None:
    result = _run(cluster)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "modules=" not in result.stdout
