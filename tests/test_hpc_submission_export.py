"""What the submitter hands to sbatch, parsed the way Slurm parses it.

A 60-task tier ran exactly one task per tier and reported COMPLETED, twice,
because --export is itself a comma-separated list: AF_TASK_INDICES=0,1,2,...
arrived as AF_TASK_INDICES=0 with the rest read as variable names. The batch
harness could not see it, because it sets the job environment directly and
never goes through sbatch.

These tests re-implement Slurm's --export splitting and assert the whole task
list survives it.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SUBMITTERS = {
    "d3": ("submit_phase_b_d3_vllm.sh", "configs/phase_b_d3_full_v1.json",
           "AF_D3_OUTPUT_ROOT", "phase_b_d3_vllm_batch.slurm"),
    "llm_ode": ("submit_phase_b_llm_ode_vllm.sh",
                "configs/phase_b_llm_ode_campaign_v1.json",
                "AF_LLM_ODE_OUTPUT_ROOT", "phase_b_llm_ode_vllm_batch.slurm"),
}


def _fake(directory: Path, name: str, body: str) -> None:
    path = directory / name
    path.write_text("#!/bin/bash\n" + body, encoding="utf-8")
    path.chmod(0o755)


def slurm_export_environment(export_argument: str) -> dict[str, str]:
    """Split --export as Slurm does: on commas, each element NAME=value."""
    environment: dict[str, str] = {}
    for element in export_argument.split(","):
        if element == "ALL":
            continue
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", element):
            raise AssertionError(
                f"--export element {element!r} is not NAME=value; a value "
                "containing a comma has been split across elements"
            )
        name, _, value = element.partition("=")
        environment[name] = value
    return environment


@pytest.fixture
def submitted(tmp_path: Path, request) -> dict:
    """Run one submitter with sbatch stubbed, and capture its arguments."""
    script, config, root_variable, batch = SUBMITTERS[request.param]
    repo = tmp_path / "repo"
    (repo / "scripts" / "hpc").mkdir(parents=True)
    for name in (script, batch):
        shutil.copy(REPO / "scripts" / "hpc" / name, repo / "scripts" / "hpc" / name)
    shutil.copy(
        REPO / "scripts" / "list_phase_b_d3_task_indices.py",
        repo / "scripts" / "list_phase_b_d3_task_indices.py",
    )

    # A campaign must run from a pinned checkout, so the fixture is one.
    for command in (
        ("git", "init", "-q"),
        ("git", "config", "user.email", "t@example.com"),
        ("git", "config", "user.name", "t"),
        ("git", "add", "-A"),
        ("git", "commit", "-qm", "pinned"),
    ):
        subprocess.run(command, cwd=repo, check=True, capture_output=True)
    head = subprocess.run(
        ("git", "rev-parse", "HEAD"), cwd=repo, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    subprocess.run(("git", "checkout", "-q", head), cwd=repo, check=True,
                   capture_output=True)

    stubs = tmp_path / "bin"
    stubs.mkdir()
    recorded = tmp_path / "sbatch.args"
    _fake(stubs, "sbatch", f'printf "%s\\n" "$@" > {recorded}\necho 999001\n')
    _fake(stubs, "jq", 'echo "{}"\n')
    upstream = tmp_path / "llm-ode"
    (upstream / "llmode").mkdir(parents=True)
    (upstream / "llmode" / "llmode.py").write_text("", encoding="utf-8")
    # Behaves like the pinned checkout a campaign requires: detached HEAD,
    # nothing uncommitted. A stub that answered everything made the new guard
    # believe the repository was on a branch.
    _fake(
        stubs,
        "git",
        'case "$*" in\n'
        '  *symbolic-ref*) exit 1 ;;\n'
        '  *"diff --quiet"*) exit 0 ;;\n'
        '  *rev-parse*) echo 31667d9ac948e8f411cd2f07726d7744784b5d5a ;;\n'
        'esac\n',
    )

    output = tmp_path / "out"
    output.mkdir()
    environment = {
        **os.environ,
        "PATH": f"{stubs}:{os.environ['PATH']}",
        "AF_REPO_ROOT": str(repo),
        "AF_PYTHON": sys.executable,
        "AF_PROJECT": str(tmp_path / "project"),
        "AF_WORK": str(tmp_path / "work"),
        root_variable: str(output),
        "AF_TIER": "easy",
        "AF_BATCH_SIZE": "10",
        "AF_CLUSTER": "delta",
        "AF_LLM_ODE_ROOT": str(upstream),
        "AF_D3_CONFIG": str(REPO / config),
        "AF_LLM_ODE_CONFIG": str(REPO / config),
    }
    environment.pop("AF_TASK_INDICES", None)
    result = subprocess.run(
        ["bash", str(repo / "scripts" / "hpc" / script)],
        cwd=repo, env=environment, capture_output=True, text=True,
        timeout=120, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    arguments = recorded.read_text(encoding="utf-8").splitlines()
    export = next(
        item.split("=", 1)[1]
        for item in arguments
        if item.startswith("--export=")
    )
    return {
        "env": environment, "repo": repo, "output": output, "batch": batch,
        "export": export, "arguments": arguments, "stdout": result.stdout,
    }


@pytest.mark.parametrize("submitted", ["d3", "llm_ode"], indirect=True)
def test_the_whole_task_list_survives_slurms_export_parsing(submitted: dict) -> None:
    """The easy tier is 60 tasks; the job must receive all 60."""
    passed = slurm_export_environment(submitted["export"])
    index_file = Path(passed["AF_SUBMISSION_DIR"]) / "tasks.txt"
    assert index_file.is_file()
    indices = [line for line in index_file.read_text().splitlines() if line]
    assert len(indices) == 60, f"job would see {len(indices)} of 60 tasks"
    assert indices[0] == "0"


@pytest.mark.parametrize("submitted", ["d3", "llm_ode"], indirect=True)
def test_no_exported_value_contains_a_comma(submitted: dict) -> None:
    """Any comma in a value silently truncates it and shifts the rest."""
    passed = slurm_export_environment(submitted["export"])
    for name, value in passed.items():
        assert "," not in value, f"{name} would be split by --export"


@pytest.mark.parametrize("submitted", ["d3", "llm_ode"], indirect=True)
def test_the_batch_job_reads_every_index_it_was_given(submitted: dict) -> None:
    """End to end: what sbatch was handed is what the batch script resolves."""
    stubs = Path(submitted["env"]["PATH"].split(":")[0])
    _fake(stubs, "nvidia-smi", 'echo "FAKE GPU, 46068 MiB"\n')
    _fake(stubs, "df", 'echo "h"\necho "fake 100 1 999999999 1%"\n')
    # stop before the model is served; the task list is already resolved
    _fake(stubs, "apptainer", 'echo "apptainer $*"\nexit 9\n')

    environment = {
        **submitted["env"],
        **slurm_export_environment(submitted["export"]),
        "SLURM_ARRAY_TASK_ID": "0",
        "SLURM_JOB_ID": "1",
        "SLURM_CPUS_PER_TASK": "8",
        "AF_VLLM_IMAGE": str(submitted["repo"] / "missing.sif"),
    }
    result = subprocess.run(
        ["bash", str(submitted["repo"] / "scripts" / "hpc" / submitted["batch"])],
        cwd=submitted["repo"], env=environment, capture_output=True, text=True,
        timeout=120, check=False,
    )
    assert "task_list_size=60" in result.stdout, result.stdout + result.stderr
    # batch 0 of 10 takes the first ten of the list, not the first one. The
    # easy tier interleaves with hard, so the indices are not contiguous.
    passed = slurm_export_environment(submitted["export"])
    indices = [
        line
        for line in (
            Path(passed["AF_SUBMISSION_DIR"]) / "tasks.txt"
        ).read_text().splitlines()
        if line
    ]
    assert f"tasks={' '.join(indices[:10])}" in result.stdout, result.stdout


@pytest.mark.parametrize("submitted", ["d3", "llm_ode"], indirect=True)
def test_every_array_element_covers_its_slice_including_the_short_last_one(
    submitted: dict,
) -> None:
    """Element zero passing says nothing about the final, partial batch.

    60 tasks at 10 per batch is six elements; the bug that lost the campaign
    showed up as a later element resolving to nothing at all.
    """
    stubs = Path(submitted["env"]["PATH"].split(":")[0])
    _fake(stubs, "nvidia-smi", 'echo "FAKE GPU, 46068 MiB"\n')
    _fake(stubs, "df", 'echo "h"\necho "fake 100 1 999999999 1%"\n')
    _fake(stubs, "apptainer", 'echo "apptainer $*"\nexit 9\n')
    passed = slurm_export_environment(submitted["export"])
    indices = [
        line
        for line in (Path(passed["AF_SUBMISSION_DIR"]) / "tasks.txt")
        .read_text().splitlines()
        if line
    ]
    assert len(indices) == 60

    seen: list[str] = []
    for element in range(6):
        result = subprocess.run(
            ["bash", str(submitted["repo"] / "scripts" / "hpc" / submitted["batch"])],
            cwd=submitted["repo"],
            env={
                **submitted["env"], **passed,
                "SLURM_ARRAY_TASK_ID": str(element),
                "SLURM_JOB_ID": "1", "SLURM_CPUS_PER_TASK": "8",
                "AF_VLLM_IMAGE": str(submitted["repo"] / "missing.sif"),
            },
            capture_output=True, text=True, timeout=120, check=False,
        )
        line = next(
            item for item in result.stdout.splitlines()
            if item.startswith(f"batch={element} tasks=")
        )
        assert "is empty" not in result.stdout, f"element {element}: {result.stdout}"
        seen.extend(line.split("tasks=", 1)[1].split())
    # every submitted task is covered exactly once across the array
    assert seen == indices


@pytest.mark.parametrize("submitted", ["d3"], indirect=True)
def test_a_second_submission_cannot_rewrite_a_queued_ones_task_list(
    submitted: dict,
) -> None:
    """A retry for the same tier must not change what a queued array will read.

    The first fix wrote one file per tier in the campaign root, so a smoke run
    submitted afterwards would silently redefine the membership of an array
    still sitting in the queue.
    """
    first = Path(slurm_export_environment(submitted["export"])["AF_SUBMISSION_DIR"])
    before = (first / "tasks.txt").read_text()

    second = subprocess.run(
        ["bash", str(submitted["repo"] / "scripts" / "hpc"
                     / SUBMITTERS["d3"][0])],
        cwd=submitted["repo"],
        env={**submitted["env"], "AF_TASK_INDICES": "7"},
        capture_output=True, text=True, timeout=120, check=False,
    )
    assert second.returncode == 0, second.stdout + second.stderr
    assert (first / "tasks.txt").read_text() == before
    # and the two submissions are separately recorded
    roots = sorted((first.parent).iterdir())
    assert len(roots) == 2, [item.name for item in roots]


@pytest.mark.parametrize("submitted", ["d3"], indirect=True)
def test_a_tampered_task_list_is_refused_before_the_model_loads(
    submitted: dict,
) -> None:
    """The membership acted on must be the membership submitted."""
    stubs = Path(submitted["env"]["PATH"].split(":")[0])
    _fake(stubs, "nvidia-smi", 'echo "FAKE GPU"\n')
    _fake(stubs, "apptainer", 'echo "apptainer $*"\nexit 9\n')
    passed = slurm_export_environment(submitted["export"])
    tasks = Path(passed["AF_SUBMISSION_DIR"]) / "tasks.txt"
    tasks.write_text(tasks.read_text().replace("0\n", "99\n", 1))

    result = subprocess.run(
        ["bash", str(submitted["repo"] / "scripts" / "hpc" / submitted["batch"])],
        cwd=submitted["repo"],
        env={
            **submitted["env"], **passed,
            "SLURM_ARRAY_TASK_ID": "0", "SLURM_JOB_ID": "1",
            "SLURM_CPUS_PER_TASK": "8",
            "AF_VLLM_IMAGE": str(submitted["repo"] / "missing.sif"),
        },
        capture_output=True, text=True, timeout=120, check=False,
    )
    assert result.returncode != 0
    assert "does not match the manifest" in result.stderr
    assert "apptainer" not in result.stdout


@pytest.mark.parametrize(
    "script",
    ["submit_phase_b_llm_ode_vllm.sh", "submit_phase_b_llm_sr_vllm.sh"],
)
def test_a_campaign_refuses_a_checkout_that_is_still_moving(
    tmp_path: Path, script: str
) -> None:
    """A plan's identity covers every .py in the package.

    Seventeen of eighteen LLM-ODE batches died on the stale-plan guard because
    the campaign ran from the branch being edited, and commits landed while it
    was queued. A campaign must run from a pinned checkout.
    """
    repo = tmp_path / "repo"
    (repo / "scripts" / "hpc").mkdir(parents=True)
    shutil.copy(REPO / "scripts" / "hpc" / script, repo / "scripts" / "hpc" / script)
    run = lambda *a: subprocess.run(a, cwd=repo, check=True, capture_output=True)  # noqa: E731
    run("git", "init", "-q")
    run("git", "config", "user.email", "t@example.com")
    run("git", "config", "user.name", "t")
    run("git", "add", "-A")
    run("git", "commit", "-qm", "pinned")

    stubs = tmp_path / "bin"
    stubs.mkdir()
    _fake(stubs, "sbatch", "echo 1\n")

    environment = {
        **os.environ,
        "PATH": f"{stubs}:{os.environ['PATH']}",
        "AF_REPO_ROOT": str(repo),
        "AF_PROJECT": str(tmp_path / "p"),
        "AF_WORK": str(tmp_path / "w"),
    }
    result = subprocess.run(
        ["bash", str(repo / "scripts" / "hpc" / script)],
        cwd=repo, env=environment, capture_output=True, text=True,
        timeout=60, check=False,
    )
    # on a branch, so refused before anything is submitted
    assert result.returncode == 2, result.stdout + result.stderr
    assert "not a pinned checkout" in result.stderr
    assert "worktree" in result.stderr
