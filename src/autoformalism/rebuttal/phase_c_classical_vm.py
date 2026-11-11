"""Run the frozen Phase C classical baselines on a Jetstream2 CPU VM.

On Delta each task is a Slurm array element, and
scripts/hpc/phase_c_public_baseline_delta_cpu.slurm reads it from the frozen
ledger and starts scripts/run_baseline.py. A VM has no scheduler, so this
module does that worker's part: the same arguments, the same thread counts
(the task's declared CPUs), the same PySR runtime record. The launcher,
scripts/jetstream/run_phase_c_classical.sh, pins each task to its declared CPUs
and runs as many at once as the VM holds.

A task whose run recorded an outcome (complete, failed or timed_out) is never
started again: running it twice would give it a second try at its budget. A
task that was interrupted recorded none and starts again from the beginning;
SINDy and PySR keep no checkpoints.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

from autoformalism.rebuttal.baseline_pilot import BaselinePilotTask
from autoformalism.rebuttal.phase_c_baseline_plan import (
    CLASSICAL_METHODS,
    freeze_phase_c_baseline_plan,
    load_phase_c_baseline_plan,
)

PLATFORM = "jetstream2_cpu"
OUTCOMES = frozenset({"complete", "failed", "timed_out"})
REPO = Path(__file__).resolve().parents[3]
#: The variables the Delta worker sets to the task's CPU count.
THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verified(path: Path) -> Path:
    """Return a file after checking it against the digest written beside it."""
    record = path.with_name(f"{path.name}.sha256")
    if not path.is_file() or not record.is_file():
        raise ValueError(f"{path} or its recorded digest is missing")
    if _sha256(path) != record.read_text(encoding="utf-8").split()[0]:
        raise ValueError(f"{path} differs from its recorded digest")
    return path


def prepare(config: Path, release: Path, root: Path) -> dict[str, object]:
    """Freeze a Jetstream2 plan into ROOT/frozen; report what the launcher needs.

    Freezing again with the same plan and release changes nothing, so a
    relaunch passes through here; a different plan or release is refused.
    """
    plan = load_phase_c_baseline_plan(config)
    if plan.platform != PLATFORM:
        raise ValueError(f"this plan runs on {plan.platform}, not a Jetstream2 VM")
    manifest = freeze_phase_c_baseline_plan(config, root / "frozen", release)
    return {
        "expected": manifest["task_count"],
        "tasks_by_method": manifest["tasks_by_method"],
        "cpus_per_task": max(method.cpus_per_task for method in plan.methods),
        "release_summary_sha256": manifest["release_summary_sha256"],
        "plan_sha256": manifest["plan_sha256"],
    }


def load_task(root: Path, index: int) -> BaselinePilotTask:
    """Read one task from the frozen ledger, which must be unchanged."""
    ledger = _verified(root / "frozen" / "task_plan.jsonl")
    lines = ledger.read_text(encoding="utf-8").splitlines()
    if not 0 <= index < len(lines):
        raise ValueError(f"the frozen plan has no task {index}")
    task = BaselinePilotTask.model_validate_json(lines[index])
    if task.task_index != index:
        raise ValueError(f"line {index} of the ledger is task {task.task_index}")
    if task.platform != PLATFORM or task.method not in CLASSICAL_METHODS:
        raise ValueError(f"task {index} is not a classical Jetstream2 CPU task")
    return task


def run_directory(root: Path, task: BaselinePilotTask) -> Path:
    """Where scripts/run_baseline.py writes this task's run."""
    name = f"{task.benchmark_id}_{task.tier}_seed{task.repetition}"
    return root / "runs" / task.method / name


def outcome(directory: Path) -> str | None:
    """The outcome a run recorded, or None if it never recorded one."""
    path = directory / "run_status.json"
    if not path.is_file():
        return None
    status = json.loads(path.read_text(encoding="utf-8")).get("status")
    return status if status in OUTCOMES else None


def baseline_arguments(task: BaselinePilotTask, release: Path, root: Path) -> list[str]:
    """scripts/run_baseline.py's arguments, as the Delta worker builds them."""
    arguments = [
        "--phase-c-release",
        str(release),
        "--benchmark-id",
        task.benchmark_id,
        "--method",
        task.method,
        "--seed",
        str(task.repetition),
        "--output-root",
        str(root / "runs"),
        "--wall-timeout-seconds",
        repr(task.wall_timeout_seconds),
        "--development-only",
    ]
    if task.method == "sindy":
        if not task.sindy_thresholds:
            raise ValueError("SINDy task has no frozen threshold grid")
        arguments += ["--sindy-thresholds", *map(repr, task.sindy_thresholds)]
    elif task.method == "pysr":
        arguments += [
            "--pysr-iterations",
            str(task.pysr_iterations),
            "--maximum-expression-size",
            str(task.maximum_expression_size),
        ]
    return arguments


def task_environment(
    task: BaselinePilotTask, *, julia_depot: Path, base: Mapping[str, str]
) -> dict[str, str]:
    """The Delta worker's environment: thread counts are the task's CPUs."""
    environment = dict(base)
    environment.update(dict.fromkeys(THREAD_VARIABLES, str(task.cpus_per_task)))
    environment.update(
        PYTHONHASHSEED="0",
        PYTHONPATH=str(REPO / "src"),
        JULIA_DEPOT_PATH=str(julia_depot),
    )
    return environment


def run(
    root: Path,
    index: int,
    *,
    release: Path,
    julia_depot: Path,
    base: Mapping[str, str],
) -> int:
    """Run one task unless it already has an outcome; return its exit status."""
    root, release = root.resolve(), release.resolve()
    task = load_task(root, index)
    plan = load_phase_c_baseline_plan(_verified(root / "frozen" / "plan.json"))
    if _sha256(release / "summary.json") != plan.release_summary_sha256:
        raise ValueError("the release receipt differs from the frozen plan")
    ended = outcome(run_directory(root, task))
    if ended is not None:
        print(f"task {index} already ended: {ended}; it is not run again")
        return 0
    if task.method == "pysr":
        _verified(root / "runtime" / "pysr_runtime.json")
    command = [
        sys.executable,
        str(REPO / "scripts" / "run_baseline.py"),
        *baseline_arguments(task, release, root),
    ]
    environment = task_environment(task, julia_depot=julia_depot, base=base)
    return subprocess.run(command, cwd=REPO, env=environment, check=False).returncode
