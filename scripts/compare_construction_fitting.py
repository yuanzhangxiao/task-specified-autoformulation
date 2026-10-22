#!/usr/bin/env python3
"""Launch a bounded CPU comparison of frozen parents and revised children."""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
from types import ModuleType

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import public_fitting as public
from scripts.submit_construction_fit_aces import confirmed_job, file_hash
from scripts.submit_construction_fit_aces import runtime_identity as _runtime_identity
from scripts.submit_review_continuation import submit_job

REPO = Path(__file__).resolve().parents[1]
ENTRYPOINTS = (
    "scripts/compare_construction_fitting.py",
    "scripts/hpc/start_construction_paired_fit.sh",
    "scripts/hpc/run_construction_paired_fit.sh",
    "scripts/submit_construction_fit_aces.py",
    "scripts/submit_review_continuation.py",
)
SUBMISSIONS = "submissions/cpu-paired-fit-1"
TASKS = 24


def _campaign() -> ModuleType:
    """Load the opt-in campaign without coupling other launchers to it."""
    return importlib.import_module("autoformalism.research.construction_paired_fit")


def runtime_identity() -> dict:
    """Pin native scientific Python and the explicit dependency overlay."""
    if os.environ.get("AF_FIT_RUNTIME", "native") != "native":
        raise ValueError("paired fitting requires native AF_PYTHON")
    value = _runtime_identity()
    return value | {
        "casadi_root": os.environ.get("AF_CASADI_ROOT", ""),
        "pythonpath": os.environ.get("PYTHONPATH", ""),
    }


def execution_identity(root: Path) -> dict:
    """Bind all scientific, source, runtime and CPU resource choices before submit."""
    plan, _ = _campaign().verify(root)
    platform = os.environ.get("AF_PLATFORM", "aces")
    if platform not in {"aces", "delta"}:
        raise ValueError("AF_PLATFORM must be aces or delta")
    if len(plan.get("tasks", [])) != TASKS:
        raise ValueError("paired comparison requires 24 frozen model tasks")
    return {
        "protocol": "construction-paired-fit-cpu-execution-2",
        "plan_sha256": public.content_sha256(plan),
        "source_sha256": plan["source_sha256"],
        "root": str(root.resolve()),
        "repo": str(REPO),
        "platform": platform,
        "runtime": runtime_identity(),
        "entrypoint_sha256": {name: file_hash(REPO / name) for name in ENTRYPOINTS},
        "account": os.environ.get(
            "AF_ACCOUNT", "156264627414" if platform == "aces" else "bibo-delta-cpu"
        ),
        "partition": "cpu",
        "tasks": TASKS,
        "array": "0-23%8",
        "maximum_parallel_tasks": 8,
        "cpus_per_task": 1,
        "memory": "8G",
        "wall_time": "03:00:00",
        "report_wall_time": "00:15:00",
        "preflight_wall_time": "00:05:00",
        "exclude": "ac042" if platform == "aces" else None,
        "gpus": 0,
        "llm_calls": 0,
        "test_data_opened": False,
        "automatic_revision": False,
        "automatic_followup": False,
    }


def _identity_differences(saved: dict, current: dict, prefix: str = "") -> list[dict]:
    """Name every differing field without dumping the complete identity."""
    differences = []
    for key in sorted(saved.keys() | current.keys()):
        field = f"{prefix}.{key}" if prefix else key
        if key not in saved or key not in current:
            differences.append({
                "field": field,
                "saved_present": key in saved,
                "current_present": key in current,
                "saved": saved.get(key),
                "current": current.get(key),
            })
        elif isinstance(saved[key], dict) and isinstance(current[key], dict):
            differences.extend(_identity_differences(saved[key], current[key], field))
        elif saved[key] != current[key]:
            differences.append({
                "field": field, "saved": saved[key], "current": current[key],
            })
    return differences


def verify_execution(root: Path) -> dict:
    """Reject a changed plan, source, interpreter, dependency set or worker."""
    value = read_seal(root / "launcher_execution.json")
    differences = _identity_differences(value, execution_identity(root))
    if differences:
        raise ValueError(
            "paired fitting launcher identity differs: "
            + json.dumps(differences, sort_keys=True)
        )
    return value


def submit(root: Path) -> dict:
    """Submit once, retaining ambiguous receipts without issuing duplicate jobs."""
    root = root.resolve()
    identity = execution_identity(root)
    os.environ.update(AF_REPO_ROOT=str(REPO), AF_PAIRED_FIT_ROOT=str(root))
    directory = root / SUBMISSIONS
    with public._lock(directory):
        seal(root / "launcher_execution.json", identity)
        execution_sha256 = public.content_sha256(identity)
        seal(directory / "identity.json", {"execution_sha256": execution_sha256})
        manifest = directory / "manifest.json"
        saved = read_seal(manifest) if manifest.exists() else None
        if saved and saved.get("execution_sha256") != execution_sha256:
            raise ValueError("paired fitting submission identity differs")
        (root / "logs").mkdir(exist_ok=True)
        base = [
            "--kill-on-invalid-dep=yes",
            "--nodes=1",
            "--ntasks=1",
            "--export=ALL",
            f"--account={identity['account']}",
            "--partition=cpu",
            "--cpus-per-task=1",
            "--mem=8G",
            *([f"--exclude={identity['exclude']}"] if identity["exclude"] else []),
        ]
        worker = REPO / "scripts/hpc/run_construction_paired_fit.sh"
        jobs = {}
        for stage in ("preflight", "run", "report"):
            worker_stage = "verify-execution" if stage == "preflight" else stage
            if stage == "preflight":
                scheduling = [f"--time={identity['preflight_wall_time']}"]
            elif stage == "run":
                scheduling = [
                    f"--dependency=afterok:{jobs['preflight']}",
                    f"--array={identity['array']}",
                    f"--time={identity['wall_time']}",
                ]
            else:
                scheduling = [
                    f"--dependency=afterany:{jobs['run']}",
                    f"--time={identity['report_wall_time']}",
                ]
            options = [
                *base,
                f"--job-name=construction-paired-{stage}",
                f"--output={root}/logs/{stage}-%A_%a.out",
                f"--error={root}/logs/{stage}-%A_%a.err",
                *scheduling,
            ]
            argv = ["sbatch", "--parsable", *options, str(worker), worker_stage, "0"]
            job = confirmed_job(directory, stage, argv)
            if job is None:
                if saved:
                    raise ValueError(
                        "submission manifest lacks a confirmed job receipt"
                    )
                job = submit_job(directory, stage, options, worker, worker_stage, 0)
            jobs[stage] = job
        value = {
            "execution_sha256": execution_sha256,
            "jobs": jobs,
            "tasks": TASKS,
            "maximum_parallel_tasks": 8,
            "gpus": 0,
            "llm_calls": 0,
            "test_data_opened": False,
            "automatic_revision": False,
            "automatic_followup": False,
        }
        if saved is not None and saved != value:
            raise ValueError("paired fitting manifest differs from confirmed receipts")
        seal(manifest, value)
        return value


def _summary(value: dict) -> dict:
    """Print completion counts, never the full model/data packet."""
    return {
        key: value[key]
        for key in (
            "status", "expected", "recorded", "status_counts", "task_status_counts",
            "paired_status_counts", "completed", "pending", "failed",
            "numerically_verified", "validation_available",
        )
        if key in value
    }


def status(root: Path) -> dict:
    """Rebuild receipt counts without restarting any numerical operation."""
    summary = _campaign().report(root)
    manifest = root / SUBMISSIONS / "manifest.json"
    return _summary(summary) | {
        "jobs": read_seal(manifest)["jobs"] if manifest.exists() else {},
        "report": str(root / "report/summary.json"),
        "logs": str(root / "logs"),
    }


def main(argv: list[str] | None = None) -> None:
    """Separate schema-only preparation, scheduling and compute-node fitting."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in (
        "prepare", "submit", "verify-execution", "run", "status", "report", "inspect"
    ):
        command = commands.add_parser(name)
        command.add_argument("--root", type=Path, required=True)
        if name == "prepare":
            command.add_argument("--inputs", type=Path, required=True)
            command.add_argument(
                "--config", type=Path,
                default=REPO / "configs/construction_paired_fit_v1.json",
            )
        elif name == "run":
            command.add_argument("--index", type=int, required=True)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.command == "prepare":
        plan = _campaign().prepare(
            root, read_seal(args.inputs),
            _campaign().Policy.model_validate_json(args.config.read_text()),
        )
        result = {
            "identity": public.content_sha256(plan),
            "tasks": len(plan.get("tasks", [])),
            "root": str(root),
        }
    elif args.command == "submit":
        result = submit(root)
    elif args.command == "verify-execution":
        value = verify_execution(root)
        result = {
            "status": "verified", "execution_sha256": public.content_sha256(value),
            "tasks": value["tasks"],
        }
    elif args.command == "run":
        if not os.environ.get("SLURM_JOB_ID"):
            raise ValueError("numerical work requires a Slurm compute allocation")
        if not 0 <= args.index < TASKS:
            raise ValueError("task index must be between 0 and 23")
        verify_execution(root)
        value = _campaign().run_task(root, args.index)
        result = _summary(value) | {"index": args.index}
    elif args.command == "status":
        result = status(root)
    elif args.command == "report":
        result = _summary(_campaign().report(root)) | {
            "report": str(root / "report/summary.json")
        }
    else:
        result = _campaign().inspect(root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
