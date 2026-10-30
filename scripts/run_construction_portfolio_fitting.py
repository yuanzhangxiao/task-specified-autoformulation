#!/usr/bin/env python3
"""Freeze and run the 23-model construction portfolio on Delta CPUs."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import public_fitting as public
from autoformalism.research import construction_portfolio_fitting as campaign
from scripts.submit_construction_fit_aces import confirmed_job, file_hash
from scripts.submit_construction_fit_aces import runtime_identity as _runtime_identity
from scripts.submit_review_continuation import submit_job

REPO = Path(__file__).resolve().parents[1]
ENTRYPOINTS = (
    "scripts/run_construction_portfolio_fitting.py",
    "scripts/hpc/start_construction_portfolio_fitting_delta.sh",
    "scripts/submit_construction_fit_aces.py",
    "scripts/submit_review_continuation.py",
)
SUBMISSIONS = "submissions/cpu-construction-portfolio-1"


def progress(message: str) -> None:
    """Flush progress before filesystem or scheduler operations can block."""
    print(f"[{datetime.now(UTC).isoformat()}] {message}", file=sys.stderr, flush=True)


def runtime_identity() -> dict:
    """Bind the actual native interpreter, scientific packages and import paths."""
    if os.environ.get("AF_FIT_RUNTIME", "native") != "native":
        raise ValueError("construction portfolio requires native AF_PYTHON")
    import mpmath

    return _runtime_identity() | {
        "pythonpath": os.environ.get("PYTHONPATH", ""),
        "casadi_root": os.environ.get("AF_CASADI_ROOT", ""),
        "mpmath_version": mpmath.__version__,
    }


def execution_identity(root: Path, *, verified_plan: dict | None = None) -> dict:
    """Bind one immutable plan to its launcher, runtime and CPU resource ceiling."""
    plan = campaign.verify(root)[0] if verified_plan is None else verified_plan
    count = len(plan["tasks"])
    if not count:
        raise ValueError("construction portfolio requires a nonempty task array")
    raw = os.environ.get("AF_CONSTRUCTION_PORTFOLIO_CONCURRENCY", "2")
    if not raw.isdecimal() or not 1 <= int(raw) <= 23:
        raise ValueError("AF_CONSTRUCTION_PORTFOLIO_CONCURRENCY must be 1 through 23")
    concurrency = int(raw)
    return {
        "protocol": "construction-portfolio-cpu-execution-1",
        "plan_sha256": public.content_sha256(plan),
        "source_sha256": plan["source_sha256"],
        "root": str(root.resolve()),
        "repo": str(REPO),
        "runtime": runtime_identity(),
        "entrypoint_sha256": {name: file_hash(REPO / name) for name in ENTRYPOINTS},
        "account": os.environ.get("AF_ACCOUNT", "bibo-delta-cpu"),
        "partition": "cpu",
        "tasks": count,
        "array": f"0-{count - 1}%{concurrency}",
        "maximum_parallel_tasks": concurrency,
        "cpus_per_task": 1,
        "memory": "8G",
        "wall_time": "02:00:00",
        "report_wall_time": "00:30:00",
        "gpus": 0,
        "llm_calls": 0,
        "validation_or_test_access": False,
        "automatic_revision": False,
    }


def verify_execution(root: Path) -> dict:
    """Reject a changed plan, source, worker or interpreter before numerical work."""
    value = read_seal(root / "launcher_execution.json")
    if value != execution_identity(root):
        raise ValueError("construction portfolio launcher identity differs")
    return value


def submit(root: Path) -> dict:
    """Reuse confirmed job receipts and refuse retries after ambiguous submission."""
    root = root.resolve()
    progress("Authenticating the frozen 23-model campaign and runtime...")
    plan, _ = campaign.verify(root)
    identity = execution_identity(root, verified_plan=plan)
    os.environ.update(AF_REPO_ROOT=str(REPO), AF_CONSTRUCTION_PORTFOLIO_ROOT=str(root))
    directory = root / SUBMISSIONS
    with public._lock(directory):
        progress("Saving execution identity and checking scheduler receipts...")
        seal(root / "launcher_execution.json", identity)
        execution_sha256 = public.content_sha256(identity)
        seal(directory / "identity.json", {"execution_sha256": execution_sha256})
        manifest = directory / "manifest.json"
        saved = read_seal(manifest) if manifest.exists() else None
        if saved and saved.get("execution_sha256") != execution_sha256:
            raise ValueError("construction portfolio submission identity differs")
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
        ]
        worker = REPO / "scripts/hpc/start_construction_portfolio_fitting_delta.sh"
        jobs = {}
        for stage in ("run", "report"):
            options = [
                *base,
                f"--job-name=construction-portfolio-{stage}",
                f"--output={root}/logs/{stage}-%A_%a.out",
                f"--error={root}/logs/{stage}-%A_%a.err",
                *(
                    [f"--array={identity['array']}", f"--time={identity['wall_time']}"]
                    if stage == "run"
                    else [
                        f"--dependency=afterany:{jobs['run']}",
                        f"--time={identity['report_wall_time']}",
                    ]
                ),
            ]
            worker_stage = f"worker-{stage}"
            argv = ["sbatch", "--parsable", *options, str(worker), worker_stage, "0"]
            job = confirmed_job(directory, stage, argv)
            if job is None:
                if saved:
                    raise ValueError("completed submission lacks a confirmed receipt")
                progress(
                    f"Submitting {stage} to Slurm (60-second scheduler timeout)..."
                )
                job = submit_job(directory, stage, options, worker, worker_stage, 0)
                progress(f"Slurm accepted {stage}: job {job}; receipt saved.")
            else:
                progress(f"Reusing confirmed {stage}: job {job}.")
            jobs[stage] = job
        result = {
            "execution_sha256": execution_sha256,
            "jobs": jobs,
            "tasks": identity["tasks"],
            "maximum_parallel_tasks": identity["maximum_parallel_tasks"],
            "gpus": 0,
            "llm_calls": 0,
            "validation_or_test_access": False,
            "automatic_revision": False,
        }
        if saved is not None and saved != result:
            raise ValueError("submission manifest differs from confirmed job receipts")
        seal(manifest, result)
        progress("Submission complete; jobs run independently of this terminal.")
        return result


def status(root: Path) -> dict:
    """Read sealed terminal receipts without rerunning numerical work or reporting."""
    plan = read_seal(root / "plan.json")
    plan_sha256 = public.content_sha256(plan)
    rows = []
    for task in plan["tasks"]:
        name = task["task_id"]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ValueError("unsafe portfolio task path")
        folder = root / "results" / name
        terminal = folder / "result.json"
        if terminal.exists():
            saved = read_seal(terminal)
            if saved.get("identity") != {"plan_sha256": plan_sha256, "task": task}:
                raise ValueError("task status receipt differs from the frozen plan")
            row = {
                "status": saved["status"],
                "fitting_status": saved.get("fitting_status"),
                "terminal_receipt_present": True,
            }
        else:
            operations = folder / "operations"
            row = {
                "status": "terminal_receipt_missing",
                "terminal_receipt_present": False,
                "started_operations": sum(1 for _ in operations.glob("*/started.json")),
                "recorded_operations": sum(1 for _ in operations.glob("*/result.json")),
            }
        rows.append({"index": task["index"], "candidate_id": name, **row})
    recorded = sum(row["terminal_receipt_present"] for row in rows)
    manifest = root / SUBMISSIONS / "manifest.json"
    return {
        "status": "terminal_receipts_present"
        if recorded == len(rows)
        else "incomplete",
        "expected": len(rows),
        "recorded": recorded,
        "jobs": read_seal(manifest)["jobs"] if manifest.exists() else {},
        "tasks": rows,
        "summary": str(root / "summary.json"),
        "summary_present": (root / "summary.json").is_file(),
        "logs": str(root / "logs"),
        "scope": "Receipt snapshot; report/inspect reconstruct numerical evidence.",
    }


def main(argv: list[str] | None = None) -> None:
    """Keep preparation and read-only inspection separate from fitting/submission."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in (
        "prepare",
        "verify",
        "submit",
        "verify-execution",
        "run",
        "status",
        "report",
        "inspect",
    ):
        command = commands.add_parser(name)
        command.add_argument("--root", type=Path, required=True)
        if name == "prepare":
            command.add_argument("--source", type=Path, required=True)
            command.add_argument(
                "--config",
                type=Path,
                default=REPO / "configs/construction_portfolio_fitting_v1.json",
            )
        elif name == "run":
            command.add_argument("--index", type=int, required=True)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    progress(f"Construction portfolio: command={args.command}, root={root}")
    if args.command == "prepare":
        result = campaign.prepare(
            root,
            args.source.resolve(),
            campaign.Policy.model_validate_json(args.config.read_text()),
        )
    elif args.command == "verify":
        plan, _ = campaign.verify(root)
        result = {"verified": True, "tasks": len(plan["tasks"])}
    elif args.command == "submit":
        result = submit(root)
    elif args.command == "verify-execution":
        result = verify_execution(root)
    elif args.command == "run":
        verify_execution(root)
        row = campaign.run_task(root, args.index)
        result = {
            "index": args.index,
            "status": row["status"],
            "fitting_status": row.get("fitting_status"),
        }
    elif args.command == "status":
        result = status(root)
    elif args.command == "report":
        summary = campaign.report(root)
        result = {key: summary[key] for key in ("status", "expected", "recorded")}
        result["summary"] = str(root / "summary.json")
    else:
        result = {"archive": str(campaign.inspection(root).resolve())}
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
