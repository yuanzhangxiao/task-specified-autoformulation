#!/usr/bin/env python3
"""Recover detention derivative probes and continue two retained fits."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tarfile
from datetime import UTC, datetime
from pathlib import Path

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import public_fitting as public
from autoformalism.research import llm_sr_detention_probe_recovery_campaign as campaign
from scripts.submit_construction_fit_aces import confirmed_job, file_hash
from scripts.submit_review_continuation import submit_job

REPO = Path(__file__).resolve().parents[1]
ENTRYPOINTS = (
    "scripts/run_llm_sr_detention_probe_recovery.py",
    "scripts/hpc/run_llm_sr_detention_probe_recovery_delta.sh",
    "scripts/hpc/start_llm_sr_detention_probe_recovery_delta.sh",
    "scripts/submit_construction_fit_aces.py",
    "scripts/submit_review_continuation.py",
)
SUBMISSIONS = "submissions/cpu-llm-sr-detention-probe-recovery-1"


def progress(message: str) -> None:
    """Flush stage progress before potentially slow shared-filesystem work."""
    print(f"[{datetime.now(UTC).isoformat()}] {message}", file=sys.stderr, flush=True)


def execution_identity(root: Path, *, verified_plan: dict | None = None) -> dict:
    """Pin the plan, source, worker, interpreter, imports and CPU ceiling."""
    if verified_plan is None:
        plan, _, _ = campaign.verify(root)
    else:
        plan = verified_plan
    selected = os.environ.get("AF_PYTHON")
    if not selected or Path(selected).resolve() != Path(sys.executable).resolve():
        raise ValueError("use the selected AF_PYTHON for submission and execution")
    return {
        "protocol": "llm-sr-detention-probe-recovery-cpu-execution-1",
        "plan_sha256": public.content_sha256(plan),
        "source_sha256": plan["source_sha256"],
        "root": str(root.resolve()),
        "repo": str(REPO),
        "runtime": public._runtime()
        | {
            "python": sys.executable,
            "python_sha256": file_hash(Path(sys.executable)),
            "selected_python": selected,
            "pythonpath": os.environ.get("PYTHONPATH", ""),
            "casadi_root": os.environ.get("AF_CASADI_ROOT", ""),
        },
        "entrypoint_sha256": {name: file_hash(REPO / name) for name in ENTRYPOINTS},
        "account": os.environ.get("AF_ACCOUNT", "bibo-delta-cpu"),
        "partition": "cpu",
        "tasks": 6,
        "run_array": "0-5%2",
        "cpus_per_task": 1,
        "memory": "8G",
        "wall_time": "06:00:00",
        "report_wall_time": "00:30:00",
        "gpus": 0,
        "llm_calls": 0,
        "validation_or_test_access": False,
        "optimization_allowed": True,
    }


def verify_execution(root: Path, *, verified_plan: dict | None = None) -> dict:
    """Refuse a changed runtime, input, worker or source before diagnostics."""
    saved = read_seal(root / "launcher_execution.json")
    if saved != execution_identity(root, verified_plan=verified_plan):
        raise ValueError("LLM-SR controls launcher identity differs")
    return saved


def submit(root: Path) -> dict:
    """Reuse confirmed jobs; an ambiguous scheduler response blocks resubmission."""
    root = root.resolve()
    progress("Authenticating compact probe-recovery inputs and runtime...")
    context = campaign.load_run(root)
    progress("Input authentication complete; checking launcher execution identity...")
    identity = execution_identity(root, verified_plan=context.plan)
    os.environ.update(
        AF_REPO_ROOT=str(REPO), AF_LLMSR_DETENTION_PROBE_RECOVERY_ROOT=str(root)
    )
    directory = root / SUBMISSIONS
    with public._lock(directory):
        progress("Saving execution identity and scheduler intents...")
        seal(root / "launcher_execution.json", identity)
        execution_sha256 = public.content_sha256(identity)
        seal(directory / "identity.json", {"execution_sha256": execution_sha256})
        manifest = directory / "manifest.json"
        saved = read_seal(manifest) if manifest.exists() else None
        if saved and saved.get("execution_sha256") != execution_sha256:
            raise ValueError("LLM-SR controls submission identity differs")
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
        worker = REPO / "scripts/hpc/run_llm_sr_detention_probe_recovery_delta.sh"
        jobs = {}
        for stage in ("run", "report"):
            options = [
                *base,
                "--time="
                + (
                    identity["report_wall_time"]
                    if stage == "report"
                    else identity["wall_time"]
                ),
                f"--job-name=llmsr-detention-probe-recovery-{stage}",
                f"--output={root}/logs/{stage}-%A_%a.out",
                f"--error={root}/logs/{stage}-%A_%a.err",
                *(
                    [f"--array={identity['run_array']}"]
                    if stage == "run"
                    else [f"--dependency=afterany:{jobs['run']}"]
                ),
            ]
            argv = ["sbatch", "--parsable", *options, str(worker), stage, "0"]
            job = confirmed_job(directory, stage, argv)
            if job is None:
                if saved:
                    raise ValueError("completed submission lacks a confirmed receipt")
                progress(
                    f"Submitting {stage} to Slurm (scheduler timeout: 60 seconds)..."
                )
                job = submit_job(directory, stage, options, worker, stage, 0)
                progress(f"Slurm accepted {stage}: job {job}; receipt saved.")
            else:
                progress(f"Reusing confirmed {stage}: job {job}.")
            jobs[stage] = job
        result = {
            "execution_sha256": execution_sha256,
            "jobs": jobs,
            "tasks": 6,
            "maximum_parallel_tasks": 2,
            "gpus": 0,
            "llm_calls": 0,
            "validation_or_test_access": False,
            "optimization_allowed": True,
        }
        if saved is not None and saved != result:
            raise ValueError("submission manifest differs from confirmed receipts")
        progress("Saving completed submission manifest...")
        seal(manifest, result)
        progress("Submission complete; jobs run independently of this terminal.")
        return result


def report(root: Path) -> dict:
    """Publish a derived report without renewing work or requiring old runtimes."""
    progress("Reconstructing report from saved operation and probe receipts...")
    result = campaign.report(root)
    with public._lock(root / "report"):
        public._write(root / "report/summary.json", result)
    progress(f"Report saved: status={result['status']}.")
    return result


def compact_report(result: dict) -> dict:
    """Keep interactive output small while full receipts remain in the report."""
    return {
        key: result[key] for key in ("protocol", "status", "expected", "recorded")
    } | {
        "tasks": [
            {
                "index": row["identity"]["task"]["index"],
                "kind": row["identity"]["task"]["kind"],
                "start": row["identity"]["task"].get("start_id"),
                "arm": row["identity"]["task"].get("arm"),
                "status": row["status"],
                "training_nmse": (row.get("retained") or {}).get("training_nmse"),
                "phases_completed": row.get("phases_completed"),
                "active_names": row.get("active_names"),
                "held_names": row.get("held_names"),
                "stop_reason": row.get("stop_reason"),
                "optimization_status": row.get("optimization_status"),
                "optimization_performed": row.get("optimization_performed"),
                "diagnostic_status": row.get("diagnostic_status"),
            }
            for row in result["tasks"]
        ]
    }


def status(root: Path) -> dict:
    """Read sealed terminal receipts without rebuilding history or numerical work.

    This is a progress snapshot, not requalification of intermediate or terminal
    numerical evidence. The report command independently reconstructs that evidence.
    """
    plan = read_seal(root / "plan.json")
    plan_sha256 = public.content_sha256(plan)
    rows = []
    for task in plan["tasks"]:
        directory = root / "tasks" / str(task["index"])
        path = directory / "result.json"
        if path.exists():
            saved = read_seal(path)
            identity = saved.get("identity", {})
            if (
                identity.get("plan_sha256") != plan_sha256
                or identity.get("task") != task
            ):
                raise ValueError("task status receipt differs from the frozen plan")
            row = {
                "index": task["index"],
                "kind": task.get("kind"),
                "status": saved["status"],
                "stop_reason": saved.get("stop_reason"),
                "optimization_status": saved.get("optimization_status"),
                "optimization_performed": saved.get("optimization_performed"),
                "diagnostic_status": saved.get("diagnostic_status"),
                "training_nmse": (saved.get("retained") or {}).get("training_nmse"),
            }
        else:
            operations = directory / "operations"
            row = {
                "index": task["index"],
                "kind": task.get("kind"),
                "status": "terminal_receipt_missing",
                "started_operations": sum(1 for _ in operations.glob("*/started.json")),
                "recorded_operations": sum(1 for _ in operations.glob("*/result.json")),
            }
        rows.append(row)
    recorded = sum(row["status"] != "terminal_receipt_missing" for row in rows)
    return {
        "protocol": plan["protocol"],
        "status": (
            "terminal_receipts_present" if recorded == len(rows) else "incomplete"
        ),
        "expected": len(rows),
        "recorded": recorded,
        "tasks": rows,
        "scope": (
            "Sealed receipt snapshot; run report/inspect for numerical reconstruction."
        ),
        "fitting_performed": False,
        "submission_performed": False,
    }


def inspect(root: Path) -> dict:
    """Archive the plan, stage receipts, diagnostics and scheduler evidence."""
    result = report(root)
    destination = root.parent / f"{root.name}-inspection.tar.gz"
    temporary = destination.with_suffix(".tmp")
    progress(
        "Archiving compact inputs, operations, probe caches and scheduler receipts..."
    )
    with tarfile.open(temporary, "w:gz") as archive:
        for name in (
            "plan.json",
            "inputs.json",
            "launcher_execution.json",
            "tasks",
            "submissions",
            "logs",
            "report",
            "invocations",
        ):
            source = root / name
            paths = sorted(source.rglob("*")) if source.is_dir() else [source]
            for path in paths:
                if path.is_file() and not path.is_symlink() and path.name != ".lock":
                    archive.add(
                        path,
                        arcname=str(Path(root.name) / path.relative_to(root)),
                        recursive=False,
                    )
    temporary.replace(destination)
    progress(f"Inspection archive saved: {destination}")
    return {
        "status": result["status"],
        "archive": str(destination),
        "report": str(root / "report/summary.json"),
    }


def main(argv: list[str] | None = None) -> None:
    """Run only the explicitly requested stage; preparation never runs a fit."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser(
        "prepare",
        help="freeze two incumbent continuations and four diagnostic recoveries",
    )
    prepare.add_argument("--inputs", type=Path, required=True)
    prepare.add_argument("--root", type=Path, required=True)
    prepare.add_argument("--direction-seconds", type=float, default=360)
    export = commands.add_parser(
        "export", help="authenticate and package saved lazy-control endpoints"
    )
    export.add_argument("--source", type=Path, required=True)
    export.add_argument("--destination", type=Path, required=True)
    run = commands.add_parser("run", help="execute one frozen roster position once")
    run.add_argument("--root", type=Path, required=True)
    run.add_argument("--index", type=int, choices=range(6), required=True)
    for name in ("submit", "verify-execution", "status", "report", "inspect"):
        command = commands.add_parser(name)
        command.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command != "export":
        args.root = args.root.resolve()
    if args.command == "export":
        progress(
            "Authenticating saved lazy-control results and exporting compact inputs..."
        )
        result = campaign.export_inputs(args.source, args.destination)
    elif args.command == "prepare":
        progress("Authenticating compact inputs and freezing six recovery tasks...")
        policy = campaign.Policy(direction_seconds=args.direction_seconds)
        result = campaign.prepare(args.root, read_seal(args.inputs), policy)
        progress("Six-task preparation complete.")
    elif args.command == "run":
        progress(
            f"Task {args.index}: authenticating source, compact inputs and runtime..."
        )
        context = campaign.load_run(args.root)
        verify_execution(args.root, verified_plan=context.plan)
        progress(
            f"Task {args.index}: authentication complete; "
            "entering saved operation ledger."
        )
        row = campaign.run_task(args.root, args.index, context=context)
        result = {
            "index": args.index,
            "status": row["status"],
            "training_nmse": (row.get("retained") or {}).get("training_nmse"),
            "record": str(args.root / "tasks" / str(args.index) / "result.json"),
        }
    elif args.command == "submit":
        result = submit(args.root)
    elif args.command == "verify-execution":
        result = verify_execution(args.root)
    elif args.command == "inspect":
        result = inspect(args.root)
    elif args.command == "report":
        result = compact_report(report(args.root))
    else:
        result = status(args.root)
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
