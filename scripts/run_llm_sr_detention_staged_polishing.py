#!/usr/bin/env python3
"""Run two 30-minute detention searches with separately allocated endpoint audits."""

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
from autoformalism.research import (
    llm_sr_detention_staged_polishing_campaign as campaign,
)
from scripts.submit_construction_fit_aces import confirmed_job, file_hash
from scripts.submit_review_continuation import submit_job

REPO = Path(__file__).resolve().parents[1]
ENTRYPOINTS = (
    "scripts/run_llm_sr_detention_staged_polishing.py",
    "scripts/hpc/run_llm_sr_detention_staged_polishing_delta.sh",
    "scripts/hpc/start_llm_sr_detention_staged_polishing_delta.sh",
    "scripts/submit_construction_fit_aces.py",
    "scripts/submit_review_continuation.py",
)
SUBMISSIONS = "submissions/cpu-llm-sr-detention-staged-polishing-1"


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
        "protocol": "llm-sr-detention-staged-polishing-cpu-execution-1",
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
        "tasks": 2,
        "run_array": "0-1%2",
        "cpus_per_task": 1,
        "memory": "8G",
        "wall_time": "04:00:00",
        "report_wall_time": "00:30:00",
        "numerical_reservation_ceiling_seconds": 8400,
        "maximum_numerical_allowances": {
            "search_segments": 1,
            "search_seconds": 1800,
            "independent_verification_seconds": 120,
            "search_audit_coordinates": 9,
            "search_audit_seconds_per_coordinate": 360,
            "strict_audit_coordinates_if_required": 9,
            "strict_audit_seconds_per_coordinate": 360,
        },
        "gpus": 0,
        "llm_calls": 0,
        "validation_or_test_access": False,
        "optimization_allowed": True,
    }


def verify_execution(root: Path, *, verified_plan: dict | None = None) -> dict:
    """Refuse a changed runtime, input, worker or source before diagnostics."""
    saved = read_seal(root / "launcher_execution.json")
    if saved != execution_identity(root, verified_plan=verified_plan):
        raise ValueError("LLM-SR detention staged polishing launcher identity differs")
    return saved


def submit(root: Path) -> dict:
    """Reuse confirmed jobs; an ambiguous scheduler response blocks resubmission."""
    root = root.resolve()
    progress("Authenticating compact staged-polishing inputs and runtime...")
    context = campaign.load_run(root)
    progress("Input authentication complete; checking launcher execution identity...")
    identity = execution_identity(root, verified_plan=context.plan)
    os.environ.update(
        AF_REPO_ROOT=str(REPO), AF_LLMSR_DETENTION_STAGED_POLISHING_ROOT=str(root)
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
            raise ValueError(
                "LLM-SR detention staged polishing submission identity differs"
            )
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
        worker = REPO / "scripts/hpc/run_llm_sr_detention_staged_polishing_delta.sh"
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
                f"--job-name=llmsr-detention-staged-polishing-{stage}",
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
            "tasks": 2,
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


def task_summary(row: dict, task: dict) -> dict:
    """Separate search stops, retention, endpoint assessment and audit allocation."""
    optimization = row.get("optimization") or {}
    retained = row.get("retained") or {}
    verification = row.get("candidate_verification") or {}
    decision = row.get("audit_decision") or {}
    strict_audit = row.get("strict_audit") or {}
    reuse = row.get("initial_sample_reuse") or optimization.get(
        "initial_sample_reuse"
    ) or {}
    return {
        "index": task["index"],
        "kind": task.get("kind"),
        "start": task.get("start_id"),
        "source_index": task.get("source_index"),
        "mode": task.get("mode"),
        "status": row["status"],
        "initial_training_nmse": row.get("initial_training_nmse"),
        "training_nmse": retained.get("training_nmse"),
        "retained_independently_verified": retained.get("usable_for_retention"),
        "verified_improvement": row.get("verified_improvement"),
        "candidate_verification_status": verification.get("status"),
        "candidate_usable_for_retention": verification.get("usable_for_retention"),
        "stop_reason": row.get("stop_reason"),
        "optimizer_stop_reason": optimization.get("stop_reason"),
        "optimization_status": row.get("optimization_status"),
        "optimization_performed": row.get("optimization_performed"),
        "diagnostic_status": row.get("diagnostic_status"),
        "endpoint_assessment_status": (row.get("endpoint_assessment") or {}).get(
            "status"
        ),
        "endpoint_assessment_reason": (row.get("endpoint_assessment") or {}).get(
            "reason"
        ),
        "search_audit_complete": row.get("search_audit_complete"),
        "audit_decision": {
            key: decision[key]
            for key in (
                "decision",
                "strict_audit_required",
                "strict_audit_reason",
                "near_stationarity_plausible",
                "full_chart_search_qualified",
                "full_chart_evidence_available",
                "maximum_lower_empirical_descent_magnitude",
                "gradient_tolerance",
            )
            if key in decision
        },
        "strict_audit_status": strict_audit.get("status"),
        "strict_audit_performed": strict_audit.get("performed"),
        "strict_audit_skipped_reason": strict_audit.get("skipped_reason"),
        "full_model_stationarity_qualified": row.get(
            "full_model_stationarity_qualified"
        ),
        "initial_sample_reuse": {
            key: reuse[key]
            for key in (
                "seeded_residuals",
                "seeded_active_columns",
                "seeded_held_columns",
                "residual_reuse_hits",
                "jacobian_reuse_hits",
            )
            if key in reuse
        },
    }


def compact_report(result: dict) -> dict:
    """Keep interactive output small while full receipts remain in the report."""
    return {
        key: result[key] for key in ("protocol", "status", "expected", "recorded")
    } | {
        "tasks": [task_summary(row, row["identity"]["task"]) for row in result["tasks"]]
    }


def status(root: Path) -> dict:
    """Read sealed terminal receipts without rebuilding history or numerical work.

    This is a progress snapshot, not requalification of intermediate or terminal
    numerical evidence. The report command independently reconstructs that evidence.
    """
    progress("Reading sealed plan and task progress receipts...")
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
            row = task_summary(saved, task)
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
        help="freeze two 30-minute retained-incumbent staged-polishing tasks",
    )
    prepare.add_argument("--inputs", type=Path, required=True)
    prepare.add_argument("--root", type=Path, required=True)
    prepare.add_argument("--direction-seconds", type=float, default=360)
    export = commands.add_parser(
        "export", help="authenticate and package saved polishing incumbents"
    )
    export.add_argument("--source", type=Path, required=True)
    export.add_argument("--destination", type=Path, required=True)
    run = commands.add_parser("run", help="execute one frozen roster position once")
    run.add_argument("--root", type=Path, required=True)
    run.add_argument("--index", type=int, choices=range(2), required=True)
    for name in ("submit", "verify-execution", "status", "report", "inspect"):
        command = commands.add_parser(name)
        command.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command != "export":
        args.root = args.root.resolve()
    if args.command == "export":
        progress(
            "Authenticating saved polishing results "
            "and exporting compact inputs..."
        )
        result = campaign.export_inputs(args.source, args.destination)
    elif args.command == "prepare":
        progress(
            "Authenticating compact inputs and freezing two staged-polishing tasks..."
        )
        policy = campaign.Policy(direction_seconds=args.direction_seconds)
        result = campaign.prepare(args.root, read_seal(args.inputs), policy)
        progress("Two-task preparation complete.")
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
        result = task_summary(row, context.plan["tasks"][args.index]) | {
            "record": str(args.root / "tasks" / str(args.index) / "result.json"),
        }
    elif args.command == "submit":
        result = submit(args.root)
    elif args.command == "verify-execution":
        progress("Authenticating frozen inputs, worker and execution identity...")
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
