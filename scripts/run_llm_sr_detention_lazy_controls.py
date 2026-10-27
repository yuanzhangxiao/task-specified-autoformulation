#!/usr/bin/env python3
"""Run controlled detention searches with lazy derivative evaluation."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tarfile
from pathlib import Path

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import public_fitting as public
from autoformalism.research import llm_sr_detention_lazy_controls_campaign as campaign
from scripts.submit_construction_fit_aces import confirmed_job, file_hash
from scripts.submit_review_continuation import submit_job

REPO = Path(__file__).resolve().parents[1]
ENTRYPOINTS = (
    "scripts/run_llm_sr_detention_lazy_controls.py",
    "scripts/hpc/run_llm_sr_detention_lazy_controls_delta.sh",
    "scripts/hpc/start_llm_sr_detention_lazy_controls_delta.sh",
    "scripts/submit_construction_fit_aces.py",
    "scripts/submit_review_continuation.py",
)
SUBMISSIONS = "submissions/cpu-llm-sr-detention-lazy-controls-1"


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
        "protocol": "llm-sr-detention-lazy-controls-cpu-execution-1",
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
        "tasks": 27,
        "qualification_array": "0-5%2",
        "run_array": "6-26%2",
        "cpus_per_task": 1,
        "memory": "8G",
        "wall_time": "04:00:00",
        "qualification_wall_time": "02:00:00",
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
    identity = execution_identity(root)
    os.environ.update(
        AF_REPO_ROOT=str(REPO), AF_LLMSR_DETENTION_LAZY_CONTROLS_ROOT=str(root)
    )
    directory = root / SUBMISSIONS
    with public._lock(directory):
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
        worker = REPO / "scripts/hpc/run_llm_sr_detention_lazy_controls_delta.sh"
        jobs = {}
        for stage in ("qualify", "run", "report"):
            options = [
                *base,
                "--time="
                + (
                    identity["report_wall_time"]
                    if stage == "report"
                    else identity["qualification_wall_time"]
                    if stage == "qualify"
                    else identity["wall_time"]
                ),
                f"--job-name=llmsr-detention-lazy-controls-{stage}",
                f"--output={root}/logs/{stage}-%A_%a.out",
                f"--error={root}/logs/{stage}-%A_%a.err",
                *(
                    [f"--array={identity['qualification_array']}"]
                    if stage == "qualify"
                    else [
                        f"--array={identity['run_array']}",
                        f"--dependency=afterany:{jobs['qualify']}",
                    ]
                    if stage == "run"
                    else [f"--dependency=afterany:{jobs['run']}"]
                ),
            ]
            argv = ["sbatch", "--parsable", *options, str(worker), stage, "0"]
            job = confirmed_job(directory, stage, argv)
            if job is None:
                if saved:
                    raise ValueError("completed submission lacks a confirmed receipt")
                job = submit_job(directory, stage, options, worker, stage, 0)
            jobs[stage] = job
        result = {
            "execution_sha256": execution_sha256,
            "jobs": jobs,
            "tasks": 27,
            "maximum_parallel_tasks": 2,
            "gpus": 0,
            "llm_calls": 0,
            "validation_or_test_access": False,
            "optimization_allowed": True,
        }
        if saved is not None and saved != result:
            raise ValueError("submission manifest differs from confirmed receipts")
        seal(manifest, result)
        return result


def report(root: Path) -> dict:
    """Publish a derived report without renewing work or requiring old runtimes."""
    result = campaign.report(root)
    with public._lock(root / "report"):
        public._write(root / "report/summary.json", result)
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
            }
            for row in result["tasks"]
        ]
    }


def status(root: Path) -> dict:
    """Reconstruct progress without performing or renewing numerical work."""
    return compact_report(campaign.report(root))


def inspect(root: Path) -> dict:
    """Archive the plan, stage receipts, diagnostics and scheduler evidence."""
    result = report(root)
    destination = root.parent / f"{root.name}-inspection.tar.gz"
    temporary = destination.with_suffix(".tmp")
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
        help="freeze six shared qualifications, eighteen fits and three checks",
    )
    prepare.add_argument("--inputs", type=Path, required=True)
    prepare.add_argument("--root", type=Path, required=True)
    prepare.add_argument("--direction-seconds", type=float, default=120)
    export = commands.add_parser(
        "export", help="authenticate and package chart audit history"
    )
    export.add_argument("--source", type=Path, required=True)
    export.add_argument("--destination", type=Path, required=True)
    run = commands.add_parser("run", help="execute one frozen roster position once")
    run.add_argument("--root", type=Path, required=True)
    run.add_argument("--index", type=int, choices=range(27), required=True)
    for name in ("submit", "verify-execution", "status", "report", "inspect"):
        command = commands.add_parser(name)
        command.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command != "export":
        args.root = args.root.resolve()
    if args.command == "export":
        print(
            "Authenticating the saved active-block results and their history...",
            file=sys.stderr,
            flush=True,
        )
        result = campaign.export_inputs(args.source, args.destination)
    elif args.command == "prepare":
        print(
            "Authenticating saved results and freezing 27 tasks...",
            file=sys.stderr,
            flush=True,
        )
        policy = campaign.Policy(direction_seconds=args.direction_seconds)
        result = campaign.prepare(args.root, read_seal(args.inputs), policy)
    elif args.command == "run":
        context = campaign.load_run(args.root)
        verify_execution(args.root, verified_plan=context.plan)
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
