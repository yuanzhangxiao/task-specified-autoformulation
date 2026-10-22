#!/usr/bin/env python3
"""Prepare, submit and inspect isolated training-only construction reliability."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import public_fitting as public
from autoformalism.research import construction_fit_reliability as campaign
from scripts.submit_construction_fit_aces import (
    confirmed_job,
    file_hash,
)
from scripts.submit_construction_fit_aces import runtime_identity as _runtime_identity
from scripts.submit_review_continuation import submit_job

REPO = Path(__file__).resolve().parents[1]
ENTRYPOINTS = (
    "scripts/repair_construction_fitting.py",
    "scripts/hpc/run_construction_reliability.sh",
    "scripts/hpc/start_construction_reliability.sh",
    "scripts/submit_construction_fit_aces.py",
    "scripts/submit_review_continuation.py",
)
SUBMISSIONS = "submissions/cpu-reliability-1"


def runtime_identity() -> dict:
    """Pin native scientific Python, dependencies and the selected CasADi overlay."""
    if os.environ.get("AF_FIT_RUNTIME", "native") != "native":
        raise ValueError("construction reliability requires native AF_PYTHON")
    return _runtime_identity() | {
        "casadi_root": os.environ.get("AF_CASADI_ROOT", ""),
        "pythonpath": os.environ.get("PYTHONPATH", ""),
    }


def execution_identity(root: Path) -> dict:
    """Bind the exact plan/source/runtime/entrypoints and CPU resource ceiling."""
    plan, _ = campaign.verify(root)
    platform = os.environ.get("AF_PLATFORM", "delta")
    if platform not in {"delta", "aces"}:
        raise ValueError("AF_PLATFORM must be delta or aces")
    count = len(plan["tasks"])
    if not count:
        raise ValueError("reliability requires a nonempty task array")
    return {
        "protocol": "construction-reliability-cpu-execution-1",
        "plan_sha256": public.content_sha256(plan),
        "source_sha256": plan["source_sha256"],
        "root": str(root.resolve()),
        "repo": str(REPO),
        "platform": platform,
        "runtime": runtime_identity(),
        "entrypoint_sha256": {name: file_hash(REPO / name) for name in ENTRYPOINTS},
        "account": os.environ.get(
            "AF_ACCOUNT", "bibo-delta-cpu" if platform == "delta" else "156264627414"
        ),
        "partition": "cpu",
        "tasks": count,
        "array": f"0-{count - 1}%2",
        "maximum_parallel_tasks": 2,
        "cpus_per_task": 1,
        "memory": "8G",
        "wall_time": "01:00:00",
        "report_wall_time": "00:15:00",
        "exclude": "ac042" if platform == "aces" else None,
        "gpus": 0,
        "llm_calls": 0,
        "validation_data_opened": False,
        "test_data_opened": False,
        "automatic_revision": False,
    }


def verify_execution(root: Path) -> dict:
    """Workers cannot switch interpreter, source, entrypoints or scientific plan."""
    value = read_seal(root / "launcher_execution.json")
    if value != execution_identity(root):
        raise ValueError("construction reliability launcher identity differs")
    return value


def submit(root: Path) -> dict:
    """Submit a bounded CPU array and afterany report; refuse ambiguous retries."""
    root = root.resolve()
    identity = execution_identity(root)
    os.environ.update(AF_REPO_ROOT=str(REPO), AF_RELIABILITY_ROOT=str(root))
    directory = root / SUBMISSIONS
    with public._lock(directory):
        seal(root / "launcher_execution.json", identity)
        execution_sha256 = public.content_sha256(identity)
        seal(directory / "identity.json", {"execution_sha256": execution_sha256})
        manifest = directory / "manifest.json"
        saved = read_seal(manifest) if manifest.exists() else None
        if saved and saved.get("execution_sha256") != execution_sha256:
            raise ValueError("reliability submission identity differs")
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
        worker = REPO / "scripts/hpc/run_construction_reliability.sh"
        jobs = {}
        for stage in ("run", "report"):
            options = [
                *base,
                f"--job-name=construction-reliability-{stage}",
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
            argv = ["sbatch", "--parsable", *options, str(worker), stage, "0"]
            job = confirmed_job(directory, stage, argv)
            if job is None:
                if saved:
                    raise ValueError("completed submission lacks confirmed job receipt")
                job = submit_job(directory, stage, options, worker, stage, 0)
            jobs[stage] = job
        value = {
            "execution_sha256": execution_sha256,
            "jobs": jobs,
            "tasks": identity["tasks"],
            "maximum_parallel_tasks": 2,
            "gpus": 0,
            "llm_calls": 0,
            "validation_data_opened": False,
            "test_data_opened": False,
            "automatic_revision": False,
        }
        if saved is not None and saved != value:
            raise ValueError(
                "reliability manifest differs from confirmed job receipts"
            )
        seal(manifest, value)
        return value


def status(root: Path) -> dict:
    """Reconstruct receipt status without fitting or requiring the old runtime."""
    summary = campaign.report(root)
    manifest = root / SUBMISSIONS / "manifest.json"
    return {
        "status": summary["status"],
        "expected": summary["expected"],
        "recorded": summary["recorded"],
        "jobs": read_seal(manifest)["jobs"] if manifest.exists() else {},
        "report": str(root / "report/summary.json"),
        "logs": str(root / "logs"),
    }


def main(argv: list[str] | None = None) -> None:
    """Keep schema-only preparation distinct from identity-checked numerical work."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export")
    export.add_argument("--source", type=Path, required=True)
    export.add_argument("--destination", type=Path, required=True)
    for name in (
        "prepare",
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
            command.add_argument("--inputs", type=Path, required=True)
            command.add_argument(
                "--config",
                type=Path,
                default=REPO / "configs/construction_fit_reliability_v1.json",
            )
        elif name == "run":
            command.add_argument("--index", type=int, required=True)
    args = parser.parse_args(argv)
    if args.command == "export":
        result = campaign.export_inputs(
            args.source.resolve(), args.destination.resolve()
        )
    else:
        root = args.root.resolve()
        if args.command == "prepare":
            result = campaign.prepare(
                root,
                read_seal(args.inputs),
                campaign.Policy.model_validate_json(args.config.read_text()),
            )
        elif args.command == "submit":
            result = submit(root)
        elif args.command == "verify-execution":
            result = verify_execution(root)
        elif args.command == "run":
            verify_execution(root)
            value = campaign.run_task(root, args.index)
            result = {
                "status": value["status"],
                "optimization_status": value.get("optimization_status"),
                "retained_origin": value.get("retained_origin"),
                "training_nmse": (value.get("retained") or {}).get("training_nmse"),
            }
        elif args.command == "status":
            result = status(root)
        elif args.command == "report":
            value = campaign.report(root)
            result = {k: value[k] for k in ("status", "expected", "recorded")}
            result["report"] = str(root / "report/summary.json")
        else:
            result = campaign.inspect(root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
