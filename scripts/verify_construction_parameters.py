#!/usr/bin/env python3
"""Freeze and independently verify saved parameters; never repeat fitting."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import public_fitting as public
from autoformalism.research import construction_verification_campaign as campaign
from scripts.submit_construction_fit_aces import confirmed_job, runtime_identity
from scripts.submit_review_continuation import submit_job

REPO = Path(__file__).resolve().parents[1]


def execution_identity(root: Path) -> dict:
    """Pin the scientific plan, interpreter and CPU-only scheduler resources."""
    plan, _ = campaign.verify(root)
    return {
        "protocol": "construction-saved-verification-aces-1",
        "plan_sha256": public.content_sha256(plan),
        "root": str(root.resolve()),
        "runtime": runtime_identity(),
        "account": os.environ.get("AF_ACCOUNT", "156264627414"),
        "partition": "cpu",
        "array": f"0-{len(plan['candidates']) - 1}%4",
        "cpus_per_task": 1,
        "memory": "8G",
        "wall_time": "00:15:00",
        "exclude": "ac042",
        "gpus": 0,
    }


def verify_execution(root: Path) -> dict:
    """Workers must use exactly the submitted interpreter and campaign."""
    value = read_seal(root / "launcher_execution.json")
    if value != execution_identity(root):
        raise ValueError("verification launcher identity differs")
    return value


def submit(root: Path) -> dict:
    """Reserve one CPU array and one report; never retry uncertain submissions."""
    root = root.resolve()
    identity = execution_identity(root)
    os.environ.update(AF_REPO_ROOT=str(REPO), AF_OUTPUT_ROOT=str(root))
    directory = root / "submissions/output-verification-1"
    with public._lock(directory):
        seal(root / "launcher_execution.json", identity)
        manifest = directory / "manifest.json"
        if manifest.exists():
            value = read_seal(manifest)
            if value["execution_sha256"] != public.content_sha256(identity):
                raise ValueError("submission identity differs")
            return value
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
            "--time=00:15:00",
            "--exclude=ac042",
        ]
        worker = REPO / "scripts/hpc/run_construction_verification_aces.sh"
        jobs = {}
        for stage in ("verify", "report"):
            options = [
                *base,
                f"--job-name=saved-parameters-{stage}",
                f"--output={root}/logs/{stage}-%A_%a.out",
                f"--error={root}/logs/{stage}-%A_%a.err",
                *(
                    [f"--array={identity['array']}"]
                    if stage == "verify"
                    else [f"--dependency=afterany:{jobs['verify']}"]
                ),
            ]
            argv = ["sbatch", "--parsable", *options, str(worker), stage, "0"]
            job = confirmed_job(directory, stage, argv)
            if job is None:
                job = submit_job(directory, stage, options, worker, stage, 0)
            jobs[stage] = job
        value = {
            "execution_sha256": public.content_sha256(identity),
            "jobs": jobs,
            "optimizer_calls": 0,
            "llm_calls": 0,
            "test_data_opened": False,
            "automatic_followup": False,
        }
        seal(manifest, value)
        return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("prepare", "submit", "verify-execution", "run", "report", "inspect"),
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command == "prepare":
        if args.source is None:
            parser.error("prepare requires --source")
        result = campaign.prepare(root, args.source)
    elif args.command == "submit":
        result = submit(root)
    elif args.command == "verify-execution":
        result = verify_execution(root)
    elif args.command == "run":
        if args.index is None:
            parser.error("run requires --index")
        operation = campaign.run_task(root, args.index)
        value = operation.get("value") or {}
        result = {
            "status": operation["status"],
            "success": value.get("success"),
            "normalized_mse": value.get("normalized_mse"),
        }
    else:
        result = getattr(campaign, args.command)(root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
