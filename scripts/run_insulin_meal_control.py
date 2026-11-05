#!/usr/bin/env python3
"""Prepare, submit, inspect or resume the matched candidate-15 CPU comparison."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import public_fitting as public
from autoformalism.research import insulin_meal_campaign as campaign
from scripts.submit_construction_fit_aces import (
    confirmed_job,
    file_hash,
    runtime_identity,
)
from scripts.submit_review_continuation import submit_job

REPO = Path(__file__).resolve().parents[1]
WORKER = REPO / "scripts/hpc/start_insulin_meal_control_delta.sh"
SUBMISSIONS = "submissions/insulin-meal-control-1"


def launcher(root: Path, plan: dict) -> dict:
    """Pin the submission and compute-node interpreter, scripts and environment."""
    from autoformalism.fitting import insulin_meal_comparison as fit

    if os.environ.get("AF_FIT_RUNTIME", "native") != "native":
        raise ValueError("select a native Delta scientific Python for this comparison")
    reservations = fit.allowances(fit.Policy.model_validate(plan["policy"]))
    scripts = (
        Path(__file__),
        WORKER,
        REPO / "scripts/submit_construction_fit_aces.py",
        REPO / "scripts/submit_review_continuation.py",
    )
    return {
        "repo": str(REPO),
        "root": str(root),
        "plan_sha256": public.content_sha256(plan),
        "source_sha256": plan["source_sha256"],
        "runtime": runtime_identity(),
        "pythonpath": os.environ.get("PYTHONPATH", ""),
        "entrypoints": {str(p.relative_to(REPO)): file_hash(p) for p in scripts},
        "account": os.environ.get("AF_ACCOUNT", "bibo-delta-cpu"),
        "partition": "cpu",
        "array": "0-1%2",
        "time": "04:00:00",
        "reservations_seconds": reservations,
    }


def submit(root: Path) -> dict:
    plan, _ = campaign.verify(root, runtime=True)
    identity = launcher(root, plan)
    directory = root / SUBMISSIONS
    os.environ.update(AF_INSULIN_MEAL_CODE=str(REPO), AF_INSULIN_MEAL_ROOT=str(root))
    jobs = {}
    with public._lock(directory):
        seal(root / "launcher_execution.json", identity)
        (root / "logs").mkdir(exist_ok=True)
        for stage in ("run", "report"):
            options = [
                "--kill-on-invalid-dep=yes",
                "--nodes=1",
                "--ntasks=1",
                "--export=ALL",
                f"--account={identity['account']}",
                "--partition=cpu",
                "--cpus-per-task=1",
                "--mem=8G",
                f"--job-name=insulin-meal-{stage}",
                f"--output={root}/logs/{stage}-%A_%a.out",
                f"--error={root}/logs/{stage}-%A_%a.err",
                *(
                    [f"--array={identity['array']}", f"--time={identity['time']}"]
                    if stage == "run"
                    else [f"--dependency=afterany:{jobs['run']}", "--time=00:30:00"]
                ),
            ]
            worker_stage = f"worker-{stage}"
            argv = ["sbatch", "--parsable", *options, str(WORKER), worker_stage, "0"]
            job = confirmed_job(directory, stage, argv)
            if job is None:
                if (directory / "manifest.json").exists():
                    raise ValueError("saved submission is missing a confirmed receipt")
                print(f"Submitting {stage}...", flush=True)
                job = submit_job(directory, stage, options, WORKER, worker_stage, 0)
            jobs[stage] = job
        manifest = {"jobs": jobs, "execution_sha256": public.content_sha256(identity)}
        seal(directory / "manifest.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    export = sub.add_parser("export-source", help="rebuild the reviewed public packet")
    export.add_argument("--source", required=True, type=Path)
    export.add_argument("--evidence", required=True, type=Path)
    export.add_argument("--packet", required=True, type=Path)
    for action in ("prepare", "submit", "worker", "status", "report", "inspect"):
        arg = sub.add_parser(action)
        arg.add_argument("--root", required=True, type=Path)
        if action == "prepare":
            arg.add_argument(
                "--packet", type=Path, default=REPO / "inputs/candidate-15-source.json"
            )
        if action == "worker":
            arg.add_argument("--index", type=int, choices=(0, 1), required=True)
    args = parser.parse_args(argv)
    if args.action == "export-source":
        packet = campaign.source_packet(args.source, args.evidence)
        if public.content_sha256(packet) != campaign.SOURCE_PACKET:
            raise ValueError("exported packet differs from the reviewed source")
        seal(args.packet, packet)
        print(
            json.dumps({"packet": str(args.packet), "sha256": campaign.SOURCE_PACKET})
        )
        return
    root = args.root.resolve()
    if args.action == "prepare":
        result = campaign.prepare(root, args.packet, {})
    elif args.action == "submit":
        result = submit(root)
    elif args.action == "worker":
        plan, _ = campaign.verify(root, runtime=True)
        if read_seal(root / "launcher_execution.json") != launcher(root, plan):
            raise ValueError("worker and submission identities differ")
        terminal = campaign.run(root, args.index)
        result = {
            k: terminal[k] for k in ("ready_for_comparison", "search_termination")
        }
    elif args.action == "inspect":
        result = {"archive": str(campaign.inspect(root))}
    else:
        if args.action == "report":
            report = campaign.report(root)
        else:
            plan, _ = campaign.verify(root)
            report, _ = campaign.comparison(root, plan)
        receipt = root / SUBMISSIONS / "manifest.json"
        result = {
            "status": report["status"],
            "arms": report["arms"],
            "jobs": read_seal(receipt)["jobs"] if receipt.is_file() else {},
            "ready_for_comparison": report["ready_for_comparison"],
            "routing": (report["comparison"] or {}).get("routing"),
            "report": str(root / "SUMMARY.md"),
        }
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
