#!/usr/bin/env python3
"""Run, exchange and inspect the six T2 fitting tasks, locally or on Delta CPUs."""

import argparse
import json
import os
from pathlib import Path

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import public_fitting as public
from autoformalism.research import t2_breadth_fitting as campaign
from scripts.submit_construction_fit_aces import (
    confirmed_job,
    file_hash,
    runtime_identity,
)
from scripts.submit_review_continuation import submit_job

REPO = Path(__file__).resolve().parents[1]
WORKER = "scripts/hpc/start_t2_breadth_fitting_delta.sh"
SUBMISSIONS = "submissions/cpu-t2-breadth-fit-1"
ENTRYPOINTS = (
    WORKER,
    "scripts/run_t2_breadth_fitting.py",
    "scripts/submit_construction_fit_aces.py",
    "scripts/submit_review_continuation.py",
)


def export_models(root: Path):
    """Return original equations and both retained parameter sets to construction."""
    summary = campaign.report(root)
    if summary["status"] != "complete":
        raise ValueError("T2 exchange requires all six verified terminal records")
    plan, data = campaign.verify(root, runtime=False)
    models = []
    for row in summary["rows"]:
        source = data["models"][row["branch"]]
        fits = {}
        for name in ("best_numerical", "best_sampled_admissible"):
            record = row[name]["endpoint"]
            fits[name] = (
                {
                    key: record[key]
                    for key in (
                        "parameters",
                        "parameter_sha256",
                        "training_nmse",
                        "target_nmse",
                        "domain",
                        "sampled_domain_passed",
                        "shifted_90g",
                        "extension_150g",
                        "own_depletion_rate",
                        "common_depletion_rate",
                    )
                }
                if record
                else None
            )
        models.append(
            {
                "task": row["task"],
                "nested_control": row["is_nested_control"],
                "source_handoff_sha256": source["receipt"],
                "model_sha256": source["model_sha256"],
                "problem": source["problem"],
                **fits,
            }
        )
    out = root / "fitted-models.json"
    seal(
        out,
        {
            "protocol": campaign.PROTOCOL,
            "plan_sha256": summary["plan_sha256"],
            "training_sha256": plan["training_sha256"],
            "prefix_policy": plan["prefix"],
            "initial_values": "trajectory:* parameters are training-only fitted roots",
            "prefix_forecast_evaluated": False,
            "validation_or_test_access": False,
            "models": models,
        },
    )
    return {"models": len(models), "file": str(out)}


def execution(root: Path, plan: dict):
    raw = os.environ.get("AF_T2_FIT_CONCURRENCY", "2")
    if raw not in ("1", "2"):
        raise ValueError("AF_T2_FIT_CONCURRENCY must be 1 or 2")
    return {
        "protocol": "t2-breadth-fit-delta-cpu-1",
        "repo": str(REPO),
        "root": str(root.resolve()),
        "plan_sha256": public.content_sha256(plan),
        "source_sha256": plan["source_sha256"],
        "runtime": runtime_identity(),
        "pythonpath": os.environ.get("PYTHONPATH", ""),
        "entrypoints": {n: file_hash(REPO / n) for n in ENTRYPOINTS},
        "account": os.environ.get("AF_ACCOUNT", "bibo-delta-cpu"),
        "array": f"0-{len(plan['tasks']) - 1}%{raw}",
        "wall": "02:00:00",
        "memory": "8G",
        "cpus_per_task": 1,
        "gpu_count": 0,
        "validation_or_test_access": False,
    }


def submit(root: Path):
    plan, _ = campaign.verify(root)
    identity = execution(root, plan)
    directory = root / SUBMISSIONS
    os.environ.update(
        AF_REPO_ROOT=str(REPO), AF_T2_BREADTH_FIT_ROOT=str(root.resolve())
    )
    with public._lock(directory):
        seal(root / "launcher_execution.json", identity)
        digest = public.content_sha256(identity)
        seal(directory / "identity.json", {"execution_sha256": digest})
        jobs = {}
        worker = REPO / WORKER
        (root / "logs").mkdir(exist_ok=True)
        for name in ("task", "report"):
            options = [
                "--kill-on-invalid-dep=yes",
                "--nodes=1",
                "--ntasks=1",
                "--export=ALL",
                f"--account={identity['account']}",
                "--partition=cpu",
                "--cpus-per-task=1",
                "--mem=8G",
                f"--job-name=t2-breadth-fit-{name}",
                f"--output={root}/logs/{name}-%A_%a.out",
                f"--error={root}/logs/{name}-%A_%a.err",
            ]
            if name == "task":
                options.extend(
                    [f"--array={identity['array']}", f"--time={identity['wall']}"]
                )
            else:
                options.extend(
                    [f"--dependency=afterany:{jobs['task']}", "--time=01:00:00"]
                )
            stage = "slurm-" + name
            argv = ["sbatch", "--parsable", *options, str(worker), stage, "0"]
            job = confirmed_job(directory, name, argv)
            if job is None:
                if (directory / "manifest.json").exists():
                    raise ValueError("submission manifest lacks a confirmed job")
                job = submit_job(directory, name, options, worker, stage, 0)
            jobs[name] = job
        result = {"execution_sha256": digest, "jobs": jobs, "tasks": plan["tasks"]}
        seal(directory / "manifest.json", result)
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "prepare",
            "submit",
            "task",
            "slurm-task",
            "slurm-report",
            "status",
            "report",
            "inspect",
            "export",
        ),
    )
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--index", type=int)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.command.startswith("slurm-"):
        plan, _ = campaign.verify(root)
        if read_seal(root / "launcher_execution.json") != execution(root, plan):
            raise ValueError("T2 Delta worker identity differs")
    if args.command == "prepare":
        if not args.archive:
            parser.error("prepare requires --archive")
        result = campaign.prepare(root, args.archive, campaign.Policy())
    elif args.command == "submit":
        result = submit(root)
    elif args.command in ("task", "slurm-task"):
        result = campaign.compact(campaign.run_task(root, args.index))
    elif args.command == "status":
        result = campaign.status(root)
    elif args.command == "inspect":
        result = {"archive": str(campaign.inspection(root))}
    elif args.command == "export":
        result = export_models(root)
    else:
        summary = campaign.report(root)
        result = {k: summary[k] for k in ("status", "fitted", "expected", "recorded")}
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
