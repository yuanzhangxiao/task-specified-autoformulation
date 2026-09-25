#!/usr/bin/env python3
"""Submit independent repaired/control CPU tasks and a dependent report."""

import argparse
import json
import os
from pathlib import Path

from autoformalism.fitting import public_fitting as public
from autoformalism.rebuttal import dalla_sign_diagnostic as io
from scripts.recover_review_continuation_submission import scheduler_record
from scripts.submit_shared_process_pilot import source_commit, submit_job


def submit(
    source: Path,
    decisions: Path,
    root: Path,
    *,
    site: str = "aces",
    adopt: dict | None = None,
) -> dict:
    """Freeze on the login node; numerical work runs only in CPU allocations."""
    root = root.resolve()
    commit = source_commit(io.REPO)
    if os.environ.get("AF_COMMIT", commit) != commit:
        raise ValueError("pinned commit differs")
    if not Path(os.environ["AF_PYTHON"]).is_file():
        raise ValueError("AF_PYTHON is missing")
    plan = io.freeze(source, decisions, root, site=site)
    identity = {"commit": commit, "plan_sha256": plan["artifact_sha256"], "site": site}
    adopt = adopt or {}
    if set(adopt) - {"fit", "report"}:
        raise ValueError("unknown adoption stage")
    os.environ.update(
        AF_REPO_ROOT=str(io.REPO), AF_OUTPUT_ROOT=str(root), AF_COMMIT=commit
    )
    with public._lock(root / "scheduler"):
        manifest = root / "submission_manifest.json"
        if manifest.exists():
            value = public._read(manifest)
            if value["identity"] != identity:
                raise ValueError("submission identity differs")
            return value
        directory = root / "submission-intent"
        directory.mkdir(exist_ok=True)
        frozen = directory / "identity.json"
        if frozen.exists() and public._read(frozen) != identity:
            raise ValueError("submission intent differs")
        public._write(frozen, identity)
        (root / "logs").mkdir(exist_ok=True)
        worker = io.REPO / "scripts/hpc/run_dalla_sign_diagnostic.sh"
        account = (
            os.environ.get("AF_CPU_ACCOUNT", "bibo-delta-cpu")
            if site == "delta"
            else os.environ.get("AF_ACCOUNT", "156264627414")
        )

        def queue(stage, options):
            opts = [
                "--kill-on-invalid-dep=yes",
                "--export=ALL",
                "--partition=cpu",
                "--account=" + account,
                "--nodes=1",
                "--ntasks=1",
                "--cpus-per-task=1",
                f"--job-name=dalla-manual-{stage}",
                f"--output={root}/logs/{stage}-%A_%a.out",
                f"--error={root}/logs/{stage}-%A_%a.err",
                *(["--constraint=projects&work"] if site == "delta" else []),
                *options,
            ]
            argv = ["sbatch", "--parsable", *opts, str(worker), stage, "0"]
            intent, ident = (
                directory / f"{stage}.intent.json",
                directory / f"{stage}.id",
            )
            if intent.exists():
                if public._read(intent) != {"argv": argv}:
                    raise ValueError("saved scheduler command differs")
                if ident.exists():
                    job = ident.read_text().strip()
                    if not job.isdecimal() or (stage in adopt and adopt[stage] != job):
                        raise ValueError("confirmed job differs")
                    return job
                if stage not in adopt:
                    raise ValueError(
                        f"Unconfirmed {stage}; inspect scheduler and receipts "
                        f"before --adopt {stage}=JOB_ID"
                    )
                job = adopt[stage]
                record = scheduler_record(job, argv, since=intent.stat().st_mtime)
                public._write(directory / f"{stage}.adoption.json", record)
                ident.write_text(job + "\n")
                return job
            if stage in adopt:
                raise ValueError("cannot adopt without saved intent")
            return submit_job(directory, stage, opts, worker, stage, 0)

        fit = queue("fit", ["--array=0-1%2", "--mem=16G", "--time=02:15:00"])
        report = queue(
            "report", [f"--dependency=afterany:{fit}", "--mem=4G", "--time=00:15:00"]
        )
        result = {
            "protocol": io.PROTOCOL,
            "identity": identity,
            "jobs": {"fit": fit, "report": report},
            "array_tasks": 2,
            "decision_source": io.LABEL,
            "gpus": 0,
            "live_llm_calls": 0,
            "test_data_opened": False,
        }
        public._write(manifest, result)
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--site", choices=("aces", "delta"), default="aces")
    parser.add_argument("--adopt", action="append", default=[])
    args = parser.parse_args()
    print(
        json.dumps(
            submit(
                args.source,
                args.decisions,
                args.root,
                site=args.site,
                adopt=dict(x.split("=", 1) for x in args.adopt),
            ),
            indent=2,
        )
    )
