#!/usr/bin/env python3
"""Recover a rejected sign-review dependency using the unchanged pinned checkout.

Copy this file OUTSIDE AF_REPO_ROOT on ACES. Original scientific files, plans and
submission receipts stay unchanged. New scheduler receipts have their own folder.
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import importlib
import json
import os
import re
import shlex
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

TERMINAL = {
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "OUT_OF_MEMORY",
    "NODE_FAIL",
    "PREEMPTED",
    "BOOT_FAIL",
    "DEADLINE",
    "REVOKED",
}


def load_submitter(repo: Path):
    """Use scientific code and scheduler helpers from the original archive only."""
    sys.path[:0] = [str(repo), str(repo / "src")]
    module = importlib.import_module("scripts.submit_dalla_sign_repair")
    if module.io.REPO.resolve() != repo.resolve():
        raise ValueError("imports do not belong to the pinned checkout")
    return module


def canonical(argv: list[str]) -> list[str]:
    """Slurm may resolve sbatch to an absolute executable path."""
    return [Path(argv[0]).name, *argv[1:]] if argv else []


def without_dependencies(argv: list[str]) -> list[str]:
    """Only dependency flags may change between audited submission attempts."""
    return [a for a in argv if not a.startswith("--dependency=")]


def dependency_rejected(receipt: dict) -> bool:
    """Only an explicit dependency rejection can authorize a changed retry."""
    return (
        not receipt.get("stdout", "").strip()
        and "Batch job submission failed: Job dependency problem"
        in receipt.get("stderr", "")
        and "timed out" not in receipt.get("stderr", "").lower()
        and receipt.get("status") != "uncertain_timeout"
    )


def accounting(argv: list[str], since: float) -> list[dict]:
    """Find exact owned submission commands, including individual array tasks."""
    name = next(a.split("=", 1)[1] for a in argv if a.startswith("--job-name="))
    start = (datetime.fromtimestamp(since, UTC) - timedelta(days=1)).date()
    reply = subprocess.run(
        [
            "sacct",
            "-X",
            "-n",
            "-P",
            "-u",
            getpass.getuser(),
            "--name=" + name,
            "--starttime=" + start.isoformat(),
            "--format=JobIDRaw,User%80,JobName%80,State%40,ExitCode,SubmitLine%8192",
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    records = []
    for line in reply.stdout.splitlines():
        if not line.strip():
            continue
        fields = [s.strip() for s in line.split("|", 5)]
        if len(fields) != 6:
            raise ValueError("incomplete scheduler accounting reply")
        raw, user, job_name, state, code, command = fields
        match = re.fullmatch(r"(\d+)(?:_\d+)?", raw)
        if (
            match
            and user == getpass.getuser()
            and job_name == name
            and canonical(shlex.split(command)) == canonical(argv)
        ):
            records.append(
                {
                    "job": match[1],
                    "raw_job": raw,
                    "state": state,
                    "exit_code": code,
                    "submit_line": command,
                }
            )
    return records


def fit_finished(records: list[dict], job: str, tasks: int) -> bool:
    """Drop an afterany dependency only after all expected array tasks terminated."""
    states = {
        r["raw_job"]: r["state"].split()[0].rstrip("+")
        for r in records
        if r["job"] == job
    }
    return all(states.get(f"{job}_{i}") in TERMINAL for i in range(tasks))


def configure(module, repo: Path, root: Path, plan: dict) -> None:
    """Restore the same worker environment as the original launcher."""
    group = Path("/scratch/group/p.nairr260351.000/u.yx126462")
    os.environ.setdefault(
        "AF_VLLM_IMAGE", "/scratch/user/u.yx126462/containers/vllm-openai-v0.27.1.sif"
    )
    if not os.environ.get("AF_HF_HOME"):
        revision = plan["config"]["model_settings"]["model_revision"]
        for cache in (
            group / "huggingface-cache",
            Path("/scratch/user/u.yx126462/huggingface-cache"),
        ):
            if (
                cache / "hub/models--openai--gpt-oss-20b/snapshots" / revision
            ).is_dir():
                os.environ["AF_HF_HOME"] = str(cache)
                break
    for key in ("AF_PYTHON", "AF_VLLM_IMAGE", "AF_HF_HOME"):
        if not os.environ.get(key) or not Path(os.environ[key]).exists():
            raise ValueError(f"missing {key}")
    os.environ.setdefault("AF_COMPUTE_CACHE_ROOT", str(group / "sign-repair-cache"))
    os.environ.setdefault("AF_IPC_TMP_ROOT", f"/tmp/af-sign-{os.getuid()}")
    os.environ.update(
        AF_REPO_ROOT=str(repo),
        AF_OUTPUT_ROOT=str(root),
        AF_INPUTS=plan["source"],
        AF_COMMIT=module.source_commit(repo),
    )


def recover(root: Path, prepare_job: str) -> dict:
    """Resume only the rejected chain, preserving all uncertain submission attempts."""
    repo = Path(os.environ["AF_REPO_ROOT"]).resolve()
    root = root.resolve()
    module = load_submitter(repo)
    public, io = module.public, module.io
    plan = io.verify(root)
    identity = {
        "commit": module.source_commit(repo),
        "plan_sha256": plan["artifact_sha256"],
    }
    if os.environ.get("AF_COMMIT", identity["commit"]) != identity["commit"]:
        raise ValueError("AF_COMMIT differs from pinned archive")
    original = root / "submission-intent"
    if public._read(original / "identity.json") != identity:
        raise ValueError("original submission identity differs")
    configure(module, repo, root, plan)
    with public._lock(root / "scheduler"):
        manifest = root / "submission_manifest.json"
        if manifest.exists():
            result = public._read(manifest)
            if (
                result["identity"] != identity
                or result["jobs"]["prepare"] != prepare_job
            ):
                raise ValueError("completed submission differs")
            return result
        prepare = original / "prepare.intent.json"
        prepare_argv = public._read(prepare)["argv"]
        confirmed = (original / "prepare.id").read_text().strip()
        if confirmed != prepare_job:
            raise ValueError("prepare ID differs from the confirmed receipt")
        prep = module.scheduler_record(
            prepare_job, prepare_argv, since=prepare.stat().st_mtime
        )
        if prep["state"] != "COMPLETED":
            raise ValueError("recovery requires successfully completed preparation")
        if (original / "review.id").exists() or not dependency_rejected(
            public._read(original / "review.reply.json")
        ):
            raise ValueError("original review is not a confirmed dependency rejection")
        if any((original / f"{s}.intent.json").exists() for s in ("fit", "report")):
            raise ValueError(
                "later original submission exists; inspect before recovery"
            )

        worker = repo / "scripts/hpc/run_dalla_sign_repair_aces.sh"
        review_argv = public._read(original / "review.intent.json")["argv"]
        account = next(a for a in prepare_argv if a.startswith("--account="))

        def command(stage, options):
            return [
                "sbatch",
                "--parsable",
                "--kill-on-invalid-dep=yes",
                "--export=ALL",
                account,
                "--nodes=1",
                "--ntasks=1",
                f"--job-name=dalla-sign-{stage}",
                f"--output={root}/logs/{stage}-%A_%a.out",
                f"--error={root}/logs/{stage}-%A_%a.err",
                *options,
                str(worker),
                stage,
                "0",
            ]

        gpu = [
            "--partition=gpu",
            "--gres=gpu:h100:1",
            "--cpus-per-task=8",
            "--mem=64G",
            "--time=01:15:00",
            "--signal=B:TERM@300",
        ]
        if review_argv != command(
            "review", [f"--dependency=afterok:{prepare_job}", *gpu]
        ):
            raise ValueError(
                "rejected command differs from the original review contract"
            )
        if accounting(review_argv, (original / "review.intent.json").stat().st_mtime):
            raise ValueError(
                "an original review job exists; inspect instead of resubmitting"
            )

        directory = root / "submission-recovery-1"
        directory.mkdir(exist_ok=True)
        recovery_identity = {
            **identity,
            "prepare_job": prepare_job,
            "recovery_script_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
        }
        receipt = directory / "identity.json"
        if receipt.exists() and public._read(receipt) != recovery_identity:
            raise ValueError("recovery identity changed")
        public._write(receipt, recovery_identity)
        if not (directory / "prepare-verification.json").exists():
            public._write(directory / "prepare-verification.json", prep)
        submitted = {}

        def queue(stage, options):
            # A changed dependency gets a new attempt, never an overwritten receipt.
            for attempt in range(3):
                key = f"{stage}-{attempt}"
                intent = directory / f"{key}.intent.json"
                ident = directory / f"{key}.id"
                argv = command(stage, options())
                if intent.exists():
                    saved = public._read(intent)["argv"]
                    if without_dependencies(saved) != without_dependencies(argv):
                        raise ValueError("recovery command changed")
                    argv = saved
                    if ident.exists():
                        job = ident.read_text().strip()
                        if not job.isdecimal():
                            raise ValueError("invalid saved recovery ID")
                        submitted[stage] = (job, argv, intent.stat().st_mtime)
                        return job
                else:
                    try:
                        job = module.submit_job(
                            directory, key, argv[2:-3], worker, stage, 0
                        )
                    except ValueError:
                        # Inspect receipts and accounting; never retry blindly.
                        pass
                    else:
                        submitted[stage] = (job, argv, intent.stat().st_mtime)
                        return job
                records = accounting(argv, intent.stat().st_mtime)
                matches = sorted({r["job"] for r in records})
                if len(matches) == 1:
                    job = matches[0]
                    public._write(
                        directory / f"{key}.adoption.json", {"records": records}
                    )
                    ident.write_text(job + "\n")
                    submitted[stage] = (job, argv, intent.stat().st_mtime)
                    return job
                if matches:
                    raise ValueError(f"multiple matching {stage} jobs: {matches}")
                reply = directory / f"{key}.reply.json"
                if (
                    reply.exists()
                    and dependency_rejected(public._read(reply))
                    and command(stage, options()) != argv
                ):
                    continue
                raise ValueError(
                    f"{stage} submission remains unconfirmed; keep receipts and rerun "
                    "this same recovery command after accounting updates. "
                    "No duplicate sent."
                )
            raise ValueError(f"{stage}: dependency recovery attempt limit reached")

        review = queue("review", lambda: gpu)

        def fit_options():
            job, argv, since = submitted["review"]
            record = module.scheduler_record(job, argv, since=since)
            deps = (
                []
                if record["state"] == "COMPLETED"
                else [f"--dependency=afterok:{job}"]
            )
            return [
                *deps,
                "--partition=cpu",
                "--cpus-per-task=1",
                f"--array=0-{len(plan['rows']) - 1}%3",
                "--mem=16G",
                "--time=04:30:00",
            ]

        fit = queue("fit", fit_options)

        def report_options():
            job, argv, since = submitted["fit"]
            finished = fit_finished(accounting(argv, since), job, len(plan["rows"]))
            deps = [] if finished else [f"--dependency=afterany:{job}"]
            return [
                *deps,
                "--partition=cpu",
                "--cpus-per-task=1",
                "--mem=4G",
                "--time=00:15:00",
            ]

        report = queue("report", report_options)
        settings = plan["config"]["model_settings"]
        result = {
            "protocol": plan["protocol"],
            "identity": identity,
            "jobs": {
                "prepare": prepare_job,
                "review": review,
                "fit": fit,
                "report": report,
            },
            "array_tasks": len(plan["rows"]),
            "review_gpus": 1,
            "fit_gpus": 0,
            "maximum_llm_calls": sum(
                bool(r["review_context"]["eligible_slots"]) for r in plan["rows"]
            )
            * min(
                settings["maximum_requests"],
                settings["attempts_per_step"]
                * (2 if plan["protocol"].endswith("-2") else 1),
            ),
            "test_data_opened": False,
            "recovery": recovery_identity,
        }
        public._write(manifest, result)
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--prepare-job", required=True)
    args = parser.parse_args()
    print(json.dumps(recover(args.root, args.prepare_job), indent=2))
