"""Submit or safely resume the T2 breadth pilot on a single two-H100 allocation."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from autoformalism.fitting.public_fitting import _lock
from scripts import submit_meal_insulin_revision_aces as aces
from scripts import t2_breadth as campaign
from scripts.submit_structural_revision_aces import (
    _digest,
    _read,
    _seal,
    _submit_job,
    file_hash,
)


def execution(plan, image, snapshot):
    values = aces.execution_identity(
        plan, image, snapshot, entrypoints=campaign.ENTRYPOINTS
    )
    return {
        **values,
        "protocol": "t2-three-origin-aces-1",
        "entrypoint_sha256": plan["entrypoints"],
        "generation_settings": plan["settings"],
        "maximum_output_tokens": plan["settings"]["max_output_tokens"],
        "maximum_generation_requests": campaign.MAX_CALLS,
        "maximum_charged_tokens": campaign.MAX_CALLS * 32768,
        "worker_seconds": 6600,
        "scheduler_walltime_seconds": 7200,
    }


def verify(root, check_assets=False):
    plan, _ = campaign.load(root)
    record = _read(root / "execution.json")
    image, snapshot = record["image_sha256"], record["snapshot"]
    if check_assets:
        asset, snapshot = aces._assets()
        image = file_hash(asset)
    value = execution(plan, image, snapshot)
    if record != {**value, "artifact_sha256": _digest(value)}:
        raise ValueError("T2 breadth execution identity differs")
    return record


def bind(root, runtime, endpoint):
    record = aces.checked_server(verify(root), runtime, endpoint)
    return _seal(root / "runtime" / ("bound-" + runtime.name), record)


def _finished(job):
    reply = subprocess.run(
        ["sacct", "-X", "-n", "-P", "-j", job, "--format=JobIDRaw,State"],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    rows = [
        line.split("|")[1].split()[0]
        for line in reply.stdout.splitlines()
        if line.split("|")[0] == job
    ]
    return len(rows) == 1 and rows[0] in {
        "COMPLETED",
        "FAILED",
        "TIMEOUT",
        "CANCELLED",
        "PREEMPTED",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
    }


def _manifest(wave, identity, job):
    if not isinstance(job, str) or not job.isdecimal():
        raise ValueError("Confirmed numeric Slurm job ID required")
    return _seal(
        wave / "manifest.json",
        {
            "execution_sha256": identity["artifact_sha256"],
            "job": job,
            "maximum_generation_requests": campaign.MAX_CALLS,
        },
    )


def submit(root, resume=False):
    root = root.resolve()
    with _lock(root / "submission-lock"):
        plan, _ = campaign.load(root)
        if len(campaign._rows(root, plan)) == 15:
            return {"status": "already_complete", "results": str(root)}
        base = root / "submissions"
        waves = sorted(base.glob("wave-*"))
        if waves:
            last = waves[-1]
            identity = verify(root)
            # A crash after Slurm confirmed the job may precede the manifest.
            if (last / "propose.receipt.json").is_file():
                receipt = json.loads((last / "propose.receipt.json").read_text())
                _manifest(last, identity, receipt["job"])
            if not (last / "manifest.json").is_file():
                raise ValueError(
                    f"Scheduler acknowledgment unconfirmed. Check the queue and {last}"
                )
            prior = _read(last / "manifest.json")
            if prior["execution_sha256"] != identity["artifact_sha256"]:
                raise ValueError("Submission belongs to a different execution")
            if not resume or not _finished(prior["job"]):
                return prior
        image, snapshot = aces._assets()
        identity = _seal(
            root / "execution.json", execution(plan, file_hash(image), snapshot)
        )
        wave = base / f"wave-{len(waves) + 1:04d}"
        wave.mkdir(parents=True, exist_ok=False)
        (root / "logs").mkdir(exist_ok=True)
        os.environ.update(AF_REPO_ROOT=str(campaign.REPO), AF_OUTPUT_ROOT=str(root))
        argv = [
            "sbatch",
            "--parsable",
            "--nodes=1",
            "--ntasks=1",
            "--export=ALL",
            "--account=" + os.environ.get("AF_ACCOUNT", "156264627414"),
            "--partition=gpu",
            "--gres=gpu:h100:2",
            "--cpus-per-task=16",
            "--mem=128G",
            "--time=02:00:00",
            "--signal=B:TERM@300",
            "--exclude=ac042",
            "--job-name=phasec-t2-breadth",
            f"--output={root}/logs/propose-%j.out",
            f"--error={root}/logs/propose-%j.err",
            str(campaign.REPO / "scripts/hpc/run_meal_insulin_revision_aces.sh"),
            "t2-breadth",
        ]
        job = _submit_job(wave, argv)
        return _manifest(wave, identity, job)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--check-assets", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            verify(args.root, args.check_assets)
            if args.verify_only
            else submit(args.root, args.resume),
            indent=2,
        )
    )
