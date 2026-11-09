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
from scripts import t2_breadth_scheduler as scheduler
from scripts.submit_structural_revision_aces import (
    _digest,
    _read,
    _seal,
    _submit_job,
    file_hash,
)


def execution(plan, image, snapshot):
    values = aces.execution_identity(
        plan, image, snapshot, entrypoints=tuple(plan["entrypoints"])
    )
    return {
        **values,
        "protocol": "t2-fitted-residual-aces-1"
        if plan["protocol"] == campaign.fitted.PROTOCOL
        else "t2-three-origin-aces-2",
        "entrypoint_sha256": plan["entrypoints"],
        "generation_settings": plan["settings"],
        "maximum_output_tokens": plan["settings"]["max_output_tokens"],
        "maximum_generation_requests": plan["maximum_generation_requests"],
        "maximum_charged_tokens": plan["maximum_generation_requests"] * 32768,
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
            "maximum_generation_requests": identity["maximum_generation_requests"],
        },
    )


def scheduler_status(root: Path) -> dict:
    """Inspect both the original acknowledgment and the most recent deeper wave."""
    observed = {"predecessor": scheduler.predecessor(root), "current": None}
    waves = sorted((root / "submissions").glob("wave-*"))
    if waves:
        last = waves[-1]
        manifest = last / "manifest.json"
        if manifest.exists():
            observed["current"] = _read(manifest)
        else:
            try:
                observed["current"] = scheduler.reconcile(root, last)
            except (ValueError, OSError, subprocess.SubprocessError) as exc:
                observed["current"] = {
                    "status": "acknowledgment_pending",
                    "lookup": str(exc),
                }
            reply = last / "propose.reply.json"
            if reply.exists():
                observed["scheduler_reply"] = json.loads(reply.read_text())
    return observed


def submit(root, resume=False):
    root = root.resolve()
    with _lock(root / "submission-lock"):
        plan, _ = campaign.load(root)
        if len(campaign._rows(root, plan)) == len(campaign._names(plan)):
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
            manifest = last / "manifest.json"
            if not manifest.is_file():
                try:
                    observation = scheduler.reconcile(root, last)
                except (ValueError, OSError, subprocess.SubprocessError) as exc:
                    return {
                        "status": "acknowledgment_pending",
                        "wave": str(last),
                        "lookup": str(exc),
                        "next_action": "run",
                    }
                if observation["status"] == "confirmed":
                    _manifest(last, identity, observation["job"])
                    _seal(last / "adoption.json", observation)
                elif not resume:
                    return {**observation, "next_action": "resume", "wave": str(last)}
            if manifest.exists():
                prior = _read(manifest)
                if prior["execution_sha256"] != identity["artifact_sha256"]:
                    raise ValueError("Submission belongs to a different execution")
                if not resume or not _finished(prior["job"]):
                    return prior
        earlier = (
            scheduler.predecessor(root)
            if not (root / "legacy/index.json").exists()
            else None
        )
        image, snapshot = aces._assets()
        identity = _seal(
            root / "execution.json", execution(plan, file_hash(image), snapshot)
        )
        wave = base / f"wave-{len(waves) + 1:04d}"
        wave.mkdir(parents=True, exist_ok=False)
        (root / "logs").mkdir(exist_ok=True)
        os.environ.update(AF_REPO_ROOT=str(campaign.REPO), AF_OUTPUT_ROOT=str(root))
        fitted_run = plan["protocol"] == campaign.fitted.PROTOCOL
        job_name = "phasec-t2-fitted" if fitted_run else "phasec-t2-breadth"
        tag = "t2-fitted" if fitted_run else "t2-depth3"
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
            f"--job-name={job_name}",
            "--dependency=singleton",
            f"--comment={tag}-{identity['artifact_sha256'][:12]}-{wave.name}",
            f"--output={root}/logs/propose-%j.out",
            f"--error={root}/logs/propose-%j.err",
            str(campaign.REPO / "scripts/hpc/run_meal_insulin_revision_aces.sh"),
            "t2-breadth",
        ]
        if earlier:
            _seal(wave / "predecessor-check.json", earlier)
            if earlier.get("state") in {
                "PENDING",
                "RUNNING",
                "CONFIGURING",
                "COMPLETING",
                "SUSPENDED",
                "RESIZING",
            }:
                job = earlier["job"]
                if not job.isdecimal():
                    raise ValueError("Expected a numeric predecessor job ID")
                pos = argv.index("--dependency=singleton")
                argv[pos] += ",afterany:" + job
        try:
            job = _submit_job(wave, argv)
        except ValueError as exc:
            # The site wrapper sometimes times out after the controller accepted
            # the job. Preserve sbatch's actual error even if accounting lags.
            reply_path = wave / "propose.reply.json"
            reply = json.loads(reply_path.read_text()) if reply_path.exists() else {}
            try:
                observed = scheduler.reconcile(root, wave)
            except (ValueError, OSError, subprocess.SubprocessError) as lookup:
                return {
                    "status": "acknowledgment_pending",
                    "wave": str(wave),
                    "scheduler_reply": reply,
                    "lookup": str(lookup),
                    "next_action": "Run the launcher again to reconcile this intent.",
                }
            if observed["status"] != "confirmed":
                return {
                    **observed,
                    "submission_error": str(exc),
                    "next_action": "resume",
                }
            job = observed["job"]
            _seal(wave / "adoption.json", observed)
        result = _manifest(wave, identity, job)
        return {**result, "predecessor": earlier} if earlier else result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--check-assets", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    if args.status:
        result = scheduler_status(args.root)
    elif args.verify_only:
        result = verify(args.root, args.check_assets)
    else:
        result = submit(args.root, args.resume)
    print(json.dumps(result, indent=2))
    if not (
        args.status
        or args.verify_only
        or result.get("job")
        or result.get("status") == "already_complete"
    ):
        raise SystemExit(75)
