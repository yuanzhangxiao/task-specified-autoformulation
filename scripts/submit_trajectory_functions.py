"""Queue one 20B generation job, its paired CPU fits and a final validation job."""

from __future__ import annotations

import argparse
import getpass
import json
import os
import re
import subprocess
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

from autoformalism.fitting.public_fitting import _lock
from scripts import submit_t2_breadth as gpu
from scripts import t2_breadth_scheduler as scheduler
from scripts.submit_structural_revision_aces import _digest, _read, _seal, _submit_job
from scripts.t2_breadth import ENTRYPOINTS as GPU_ENTRYPOINTS
from scripts.t2_breadth import REPO

ENTRYPOINTS = (
    *GPU_ENTRYPOINTS,
    "scripts/trajectory_functions.py",
    "scripts/trajectory_function_intake.py",
    "scripts/trajectory_function_prompts.py",
    "scripts/trajectory_function_report.py",
    "scripts/submit_trajectory_functions.py",
    "scripts/hpc/start_trajectory_functions_aces.sh",
    "scripts/hpc/run_trajectory_functions_cpu.sh",
)
FIT_SLOTS = 36
STAGES = ("fit", "validate")
JOB_NAMES = {"fit": "phasec-traj-fit", "validate": "phasec-traj-val"}


def _recover(directory: Path, job_name: str, tag: str) -> str | None:
    """Resolve a lost ACES acknowledgment using a campaign- and wave-specific tag."""
    user = getpass.getuser()
    intent_time = (directory / "propose.intent.json").stat().st_mtime
    date = (datetime.fromtimestamp(intent_time, UTC) - timedelta(days=1)).date()
    commands = (
        ["squeue", "-h", "-u", user, "-n", job_name, "-o", "%F|%k|%j"],
        [
            "sacct",
            "-X",
            "-n",
            "-P",
            "-u",
            user,
            "-S",
            str(date),
            "--name=" + job_name,
            "--format=JobID%128,Comment%128,JobName%80",
        ],
    )
    jobs, errors = set(), []
    for argv in commands:
        try:
            output = scheduler._query(argv)
        except (ValueError, OSError, subprocess.SubprocessError) as exc:
            errors.append(str(exc))
            continue
        for line in output.splitlines():
            fields = [v.strip() for v in line.split("|")]
            if len(fields) != 3 or fields[1:] != [tag, job_name]:
                continue
            # Queue %F is the array master. Accounting reports e.g. 123_0
            # or 123_[0-35] until the last pending array task is instantiated.
            match = re.fullmatch(r"(\d+)(?:_(?:\d+|\[[0-9,%\-]+\]))?", fields[0])
            if not match:
                raise ValueError(f"Unrecognized Slurm array identity {fields[0]}")
            jobs.add(match[1])
    if len(jobs) == 1:
        return jobs.pop()
    if jobs or errors:
        raise ValueError(f"Unconfirmed scheduler records for {tag}: {jobs}, {errors}")
    reply = directory / "propose.reply.json"
    acknowledged = reply.stat().st_mtime if reply.is_file() else intent_time + 60
    if time.time() - acknowledged < scheduler.CLOCK_SKEW_SECONDS:
        raise ValueError(f"Awaiting accounting for {tag}; repeat the run command")
    return None


def _stage(
    wave: Path,
    stage: str,
    predecessor: str | None,
    root: Path,
    plan_hash: str,
    *,
    resume: bool,
) -> str:
    """Reuse an accepted stage and recover ambiguous submissions before retrying."""
    if predecessor is not None and not predecessor.isdecimal():
        raise ValueError("Expected a numeric Slurm job ID")
    name = JOB_NAMES[stage]
    attempts = sorted((wave / stage).glob("attempt-*"))
    directory = attempts[-1] if attempts else None
    if directory is not None:
        manifest = directory / "manifest.json"
        if manifest.is_file():
            return _read(manifest)["job"]
        intent = directory / "propose.intent.json"
        if intent.is_file():
            argv = json.loads(intent.read_text())["argv"]
            tag = scheduler._value(argv, "comment")
            receipt = directory / "propose.receipt.json"
            job = (
                json.loads(receipt.read_text())["job"]
                if receipt.is_file()
                else _recover(directory, name, tag)
            )
            if job is not None:
                if not job.isdecimal():
                    raise ValueError("Invalid CPU scheduler receipt")
                return _seal(manifest, {"job": job, "tag": tag})["job"]
            if not resume:
                raise ValueError(f"No job found for {tag}; use resume to retry")
    directory = wave / stage / f"attempt-{len(attempts) + 1:04d}"
    directory.mkdir(parents=True, exist_ok=False)
    tag = "tf-" + _digest([str(directory), plan_hash])[:24]
    dependency = "afterok" if stage == "fit" else "afterany"
    argv = [
        "sbatch",
        "--parsable",
        "--nodes=1",
        "--ntasks=1",
        "--export=ALL",
        "--partition=cpu",
        "--cpus-per-task=1",
        "--mem=8G",
        "--exclude=ac042",
        "--account=" + os.environ.get("AF_ACCOUNT", "156264627414"),
        "--kill-on-invalid-dep=yes",
        f"--job-name={name}",
        f"--comment={tag}",
        f"--output={root}/logs/{stage}-%A_%a.out",
        f"--error={root}/logs/{stage}-%A_%a.err",
        "--time=00:20:00" if stage == "fit" else "--time=01:00:00",
    ]
    if stage == "fit":
        argv.append(f"--array=0-{FIT_SLOTS - 1}%8")
    if predecessor is not None:
        argv.append(f"--dependency={dependency}:{predecessor}")
    argv.extend([str(REPO / "scripts/hpc/run_trajectory_functions_cpu.sh"), stage])
    try:
        job = _submit_job(directory, argv)
    except ValueError as exc:
        job = _recover(directory, name, tag)
        if job is None:
            raise ValueError(f"Unconfirmed {stage} submission in {directory}") from exc
    return _seal(directory / "manifest.json", {"job": job, "tag": tag})["job"]


def submit(root: Path, *, resume: bool = False) -> dict:
    """Reserve the chain once and resume terminal incomplete waves on request."""
    from scripts import trajectory_functions as campaign

    root = root.resolve()
    plan, _ = campaign.load(root)
    if (
        plan["protocol"] != gpu.TRAJECTORY_FUNCTIONS
        or plan["maximum_fit_slots"] != FIT_SLOTS
        or plan["maximum_generation_requests"] != 24
    ):
        raise ValueError("Expected the fixed 36-fit, 24-request trajectory pilot")
    with _lock(root / "chain-lock"):
        status = campaign.status(root)
        if status["status"] == "complete":
            return {"status": "complete", "results": str(root)}
        generated = gpu.submit(root, resume=resume)
        generation = generated.get("job")
        if generation is None and generated.get("status") != "already_complete":
            return generated
        os.environ.update(AF_REPO_ROOT=str(REPO), AF_OUTPUT_ROOT=str(root))
        base = root / "cpu-submissions"
        waves = sorted(base.glob("wave-*"))
        wave = waves[-1] if waves else None
        if wave and generation is not None and (wave / "generation.json").is_file():
            old_generation = _read(wave / "generation.json")["job"]
            if old_generation != generation:
                # A resumed generation has a new Slurm ID. The older array's
                # afterok dependency cannot follow that replacement job.
                wave = None
        if wave and (wave / "manifest.json").is_file():
            prior = _read(wave / "manifest.json")
            if prior["plan_sha256"] != plan["artifact_sha256"]:
                raise ValueError("Submission belongs to a different plan")
            if not resume or not gpu._finished(prior["jobs"]["validate"]):
                return prior
            wave = None
        if wave is None:
            wave = base / f"wave-{len(waves) + 1:04d}"
            wave.mkdir(parents=True, exist_ok=False)
        # Preserve the original dependency if generation finishes between
        # submission of the array and reconciliation of its acknowledgment.
        binding = wave / "generation.json"
        if binding.exists():
            generation = _read(binding)["job"]
        else:
            _seal(binding, {"job": generation, "plan_sha256": plan["artifact_sha256"]})
        jobs = {"generate": generation}
        predecessor = generation
        for stage in STAGES:
            job = _stage(
                wave, stage, predecessor, root, plan["artifact_sha256"], resume=resume
            )
            jobs[stage] = predecessor = job
        return _seal(
            wave / "manifest.json",
            {
                "status": "submitted",
                "jobs": jobs,
                "plan_sha256": plan["artifact_sha256"],
            },
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = submit(args.root, resume=args.resume)
    print(json.dumps(result, indent=2))
    if result.get("status") not in {"complete", "submitted"}:
        raise SystemExit(75)
