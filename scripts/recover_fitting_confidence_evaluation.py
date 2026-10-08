#!/usr/bin/env python3
"""Score sealed M19 backends using their original code, without refitting."""

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import confidence_campaign as campaign
from autoformalism.fitting import public_fitting as public


def submit_job(directory: Path, stage: str, argv: list[str]) -> str:
    """Save intent before sbatch; an unknown reply never earns an automatic retry."""
    public._write(directory / f"{stage}.intent.json", {"argv": argv})
    try:
        reply = subprocess.run(argv, capture_output=True, text=True, timeout=45)
    except subprocess.TimeoutExpired:
        public._write(
            directory / f"{stage}.reply.json", {"status": "uncertain_timeout"}
        )
        raise ValueError(f"uncertain submission; inspect {directory}") from None
    public._write(
        directory / f"{stage}.reply.json",
        {
            "returncode": reply.returncode,
            "stdout": reply.stdout,
            "stderr": reply.stderr,
        },
    )
    job = reply.stdout.strip().split(";")[0]
    if reply.returncode or not job.isdecimal():
        raise ValueError(f"unconfirmed scheduler reply; inspect {directory}")
    (directory / f"{stage}.id").write_text(job + "\n")
    return job


def ready(root: Path, index: int | None = None) -> dict:
    """Require every relevant fit to be terminal before creating scoring folders."""
    plan, data = campaign.verify(root)
    if index is not None and not 0 <= index < len(plan["tasks"]):
        raise ValueError("invalid recovery task")
    indices = range(len(plan["tasks"])) if index is None else [index]
    folders = []
    for i in indices:
        folder = root / "results" / plan["tasks"][i]["task_id"]
        backend = read_seal(folder / "backend.json")
        finished = read_seal(folder / "fit" / "finished.json")
        expected = {
            "problem_sha256": public.content_sha256(
                campaign.training_problem(data, data["endpoints"][i]).model_dump(
                    mode="json"
                )
            ),
            "policy": plan["policy"],
        }
        if (
            backend != finished
            or backend.get("status") != "complete"
            or backend.get("identity") != expected
        ):
            raise ValueError("recovery requires an intact completed fit")
        folders.append(folder)
    for folder in folders:
        for stage in ("after_reliability", "selected"):
            (folder / f"evaluation-{stage}").mkdir(parents=True, exist_ok=True)
    return plan


def run(root: Path, index: int) -> dict:
    """The normal controller reuses finished.json; disallow incomplete inputs."""
    ready(root, index)
    return campaign.run_task(root, index)


def submit(root: Path, account: str = "bibo-delta-cpu") -> dict:
    """Journal scoring-only CPU jobs; never repeat an uncertain submission."""
    root = root.resolve()
    plan = ready(root)
    (root / "logs").mkdir(parents=True, exist_ok=True)
    directory = root / "submission" / "evaluation-recovery"
    repo = Path(campaign.__file__).resolve().parents[3]
    script = Path(__file__).read_text()
    environment = os.environ.get("PYTHONPATH", "")
    identity = {
        "plan_sha256": public.content_sha256(plan),
        "script_sha256": hashlib.sha256(script.encode()).hexdigest(),
        "python": sys.executable,
        "repo": str(repo),
        "pythonpath": environment,
        "account": account,
        "refitting_permitted": False,
    }
    with public._lock(directory):
        seal(directory / "identity.json", identity)
        saved_script = directory / "recover.py"
        if saved_script.exists() and saved_script.read_text() != script:
            raise ValueError("saved recovery script differs")
        saved_script.write_text(script)
        worker = directory / "worker.sh"
        worker.write_text(
            "#!/bin/bash\nset -euo pipefail\n"
            f"cd {shlex.quote(str(repo))}\n"
            f"export PYTHONPATH={shlex.quote(environment)}\n"
            "export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1\n"
            f"{shlex.quote(sys.executable)} {shlex.quote(str(saved_script))} "
            f'"$1" --root {shlex.quote(str(root))} '
            '--index "${SLURM_ARRAY_TASK_ID:-0}"\n'
        )
        jobs = {}
        for stage in ("run", "report"):
            options = [
                f"--account={account}",
                "--partition=cpu",
                "--nodes=1",
                "--ntasks=1",
                "--cpus-per-task=1",
                "--mem=8G",
                "--time=00:15:00",
                "--export=ALL",
                f"--job-name=confidence-score-{stage}",
                f"--output={root}/logs/recovery-{stage}-%A_%a.out",
                f"--error={root}/logs/recovery-{stage}-%A_%a.err",
            ]
            if stage == "run":
                options.append(f"--array=0-{len(plan['tasks']) - 1}%6")
            else:
                options.append(f"--dependency=afterany:{jobs['run']}")
            argv = ["sbatch", "--parsable", *options, str(worker), stage, "0"]
            intent, job_file = (
                directory / f"{stage}.intent.json",
                directory / f"{stage}.id",
            )
            if job_file.exists():
                if public._read(intent) != {"argv": argv}:
                    raise ValueError("recovery submission differs")
                job = job_file.read_text().strip()
                if not job.isdecimal():
                    raise ValueError("invalid saved scheduler job")
            else:
                if intent.exists():
                    raise ValueError(f"uncertain submission; inspect {directory}")
                job = submit_job(directory, stage, argv)
            jobs[stage] = job
        result = {"identity": identity, "jobs": jobs, "tasks": len(plan["tasks"])}
        public._write(directory / "manifest.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("submit", "run", "report"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--account", default="bibo-delta-cpu")
    args = parser.parse_args()
    if args.mode == "submit":
        result = submit(args.root, args.account)
    elif args.mode == "run":
        result = run(args.root, args.index)
    else:
        result = campaign.report(args.root)
    print(json.dumps(result, indent=2))
