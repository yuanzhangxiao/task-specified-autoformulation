#!/usr/bin/env python3
"""Recover M47 reporting as per-task audits followed by one exact reduction.

Upload with both earlier recovery helpers and the sharded reporter, outside the
original source bundle. All scientific execution continues to use that bundle.
"""

from __future__ import annotations

import argparse
import fcntl
import getpass
import json
import os
import subprocess
import sys
from pathlib import Path

if __package__:
    from scripts import recover_prefix_noise_report as prior
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import recover_prefix_noise_report as prior

previous = prior.previous
RECOVERY = "submission/sharded-report-v1"
FILES = (
    "recover_prefix_noise_sharded.py",
    "recover_prefix_noise_report.py",
    "recover_prefix_noise_preparation.py",
    "report_prefix_noise_sharded.py",
    "run_prefix_noise_sharded.sh",
)


def stopped_retry(root: Path, chain: dict, retry: dict) -> dict:
    """Require completed fitting and the terminal thirty-minute report timeout."""
    evidence = prior.terminal_evidence(root, chain["jobs"])
    job = retry["jobs"]["report"]
    output = subprocess.run(
        [
            "sacct", "-X", "-n", "-P", "-j", job,
            "--starttime=1970-01-01", "--endtime=now",
            "--format=JobID%200,JobName%100,State%100,ExitCode",
        ],
        check=True, capture_output=True, text=True, timeout=60,
    ).stdout
    rows = [line.strip().split("|") for line in output.splitlines() if line.strip()]
    if len(rows) != 1 or rows[0][:3] != [
        job, "prefix-noise-reliability-report-retry", "TIMEOUT",
    ] or len(rows[0]) != 4:
        raise ValueError("thirty-minute report retry is not confirmed TIMEOUT")
    queued = subprocess.run(
        ["squeue", "--noheader", "--user", getpass.getuser(), "--format=%i"],
        check=True, capture_output=True, text=True, timeout=60,
    ).stdout
    if job in {line.strip().split("_", 1)[0] for line in queued.splitlines()}:
        raise ValueError("report retry is still queued")
    return evidence | {"retry_sacct": output, "retry_squeue": queued}


def recover(root: Path) -> dict:
    """Journal one validation array and reducer, with no new fitting requests."""
    root = root.resolve(strict=True)
    helper = Path(__file__).resolve().parent
    directory = root / RECOVERY
    directory.mkdir(exist_ok=True)
    with (directory / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(
            "Verifying original experiment and previous reporting receipts...",
            flush=True,
        )
        original, chain = prior.verify_chain(root)
        # Require confirmed receipts before invoking the old idempotent helper;
        # it must never create a replacement thirty-minute job here.
        for name in ("identity.json", "report.accepted.json", "report.intent.json",
                     "manifest.json"):
            if not (root / prior.RECOVERY / name).is_file():
                raise ValueError("missing confirmed report-only retry receipt")
        retry = prior.recover(root)
        identity = {
            "protocol": "prefix-noise-sharded-report-recovery-1",
            "original_manifest_sha256": previous.digest(original),
            "previous_retry_sha256": previous.digest(retry),
            "helpers": {name: previous.file_digest(helper / name) for name in FILES},
            "helper_directory": str(helper),
            "concurrency": 2,
            "check_time": "01:00:00",
            "reduce_time": "01:00:00",
            "fitting_submitted": False,
        }
        identity_path = directory / "identity.json"
        if identity_path.exists():
            if previous.read(identity_path) != identity:
                raise ValueError("sharded recovery identity differs")
        else:
            print("Checking saved tasks and terminal reporting jobs...", flush=True)
            evidence = stopped_retry(root, chain, retry)
            previous.write(directory / "prior-accounting.json", evidence)
            previous.write(
                directory / "prior-summary.json", previous.read(root / "summary.json")
            )
            previous.write(identity_path, identity)
        source = original["identity"]
        env = os.environ | {
            "AF_REPO_ROOT": source["repo"],
            "AF_OUTPUT_ROOT": str(root),
            "AF_PYTHON": source["python"],
            "AF_COMMIT": source["commit"],
            "AF_CASADI_ROOT": chain["identity"]["casadi_root"],
            "AF_REPORT_HELPER_ROOT": str(helper),
            "PYTHONDONTWRITEBYTECODE": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
        jobs: dict[str, str] = {}
        for stage in ("check", "reduce"):
            argv = [
                "sbatch", "--parsable", f"--account={source['account']}",
                "--partition=cpu", "--nodes=1", "--ntasks=1", "--cpus-per-task=1",
                "--mem=16G", f"--time={identity[stage + '_time']}", "--export=ALL",
                "--kill-on-invalid-dep=yes",
                f"--job-name=prefix-noise-report-{stage}",
                f"--output={root}/logs/report-{stage}-%A_%a.out",
                f"--error={root}/logs/report-{stage}-%A_%a.err",
            ]
            if stage == "check":
                argv += ["--array=0-35%2"]
            else:
                argv += [f"--dependency=afterok:{jobs['check']}"]
            argv += [str(helper / "run_prefix_noise_sharded.sh"), stage]
            jobs[stage] = previous.submit_once(directory, stage, argv, env)
        result = {
            "identity": identity, "jobs": jobs, "previous_jobs": retry["jobs"],
            "array_tasks": 36, "gpus": 0, "live_llm_calls": 0,
            "fitting_submitted": False,
            "artifact": str(directory / "manifest.json"),
        }
        manifest = directory / "manifest.json"
        if manifest.exists() and previous.read(manifest) != result:
            raise ValueError("sharded reporting manifest differs")
        if not manifest.exists():
            previous.write(manifest, result)
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    print(json.dumps(recover(parser.parse_args().root), indent=2))
