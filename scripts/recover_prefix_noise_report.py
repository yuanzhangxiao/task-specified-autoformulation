#!/usr/bin/env python3
"""Submit one report-only M47 retry using the unchanged original source bundle.

Deliver alongside recover_prefix_noise_preparation.py outside the frozen bundle.
No preparation, fitting, inference or forecast operation is submitted.
"""

from __future__ import annotations

import argparse
import fcntl
import getpass
import json
import os
import re
import subprocess
import sys
from pathlib import Path

if __package__:
    from scripts import recover_prefix_noise_preparation as previous
else:
    # Resolve the delivered sibling even with Python's isolated (-I) mode.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import recover_prefix_noise_preparation as previous

RECOVERY = "submission/report-timeout-recovery"


def verify_chain(root: Path) -> tuple[dict, dict]:
    """Bind the retry to original inputs and the confirmed preparation recovery."""
    original = previous.verify_original(root)
    folder = root / previous.RECOVERY
    manifest = previous.read(folder / "manifest.json")
    identity = previous.read(folder / "identity.json")
    jobs = manifest["jobs"]
    if (
        manifest["identity"] != identity
        or manifest["original_jobs"] != original["jobs"]
        or manifest["array_tasks"] != 36
        or set(jobs) != set(previous.STAGES)
        or any(not re.fullmatch(r"[0-9]+", job) for job in jobs.values())
        or len(set(jobs.values())) != 3
        or identity["original_manifest_sha256"] != previous.digest(original)
        or identity["inputs_file_sha256"] != previous.file_digest(root / "inputs.json")
        or identity["bundle_manifest_sha256"]
        != previous.file_digest(Path(original["identity"]["repo"]) / "SHA256SUMS")
        or identity["helper_sha256"] != previous.file_digest(Path(previous.__file__))
    ):
        raise ValueError("preparation recovery identity differs")
    for stage in previous.STAGES:
        if previous.read(folder / f"{stage}.accepted.json") != {
            "job": jobs[stage]
        } or previous.read(folder / f"{stage}.intent.json") != {
            "argv": previous.command(
                root,
                original["identity"],
                jobs,
                stage,
                recovery=True,
            ),
        }:
            raise ValueError(f"preparation recovery {stage} receipt differs")
    return original, manifest


def check_accounting(output: str, jobs: dict) -> None:
    """Require all 36 successful array allocations and a timed-out report."""
    seen: set[int] = set()
    singles: set[str] = set()
    prefix = "prefix-noise-reliability-recovery-"
    for line in output.splitlines():
        fields = [value.strip() for value in line.split("|")]
        if len(fields) != 4:
            raise ValueError("unexpected accounting format")
        job, name, state, exit_code = fields
        if job.endswith((".batch", ".extern")):
            continue
        if job in (jobs["prepare"], jobs["report"]):
            stage = "prepare" if job == jobs["prepare"] else "report"
            expected = "COMPLETED" if stage == "prepare" else "TIMEOUT"
            if (
                stage in singles
                or name != prefix + stage
                or state != expected
                or (stage == "prepare" and exit_code != "0:0")
            ):
                raise ValueError(f"recovery {stage} not confirmed {expected}")
            singles.add(stage)
        else:
            indices = previous.array_indices(job, jobs["run"])
            if (
                seen & indices
                or name != prefix + "run"
                or state != "COMPLETED"
                or exit_code != "0:0"
            ):
                raise ValueError("recovery array is not entirely successful")
            seen.update(indices)
    if singles != {"prepare", "report"} or seen != set(range(36)):
        raise ValueError("incomplete recovery accounting; no report submitted")


def terminal_evidence(root: Path, jobs: dict) -> dict:
    """Check terminal accounting and result presence before creating a retry."""
    accounting = subprocess.run(
        [
            "sacct",
            "-X",
            "-n",
            "-P",
            "-j",
            ",".join(jobs.values()),
            "--starttime=1970-01-01",
            "--endtime=now",
            "--format=JobID%200,JobName%100,State%100,ExitCode",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    check_accounting(accounting.stdout, jobs)
    queue = subprocess.run(
        ["squeue", "--noheader", "--user", getpass.getuser(), "--format=%i"],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if any(
        line.strip().split("_", 1)[0] in jobs.values()
        for line in queue.stdout.splitlines()
    ):
        raise ValueError("a previous recovery job is still queued")
    plan = previous.read_seal(root / "plan.json")
    paths = [
        root / "results" / task["task_id"] / "result.json" for task in plan["tasks"]
    ]
    if any(
        not p.resolve().is_relative_to(root / "results") or not p.is_file()
        for p in paths
    ):
        raise ValueError("not all 36 frozen tasks have saved results")
    return {"sacct": accounting.stdout, "squeue": queue.stdout, "result_files": 36}


def recover(root: Path) -> dict:
    """Journal one CPU report; repeated invocations reuse the same job ID."""
    root = root.resolve(strict=True)
    directory = root / RECOVERY
    directory.mkdir(exist_ok=True)
    with (directory / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print("Verifying frozen M47 source and completed-run receipts...", flush=True)
        original, chain = verify_chain(root)
        identity = {
            "protocol": "prefix-noise-report-recovery-1",
            "original_manifest_sha256": previous.digest(original),
            "previous_recovery_sha256": previous.digest(chain),
            "helper_sha256": previous.file_digest(Path(__file__)),
            "report_time": "00:30:00",
            "fitting_submitted": False,
        }
        identity_path = directory / "identity.json"
        if identity_path.exists():
            if previous.read(identity_path) != identity:
                raise ValueError("report recovery identity differs")
        else:
            print("Checking all 36 runs completed and report timed out...", flush=True)
            evidence = terminal_evidence(root, chain["jobs"])
            previous.write(directory / "prior-accounting.json", evidence)
            previous.write(
                directory / "prior-summary.json", previous.read(root / "summary.json")
            )
            previous.write(identity_path, identity)
        original_identity = original["identity"]
        env = os.environ | {
            "AF_REPO_ROOT": original_identity["repo"],
            "AF_OUTPUT_ROOT": str(root),
            "AF_PYTHON": original_identity["python"],
            "AF_COMMIT": original_identity["commit"],
            "AF_CASADI_ROOT": chain["identity"]["casadi_root"],
            "PYTHONDONTWRITEBYTECODE": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
        # All previous jobs are terminal. Omit dependencies on aged scheduler IDs.
        argv = [
            "sbatch",
            "--parsable",
            f"--account={original_identity['account']}",
            "--partition=cpu",
            "--nodes=1",
            "--ntasks=1",
            "--cpus-per-task=1",
            "--mem=16G",
            "--time=00:30:00",
            "--export=ALL",
            "--job-name=prefix-noise-reliability-report-retry",
            f"--output={root}/logs/report-retry-%j.out",
            f"--error={root}/logs/report-retry-%j.err",
            str(Path(original_identity["repo"]) / previous.WORKER),
            "report",
        ]
        job = previous.submit_once(directory, "report", argv, env)
        result = {
            "identity": identity,
            "jobs": {"report": job},
            "previous_jobs": chain["jobs"],
            "gpus": 0,
            "live_llm_calls": 0,
            "fitting_submitted": False,
            "artifact": str(directory / "manifest.json"),
        }
        manifest = directory / "manifest.json"
        if not manifest.exists():
            previous.write(manifest, result)
        elif previous.read(manifest) != result:
            raise ValueError("report recovery manifest differs")
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    print(json.dumps(recover(parser.parse_args().root), indent=2))
