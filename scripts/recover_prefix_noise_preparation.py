#!/usr/bin/env python3
"""Recover an M47 preparation timeout without replacing its frozen experiment.

Standalone standard-library helper: upload outside the original source bundle.
Only a timed-out preparation with an entirely unstarted, cancelled array qualifies.
"""

from __future__ import annotations

import argparse
import fcntl
import getpass
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

STAGES = ("prepare", "run", "report")
WORKER = "scripts/hpc/run_phase_c_prefix_noise_reliability_delta.sh"
RECOVERY = "submission/prepare-timeout-recovery"


def digest(value: object) -> str:
    """Match the canonical JSON convention of the original campaign."""
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def file_digest(path: Path) -> str:
    """Hash a file without changing it."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> dict:
    """Read a plain JSON receipt."""
    return json.loads(path.read_text())


def read_seal(path: Path) -> dict:
    """Verify an existing sealed artifact without importing fitting code."""
    payload = read(path)
    if digest(payload["value"]) != payload["sha256"]:
        raise ValueError(f"sealed artifact differs: {path}")
    return payload["value"]


def write(path: Path, value: dict) -> None:
    """Atomically persist a receipt before any subsequent scheduler action."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def command(
    root: Path,
    identity: dict,
    jobs: dict,
    stage: str,
    *,
    recovery: bool,
) -> list[str]:
    """Reproduce the original stage contract, changing only recovery scheduling."""
    stem = "recovery-" if recovery else ""
    duration = {
        "prepare": "00:30:00" if recovery else identity["prepare_time"],
        "run": identity["run_time"],
        "report": "00:05:00",
    }[stage]
    argv = [
        "sbatch",
        "--parsable",
        f"--account={identity['account']}",
        "--partition=cpu",
        "--nodes=1",
        "--ntasks=1",
        "--cpus-per-task=1",
        "--mem=16G",
        f"--time={duration}",
        "--export=ALL",
        "--kill-on-invalid-dep=yes",
        f"--job-name=prefix-noise-reliability-{stem}{stage}",
        f"--output={root}/logs/{stem}{stage}-%A_%a.out",
        f"--error={root}/logs/{stem}{stage}-%A_%a.err",
    ]
    if stage == "run":
        argv += [
            f"--dependency=afterok:{jobs['prepare']}",
            f"--array=0-35%{identity['concurrency']}",
        ]
    elif stage == "report":
        argv += [f"--dependency=afterany:{jobs['prepare']}:{jobs['run']}"]
    return [*argv, str(Path(identity["repo"]) / WORKER), stage]


def verify_original(root: Path) -> dict:
    """Require the original M47 bundle and its complete three-stage receipts."""
    identity = read_seal(root / "submission/identity.json")
    manifest = read(root / "submission_manifest.json")
    plan = read_seal(root / "plan.json")
    data = read_seal(root / "inputs.json")
    jobs = manifest["jobs"]
    repo = Path(identity["repo"])
    if (
        manifest["identity"] != identity
        or manifest["array_tasks"] != 36
        or set(jobs) != set(STAGES)
        or any(not re.fullmatch(r"[0-9]+", job) for job in jobs.values())
        or len(set(jobs.values())) != 3
        or plan["protocol"] != "phase-c-prefix-noise-reliability-1"
        or len(plan["tasks"]) != 36
        or digest(plan) != identity["plan_sha256"]
        or digest(data) != plan["inputs_sha256"]
        or identity["prepare_time"] != "00:10:00"
        or identity["run_time"] != "00:45:00"
        or not 1 <= identity["concurrency"] <= 8
        or not Path(identity["python"]).is_file()
        or (repo / "SOURCE_COMMIT").read_text().strip() != identity["commit"]
        or file_digest(repo / WORKER) != identity["worker_sha256"]
    ):
        raise ValueError("original M47 identity differs")
    for stage in STAGES:
        saved = root / "submission" / stage
        if saved.with_suffix(".id").read_text().strip() != jobs[stage] or read(
            saved.with_suffix(".intent.json")
        ) != {
            "argv": command(root, identity, jobs, stage, recovery=False),
        }:
            raise ValueError(f"original {stage} receipt differs")
    subprocess.run(
        ["sha256sum", "-c", "SHA256SUMS", "--quiet"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return manifest


def array_indices(job: str, base: str) -> set[int]:
    """Expand complete Slurm array IDs; reject truncation and ambiguous ranges."""
    match = re.fullmatch(rf"{re.escape(base)}_(\d+|\[[0-9,\-]+(?:%\d+)?\])", job)
    if match is None:
        raise ValueError(f"unrecognized array record: {job}")
    expression = match[1].strip("[]").split("%", 1)[0]
    indices: set[int] = set()
    for part in expression.split(","):
        bounds = part.split("-")
        if len(bounds) not in (1, 2) or not all(x.isdecimal() for x in bounds):
            raise ValueError("invalid array range")
        lo, hi = int(bounds[0]), int(bounds[-1])
        if not 0 <= lo <= hi < 36:
            raise ValueError("array index outside frozen roster")
        block = set(range(lo, hi + 1))
        if indices & block:
            raise ValueError("overlapping array indices")
        indices.update(block)
    return indices


def check_accounting(output: str, jobs: dict) -> None:
    """Require exact terminal coverage and no started original array elements."""
    seen: set[int] = set()
    singles: set[str] = set()
    for line in output.splitlines():
        fields = [item.strip() for item in line.split("|")]
        if len(fields) != 5:
            raise ValueError("unexpected accounting format")
        job, name, state, elapsed, start = fields
        if job.endswith((".batch", ".extern")):
            continue
        state = state.split(" ", 1)[0]
        if job in (jobs["prepare"], jobs["report"]):
            stage = "prepare" if job == jobs["prepare"] else "report"
            expected = "TIMEOUT" if stage == "prepare" else "COMPLETED"
            if (
                stage in singles
                or state != expected
                or name != f"prefix-noise-reliability-{stage}"
            ):
                raise ValueError(f"original {stage} not confirmed {expected}")
            singles.add(stage)
        else:
            indices = array_indices(job, jobs["run"])
            if (
                state != "CANCELLED"
                or elapsed != "0"
                or start not in ("Unknown", "None", "N/A", "")
                or name != "prefix-noise-reliability-run"
                or seen & indices
            ):
                raise ValueError("array is not entirely cancelled and unstarted")
            seen.update(indices)
    if singles != {"prepare", "report"} or seen != set(range(36)):
        raise ValueError("incomplete original job accounting; no recovery submitted")


def original_stopped(jobs: dict) -> dict:
    """Read scheduler evidence before allowing the first recovery submission."""
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
            "--format=JobID%200,JobName%100,State%100,ElapsedRaw,Start",
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
        raise ValueError("original job is still queued; no recovery submitted")
    return {"sacct": accounting.stdout, "squeue": queue.stdout}


def submit_once(directory: Path, stage: str, argv: list[str], env: dict) -> str:
    """Reuse confirmed jobs; never repeat an unknown scheduler request."""
    intent, reply_path, receipt = (
        directory / f"{stage}.{suffix}"
        for suffix in ("intent.json", "reply.json", "accepted.json")
    )
    if receipt.exists():
        saved = read(receipt)
        if read(intent) != {"argv": argv} or not re.fullmatch(r"[0-9]+", saved["job"]):
            raise ValueError("recovery job identity differs")
        return saved["job"]
    if intent.exists() or reply_path.exists():
        raise ValueError(f"Unconfirmed {stage} submission; inspect {directory}")
    write(intent, {"argv": argv})
    try:
        reply = subprocess.run(
            argv, env=env, capture_output=True, text=True, timeout=45
        )
    except (subprocess.TimeoutExpired, OSError) as error:
        write(reply_path, {"status": "uncertain", "error": str(error)})
        raise ValueError(f"Scheduler outcome unknown; inspect {directory}") from error
    write(
        reply_path,
        {
            "returncode": reply.returncode,
            "stdout": reply.stdout,
            "stderr": reply.stderr,
        },
    )
    if reply.returncode or not re.fullmatch(
        r"[0-9]+(?:;[A-Za-z0-9_-]+)?",
        reply.stdout.strip(),
    ):
        raise ValueError(f"Scheduler reply unconfirmed; inspect {reply_path}")
    job = reply.stdout.strip().split(";", 1)[0]
    write(receipt, {"job": job})
    print(f"Recovery {stage}: {job}", flush=True)
    return job


def recover(root: Path, casadi_root: Path) -> dict:
    """Submit one journaled replacement chain using the unchanged M47 worker."""
    root = root.resolve(strict=True)
    directory = root / RECOVERY
    directory.mkdir(exist_ok=True)
    with (directory / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print("Verifying original M47 bundle and receipts...", flush=True)
        manifest = verify_original(root)
        original = manifest["identity"]
        identity = {
            "protocol": "prefix-noise-preparation-recovery-1",
            "original_manifest_sha256": digest(manifest),
            "inputs_file_sha256": file_digest(root / "inputs.json"),
            "bundle_manifest_sha256": file_digest(
                Path(original["repo"]) / "SHA256SUMS"
            ),
            "helper_sha256": file_digest(Path(__file__)),
            "casadi_root": str(casadi_root.resolve(strict=True)),
            "prepare_time": "00:30:00",
        }
        identity_path = directory / "identity.json"
        if identity_path.exists():
            if read(identity_path) != identity:
                raise ValueError("recovery identity differs")
        else:
            if any((root / "results").glob("*")):
                raise ValueError(
                    "results already exist; this recovery requires no started work"
                )
            print(
                "Checking original Slurm jobs are terminal and array never started...",
                flush=True,
            )
            evidence = original_stopped(manifest["jobs"])
            write(directory / "original-accounting.json", evidence)
            write(directory / "original-summary.json", read(root / "summary.json"))
            write(identity_path, identity)
        env = os.environ | {
            "AF_REPO_ROOT": original["repo"],
            "AF_OUTPUT_ROOT": str(root),
            "AF_COMMIT": original["commit"],
            "AF_PYTHON": original["python"],
            "AF_CASADI_ROOT": identity["casadi_root"],
            "PYTHONDONTWRITEBYTECODE": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
        jobs: dict[str, str] = {}
        for stage in STAGES:
            jobs[stage] = submit_once(
                directory,
                stage,
                command(root, original, jobs, stage, recovery=True),
                env,
            )
        result = {
            "identity": identity,
            "jobs": jobs,
            "original_jobs": manifest["jobs"],
            "array_tasks": 36,
            "gpus": 0,
            "live_llm_calls": 0,
            "test_data_opened": False,
            "artifact": str(directory / "manifest.json"),
        }
        if not (directory / "manifest.json").exists():
            write(directory / "manifest.json", result)
        elif read(directory / "manifest.json") != result:
            raise ValueError("recovery manifest differs")
        return result


def main() -> None:
    """Run from outside the sealed source tree with system Python 3.11 or newer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument(
        "--casadi-root",
        type=Path,
        default=Path("/projects/bibo/yxiao2/venvs/fitter-methods-v1-deps"),
    )
    args = parser.parse_args()
    print(json.dumps(recover(args.root, args.casadi_root), indent=2))


if __name__ == "__main__":
    main()
