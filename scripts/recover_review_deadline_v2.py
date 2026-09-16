#!/usr/bin/env python3
"""Recover unsubmitted visits using the original frozen scientific checkout.

This standalone scheduler driver deliberately imports no autoformalism modules.
It never re-freezes a plan, repeats a numerical attempt, or edits the old intent.
"""

from __future__ import annotations

import argparse
import fcntl
import getpass
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

FAILED = {"FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "BOOT_FAIL"}
PRIOR_RESULTS = """
import json, sys
from pathlib import Path
from autoformalism.rebuttal.prefit_replay import sealed_read
root, index = Path(sys.argv[1]), int(sys.argv[2])
plan = sealed_read(root / 'plan.json')
artifacts = {}
for task in plan['tasks']:
    path = root / 'results' / task['task_id'] / f'round_{index:02d}' / 'result.json'
    result = sealed_read(path)
    if (result.get('task') != task or result.get('round') != index
        or not result.get('status')):
        raise ValueError(f'prior result identity differs: {path}')
    artifacts[task['task_id']] = result['artifact_sha256']
print(json.dumps(artifacts))
"""


def run(
    argv: list[str], *, env: dict | None = None, receipt: Path | None = None
) -> str:
    """Require a successful, bounded scheduler or verification command."""
    try:
        result = subprocess.run(
            argv, env=env, check=False, text=True, capture_output=True, timeout=180
        )
    except subprocess.TimeoutExpired as error:
        if receipt is not None:
            write(receipt, {"argv": argv, "status": "timeout_reply_uncertain"})
        raise error
    if receipt is not None:
        write(
            receipt,
            {
                "argv": argv,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
            },
        )
    result.check_returncode()
    return result.stdout.strip()


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def write(path: Path, value: dict) -> None:
    """Atomically publish metadata, without touching scientific artifacts."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def accounting(job: str) -> dict:
    """Use durable accounting, not dependency-addressable controller records."""
    if not re.fullmatch(r"[0-9]+", job):
        raise ValueError("invalid recorded job ID")
    rows = run(
        [
            "sacct",
            "-n",
            "-P",
            "-j",
            job,
            "--format=JobIDRaw,State%40,ExitCode,Start",
        ]
    ).splitlines()
    matches = [r.split("|") for r in rows if r.split("|")[0] == job]
    if len(matches) != 1 or len(matches[0]) < 4:
        raise ValueError(f"accounting is unavailable or ambiguous for {job}")
    _, state, code, start, *_ = (field.strip() for field in matches[0])
    return {"job": job, "state": state.split()[0], "exit_code": code, "start": start}


def completed(job: str) -> dict:
    value = accounting(job)
    if value["state"] != "COMPLETED" or value["exit_code"] != "0:0":
        raise ValueError(f"prerequisite has not completed successfully: {value}")
    return value


def check_unsubmitted(root: Path, index: int, start: str) -> None:
    """Refuse known or possibly accepted work instead of guessing at a retry."""
    names = [f"review-v2-{stage}-{index}" for stage in ("propose", "fit", "finish")]
    for stage in ("propose", "fit", "finish"):
        if (root / "submission-intent" / f"{stage}-{index}.id").exists():
            raise ValueError(
                "original visit has recorded jobs; inspect before recovery"
            )
    for name in ("proposal.json", "worker_started.json", "result.json"):
        if any((root / "results").glob(f"*/round_{index:02d}/{name}")):
            raise ValueError(
                "visit already has scientific work; inspect before recovery"
            )
    check_absent(names, start)


def check_absent(names: list[str], start: str) -> None:
    """Check accounting and the live queue before a possibly duplicate submission."""
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", start):
        raise ValueError("missing preparation start time for duplicate-job audit")
    history = run(
        [
            "sacct",
            "-n",
            "-P",
            "--user",
            getpass.getuser(),
            "--starttime",
            start,
            "--name",
            ",".join(names),
            "--format=JobIDRaw,JobName%80,State",
        ]
    )
    active = run(
        ["squeue", "-h", "--user", getpass.getuser(), "--name", ",".join(names)]
    )
    if history or active:
        raise ValueError("scheduler has visit jobs; inspect before recovery")


def recorded_job(job: str, key: str, *, indices: set[int] | None = None) -> dict:
    """Confirm the identity of an already accepted job without resubmitting it."""
    if not re.fullmatch(r"[0-9]+", job):
        raise ValueError("invalid saved recovery job ID")
    rows = run(
        [
            "sacct",
            "--array",
            "-n",
            "-P",
            "-j",
            job,
            "--format=JobID%40,JobName%80,State%40,ExitCode",
        ]
    ).splitlines()
    allocations = [[f.strip() for f in r.split("|")] for r in rows]
    if indices is not None:
        matches = [
            r for r in allocations if re.fullmatch(re.escape(job) + r"_\d+", r[0])
        ]
        actual = [int(r[0].rsplit("_", 1)[1]) for r in matches]
        if len(actual) != len(indices) or set(actual) != indices:
            raise ValueError(
                f"cannot confirm complete saved array {job}; accounting may be "
                "catching up. No repair was submitted; wait and retry."
            )
    else:
        matches = [r for r in allocations if r[0] == job]
        if len(matches) != 1:
            raise ValueError(f"cannot confirm existing job {job}")
    for fields in matches:
        if (
            len(fields) != 4
            or fields[1] != f"review-v2-{key}"
            or fields[2] not in {"PENDING", "RUNNING", "COMPLETING", "COMPLETED"}
            or fields[3] != "0:0"
        ):
            raise ValueError(f"existing job needs inspection: {fields}")
    return {
        "job": job,
        "name": f"review-v2-{key}",
        "state": matches[0][2] if len(matches) == 1 else "ARRAY_CONFIRMED",
        "allocations": matches,
    }


def check_prior_reply(attempt: Path, key: str) -> None:
    """Positive or uncertain local submission evidence outweighs empty snapshots."""
    path = attempt / f"{key}.reply.json"
    if not path.exists():
        return  # The legacy driver did not save replies; inspect queue/accounting.
    value = read(path)
    if (
        value.get("status") is not None
        or not isinstance(value.get("returncode"), int)
        or value["returncode"] <= 0
        or value["returncode"] >= 128
        or str(value.get("stdout", "")).strip()
    ):
        raise ValueError(
            f"prior dispatcher reply records acceptance or uncertainty: {path}; "
            "inspect it before any new submission"
        )


def recover(index: int, *, complete_dispatcher_only: bool = False) -> dict:
    """Queue one untouched visit and, if needed, this same bounded dispatcher."""
    root = Path(os.environ["AF_OUTPUT_ROOT"]).resolve()
    repo = Path(os.environ["AF_REPO_ROOT"]).resolve()
    # Preserve the venv executable path: resolving its symlink loses the venv.
    python = str(Path(os.environ["AF_PYTHON"]).absolute())
    script = Path(__file__).resolve()
    digest = hashlib.sha256(script.read_bytes()).hexdigest()
    expected = os.environ.get("AF_RECOVERY_SCRIPT_SHA256", digest)
    if expected != digest:
        raise ValueError("recovery dispatcher source changed")
    if index < 1:
        raise ValueError("recovery never reruns visit zero")
    ledger = root / "scheduler-recovery"
    ledger.mkdir(exist_ok=True)
    with (ledger / "lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        existing = root / f"submission-round-{index}.json"
        if existing.exists():
            return read(existing)
        attempt = ledger / f"attempt-{index}"
        if attempt.exists() and not complete_dispatcher_only:
            raise ValueError("recovery intent exists; inspect jobs before retrying")
        if complete_dispatcher_only and not attempt.exists():
            raise ValueError("dispatcher-only repair requires an existing attempt")
        parent = read(root / f"submission-round-{index - 1}.json")
        plan = read(root / "plan.json")
        if (
            parent["protocol"] != "review-deadline-2"
            or plan["protocol"] != parent["protocol"]
        ):
            raise ValueError("requires the original review-deadline-2 campaign")
        if not index < plan["config"]["rounds"]:
            raise ValueError("visit outside frozen campaign")
        commit = run(["git", "-C", str(repo), "rev-parse", "HEAD"])
        if commit != parent["commit"] or run(
            ["git", "-C", str(repo), "status", "--porcelain"]
        ):
            raise ValueError("use the clean ORIGINAL pinned scientific checkout")
        env = {
            **os.environ,
            "AF_COMMIT": commit,
            "AF_CONFIG": str(repo / "configs/review_deadline_v2.json"),
            "AF_RECOVERY_SCRIPT_SHA256": digest,
            "PYTHONPATH": str(repo / "src"),
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
        base = "/scratch/user/u.yx126462"
        for key, default in {
            "AF_VLLM_IMAGE": f"{base}/containers/vllm-openai-v0.27.1.sif",
            "AF_HF_HOME": f"{base}/huggingface-cache",
            "AF_COMPUTE_CACHE_ROOT": f"{base}/autoformalism-runtime-cache/review",
            "AF_IPC_TMP_ROOT": f"{base}/af-ipc",
        }.items():
            env.setdefault(key, default)
        # Verify scientific source, launcher, runtime and public assets with OLD code.
        run(
            [
                python,
                str(repo / "scripts/review_deadline.py"),
                "verify",
                "--root",
                str(root),
            ],
            env=env,
        )
        prepare = completed(parent["jobs"]["prepare-0"])
        finish = completed(parent["jobs"][f"finish-{index - 1}"])
        prior_results = json.loads(
            run([python, "-c", PRIOR_RESULTS, str(root), str(index - 1)], env=env)
        )
        failed_dispatch = None
        if index == 1:
            failed_dispatch = accounting(parent["jobs"]["submit-next-1"])
            if failed_dispatch["state"] not in FAILED:
                raise ValueError(
                    "original dispatcher is not a confirmed terminal failure"
                )
        adopted = {}
        if complete_dispatcher_only:
            if index + 1 >= plan["config"]["rounds"]:
                raise ValueError("last visit does not need a dispatcher")
            old = read(attempt / "evidence.json")
            for key, value in {
                "scientific_commit": commit,
                "plan_sha256": plan["artifact_sha256"],
                "round": index,
                "prior_results": prior_results,
            }.items():
                if old.get(key) != value:
                    raise ValueError(f"partial recovery identity differs: {key}")
            required = [f"{s}-{index}" for s in ("propose", "fit", "finish")]
            if {p.stem for p in attempt.glob("*.id")} != set(required):
                raise ValueError("repair requires exactly the three original job IDs")
            check_prior_reply(attempt, f"submit-next-{index + 1}")
            for key in required:
                job = (attempt / f"{key}.id").read_text().strip()
                adopted[key] = recorded_job(
                    job,
                    key,
                    indices={t["index"] for t in plan["tasks"]}
                    if key.startswith("fit-")
                    else None,
                )
            check_absent([f"review-v2-submit-next-{index + 1}"], prepare["start"])
            submission_directory = attempt / "dispatcher-repair"
            # A repeated uncertain dispatcher repair also requires inspection.
            submission_directory.mkdir()
        else:
            check_unsubmitted(root, index, prepare["start"])
            # Permanent intent precedes the first possible sbatch side effect.
            attempt.mkdir()
            submission_directory = attempt
        evidence = {
            "scientific_commit": commit,
            "plan_sha256": plan["artifact_sha256"],
            "driver_sha256": digest,
            "round": index,
            "prepare": prepare,
            "prior_finish": finish,
            "prior_results": prior_results,
            "failed_original_dispatcher": failed_dispatch,
            "scheduler_only": True,
            "new_scientific_budget": False,
            "adopted_jobs": adopted,
        }
        write(submission_directory / "evidence.json", evidence)
        jobs = dict(parent["jobs"])
        jobs.update({key: value["job"] for key, value in adopted.items()})
        account = env.get("AF_ACCOUNT", "156264627414")
        worker = str(repo / "scripts/hpc/run_review_deadline_aces.sh")

        def submit(
            stage: str, visit: int, options: list[str], command: list[str]
        ) -> str:
            key = f"{stage}-{visit}"
            reply_path = submission_directory / f"{key}.reply.json"
            reply = run(
                [
                    "sbatch",
                    "--parsable",
                    "--kill-on-invalid-dep=yes",
                    f"--account={account}",
                    "--nodes=1",
                    "--ntasks=1",
                    "--export=ALL",
                    f"--job-name=review-v2-{key}",
                    f"--output={root}/logs/{key}-%A_%a.out",
                    f"--error={root}/logs/{key}-%A_%a.err",
                    *options,
                    *command,
                ],
                env=env,
                receipt=reply_path,
            )
            job = reply.split(";")[0]
            if not re.fullmatch(r"[0-9]+", job):
                raise ValueError(
                    "uncertain scheduler reply; intent retained. "
                    f"Inspect jobs and {reply_path} before retrying"
                )
            (submission_directory / f"{key}.id").write_text(job + "\n")
            jobs[key] = job
            return job

        gpu = (
            jobs[f"propose-{index}"]
            if complete_dispatcher_only
            else submit(
                "propose",
                index,
                [
                    "--partition=gpu",
                    "--gres=gpu:h100:1",
                    "--cpus-per-task=8",
                    "--mem=64G",
                    "--time=06:30:00",
                    "--signal=B:TERM@300",
                ],
                [worker, "propose", str(index)],
            )
        )
        indices = ",".join(str(t["index"]) for t in plan["tasks"])
        cpu = (
            jobs[f"fit-{index}"]
            if complete_dispatcher_only
            else submit(
                "fit",
                index,
                [
                    f"--dependency=afterany:{gpu}",
                    "--partition=cpu",
                    f"--array={indices}%16",
                    "--cpus-per-task=1",
                    "--mem=16G",
                    "--time=00:40:00",
                    "--signal=B:TERM@120",
                ],
                [worker, "fit", str(index)],
            )
        )
        finish_id = (
            jobs[f"finish-{index}"]
            if complete_dispatcher_only
            else submit(
                "finish",
                index,
                [
                    f"--dependency=afterany:{cpu}",
                    "--partition=cpu",
                    "--cpus-per-task=1",
                    "--mem=4G",
                    "--time=00:15:00",
                ],
                [worker, "finish", str(index)],
            )
        )
        last = index + 1 == plan["config"]["rounds"]
        if not last:
            shell = (
                "#!/bin/bash\nset -euo pipefail\n"
                "module load GCCcore/13.2.0 Python/3.11.5\nexec "
                + shlex.join(
                    [
                        python,
                        str(script),
                        "--round",
                        str(index + 1),
                    ]
                )
            )
            dispatch_script = submission_directory / f"dispatch-{index + 1}.sh"
            dispatch_script.write_text(shell + "\n")
            # ACES's wrapper expands its argument string without preserving
            # quoting. Pass a script file; --wrap with spaces is split by it.
            dependency = (
                []
                if (
                    complete_dispatcher_only
                    and adopted[f"finish-{index}"]["state"] == "COMPLETED"
                )
                else [f"--dependency=afterany:{finish_id}"]
            )
            submit(
                "submit-next",
                index + 1,
                [
                    *dependency,
                    "--partition=cpu",
                    "--cpus-per-task=1",
                    "--mem=4G",
                    "--time=00:15:00",
                ],
                [str(dispatch_script)],
            )
        manifest = {
            **parent,
            "jobs": jobs,
            "submitted_through_round": index,
            "all_rounds_submitted": last,
            "submission_complete": last,
            "round_submission_complete": True,
            "scheduler_recovery": evidence,
        }
        write(existing, manifest)
        write(root / "submission_manifest.json", manifest)
        return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--complete-dispatcher-only", action="store_true")
    args = parser.parse_args()
    try:
        print(
            json.dumps(
                recover(
                    args.round, complete_dispatcher_only=args.complete_dispatcher_only
                ),
                indent=2,
            )
        )
    except subprocess.CalledProcessError as error:
        print(error.stderr or str(error), file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
