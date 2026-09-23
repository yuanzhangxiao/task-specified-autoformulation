#!/usr/bin/env python3
"""Adopt one confirmed prepare job and finish a continuation's first submission.

Run this file outside the pinned scientific checkout with AF_REPO_ROOT pointing
to that checkout. No scientific files, raw receipts or existing jobs are changed.
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import importlib.util
import json
import os
import re
import shlex
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path


def scheduler_record(job: str, argv: list[str], *, since: float) -> dict:
    """Verify job ownership and the complete saved submission command."""
    if not re.fullmatch(r"[0-9]+", job):
        raise ValueError("prepare job ID must be numeric")
    reply = subprocess.run(
        ["scontrol", "show", "job", job, "--oneliner"],
        text=True,
        capture_output=True,
        timeout=60,
    )
    fields = dict(
        re.findall(
            r"(?:^|\s)([A-Za-z][A-Za-z0-9]*)=(.*?)(?=\s[A-Za-z][A-Za-z0-9]*=|$)",
            reply.stdout.strip(),
        )
    )
    if reply.returncode == 0 and fields.get("JobId") == job:
        record = {
            "job": job,
            "user": fields.get("UserId", "").split("(")[0],
            "name": fields.get("JobName"),
            "state": fields.get("JobState"),
            "exit_code": fields.get("ExitCode"),
            "submit_line": fields.get("SubmitLine", ""),
            "source": "scontrol",
        }
    else:
        # Completed jobs can leave the controller before recovery is run.
        reply = subprocess.run(
            [
                "sacct",
                "-X",
                "-n",
                "-P",
                "-j",
                job,
                "--starttime="
                + (datetime.fromtimestamp(since, UTC) - timedelta(days=1))
                .date()
                .isoformat(),
                "--format=JobIDRaw,User%80,JobName%80,State%40,ExitCode,SubmitLine%4096",
            ],
            text=True,
            capture_output=True,
            timeout=60,
            check=True,
        )
        rows = [
            line.split("|")
            for line in reply.stdout.splitlines()
            if line.split("|")[0].strip() == job
        ]
        if len(rows) != 1 or len(rows[0]) != 6:
            raise ValueError("cannot confirm prepare job in scheduler accounting")
        ident, user, name, state, code, line = (part.strip() for part in rows[0])
        record = {
            "job": ident,
            "user": user,
            "name": name,
            "state": state,
            "exit_code": code,
            "submit_line": line,
            "source": "sacct",
        }
    observed = shlex.split(record["submit_line"])
    if not observed or [Path(observed[0]).name, *observed[1:]] != argv:
        raise ValueError("prepare job submission command differs from saved intent")
    expected_name = next(
        a.split("=", 1)[1] for a in argv if a.startswith("--job-name=")
    )
    if record["user"] != getpass.getuser() or record["name"] != expected_name:
        raise ValueError("prepare job owner or name differs")
    if record["state"] not in {"PENDING", "RUNNING", "COMPLETING", "COMPLETED"}:
        raise ValueError(f"prepare job is not viable: {record['state']}")
    if record["exit_code"] != "0:0":
        raise ValueError("prepare job exit code is not successful")
    return record


def load_submitter(repo: Path):
    """Import only the original submitter and its original scientific modules."""
    sys.path.insert(0, str(repo / "src"))
    spec = importlib.util.spec_from_file_location(
        "pinned_continuation_submit", repo / "scripts/submit_review_continuation.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.io.REPO.resolve() != repo:
        raise ValueError("scientific imports do not belong to AF_REPO_ROOT")
    return module


def recover(root: Path, prepare_job: str) -> dict:
    """Continue the saved first chain; never retry an uncertain submission."""
    root = root.resolve()
    repo = Path(os.environ["AF_REPO_ROOT"]).resolve()
    submitter = load_submitter(repo)
    public, io = submitter.public, submitter.io
    io.require_open(root)
    plan = io.verify(root)
    if plan["protocol"] not in io.CONTINUATION_PROTOCOLS:
        raise ValueError("requires a frozen continuation campaign")
    if subprocess.check_output(
        ["git", "-C", str(repo), "status", "--porcelain"], text=True
    ).strip():
        raise ValueError("use the clean original scientific checkout")
    commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if os.environ.get("AF_COMMIT") != commit:
        raise ValueError("set AF_COMMIT to the original scientific commit")
    for key in (
        "AF_PYTHON",
        "AF_VLLM_IMAGE",
        "AF_HF_HOME",
        "AF_COMPUTE_CACHE_ROOT",
        "AF_IPC_TMP_ROOT",
    ):
        if not os.environ.get(key):
            raise ValueError(f"set {key} before recovery")
    if not all(
        Path(os.environ[key]).is_file() for key in ("AF_PYTHON", "AF_VLLM_IMAGE")
    ):
        raise ValueError("Python or serving image is missing")
    os.environ["AF_OUTPUT_ROOT"] = str(root)
    directory = root / "submission-intent/round-1"
    with public._lock(root / "scheduler"):
        manifest = root / "submission-round-1.json"
        if manifest.exists():
            value = public._read(manifest)
            if (
                value["plan_sha256"] != plan["artifact_sha256"]
                or value["commit"] != commit
                or value["jobs"].get("prepare-0") != prepare_job
            ):
                raise ValueError("completed submission identity differs")
            return value
        prior = {}
        for task in plan["tasks"]:
            result = io.read_round(root, task, 0)
            if result is None or result["task"] != task or result["round"] != 0:
                raise ValueError("imported checkpoint is incomplete")
            prior[task["task_id"]] = result["artifact_sha256"]
        if public._read(directory / "prerequisites.json") != {
            "plan": plan["artifact_sha256"],
            "results": prior,
        }:
            raise ValueError("saved submission prerequisites differ")
        argv = public._read(directory / "prepare-0.intent.json")["argv"]
        worker = repo / "scripts/hpc/run_review_continuation_aces.sh"
        version = plan["protocol"].rsplit("-", 1)[-1]
        if (
            argv[0] != "sbatch"
            or argv[-3:] != [str(worker), "prepare", "0"]
            or f"--output={root}/logs/prepare-0-%A_%a.out" not in argv
            or f"--error={root}/logs/prepare-0-%A_%a.err" not in argv
            or f"--job-name=review-v{version}-prepare-0" not in argv
        ):
            raise ValueError("saved prepare intent belongs to another campaign")
        audit_path = directory / "prepare-0.adoption.json"
        if audit_path.exists():
            adoption = public._read(audit_path)
            if (
                adoption["job"] != prepare_job
                or adoption["commit"] != commit
                or adoption["plan_sha256"] != plan["artifact_sha256"]
            ):
                raise ValueError("prior adoption identity differs")
        else:
            record = scheduler_record(
                prepare_job,
                argv,
                since=(directory / "prepare-0.intent.json").stat().st_mtime,
            )
            adoption = {
                "schema_version": "continuation-prepare-adoption-1",
                "recovery_script_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                "job": prepare_job,
                "commit": commit,
                "plan_sha256": plan["artifact_sha256"],
                "scheduler": record,
                "dependency": []
                if record["state"] == "COMPLETED"
                else [f"--dependency=afterok:{prepare_job}"],
            }
            public._write(audit_path, adoption)
        prepare_id = directory / "prepare-0.id"
        if prepare_id.exists() and prepare_id.read_text().strip() != prepare_job:
            raise ValueError("recorded prepare job differs")
        prepare_id.write_text(prepare_job + "\n")
        account = next(a for a in argv if a.startswith("--account="))
        base = [
            "--kill-on-invalid-dep=yes",
            account,
            "--nodes=1",
            "--ntasks=1",
            "--export=ALL",
        ]

        def queue(stage: str, index: int, options: list[str]) -> str:
            key = f"{stage}-{index}"
            flags = [
                *base,
                f"--job-name=review-v{version}-{key}",
                f"--output={root}/logs/{key}-%A_%a.out",
                f"--error={root}/logs/{key}-%A_%a.err",
                *options,
            ]
            expected = ["sbatch", "--parsable", *flags, str(worker), stage, str(index)]
            ident = directory / f"{key}.id"
            intent = directory / f"{key}.intent.json"
            if ident.exists():
                job = ident.read_text().strip()
                if not job.isdecimal() or public._read(intent)["argv"] != expected:
                    raise ValueError(f"recorded {key} submission differs")
                return job
            if intent.exists() or (directory / f"{key}.reply.json").exists():
                raise ValueError(
                    f"uncertain {key} intent exists; inspect scheduler before recovery"
                )
            return submitter.submit_job(directory, key, flags, worker, stage, index)

        proposer = queue(
            "propose",
            1,
            [
                "--partition=gpu",
                "--gres=gpu:h100:1",
                "--cpus-per-task=8",
                "--mem=64G",
                "--time=06:30:00",
                "--signal=B:TERM@300",
                *adoption["dependency"],
            ],
        )
        success = "afterok" if plan["protocol"] in io.MULTI_PROTOCOLS else "afterany"
        fit = queue(
            "fit",
            1,
            [
                f"--dependency={success}:{proposer}",
                "--partition=cpu",
                f"--array=0-{len(plan['tasks']) - 1}%16",
                "--cpus-per-task=1",
                "--mem=16G",
                "--time=00:40:00",
                "--signal=B:TERM@120",
            ],
        )
        finish = queue(
            "finish",
            1,
            [
                f"--dependency=afterany:{fit}",
                "--partition=cpu",
                "--cpus-per-task=1",
                "--mem=4G",
                "--time=00:15:00",
            ],
        )
        if plan["config"]["rounds"] > 2:
            queue(
                "dispatch",
                2,
                [
                    f"--dependency={success}:{finish}",
                    "--partition=cpu",
                    "--cpus-per-task=1",
                    "--mem=4G",
                    "--time=00:15:00",
                ],
            )
        value = {
            "protocol": plan["protocol"],
            "commit": commit,
            "plan_sha256": plan["artifact_sha256"],
            "phase_round": 1,
            "global_round": plan["continuation"]["source_round"] + 1,
            "additional_visits": plan["continuation"]["additional_visits"],
            "jobs": {
                p.stem: p.read_text().strip()
                for p in sorted((root / "submission-intent").glob("round-*/*.id"))
            },
            "all_rounds_submitted": plan["config"]["rounds"] == 2,
            "automatic_test_access": False,
            "recovered_prepare_job": prepare_job,
        }
        public._write(manifest, value)
        public._write(root / "submission_manifest.json", value)
        return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--prepare-job", required=True)
    args = parser.parse_args()
    print(json.dumps(recover(args.root, args.prepare_job), indent=2))
