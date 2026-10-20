#!/usr/bin/env python3
"""Adopt an existing function-construction job without submitting new work.

Run this standalone standard-library helper outside the pinned source bundle.
It preserves frozen scientific files and the original scheduler intent/reply.
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
import tempfile
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path

PROTOCOL = "saved-construction-functions-1"
EXECUTION_PROTOCOL = "construction-functions-aces-execution-1"
ADOPTION_PROTOCOL = "construction-functions-submission-adoption-1"
MODEL = "openai/gpt-oss-20b"
REVISION = "6cee5e81ee83917806bbde320786a8fb61efebee"
CLOCK_TOLERANCE_SECONDS = 120
ENTRYPOINTS = (
    "scripts/construction_functions.py",
    "scripts/submit_construction_functions_aces.py",
    "scripts/submit_review_continuation.py",
    "scripts/saved_initialization_aces.py",
    "scripts/harmony_vocab.py",
    "scripts/hpc/start_construction_functions_aces.sh",
    "scripts/hpc/run_construction_functions_aces.sh",
)
STATES = {
    "PENDING",
    "RUNNING",
    "COMPLETING",
    "CONFIGURING",
    "SUSPENDED",
    "STOPPED",
    "RESIZING",
    "REQUEUED",
    "REQUEUE_FED",
    "REQUEUE_HOLD",
    "RESV_DEL_HOLD",
    "SPECIAL_EXIT",
    "STAGE_OUT",
    "SIGNALING",
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


def content_hash(value: object) -> str:
    """Match staged_topology.content_hash exactly, without importing it."""
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_hash(path: Path) -> str:
    """Hash a regular file in bounded chunks."""
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"expected a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_source_hash(repo: Path) -> str:
    """Match staged_topology_campaign.runtime_source_hash exactly."""
    package = repo / "src/autoformalism"
    paths = sorted(package.rglob("*.py"))
    if not paths:
        raise ValueError("pinned runtime source package is missing")
    return content_hash(
        {str(path.relative_to(package)): file_hash(path) for path in paths}
    )


def _read(path: Path) -> dict:
    """Read a JSON object without interpreting executable content."""
    file_hash(path)
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def sealed_read(path: Path) -> dict:
    """Verify the campaign's original JSON content seal."""
    value = _read(path)
    if value.get("artifact_sha256") != content_hash(
        {key: item for key, item in value.items() if key != "artifact_sha256"}
    ):
        raise ValueError(f"artifact digest differs: {path}")
    return value


def _canonical(argv: list[str]) -> list[str]:
    """Allow only the scheduler executable's absolute-path normalization."""
    return [Path(argv[0]).name, *argv[1:]] if argv else []


def verify_campaign(root: Path, argv: list[str]) -> tuple[Path, dict, dict]:
    """Verify frozen identities and the complete original resource request."""
    if not isinstance(argv, list) or not all(isinstance(a, str) for a in argv):
        raise ValueError("saved submission argv must be a string list")
    if len(argv) < 4 or argv[-2:] != ["propose", "0"]:
        raise ValueError("saved submission is not a function proposal")
    worker = Path(argv[-3])
    if not worker.is_absolute() or worker.parts[-3:] != (
        "scripts",
        "hpc",
        "run_construction_functions_aces.sh",
    ):
        raise ValueError("saved worker path differs")
    repo = worker.parents[2].resolve()
    accounts = [a for a in argv if a.startswith("--account=")]
    if len(accounts) != 1 or not re.fullmatch(
        r"--account=[A-Za-z0-9_.-]+", accounts[0]
    ):
        raise ValueError("saved account is missing or ambiguous")
    expected = [
        "sbatch",
        "--parsable",
        "--kill-on-invalid-dep=yes",
        "--nodes=1",
        "--ntasks=1",
        "--export=ALL",
        accounts[0],
        "--partition=gpu",
        "--gres=gpu:h100:1",
        "--cpus-per-task=8",
        "--mem=64G",
        "--time=02:00:00",
        "--signal=B:TERM@300",
        "--exclude=ac042",
        "--job-name=phasec-functions",
        f"--output={root}/logs/functions-%j.out",
        f"--error={root}/logs/functions-%j.err",
        str(repo / "scripts/hpc/run_construction_functions_aces.sh"),
        "propose",
        "0",
    ]
    if argv != expected:
        raise ValueError("saved submission intent differs from the frozen request")
    plan = sealed_read(root / "plan.json")
    execution = sealed_read(root / "execution.json")
    if (
        plan.get("protocol") != PROTOCOL
        or execution.get("protocol") != EXECUTION_PROTOCOL
        or execution.get("plan_sha256") != plan["artifact_sha256"]
    ):
        raise ValueError("campaign protocol or execution plan identity differs")
    cases = plan.get("cases")
    budgets = {
        "maximum_calls_per_case": 96,
        "maximum_charged_tokens_per_case": 2097152,
        "maximum_output_tokens": 8192,
        "attempts_per_step": 2,
    }
    if (
        not isinstance(cases, list)
        or not 1 <= len(cases) <= 8
        or plan.get("budgets") != budgets
        or plan.get("automatic_followup") is not False
        or plan.get("test_data_opened") is not False
        or type(plan.get("optimizer_calls")) is not int
        or plan["optimizer_calls"] != 0
    ):
        raise ValueError("frozen function campaign limits differ")
    settings = plan.get("settings", {})
    required_settings = {
        "model": MODEL,
        "model_revision": REVISION,
        "max_output_tokens": 8192,
        "maximum_requests": 96,
        "maximum_total_tokens": 2097152,
    }
    required_execution = {
        "model": MODEL,
        "model_revision": REVISION,
        "platform": "aces-h100x1",
        "maximum_cases": len(cases),
        "maximum_generation_requests": 96 * len(cases),
        "maximum_charged_tokens": 2097152 * len(cases),
        "worker_seconds": 5400,
        "scheduler_walltime_seconds": 7200,
        "cpus_per_task": 8,
        "memory_gib": 64,
        "automatic_followup": False,
        "test_data_opened": False,
        "optimizer_calls": 0,
        "model_promotion": False,
    }
    for observed, expected_values in (
        (settings, required_settings),
        (execution, required_execution),
    ):
        if any(
            observed.get(key) != value or type(observed.get(key)) is not type(value)
            for key, value in expected_values.items()
        ):
            raise ValueError("frozen serving identity or limits differ")
    hashes = {name: file_hash(repo / name) for name in ENTRYPOINTS}
    if (
        plan.get("entrypoints") != hashes
        or execution.get("entrypoint_sha256") != hashes
    ):
        raise ValueError("pinned entrypoint hashes differ")
    if plan.get("runtime_source_sha256") != runtime_source_hash(repo):
        raise ValueError("pinned runtime source hash differs")
    if plan.get("source_archive_sha256") != file_hash(
        root / "historical/source.tar.gz"
    ):
        raise ValueError("historical source archive hash differs")
    names = set()
    for case in cases:
        name = case.get("case_id") if isinstance(case, dict) else None
        if not isinstance(name, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]{0,199}", name
        ):
            raise ValueError("invalid frozen case identity")
        if name in names:
            raise ValueError("duplicate frozen case identity")
        names.add(name)
        value = sealed_read(root / "inputs" / f"{name}.json")
        if value["artifact_sha256"] != case.get("input_sha256"):
            raise ValueError("frozen function input identity differs")
    return repo, plan, execution


def _timestamp(value: str) -> float:
    """Interpret scheduler timestamps requested explicitly in UTC."""
    if not isinstance(value, str):
        raise ValueError("scheduler submission timestamp is unavailable")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        raise ValueError("scheduler submission timestamp is unavailable") from None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.timestamp()


def _validate_record(
    record: dict, job_id: str, argv: list[str], since: float, until: float
) -> None:
    """Require one owned exact command submitted during the saved attempt."""
    if not isinstance(record, dict):
        raise ValueError("scheduler record is malformed")
    if record.get("job") != job_id or record.get("user") != getpass.getuser():
        raise ValueError("scheduler job ID or owner differs")
    if record.get("name") != "phasec-functions":
        raise ValueError("scheduler job name differs")
    command = record.get("submit_line")
    if not isinstance(command, str):
        raise ValueError("scheduler submission command is unavailable")
    try:
        observed = shlex.split(command)
    except ValueError:
        raise ValueError("scheduler submission command is malformed") from None
    if _canonical(observed) != _canonical(argv):
        raise ValueError("scheduler submission command differs from saved intent")
    stamp = _timestamp(record.get("submit_time", ""))
    if not since - CLOCK_TOLERANCE_SECONDS <= stamp <= until + CLOCK_TOLERANCE_SECONDS:
        raise ValueError("scheduler submission timestamp is outside the saved attempt")
    state_text = record.get("state")
    state = state_text.split() if isinstance(state_text, str) else []
    if not state or state[0] not in STATES:
        raise ValueError("scheduler job state is missing or unrecognized")
    code = record.get("exit_code")
    if not isinstance(code, str) or not re.fullmatch(r"\d+:\d+", code):
        raise ValueError("scheduler exit code is missing or unrecognized")
    if record.get("source") not in {"scontrol", "sacct"}:
        raise ValueError("scheduler record source differs")


def scheduler_record(
    job_id: str, argv: list[str], *, since: float, until: float
) -> dict:
    """Read controller state or accounting; never submit, cancel, or wait for a GPU."""
    if not re.fullmatch(r"[0-9]+", job_id):
        raise ValueError("job ID must be numeric")
    environment = {**os.environ, "TZ": "UTC", "SLURM_TIME_FORMAT": "standard"}
    reply = subprocess.run(
        ["scontrol", "show", "job", job_id, "--oneliner"],
        capture_output=True,
        text=True,
        timeout=60,
        env=environment,
    )
    if reply.returncode == 0 and reply.stdout.strip():
        pairs = re.findall(
            r"(?:^|\s)([A-Za-z][A-Za-z0-9]*)=(.*?)(?=\s[A-Za-z][A-Za-z0-9]*=|$)",
            reply.stdout.strip(),
        )
        fields = dict(pairs)
        if len(pairs) != len(fields):
            raise ValueError("ambiguous scheduler controller reply")
        record = {
            "job": fields.get("JobId"),
            "user": fields.get("UserId", "").split("(")[0],
            "name": fields.get("JobName"),
            "state": fields.get("JobState"),
            "exit_code": fields.get("ExitCode"),
            "submit_time": fields.get("SubmitTime"),
            "submit_line": fields.get("SubmitLine", ""),
            "source": "scontrol",
        }
    else:
        start = (datetime.fromtimestamp(since, UTC) - timedelta(days=1)).date()
        reply = subprocess.run(
            [
                "sacct",
                "-X",
                "-n",
                "-P",
                "-j",
                job_id,
                "--starttime=" + start.isoformat(),
                "--format=JobIDRaw,User%80,JobName%80,State%80,ExitCode,Submit,SubmitLine%16384",
            ],
            capture_output=True,
            text=True,
            timeout=60,
            env=environment,
        )
        rows = [
            line.split("|", 6) for line in reply.stdout.splitlines() if line.strip()
        ]
        if reply.returncode or len(rows) != 1 or len(rows[0]) != 7:
            raise ValueError("cannot confirm one job in scheduler accounting")
        job, user, name, state, code, submitted, command = [
            part.strip() for part in rows[0]
        ]
        record = {
            "job": job,
            "user": user,
            "name": name,
            "state": state,
            "exit_code": code,
            "submit_time": submitted,
            "submit_line": command,
            "source": "sacct",
        }
    _validate_record(record, job_id, argv, since, until)
    return record


@contextmanager
def _lock(root: Path):
    """Use the original submitter's nonblocking scheduler lock."""
    directory = root / "scheduler"
    directory.mkdir(exist_ok=True)
    with (directory / ".lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("campaign scheduler directory is in use") from None
        yield


def atomic_write(path: Path, text: str) -> None:
    """Publish one durable recovery file by an atomic same-directory rename."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _json_text(value: dict) -> str:
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"


def recover(root: Path, job_id: str, wave: str = "functions-1") -> dict:
    """Adopt a verified existing job, resuming only missing recovery writes."""
    if not re.fullmatch(r"[0-9]+", job_id):
        raise ValueError("job ID must be numeric")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", wave):
        raise ValueError("wave must be a simple identifier")
    root = root.resolve()
    directory = root / "submissions" / wave
    intent_path, reply_path = (
        directory / "propose.intent.json",
        directory / "propose.reply.json",
    )
    if not directory.is_dir():
        raise ValueError("saved submission directory is missing")
    with _lock(root):
        intent, reply = _read(intent_path), _read(reply_path)
        argv = intent.get("argv")
        repo, plan, execution = verify_campaign(root, argv)
        if reply.get("status") != "uncertain_timeout" and not (
            type(reply.get("returncode")) is int
            and isinstance(reply.get("stdout"), str)
            and isinstance(reply.get("stderr"), str)
        ):
            raise ValueError("saved scheduler reply is malformed")
        acknowledged = reply.get("stdout", "").strip().split(";")[0]
        if acknowledged.isdecimal() and acknowledged != job_id:
            raise ValueError("saved scheduler reply names a different job ID")
        since, until = intent_path.stat().st_mtime, reply_path.stat().st_mtime
        if until < since:
            raise ValueError("saved scheduler reply predates its intent")
        binding = {
            "protocol": ADOPTION_PROTOCOL,
            "root": str(root),
            "repo": str(repo),
            "wave": wave,
            "job": job_id,
            "plan_sha256": plan["artifact_sha256"],
            "execution_sha256": execution["artifact_sha256"],
            "intent_sha256": file_hash(intent_path),
            "reply_sha256": file_hash(reply_path),
            "intent_mtime_ns": intent_path.stat().st_mtime_ns,
            "reply_mtime_ns": reply_path.stat().st_mtime_ns,
        }
        manifest = {
            "execution_sha256": execution["artifact_sha256"],
            "wave": wave,
            "job": job_id,
            "automatic_followup": False,
            "maximum_cases": execution["maximum_cases"],
            "maximum_generation_requests": execution["maximum_generation_requests"],
            "maximum_charged_tokens": execution["maximum_charged_tokens"],
        }
        manifest_path, id_path = directory / "manifest.json", directory / "propose.id"
        audit_path = directory / "propose.adoption.json"
        if manifest_path.exists() and _read(manifest_path) != manifest:
            raise ValueError("existing submission manifest identity differs")
        if id_path.exists():
            file_hash(id_path)
            if id_path.read_text().strip() != job_id:
                raise ValueError("existing submission job ID differs")
        if audit_path.exists():
            audit = sealed_read(audit_path)
            if any(audit.get(key) != value for key, value in binding.items()):
                raise ValueError("existing adoption identity differs")
            record = audit.get("scheduler", {})
            _validate_record(record, job_id, argv, since, until)
        else:
            record = scheduler_record(job_id, argv, since=since, until=until)
            audit = {
                **binding,
                "scheduler": record,
                "recovery_script_sha256": file_hash(Path(__file__).resolve()),
                "automatic_retry": False,
            }
            audit["artifact_sha256"] = content_hash(audit)
        if (
            file_hash(intent_path) != binding["intent_sha256"]
            or file_hash(reply_path) != binding["reply_sha256"]
        ):
            raise ValueError("scheduler receipts changed during recovery")
        for path, value in (
            (audit_path, _json_text(audit)),
            (id_path, job_id + "\n"),
            (manifest_path, _json_text(manifest)),
        ):
            if not path.exists():
                atomic_write(path, value)
        return manifest


def main() -> None:
    """Print the adopted identity and the observed scheduler state."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--wave", default="functions-1")
    args = parser.parse_args()
    manifest = recover(args.root, args.job_id, args.wave)
    audit = sealed_read(args.root / "submissions" / args.wave / "propose.adoption.json")
    print(
        _json_text(
            {**manifest, "scheduler": audit["scheduler"], "automatic_retry": False}
        ),
        end="",
    )


if __name__ == "__main__":
    main()
