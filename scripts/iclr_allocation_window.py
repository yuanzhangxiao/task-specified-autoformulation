"""Cap targeted search work using the actual Slurm allocation's remaining time.

The two-hour completion margin matches the original twelve-hour allocation and
ten-hour worker allowance. It reduces the chance of interrupting active work;
it is not a hard upper bound on critic retries, fitting or pruning duration.
This adapter changes no source, plan, scientific budget or saved result.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import tempfile
import time
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path

PROTOCOL = "iclr-targeted-perturbations-1"
MARGIN_SECONDS = 7200


def applies(plan: dict, environment: Mapping[str, str] | None = None) -> bool:
    """Restrict operational timing changes to this targeted Slurm campaign."""
    environment = os.environ if environment is None else environment
    return plan.get("targeted_protocol") == PROTOCOL and bool(
        environment.get("SLURM_JOB_ID")
    )


def end_timestamp(value: str) -> float:
    """Read Slurm's local wall clock, preserving an explicit offset if present."""
    if not re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:Z|[+-]\d{2}:\d{2})?", value
    ):
        raise ValueError("Slurm EndTime is missing, unknown or invalid")
    try:
        # A naive Slurm timestamp uses the compute host's local timezone, not UTC.
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except (ValueError, OverflowError, OSError) as error:
        raise ValueError("Slurm EndTime is invalid") from error


def limit_seconds(value: str) -> int:
    """Require the finite [days-]hours:minutes:seconds form returned by Slurm."""
    match = re.fullmatch(r"(?:(\d+)-)?(\d+):(\d{2}):(\d{2})", value)
    if match is None:
        raise ValueError("Slurm TimeLimit is missing, unlimited or invalid")
    days, hours, minutes, seconds = (int(v or 0) for v in match.groups())
    total = days * 86400 + hours * 3600 + minutes * 60 + seconds
    if minutes >= 60 or seconds >= 60 or total <= 0:
        raise ValueError("Slurm TimeLimit is invalid")
    return total


def _job_record(
    job: str, array_job: str | None = None, array_task: str | None = None
) -> dict[str, str]:
    """Read only the requested allocation; omit scheduler output on failure."""
    query_id = f"{array_job}_{array_task}" if array_job is not None else job
    try:
        result = subprocess.run(
            ["scontrol", "-o", "show", "job", query_id],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError("Cannot verify the Slurm allocation deadline") from error
    values = re.findall(
        r"(?:^|\s)(JobId|ArrayJobId|ArrayTaskId|JobState|EndTime|TimeLimit)=(\S+)",
        result.stdout,
    )
    record = dict(values)
    required = {"JobId", "JobState", "EndTime", "TimeLimit"}
    if array_job is not None:
        required |= {"ArrayJobId", "ArrayTaskId"}
    if (
        len(values) != len(record)
        or not required <= record.keys()
        or record["JobId"] != job
        or (
            array_job is not None
            and (
                record["ArrayJobId"] != array_job or record["ArrayTaskId"] != array_task
            )
        )
    ):
        raise ValueError("Slurm returned an ambiguous or mismatched allocation")
    if record["JobState"] != "RUNNING":
        raise ValueError("Slurm allocation must be RUNNING before starting work")
    return record


def _write_receipt(path: Path, receipt: dict) -> None:
    """Atomically record operational timing, without environment or credentials."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(receipt, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def allowance(
    root: Path,
    plan: dict,
    block: int,
    requested_seconds: int,
    *,
    environment: Mapping[str, str] | None = None,
    now: float | None = None,
) -> int:
    """Record and return the remaining admission window; zero means defer work."""
    environment = os.environ if environment is None else environment
    if not applies(plan, environment):
        return requested_seconds
    job = environment["SLURM_JOB_ID"]
    if not re.fullmatch(r"[0-9]+", job):
        raise ValueError("Invalid Slurm job identifier")
    array_job = environment.get("SLURM_ARRAY_JOB_ID")
    array_task = environment.get("SLURM_ARRAY_TASK_ID")
    if (array_job is None) != (array_task is None) or any(
        value is not None and not re.fullmatch(r"[0-9]+", value)
        for value in (array_job, array_task)
    ):
        raise ValueError("Invalid or incomplete Slurm array identifiers")
    if type(requested_seconds) is not int or requested_seconds <= 0:
        raise ValueError("Worker allowance must be a positive number of seconds")
    if type(block) is not int or block < 0:
        raise ValueError("Invalid campaign block")
    record = _job_record(job, array_job, array_task)
    end = end_timestamp(record["EndTime"])
    limit = limit_seconds(record["TimeLimit"])
    observed = time.time() if now is None else now
    if not math.isfinite(observed):
        raise ValueError("Invalid observation time")
    effective = max(
        0, min(requested_seconds, math.floor(end - observed - MARGIN_SECONDS))
    )
    receipt = {
        "protocol": "iclr-targeted-allocation-window-1",
        "job": job,
        "array_job": array_job,
        "array_task": array_task,
        "slurm_query_id": f"{array_job}_{array_task}" if array_job is not None else job,
        "block": block,
        "plan_sha256": plan.get("artifact_sha256"),
        "source_sha256": plan.get("source_sha256"),
        "requested_seconds": requested_seconds,
        "effective_seconds": effective,
        "completion_margin_seconds": MARGIN_SECONDS,
        "slurm_time_limit_seconds": limit,
        "slurm_end_time": record["EndTime"],
        "slurm_end_timestamp": end,
        "observed_timestamp": observed,
        "status": "ready" if effective else "pending_insufficient_time",
        "completion_margin_is_hard_bound": False,
        "scientific_budgets_modified": False,
    }
    _write_receipt(root / "runtime" / f"allocation-{job}.json", receipt)
    return effective
