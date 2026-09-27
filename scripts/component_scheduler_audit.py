"""Reconcile saved sacct rows with a portable campaign audit; never resume work."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from collections import Counter

JOB = re.compile(r"\d+(?:_\d+)?(?:\.[A-Za-z0-9_-]+)?\Z")
TERMINAL = {
    "BOOT_FAIL",
    "CANCELLED",
    "COMPLETED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "REVOKED",
    "TIMEOUT",
}
ACTIVE = {
    "PENDING",
    "RUNNING",
    "CONFIGURING",
    "COMPLETING",
    "SUSPENDED",
    "RESIZING",
    "REQUEUED",
    "REQUEUE_FED",
    "REQUEUE_HOLD",
    "SIGNALING",
    "SPECIAL_EXIT",
    "STAGE_OUT",
    "STOPPED",
}


def parse(text: str) -> list[dict]:
    """Require untruncated job IDs; retain steps and conflicting duplicate rows."""
    reader = csv.DictReader(io.StringIO(text), delimiter="|")
    required = {"JobID", "JobName", "State", "ExitCode", "Elapsed"}
    if not required.issubset(reader.fieldnames or []):
        raise ValueError(
            "scheduler PSV requires JobID, JobName, State, ExitCode, Elapsed"
        )
    rows = []
    for row in reader:
        if None in row:
            raise ValueError("malformed scheduler row")
        item = {key: (row.get(key) or "").strip() for key in required}
        if not JOB.fullmatch(item["JobID"]):
            raise ValueError("unsupported or truncated scheduler JobID")
        raw = (row.get("JobIDRaw") or "").strip()
        if raw and not JOB.fullmatch(raw):
            raise ValueError("unsupported scheduler JobIDRaw")
        item["JobIDRaw"] = raw or None
        for field in ("Submit", "Start", "End"):
            item[field] = (row.get(field) or "").strip() or None
        # 'CANCELLED by ...' is terminal; truncated states (e.g. OUT_OF_ME+)
        # stay unknown rather than being guessed.
        state = item["State"].split()[0] if item["State"] else "UNKNOWN"
        item["state_class"] = (
            "terminal"
            if state in TERMINAL
            else "active"
            if state in ACTIVE
            else "unknown"
        )
        item["state"] = state
        if item not in rows:
            rows.append(item)
    return rows


def reconcile(audit: dict, text: str) -> dict:
    """Match submission arrays and raw worker IDs without inferring live absence."""
    rows = parse(text)
    submissions = {j for s in audit["submissions"] for j in s["jobs"].values()}
    worker_ids = {w["job_id"] for w in audit["workers"] if w["job_id"]}
    matched = [
        r
        for r in rows
        if r["JobID"].split(".")[0].split("_")[0] in submissions
        or (r["JobIDRaw"] or "").split(".")[0] in worker_ids
        or r["JobID"].split(".")[0] in worker_ids
    ]
    jobs = [r for r in matched if "." not in r["JobID"]]
    steps = [r for r in matched if "." in r["JobID"]]
    arrays = {r["JobID"].split("_")[0] for r in jobs if "_" in r["JobID"]}
    # A parent summary and its array children must not both count as tasks.
    tasks = [r for r in jobs if r["JobID"] not in arrays]
    conflicting = sorted(
        j for j, n in Counter(r["JobID"] for r in jobs).items() if n > 1
    )
    worker_coverage = []
    for worker in audit["workers"]:
        ident = worker["job_id"]
        hits = [
            r
            for r in tasks
            if ident
            and (
                r["JobIDRaw"] == ident
                or ("_" not in r["JobID"] and r["JobID"] == ident)
            )
        ]
        worker_coverage.append(
            {
                **worker,
                "scheduler_match": "exact"
                if len(hits) == 1
                else "ambiguous"
                if hits
                else "unresolved",
                "scheduler_rows": hits,
            }
        )
    grouped = []
    for submission in audit["submissions"]:
        for stage, parent in sorted(submission["jobs"].items()):
            members = [r for r in tasks if r["JobID"].split("_")[0] == parent]
            grouped.append(
                {
                    "wave": submission["wave"],
                    "stage": stage,
                    "submission_id": parent,
                    "observed_tasks": len(members),
                    "states": dict(
                        sorted(Counter(r["state"] for r in members).items())
                    ),
                }
            )
    return {
        "protocol": "component-scheduler-observation-1",
        "audit_sha256": audit["artifact_sha256"],
        "scheduler_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "observed_tasks": len(tasks),
        "state_counts": dict(sorted(Counter(r["state"] for r in tasks).items())),
        "submissions": grouped,
        "non_success_tasks": [
            r for r in tasks if r["state"] != "COMPLETED" or r["ExitCode"] != "0:0"
        ],
        "non_success_steps": [
            r for r in steps if r["state"] != "COMPLETED" or r["ExitCode"] != "0:0"
        ],
        "observed_active_tasks": [
            r["JobID"] for r in tasks if r["state_class"] == "active"
        ],
        "unknown_state_tasks": [
            r["JobID"] for r in tasks if r["state_class"] == "unknown"
        ],
        "conflicting_job_ids": conflicting,
        "unobserved_submissions": [s for s in grouped if not s["observed_tasks"]],
        "worker_match_counts": dict(
            sorted(Counter(w["scheduler_match"] for w in worker_coverage).items())
        ),
        "workers": worker_coverage,
        "unmatched_rows": len(rows) - len(matched),
        "remaining_work": [
            {"task": t["task"], **t["frontier"]} for t in audit["lineages"]
        ],
        "resume_authorized": False,
        "scope": (
            "Saved scheduler observation, not a live writer check. Missing "
            "rows are unknown. "
            "Array JobID suffixes cannot be converted arithmetically to SLURM_JOB_ID; "
            "JobIDRaw is needed. COMPLETED is allocation status, not campaign "
            "completion. "
            "No failure cause, unused budget, or permission to reset an "
            "attempt is inferred."
        ),
    }


def outputs(audit: dict, text: str) -> dict[str, str]:
    """Keep reconciliation separate from the sealed filesystem observation."""
    value = reconcile(audit, text)
    lines = [
        "# Scheduler and campaign reconciliation",
        "",
        value["scope"],
        "",
        "| Wave | Stage | Submission | Observed tasks | States |",
        "|---|---|---|---:|---|",
    ]
    for s in value["submissions"]:
        lines.append(
            f"| {s['wave']} | {s['stage']} | {s['submission_id']} | "
            f"{s['observed_tasks']} | {s['states']} |"
        )
    lines += [
        "",
        "Worker identity matches: " + str(value["worker_match_counts"]),
        "",
        "Non-success tasks:",
        "",
        "```json",
        json.dumps(value["non_success_tasks"], indent=2),
        "```",
        "",
        "Read worker receipts/logs, refresh live queue and campaign observations, "
        "and validate the original runtime before scheduling any remaining work.",
        "",
    ]
    return {
        "scheduler_summary.json": json.dumps(value, indent=2, sort_keys=True) + "\n",
        "SCHEDULER.md": "\n".join(lines),
        "scheduler.psv": text,
    }
