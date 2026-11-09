"""Reconcile an ACES submission from Slurm records without issuing another job."""

from __future__ import annotations

import getpass
import json
import os
import re
import shlex
import subprocess
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

from autoformalism.rebuttal.prefit_replay import sealed_read

# Protect the interval during which a controller reply and accounting can cross.
CLOCK_SKEW_SECONDS = 120


def _query(argv):
    reply = subprocess.run(
        argv,
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, "TZ": "UTC", "SLURM_TIME_FORMAT": "standard"},
    )
    if reply.returncode or "error:" in reply.stderr.lower():
        raise ValueError(f"Scheduler lookup failed ({argv[0]}): {reply.stderr.strip()}")
    return reply.stdout


def _stamp(text):
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    return (parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)).timestamp()


def _value(argv, key):
    matches = [arg.split("=", 1)[1] for arg in argv if arg.startswith("--" + key + "=")]
    if len(matches) != 1:
        raise ValueError(f"Expected one --{key} in the saved scheduler intent")
    return matches[0]


def _matches(command, argv):
    """ACES can wrap sbatch. Bind to the worker, job name and both output paths."""
    try:
        fields = shlex.split(command)
        keys = ["job-name", "output", "error"]
        if any(arg.startswith("--comment=") for arg in argv):
            keys.append("comment")
        return fields[-2:] == argv[-2:] and all(
            _value(fields, k) == _value(argv, k) for k in keys
        )
    except ValueError:
        return False


def _controller_match(job, argv):
    output = _query(["scontrol", "show", "job", job, "--oneliner"]).strip()
    fields = dict(
        re.findall(
            r"(?:^|\s)([A-Za-z][A-Za-z0-9]*)=(.*?)(?=\s[A-Za-z][A-Za-z0-9]*=|$)", output
        )
    )
    if (
        fields.get("JobId") != job
        or fields.get("UserId", "").split("(")[0] != getpass.getuser()
        or fields.get("JobName") != _value(argv, "job-name")
    ):
        return False
    if _matches(fields.get("SubmitLine", ""), argv):
        return True
    if any(arg.startswith("--comment=") for arg in argv) and fields.get(
        "Comment"
    ) != _value(argv, "comment"):
        return False
    return fields.get("Command", "").split()[:1] == [argv[-2]] and all(
        fields.get(field) in (expected, expected.replace("%j", job))
        for field, expected in (
            ("StdOut", _value(argv, "output")),
            ("StdErr", _value(argv, "error")),
        )
    )


def _accounting_paths(job: str, argv: list[str]) -> bool:
    """Terminal jobs can outlive their controller record and full submit line."""
    output = _query(
        [
            "sacct",
            "-X",
            "-n",
            "-P",
            "-j",
            job,
            "--format=JobIDRaw,StdOut%4096,StdErr%4096,Comment%256",
        ]
    )
    rows = [line.strip().split("|") for line in output.splitlines() if line.strip()]
    if len(rows) != 1 or len(rows[0]) != 4 or rows[0][0] != job:
        return False
    _, stdout, stderr, comment = rows[0]
    tag = next((a.split("=", 1)[1] for a in argv if a.startswith("--comment=")), None)
    return (tag is None or tag == comment) and all(
        actual == expected or actual == expected.replace("%j", job)
        for actual, expected in (
            (stdout, _value(argv, "output")),
            (stderr, _value(argv, "error")),
        )
    )


def reconcile(root: Path, wave: Path) -> dict:
    """A name or a zero wrapper exit alone never identifies the submitted worker."""
    intent = wave / "propose.intent.json"
    if not intent.is_file():
        if any(p.name != "predecessor-check.json" for p in wave.iterdir()):
            raise ValueError(f"Missing scheduler intent in {wave}")
        return {"status": "no_submission_intent"}
    argv = json.loads(intent.read_text())["argv"]
    name = _value(argv, "job-name")
    if (
        name not in {"phasec-t2-breadth", "phasec-t2-depth3", "phasec-t2-fitted"}
        or _value(argv, "output") != str(root / "logs/propose-%j.out")
        or _value(argv, "error") != str(root / "logs/propose-%j.err")
        or Path(argv[-2]).name != "run_meal_insulin_revision_aces.sh"
        or argv[-1] != "t2-breadth"
    ):
        raise ValueError("Original sbatch intent does not address this T2 result root")
    reply_path = wave / "propose.reply.json"
    reply = json.loads(reply_path.read_text()) if reply_path.exists() else {}
    start = intent.stat().st_mtime
    end = reply_path.stat().st_mtime if reply_path.exists() else start + 60
    user = getpass.getuser()
    date = (datetime.fromtimestamp(start, UTC) - timedelta(days=1)).date().isoformat()
    commands = {
        "sacct": [
            "sacct",
            "-X",
            "-n",
            "-P",
            "-u",
            user,
            "--name=" + name,
            "-S",
            date,
            "--format=JobIDRaw,User%80,JobName%80,State%48,Submit,SubmitLine%4096",
        ],
        "squeue": ["squeue", "-h", "-u", user, "-n", name, "-o", "%i|%u|%j|%T|%V"],
    }
    lookups = {}
    unavailable = {}
    for tool, command in commands.items():
        try:
            lookups[tool] = _query(command)
        except (ValueError, OSError, subprocess.SubprocessError) as exc:
            unavailable[tool] = str(exc)
    matches = {}
    evidence = {}
    other_jobs = set()
    for source, text in lookups.items():
        for line in text.splitlines():
            if not line.strip():
                continue
            parts = [x.strip() for x in line.split("|", 5)]
            if len(parts) != (6 if source == "sacct" else 5):
                raise ValueError(f"Incomplete {source} reply for {name}")
            job, owner, job_name, state, stamp = parts[:5]
            if owner != user or job_name != name:
                continue
            if not job.isdecimal():
                raise ValueError("Unexpected job identity in scheduler reply")
            if (
                not start - CLOCK_SKEW_SECONDS
                <= _stamp(stamp)
                <= end + CLOCK_SKEW_SECONDS
            ):
                continue
            # A different wave/root can have the same singleton job name.
            if source == "sacct" and parts[5]:
                try:
                    other = shlex.split(parts[5])
                    if _value(other, "output") != _value(argv, "output") or (
                        any(a.startswith("--comment=") for a in argv)
                        and any(a.startswith("--comment=") for a in other)
                        and _value(other, "comment") != _value(argv, "comment")
                    ):
                        other_jobs.add(job)
                except ValueError:
                    pass
            if job in other_jobs:
                continue
            matches[job] = state.split()[0]
            if source == "sacct" and _matches(parts[5], argv):
                evidence[job] = "sacct SubmitLine"
            elif (
                job not in evidence
                and source == "squeue"
                and _controller_match(job, argv)
            ):
                evidence[job] = "scontrol worker and log paths"
    if matches:
        for job in matches.keys() - evidence.keys():
            if _accounting_paths(job, argv):
                evidence[job] = "sacct output and error paths"
        if len(matches) != 1 or evidence.keys() != matches.keys():
            raise ValueError(f"Ambiguous {name} jobs {sorted(matches)} in {wave}")
        job, state = next(iter(matches.items()))
        return {
            "status": "confirmed",
            "job": job,
            "state": state,
            "evidence": evidence[job],
        }
    if unavailable:
        raise ValueError(f"Cannot establish absence from Slurm: {unavailable}")
    error = reply.get("stderr", "").strip()
    refused = any(
        text in error
        for text in (
            "Job violates accounting/QOS policy",
            "Invalid account",
            "Invalid partition",
            "Requested node configuration is not available",
            "Invalid time limit",
            "not allowed to submit",
            "not permitted to submit",
        )
    )
    if re.search(
        r"(?m)^(?:Submitted batch job )?[0-9]+(?:;[^\s;]+)?$",
        reply.get("stdout", "").strip(),
    ):
        raise ValueError(f"Scheduler stdout contains a job ID not yet verified: {wave}")
    if not refused and time.time() - end < CLOCK_SKEW_SECONDS:
        raise ValueError(
            f"No matching job yet. Repeat after accounting updates: {wave}"
        )
    return {
        "status": "scheduler_rejected" if refused else "absent_in_queue_and_accounting",
        "scheduler_stderr": error,
        "wave": wave.name,
    }


def predecessor(root: Path) -> dict | None:
    """Identify the existing first-wave job so depth three can wait for it."""
    bound = root / "predecessor.json"
    if not bound.exists():
        return None
    binding = sealed_read(bound)
    if binding["path"] is None:
        return None
    source = Path(binding["path"])
    waves = sorted((source / "submissions").glob("wave-*"))
    if not waves:
        return None
    wave = waves[-1]
    execution = sealed_read(source / "execution.json")
    if execution["plan_sha256"] != binding["plan_sha256"]:
        raise ValueError("Original Slurm execution belongs to a different plan")
    manifest = wave / "manifest.json"
    if manifest.exists():
        record = sealed_read(manifest)
        if record["execution_sha256"] != execution["artifact_sha256"]:
            raise ValueError(
                "Original submission manifest belongs to another execution"
            )
        return {
            "status": "confirmed",
            "job": record["job"],
            "evidence": "original manifest",
        }
    try:
        return reconcile(source, wave)
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        # Both plans use the same Slurm user and job name. The singleton waits
        # for earlier allocations even when accounting cannot supply an ID.
        return {
            "status": "unconfirmed_original",
            "wave": str(wave),
            "lookup": str(exc),
            "dependency": "singleton",
        }
