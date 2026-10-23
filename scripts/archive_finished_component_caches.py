#!/usr/bin/env python3
"""Archive verified compiler caches of terminal jobs before removal.

Default is a read-only plan. --apply retains a tar archive and a per-file checksum
receipt for every explicitly named job. Scientific call caches are not accepted.
The default component profile requires successful component proposer jobs. The
explicit phase-c and review profiles require an empty user queue and terminal
accounting. Only review supports bounded discovery of numeric job directories.
This script is outside the frozen experiment runtime.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import io
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
from collections.abc import Callable
from pathlib import Path

CACHE_KINDS = {"triton", "torchinductor", "vllm", "xdg", "cuda"}
MAX_DISCOVERED_JOBS = 256
PROFILES = {
    "component": (
        "component-runtime-cache",
        "finished-component-cache-archive-1",
    ),
    "phase-c": ("phase-c-runtime-cache", "finished-phase-c-cache-archive-1"),
    "review": ("runtime-cache/review", "finished-review-cache-archive-1"),
}
TERMINAL_STATES = {
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


def digest(stream) -> str:
    """Hash bytes without interpreting or loading an entire cache file."""
    value = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        value.update(block)
    return value.hexdigest()


def physical(path: Path) -> Path:
    """Reject symlinked roots and ancestors instead of following them."""
    absolute = path.expanduser().absolute()
    if absolute.resolve() != absolute:
        raise ValueError(f"symlinked path: {absolute}")
    return absolute


def identity(metadata: os.stat_result) -> tuple:
    """Track identity/content metadata while allowing read-induced atime updates."""
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_uid,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def snapshot(root: Path, job: str) -> dict:
    """Read owned ordinary files under the reviewed compiler-cache namespaces."""
    base = root / job
    if base.is_symlink():
        raise ValueError("symlinked job directory")
    if not base.exists():
        return {}
    if not base.is_dir():
        raise ValueError("job path is not a directory")
    if any(p.name not in CACHE_KINDS or not p.is_dir() for p in base.iterdir()):
        raise ValueError("unexpected top-level cache entry")
    rows, pending = {}, [base]
    device = root.stat().st_dev
    while pending:
        path = pending.pop()
        before = path.lstat()
        if before.st_dev != device or before.st_uid != os.getuid():
            raise ValueError("cache mount or owner differs")
        row = {"mode": stat.S_IMODE(before.st_mode)}
        if stat.S_ISDIR(before.st_mode):
            row["kind"] = "directory"
            pending.extend(path.iterdir())
        elif stat.S_ISREG(before.st_mode) and before.st_nlink == 1:
            with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "rb") as stream:
                if identity(os.fstat(stream.fileno())) != identity(before):
                    raise ValueError("cache changed while opening")
                row.update(kind="file", size=before.st_size, sha256=digest(stream))
            after = path.lstat()
            if identity(before) != identity(after):
                raise ValueError("cache changed while reading")
        else:
            raise ValueError("symlink, hard link or special cache entry")
        rows[path.relative_to(root).as_posix()] = row
    return dict(sorted(rows.items()))


def verify_archive(path: Path, expected: dict) -> str:
    """Read back every member; validate content without extracting anything."""
    seen = set()
    with tarfile.open(path, "r:") as archive:
        for item in archive:
            if item.name in seen or item.name not in expected:
                raise ValueError("unexpected or duplicate archive member")
            seen.add(item.name)
            row = expected[item.name]
            if item.mode != row["mode"]:
                raise ValueError("archive mode differs")
            if row["kind"] == "directory":
                if not item.isdir():
                    raise ValueError("archive directory differs")
            elif not item.isfile() or item.size != row["size"]:
                raise ValueError("archive file differs")
            else:
                with archive.extractfile(item) as stream:
                    if digest(stream) != row["sha256"]:
                        raise ValueError("archive file checksum differs")
    if seen != set(expected):
        raise ValueError("archive is incomplete")
    with path.open("rb") as stream:
        return digest(stream)


def guard_jobs(jobs: list[str]) -> None:
    """Require current scheduler evidence; never infer array raw IDs."""
    queue = subprocess.check_output(
        ["squeue", "--noheader", "--user", str(os.getuid()), "--format=%j"],
        text=True,
    )
    if any(name.strip().startswith("component-") for name in queue.splitlines()):
        raise ValueError("component workers are queued/running")
    text = subprocess.check_output(
        [
            "sacct",
            "-j",
            ",".join(jobs),
            "--parsable2",
            "--format=JobID%40,JobIDRaw%40,JobName%40,State%30,ExitCode",
        ],
        text=True,
    )
    records = list(csv.DictReader(io.StringIO(text), delimiter="|"))
    for job in jobs:
        matches = [
            r
            for r in records
            if (r.get("JobIDRaw") or "").strip() == job
            and "." not in (r.get("JobID") or "")
        ]
        if len(matches) != 1 or any(
            matches[0].get(k, "").strip() != v
            for k, v in {
                "JobName": "component-propose",
                "State": "COMPLETED",
                "ExitCode": "0:0",
            }.items()
        ):
            raise ValueError(f"unconfirmed successful proposer job: {job}")


def guard_phase_c_jobs(jobs: list[str]) -> None:
    """Require an empty user queue and exact, owned, terminal allocation records."""
    owner = str(os.getuid())
    queue = subprocess.check_output(
        ["squeue", "--noheader", "--user", owner, "--format=%i"], text=True
    )
    if queue.strip():
        raise ValueError(
            "user jobs are queued/running; compiler caches remain in place"
        )
    if not jobs:
        return
    text = subprocess.check_output(
        [
            "sacct",
            "-j",
            ",".join(jobs),
            "--parsable2",
            "--format=JobID%40,JobIDRaw%40,JobName%40,State%40,ExitCode,UID",
        ],
        text=True,
    )
    records = list(csv.DictReader(io.StringIO(text), delimiter="|"))
    for job in jobs:
        matches = [
            row
            for row in records
            if (row.get("JobIDRaw") or "").strip() == job
            and "." not in (row.get("JobID") or "")
        ]
        if len(matches) != 1:
            raise ValueError(f"unconfirmed terminal allocation: {job}")
        row = matches[0]
        state = (row.get("State") or "").strip()
        # Slurm may append the cancelling UID. Do not accept arbitrary state
        # prefixes, truncation markers, or held/requeued states as terminal.
        if re.fullmatch(r"CANCELLED by [0-9]+", state):
            state = "CANCELLED"
        if state not in TERMINAL_STATES or (row.get("UID") or "").strip() != owner:
            raise ValueError(f"unconfirmed owned terminal allocation: {job}")


def discover_review_jobs(root: Path) -> list[str]:
    """Bound discovery and reject anything except real numeric job directories."""
    jobs = []
    with os.scandir(root) as entries:
        for entry in entries:
            if len(jobs) >= MAX_DISCOVERED_JOBS:
                raise ValueError("review cache job discovery limit exceeded")
            if re.fullmatch(r"[0-9]+", entry.name) is None or not entry.is_dir(
                follow_symlinks=False
            ):
                raise ValueError(f"unexpected review cache entry: {entry.name}")
            jobs.append(entry.name)
    return sorted(jobs)


def prepare_archive(
    root: Path, destination: Path, job: str, *, profile: str = "component"
) -> dict:
    """Create or verify one recoverable archive; never overwrite old receipts."""
    if profile not in PROFILES:
        raise ValueError(f"unknown cache profile: {profile}")
    protocol = PROFILES[profile][1]
    archive, receipt = destination / f"{job}.tar", destination / f"{job}.json"
    if archive.is_symlink() or receipt.is_symlink():
        raise ValueError("symlinked archive or receipt")
    if receipt.exists():
        record = json.loads(receipt.read_text())
        if (
            record.get("protocol") != protocol
            or record.get("root") != str(root)
            or record.get("job") != job
        ):
            raise ValueError("archive receipt identity differs")
        if verify_archive(archive, record["entries"]) != record["archive_sha256"]:
            raise ValueError("archive checksum differs")
        return record
    entries = snapshot(root, job)
    if not entries:
        raise ValueError("missing cache without verified archive")
    if archive.exists():
        # A crash after rename but before receipt is recoverable only by checking
        # the entire archive against the still-present original cache.
        temporary = archive
    else:
        # Preserve incomplete files after interruption, never reuse them.
        with tempfile.NamedTemporaryFile(
            dir=destination, suffix=".partial", delete=False
        ) as f:
            temporary = Path(f.name)
        with tarfile.open(temporary, "w:", dereference=False) as output:
            for name in entries:
                output.add(root / name, arcname=name, recursive=False)
    sha = verify_archive(temporary, entries)
    if snapshot(root, job) != entries:
        raise ValueError("cache changed during archiving")
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    record = {
        "protocol": protocol,
        "root": str(root),
        "job": job,
        "archive_sha256": sha,
        "entries": entries,
    }
    if temporary != archive:
        temporary.rename(archive)
    with receipt.open("x") as stream:
        json.dump(record, stream, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    return record


def run(
    root: Path,
    destination: Path,
    jobs: list[str] | None = None,
    *,
    apply: bool,
    profile: str = "component",
    all_jobs: bool = False,
    guard: Callable[[list[str]], None] | None = None,
) -> dict:
    """Validate all jobs before changes; resume deletion only from verified archives."""
    if profile not in PROFILES:
        raise ValueError(f"unknown cache profile: {profile}")
    root, destination = physical(root), physical(destination)
    expected_root = PROFILES[profile][0]
    expected_parts = Path(expected_root).parts
    if root.parts[-len(expected_parts) :] != expected_parts or not root.is_dir():
        raise ValueError(f"expected existing {expected_root} root")
    if destination.is_relative_to(root) or root.is_relative_to(destination):
        raise ValueError("archive and cache roots overlap")
    if all_jobs and (profile != "review" or jobs is not None):
        raise ValueError("--all-jobs is exclusive with --jobs and requires review")
    review_roster = discover_review_jobs(root) if profile == "review" else None
    if all_jobs:
        jobs = review_roster
    if (
        jobs is None
        or (not jobs and not all_jobs)
        or len(set(jobs)) != len(jobs)
        or any(re.fullmatch(r"[0-9]+", job) is None for job in jobs)
    ):
        raise ValueError("unique numeric job IDs required")
    if guard is None:
        guard = guard_jobs if profile == "component" else guard_phase_c_jobs
    guard(jobs)
    if not apply:
        return {"apply": False, "entries": {j: len(snapshot(root, j)) for j in jobs}}
    if not jobs:
        return {
            "apply": True,
            "archive_root": str(destination),
            "jobs": [],
            "entries_removed": 0,
            "archive_count": 0,
        }
    if not shutil.rmtree.avoids_symlink_attacks:
        raise ValueError("this platform lacks safe directory removal")
    destination.mkdir(parents=True, exist_ok=True)
    lock_path = destination / ".lock"
    with os.fdopen(
        os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600), "r+"
    ) as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        records = {}
        for job in jobs:
            print(f"Archiving/verifying completed-job cache {job}...", file=sys.stderr)
            records[job] = prepare_archive(root, destination, job, profile=profile)
        directory_fd = os.open(destination, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        guard(jobs)
        if profile == "review" and discover_review_jobs(root) != review_roster:
            raise ValueError("review cache job roster changed during archiving")
        # Preflight every remaining subtree before removing any of them.
        print("Rechecking live cache contents before removal...", file=sys.stderr)
        current = {j: snapshot(root, j) for j in jobs}
        for job, entries in current.items():
            if any(records[job]["entries"].get(k) != v for k, v in entries.items()):
                raise ValueError("live cache differs from verified archive")
        removed = 0
        for job in jobs:
            if current[job]:
                print(f"Removing verified loose cache {job}...", file=sys.stderr)
                shutil.rmtree(root / job)
                removed += len(current[job])
        return {
            "apply": True,
            "archive_root": str(destination),
            "jobs": jobs,
            "entries_removed": removed,
            "archive_count": len(records),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--jobs", nargs="+")
    selection.add_argument("--all-jobs", action="store_true")
    parser.add_argument("--profile", choices=tuple(PROFILES), default="component")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if args.all_jobs and args.profile != "review":
        parser.error("--all-jobs requires --profile review")
    print(
        json.dumps(
            run(
                args.root,
                args.archive_root,
                args.jobs,
                apply=args.apply,
                profile=args.profile,
                all_jobs=args.all_jobs,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
