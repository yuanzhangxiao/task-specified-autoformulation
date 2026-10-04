#!/usr/bin/env python3
"""Recover derived Phase C reports using the original, verified scientific runtime.

Only TRACE.html and trace.json are removable. No model, call, transaction,
plan, log, temporary checkpoint or scheduler receipt is deleted or rewritten.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import html
import importlib
import json
import os
import sys
import tarfile
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from urllib.parse import quote

POLICY = "phase-c-linked-trace-recovery-1"
DERIVED = ("TRACE.html", "trace.json")


def safe_file(path: Path, root: Path) -> bool:
    """Do not follow links out of the selected campaign during cleanup/export."""
    if not path.exists() and not path.is_symlink():
        return False
    if path.is_symlink() or path.resolve() != root / path.relative_to(root):
        raise ValueError(f"linked campaign path is not supported: {path}")
    return path.is_file()


def runtime(repo: Path):
    """Import the original checkout, never substitute this tool's newer runtime."""
    expected = repo / "src/autoformalism/research/construction_comparison.py"
    if not expected.is_file():
        raise ValueError(f"missing original construction runtime: {expected}")
    sys.path.insert(0, str(repo / "src"))
    module = importlib.import_module("autoformalism.research.construction_comparison")
    if Path(module.__file__).resolve() != expected.resolve():
        raise ValueError("a different autoformalism runtime was already imported")
    return module


def inventory(root: Path, campaign, plan: dict) -> dict:
    """Validate authoritative evidence before classifying expendable views."""
    rows = []
    for task in plan["tasks"]:
        directory = campaign.baseline.location(root, task)
        if directory.resolve() != directory:
            raise ValueError(f"linked result directory: {directory}")
        row = {"task": task["task_id"], "derived_bytes": 0, "derived_files": []}
        for name in DERIVED:
            path = directory / name
            if safe_file(path, root):
                row["derived_bytes"] += path.stat().st_size
                row["derived_files"].append(str(path.relative_to(root)))
        try:
            records = campaign.checked_records(
                directory, campaign.baseline.namespace(plan, task)
            )
            saved = campaign.baseline.read_outcome(
                directory / "proposal.json", plan, task
            )
            if row["derived_files"] and not records and not saved:
                raise ValueError("derived views have no surviving source records")
            if saved and saved["cost"] != campaign._cost(records):
                raise ValueError("saved proposal accounting differs from calls")
            row.update(
                status=saved["status"]
                if saved
                else "partial"
                if records
                else "unstarted",
                physical_requests=len(records),
                call_status_counts=dict(Counter(r["status"] for r in records)),
                evidence_valid=True,
            )
        except (ValueError, KeyError, TypeError, OSError) as exc:
            row.update(
                status="checkpoint_error",
                evidence_valid=False,
                error=f"{type(exc).__name__}: {exc}",
            )
        rows.append(row)
    return {
        "policy": POLICY,
        "identity": plan["artifact_sha256"],
        "status_counts": dict(Counter(r["status"] for r in rows)),
        "derived_bytes": sum(r["derived_bytes"] for r in rows),
        "reclaimable_bytes": sum(
            r["derived_bytes"] for r in rows if r["evidence_valid"]
        ),
        "checkpoint_errors": [r for r in rows if not r["evidence_valid"]],
        "rows": rows,
        "llm_calls": 0,
        "optimizer_calls": 0,
        "test_data_opened": False,
    }


def reclaim(root: Path, audit: dict) -> dict:
    """Remove only two reproducible views from tasks with verified source records."""
    count, size = 0, 0
    for row in audit["rows"]:
        if not row["evidence_valid"]:
            continue
        for name in row["derived_files"]:
            path = root / name
            if (
                path.parent != root / "results" / row["task"]
                or path.name not in DERIVED
            ):
                raise ValueError("cleanup path is not an approved derived view")
            if safe_file(path, root):
                size += path.stat().st_size
                path.unlink()
                count += 1
    return {"removed_derived_files": count, "reclaimed_bytes": size}


def linked_trace(directory: Path, identity: str, campaign) -> dict:
    """Index original call/event files without duplicating their full contents."""
    records = campaign.checked_records(directory, identity)
    events = {}
    for path in sorted((directory / "construction/events").glob("*.json")):
        event = campaign.sealed_read(path)
        events.setdefault(event["request_hash"], []).append(
            str(path.relative_to(directory))
        )
    rows, parts = (
        [],
        [
            f"<!-- {POLICY} -->",
            "<!doctype html><meta charset='utf-8'>"
            "<title>Construction trace index</title>",
            "<h1>Construction trace index</h1>"
            "<p>Exact requests and responses remain in "
            "the linked call JSON files. Runtime decisions and model snapshots remain "
            "in the linked event files. "
            "These views do not duplicate their contents.</p>",
        ],
    )
    for record in records:
        key = record["request_hash"]
        row = {
            "request_hash": key,
            "step": record["step"],
            "attempt": record["attempt"],
            "status": record["status"],
            "call_file": f"calls/{key}.json",
            "event_files": events.get(key, []),
        }
        rows.append(row)
        label = html.escape(f"{row['step']} / {row['attempt']} / {row['status']}")
        links = " ".join(
            f'<a href="{quote(path, safe="/")}">{html.escape(Path(path).name)}</a>'
            for path in (row["call_file"], *row["event_files"])
        )
        parts.append(f"<p>{label}: {links}</p>")
    value = {
        "policy": POLICY,
        "ordering": "request_hash; step shown explicitly",
        "calls": rows,
    }
    # Keep only one small index per task: quota may limit file count as well as
    # bytes. Every exact JSON request/response remains in the original call file.
    path = directory / "TRACE.html"
    temporary = directory / f".TRACE.html.recovery-{os.getpid()}.tmp"
    try:
        temporary.write_text("\n".join(parts), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return value


def recover(root: Path, campaign, plan: dict) -> dict:
    """Refuse active construction locks, then revalidate before any deletion."""
    with ExitStack() as locks:
        for task in plan["tasks"]:
            directory = campaign.baseline.location(root, task)
            path = directory / "construction-lock/.lock"
            if safe_file(path, root):
                stream = locks.enter_context(path.open("r"))
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        audit = inventory(root, campaign, plan)
        return _recover(root, campaign, plan, audit)


def _recover(root: Path, campaign, plan: dict, audit: dict) -> dict:
    """Regenerate reporting only; no fitting, proposal generation or promotion."""
    cleaned = reclaim(root, audit)
    if audit["checkpoint_errors"]:
        return {
            **cleaned,
            "status": "checkpoint_errors_require_inspection",
            "checkpoint_errors": audit["checkpoint_errors"],
        }
    original = campaign.construction_trace.render
    try:
        # Scoped presentation adapter. Original source verification, record checks,
        # namespace and report calculations remain those of the frozen campaign.
        campaign.construction_trace.render = lambda directory, identity: linked_trace(
            directory, identity, campaign
        )
        result = campaign.report(root, plan)
    finally:
        campaign.construction_trace.render = original
    return {
        **cleaned,
        "status": "reported",
        "status_counts": result["status_counts"],
        "physical_requests": result["physical_requests"],
        "observed_total_tokens": result["observed_total_tokens"],
        "unmeasured_requests": result["unmeasured_requests"],
    }


def export(root: Path, destination: Path) -> dict:
    """Package original evidence and small reports; avoid duplicating derived traces."""
    names = [
        root / name
        for name in ("plan.json", "summary.json", "SUMMARY.md", "TOPOLOGY.html")
    ]
    for name in ("results", "logs", "submissions", "runtime"):
        base = root / name
        if base.is_symlink():
            raise ValueError(f"linked archive input: {base}")
        if base.exists():
            names.extend(sorted(base.rglob("*")))
    destination = destination.resolve()
    if destination.is_relative_to(root) and (
        destination.parent != root or not destination.name.endswith(".tar.gz")
    ):
        raise ValueError("archive destination must not replace campaign evidence")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".{os.getpid()}.tmp")
    count = 0
    try:
        with tarfile.open(temporary, "w:gz") as archive:
            for path in names:
                if path in (destination, temporary) or path.name in (
                    ".lock",
                    "trace.json",
                ):
                    continue
                if path.name == "TRACE.html":
                    with path.open() as stream:
                        marker = stream.readline().strip()
                    if marker != f"<!-- {POLICY} -->":
                        raise ValueError("recover linked traces before exporting")
                if safe_file(path, root):
                    archive.add(
                        path, arcname=str(path.relative_to(root)), recursive=False
                    )
                    count += 1
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "archive": str(destination),
        "files": count,
        "bytes": destination.stat().st_size,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("inspect", "recover"))
    parser.add_argument(
        "--repo", type=Path, required=True, help="Original pinned source directory"
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--archive", type=Path, help="Optional compact evidence archive after recovery"
    )
    args = parser.parse_args()
    if args.archive and args.command != "recover":
        parser.error("--archive requires recover")
    root, repo = args.root.resolve(), args.repo.resolve()
    campaign = runtime(repo)
    plan = campaign.verify(root)
    if plan.get("test_data_opened") is not False:
        raise ValueError("requires a development-only campaign")
    audit = inventory(root, campaign, plan)
    print(
        json.dumps({k: v for k, v in audit.items() if k != "rows"}, indent=2),
        flush=True,
    )
    if args.command == "recover":
        result = recover(root, campaign, plan)
        result.update(
            policy=POLICY,
            tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            identity=plan["artifact_sha256"],
            llm_calls=0,
            optimizer_calls=0,
            source_and_plan_modified=False,
        )
        print(json.dumps(result, indent=2), flush=True)
        if result["status"] != "reported":
            raise SystemExit(2)
        if args.archive:
            print(json.dumps(export(root, args.archive), indent=2), flush=True)


if __name__ == "__main__":
    main()
