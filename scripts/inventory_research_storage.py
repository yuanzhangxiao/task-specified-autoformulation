#!/usr/bin/env python3
"""Inventory explicit storage roots using metadata only; never delete or move files."""

from __future__ import annotations

import argparse
import json
import os
import stat
from collections import Counter
from pathlib import Path

GENERATED = {"__pycache__", ".pytest_cache", ".ruff_cache"}
RUNTIME = {".venv", "venvs", "containers", "huggingface-cache", "huggingface"}


def overlaps(left: Path, right: Path) -> bool:
    """Ancestors and descendants of a protected path both require retention."""
    return left.is_relative_to(right) or right.is_relative_to(left)


def policy(path: Path, protected: list[Path]) -> str:
    """Names suggest review order, never reproducibility or deletion permission."""
    if any(overlaps(path, item) for item in protected):
        return "protected"
    if GENERATED.intersection(path.parts):
        return "generated_cache_review"
    if RUNTIME.intersection(path.parts):
        return "runtime_dependency_review"
    return "retain_pending_dependency_and_archive_review"


def scan(root: Path, protected: list[Path], limit: int, depth: int) -> dict:
    """Bound metadata visits, count hard-linked allocation once, and skip mounts."""
    root_stat = root.stat()
    pending = [root]
    seen = set()
    buckets: dict[str, dict] = {}
    issues = []
    issue_counts = Counter()
    visited = 0
    truncated = False

    def issue(path: Path, code: str) -> None:
        issue_counts[code] += 1
        if len(issues) < 100:
            issues.append({"path": str(path.relative_to(root)), "code": code})

    while pending and not truncated:
        directory = pending.pop()
        try:
            # A directory may have changed since it was enumerated.
            current = directory.lstat()
            if not stat.S_ISDIR(current.st_mode) or current.st_dev != root_stat.st_dev:
                issue(directory, "directory_changed_or_mount_skipped")
                continue
            with os.scandir(directory) as entries:
                for entry in entries:
                    if visited >= limit:
                        truncated = True
                        break
                    visited += 1
                    path = Path(entry.path)
                    try:
                        metadata = entry.stat(follow_symlinks=False)
                    except OSError:
                        issue(path, "stat_failed")
                        continue
                    parts = path.relative_to(root).parts
                    kind = (
                        "directories"
                        if stat.S_ISDIR(metadata.st_mode)
                        else "files"
                        if stat.S_ISREG(metadata.st_mode)
                        else "symlinks"
                        if stat.S_ISLNK(metadata.st_mode)
                        else "other"
                    )
                    scope = parts if kind == "directories" else parts[:-1]
                    name = "/".join(scope[:depth]) or "."
                    bucket = buckets.setdefault(
                        name,
                        {
                            "path": name,
                            "files": 0,
                            "directories": 0,
                            "symlinks": 0,
                            "other": 0,
                            "apparent_file_bytes": 0,
                            "allocated_bytes_unique_inodes": 0,
                            "metadata_policy_counts": Counter(),
                            "generated_cache_file_bytes": 0,
                            "deletion_eligible": False,
                        },
                    )
                    bucket[kind] += 1
                    category = policy(path, protected)
                    bucket["metadata_policy_counts"][category] += 1
                    if metadata.st_dev != root_stat.st_dev:
                        issue(path, "mount_skipped")
                        continue
                    inode = (metadata.st_dev, metadata.st_ino)
                    if inode not in seen:
                        bucket["allocated_bytes_unique_inodes"] += (
                            getattr(metadata, "st_blocks", 0) * 512
                        )
                        seen.add(inode)
                    if kind == "files":
                        bucket["apparent_file_bytes"] += metadata.st_size
                        if category == "generated_cache_review":
                            bucket["generated_cache_file_bytes"] += metadata.st_size
                    elif kind == "directories":
                        pending.append(path)
        except OSError:
            issue(directory, "directory_unreadable")
    rows = sorted(
        buckets.values(), key=lambda b: (-b["allocated_bytes_unique_inodes"], b["path"])
    )
    totals = {
        k: sum(b[k] for b in rows)
        for k in (
            "files",
            "directories",
            "symlinks",
            "other",
            "apparent_file_bytes",
            "allocated_bytes_unique_inodes",
            "generated_cache_file_bytes",
        )
    }
    return {
        "root": str(root),
        "entries_visited": visited,
        "entry_limit": limit,
        "truncated": truncated,
        "complete": not truncated and not issue_counts,
        "totals": totals,
        "buckets": rows,
        "issue_counts": dict(issue_counts),
        "issue_examples": issues,
    }


def inventory(roots: list[Path], protected: list[Path], limit: int, depth: int) -> dict:
    """Require existing, disjoint roots and resolve declared retention anchors."""
    if limit < 1 or not 1 <= depth <= 5:
        raise ValueError("positive entry limit and depth between 1 and 5 required")
    if any(p.is_symlink() for p in roots):
        raise ValueError("use the physical directory, not a symlink root")
    roots = [p.expanduser().resolve(strict=True) for p in roots]
    if not roots or any(not p.is_dir() for p in roots):
        raise ValueError("existing directory roots required")
    if any(overlaps(a, b) for i, a in enumerate(roots) for b in roots[i + 1 :]):
        raise ValueError("inventory roots overlap; scan their common parent once")
    protected = [p.expanduser().resolve() for p in protected]
    return {
        "protocol": "research-storage-metadata-1",
        "protected_paths": [str(p) for p in protected],
        "roots": [scan(p, protected, limit, depth) for p in roots],
        "content_files_opened": 0,
        "files_deleted": 0,
        "scope": (
            "Filesystem metadata only; no payloads, model calls, quota API or "
            "live-job check. "
            "Symlinks are not followed; other filesystems are skipped. "
            "Partial scans are "
            "lower bounds, not zero usage. Apparent bytes count file names; "
            "allocated bytes "
            "deduplicate hard links within each root and attribute them to "
            "the first visited "
            "bucket. They do not measure deduplication, snapshots or provider billing. "
            "Directory metadata may change during observation. Every "
            "deletion_eligible is "
            "false: cache/runtime names are review hints, not evidence of "
            "safe deletion."
        ),
    }


def render(value: dict) -> str:
    """Display large directories and explicit incomplete-coverage notices."""
    lines = ["# Research storage inventory", "", value["scope"], ""]
    for root in value["roots"]:
        lines += [
            f"## {root['root']}",
            "",
            f"Complete traversal: {root['complete']}. "
            f"Visited: {root['entries_visited']}. Issues: {root['issue_counts']}.",
            "",
            "| Directory | Files | Directories | Allocated GiB | "
            "Apparent GiB | Policies |",
            "|---|---:|---:|---:|---:|---|",
        ]
        for b in root["buckets"]:
            name = b["path"].replace("|", "&#124;").replace("\n", " ")
            lines.append(
                f"| {name} | {b['files']} | {b['directories']} | "
                f"{b['allocated_bytes_unique_inodes'] / 2**30:.3f} | "
                f"{b['apparent_file_bytes'] / 2**30:.3f} | "
                f"{dict(b['metadata_policy_counts'])} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, action="append", required=True)
    parser.add_argument("--protect", type=Path, action="append", default=[])
    parser.add_argument("--max-entries", type=int, default=250000)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if any(overlaps(output, p.expanduser().resolve()) for p in args.root):
        raise ValueError("output must be outside and not contain scanned roots")
    if output.exists():
        raise ValueError("use a new output directory")
    value = inventory(args.root, args.protect, args.max_entries, args.depth)
    output.mkdir(parents=True)
    (output / "inventory.json").write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n"
    )
    (output / "INVENTORY.md").write_text(render(value))
    print(
        json.dumps(
            {
                "output": str(output),
                "roots": [
                    {k: r[k] for k in ("root", "complete", "totals", "issue_counts")}
                    for r in value["roots"]
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
