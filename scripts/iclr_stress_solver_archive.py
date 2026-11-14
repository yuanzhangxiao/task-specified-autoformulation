"""Compact numerical scratch only after both arms have terminal, verified endpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tarfile
from pathlib import Path
from tempfile import NamedTemporaryFile

from autoformalism.fitting import public_fitting as public
from autoformalism.rebuttal import iclr_controlled_search as study
from autoformalism.rebuttal import review_deadline_io as io
from autoformalism.rebuttal.prefit_replay import sealed_read

PLAN = "3cbcba883d2a2437c9cd897ff3957fa321f2ae9aa813cef24fc67b54404f0562"
STORE = "solver-archives-v1"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def publish(path: Path, write) -> None:
    """Preserve an earlier valid publication if creating a new copy fails."""
    with NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        tmp = Path(stream.name)
    try:
        write(tmp)
        with tmp.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(tmp, path)
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        tmp.unlink(missing_ok=True)


def terminal(root: Path, plan: dict, block: dict) -> list[dict]:
    """Read the native endpoint/pruning/critic checks; pending work stays loose."""
    raw = sealed_read(root / "cells" / (block["cell"] + ".json"))
    if raw["artifact_sha256"] != plan["cells"][block["cell"]]:
        raise ValueError("cell differs from the sealed roster")
    result = []
    for task in block["tasks"]:
        row, trace = study.summarize_row(root, plan, raw, task)
        if row["status"] == "pending":
            return []
        if (
            row["status"] not in ("ready", "no_retained_model")
            or len(trace) != plan["visits"]
        ):
            raise ValueError("incomplete or unrecognized endpoint")
        result.append(
            {
                k: row.get(k)
                for k in (
                    "source_id",
                    "source_sha256",
                    "status",
                    "model_sha256",
                    "validation_nmse",
                )
            }
        )
    return result


def permitted(root: Path, block: dict, visits: int) -> set[str]:
    """Only fit/backend below a completed visit or a completed pruning fit."""
    paths = set()
    for task in block["tasks"]:
        rounds = [io.round_path(root, task, i) / "fit" for i in range(visits)]
        pr = root / "pruning" / task["task_id"]
        for fit in [*rounds, pr / "control", pr / "pruned"]:
            paths.add((fit / "backend").relative_to(root).as_posix())
    return paths


def inventory(root: Path, names: list[str]) -> dict[str, os.stat_result]:
    """lstat without following links, including on an interrupted retirement."""
    entries = {}
    device = root.stat().st_dev
    for name in names:
        start = root / name
        if start.resolve() != start:
            raise ValueError(f"linked optimizer directory: {name}")
        if not start.exists():
            continue
        todo = [start]
        while todo:
            path = todo.pop()
            st = path.lstat()
            if st.st_dev != device or st.st_uid != os.getuid():
                raise ValueError("unexpected optimizer owner or device")
            if stat.S_ISDIR(st.st_mode):
                todo.extend(path.iterdir())
            elif not stat.S_ISREG(st.st_mode) or st.st_nlink != 1:
                raise ValueError(f"unexpected link or special file: {path}")
            entries[path.relative_to(root).as_posix()] = st
    return entries


def checked_contents(archive: Path, allowed: set[str]) -> dict[str, str | None]:
    """Validate the full member set and hash every archived regular file."""
    contents = {}
    with tarfile.open(archive, "r|gz") as tar:
        for m in tar:
            name = m.name.rstrip("/")
            p = Path(name)
            if (
                p.is_absolute()
                or ".." in p.parts
                or p.as_posix() != name
                or name in contents
                or not ({name, *(a.as_posix() for a in p.parents)} & allowed)
            ):
                raise ValueError("unrecognized or duplicated archive path")
            if m.isdir():
                contents[name] = None
            elif m.isreg():
                with tar.extractfile(m) as stream:
                    h = hashlib.sha256()
                    for chunk in iter(lambda: stream.read(1 << 20), b""):
                        h.update(chunk)
                    contents[name] = h.hexdigest()
            else:
                raise ValueError("unexpected archive member type")
    return contents


def archive_paths(
    root: Path, allowed: set[str], name: str, mirror: Path, binding: dict
) -> dict:
    """Verify two copies and resume exact-member cleanup after interruption."""
    store = root / STORE
    mirror.mkdir(mode=0o700, parents=True, exist_ok=True)
    store.mkdir(mode=0o700, exist_ok=True)
    if (
        store.resolve() != store
        or mirror.resolve() != mirror
        or mirror.is_relative_to(root)
        or root.is_relative_to(mirror)
    ):
        raise ValueError("archive mirror must be a separate physical directory")
    tar = store / (name + ".tar.gz")
    backup = mirror / tar.name
    verified = store / (name + ".verified.json")
    closed = store / (name + ".closed.json")
    if closed.exists() and not verified.exists():
        raise ValueError("missing archive verification receipt")
    if verified.exists():
        info = json.loads(verified.read_bytes())
        names = info["roots"]
        if (
            info["binding"] != binding
            or not set(names) <= allowed
            or digest(tar) != info["sha256"]
            or digest(backup) != info["sha256"]
        ):
            raise ValueError("recorded archive, mirror or endpoint identity differs")
        if closed.exists():
            if json.loads(closed.read_bytes()) != info:
                raise ValueError("archive closeout differs")
            return info
    else:
        names = []
        for n in sorted(allowed):
            work = root / n
            if not work.exists():
                continue
            if not (work.parent / "result.json").is_file():
                continue  # No completed outer fit receipt; leave this subtree alone.
            receipt = public._read(work.parent / "result.json")
            fit = receipt["result"]
            if receipt["sha256"] != public.content_sha256(fit):
                raise ValueError("outer fit digest differs")
            backend = fit.get("backend_result_sha256")
            if backend and backend != public.content_sha256(
                public._read(work.parent / "backend_result.json")
            ):
                raise ValueError("outer optimizer receipt differs")
            names.append(n)
        entries = inventory(root, names)

        def pack(tmp):
            # BSD tar emits extra AppleDouble files for the host's extended attributes.
            # ACES uses GNU tar, which stores ACLs and xattrs in PAX headers.
            metadata = (
                ["--no-mac-metadata"]
                if sys.platform == "darwin"
                else ["--acls", "--xattrs"]
            )
            cmd = [
                "tar",
                "--format=posix",
                *metadata,
                "-czf",
                str(tmp),
                "-C",
                str(root),
            ]
            cmd += ["--", *names] if names else ["-T", "/dev/null"]
            subprocess.run(cmd, check=True)

        if not tar.exists():
            publish(tar, pack)
        contents = checked_contents(tar, set(names))
        if set(contents) != set(entries):
            raise ValueError("archive member set differs from the live directories")
        if sys.platform.startswith("linux"):
            subprocess.run(
                ["tar", "--acls", "--xattrs", "-dzf", str(tar), "-C", str(root)],
                check=True,
            )
        h = digest(tar)
        if not backup.exists():
            publish(backup, lambda p: shutil.copyfile(tar, p))
        if digest(backup) != h:
            raise ValueError("second archive copy differs")
        info = {
            "binding": binding,
            "roots": names,
            "archive": tar.name,
            "sha256": h,
            "entries": len(contents),
            "archive_bytes": tar.stat().st_size,
            "backup": str(backup),
        }
        publish(
            verified,
            lambda p: p.write_text(json.dumps(info, indent=2, sort_keys=True) + "\n"),
        )
    # A partial cleanup can resume: absent members must already be in both archives.
    contents = checked_contents(tar, set(names))
    entries = inventory(root, names)
    if not entries.keys() <= contents.keys() or len(contents) != info["entries"]:
        raise ValueError("loose paths differ from verified members")
    for n, st in entries.items():
        if stat.S_ISDIR(st.st_mode):
            if contents[n] is not None:
                raise ValueError("directory replaced an archived file")
        elif contents[n] != digest(root / n):
            raise ValueError("loose optimizer file differs from archive")
    for n in sorted(entries, key=lambda v: v.count("/"), reverse=True):
        path = root / n
        a, b = entries[n], path.lstat()
        if (a.st_dev, a.st_ino, a.st_mode, a.st_uid, a.st_gid) != (
            b.st_dev,
            b.st_ino,
            b.st_mode,
            b.st_uid,
            b.st_gid,
        ):
            raise ValueError("loose member changed during retirement")
        if stat.S_ISDIR(b.st_mode):
            path.rmdir()
        elif (a.st_size, a.st_mtime_ns, b.st_nlink) == (b.st_size, b.st_mtime_ns, 1):
            path.unlink()
        else:
            raise ValueError("loose file changed during retirement")
    return info


def compact(root: Path, mirror: Path, number: int) -> dict:
    """Hold the same block and execution locks as the worker throughout compaction."""
    root = root.resolve()
    plan = sealed_read(root / "plan.json")
    if plan["artifact_sha256"] != PLAN or not 0 <= number < len(plan["blocks"]):
        raise ValueError("wrong stress roster or block")
    block = plan["blocks"][number]
    with io.execution_lease(root), public._lock(root / "locks" / str(number)):
        ends = terminal(root, plan, block)
        if not ends:
            return {"block": number, "status": "pending", "entries": 0}
        name = f"block{number:04d}"
        info = archive_paths(
            root,
            permitted(root, block, plan["visits"]),
            name,
            mirror,
            {"plan": PLAN, "block": number, "endpoints": ends},
        )
        if terminal(root, plan, block) != ends:
            raise ValueError("native endpoint report changed after compaction")
        close = root / STORE / (name + ".closed.json")
        if not close.exists():
            publish(
                close,
                lambda p: p.write_text(
                    json.dumps(info, indent=2, sort_keys=True) + "\n"
                ),
            )
        return {
            "block": number,
            "status": "complete",
            "entries": info["entries"],
            "sha256": info["sha256"],
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--mirror", required=True, type=Path)
    parser.add_argument("--blocks", nargs="+", type=int, required=True)
    args = parser.parse_args()
    for block_number in args.blocks:
        result = compact(args.root, args.mirror, block_number)
        print(json.dumps(result), flush=True)
