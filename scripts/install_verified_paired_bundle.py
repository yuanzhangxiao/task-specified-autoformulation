#!/usr/bin/env python3
"""Install a complete authenticated paired-fit bundle without running its code."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import BinaryIO

MANIFEST = "MANIFEST.sha256"
REQUIRED = {
    "run_fit.sh", "configs/construction_paired_fit_v1.json",
    "inputs/paired-inputs.json", "scripts/hpc/start_construction_paired_fit.sh",
}


def _digest(stream: BinaryIO) -> str:
    """Hash a stream with bounded memory."""
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def _relative(name: str) -> PurePosixPath:
    """Reject ambiguous or escaping names before any filesystem access."""
    path = PurePosixPath(name)
    if (
        not path.parts or path.is_absolute() or "\\" in name or "\x00" in name
        or path.as_posix() != name or any(p in {".", ".."} for p in path.parts)
    ):
        raise ValueError(f"unsafe bundle path: {name!r}")
    return path


def _inspect(
    bundle: tarfile.TarFile,
) -> tuple[str, dict[str, tarfile.TarInfo], dict[str, str]]:
    """Authenticate the exact archive roster and every manifest entry."""
    root, members = "", {}
    for member in bundle.getmembers():
        path = _relative(member.name)
        if (
            len(path.parts) < 2
            or not re.fullmatch(r"phase-c-paired-fit-[0-9a-f]{7,40}", path.parts[0])
            or member.type not in {tarfile.REGTYPE, tarfile.AREGTYPE}
        ):
            raise ValueError(f"unsupported bundle member: {member.name!r}")
        if root and root != path.parts[0]:
            raise ValueError("archive must contain exactly one bundle root")
        root, name = path.parts[0], str(PurePosixPath(*path.parts[1:]))
        if name in members:
            raise ValueError(f"duplicate archive member: {name}")
        members[name] = member
    if MANIFEST not in members or not members.keys() >= REQUIRED:
        raise ValueError("bundle is missing its manifest or required launch inputs")
    with bundle.extractfile(members[MANIFEST]) as stream:
        manifest = stream.read()
    expected = {}
    for line in manifest.decode("utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if not match:
            raise ValueError("invalid MANIFEST.sha256 entry")
        digest, name = match.groups()
        _relative(name)
        if name == MANIFEST or name in expected:
            raise ValueError(f"duplicate or self-referential manifest entry: {name}")
        expected[name] = digest
    if set(expected) != members.keys() - {MANIFEST}:
        raise ValueError("archive file set differs from MANIFEST.sha256")
    expected[MANIFEST] = hashlib.sha256(manifest).hexdigest()
    for name, member in members.items():
        if any(str(parent) in members for parent in PurePosixPath(name).parents):
            raise ValueError(f"conflicting file and directory paths: {name}")
        with bundle.extractfile(member) as stream:
            if _digest(stream) != expected[name]:
                raise ValueError(f"archive member hash differs: {name}")
    return root, members, expected


def _verify_tree(root: Path, expected: dict[str, str]) -> None:
    """Require exact regular files, hashes, and their necessary directories."""
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"existing destination is not a regular directory: {root}")
    directories = {
        str(parent) for name in expected for parent in PurePosixPath(name).parents
        if str(parent) != "."
    }
    found = set()
    for current, dirs, files in os.walk(root, followlinks=False):
        for name in dirs + files:
            path = Path(current) / name
            relative = path.relative_to(root).as_posix()
            mode = path.lstat().st_mode
            if stat.S_ISDIR(mode) and relative in directories:
                continue
            if not stat.S_ISREG(mode) or relative not in expected:
                raise ValueError(f"unexpected destination entry: {relative}")
            with path.open("rb") as stream:
                if _digest(stream) != expected[relative]:
                    raise ValueError(f"destination file hash differs: {relative}")
            found.add(relative)
    if found != expected.keys():
        missing = sorted(expected.keys() - found)
        raise ValueError(f"destination is incomplete; missing: {missing}")


def _publish(staged: Path, destination: Path) -> None:
    """Reserve an absent name and atomically replace only our empty directory."""
    destination.mkdir()
    reservation = destination.lstat()
    try:
        current = destination.lstat()
        if (current.st_dev, current.st_ino) != (reservation.st_dev, reservation.st_ino):
            raise RuntimeError("installation reservation changed; refusing publication")
        staged.rename(destination)
    except BaseException:
        try:
            current = destination.lstat()
            if (current.st_dev, current.st_ino) == (
                reservation.st_dev, reservation.st_ino,
            ):
                destination.rmdir()
        except OSError:
            pass  # Leave changed/nonempty directories untouched.
        raise


def install_bundle(
    archive: Path, expected_sha256: str, destination_parent: Path,
) -> dict[str, str | int | bool]:
    """Verify first, stage privately, then publish or reuse an exact complete copy."""
    if not re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha256):
        raise ValueError("expected archive SHA256 must contain 64 hexadecimal digits")
    try:
        with archive.open("rb") as source:
            if _digest(source) != expected_sha256.lower():
                raise ValueError("archive SHA256 differs; nothing was extracted")
            source.seek(0)
            with tarfile.open(fileobj=source, mode="r:*") as bundle:
                name, members, expected = _inspect(bundle)
                destination_parent = destination_parent.resolve(strict=True)
                destination = destination_parent / name
                if os.path.lexists(destination):
                    _verify_tree(destination, expected)
                    return {"code_path": str(destination), "files": len(expected),
                            "reused": True}
                reused = False
                with tempfile.TemporaryDirectory(
                    prefix=f".{name}.install-", dir=destination_parent,
                ) as temporary:
                    staged = Path(temporary)
                    for relative, member in members.items():
                        path = staged / relative
                        path.parent.mkdir(parents=True, exist_ok=True)
                        with bundle.extractfile(member) as data, path.open("xb") as out:
                            shutil.copyfileobj(data, out)
                        path.chmod(member.mode & 0o777)
                    _verify_tree(staged, expected)
                    try:
                        _publish(staged, destination)
                    except FileExistsError:
                        _verify_tree(destination, expected)
                        reused = True
                return {"code_path": str(destination), "files": len(expected),
                        "reused": reused}
    except OSError as error:
        raise RuntimeError(
            f"bundle installation failed (storage/quota or filesystem error): {error}; "
            "existing copies and scientific results were not replaced"
        ) from error


def main(argv: list[str] | None = None) -> None:
    """Install without importing scientific packages or launching jobs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--destination-parent", type=Path, required=True)
    args = parser.parse_args(argv)
    result = install_bundle(args.archive, args.sha256, args.destination_parent)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
