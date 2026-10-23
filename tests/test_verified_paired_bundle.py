"""Verified bundle installation never publishes incomplete or substituted code."""

from __future__ import annotations

import errno
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from scripts import install_verified_paired_bundle as installer

ROOT = "phase-c-paired-fit-0bf4747"


def make_archive(tmp_path, *, changes=None, manifest=None, extra=()):
    """Build regular-file fixtures, optionally retaining a stale manifest."""
    files = dict.fromkeys(installer.REQUIRED, b"fixture\n")
    files["run_fit.sh"] = b"#!/bin/bash\nexit 0\n"
    if manifest is None:
        manifest = "".join(
            f"{hashlib.sha256(data).hexdigest()}  {name}\n"
            for name, data in sorted(files.items())
        ).encode()
    files[installer.MANIFEST] = manifest
    for name, data in (changes or {}).items():
        if data is None:
            del files[name]
        else:
            files[name] = data
    archive = tmp_path / "bundle.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        for name, data in files.items():
            member = tarfile.TarInfo(f"{ROOT}/{name}")
            member.size = len(data)
            member.mode = 0o755 if name.endswith(".sh") else 0o644
            bundle.addfile(member, io.BytesIO(data))
        for member in extra:
            bundle.addfile(member, io.BytesIO(b""))
    return archive, hashlib.sha256(archive.read_bytes()).hexdigest()


def test_install_success_and_exact_reuse(tmp_path):
    archive, digest = make_archive(tmp_path)
    parent = tmp_path / "code"
    parent.mkdir()
    result = installer.install_bundle(archive, digest, parent)
    destination = parent / ROOT
    assert result == {"code_path": str(destination), "files": 5, "reused": False}
    assert (destination / "run_fit.sh").stat().st_mode & 0o777 == 0o755
    assert {path.name for path in parent.iterdir()} == {ROOT}
    inode = destination.stat().st_ino
    reused = installer.install_bundle(archive, digest, parent)
    assert reused == result | {"reused": True}
    assert destination.stat().st_ino == inode


def test_wrong_archive_hash_never_creates_staging(tmp_path):
    archive, _ = make_archive(tmp_path)
    with pytest.raises(ValueError, match="archive SHA256 differs"):
        installer.install_bundle(archive, "0" * 64, tmp_path)
    assert {path.name for path in tmp_path.iterdir()} == {archive.name}


@pytest.mark.parametrize(
    "changes,message",
    [
        ({"run_fit.sh": b"corrupted"}, "member hash differs"),
        ({"run_fit.sh": None}, "missing its manifest or required"),
        ({installer.MANIFEST: None}, "missing its manifest"),
        ({"extra.py": b"unlisted"}, "file set differs"),
    ],
)
def test_corrupt_or_missing_member_fails_before_extraction(tmp_path, changes, message):
    archive, digest = make_archive(tmp_path, changes=changes)
    with pytest.raises(ValueError, match=message):
        installer.install_bundle(archive, digest, tmp_path)
    assert {path.name for path in tmp_path.iterdir()} == {archive.name}


@pytest.mark.parametrize(
    "name,kind",
    [
        (f"{ROOT}/../escaped", tarfile.REGTYPE),
        (f"/{ROOT}/absolute", tarfile.REGTYPE),
        (f"{ROOT}//double", tarfile.REGTYPE),
        (f"{ROOT}/./dot", tarfile.REGTYPE),
        (f"{ROOT}/back\\slash", tarfile.REGTYPE),
        (f"{ROOT}/run_fit.sh", tarfile.REGTYPE),
        ("another-root/file", tarfile.REGTYPE),
        ("phase-c-paired-fit-deadbee/file", tarfile.REGTYPE),
        (f"{ROOT}/link", tarfile.SYMTYPE),
        (f"{ROOT}/hardlink", tarfile.LNKTYPE),
        (f"{ROOT}/fifo", tarfile.FIFOTYPE),
        (f"{ROOT}/device", tarfile.CHRTYPE),
        (f"{ROOT}/directory", tarfile.DIRTYPE),
    ],
)
def test_unsafe_archive_members_are_rejected(tmp_path, name, kind):
    member = tarfile.TarInfo(name)
    member.type = kind
    member.linkname = "../outside"
    archive, digest = make_archive(tmp_path, extra=[member])
    with pytest.raises(ValueError):
        installer.install_bundle(archive, digest, tmp_path)
    assert {path.name for path in tmp_path.iterdir()} == {archive.name}


@pytest.mark.parametrize("suffix", ["duplicate", "self", "traversal", "invalid"])
def test_invalid_manifest_is_rejected(tmp_path, suffix):
    line = f"{'0' * 64}  run_fit.sh\n"
    values = {
        "duplicate": line * 2,
        "self": f"{'0' * 64}  MANIFEST.sha256\n",
        "traversal": f"{'0' * 64}  ../outside\n",
        "invalid": "not a checksum\n",
    }
    archive, digest = make_archive(tmp_path, manifest=values[suffix].encode())
    with pytest.raises(ValueError):
        installer.install_bundle(archive, digest, tmp_path)
    assert not (tmp_path / ROOT).exists()


def test_existing_partial_copy_and_scientific_results_are_preserved(tmp_path):
    archive, digest = make_archive(tmp_path)
    destination = tmp_path / ROOT
    destination.mkdir()
    partial = destination / "run_fit.sh"
    partial.write_bytes(b"partial user copy")
    results = tmp_path / "scientific-results.json"
    results.write_bytes(b"saved results")
    with pytest.raises(ValueError, match="destination file hash differs"):
        installer.install_bundle(archive, digest, tmp_path)
    assert partial.read_bytes() == b"partial user copy"
    assert results.read_bytes() == b"saved results"
    assert {path.name for path in tmp_path.iterdir()} == {
        archive.name, ROOT, results.name,
    }


@pytest.mark.parametrize("change", ["missing", "extra", "directory", "symlink"])
def test_existing_mutated_copy_is_never_reused(tmp_path, change):
    archive, digest = make_archive(tmp_path)
    installer.install_bundle(archive, digest, tmp_path)
    destination = tmp_path / ROOT
    if change == "missing":
        (destination / "run_fit.sh").unlink()
    elif change == "extra":
        (destination / "extra").write_bytes(b"user data")
    elif change == "directory":
        (destination / "extra-directory").mkdir()
    else:
        (destination / "run_fit.sh").unlink()
        (destination / "run_fit.sh").symlink_to(archive)
    with pytest.raises(ValueError, match="destination"):
        installer.install_bundle(archive, digest, tmp_path)


@pytest.mark.parametrize("number", [errno.ENOSPC, errno.EDQUOT])
def test_write_failure_cleans_only_own_staging_directory(tmp_path, monkeypatch, number):
    archive, digest = make_archive(tmp_path)
    preserved = tmp_path / ".unrelated-stage"
    preserved.mkdir()
    (preserved / "saved").write_bytes(b"preserve")

    def fail(source, target):
        target.write(b"partial")
        raise OSError(number, os.strerror(number))

    monkeypatch.setattr(installer.shutil, "copyfileobj", fail)
    with pytest.raises(RuntimeError, match="storage/quota"):
        installer.install_bundle(archive, digest, tmp_path)
    assert {path.name for path in tmp_path.iterdir()} == {archive.name, preserved.name}
    assert (preserved / "saved").read_bytes() == b"preserve"


def test_atomic_publication_refuses_concurrent_empty_destination(tmp_path, monkeypatch):
    archive, digest = make_archive(tmp_path)
    publish = installer._publish
    destination = tmp_path / ROOT

    def race(staged, target):
        target.mkdir()
        publish(staged, target)

    monkeypatch.setattr(installer, "_publish", race)
    with pytest.raises(ValueError, match="destination is incomplete"):
        installer.install_bundle(archive, digest, tmp_path)
    assert destination.is_dir() and list(destination.iterdir()) == []
    assert {path.name for path in tmp_path.iterdir()} == {archive.name, ROOT}


@pytest.mark.parametrize("new_user_content", [False, True])
def test_publication_failure_cleans_only_matching_empty_reservation(
    tmp_path, monkeypatch, new_user_content,
):
    archive, digest = make_archive(tmp_path)
    destination = tmp_path / ROOT

    def fail(staged, target):
        if new_user_content:
            (target / "user-content").write_bytes(b"preserve")
        raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC))

    monkeypatch.setattr(Path, "rename", fail)
    with pytest.raises(RuntimeError, match="storage/quota"):
        installer.install_bundle(archive, digest, tmp_path)
    if new_user_content:
        assert (destination / "user-content").read_bytes() == b"preserve"
    else:
        assert not destination.exists()
    assert not list(tmp_path.glob(f".{ROOT}.install-*"))


def test_standalone_cli_works_without_site_packages_and_prints_json(tmp_path):
    archive, digest = make_archive(tmp_path)
    script = Path(installer.__file__)
    result = subprocess.run(
        [sys.executable, "-I", "-S", str(script), "--archive", str(archive),
         "--sha256", digest, "--destination-parent", str(tmp_path)],
        check=True, capture_output=True, text=True,
    )
    assert json.loads(result.stdout) == {
        "code_path": str(tmp_path / ROOT), "files": 5, "reused": False,
    }


def test_cli_never_imports_ctypes_or_scientific_libraries(tmp_path):
    archive, digest = make_archive(tmp_path)
    code = '''
import importlib.abc
import runpy
import sys

class Unavailable(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {
            "ctypes", "numpy", "scipy", "pandas", "pydantic", "casadi", "autoformalism"
        }:
            raise ModuleNotFoundError(fullname)

sys.meta_path.insert(0, Unavailable())
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
'''
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", code, str(Path(installer.__file__)),
         "--archive", str(archive), "--sha256", digest,
         "--destination-parent", str(tmp_path)],
        check=True, capture_output=True, text=True,
    )
    assert json.loads(result.stdout)["files"] == 5
