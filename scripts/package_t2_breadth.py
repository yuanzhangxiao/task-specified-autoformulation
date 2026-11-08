"""Build the ACES delivery from a Git commit, the pinned intake and Harmony asset."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path

from autoformalism.rebuttal.prefit_replay import sealed_read
from scripts import harmony_vocab, t2_breadth

NAME = "phase-c-t2-breadth-v2"


def package(destination: Path, intake: Path, harmony: Path) -> Path:
    """Include committed runtime files and exactly the two reviewed external inputs."""
    if sealed_read(intake)["artifact_sha256"] != t2_breadth.INTAKE:
        raise ValueError("Different T2 training intake")
    harmony_vocab.verify(harmony)
    repo = t2_breadth.REPO
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True)
    tracked = [
        "src",
        "pyproject.toml",
        "docs/PHASE_C_T2_BREADTH.md",
        *t2_breadth.ENTRYPOINTS,
    ]
    source = subprocess.check_output(
        ["git", "archive", "--format=tar", commit.strip(), *tracked], cwd=repo
    )
    files = {}
    with tarfile.open(fileobj=io.BytesIO(source)) as archive:
        for member in archive:
            if member.isfile():
                files[member.name] = archive.extractfile(member).read(), member.mode
    for target, path in [
        ("inputs/intake.json", intake),
        ("assets/harmony/o200k_base.tiktoken", harmony / "o200k_base.tiktoken"),
    ]:
        files[target] = path.read_bytes(), 0o644
    files["SOURCE_COMMIT"] = commit.encode(), 0o644
    checks = [
        f"{hashlib.sha256(data).hexdigest()}  {path}\n"
        for path, (data, _) in sorted(files.items())
    ]
    files["SHA256SUMS"] = "".join(checks).encode(), 0o644
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / (NAME + ".tar.gz")
    temporary = target.with_suffix(".tmp")
    with (
        temporary.open("wb") as raw,
        gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as stream,
        tarfile.open(fileobj=stream, mode="w") as archive,
    ):
        for path, (data, mode) in sorted(files.items()):
            member = tarfile.TarInfo(f"{NAME}/{path}")
            member.mode, member.size = mode, len(data)
            archive.addfile(member, io.BytesIO(data))
    temporary.replace(target)
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--intake", required=True, type=Path)
    parser.add_argument("--harmony", required=True, type=Path)
    args = parser.parse_args()
    result = package(args.out, args.intake, args.harmony)
    print(
        json.dumps(
            {
                "archive": str(result.resolve()),
                "sha256": hashlib.sha256(result.read_bytes()).hexdigest(),
                "bytes": result.stat().st_size,
            },
            indent=2,
        )
    )
